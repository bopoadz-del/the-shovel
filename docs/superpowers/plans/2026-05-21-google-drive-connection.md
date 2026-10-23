# Google Drive Connection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the user connect a personal Google Drive (OAuth) and import Drive files into a project as documents.

**Architecture:** A new `app/core/drive_auth.py` owns one app-wide OAuth token (encrypted at rest via `file_crypto`) and refreshes it. A new `app/routers/drive.py` exposes `/v1/drive/*` — the consent redirect, the callback, status/disconnect, file listing, and a per-project import that reuses the existing upload storage path. The existing `GoogleDriveBlock` is consumed as-is for `list`/`download`. The frontend's "Connect Drive" modal is wired to these routes.

**Tech Stack:** FastAPI, `httpx` (already a dep — no google-api libraries needed), the existing `file_crypto` and `projects` modules. Spec: `docs/superpowers/specs/2026-05-21-google-drive-connection-design.md`.

**Conventions:** run the venv python `C:\Users\example\The_Fork\.venv\Scripts\python.exe`; tests with `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest ... -q`. End commit messages with `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`.

## File structure

| File | Responsibility |
|------|----------------|
| `app/core/drive_auth.py` | CREATE — token persistence (encrypted) + access-token refresh + typed errors |
| `app/routers/drive.py` | CREATE — `/v1/drive/*` routes + the per-project import route |
| `app/main.py` | MODIFY — mount the drive router |
| `app/static/index.html` | MODIFY — wire the Connect Drive modal + Drive file browser |
| `tests/test_drive_auth.py` | CREATE — drive_auth unit tests |
| `tests/test_drive_router.py` | CREATE — drive router tests (Google calls mocked) |
| `tests/test_drive_router_live.py` | CREATE — one skipif-on-key live test |

`app/blocks/google_drive.py` is **not modified** — its `list`/`download` operations are called as-is with an `access_token` passed in `params`.

---

### Task 1: `drive_auth.py` — token store + refresh

**Files:**
- Create: `app/core/drive_auth.py`
- Test: `tests/test_drive_auth.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_drive_auth.py
import json, time, os
import pytest
from pathlib import Path
from app.core import drive_auth


@pytest.fixture
def tmp_data(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "cid")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "csecret")
    return tmp_path


def test_save_load_roundtrip(tmp_data):
    tok = {"access_token": "a", "refresh_token": "r", "expiry": time.time() + 9999, "email": "x@y.z"}
    drive_auth.save_token(tok)
    assert drive_auth.load_token() == tok


def test_load_returns_none_when_absent(tmp_data):
    assert drive_auth.load_token() is None


def test_clear_token(tmp_data):
    drive_auth.save_token({"access_token": "a"})
    assert drive_auth.clear_token() is True
    assert drive_auth.load_token() is None
    assert drive_auth.clear_token() is False


@pytest.mark.asyncio
async def test_get_access_token_returns_unexpired(tmp_data):
    drive_auth.save_token({"access_token": "live", "refresh_token": "r",
                           "expiry": time.time() + 9999})
    assert await drive_auth.get_access_token() == "live"


@pytest.mark.asyncio
async def test_get_access_token_refreshes_when_expired(tmp_data, monkeypatch):
    drive_auth.save_token({"access_token": "old", "refresh_token": "r",
                           "expiry": time.time() - 10})

    async def fake_refresh(refresh_token):
        assert refresh_token == "r"
        return {"access_token": "fresh", "expires_in": 3600}

    monkeypatch.setattr(drive_auth, "_refresh_request", fake_refresh)
    assert await drive_auth.get_access_token() == "fresh"
    # persisted
    assert drive_auth.load_token()["access_token"] == "fresh"


@pytest.mark.asyncio
async def test_get_access_token_raises_when_not_connected(tmp_data):
    with pytest.raises(drive_auth.DriveNotConnected):
        await drive_auth.get_access_token()


@pytest.mark.asyncio
async def test_get_access_token_raises_on_failed_refresh(tmp_data, monkeypatch):
    drive_auth.save_token({"access_token": "old", "refresh_token": "r",
                           "expiry": time.time() - 10})

    async def boom(refresh_token):
        raise drive_auth.DriveAuthError("refresh failed")

    monkeypatch.setattr(drive_auth, "_refresh_request", boom)
    with pytest.raises(drive_auth.DriveAuthError):
        await drive_auth.get_access_token()
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest tests/test_drive_auth.py -q`
Expected: FAIL — `ModuleNotFoundError: app.core.drive_auth`.

- [ ] **Step 3: Implement `app/core/drive_auth.py`**

```python
"""App-wide Google Drive OAuth token store + refresh.

One connection for the whole app. The token — including the refresh token, a
secret — is persisted to DATA_DIR/google_drive_token.json via file_crypto, so
it is encrypted at rest when DATA_ENCRYPTION_KEY is set.
"""
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

from app.core import file_crypto

_TOKEN_URL = "https://oauth2.googleapis.com/token"


class DriveNotConnected(Exception):
    """No Google Drive token is stored."""


class DriveAuthError(Exception):
    """Token exchange or refresh failed."""


def _token_path() -> Path:
    return Path(os.getenv("DATA_DIR", "./data")) / "google_drive_token.json"


def save_token(token: Dict[str, Any]) -> None:
    path = _token_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    file_crypto.write_document(str(path), json.dumps(token).encode("utf-8"))


def load_token() -> Optional[Dict[str, Any]]:
    path = _token_path()
    if not path.exists():
        return None
    return json.loads(file_crypto.read_document(str(path)).decode("utf-8"))


def clear_token() -> bool:
    path = _token_path()
    if path.exists():
        path.unlink()
        return True
    return False


async def _refresh_request(refresh_token: str) -> Dict[str, Any]:
    """POST the refresh grant to Google. Overridable seam for tests."""
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(_TOKEN_URL, data={
            "client_id": os.getenv("GOOGLE_CLIENT_ID", ""),
            "client_secret": os.getenv("GOOGLE_CLIENT_SECRET", ""),
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        })
    if resp.status_code != 200:
        raise DriveAuthError(f"Token refresh failed (HTTP {resp.status_code})")
    return resp.json()


async def get_access_token() -> str:
    """Return a valid access token, refreshing if it has expired."""
    token = load_token()
    if not token:
        raise DriveNotConnected("Google Drive is not connected.")
    if token.get("expiry", 0) > time.time() + 60:
        return token["access_token"]
    refresh_token = token.get("refresh_token")
    if not refresh_token:
        raise DriveAuthError("No refresh token stored — reconnect Google Drive.")
    data = await _refresh_request(refresh_token)
    token["access_token"] = data["access_token"]
    token["expiry"] = time.time() + int(data.get("expires_in", 3600))
    save_token(token)
    return token["access_token"]
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest tests/test_drive_auth.py -q`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add app/core/drive_auth.py tests/test_drive_auth.py
git commit -m "feat(drive): app-wide OAuth token store + refresh"
```

---

### Task 2: `drive.py` router — OAuth connect + callback, mounted

**Files:**
- Create: `app/routers/drive.py`
- Modify: `app/main.py` (mount the router)
- Test: `tests/test_drive_router.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_drive_router.py
import time
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.core import drive_auth

H = {"Authorization": "Bearer $CEREBRUM_DEV_KEY"}


@pytest.fixture(autouse=True)
def tmp_data(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "cid")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "csecret")
    monkeypatch.setenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/v1/drive/callback")
    yield


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_connect_returns_auth_url(client):
    # /connect returns the consent URL as JSON (NOT a redirect) so the
    # browser can fetch it with the Bearer header, then navigate client-side.
    r = client.get("/v1/drive/connect", headers=H)
    assert r.status_code == 200
    url = r.json()["auth_url"]
    assert url.startswith("https://accounts.google.com/o/oauth2/v2/auth")
    assert "scope=" in url and "drive.readonly" in url
    assert "state=" in url and "access_type=offline" in url


def test_connect_requires_auth(client):
    assert client.get("/v1/drive/connect", follow_redirects=False).status_code == 401


def test_callback_rejects_bad_state(client):
    r = client.get("/v1/drive/callback?code=x&state=never-issued",
                    follow_redirects=False)
    assert r.status_code == 400


def test_callback_exchanges_code_and_stores_token(client, monkeypatch):
    # issue a state via /connect
    r = client.get("/v1/drive/connect", headers=H)
    state = r.json()["auth_url"].split("state=")[1].split("&")[0]

    async def fake_exchange(code):
        assert code == "auth-code"
        return {"access_token": "AT", "refresh_token": "RT", "expires_in": 3600}

    async def fake_email(access_token):
        return "me@example.com"

    import app.routers.drive as drive_mod
    monkeypatch.setattr(drive_mod, "_exchange_code", fake_exchange)
    monkeypatch.setattr(drive_mod, "_fetch_email", fake_email)

    r = client.get(f"/v1/drive/callback?code=auth-code&state={state}",
                    follow_redirects=False)
    assert r.status_code in (302, 307)
    tok = drive_auth.load_token()
    assert tok["access_token"] == "AT" and tok["refresh_token"] == "RT"
    assert tok["email"] == "me@example.com"
    assert tok["expiry"] > time.time()
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest tests/test_drive_router.py -q`
Expected: FAIL — no `/v1/drive/*` routes.

- [ ] **Step 3: Create `app/routers/drive.py`**

```python
"""Google Drive connection — /v1/drive/* OAuth flow + file import.

All routes require Authorization: Bearer like other /v1/* routes, EXCEPT
/v1/drive/callback — Google calls that directly and cannot send our header,
so it is protected by the single-use OAuth `state` value instead.
"""
import os
import secrets
import time
from typing import Any, Dict, Optional

import httpx
from fastapi import APIRouter, Body, Depends, HTTPException, Query
from fastapi.responses import RedirectResponse

from app.dependencies import require_api_key
from app.core import drive_auth

router = APIRouter()

_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_TOKEN_URL = "https://oauth2.googleapis.com/token"
_SCOPE = "https://www.googleapis.com/auth/drive.readonly"
_DRIVE_API = "https://www.googleapis.com/drive/v3"

# Single-use OAuth state values issued by /connect (in-memory; single process).
_pending_states: set[str] = set()


def _redirect_uri() -> str:
    return os.getenv("GOOGLE_REDIRECT_URI",
                     "http://localhost:8000/v1/drive/callback")


def _configured() -> bool:
    return bool(os.getenv("GOOGLE_CLIENT_ID") and os.getenv("GOOGLE_CLIENT_SECRET"))


async def _exchange_code(code: str) -> Dict[str, Any]:
    """Exchange an auth code for tokens. Overridable seam for tests."""
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(_TOKEN_URL, data={
            "code": code,
            "client_id": os.getenv("GOOGLE_CLIENT_ID", ""),
            "client_secret": os.getenv("GOOGLE_CLIENT_SECRET", ""),
            "redirect_uri": _redirect_uri(),
            "grant_type": "authorization_code",
        })
    if resp.status_code != 200:
        raise drive_auth.DriveAuthError(
            f"Code exchange failed (HTTP {resp.status_code})")
    return resp.json()


async def _fetch_email(access_token: str) -> str:
    """Read the connected account's email via the Drive `about` endpoint
    (works with the drive.readonly scope). Overridable seam for tests."""
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(
            f"{_DRIVE_API}/about", params={"fields": "user"},
            headers={"Authorization": f"Bearer {access_token}"})
    if resp.status_code != 200:
        return ""
    return resp.json().get("user", {}).get("emailAddress", "")


@router.get("/v1/drive/connect")
async def drive_connect(auth: dict = Depends(require_api_key)):
    # Returns the Google consent URL as JSON — NOT a redirect. A browser
    # cannot attach the Bearer header to a top-level navigation, so the
    # frontend fetches this (header attaches fine on a same-origin fetch),
    # reads auth_url, and does window.location = auth_url itself. Keeps the
    # route gated like every other /v1/* and puts no key in any URL.
    if not _configured():
        raise HTTPException(503, "Google Drive not configured — set "
                                 "GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET.")
    state = secrets.token_urlsafe(24)
    _pending_states.add(state)
    from urllib.parse import urlencode
    url = _AUTH_URL + "?" + urlencode({
        "client_id": os.getenv("GOOGLE_CLIENT_ID", ""),
        "redirect_uri": _redirect_uri(),
        "response_type": "code",
        "scope": _SCOPE,
        "access_type": "offline",
        "prompt": "consent",
        "state": state,
    })
    return {"auth_url": url}


@router.get("/v1/drive/callback")
async def drive_callback(code: str = Query(""), state: str = Query("")):
    # No Bearer auth here — Google calls this. The single-use state is the gate.
    if state not in _pending_states:
        raise HTTPException(400, "Invalid or expired OAuth state.")
    _pending_states.discard(state)
    if not code:
        raise HTTPException(400, "Missing authorization code.")
    data = await _exchange_code(code)
    access_token = data["access_token"]
    email = await _fetch_email(access_token)
    drive_auth.save_token({
        "access_token": access_token,
        "refresh_token": data.get("refresh_token", ""),
        "expiry": time.time() + int(data.get("expires_in", 3600)),
        "email": email,
    })
    return RedirectResponse("/", status_code=302)
```

- [ ] **Step 4: Mount the router in `app/main.py`**

Find `app.include_router(mcp.router)` and the `mcp.mount_message_endpoint(app)` line; add `drive` alongside. First ensure `drive` is imported with the other routers (the routers are imported near the top of `main.py` — add `drive` to that import group, matching the existing style). Then add the include:

```python
app.include_router(mcp.router)
mcp.mount_message_endpoint(app)
app.include_router(drive.router)
```

- [ ] **Step 5: Run tests, verify they pass**

Run: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest tests/test_drive_router.py -q`
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add app/routers/drive.py app/main.py tests/test_drive_router.py
git commit -m "feat(drive): OAuth connect + callback routes"
```

---

### Task 3: `drive.py` router — status, disconnect, file listing

**Files:**
- Modify: `app/routers/drive.py`
- Test: `tests/test_drive_router.py` (append)

- [ ] **Step 1: Write the failing tests** (append to `tests/test_drive_router.py`)

```python
def test_status_not_connected(client):
    r = client.get("/v1/drive/status", headers=H)
    assert r.status_code == 200
    body = r.json()
    assert body["connected"] is False and body["configured"] is True


def test_status_connected_after_token(client):
    drive_auth.save_token({"access_token": "AT", "refresh_token": "RT",
                           "expiry": time.time() + 9999, "email": "me@example.com"})
    body = client.get("/v1/drive/status", headers=H).json()
    assert body["connected"] is True and body["email"] == "me@example.com"


def test_disconnect_clears_token(client):
    drive_auth.save_token({"access_token": "AT", "expiry": time.time() + 9999})
    assert client.post("/v1/drive/disconnect", headers=H).status_code == 200
    assert drive_auth.load_token() is None


def test_files_requires_connection(client):
    r = client.get("/v1/drive/files", headers=H)
    assert r.status_code == 409  # not connected


def test_files_lists_when_connected(client, monkeypatch):
    drive_auth.save_token({"access_token": "AT", "refresh_token": "RT",
                           "expiry": time.time() + 9999})

    async def fake_token():
        return "AT"

    async def fake_process(self, input_data, params=None):
        assert params.get("operation") == "list"
        assert params.get("access_token") == "AT"
        return {"status": "success", "files": [
            {"id": "f1", "name": "Plan.pdf", "type": "pdf"}]}

    monkeypatch.setattr(drive_auth, "get_access_token", fake_token)
    from app.blocks.google_drive import GoogleDriveBlock
    monkeypatch.setattr(GoogleDriveBlock, "process", fake_process)

    r = client.get("/v1/drive/files?q=plan", headers=H)
    assert r.status_code == 200
    assert r.json()["files"][0]["name"] == "Plan.pdf"


def test_files_requires_auth(client):
    assert client.get("/v1/drive/files").status_code == 401
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest tests/test_drive_router.py -q`
Expected: the 6 new tests FAIL (routes missing).

- [ ] **Step 3: Append the routes to `app/routers/drive.py`**

```python
@router.get("/v1/drive/status")
async def drive_status(auth: dict = Depends(require_api_key)):
    token = drive_auth.load_token()
    return {
        "connected": token is not None,
        "email": (token or {}).get("email") or None,
        "configured": _configured(),
    }


@router.post("/v1/drive/disconnect")
async def drive_disconnect(auth: dict = Depends(require_api_key)):
    cleared = drive_auth.clear_token()
    return {"status": "ok", "was_connected": cleared}


@router.get("/v1/drive/files")
async def drive_files(q: str = Query(""),
                      auth: dict = Depends(require_api_key)):
    try:
        access_token = await drive_auth.get_access_token()
    except drive_auth.DriveNotConnected:
        raise HTTPException(409, "Google Drive is not connected.")
    except drive_auth.DriveAuthError as e:
        raise HTTPException(409, f"{e} Reconnect Google Drive.")
    from app.blocks.google_drive import GoogleDriveBlock
    result = await GoogleDriveBlock().process(
        q, {"operation": "list", "access_token": access_token, "limit": 50})
    if result.get("status") != "success":
        raise HTTPException(502, result.get("error", "Drive list failed."))
    return {"files": result.get("files", [])}
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest tests/test_drive_router.py -q`
Expected: PASS (10 tests).

- [ ] **Step 5: Commit**

```bash
git add app/routers/drive.py tests/test_drive_router.py
git commit -m "feat(drive): status, disconnect, file listing routes"
```

---

### Task 4: per-project Drive import

**Files:**
- Modify: `app/routers/drive.py`
- Test: `tests/test_drive_router.py` (append)

The import downloads a Drive file and stores it as a project document through
the SAME path an upload takes. First read `app/routers/projects.py`'s
`add_document` endpoint and `app/routers/upload.py` to see exactly how an
uploaded file is written (`file_crypto.write_document`) and registered
(`app/core/projects.add_document(...)`), plus how the stored filename is
derived. Reuse those calls — do not invent a parallel storage path.

- [ ] **Step 1: Write the failing test** (append to `tests/test_drive_router.py`)

```python
def test_drive_import_adds_project_document(client, monkeypatch):
    # a real project to import into
    proj = client.post("/v1/projects", headers=H, json={"name": "Drive Test"}).json()
    pid = proj["id"]

    drive_auth.save_token({"access_token": "AT", "refresh_token": "RT",
                           "expiry": time.time() + 9999})

    async def fake_token():
        return "AT"

    async def fake_process(self, input_data, params=None):
        assert params.get("operation") == "download"
        return {"status": "success", "file_id": "f1",
                "filename": "Spec.pdf",
                "content_base64": __import__("base64").b64encode(b"PDFDATA").decode()}

    monkeypatch.setattr(drive_auth, "get_access_token", fake_token)
    from app.blocks.google_drive import GoogleDriveBlock
    monkeypatch.setattr(GoogleDriveBlock, "process", fake_process)

    r = client.post(f"/v1/projects/{pid}/drive/import", headers=H,
                     json={"file_id": "f1", "name": "Spec.pdf"})
    assert r.status_code in (200, 201), r.text
    body = r.json()
    assert body["status"] == "stored"
    assert body["document"]["name"] == "Spec.pdf"
    # the document now shows up on the project
    docs = client.get(f"/v1/projects/{pid}/documents", headers=H).json()
    assert any(d["name"] == "Spec.pdf" for d in docs.get("documents", docs))


def test_drive_import_requires_connection(client):
    proj = client.post("/v1/projects", headers=H, json={"name": "P2"}).json()
    r = client.post(f"/v1/projects/{proj['id']}/drive/import", headers=H,
                     json={"file_id": "f1"})
    assert r.status_code == 409
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest tests/test_drive_router.py -q`
Expected: the 2 new tests FAIL.

- [ ] **Step 3: Append the import route to `app/routers/drive.py`**

Implement `POST /v1/projects/{project_id}/drive/import`. Use the real symbols
discovered in the task preamble — verify the project exists (the same
not-found check `projects.py` uses, → 404), call
`GoogleDriveBlock().process(file_id, {"operation": "download", "access_token": ...})`,
base64-decode `content_base64`, write the bytes with the SAME
`file_crypto.write_document` call + stored-filename scheme `upload.py` uses,
register the document with the SAME `app.core.projects` add-document call
`projects.py` uses, and return the same `{status, message, document, ...}`
shape `projects.py`'s `add_document` returns. Reuse `_get_access_token` error
handling from Task 3 (→ 409 when not connected). The request body is
`{"file_id": str, "name": str|optional}` — fall back to the block's returned
`filename` when `name` is omitted.

- [ ] **Step 4: Run tests, verify they pass**

Run: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest tests/test_drive_router.py -q`
Expected: PASS (12 tests).

- [ ] **Step 5: Regression check + commit**

Run: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest tests/ -q --ignore=tests/browser`
Expected: PASS (≥ 437 passed — the prior 425 plus the new drive tests).

```bash
git add app/routers/drive.py tests/test_drive_router.py
git commit -m "feat(drive): import a Drive file into a project as a document"
```

---

### Task 5: live acceptance test

**Files:**
- Create: `tests/test_drive_router_live.py`

- [ ] **Step 1: Write the live test**

```python
"""Live Google Drive acceptance test — skipped unless GOOGLE_CLIENT_ID is set
AND a token has been obtained by completing the OAuth flow once in a browser.
Run manually after connecting Drive."""
import os
import pytest
from app.core import drive_auth

pytestmark = pytest.mark.skipif(
    not os.getenv("GOOGLE_CLIENT_ID") or drive_auth.load_token() is None,
    reason="Needs GOOGLE_CLIENT_ID and a connected Drive (run /v1/drive/connect first)",
)


@pytest.mark.asyncio
async def test_live_get_access_token():
    token = await drive_auth.get_access_token()
    assert isinstance(token, str) and len(token) > 10


@pytest.mark.asyncio
async def test_live_list_files():
    from app.blocks.google_drive import GoogleDriveBlock
    token = await drive_auth.get_access_token()
    result = await GoogleDriveBlock().process(
        "", {"operation": "list", "access_token": token, "limit": 5})
    assert result["status"] == "success"
    assert isinstance(result["files"], list)
```

- [ ] **Step 2: Run it (will skip)**

Run: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest tests/test_drive_router_live.py -q`
Expected: 2 skipped (no token yet).

- [ ] **Step 3: Commit**

```bash
git add tests/test_drive_router_live.py
git commit -m "test(drive): live acceptance test, skipif on connection"
```

---

### Task 6: frontend — wire the Connect Drive modal + Drive browser

**Files:**
- Modify: `app/static/index.html`
- Test: `tests/test_project_ui.py` (append markup/JS smoke checks)

First read `app/static/index.html` — the existing `showDriveModal()`, the
`.drive-option` rows in that modal, the `drivesList`/`selectDriveFile` helpers,
the `API_KEY`/`API_BASE` constants, `escapeHtml()`, and `activeProjectId`.

- [ ] **Step 1: Implement the frontend wiring**

In `app/static/index.html`:
1. On load and when the Connect Drive modal opens, call
   `GET /v1/drive/status` (with the `Authorization` header). Render one of:
   - `configured === false` → "Google Drive not configured" (disabled).
   - `connected === false` → a "Connect Google Drive" button whose click does:
     `const r = await fetch(API_BASE + '/v1/drive/connect', {headers: {Authorization: 'Bearer ' + API_KEY}});`
     `const {auth_url} = await r.json(); window.location = auth_url;`
     `/v1/drive/connect` returns the consent URL as JSON (see Task 2), so this
     is an ordinary same-origin fetch — the Bearer header attaches fine — and
     the browser navigation to Google happens client-side. No key in any URL.
   - `connected === true` → "Connected as &lt;email&gt;" + a Disconnect button
     (`POST /v1/drive/disconnect`).
2. A Drive file browser inside the modal (or the sidebar drive section): a
   search input → `GET /v1/drive/files?q=` → render rows (escape names with
   `escapeHtml`). Each row has an "Add to project" action → if `activeProjectId`
   is set, `POST /v1/projects/{activeProjectId}/drive/import` with
   `{file_id, name}`; on success show a system message and refresh the project's
   document list. If no project is active, prompt the user to pick one first.
3. All `fetch` calls send `Authorization: Bearer ${API_KEY}`.

- [ ] **Step 2: Add smoke checks** to `tests/test_project_ui.py`

```python
def test_ui_has_drive_status_check():
    html = open("app/static/index.html", encoding="utf-8").read()
    assert "/v1/drive/status" in html
    assert "/v1/drive/files" in html
    assert "/drive/import" in html
```

- [ ] **Step 3: Run UI tests**

Run: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest tests/test_project_ui.py -q`
Expected: PASS.

- [ ] **Step 4: Manual verification**

Start the app (`.venv/Scripts/python.exe .claude/skills/run-the-fork/driver.py`
to confirm it boots, then run it via the human path). Open
`http://localhost:8000/`, click Connect Drive → Google consent → approve →
land back on `/`. Confirm status shows "Connected as …", search lists real
files, "Add to project" imports one into the active project.

- [ ] **Step 5: Regression check + commit**

Run: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest tests/ -q --ignore=tests/browser`
Expected: PASS.

```bash
git add app/static/index.html tests/test_project_ui.py
git commit -m "feat(drive): wire Connect Drive modal + Drive file browser"
```

---

## Notes for the implementer

- **No new dependencies.** Everything uses `httpx` (already installed). Do not
  uncomment the `google-*` libraries in `requirements.txt`.
- **The `/v1/drive/connect` auth-vs-navigation wrinkle is resolved:**
  `/v1/drive/connect` returns the consent URL as JSON (`{auth_url}`), stays
  gated with `Depends(require_api_key)`, and the frontend fetches it with the
  Bearer header then does `window.location = auth_url`. No `?key=` param, no
  cross-origin-redirect reading. Build it exactly that way.
- `GoogleDriveBlock.process` `download` returns `content_base64`; confirm
  whether it also returns a `filename` — if not, the import route must take the
  name from the request body or a prior `list` result.
