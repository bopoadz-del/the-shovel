import time
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.core import drive_auth
from tests._dev_key import DEV_KEY
from tests._fixture_values import ENV_SECRET_05

H = {"Authorization": "Bearer " + DEV_KEY}


@pytest.fixture(autouse=True)
def tmp_data(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "cid")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", ENV_SECRET_05)
    monkeypatch.setenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/v1/drive/callback")
    yield


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_connect_returns_auth_url(client):
    # /connect returns the consent URL as JSON (NOT a redirect) so the browser
    # can fetch it with the Bearer header, then navigate client-side.
    r = client.get("/v1/drive/connect", headers=H)
    assert r.status_code == 200
    url = r.json()["auth_url"]
    assert url.startswith("https://accounts.google.com/o/oauth2/v2/auth")
    assert "scope=" in url and "drive.readonly" in url
    assert "state=" in url and "access_type=offline" in url


def test_connect_requires_auth(client):
    assert client.get("/v1/drive/connect").status_code == 401


def test_callback_bad_state_redirects_into_app(client):
    # A forged/garbage state must NOT strand the user on a bare 400 JSON page
    # (that read as "the OAuth round-trip boots me out of the app") — it
    # redirects back into the SPA with a reason slug, session untouched.
    r = client.get("/v1/drive/callback?code=x&state=never-issued",
                    follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["location"] == "http://localhost:5173/?drive=error&reason=invalid_state"


def test_callback_tampered_state_redirects_into_app(client):
    r = client.get("/v1/drive/connect", headers=H)
    state = r.json()["auth_url"].split("state=")[1].split("&")[0]
    raw, _sig = state.rsplit(".", 1)
    r = client.get(f"/v1/drive/callback?code=x&state={raw}.{'0' * 32}",
                    follow_redirects=False)
    assert r.status_code == 302
    assert "drive=error" in r.headers["location"]
    assert "reason=invalid_state" in r.headers["location"]


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
    assert r.headers["location"] == "http://localhost:5173/?drive=connected"
    tok = drive_auth.load_token("system")
    assert tok["access_token"] == "AT" and tok["refresh_token"] == "RT"
    assert tok["email"] == "me@example.com"
    assert tok["expiry"] > time.time()


def test_callback_consent_denied(client):
    # User clicked "Deny": Google redirects back with `error` and no `code`.
    # The state was issued by /connect, so it is a valid (consumable) state —
    # the callback must redirect gracefully, not 400, and store no token.
    r = client.get("/v1/drive/connect", headers=H)
    state = r.json()["auth_url"].split("state=")[1].split("&")[0]

    r = client.get(f"/v1/drive/callback?error=access_denied&state={state}",
                    follow_redirects=False)
    assert r.status_code in (302, 307)
    assert r.headers["location"] == "http://localhost:5173/?drive=error&reason=denied"
    assert drive_auth.load_token("system") is None


def test_callback_expired_state_redirects_into_app(client, monkeypatch):
    # A state older than _STATE_TTL fails verification → redirect with
    # reason=expired_state, never a 400.
    import app.routers.drive as drive_mod
    real_time = time.time
    monkeypatch.setattr(drive_mod.time, "time",
                        lambda: real_time() - drive_mod._STATE_TTL - 5)
    r = client.get("/v1/drive/connect", headers=H)
    state = r.json()["auth_url"].split("state=")[1].split("&")[0]
    monkeypatch.setattr(drive_mod.time, "time", real_time)

    r = client.get(f"/v1/drive/callback?code=x&state={state}",
                    follow_redirects=False)
    assert r.status_code == 302
    assert "reason=expired_state" in r.headers["location"]


def test_state_survives_process_restart(client, monkeypatch):
    # THE bug: the old process-local dict lost every pending state on a
    # deploy/restart mid-consent, so Google's redirect hit "Invalid or
    # expired OAuth state" and dumped the user outside the app. The signed
    # state must verify with zero server-side memory of it.
    import app.routers.drive as drive_mod
    r = client.get("/v1/drive/connect", headers=H)
    state = r.json()["auth_url"].split("state=")[1].split("&")[0]

    drive_mod._consumed_states.clear()  # simulate a fresh process

    async def fake_exchange(code):
        return {"access_token": "AT", "refresh_token": "RT", "expires_in": 3600}

    async def fake_email(access_token):
        return "me@example.com"

    monkeypatch.setattr(drive_mod, "_exchange_code", fake_exchange)
    monkeypatch.setattr(drive_mod, "_fetch_email", fake_email)
    r = client.get(f"/v1/drive/callback?code=auth-code&state={state}",
                    follow_redirects=False)
    assert r.status_code == 302
    assert "drive=connected" in r.headers["location"]
    assert drive_auth.load_token("system")["access_token"] == "AT"


def test_state_single_use_within_process(client, monkeypatch):
    import app.routers.drive as drive_mod
    r = client.get("/v1/drive/connect", headers=H)
    state = r.json()["auth_url"].split("state=")[1].split("&")[0]

    async def fake_exchange(code):
        return {"access_token": "AT", "expires_in": 3600}

    async def fake_email(access_token):
        return "me@example.com"

    monkeypatch.setattr(drive_mod, "_exchange_code", fake_exchange)
    monkeypatch.setattr(drive_mod, "_fetch_email", fake_email)
    first = client.get(f"/v1/drive/callback?code=c&state={state}",
                       follow_redirects=False)
    assert "drive=connected" in first.headers["location"]
    replay = client.get(f"/v1/drive/callback?code=c&state={state}",
                        follow_redirects=False)
    assert "reason=invalid_state" in replay.headers["location"]


def test_callback_exchange_failure_redirects_into_app(client, monkeypatch):
    import app.routers.drive as drive_mod
    r = client.get("/v1/drive/connect", headers=H)
    state = r.json()["auth_url"].split("state=")[1].split("&")[0]

    async def boom(code):
        raise drive_auth.DriveAuthError("Code exchange failed (HTTP 400)")

    monkeypatch.setattr(drive_mod, "_exchange_code", boom)
    r = client.get(f"/v1/drive/callback?code=bad&state={state}",
                    follow_redirects=False)
    assert r.status_code == 302
    assert "reason=exchange_failed" in r.headers["location"]
    assert drive_auth.load_token("system") is None


def test_connect_return_to_lands_back_on_same_page(client, monkeypatch):
    import app.routers.drive as drive_mod
    r = client.get("/v1/drive/connect?return_to=/projects/p1", headers=H)
    state = r.json()["auth_url"].split("state=")[1].split("&")[0]

    async def fake_exchange(code):
        return {"access_token": "AT", "expires_in": 3600}

    async def fake_email(access_token):
        return "me@example.com"

    monkeypatch.setattr(drive_mod, "_exchange_code", fake_exchange)
    monkeypatch.setattr(drive_mod, "_fetch_email", fake_email)
    r = client.get(f"/v1/drive/callback?code=c&state={state}",
                    follow_redirects=False)
    assert r.headers["location"] == "http://localhost:5173/projects/p1?drive=connected"


def test_connect_return_to_rejects_external_urls(client):
    # A hostile return_to must not become an open redirect off-origin.
    for evil in ("https://evil.example", "//evil.example", "javascript:x"):
        r = client.get(f"/v1/drive/connect?return_to={evil}", headers=H)
        state = r.json()["auth_url"].split("state=")[1].split("&")[0]
        cb = client.get(f"/v1/drive/callback?error=access_denied&state={state}",
                        follow_redirects=False)
        loc = cb.headers["location"]
        assert loc.startswith("http://localhost:5173/?"), loc


def test_status_not_connected(client):
    r = client.get("/v1/drive/status", headers=H)
    assert r.status_code == 200
    body = r.json()
    assert body["connected"] is False and body["configured"] is True


def test_status_connected_after_token(client):
    drive_auth.save_token("system", {"access_token": "AT", "refresh_token": "RT",
                           "expiry": time.time() + 9999, "email": "me@example.com"})
    body = client.get("/v1/drive/status", headers=H).json()
    assert body["connected"] is True and body["email"] == "me@example.com"


def test_disconnect_clears_token(client, monkeypatch):
    revoked = []

    async def fake_revoke(token):
        revoked.append(token)
        return True

    monkeypatch.setattr(drive_auth, "_revoke_request", fake_revoke)
    drive_auth.save_token("system", {"access_token": "AT", "expiry": time.time() + 9999})
    body = client.post("/v1/drive/disconnect", headers=H).json()
    assert body["was_connected"] is True and body["revoked"] is True
    assert revoked == ["AT"]
    assert drive_auth.load_token("system") is None


def test_files_requires_connection(client):
    r = client.get("/v1/drive/files", headers=H)
    assert r.status_code == 409  # not connected


def test_files_lists_when_connected(client, monkeypatch):
    drive_auth.save_token("system", {"access_token": "AT", "refresh_token": "RT",
                           "expiry": time.time() + 9999})

    async def fake_token(*_a, **_k):
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


@pytest.mark.usefixtures("dev_key_is_admin")
def test_drive_import_adds_project_document(client, monkeypatch):
    proj = client.post("/v1/projects", headers=H, json={"name": "Drive Test"}).json()
    pid = proj["id"]

    drive_auth.save_token("system", {"access_token": "AT", "refresh_token": "RT",
                           "expiry": time.time() + 9999})

    async def fake_token(*_a, **_k):
        return "AT"

    async def fake_process(self, input_data, params=None):
        assert params.get("operation") == "download"
        # Matches the REAL GoogleDriveBlock download response shape — no
        # `filename` key; the route stores the file under the request `name`.
        return {"status": "success", "operation": "download", "file_id": "f1",
                "size_bytes": 7,
                "content_base64": __import__("base64").b64encode(b"PDFDATA").decode()}

    monkeypatch.setattr(drive_auth, "get_access_token", fake_token)
    from app.blocks.google_drive import GoogleDriveBlock
    monkeypatch.setattr(GoogleDriveBlock, "process", fake_process)

    r = client.post(f"/v1/projects/{pid}/drive/import", headers=H,
                     json={"file_id": "f1", "name": "Spec.pdf"})
    assert r.status_code in (200, 201), r.text
    body = r.json()
    assert body["status"] == "stored"
    # `add_document` stores the document name under `original_name` (the real
    # DB column / response key) — not `name`.
    assert body["document"]["original_name"] == "Spec.pdf"
    # There is no GET /v1/projects/{id}/documents route; documents are exposed
    # via the project-detail route, which `get_project` populates.
    detail = client.get(f"/v1/projects/{pid}", headers=H).json()
    assert any(d["original_name"] == "Spec.pdf" for d in detail["documents"])


@pytest.mark.usefixtures("dev_key_is_admin")
def test_drive_import_requires_connection(client):
    proj = client.post("/v1/projects", headers=H, json={"name": "P2"}).json()
    r = client.post(f"/v1/projects/{proj['id']}/drive/import", headers=H,
                     json={"file_id": "f1", "name": "x.pdf"})
    assert r.status_code == 409


@pytest.mark.usefixtures("dev_key_is_admin")
def test_drive_import_rejects_disallowed_extension(client, monkeypatch):
    proj = client.post("/v1/projects", headers=H, json={"name": "P3"}).json()
    pid = proj["id"]

    drive_auth.save_token("system", {"access_token": "AT", "refresh_token": "RT",
                           "expiry": time.time() + 9999})

    async def fake_token(*_a, **_k):
        return "AT"

    async def fake_process(self, input_data, params=None):
        return {"status": "success", "operation": "download", "file_id": "f1",
                "size_bytes": 4,
                "content_base64": __import__("base64").b64encode(b"EXE!").decode()}

    monkeypatch.setattr(drive_auth, "get_access_token", fake_token)
    from app.blocks.google_drive import GoogleDriveBlock
    monkeypatch.setattr(GoogleDriveBlock, "process", fake_process)

    r = client.post(f"/v1/projects/{pid}/drive/import", headers=H,
                     json={"file_id": "f1", "name": "evil.exe"})
    assert r.status_code == 415, r.text
    assert "not ingestible" in r.json()["detail"]
