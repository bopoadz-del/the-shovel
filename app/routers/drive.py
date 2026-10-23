"""Google Drive connection — /v1/drive/* OAuth flow + file import.

All routes require Authorization: Bearer like other /v1/* routes, EXCEPT
/v1/drive/callback — Google calls that directly and cannot send our header,
so it is protected by the signed OAuth `state` value instead.
"""
import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
import uuid
from typing import Any, Dict, Optional, Tuple

import httpx
from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from app.dependencies import require_user
from app.core import audit, doc_index, drive_auth, file_crypto, jwt_auth, privileges, projects as store
from app.routers import projects as projects_router
from app.routers.projects import ALLOWED_DOC_EXTENSIONS

router = APIRouter()

_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_TOKEN_URL = "https://oauth2.googleapis.com/token"
_SCOPE = "https://www.googleapis.com/auth/drive.readonly"
_DRIVE_API = "https://www.googleapis.com/drive/v3"

_STATE_TTL = 600  # seconds — OAuth states expire after 10 minutes.

# Replay guard for already-consumed states. Best-effort and process-local by
# design: the state itself is stateless (HMAC-signed, survives restarts and
# multiple workers — the fix for users getting dumped on an error page when a
# deploy landed mid-consent). If a restart empties this set, a replayed state
# still can't mint a token because Google auth codes are single-use.
_consumed_states: Dict[str, float] = {}

# In-memory registry for async Drive-folder imports keyed by job_id.
# Process-local; a restart loses pending jobs (acceptable for this path).
_DRIVE_FOLDER_JOBS: Dict[str, Dict[str, Any]] = {}


def _sign_state(raw: str) -> str:
    return hmac.new(jwt_auth.signing_secret().encode("utf-8"),
                    raw.encode("utf-8"), hashlib.sha256).hexdigest()[:32]


def _make_state(user_id: str, return_to: str = "") -> str:
    """Self-contained signed state: base64(payload).sig — no server storage.

    Carries the user this consent flow belongs to, an expiry, and the SPA
    path to send the user back to, all authenticated by the app secret."""
    payload = {
        "p": "drive_oauth",
        "u": user_id,
        "t": int(time.time()),
        "n": secrets.token_urlsafe(8),
        "r": return_to,
    }
    raw = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).rstrip(b"=").decode("ascii")
    return f"{raw}.{_sign_state(raw)}"


def _verify_state(state: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """Return (payload, "") if valid, else (None, reason-slug)."""
    if not state or "." not in state:
        return None, "invalid_state"
    raw, sig = state.rsplit(".", 1)
    if not hmac.compare_digest(_sign_state(raw), sig):
        return None, "invalid_state"
    try:
        pad = "=" * (-len(raw) % 4)
        payload = json.loads(base64.urlsafe_b64decode(raw + pad))
    except (ValueError, TypeError):
        return None, "invalid_state"
    if payload.get("p") != "drive_oauth":
        return None, "invalid_state"
    if time.time() - float(payload.get("t", 0)) > _STATE_TTL:
        return None, "expired_state"
    now = time.time()
    for s, seen in list(_consumed_states.items()):
        if now - seen > _STATE_TTL:
            del _consumed_states[s]
    if state in _consumed_states:
        return None, "invalid_state"
    _consumed_states[state] = now
    return payload, ""


def _safe_return_to(value: str) -> str:
    """Only same-app absolute paths — no scheme, no host, no protocol-relative."""
    if value and value.startswith("/") and not value.startswith("//"):
        return value
    return "/"


def _redirect_uri() -> str:
    return os.getenv("GOOGLE_REDIRECT_URI",
                     "http://localhost:8000/v1/drive/callback")


def _frontend_url() -> str:
    # NOTE: if FRONTEND_URL is a different origin, ensure it is included in the
    # CORS allow_origins list in app/main.py.
    return os.getenv("FRONTEND_URL", "http://localhost:5173").rstrip("/")


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
async def drive_connect(return_to: str = Query("", description="SPA path to land back on"),
                        auth: dict = Depends(require_user)):
    # Returns the Google consent URL as JSON — NOT a redirect. A browser cannot
    # attach the Bearer header to a top-level navigation, so the frontend
    # fetches this (header attaches fine on a same-origin fetch), reads
    # auth_url, and does window.location = auth_url itself. Keeps the route
    # gated like every other /v1/* and puts no key in any URL.
    if not _configured():
        raise HTTPException(503, "Google Drive not configured — set "
                                 "GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET.")
    state = _make_state(auth["user_id"], _safe_return_to(return_to))
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


def _back_to_app(return_to: str, outcome: str, reason: str = "") -> RedirectResponse:
    """302 back INTO the SPA — the callback must never strand the user on a
    bare JSON error page outside the app (that is what read as 'the OAuth
    round-trip boots me out'). The app session (localStorage JWT) is untouched
    by this whole flow, so landing back on an app path keeps them logged in."""
    from urllib.parse import urlencode
    q = {"drive": outcome}
    if reason:
        q["reason"] = reason
    path = _safe_return_to(return_to)
    return RedirectResponse(
        f"{_frontend_url()}{path}?{urlencode(q)}", status_code=302)


@router.get("/v1/drive/callback")
async def drive_callback(code: str = Query(""), state: str = Query(""),
                         error: str = Query("")):
    # No Bearer auth here — Google calls this. The signed state is the gate,
    # and it carries the user_id this consent flow belongs to. Every failure
    # path redirects back into the app with a reason slug — never a raw 4xx.
    payload, reason = _verify_state(state)
    if payload is None:
        return _back_to_app("/", "error", reason)
    user_id = str(payload.get("u", ""))
    return_to = str(payload.get("r", "") or "/")
    # User clicked "Deny" (or consent otherwise failed): Google sends `error`
    # and no `code`.
    if error:
        return _back_to_app(return_to, "error", "denied")
    if not code:
        return _back_to_app(return_to, "error", "missing_code")
    try:
        data = await _exchange_code(code)
        access_token = data["access_token"]
        email = await _fetch_email(access_token)
        drive_auth.save_token(user_id, {
            "access_token": access_token,
            "refresh_token": data.get("refresh_token", ""),
            "expiry": time.time() + int(data.get("expires_in", 3600)),
            "email": email,
        })
    except drive_auth.DriveAuthError:
        logging.getLogger(__name__).warning(
            "Drive OAuth code exchange failed for user %s", user_id)
        return _back_to_app(return_to, "error", "exchange_failed")
    except Exception:
        logging.getLogger(__name__).exception(
            "Drive OAuth callback failed for user %s", user_id)
        return _back_to_app(return_to, "error", "exchange_failed")
    return _back_to_app(return_to, "connected")


@router.get("/v1/drive/status")
async def drive_status(auth: dict = Depends(require_user)):
    token = drive_auth.load_token(auth["user_id"])
    return {
        "connected": token is not None,
        "email": (token or {}).get("email") or None,
        "configured": _configured(),
    }


@router.post("/v1/drive/disconnect")
async def drive_disconnect(auth: dict = Depends(require_user)):
    return {"status": "ok", **await drive_auth.revoke_and_clear(auth["user_id"])}


@router.get("/v1/drive/files")
async def drive_files(q: str = Query(""),
                      folder_id: str = Query("", description="Drive folder id; empty = root"),
                      auth: dict = Depends(require_user)):
    """List Drive files/folders.

    When ``folder_id`` is empty, returns items at the root of the user's Drive
    (root-level folders + loose files). When ``folder_id`` is set, returns
    that folder's children — letting the UI drill into subfolders. A free-text
    ``q`` query is a name-contains search that ignores folder_id (search is
    Drive-wide). Response items carry ``mime_type`` so the UI can distinguish
    folders (``application/vnd.google-apps.folder``) from files.
    """
    try:
        access_token = await drive_auth.get_access_token(auth["user_id"])
    except drive_auth.DriveNotConnected:
        raise HTTPException(409, "Google Drive is not connected.")
    except drive_auth.DriveAuthError as e:
        raise HTTPException(409, f"{e} Reconnect Google Drive.")
    from app.blocks.google_drive import GoogleDriveBlock
    result = await GoogleDriveBlock().process(
        q,
        {
            "operation": "list",
            "access_token": access_token,
            "limit": 100,
            "folder_id": folder_id or None,
        },
    )
    if result.get("status") != "success":
        raise HTTPException(502, result.get("error", "Drive list failed."))
    return {"files": result.get("files", []), "folder_id": folder_id or "root"}


# ── per-project Drive import — store a Drive file as a project document ──────

class DriveImportRequest(BaseModel):
    file_id: str
    name: str


class DriveIndexFolderRequest(BaseModel):
    folder_id: str | None = None        # default: My Drive root
    max_files: int = 100                 # hard cap so a Drive of 10k files can't DoS
    max_depth: int = 4                   # how deep to recurse
    role: str = "other"                  # doc_role to tag the imports with
    include_extensions: list[str] | None = None  # narrows ALLOWED_DOC_EXTENSIONS; never widens it


async def _run_index_folder_bg(
    *,
    project_id: str,
    user_id: str,
    folder_id: str | None,
    max_files: int,
    max_depth: int,
    role: str,
    include_extensions: list[str] | None,
    job_id: str,
) -> None:
    """Background worker: walk a Drive folder into ``project_id``, then DRAIN
    the per-file indexing tasks so chunks are written before the worker exits.
    Mirrors ``admin._run_drive_folder_import`` so a folder import never blocks
    the HTTP request — a synchronous walk of a real folder (download + encrypt
    + store + index each file) easily exceeds the Render proxy timeout (502)."""
    log = logging.getLogger(__name__)
    job = _DRIVE_FOLDER_JOBS.get(job_id)
    if job is not None:
        job["status"] = "running"
    try:
        access_token = await drive_auth.get_access_token(user_id)
        _bg = BackgroundTasks()
        walk_result = await _walk_drive_folder_into_project(
            project_id=project_id,
            user_id=user_id,
            access_token=access_token,
            folder_id=folder_id,
            max_files=max_files,
            max_depth=max_depth,
            role=role,
            background_tasks=_bg,
            include_extensions=include_extensions,
        )
        # Drain the queued maybe_eager_index tasks inline so the chunks land
        # before this worker returns (keeps doc/chunk counters consistent).
        zero_chunk_docs: list[Dict[str, Any]] = []
        for task in _bg.tasks:
            if asyncio.iscoroutinefunction(task.func):
                idx_result = await task.func(*task.args, **task.kwargs)
            else:
                idx_result = task.func(*task.args, **task.kwargs)
            if isinstance(idx_result, dict) and idx_result.get("error") == "ZERO_CHUNK":
                zero_chunk_docs.append(idx_result)

        imported_count = walk_result.get("imported_count", 0)
        if zero_chunk_docs:
            banner = (
                f"ERROR: {len(zero_chunk_docs)} imported Drive file(s) produced 0 chunks — "
                "check extractor/OCR logs."
            )
            if job is not None:
                job["status"] = "error"
                job["banner"] = banner
                job["imported_count"] = imported_count
                job["zero_chunk_count"] = len(zero_chunk_docs)
            log.error(banner)
            return

        if job is not None:
            job["status"] = "done"
            job["imported_count"] = imported_count
            job["skipped_count"] = walk_result.get("skipped_count", 0)
    except Exception as exc:  # noqa: BLE001 — background; never raise
        log.exception(
            "index-folder bg import failed project=%s folder=%s: %s",
            project_id, folder_id, exc,
        )
        if job is not None:
            job["status"] = "error"
            job["error"] = f"{type(exc).__name__}: {exc}"


@router.post("/v1/projects/{project_id}/drive/index-folder", status_code=202)
async def drive_index_folder(project_id: str, req: DriveIndexFolderRequest,
                             background_tasks: BackgroundTasks,
                             auth: dict = Depends(require_user)):
    """Queue an async Drive-folder import into the project.

    Validates ownership + the Drive connection up front, then runs the
    recursive walk (download → encrypt-at-rest → RAG index, up to ``max_depth``
    / ``max_files``) in the BACKGROUND and returns immediately. A synchronous
    walk of a real folder exceeds the request timeout (502); the import now
    progresses after the response and the caller polls the project's document
    list for progress.
    """
    proj = store.get_project(project_id, user_id=auth["user_id"])
    if not proj:
        raise HTTPException(404, f"Project '{project_id}' not found")
    privileges.raise_unless_may_add_to_project_rag(auth.get("role"))
    # Validate the connection now so a disconnected Drive fails fast (409)
    # instead of silently no-op'ing in the background.
    try:
        await drive_auth.get_access_token(auth["user_id"])
    except drive_auth.DriveNotConnected:
        raise HTTPException(409, "Google Drive is not connected.")
    except drive_auth.DriveAuthError as e:
        raise HTTPException(409, f"{e} Reconnect Google Drive.")

    job_id = uuid.uuid4().hex[:12]
    _DRIVE_FOLDER_JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "project_id": project_id,
        "folder_id": req.folder_id,
        "created_at": time.time(),
    }
    background_tasks.add_task(
        _run_index_folder_bg,
        project_id=project_id,
        user_id=auth["user_id"],
        folder_id=req.folder_id,
        max_files=req.max_files,
        max_depth=req.max_depth,
        role=req.role,
        include_extensions=req.include_extensions,
        job_id=job_id,
    )
    return {
        "status": "queued",
        "job_id": job_id,
        "status_url": f"/v1/projects/{project_id}/drive/index-folder/job/{job_id}",
        "project_id": project_id,
        "folder_id": req.folder_id,
        "max_files": req.max_files,
        "max_depth": req.max_depth,
    }


@router.get("/v1/projects/{project_id}/drive/index-folder/job/{job_id}")
async def drive_index_folder_status(
    project_id: str,
    job_id: str,
    auth: dict = Depends(require_user),
):
    """Poll the status of an async Drive-folder import job."""
    # Same owner-only grant as POST index-folder. Job lookup alone is not
    # enough: the registry is process-global, so a caller who knows
    # project_id + job_id must still own the project.
    proj = store.get_project(project_id, user_id=auth["user_id"])
    if not proj:
        raise HTTPException(404, "job not found")
    job = _DRIVE_FOLDER_JOBS.get(job_id)
    if not job or job.get("project_id") != project_id:
        raise HTTPException(404, "job not found")
    return job


async def _walk_drive_folder_into_project(
    *,
    project_id: str,
    user_id: str,
    access_token: str,
    folder_id: str | None,
    max_files: int = 100,
    max_depth: int = 4,
    role: str = "other",
    background_tasks: BackgroundTasks | None = None,
    include_extensions: list[str] | None = None,
) -> Dict[str, Any]:
    """Recursively import every supported file under ``folder_id`` into
    ``project_id``.

    This is the shared engine behind both the user-facing
    ``POST /v1/projects/{id}/drive/index-folder`` route and the admin
    ``approve-from-drive`` background worker. It walks the Drive tree using
    OAuth ``access_token``, paginates through large folders, and asks Drive to
    include items from shared drives so approved project folders that live on a
    shared drive still recurse correctly.
    """
    # A caller may narrow the formats, never widen them past the one text-only
    # declaration (docs/INGEST_EXCLUSION_RULE.md).
    allowed = set(ALLOWED_DOC_EXTENSIONS)
    if include_extensions:
        allowed &= {ext.lower() for ext in include_extensions}
    folder_mt = "application/vnd.google-apps.folder"
    imported: list[Dict[str, Any]] = []
    skipped: list[Dict[str, Any]] = []

    # Track folder path while recursing so each imported file carries its
    # Drive location as provenance metadata.
    folder_paths: Dict[str, str] = {}

    async def _list_folder(client: httpx.AsyncClient, fid: str) -> list[Dict[str, Any]]:
        q = f"'{fid}' in parents and trashed=false"
        out: list[Dict[str, Any]] = []
        page_token: str | None = None
        while True:
            params: Dict[str, Any] = {
                "q": q,
                "pageSize": 200,
                "orderBy": "folder,name",
                "fields": "nextPageToken, files(id,name,mimeType,size,parents)",
                "supportsAllDrives": "true",
                "includeItemsFromAllDrives": "true",
            }
            if page_token:
                params["pageToken"] = page_token
            r = await client.get(
                f"{_DRIVE_API}/files",
                headers={"Authorization": f"Bearer {access_token}"},
                params=params,
            )
            r.raise_for_status()
            payload = r.json()
            out.extend(payload.get("files", []))
            page_token = payload.get("nextPageToken")
            if not page_token:
                break
        return out

    async def _download_bytes(client: httpx.AsyncClient, file_id: str, mime: str, name: str):
        """Native Google Docs/Sheets/Slides need /export?mimeType=...; everything
        else uses ?alt=media. Returns (bytes, exported_extension)."""
        from app.core import drive_mime

        meta = await client.get(
            f"{_DRIVE_API}/files/{file_id}",
            headers={"Authorization": f"Bearer {access_token}"},
            params={"fields": "mimeType,name,shortcutDetails", "supportsAllDrives": "true"},
        )
        meta.raise_for_status()
        meta_json = meta.json()
        mime = meta_json.get("mimeType", mime)
        name = meta_json.get("name", name) or name

        # Follow Drive shortcuts so approved folders that contain shortcuts to
        # documents still import the target file.
        if mime == "application/vnd.google-apps.shortcut":
            sd = meta_json.get("shortcutDetails") or {}
            target_id = sd.get("targetId")
            target_mime = sd.get("targetMimeType") or ""
            if not target_id:
                raise ValueError("shortcut has no target")
            file_id = target_id
            mime = target_mime

        target = drive_mime.export_target(mime)
        if target is not None:
            export_mime, ext = target
            r = await client.get(
                f"{_DRIVE_API}/files/{file_id}/export",
                headers={"Authorization": f"Bearer {access_token}"},
                params={"mimeType": export_mime, "supportsAllDrives": "true"},
            )
            r.raise_for_status()
            return r.content, ext
        r = await client.get(
            f"{_DRIVE_API}/files/{file_id}",
            headers={"Authorization": f"Bearer {access_token}"},
            params={"alt": "media", "supportsAllDrives": "true"},
        )
        r.raise_for_status()
        return r.content, os.path.splitext(name)[1].lower()

    def _schedule_index(doc_id: str) -> None:
        if background_tasks is not None:
            background_tasks.add_task(doc_index.maybe_eager_index, project_id, doc_id)
        else:
            doc_index.maybe_eager_index(project_id, doc_id)

    # Drive file id -> the visible document imported from it, read ONCE.
    imported_from: Dict[str, str] = {}
    for d in store.list_documents(project_id):
        if d.get("retrieval_visible", True):
            fid = store.extract_document_source_pointers(d).get("drive_file_id")
            if fid:
                imported_from[fid] = d["id"]

    async with httpx.AsyncClient(timeout=60) as client:
        start = folder_id or "root"
        folder_paths[start] = ""
        stack: list[tuple[str, int, str]] = [(start, 0, "")]
        while stack and len(imported) < max_files:
            fid, depth, path_prefix = stack.pop(0)
            try:
                children = await _list_folder(client, fid)
            except Exception as e:
                skipped.append({"folder_id": fid, "reason": f"list failed: {str(e)[:80]}"})
                continue
            for child in children:
                if len(imported) >= max_files:
                    skipped.append({"name": child.get("name"), "reason": "max_files cap"})
                    continue
                child_name = child.get("name") or "unnamed"
                child_path = f"{path_prefix}/{child_name}".lstrip("/")
                if child.get("mimeType") == folder_mt:
                    if depth + 1 <= max_depth:
                        folder_paths[child["id"]] = child_path
                        stack.append((child["id"], depth + 1, child_path))
                    else:
                        skipped.append({"name": child.get("name"), "reason": "max_depth cap"})
                    continue
                # Pre-flight extension check (override for native Google types).
                mime = child.get("mimeType", "")
                name_raw = child.get("name") or "unnamed"
                pre_ext = os.path.splitext(name_raw.lower())[1]
                if mime.startswith("application/vnd.google-apps."):
                    # Will be exported to a supported ext below; skip the
                    # pre-check.
                    pass
                elif pre_ext not in allowed:
                    skipped.append({"name": name_raw, "mime": mime, "reason": f"extension {pre_ext} not in allowlist"})
                    continue
                try:
                    raw_bytes, exported_ext = await _download_bytes(
                        client, child["id"], mime, name_raw,
                    )
                except Exception as e:
                    skipped.append({"name": name_raw, "reason": f"download failed: {str(e)[:80]}"})
                    continue
                # A 0-byte file was stored and indexed as a document that can
                # never answer anything. Say so instead.
                if not raw_bytes:
                    skipped.append({"name": name_raw, "reason": "empty file (0 bytes)"})
                    continue
                # Re-validate the post-export extension for native Google types.
                if exported_ext and exported_ext not in allowed:
                    skipped.append({"name": name_raw, "reason": f"exported ext {exported_ext} not allowed"})
                    continue
                # Same storage path as upload/import.
                stored_basename = name_raw
                if mime.startswith("application/vnd.google-apps.") and exported_ext:
                    # Strip any old extension, append the exported one so
                    # downstream parsers recognise the format.
                    root, _ = os.path.splitext(name_raw)
                    stored_basename = f"{root}{exported_ext}"
                stored_basename = os.path.basename(stored_basename.replace("\\", "/"))
                # SHA-256 dedupe: skip files whose bytes are already in the
                # project so a re-walk doesn't re-encrypt/re-index them.
                import hashlib as _hashlib
                content_sha = _hashlib.sha256(raw_bytes).hexdigest()
                existing = store.find_document_by_sha(project_id, content_sha)
                if existing:
                    skipped.append({
                        "name": stored_basename,
                        "reason": f"unchanged (sha {content_sha[:12]}...)",
                    })
                    continue
                # The SAME Drive file with new bytes replaces what was imported
                # from it: the old version was kept searchable next to the new
                # one, so an answer could come from either. Superseded, never
                # deleted (see store.add_document reingest_of).
                previous = imported_from.get(child["id"])
                file_uuid = str(uuid.uuid4())[:8]
                stored_as = f"{file_uuid}_{stored_basename}"
                filepath = os.path.join(projects_router.DATA_DIR, stored_as)
                file_crypto.write_document(filepath, raw_bytes)
                doc = store.add_document(
                    project_id, stored_basename, stored_as,
                    filepath, len(raw_bytes),
                    content_sha256=content_sha,
                    reingest_of=previous,
                    metadata={
                        "drive_file_id": child["id"],
                        "drive_path": child_path,
                        "source": "drive_oauth_folder",
                    },
                )
                await audit.arecord("document.added", project_id=project_id,
                             document_id=doc["id"], name=stored_basename,
                             size=len(raw_bytes), user_id=user_id,
                             source="drive_walker")
                imported_from[child["id"]] = doc["id"]
                _schedule_index(doc["id"])
                imported.append({
                    "drive_id": child["id"],
                    "name": stored_basename,
                    "doc_id": doc["id"],
                    "size": len(raw_bytes),
                    "doc_type": doc.get("doc_type"),
                })

    return {
        "status": "ok",
        "imported_count": len(imported),
        "skipped_count": len(skipped),
        "imported": imported,
        "skipped": skipped[:50],
        "readiness": store.compute_readiness(project_id),
    }


@router.post("/v1/projects/{project_id}/drive/import", status_code=201)
async def drive_import(project_id: str, req: DriveImportRequest,
                       background_tasks: BackgroundTasks,
                       auth: dict = Depends(require_user)):
    """Import a Google Drive file into a project as a document.

    Downloads the file via the Drive block and stores it through the SAME
    path an uploaded file takes (`projects.py add_document`): encrypted at
    rest, registered as a project document, no analysis run. A Drive-imported
    file is therefore indistinguishable from an uploaded one.
    """
    # Ownership check — scoped to the calling user (same 404-not-403 pattern
    # used in projects.py to avoid leaking project existence to other users).
    proj = store.get_project(project_id, user_id=auth["user_id"])
    if not proj:
        raise HTTPException(404, f"Project '{project_id}' not found")
    privileges.raise_unless_may_add_to_project_rag(auth.get("role"))

    # Same 409 handling as the /v1/drive/files route (Task 3).
    try:
        access_token = await drive_auth.get_access_token(auth["user_id"])
    except drive_auth.DriveNotConnected:
        raise HTTPException(409, "Google Drive is not connected.")
    except drive_auth.DriveAuthError as e:
        raise HTTPException(409, f"{e} Reconnect Google Drive.")

    from app.blocks.google_drive import GoogleDriveBlock
    result = await GoogleDriveBlock().process(
        req.file_id, {"operation": "download", "access_token": access_token})

    # Folder detection (PR #86 — revises PR #85). The Drive picker UI
    # lets users click "Add" on folder rows. PR #85 auto-redirected the
    # call to drive_index_folder but big folders blow past Render's
    # ~100s edge timeout, leaving the user with an opaque 502 HTML page.
    # Return a structured 400 instead so the frontend can show
    # "this is a folder" and offer a recursive-import button. The
    # explicit endpoint at /v1/projects/<id>/drive/index-folder still
    # works for callers that want that flow.
    if result.get("status") == "error" and result.get("is_folder"):
        raise HTTPException(
            400,
            {
                "code": "IS_FOLDER",
                "message": result.get("error"),
                "file_id": req.file_id,
                "name": result.get("name", ""),
                "hint": (
                    "Open the folder to browse into it and Add individual "
                    "files. Or POST /v1/projects/<id>/drive/index-folder "
                    "with this folder_id to recursively import (slow for "
                    "large folders)."
                ),
            },
        )

    if result.get("status") != "success":
        raise HTTPException(502, result.get("error", "Drive download failed."))

    raw_bytes = base64.b64decode(result.get("content_base64", ""))
    # The Drive block's download response has no `filename` key — the caller
    # must supply `name` (the frontend always has it from /v1/drive/files).
    original_name = os.path.basename(str(req.name).replace("\\", "/"))
    # Same extension allowlist a direct upload enforces (projects.py
    # add_document) — applied after the name is known, before writing.
    _, ext = os.path.splitext(original_name.lower())
    if ext not in ALLOWED_DOC_EXTENSIONS:
        raise HTTPException(
            415,
            f"File type '{ext}' is not ingestible: the knowledge base takes text "
            f"formats only (docs/INGEST_EXCLUSION_RULE.md). Drawings, photos, "
            f"video and CAD can still be attached to a chat for the session.",
        )

    # Reuse the upload storage scheme: UUID-prefixed stored filename, written
    # via file_crypto.write_document (encrypted at rest iff DATA_ENCRYPTION_KEY
    # is set) — exactly as projects.py add_document / upload.py do.
    file_id = str(uuid.uuid4())[:8]
    stored_as = f"{file_id}_{original_name}"
    filepath = os.path.join(projects_router.DATA_DIR, stored_as)
    file_crypto.write_document(filepath, raw_bytes)
    size = len(raw_bytes)

    # Register the document through the SAME app.core.projects call.
    doc = store.add_document(
        project_id, original_name, stored_as, filepath, size,
        metadata={
            "drive_file_id": req.file_id,
            "source": "drive_oauth_single",
        },
    )
    await audit.arecord("document.added", project_id=project_id,
                 document_id=doc["id"], name=original_name, size=size)
    background_tasks.add_task(doc_index.maybe_eager_index, project_id, doc["id"])
    return {
        "status": "stored",
        "message": (
            f"Added '{original_name}' — classified as {doc['doc_type']} "
            f"(role: {doc['doc_role']}). No analysis was run; ask in chat to "
            f"analyze it."
        ),
        "document": doc,
        "readiness": store.compute_readiness(project_id),
    }
