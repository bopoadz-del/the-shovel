"""Google Drive Block - real OAuth 2.0 + Drive API (service account or user token)"""

import json
import logging
import os
from typing import Any, Dict

from app.core.universal_base import UniversalBlock

_LOG = logging.getLogger(__name__)

_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
_OAUTH_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_OAUTH_TOKEN_URL = "https://oauth2.googleapis.com/token"
_DRIVE_API = "https://www.googleapis.com/drive/v3"


def _build_service(access_token: str = None):
    """Build an authenticated Drive HTTP client."""
    import httpx
    token = access_token or os.getenv("GOOGLE_ACCESS_TOKEN", "")
    if not token:
        raise ValueError("No access token — call with operation=auth first")
    return httpx.AsyncClient(
        headers={"Authorization": f"Bearer {token}"},
        base_url=_DRIVE_API,
        timeout=20,
    )


def _oauth_url() -> str:
    client_id = os.getenv("GOOGLE_CLIENT_ID", "")
    if not client_id:
        return ""
    redirect = os.getenv("GOOGLE_REDIRECT_URI", "urn:ietf:wg:oauth:2.0:oob")
    scope = " ".join(_SCOPES)
    return (
        f"{_OAUTH_AUTH_URL}?client_id={client_id}"
        f"&redirect_uri={redirect}"
        f"&response_type=code"
        f"&scope={scope}"
        f"&access_type=offline"
    )


class GoogleDriveBlock(UniversalBlock):
    """Google Drive: list, read, download files via OAuth 2.0"""

    auto_validate = False
    name = "google_drive"
    version = "2.0"
    description = "Google Drive file operations — set GOOGLE_CLIENT_ID + GOOGLE_CLIENT_SECRET or GOOGLE_ACCESS_TOKEN"
    layer = 4
    tags = ["integration", "storage", "cloud", "google"]
    requires = []

    ui_schema = {
        "input": {
            "type": "text",
            "accept": ["*/*"],
            "placeholder": "File ID, folder name, or search query...",
            "multiline": False,
        },
        "output": {
            "type": "list",
            "fields": [{"name": "files", "type": "array", "label": "Files"}],
        },
        "quick_actions": [
            {"icon": "️", "label": "Browse Drive", "prompt": "List files from Google Drive"},
            {"icon": "", "label": "Auth", "prompt": "Authenticate with Google Drive"},
        ],
    }

    async def process(self, input_data: Any, params: Dict = None) -> Dict:
        params = params or {}
        operation = params.get("operation", "list")

        query = ""
        if isinstance(input_data, str):
            query = input_data
        elif isinstance(input_data, dict):
            query = input_data.get("query") or input_data.get("text") or ""
            operation = input_data.get("operation", operation)

        # ── Auth status / URL ─────────────────────────────────────────────────
        if operation in ("auth", "status"):
            has_token = bool(os.getenv("GOOGLE_ACCESS_TOKEN"))
            has_creds = bool(os.getenv("GOOGLE_CLIENT_ID"))
            url = _oauth_url()
            return {
                "status": "success",
                "operation": "auth",
                "authenticated": has_token,
                "credentials_configured": has_creds,
                "auth_url": url or None,
                "instructions": (
                    "Visit auth_url in a browser, approve, then set GOOGLE_ACCESS_TOKEN env var with the returned token."
                    if url and not has_token else
                    "Set GOOGLE_CLIENT_ID + GOOGLE_CLIENT_SECRET as env vars to enable OAuth."
                    if not has_creds else
                    "Access token is set. Use operation=list to browse files."
                ),
            }

        # ── List files ────────────────────────────────────────────────────────
        if operation == "list":
            access_token = params.get("access_token") or os.getenv("GOOGLE_ACCESS_TOKEN", "")
            if not access_token:
                return {
                    "status": "error",
                    "error": "Not authenticated. Run with operation=auth to get the auth URL, then set GOOGLE_ACCESS_TOKEN.",
                    "auth_url": _oauth_url() or None,
                }
            try:
                import httpx
                # Build the Drive-API q filter. Search wins if present (name
                # contains, no folder filter); otherwise list children of a
                # specific folder (defaults to root so the user sees their
                # actual top-level Drive, not the 50 newest files at any
                # depth which was the prior behaviour).
                folder_id = params.get("folder_id")
                if query:
                    q = f"name contains '{query}' and trashed=false"
                elif folder_id:
                    q = f"'{folder_id}' in parents and trashed=false"
                else:
                    q = "'root' in parents and trashed=false"
                async with httpx.AsyncClient(timeout=20) as client:
                    resp = await client.get(
                        f"{_DRIVE_API}/files",
                        headers={"Authorization": f"Bearer {access_token}"},
                        params={
                            "q": q,
                            "pageSize": params.get("limit", 100),
                            "orderBy": "folder,name",  # folders first, then alpha
                            "fields": "files(id,name,mimeType,size,modifiedTime,webViewLink,parents)",
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()

                FOLDER_MT = "application/vnd.google-apps.folder"
                files = [
                    {
                        "id": f.get("id"),
                        "name": f.get("name"),
                        "mime_type": f.get("mimeType", ""),
                        "is_folder": f.get("mimeType") == FOLDER_MT,
                        "type": f.get("mimeType", "").split("/")[-1],
                        "size_bytes": int(f.get("size", 0)),
                        "modified": f.get("modifiedTime", "")[:10],
                        "url": f.get("webViewLink", ""),
                    }
                    for f in data.get("files", [])
                ]
                return {"status": "success", "operation": "list", "files": files, "total": len(files)}
            except Exception:
                return {
                    "status": "error",
                    "error": "Unable to list Google Drive files at this time.",
                    "operation": "list",
                }

        # ── Download / read file ──────────────────────────────────────────────
        if operation == "download":
            file_id = query or params.get("file_id", "")
            access_token = params.get("access_token") or os.getenv("GOOGLE_ACCESS_TOKEN", "")
            if not file_id:
                return {"status": "error", "error": "file_id required for download"}
            if not access_token:
                return {"status": "error", "error": "Not authenticated"}
            try:
                import httpx
                from app.core import drive_mime
                async with httpx.AsyncClient(timeout=60) as client:
                    meta = await client.get(
                        f"{_DRIVE_API}/files/{file_id}",
                        headers={"Authorization": f"Bearer {access_token}"},
                        params={"fields": "mimeType,name,shortcutDetails"},
                    )
                    meta.raise_for_status()
                    meta_json = meta.json()
                    mime = meta_json.get("mimeType", "")

                    # Folder check (operator brief 2026-06-19, PR #85).
                    # The Drive picker UI lets the user click "Add" on
                    # folder rows too, sending the folder's file_id here.
                    # Folders have no downloadable content; the right
                    # action is to call /v1/projects/{id}/drive/index-folder
                    # instead. Surface a structured error the frontend can
                    # use to either auto-redirect or render a clear prompt.
                    if mime == "application/vnd.google-apps.folder":
                        return {
                            "status": "error",
                            "error": (
                                f"'{meta_json.get('name','?')}' is a Drive folder, not a file. "
                                f"Use the folder-import flow (/v1/projects/<id>/drive/index-folder) "
                                f"to recursively pull every supported file in it."
                            ),
                            "operation": "download",
                            "file_id": file_id,
                            "mime_type": mime,
                            "is_folder": True,
                            "name": meta_json.get("name", ""),
                        }

                    # Shortcut handling (operator brief 2026-06-19, PR #83).
                    # `application/vnd.google-apps.shortcut` files don't
                    # have downloadable content of their own — they point
                    # at another file. Drive returns `shortcutDetails`
                    # carrying the real `targetId` and `targetMimeType`.
                    # Re-target the download at the real file.
                    if mime == "application/vnd.google-apps.shortcut":
                        sd = meta_json.get("shortcutDetails") or {}
                        target_id = sd.get("targetId")
                        target_mime = sd.get("targetMimeType") or ""
                        if not target_id:
                            return {
                                "status": "error",
                                "error": "Drive shortcut has no targetId; original file may have been deleted.",
                                "operation": "download",
                                "file_id": file_id,
                            }
                        file_id = target_id
                        mime = target_mime
                    target = drive_mime.export_target(mime)
                    if target is not None:
                        # Known native type — use the operator-curated export
                        # mapping (Doc -> .docx, Sheet -> .xlsx, etc.).
                        export_mime, exported_ext = target
                        resp = await client.get(
                            f"{_DRIVE_API}/files/{file_id}/export",
                            headers={"Authorization": f"Bearer {access_token}"},
                            params={"mimeType": export_mime},
                        )
                    elif drive_mime.is_native(mime):
                        # Unknown native type (form/site/shortcut/script/…
                        # any application/vnd.google-apps.* not in
                        # NATIVE_EXPORTS). Google rejects alt=media on
                        # every native type with HTTP 403 "Only files with
                        # binary content can be downloaded. Use Export with
                        # Docs Editors files." — so default to PDF export
                        # which Drive accepts for nearly every Docs-Editors
                        # type. The operator hit this 2026-06-19 on a file
                        # the picker labelled as a PDF but Drive served as
                        # a native Google Doc.
                        export_mime, exported_ext = "application/pdf", ".pdf"
                        resp = await client.get(
                            f"{_DRIVE_API}/files/{file_id}/export",
                            headers={"Authorization": f"Bearer {access_token}"},
                            params={"mimeType": export_mime},
                        )
                    else:
                        # Binary file — direct download.
                        exported_ext = None
                        resp = await client.get(
                            f"{_DRIVE_API}/files/{file_id}",
                            headers={"Authorization": f"Bearer {access_token}"},
                            params={"alt": "media"},
                        )
                    resp.raise_for_status()
                    content = resp.content
                return {
                    "status": "success",
                    "operation": "download",
                    "file_id": file_id,
                    "mime_type": mime,
                    "exported_extension": exported_ext,
                    "size_bytes": len(content),
                    "content_base64": __import__("base64").b64encode(content).decode(),
                }
            except httpx.HTTPStatusError as e:
                # Google Drive returned non-2xx — surface the status + body
                # so the caller can act on it (token expired -> reconnect,
                # 404 -> file moved, 403 -> permission, 429 -> back off).
                _LOG.warning(
                    "drive download HTTP %s for file_id=%s mime=%s: %s",
                    e.response.status_code, file_id, mime, e.response.text[:200],
                )
                return {
                    "status": "error",
                    "error": (
                        f"Drive download failed: HTTP {e.response.status_code} "
                        f"(mime={mime!r}) — {e.response.text[:200]}"
                    ),
                    "operation": "download",
                    "file_id": file_id,
                    "mime_type": mime,
                    "http_status": e.response.status_code,
                }
            except Exception as e:  # noqa: BLE001
                _LOG.exception("drive download crashed for file_id=%s", file_id)
                return {
                    "status": "error",
                    "error": f"Drive download crashed: {type(e).__name__}: {e}",
                    "operation": "download",
                    "file_id": file_id,
                }

        return {"status": "error", "error": f"Unknown operation: {operation}. Use: auth, list, download"}
