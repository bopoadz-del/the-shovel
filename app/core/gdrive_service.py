"""Google Drive — service-account access for unattended jobs.

The user-facing :class:`app.blocks.google_drive.GoogleDriveBlock` is built
around per-user OAuth — a session token the user obtained by signing in.
The nightly hydration job has no signed-in user, so it can't use that path.

This module fills the gap with the standard pattern for unattended Google API
access: a **service account**. The account is created in GCP, has its email
shared into the Drive folders that should be visible, and its JSON key is
mounted into the container via an env var. At call time we mint a short-lived
access token from the key and hit the same Drive v3 REST surface the user-mode
block already talks to.

Configuration (env vars):

* ``GDRIVE_SERVICE_ACCOUNT_JSON`` — either an absolute path to the service
  account's JSON key file, or the JSON content itself (one line). Required.
* ``GDRIVE_PROJECT_FOLDERS`` — comma-separated ``project_id:drive_folder_id``
  pairs that tell hydration which Drive folder belongs to which platform
  project. Without this mapping, the scheduler has no way to know.

Dependencies: this module lazy-imports ``google-auth`` only when a call
actually needs to mint a token. If the library isn't installed, the helpers
return a clean ``status=disabled`` payload — the hydration block treats this
as "not configured" and continues without crashing.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


logger = logging.getLogger(__name__)


_DRIVE_API = "https://www.googleapis.com/drive/v3"
_DRIVE_UC_DOWNLOAD = "https://drive.google.com/uc"
_DRIVE_USERCONTENT_DOWNLOAD = "https://drive.usercontent.google.com/download"
# Drive file ids are opaque tokens (typically 28–44 chars). Reject URLs /
# paths so a PATCH or public-download fallback cannot be pointed at an
# arbitrary host.
_DRIVE_FILE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{10,128}$")
_CONFIRM_TOKEN_RE = re.compile(
    r"(?:confirm=|name=[\"']confirm[\"']\s+value=[\"'])([0-9A-Za-z_-]+)",
    re.IGNORECASE,
)
_UUID_TOKEN_RE = re.compile(
    r"(?:[?&]uuid=|name=[\"']uuid[\"']\s+value=[\"'])([0-9A-Fa-f-]+)",
    re.IGNORECASE,
)
_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# Drive metadata MIME types we cannot blob-download. The export path (for
# Docs/Sheets/Slides) is a separate concern handled at download time.
_GOOGLE_DOC_MIMES = {
    "application/vnd.google-apps.document",
    "application/vnd.google-apps.spreadsheet",
    "application/vnd.google-apps.presentation",
    "application/vnd.google-apps.drawing",
    "application/vnd.google-apps.folder",
}


# ── Configuration ────────────────────────────────────────────────────────


def is_configured() -> bool:
    """True iff the service-account key env var is present.

    Does NOT validate the key itself — only that the operator intended to
    enable this path. ``get_credentials_info`` does the parse, and minting
    a token surfaces the deeper "is this key actually valid" answer.
    """
    return bool((os.getenv("GDRIVE_SERVICE_ACCOUNT_JSON") or "").strip())


def parse_project_folder_map() -> Dict[str, str]:
    """Parse ``GDRIVE_PROJECT_FOLDERS`` into ``{project_id: folder_id}``.

    Format: comma-separated ``proj:folder`` pairs. Malformed entries are
    logged and skipped — one bad row doesn't poison the whole mapping.
    """
    raw = (os.getenv("GDRIVE_PROJECT_FOLDERS") or "").strip()
    if not raw:
        return {}
    out: Dict[str, str] = {}
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            logger.warning("GDRIVE_PROJECT_FOLDERS: ignoring malformed entry %r", chunk)
            continue
        pid, fid = chunk.split(":", 1)
        pid, fid = pid.strip(), fid.strip()
        if pid and fid:
            out[pid] = fid
    return out


def _load_service_account_info() -> Optional[Dict[str, Any]]:
    """Read the service-account JSON from env. Accepts either a file path or
    inline JSON content. Returns ``None`` (with a warning logged) when the
    value is set but unreadable — callers treat that as "not configured"."""
    raw = (os.getenv("GDRIVE_SERVICE_ACCOUNT_JSON") or "").strip()
    if not raw:
        return None
    # File path?
    if os.path.isfile(raw):
        try:
            with open(raw, "r") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("GDRIVE_SERVICE_ACCOUNT_JSON file unreadable: %s", exc)
            return None
    # Inline JSON?
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("GDRIVE_SERVICE_ACCOUNT_JSON inline value is not valid JSON: %s", exc)
        return None


# ── Token minting (cached) ───────────────────────────────────────────────

_token_cache: Dict[str, Any] = {"token": None, "expiry": 0.0}


def _mint_access_token() -> Optional[str]:
    """Return a Drive-scoped access token, or ``None`` if anything is wrong
    (lib missing, key invalid, network failure). Tokens are cached in-process
    until shortly before their server-side expiry — a single hydration pass
    won't hammer the token endpoint."""
    # Reuse a cached token until 60 s before its expiry.
    now = time.time()
    if _token_cache["token"] and now < _token_cache["expiry"] - 60:
        return _token_cache["token"]

    info = _load_service_account_info()
    if info is None:
        return None

    try:
        from google.oauth2 import service_account  # lazy import
        from google.auth.transport.requests import Request as GoogleAuthRequest
    except ImportError:
        logger.warning(
            "google-auth is not installed; hydration cannot mint Drive tokens. "
            "Install it (or uncomment in requirements.txt) to enable the "
            "service-account path."
        )
        return None

    try:
        creds = service_account.Credentials.from_service_account_info(
            info, scopes=_SCOPES
        )
        creds.refresh(GoogleAuthRequest())
    except Exception as exc:  # noqa: BLE001
        logger.warning("Drive service-account token mint failed: %s", exc)
        return None

    _token_cache["token"] = creds.token
    # creds.expiry is naive UTC; convert to epoch
    if creds.expiry is not None:
        from datetime import timezone as _tz
        _token_cache["expiry"] = creds.expiry.replace(tzinfo=_tz.utc).timestamp()
    else:
        _token_cache["expiry"] = now + 3000  # be conservative — ~50 min
    return creds.token


# ── Drive REST helpers ───────────────────────────────────────────────────

# files.list rejects pageSize above 1000 with HTTP 400.
_DRIVE_LIST_PAGE_MAX = 1000


def _clamp_drive_page_size(page_size: int) -> int:
    try:
        size = int(page_size)
    except (TypeError, ValueError):
        size = 100
    if size < 1:
        return 1
    if size > _DRIVE_LIST_PAGE_MAX:
        return _DRIVE_LIST_PAGE_MAX
    return size


def list_folder_files(folder_id: str, page_size: int = 100) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """List non-trashed files directly inside ``folder_id``.

    Returns ``(files, error)``. On any failure (no token, network, 4xx/5xx)
    returns ``([], "<reason>")``; hydration logs the reason and moves on
    rather than crashing the whole pass. Subfolders are NOT recursed — one
    Drive folder maps to one platform project; deeper structure can be
    flattened later if it becomes a real need.
    """
    token = _mint_access_token()
    if not token:
        return [], "service account unavailable (key missing, invalid, or google-auth not installed)"

    try:
        import httpx
    except ImportError:
        return [], "httpx not available"

    page_size = _clamp_drive_page_size(page_size)
    escaped = _escape_drive_query_value(folder_id)
    q = f"'{escaped}' in parents and trashed = false"
    files: List[Dict[str, Any]] = []
    page_token: Optional[str] = None
    try:
        with httpx.Client(timeout=30) as client:
            while True:
                params: Dict[str, Any] = {
                    "q": q,
                    "pageSize": page_size,
                    "fields": (
                        "nextPageToken, files(id, name, mimeType, size, "
                        "modifiedTime, md5Checksum)"
                    ),
                    "supportsAllDrives": "true",
                    "includeItemsFromAllDrives": "true",
                }
                if page_token:
                    params["pageToken"] = page_token
                resp = client.get(
                    f"{_DRIVE_API}/files",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                )
                if resp.status_code != 200:
                    return [], f"Drive list returned {resp.status_code}: {resp.text[:200]}"
                payload = resp.json()
                files.extend(payload.get("files") or [])
                page_token = payload.get("nextPageToken")
                if not page_token:
                    break
    except Exception as exc:  # noqa: BLE001
        return [], f"{type(exc).__name__}: {exc}"
    return files, None


def is_downloadable(file_meta: Dict[str, Any]) -> bool:
    """A file is downloadable if it's neither a folder nor a Google-native
    Doc/Sheet/Slide (those need ``export`` instead of ``alt=media``).

    Folders are filtered out implicitly by the list query but we double-check
    in case of permission quirks.
    """
    mime = file_meta.get("mimeType") or ""
    return mime not in _GOOGLE_DOC_MIMES


_FOLDER_MIME = "application/vnd.google-apps.folder"

# Basename classes dropped before a file is handed to an ingest assignment.
# Logged as counts and extensions only — never the basename itself.
_OS_METADATA_NAMES = frozenset({
    "desktop.ini",
    "thumbs.db",
    ".ds_store",
    "ehthumbs.db",
})
_WHATSAPP_MEDIA_RE = re.compile(
    r"^(?:"
    r"img-\d{8}-wa\d{4}\.(?:jpe?g)"
    r"|vid-\d{8}-wa\d{4}\.mp4"
    r"|(?:aud|ptt)-\d{8}-wa\d{4}\.(?:opus|m4a|aac)"
    r")$",
    re.IGNORECASE,
)
# Non-document support files. Denied by extension at the walk so they are
# never assigned. Document types (.pdf .docx .xlsx .dwg .jpg .vsdx .rtf
# .xls .zip and the rest of the extractor set) are not in these sets.
_FONT_EXTS = frozenset({".ttf", ".otf", ".shx", ".fon"})
_CAD_SUPPORT_EXTS = frozenset({
    ".ctb", ".stb", ".pc3", ".pmp", ".bak", ".bax", ".dwl", ".dwl2",
})
_GIS_EXTS = frozenset({".lyr", ".lyrx", ".mxd", ".aprx", ".atbx"})
_SCRATCH_EXTS = frozenset({".ini", ".log", ".tmp"})
_EXT_JUNK = (
    (_FONT_EXTS, "font"),
    (_CAD_SUPPORT_EXTS, "cad_support"),
    (_GIS_EXTS, "gis"),
    (_SCRATCH_EXTS, "scratch"),
)
_JUNK_CLASSES = (
    "os_metadata", "office_lock", "whatsapp",
    "font", "cad_support", "gis", "scratch",
)


def walk_junk_class(filename: str) -> Optional[str]:
    """Return a junk class for ``filename``, or None when it should be walked.

    Folders are not classified here; callers skip this for folder mime types.
    Matching is case-insensitive and uses the basename only.
    """
    base = (filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    folded = base.casefold()
    if folded in _OS_METADATA_NAMES:
        return "os_metadata"
    if folded.startswith("~$") or folded.startswith(".~lock."):
        return "office_lock"
    if _WHATSAPP_MEDIA_RE.match(base):
        return "whatsapp"
    ext = Path(base).suffix.casefold()
    for exts, cls in _EXT_JUNK:
        if ext in exts:
            return cls
    return None


def _junk_extension(filename: str) -> str:
    """Extension token for the walk summary. Never the basename."""
    base = (filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    trimmed = base[:-1] if base.endswith("#") else base
    suffix = Path(trimmed).suffix.casefold()
    if suffix:
        return suffix
    if trimmed.startswith("."):
        return trimmed.casefold()
    return "none"


def _log_walk_junk(
    counts: Dict[str, int],
    extensions: Dict[str, Set[str]],
) -> None:
    if not any(counts.values()):
        return
    parts: List[str] = []
    for cls in _JUNK_CLASSES:
        ext = ",".join(sorted(extensions[cls])) if extensions[cls] else "-"
        parts.append(f"{cls}={counts[cls]} extensions={ext}")
    logger.info("WALK_JUNK %s", " ".join(parts))


def walk_folder(
    root_folder_id: str,
    max_depth: int = 8,
    page_size: int = 100,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Recursively walk ``root_folder_id`` and return every file beneath it.

    Each yielded file dict carries an additional ``_drive_path`` field — a
    forward-slash-joined name path from the root (e.g.
    ``"200-Project Controls/2.3 Risk/2.3.1 Register.pdf"``) so callers can
    attribute files to their location for downstream training data.

    Returns ``(files, errors)``. Errors don't abort the walk — a permission
    denial or 5xx on one subtree gets logged in the errors list and the
    walker moves on. The contract matches ``list_folder_files``: callers
    see partial data rather than nothing on transient issues.

    ``max_depth`` caps recursion at 8 levels by default. This is generous
    for SOP-style folder structures (100-/200-/300-/... with one or two
    levels of section nesting) but stops cycles dead even if Drive's
    folder graph somehow contained one.

    Depth-cap fidelity (PR #25 review #2): when ``max_depth`` is hit, the
    walker counts how many subfolders went unexplored under that branch
    and surfaces the count in the error message so operators can tell
    whether one folder or fifty was skipped.
    """
    files: List[Dict[str, Any]] = []
    errors: List[str] = []
    junk_counts = {cls: 0 for cls in _JUNK_CLASSES}
    junk_exts: Dict[str, Set[str]] = {cls: set() for cls in _JUNK_CLASSES}
    visited: set = set()  # folder_ids we've already entered (cycle guard)
    # path_prefix → count of subfolders skipped at the depth cap.
    # Aggregated so we emit ONE error per truncated branch with a tally.
    depth_cap_skipped: Dict[str, int] = {}

    def _walk(folder_id: str, path_prefix: str, depth: int) -> None:
        if depth > max_depth:
            # Count this subtree as skipped against the deepest ancestor
            # the operator can act on (the parent directory at the cap).
            parent_path = "/".join(path_prefix.split("/")[:max_depth]) or "/"
            depth_cap_skipped[parent_path] = depth_cap_skipped.get(parent_path, 0) + 1
            return
        if folder_id in visited:
            return
        visited.add(folder_id)

        entries, err = list_folder_files(folder_id, page_size=page_size)
        if err:
            errors.append(f"gdrive walk({path_prefix or folder_id}): {err}")
            return

        for entry in entries:
            name = entry.get("name") or ""
            mime = entry.get("mimeType") or ""
            entry_path = f"{path_prefix}/{name}".lstrip("/")
            if mime == _FOLDER_MIME:
                _walk(entry.get("id") or "", entry_path, depth + 1)
            else:
                junk = walk_junk_class(name)
                if junk:
                    junk_counts[junk] += 1
                    junk_exts[junk].add(_junk_extension(name))
                    continue
                # Annotate with the path so downstream code can attribute
                # the file to its location in the SOP tree.
                annotated = dict(entry)
                annotated["_drive_path"] = entry_path
                files.append(annotated)

    _walk(root_folder_id, "", 0)

    # Roll up the depth-cap tallies into a single error per truncated
    # branch so operators get an actionable count rather than a stream
    # of identical messages.
    for parent_path, count in sorted(depth_cap_skipped.items()):
        suffix = f"{count} subfolder{'s' if count != 1 else ''} skipped"
        errors.append(
            f"gdrive walk: max_depth {max_depth} exceeded at {parent_path} "
            f"({suffix})"
        )

    _log_walk_junk(junk_counts, junk_exts)
    return files, errors


def _escape_drive_query_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def find_file_id_by_exact_name(filename: str) -> Tuple[Optional[str], Optional[str]]:
    """Look up a Drive file id by exact ``name`` (not trash).

    Master Corpus rag-backfill rows are RAG-citable stubs: size=0, a stale
    ``X:\\Example Drive\\...`` path, ``drive_file_id`` null.
    Preview cannot follow a pointer that was never written. The live file
    still lives on Drive under the same filename — this query is the
    durable link. Returns ``(file_id, None)`` or ``(None, reason)``.
    """
    name = (filename or "").strip()
    if not name:
        return None, "document has no original_name"
    token = _mint_access_token()
    if not token:
        return None, "service account unavailable"
    try:
        import httpx
    except ImportError:
        return None, "httpx not available"

    escaped = _escape_drive_query_value(name)
    q = f"name = '{escaped}' and trashed = false"
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.get(
                f"{_DRIVE_API}/files",
                headers={"Authorization": f"Bearer {token}"},
                params={
                    "q": q,
                    "pageSize": 10,
                    "fields": "files(id, name, mimeType, size, md5Checksum)",
                    "supportsAllDrives": "true",
                    "includeItemsFromAllDrives": "true",
                },
            )
            if resp.status_code != 200:
                return None, f"Drive name lookup returned {resp.status_code}: {resp.text[:200]}"
            files = [
                f for f in (resp.json().get("files") or [])
                if (f.get("name") or "") == name and f.get("id")
            ]
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"

    if not files:
        return None, f"no Drive file named {name}"
    if len(files) > 1:
        pdfs = [f for f in files if (f.get("mimeType") or "") == "application/pdf"]
        chosen = (pdfs or files)[0]
        logger.info(
            "Drive name lookup matched %s files for %r; using %s",
            len(files), name, chosen.get("id"),
        )
        return str(chosen["id"]), None
    return str(files[0]["id"]), None


def is_valid_drive_file_id(file_id: str) -> bool:
    """True iff ``file_id`` looks like a Drive file id, not a URL or path."""
    return bool(_DRIVE_FILE_ID_RE.fullmatch((file_id or "").strip()))


def get_file_metadata(file_id: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Drive ``files.get`` (supportsAllDrives) without ``alt=media``.

    Name-search (``files.list``) cannot see anyone-with-link files the SA
    was never shared on. ``files.get`` by id still works once the id is
    known (PATCH / prior resolve). Returns ``(meta, None)`` or
    ``(None, reason)``.
    """
    fid = (file_id or "").strip()
    if not is_valid_drive_file_id(fid):
        return None, "invalid drive file id"
    token = _mint_access_token()
    if not token:
        return None, "service account unavailable"
    try:
        import httpx
    except ImportError:
        return None, "httpx not available"
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.get(
                f"{_DRIVE_API}/files/{fid}",
                headers={"Authorization": f"Bearer {token}"},
                params={
                    "fields": "id,name,mimeType,size,md5Checksum",
                    "supportsAllDrives": "true",
                },
            )
            if resp.status_code != 200:
                return None, (
                    f"Drive files.get returned {resp.status_code}: {resp.text[:200]}"
                )
            payload = resp.json() if resp.content else {}
            if not isinstance(payload, dict) or not payload.get("id"):
                return None, "Drive files.get returned no file id"
            return payload, None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def _response_looks_like_file(content: bytes, content_type: str, content_disposition: str) -> bool:
    """True when the HTTP body is file bytes, not a Google HTML interstitial."""
    if not content:
        return False
    if content.startswith(b"%PDF") or content[:4] == b"PK\x03\x04":
        return True
    ctype = (content_type or "").split(";")[0].strip().lower()
    if ctype in {"text/html", "application/xhtml+xml", "text/plain"}:
        # text/plain can be a real .txt; accept only with a filename header.
        disp = (content_disposition or "").lower()
        return ctype == "text/plain" and (
            "filename=" in disp or "attachment" in disp
        )
    disp = (content_disposition or "").lower()
    if "filename=" in disp or "attachment" in disp:
        return True
    return ctype.startswith("application/") or ctype.startswith("image/")


def _extract_confirm_params(html: str) -> Dict[str, str]:
    """Pull confirm / uuid tokens out of Google's virus-scan HTML page."""
    out: Dict[str, str] = {}
    confirm = _CONFIRM_TOKEN_RE.search(html or "")
    if confirm:
        out["confirm"] = confirm.group(1)
    uuid_m = _UUID_TOKEN_RE.search(html or "")
    if uuid_m:
        out["uuid"] = uuid_m.group(1)
    return out


def _html_is_access_denied(html: str, final_url: str) -> bool:
    url = (final_url or "").lower()
    if "accounts.google.com" in url:
        return True
    lowered = (html or "").lower()
    return (
        "you need access" in lowered
        or "request access" in lowered
        or "sign in to continue" in lowered
        or "unable to access" in lowered
    )


def download_public_file_bytes(file_id: str) -> Tuple[Optional[bytes], Optional[str]]:
    """Download a Drive file that is already anyone-with-link / world-readable.

    Used when the service account can ``files.get`` by id (or we already
    have the id from PATCH) but cannot ``files.list`` by name, and SA
    media download fails. Does **not** use the service-account token —
    private files fail closed (403 / HTML access wall). Handles the
    confirm-token interstitial Google serves instead of the PDF.
    """
    fid = (file_id or "").strip()
    if not is_valid_drive_file_id(fid):
        return None, "invalid drive file id"
    try:
        import httpx
    except ImportError:
        return None, "httpx not available"

    attempts = [
        {"url": _DRIVE_UC_DOWNLOAD, "params": {"export": "download", "id": fid}},
        {
            "url": _DRIVE_USERCONTENT_DOWNLOAD,
            "params": {"id": fid, "export": "download"},
        },
    ]
    last_err = "public Drive download returned no bytes"
    try:
        with httpx.Client(timeout=120, follow_redirects=True) as client:
            for attempt in attempts:
                blob, err, retry_params = _public_download_once(
                    client, attempt["url"], attempt["params"],
                )
                if blob:
                    return blob, None
                if retry_params:
                    merged = dict(attempt["params"])
                    merged.update(retry_params)
                    blob, err, _ = _public_download_once(
                        client, attempt["url"], merged,
                    )
                    if blob:
                        return blob, None
                last_err = err or last_err
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"
    return None, last_err


def _public_download_once(
    client: Any, url: str, params: Dict[str, str],
) -> Tuple[Optional[bytes], Optional[str], Optional[Dict[str, str]]]:
    """One GET. Returns ``(bytes, err, confirm_params_if_html_interstitial)``."""
    resp = client.get(url, params=params)
    status = int(getattr(resp, "status_code", 0) or 0)
    if status in {401, 403, 404}:
        return None, f"public Drive download returned {status}", None
    if status != 200:
        return None, f"public Drive download returned {status}", None

    content = resp.content or b""
    headers = getattr(resp, "headers", {}) or {}
    ctype = headers.get("content-type") or headers.get("Content-Type") or ""
    disp = (
        headers.get("content-disposition")
        or headers.get("Content-Disposition")
        or ""
    )
    if _response_looks_like_file(content, ctype, disp):
        return content, None, None

    html = ""
    try:
        html = content.decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return None, "public Drive download is not world-readable", None

    final_url = str(getattr(resp, "url", "") or "")
    if _html_is_access_denied(html, final_url):
        return None, "public Drive download is not world-readable", None

    extra = _extract_confirm_params(html)
    cookies = getattr(resp, "cookies", None)
    if cookies is not None:
        try:
            items = cookies.items() if hasattr(cookies, "items") else []
            for name, value in items:
                if str(name).startswith("download_warning") and value:
                    extra.setdefault("confirm", str(value))
        except Exception:  # noqa: BLE001
            logger.info(
                "could not read Drive download_warning cookie",
                exc_info=True,
            )
    if extra.get("confirm"):
        return None, "public Drive download requires confirm token", extra
    return None, "public Drive download is not world-readable", None


def download_file_bytes(file_id: str) -> Tuple[Optional[bytes], Optional[str]]:
    """Download a single Drive file's raw bytes.

    Returns ``(bytes, None)`` on success or ``(None, error)`` on failure.
    Streams the response so a large file doesn't have to fit in one socket
    buffer; the hydration block enforces a separate size cap before calling.
    """
    fid = (file_id or "").strip()
    if not is_valid_drive_file_id(fid):
        return None, "invalid drive file id"
    token = _mint_access_token()
    if not token:
        return None, "service account unavailable"
    try:
        import httpx
    except ImportError:
        return None, "httpx not available"

    try:
        with httpx.Client(timeout=120) as client:
            resp = client.get(
                f"{_DRIVE_API}/files/{fid}",
                headers={"Authorization": f"Bearer {token}"},
                # Shared-drive corpus files 404 without supportsAllDrives.
                params={"alt": "media", "supportsAllDrives": "true"},
            )
            if resp.status_code != 200:
                return None, f"Drive download returned {resp.status_code}: {resp.text[:200]}"
            return resp.content, None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"
