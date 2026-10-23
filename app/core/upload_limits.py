"""Single source of truth for upload size caps and accepted file types.

Why this module exists
----------------------
The platform had TWO upload paths that disagreed with each other:

* ``POST /upload`` (``app/routers/upload.py``) — 10 MB cap, its own extension
  set (which included audio/video, absent from the other).
* ``POST /v1/projects/{id}/documents`` (``app/routers/projects.py``) — 50 MB
  cap, its own extension set.

The chat composer posts to the second one, so a user's mental model of "what
the product accepts" was set by whichever route they happened to hit. A file
accepted by one route and 400'd by the other is indistinguishable, from the
browser, from the product being broken. Both routes now import from here, so
the limits move together or not at all.

The caps stay env-driven (``MAX_UPLOAD_SIZE`` / ``MAX_DOC_UPLOAD_SIZE``) and
are read through functions rather than bound at import time — binding at
import made them untestable without reloading the module, and made a
deployment-time env change silently ineffective for any module already
imported.
"""

from __future__ import annotations

import os

# Formats every upload route accepts. Audio/video are included because the
# voice/transcription blocks consume them; the construction formats are the
# ones registered blocks can actually parse (.dwg is accepted so take-off and
# BIM can answer with app.core.cad_formats.DWG_NOT_SUPPORTED -- export DXF or
# PDF -- instead of the upload 400'ing with a bare "type not allowed").
ALLOWED_UPLOAD_EXTENSIONS = frozenset({
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".webp", ".tif", ".tiff",
    ".txt", ".md", ".csv", ".json", ".xml",
    ".mp3", ".mp4", ".wav", ".webm",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".dxf", ".dwg", ".ifc", ".xer", ".mpp", ".rvt",
})

# 50 MB. Deliberately NOT 10 MB: this platform's real inputs are drawing sets
# and BIM models. The pilot corpus contains a 345 MB PDF, so operators loading
# real project documents must raise MAX_DOC_UPLOAD_SIZE (and the reverse proxy's
# client_max_body_size) rather than discovering the cap as a failed upload.
_DEFAULT_MAX_UPLOAD = 50 * 1024 * 1024


def max_upload_bytes() -> int:
    """Cap for the generic ``/upload`` route."""
    return _read_limit("MAX_UPLOAD_SIZE")


def max_document_bytes() -> int:
    """Cap for the project-documents route (what the chat composer uses)."""
    return _read_limit("MAX_DOC_UPLOAD_SIZE")


#: The admin route that adds a whole reference work to general knowledge. It
#: alone takes ``ADMIN_KNOWLEDGE_MAX_UPLOAD_MB``; every other route keeps the
#: user caps above.
ADMIN_KNOWLEDGE_UPLOAD_PATH = "/v1/admin/knowledge/documents"

# A reference work is uploaded once as one document, not split by hand.
_DEFAULT_ADMIN_KNOWLEDGE_MAX_UPLOAD_MB = 200


def admin_knowledge_max_bytes() -> int:
    """Cap for ``ADMIN_KNOWLEDGE_UPLOAD_PATH`` (``ADMIN_KNOWLEDGE_MAX_UPLOAD_MB``)."""
    raw = (os.getenv("ADMIN_KNOWLEDGE_MAX_UPLOAD_MB") or "").strip()
    mb = _DEFAULT_ADMIN_KNOWLEDGE_MAX_UPLOAD_MB
    if raw:
        try:
            value = float(raw)
        except ValueError:
            value = 0
        if value > 0:
            mb = value
    return int(mb * 1024 * 1024)


def request_body_limit(path: str | None = None) -> int:
    """Largest body any upload route could legitimately accept.

    Used by the early Content-Length guard in ``app/main.py``: a request over
    this can be refused before its body is read, since no route would take it.
    ``path`` is the request path; only the admin knowledge route is allowed
    its own, larger cap.
    """
    limit = max(max_upload_bytes(), max_document_bytes())
    if path and path.rstrip("/") == ADMIN_KNOWLEDGE_UPLOAD_PATH:
        limit = max(limit, admin_knowledge_max_bytes())
    return limit


def _read_limit(var: str) -> int:
    raw = (os.getenv(var) or "").strip()
    if not raw:
        return _DEFAULT_MAX_UPLOAD
    try:
        value = int(raw)
    except ValueError:
        # A typo'd limit must not silently become "unlimited" or "zero" — both
        # are outages (OOM, or every upload rejected). Fall back to the default.
        return _DEFAULT_MAX_UPLOAD
    return value if value > 0 else _DEFAULT_MAX_UPLOAD


def extension_of(filename: str) -> str:
    """Lowercased extension of ``filename`` (``''`` when it has none)."""
    return os.path.splitext((filename or "").lower())[1]


def is_allowed(filename: str) -> bool:
    return extension_of(filename) in ALLOWED_UPLOAD_EXTENSIONS
