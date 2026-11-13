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
_SESSION_ONLY_EXTENSIONS = frozenset({
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".webp", ".tif", ".tiff",
    ".txt", ".md", ".csv", ".json", ".xml",
    ".mp3", ".mp4", ".wav", ".webm",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".dxf", ".dwg", ".ifc", ".xer", ".mpp", ".rvt",
})


def _ingestible_extensions() -> frozenset:
    """Every format a project document may be (the one ingest rule)."""
    from app.core.ingest_status import TEXT_BEARING_EXTS
    return frozenset(TEXT_BEARING_EXTS)


#: What ``/upload`` accepts without a project, and what ``/v1/upload-limits``
#: publishes: the session formats plus every ingestible text format, so a
#: format a project takes (HTML, mail, RTF) is never refused for a session.
#: Derived, not typed: adding a text extractor widens both lists at once.
ALLOWED_UPLOAD_EXTENSIONS = _SESSION_ONLY_EXTENSIONS | _ingestible_extensions()

#: The project-document subset (``POST /v1/projects/{id}/documents`` and
#: ``/upload`` with ``project_id``).
PROJECT_DOCUMENT_EXTENSIONS = _ingestible_extensions()

#: One status for "this format is not accepted here", with or without a project.
UNSUPPORTED_TYPE_STATUS = 415

# Leading bytes that prove a format whatever the upload is called. A phone's
# share sheet names a shared document after its sender ("Document from <name>")
# with no extension, and a renamed file can carry the wrong one; the type is
# then read from the content, never guessed from the name.
_SNIFF_HEAD = 1024
_LEADING_SIGNATURES = (
    (b"%PDF-", ".pdf"),
    (b"\xff\xd8\xff", ".jpg"),
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"GIF87a", ".gif"),
    (b"GIF89a", ".gif"),
    (b"II*\x00", ".tif"),
    (b"MM\x00*", ".tif"),
)
# An Office Open XML package is a zip whose parts live under one folder.
_OOXML_PART_PREFIXES = (("word/", ".docx"), ("xl/", ".xlsx"), ("ppt/", ".pptx"))
# Spellings of one format: a ``.jpeg`` holding JPEG bytes is not mis-named.
_SAME_FORMAT = {".jpeg": ".jpg", ".tiff": ".tif"}


def sniff_extension(fileobj) -> str | None:
    """The extension the content of ``fileobj`` proves, or None when the bytes
    prove nothing (plain text, CSV, an unknown binary). Leaves it rewound."""
    fileobj.seek(0)
    head = fileobj.read(_SNIFF_HEAD)
    fileobj.seek(0)
    lead = head.lstrip(b"\xef\xbb\xbf \t\r\n")
    for signature, ext in _LEADING_SIGNATURES:
        if lead.startswith(signature):
            return ext
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return ".webp"
    if head.startswith(b"PK\x03\x04"):
        return _ooxml_extension(fileobj)
    return None


def _ooxml_extension(fileobj) -> str | None:
    import zipfile

    try:
        with zipfile.ZipFile(fileobj) as package:
            names = package.namelist()
    except (zipfile.BadZipFile, OSError, ValueError):
        names = []
    finally:
        fileobj.seek(0)
    for prefix, ext in _OOXML_PART_PREFIXES:
        if any(name.startswith(prefix) for name in names):
            return ext
    return None


def upload_display_name(display_name: str | None, multipart_name: str | None) -> str:
    """The user's file name for an upload, path parts dropped.

    A client that cannot carry a non-ASCII name safely in the multipart
    ``filename`` sends an ASCII stand-in there and the real name in a
    ``display_name`` form field; the real name wins when it is present.
    """
    name = (display_name or "").strip() or (multipart_name or "").strip()
    return os.path.basename(name.replace("\\", "/")).strip()


def describe_ext(ext: str) -> str:
    """``ext`` for an error message; a missing one is named as missing."""
    return ext or "no extension, and the content is not a recognised format"


def typed_upload_name(name: str, fileobj) -> str:
    """``name`` with the extension the upload's content proves.

    The name is kept as the user gave it when the content proves nothing or
    agrees with it. A missing or unknown suffix (``Document from X``,
    ``Report v1.2``) gets the proven extension appended; a known extension the
    content contradicts is replaced by the proven one.
    """
    proven = sniff_extension(fileobj)
    stem, ext = os.path.splitext(name)
    ext = ext.lower()
    if proven is None or _SAME_FORMAT.get(ext, ext) == proven:
        return name
    if ext in ALLOWED_UPLOAD_EXTENSIONS:
        return stem + proven
    return name + proven


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
