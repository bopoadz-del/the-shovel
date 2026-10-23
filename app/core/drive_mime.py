"""Single source of truth for Google Drive MIME handling.

Google's native types (Docs, Sheets, Slides, Drawings) refuse the
``files/{id}?alt=media`` download path and must be exported through
``files/{id}/export?mimeType=...`` to a downloadable format. The two
places that need this mapping (the Drive walker in ``app/routers/drive.py``
and the single-file path in ``app/blocks/google_drive.py``) previously
each carried their own copy; this module is the de-dup.
"""
from __future__ import annotations

from typing import Optional, Tuple


# native source mime  ->  (export mime, exported extension)
NATIVE_EXPORTS: dict[str, Tuple[str, str]] = {
    "application/vnd.google-apps.document":
        ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", ".docx"),
    "application/vnd.google-apps.spreadsheet":
        ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", ".xlsx"),
    "application/vnd.google-apps.presentation":
        ("application/vnd.openxmlformats-officedocument.presentationml.presentation", ".pptx"),
    "application/vnd.google-apps.drawing":
        ("application/pdf", ".pdf"),
}


def is_native(mime_type: str) -> bool:
    """True iff this Drive mime is a native Google type that needs
    ``/export?mimeType=`` instead of ``alt=media``."""
    return (mime_type or "").startswith("application/vnd.google-apps.")


def export_target(mime_type: str) -> Optional[Tuple[str, str]]:
    """Return ``(export_mime, exported_extension)`` for a native mime,
    or ``None`` when the type isn't natively mapped (the caller should
    fall back to ``alt=media``)."""
    return NATIVE_EXPORTS.get(mime_type)
