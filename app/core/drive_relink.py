"""Re-link the shell rows (E-2 of the corpus order, replacing the count job).

2,352 master-corpus rows have size=0, no ``drive_file_id``, and a laptop-side
``source_path`` (``G:\\My Drive\\Master Folder\\...``). The files ARE in Drive
-- the link is missing, not the content. This module matches each row's
basename against the Drive subtree, relinks the decidable ones, and reports
the undecidable ones without ever guessing.

A row is NEVER relinked on a guess: exactly one basename match, or exactly
one size-consistent match among several. ``ambiguous`` and ``missing`` are
real outcomes that stay in the ledger for the next pass, not failures to
paper over.

The relink reopens the row (UNVERIFIED, ledger fields cleared) because the
row is getting a real source for the first time -- exactly the design's
"new source bytes reopen it" (ingest_status.py). It must only ever run when
the extractor fix (E-1) is in the same deploy, or the reopened rows would be
closed terminally by the family that just failed them.
"""

from __future__ import annotations

from typing import Any

FOLDER_MIME = "application/vnd.google-apps.folder"


def basename_of(source_path: str) -> str:
    """The file name of a Windows or POSIX path, platform-neutral."""
    raw = (source_path or "").replace("\\", "/")
    base = raw.rstrip("/").rsplit("/", 1)[-1]
    return base


def build_drive_index(root_folder_id: str) -> dict[str, list[dict[str, Any]]]:
    """``basename.lower()`` -> ``[{id, name, size, path}]`` for the whole
    subtree under ``root_folder_id``, via the platform's own Drive walker."""
    from app.core import gdrive_service

    files, _errors = gdrive_service.walk_folder(root_folder_id)
    index: dict[str, list[dict[str, Any]]] = {}
    for entry in files:
        if (entry.get("mimeType") or "") == FOLDER_MIME:
            continue
        name = (entry.get("name") or "").strip()
        if not name:
            continue
        index.setdefault(name.lower(), []).append({
            "id": entry.get("id"),
            "name": name,
            "size": int(entry.get("size") or 0),
            "path": entry.get("_drive_path") or name,
        })
    return index


def resolve_row(
    row: dict[str, Any],
    index: dict[str, list[dict[str, Any]]],
) -> tuple[str, dict[str, Any] | None]:
    """The relink decision for one row. Never guessed:

    EXACTLY ONE basename match            -> relink
    SEVERAL matches, ONE size-consistent  -> relink
    SEVERAL matches, none decidable       -> ambiguous
    ZERO matches                          -> missing
    """
    base = basename_of(row.get("source_path") or "").strip().lower()
    hits = index.get(base, [])
    if not hits:
        return ("missing", None)
    if len(hits) == 1:
        return ("relink", hits[0])
    want = int(row.get("size_bytes") or 0)
    sized = [h for h in hits if want and h["size"] == want]
    if len(sized) == 1:
        return ("relink", sized[0])
    return ("ambiguous", None)


def rows_without_drive_id() -> list[dict[str, Any]]:
    """Every documents row that has a source path and no drive_file_id."""
    from app.core.db import SessionLocal
    from app.core.models import Document
    from app.core.projects import coerce_document_metadata
    from sqlalchemy import select

    with SessionLocal() as session:
        docs = list(session.scalars(select(Document)).all())
    out: list[dict[str, Any]] = []
    for d in docs:
        meta = coerce_document_metadata(d.metadata_)
        if (meta.get("drive_file_id") or "").strip():
            continue
        source_path = meta.get("source_path") or meta.get("local_drive_path") or ""
        if not source_path:
            continue
        out.append({
            "id": d.id,
            "source_path": source_path,
            "size_bytes": int(d.size or 0),
        })
    return out


def relink_shell_rows(root_folder_id: str, *, apply: bool = False) -> dict[str, Any]:
    """Match every link-missing row against the Drive subtree and relink the
    decidable ones. Dry-run unless ``apply`` -- the run itself is the
    operator's, after the extractor fix is in the same deploy."""
    from app.core import projects as projects_mod

    rows = rows_without_drive_id()
    index = build_drive_index(root_folder_id)
    counts = {"relinked": 0, "ambiguous": 0, "missing": 0, "rows": len(rows)}
    for row in rows:
        outcome, hit = resolve_row(row, index)
        if outcome == "relink":
            counts["relinked"] += 1
            if apply:
                projects_mod.relink_document_drive_id(row["id"], hit["id"])
        elif outcome == "ambiguous":
            counts["ambiguous"] += 1
        else:
            counts["missing"] += 1
    counts["apply"] = apply
    return counts
