#!/usr/bin/env python3
"""In-place re-extract of stale TEXT_SPARSE .docx rows.

Not a Drive crawl and not web Shell p1b. Same document id, no new row.

Usage (Render worker the-fork-ingest, from /app):
    python scripts/reextract_stale_docx.py [--dry-run] [--limit N] [--allow-unavailable]

Selects retrieval_visible rows where
``ingest_status.docx_stale_extractor_open`` is true, fetches bytes from
the row's one Drive file id, writes the existing ``file_path``, and re-indexes
with ``stamp_as_indexed=False``.

``--limit N`` caps rows that successfully fetch bytes (source drive) and
are written. Rows with no bytes (source=none) are counted as
``source_unavailable`` and do not consume the cap. Dry-run still prints
every open row (annotate ``would_write=`` when a limit is set).

Exit 1 when any open row is ``source_unavailable`` (unfinished work), or
when the fetchable write set was not fully retried. ``--allow-unavailable``
returns 0 when the only unfinished rows are source_unavailable.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("RAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
os.environ.setdefault("RAG_VECTOR_NAMESPACE", "v2")


def _snap(doc: Dict[str, Any]) -> str:
    return (
        f"{doc.get('ingest_status') or '-'}/"
        f"{doc.get('extractor_version') or '-'}/"
        f"{int(doc.get('chunk_count') or 0)}"
    )


def select_stale_docx_rows() -> List[Dict[str, Any]]:
    """All retrieval_visible rows still open under ``docx_stale_extractor_open``.

    ``--limit`` is applied later to rows that actually fetch bytes, not here.
    """
    from sqlalchemy import select

    from app.core import ingest_status as ist
    from app.core import projects as projects_mod
    from app.core.db import SessionLocal
    from app.core.models import Document
    from app.core.projects import _document_as_dict

    projects_mod.init_db()
    with SessionLocal() as session:
        rows = session.scalars(
            select(Document).where(Document.retrieval_visible.is_(True))
        ).all()
    selected: List[Dict[str, Any]] = []
    for document in rows:
        doc = _document_as_dict(document)
        if ist.docx_stale_extractor_open(
            doc.get("ingest_status"),
            extension=ist.document_extension(doc),
            extractor_version=doc.get("extractor_version"),
        ):
            selected.append(doc)
    return selected


def fetch_row_bytes(
    doc: Dict[str, Any],
) -> Tuple[Optional[bytes], str, Optional[str], Optional[str]]:
    """One Drive file id -- the only source of an original. Never a folder walk.

    Returns ``(bytes, source, drive_file_id, err)`` where source is
    ``drive`` / ``none``.
    """
    from app.core import gdrive_service
    from app.core.projects import extract_document_source_pointers

    pointers = extract_document_source_pointers(doc)
    meta = doc.get("metadata") or {}
    if not isinstance(meta, dict):
        meta = {}
    drive_id = pointers.get("drive_file_id") or meta.get("drive_file_id") or ""

    if drive_id:
        raw, err = gdrive_service.download_file_bytes(drive_id)
        if raw:
            return raw, "drive", drive_id, None
        return None, "none", drive_id, err
    return None, "none", None, "no drive_file_id"


def _tally_ended(status: Optional[str], index_status: Optional[str]) -> str:
    from app.core import ingest_status as ist

    if status == ist.INDEXED:
        return "now_indexed"
    if status == ist.TEXT_SPARSE:
        return "still_sparse"
    if index_status == "error" or status in (
        ist.EXTRACT_FAILED, ist.ZERO_CHUNK, ist.UNSUPPORTED_TYPE,
    ):
        return "error"
    return "error"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print id / drive_file_id / source (drive|none). No writes.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Cap rows that successfully fetch bytes and are written/re-indexed. "
            "source=none does not consume the cap."
        ),
    )
    ap.add_argument(
        "--allow-unavailable",
        action="store_true",
        help=(
            "Exit 0 when retried matches the fetchable write set even if some "
            "open rows are source_unavailable."
        ),
    )
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    from app.core import doc_index, file_crypto
    from app.core import ingest_status as ist
    from app.core import projects as projects_mod

    rows = select_stale_docx_rows()
    write_cap = None if args.limit is None else max(0, int(args.limit))
    tally = {
        "open": len(rows),
        "retried": 0,
        "now_indexed": 0,
        "still_sparse": 0,
        "error": 0,
        "source_unavailable": 0,
    }
    unavailable: List[Tuple[str, Optional[str], Optional[str]]] = []
    writes_done = 0

    if args.dry_run:
        for row in rows:
            raw, source, drive_id, _err = fetch_row_bytes(row)
            suffix = ""
            if write_cap is not None:
                will = raw is not None and writes_done < write_cap
                if will:
                    writes_done += 1
                suffix = f" would_write={1 if will else 0}"
            if raw is None:
                tally["source_unavailable"] += 1
            print(
                f"DRY-RUN doc_id={row['id']} "
                f"drive_file_id={drive_id or '-'} source={source}{suffix}"
            )
        print(
            f"TALLY open={tally['open']} retried=0 now_indexed=0 "
            f"still_sparse=0 error=0 "
            f"source_unavailable={tally['source_unavailable']} dry_run=1"
        )
        return 0

    for row in rows:
        before = projects_mod.get_document(row["id"]) or row
        raw, _source, drive_id, err = fetch_row_bytes(row)
        if raw is None:
            tally["source_unavailable"] += 1
            unavailable.append((row["id"], drive_id, err))
            print(
                f"SOURCE_UNAVAILABLE doc_id={row['id']} "
                f"drive_file_id={drive_id or '-'} err={err or 'none'}"
            )
            print(
                f"VERIFICATION doc_id={row['id']} "
                f"before={_snap(before)} after={_snap(before)}"
            )
            continue
        if write_cap is not None and writes_done >= write_cap:
            continue

        dest = row.get("file_path") or ""
        after = before
        tally["retried"] += 1
        writes_done += 1
        wrote = False
        try:
            if not dest:
                raise RuntimeError("document file_path is empty")
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            file_crypto.write_document(dest, raw)
            wrote = True
            result = doc_index.index_document(
                row["project_id"], row["id"], stamp_as_indexed=False,
            )
            Path(dest).unlink(missing_ok=True)
            after = projects_mod.get_document(row["id"]) or before
            if after.get("ingest_status") not in ist.ALL_STATUSES:
                n = int(result.get("rag_indexed") or result.get("total_chunks") or 0)
                after = {
                    **after,
                    "ingest_status": ist.classify(
                        chunk_count=n,
                        extension=ist.document_extension(row) or ".docx",
                    ).status,
                    "extractor_version": after.get("extractor_version")
                    or ist.EXTRACTOR_VERSION,
                    "chunk_count": n or after.get("chunk_count") or 0,
                }
            bucket = _tally_ended(after.get("ingest_status"), result.get("status"))
            tally[bucket] += 1
        except Exception as exc:  # noqa: BLE001 — one row must not kill the run
            tally["error"] += 1
            if wrote and dest:
                Path(dest).unlink(missing_ok=True)
            after = projects_mod.get_document(row["id"]) or before
            print(
                f"ERROR doc_id={row['id']} {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

        print(
            f"VERIFICATION doc_id={row['id']} "
            f"before={_snap(before)} after={_snap(after)}"
        )

    print(
        f"TALLY open={tally['open']} retried={tally['retried']} "
        f"now_indexed={tally['now_indexed']} still_sparse={tally['still_sparse']} "
        f"error={tally['error']} source_unavailable={tally['source_unavailable']}"
    )
    for doc_id, drive_id, uerr in unavailable:
        print(
            f"UNAVAILABLE_RECORD id={doc_id} drive_file_id={drive_id or '-'} "
            f"err={uerr or 'none'}"
        )
    fetchable = tally["open"] - tally["source_unavailable"]
    expected = (
        fetchable if write_cap is None else min(write_cap, fetchable)
    )
    if tally["retried"] != expected:
        return 1
    # A selected row we could not fetch is unfinished work.
    if tally["source_unavailable"] and not args.allow_unavailable:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
