#!/usr/bin/env python3
"""Compare stored extract length to the #550 content-control extractor.

Walks every ``.docx`` documents row, measures characters currently stored
(chunk text, else doc-index JSON), re-extracts from source bytes with the
fixed walk, and emits ``artifacts/EXTRACTION_CENSUS.md``.

Columns: doc id, filename, sha, chars stored, chars fixed extractor,
delta, source available Y/N. Filenames are ``stored_as`` (opaque) so a
committed fixture run cannot leak client names.

Source resolution order: local ``file_path``, preview cache, Drive
file id. When bytes are gone: NEEDS_SOURCE. Drive has no sha256 search;
a row with only ``content_sha256`` and no pointer stays NEEDS_SOURCE.

``--apply-reingest`` creates a new row per delta>0 + source-available
doc via ``add_document(..., reingest_of=old_id)`` and indexes it
(``stamp_as_indexed=True``). Ops-safe after merge; default is report only.

Never hard-deletes a documents row.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
logger = logging.getLogger(__name__)


def _is_docx(name: str, path: str) -> bool:
    blob = f"{name or ''} {path or ''}".lower()
    return blob.endswith(".docx")


def stored_chars(project_id: str, doc_id: str) -> int:
    """Characters currently retrievable for this document."""
    texts: List[str] = []
    try:
        from app.core.rag.vector_store import get_store

        store = get_store()
        by_doc = store.doc_chunk_texts(project_id, [doc_id])
        texts = by_doc.get(doc_id) or []
    except Exception:
        texts = []
    # The doc_index ledger holds counts, never chunk text (0018): the store
    # is the only place characters can be measured.
    return sum(len(t or "") for t in texts)


def resolve_source_path(doc: Dict[str, Any]) -> tuple[Optional[str], str]:
    """Return ``(path, reason)``. reason is ok / NEEDS_SOURCE / a fetch tag."""
    from app.core import projects as projects_mod

    path, status = projects_mod.materialize_document_file(doc)
    if path and status == "ok":
        return path, "ok"
    return None, "NEEDS_SOURCE"


def census_row(doc: Dict[str, Any]) -> Dict[str, Any]:
    from app.core.doc_index import extract_document_text

    doc_id = str(doc.get("id") or "")
    project_id = str(doc.get("project_id") or "")
    filename = doc.get("stored_as") or os.path.basename(doc.get("file_path") or "") or doc_id
    original_name = doc.get("original_name") or filename
    sha = doc.get("content_sha256") or ""
    stored = stored_chars(project_id, doc_id)
    source_path, source_status = resolve_source_path(doc)
    if source_path:
        from app.core.doc_index import chunk_extracted_document

        extracted = extract_document_text(source_path, filename)
        # Same path index_document uses, so a correct re-index reports delta 0.
        extracted_n = sum(
            len(c) for c in chunk_extracted_document(extracted or "", filename=filename)
        )
        available = "Y"
    else:
        extracted_n = None
        available = "N"
    delta = (extracted_n - stored) if extracted_n is not None else None
    return {
        "id": doc_id,
        "filename": filename,
        "original_name": original_name,
        "sha": sha,
        "chars_stored": stored,
        "chars_fixed": extracted_n,
        "delta": delta,
        "source_available": available,
        "source_status": source_status,
        "project_id": project_id,
        "file_path": source_path,
    }


def render_markdown(rows: List[Dict[str, Any]]) -> str:
    lines = [
        "# Extraction census",
        "",
        "Opaque ids / stored_as only. No live client names.",
        "",
        "| doc id | filename | sha | chars stored | chars fixed | delta | source |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for r in rows:
        sha = (r.get("sha") or "")[:12]
        fixed = r["chars_fixed"] if r["chars_fixed"] is not None else "—"
        delta = r["delta"] if r["delta"] is not None else "—"
        src = r["source_available"]
        if src == "N":
            src = "NEEDS_SOURCE"
        lines.append(
            f"| {r['id']} | {r['filename']} | {sha} | {r['chars_stored']} "
            f"| {fixed} | {delta} | {src} |"
        )
    n = len(rows)
    need = sum(1 for r in rows if r["source_available"] != "Y")
    with_src = n - need
    delta_pos = sum(1 for r in rows if (r.get("delta") or 0) > 0)
    lines.extend([
        "",
        f"docs={n} with_source={with_src} needs_source={need} "
        f"delta_gt_0={delta_pos}",
        "",
        "Re-ingest (ops, after merge) for every row with delta>0 and "
        "source Y:",
        "",
        "```",
        "CEREBRUM_VIRGIN=false CEREBRUM_DOMAIN_KITS=construction \\",
        "  .venv/bin/python scripts/extraction_census.py --apply-reingest",
        "```",
        "",
        "That path calls ``add_document(..., reingest_of=<old_id>)`` then",
        "``index_document(..., stamp_as_indexed=True)``. Hard delete is",
        "forbidden. Flip ``retrieval_visible`` to undo a hide.",
        "",
    ])
    return "\n".join(lines) + "\n"


def apply_reingest(row: Dict[str, Any]) -> Dict[str, Any]:
    from app.core import doc_index, projects as projects_mod

    if (row.get("delta") or 0) <= 0 or row.get("source_available") != "Y":
        return {"skipped": True, "reason": "no_delta_or_no_source"}
    path = row.get("file_path")
    if not path:
        return {"skipped": True, "reason": "NEEDS_SOURCE"}
    new_doc = projects_mod.add_document(
        project_id=row["project_id"],
        original_name=row.get("original_name") or row["filename"],
        stored_as=row["filename"],
        file_path=path,
        size=os.path.getsize(path) if os.path.isfile(path) else 0,
        content_sha256=row.get("sha") or None,
        metadata={"source": "extraction_census_reingest"},
        reingest_of=row["id"],
    )
    result = doc_index.index_document(
        row["project_id"], new_doc["id"], stamp_as_indexed=True,
    )
    return {
        "skipped": False,
        "old_id": row["id"],
        "new_id": new_doc["id"],
        "index": result,
    }


def run_census(
    *,
    project_id: Optional[str] = None,
    do_reingest: bool = False,
    out_path: Optional[Path] = None,
) -> Dict[str, Any]:
    from app.core import projects as projects_mod
    from app.core.db import SessionLocal
    from app.core.models import Document
    from sqlalchemy import select

    projects_mod.init_db()
    with SessionLocal() as session:
        stmt = select(Document)
        if project_id:
            stmt = stmt.where(Document.project_id == project_id)
        docs = [_document_if_docx(d) for d in session.scalars(stmt).all()]
    docs = [d for d in docs if d is not None]
    rows = [census_row(d) for d in docs]
    md = render_markdown(rows)
    dest = out_path or (REPO / "artifacts" / "EXTRACTION_CENSUS.md")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(md, encoding="utf-8")

    reingested: List[Dict[str, Any]] = []
    if do_reingest:
        for row in rows:
            if (row.get("delta") or 0) > 0 and row.get("source_available") == "Y":
                reingested.append(apply_reingest(row))

    return {
        "rows": rows,
        "path": str(dest),
        "reingested": reingested,
        "markdown": md,
    }


def _document_if_docx(document: Any) -> Optional[Dict[str, Any]]:
    from app.core.projects import _document_as_dict

    d = _document_as_dict(document)
    if _is_docx(d.get("original_name") or "", d.get("file_path") or d.get("stored_as") or ""):
        return d
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", default=None)
    ap.add_argument("--apply-reingest", action="store_true")
    ap.add_argument(
        "--output",
        default=None,
        help="Markdown path (default artifacts/EXTRACTION_CENSUS.md)",
    )
    args = ap.parse_args(argv)
    report = run_census(
        project_id=args.project,
        do_reingest=args.apply_reingest,
        out_path=Path(args.output) if args.output else None,
    )
    print(
        f"[census] docs={len(report['rows'])} "
        f"reingested={len(report['reingested'])} "
        f"out={report['path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
