"""Read-only count of stored chunks outside the RQ-2 bounds and locators.

The chunkers now bound every NEW chunk (``app.core.chunk_bounds``). This
reports how much of what the older writers stored is outside those bounds,
so a later ordered re-chunk run can be sized. It never writes.

Per project it reports, over the active and legacy chunk tables:

* the length summary against the configured floor/ceiling (counts outside,
  p50, p95, max);
* PDF chunks with no ``page`` (a PDF is any document whose stored name ends
  in ``.pdf``) and PDF documents with no ``pages_in_file``.
"""
from __future__ import annotations

from typing import Any

from app.core.chunk_bounds import bounds_summary


def plan_chunk_bounds(session, project_id: str) -> dict[str, Any]:
    """Read-only: chunk-length and locator counts for ``project_id``."""
    from sqlalchemy import inspect as _inspect
    from sqlalchemy import text

    from app.core.ingest_reconcile import _guard_chunk_tables

    pdf_pages: dict[str, Any] = {
        str(did): pif for did, name, pif in session.execute(
            text(
                "SELECT id, original_name, pages_in_file FROM documents "
                "WHERE project_id = :pid"
            ),
            {"pid": project_id},
        )
        if str(name or "").lower().endswith(".pdf")
    }
    pdf_ids = set(pdf_pages)
    pdf_without_pages = sum(1 for pif in pdf_pages.values() if not pif)
    lengths: list[int] = []
    pdf_chunks = 0
    pdf_chunks_without_page = 0
    tables = _guard_chunk_tables(session)
    inspector = _inspect(session.get_bind())
    for table in tables:
        has_page = any(c["name"] == "page" for c in inspector.get_columns(table))
        page_col = "page" if has_page else "NULL"
        for did, n, page in session.execute(
            text(
                f"SELECT doc_id, length(text), {page_col} FROM {table} "
                "WHERE project_id = :pid"
            ),
            {"pid": project_id},
        ):
            lengths.append(int(n or 0))
            if str(did) in pdf_ids:
                pdf_chunks += 1
                if page is None:
                    pdf_chunks_without_page += 1
    return {
        "project_id": project_id,
        "tables": tables,
        "length": bounds_summary(lengths),
        "pdf_documents": len(pdf_ids),
        "pdf_documents_without_pages_in_file": pdf_without_pages,
        "pdf_chunks": pdf_chunks,
        "pdf_chunks_without_page": pdf_chunks_without_page,
        "executed": False,
    }


def chunk_bounds_plan(*, project_id: str) -> dict[str, Any]:
    """Open a session and return :func:`plan_chunk_bounds`."""
    from app.core.db import SessionLocal

    if not project_id:
        raise ValueError("project_id is required")
    with SessionLocal() as session:
        return plan_chunk_bounds(session, project_id)
