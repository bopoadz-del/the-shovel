#!/usr/bin/env python3
"""Re-open TEXT_SPARSE .docx that were stamped current after embed failure.

Not a Drive crawl, not p1b, and not a status-to-green hand stamp.
Same document id. Only ``extractor_version`` is rewritten.

Usage (Render worker the-fork-ingest, from /app):
    python scripts/reopen_embed_failed_docx.py [--dry-run] [--limit N]
    python scripts/reopen_embed_failed_docx.py --under-extract [--dry-run] [--limit N]

Default selects retrieval_visible ``.docx`` where
    extractor_version == EXTRACTOR_VERSION
    ingest_status == TEXT_SPARSE
    and the index ledger entry has ``rag_error`` or ``rag_indexed == 0``.

That discriminator is what separates embed-failure (re-open) from genuinely
thin text (leave closed). INDEXED rows are never selected.

``--under-extract`` is a second AND discriminator (not OR'd with embed-fail):
    retrieval_visible .docx
    ingest_status TEXT_SPARSE
    extractor_version == EXTRACTOR_VERSION (not already /embed-failed)
    chunk_count == 1
    ingest_status_reason like single_window%
    ledger rag_indexed > 0 AND no rag_error
    documents.size >= 200_000
    text length (chunks_v2 or ledger preview) < 4000

Expected live hit ~5 of the ~57 thin_no_error rows, not the whole class.

For each match, set extractor_version to the sentinel
``{EXTRACTOR_VERSION}/embed-failed``. The column is never nulled.
``docx_stale_extractor_open`` then re-qualifies the row because the sentinel
is not EXTRACTOR_VERSION. ingest_status is never painted.

``--dry-run`` prints ``doc_id=… rag_error=…`` (default) or
``doc_id=… size=… text_len=…`` (``--under-extract``) and ``count=N``.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("RAG_EMBEDDING_MODEL", "fake")


UNDER_EXTRACT_MIN_SIZE = 200_000
UNDER_EXTRACT_MAX_TEXT = 4000


def _sentinel() -> str:
    from app.core.ingest_status import embed_failed_sentinel

    return embed_failed_sentinel()


def ledger_embed_failed(entry: Optional[Mapping[str, Any]]) -> bool:
    """True when the index ledger says embed did not land."""
    if not entry:
        return False
    if entry.get("rag_error"):
        return True
    return entry.get("rag_indexed") == 0


def ledger_embed_ok(entry: Optional[Mapping[str, Any]]) -> bool:
    """True when embed landed: rag_indexed > 0 and no rag_error."""
    if not entry:
        return False
    if entry.get("rag_error"):
        return False
    return int(entry.get("rag_indexed") or 0) > 0


def _reason_is_single_window(reason: Optional[str]) -> bool:
    return (reason or "").startswith("single_window")


def _store_text_len(project_id: str, document_id: str) -> Optional[int]:
    """Sum of chunk texts in the active store (chunks_v2 in prod)."""
    from app.core.rag.vector_store import get_store

    texts = get_store().doc_chunk_texts(project_id, [document_id]).get(document_id) or []
    if not texts:
        return None
    return sum(len(t or "") for t in texts)


def document_text_len(
    doc: Mapping[str, Any],
    entry: Optional[Mapping[str, Any]],
) -> Optional[int]:
    """Sum of chunk text in chunks_v2 -- the only store of text since
    0018 (the ledger ``entry`` carries counts). None if nothing stored."""
    pid = str(doc.get("project_id") or "")
    did = str(doc.get("id") or "")
    if not (pid and did):
        return None
    return _store_text_len(pid, did)


def _ledger_by_doc_id() -> Dict[str, Dict[str, Any]]:
    from sqlalchemy import select

    from app.core import doc_index
    from app.core.db import SessionLocal
    from app.core.models import DocIndex

    doc_index.init_db()
    out: Dict[str, Dict[str, Any]] = {}
    with SessionLocal() as session:
        project_ids = session.scalars(select(DocIndex.project_id)).all()
    for project_id in project_ids:
        data = doc_index._load_index(str(project_id)) or {}
        for entry in data.get("documents") or []:
            if not isinstance(entry, dict):
                continue
            did = entry.get("document_id")
            if did:
                out[str(did)] = entry
    return out


def select_embed_failed_docx_rows(*, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """retrieval_visible .docx stamped current + TEXT_SPARSE + embed failed."""
    from sqlalchemy import select

    from app.core import ingest_status as ist
    from app.core import projects as projects_mod
    from app.core.db import SessionLocal
    from app.core.models import Document
    from app.core.projects import _document_as_dict

    projects_mod.init_db()
    ledger = _ledger_by_doc_id()
    selected: List[Dict[str, Any]] = []
    with SessionLocal() as session:
        rows = session.scalars(
            select(Document).where(Document.retrieval_visible.is_(True))
        ).all()
        docs = [_document_as_dict(document) for document in rows]
    for doc in docs:
        if ist.document_extension(doc) != ".docx":
            continue
        if (doc.get("extractor_version") or "") != ist.EXTRACTOR_VERSION:
            continue
        if doc.get("ingest_status") != ist.TEXT_SPARSE:
            continue
        entry = ledger.get(doc["id"])
        if not ledger_embed_failed(entry):
            continue
        selected.append({**doc, "_rag_error": (entry or {}).get("rag_error")})
    if limit is not None:
        selected = selected[: max(0, int(limit))]
    return selected


def select_under_extracted_docx_rows(*, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Large TEXT_SPARSE .docx with a tiny extract — not the whole thin class."""
    from sqlalchemy import select

    from app.core import ingest_status as ist
    from app.core import projects as projects_mod
    from app.core.db import SessionLocal
    from app.core.models import Document
    from app.core.projects import _document_as_dict

    projects_mod.init_db()
    ledger = _ledger_by_doc_id()
    selected: List[Dict[str, Any]] = []
    with SessionLocal() as session:
        rows = session.scalars(
            select(Document).where(Document.retrieval_visible.is_(True))
        ).all()
        docs = [_document_as_dict(document) for document in rows]
    for doc in docs:
        if ist.document_extension(doc) != ".docx":
            continue
        if (doc.get("extractor_version") or "") != ist.EXTRACTOR_VERSION:
            continue
        if ist.is_embed_failed_sentinel(doc.get("extractor_version")):
            continue
        if doc.get("ingest_status") != ist.TEXT_SPARSE:
            continue
        if int(doc.get("chunk_count") or 0) != 1:
            continue
        if not _reason_is_single_window(doc.get("ingest_status_reason")):
            continue
        entry = ledger.get(doc["id"])
        if not ledger_embed_ok(entry):
            continue
        if int(doc.get("size") or 0) < UNDER_EXTRACT_MIN_SIZE:
            continue
        text_len = document_text_len(doc, entry)
        if text_len is None or text_len >= UNDER_EXTRACT_MAX_TEXT:
            continue
        selected.append({**doc, "_text_len": text_len})
    if limit is not None:
        selected = selected[: max(0, int(limit))]
    return selected


def stamp_embed_failed_sentinel(doc_id: str) -> None:
    """Rewrite extractor_version only. Never null. Never touch ingest_status."""
    from app.core.db import SessionLocal
    from app.core.models import Document

    sentinel = _sentinel()
    with SessionLocal() as session:
        row = session.get(Document, doc_id)
        if row is None:
            return
        row.extractor_version = sentinel
        session.commit()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected rows. No writes.",
    )
    ap.add_argument(
        "--under-extract",
        action="store_true",
        help="Select large under-extracted TEXT_SPARSE .docx, not embed-fail.",
    )
    ap.add_argument("--limit", type=int, default=None, help="Cap selected rows")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.under_extract:
        rows = select_under_extracted_docx_rows(limit=args.limit)
    else:
        rows = select_embed_failed_docx_rows(limit=args.limit)
    if args.dry_run:
        for row in rows:
            if args.under_extract:
                print(
                    f"doc_id={row['id']} "
                    f"size={int(row.get('size') or 0)} "
                    f"text_len={int(row.get('_text_len') or 0)}"
                )
            else:
                print(
                    f"doc_id={row['id']} rag_error={row.get('_rag_error') or '-'}"
                )
        print(f"count={len(rows)}")
        return 0

    for row in rows:
        stamp_embed_failed_sentinel(row["id"])
        if args.under_extract:
            print(
                f"REOPENED doc_id={row['id']} "
                f"extractor_version={_sentinel()} "
                f"size={int(row.get('size') or 0)} "
                f"text_len={int(row.get('_text_len') or 0)}"
            )
        else:
            print(
                f"REOPENED doc_id={row['id']} "
                f"extractor_version={_sentinel()} "
                f"rag_error={row.get('_rag_error') or '-'}"
            )
    print(f"count={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
