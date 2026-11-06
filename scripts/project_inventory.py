#!/usr/bin/env python3
"""Read-only chunk/document inventory for ONE project, id from the environment.

Counts only: parameterised SQL, no DML, no chunk text. The chunk table is
resolved the same way the retriever does (``RAG_VECTOR_NAMESPACE`` ->
``chunks_<ns>``), never hardcoded. Refuses to run without a project id.
"""
from __future__ import annotations

import os
import sys

# Invoked as `python scripts/project_inventory.py` (entrypoint WORKER_COMMAND),
# which puts scripts/ on sys.path, not the repo root. Make `app` importable
# regardless of invocation form (also covers `python -m` and direct runs).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text

from app.core.db import SessionLocal
from app.core.models import rag_chunk_table_name
from app.core.rag.vector_store import _rag_vector_namespace

# Read-only statements. ``{table}`` is the namespace-resolved chunk table name
# (table names cannot be bound parameters); every other value is a bind param.
_CHUNKS_SQL = "SELECT count(*), count(text_search) FROM {table} WHERE project_id = :p"
_DOCUMENTS_SQL = (
    "SELECT ingest_status, retrieval_visible, count(*) FROM documents "
    "WHERE project_id = :p GROUP BY 1, 2 ORDER BY 1, 2"
)


def _counts(session, pid: str) -> tuple:
    table = rag_chunk_table_name(_rag_vector_namespace())
    chunks, fts = session.execute(
        text(_CHUNKS_SQL.format(table=table)), {"p": pid},
    ).one()
    rows = session.execute(text(_DOCUMENTS_SQL), {"p": pid}).all()
    return chunks, fts, rows


def main() -> int:
    pid = (os.environ.get("INVENTORY_PROJECT_ID") or "").strip()
    if not pid:
        sys.stderr.write("INVENTORY_PROJECT_ID is required\n")
        return 2
    with SessionLocal() as s:
        chunks, fts, rows = _counts(s, pid)
    print(f"project={pid} chunks={chunks} fts={fts}")
    print("ingest_status | retrieval_visible | documents")
    for st, vis, n in rows:
        print(f"{st} | {vis} | {n}")
    if not rows and chunks == 0:
        print("RESULT: no documents and no chunks for this project")
    return 0


if __name__ == "__main__":
    sys.exit(main())
