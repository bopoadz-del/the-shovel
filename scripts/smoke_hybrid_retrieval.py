"""Smoke-test hybrid retrieval against the live drive_archive SQLite DB.

Reads the SQLite at ``data/rag/vectors.db`` (139k+ chunks under
project_id ``drive_archive``). Runs the canonical Q2 + Q5 probes both
semantic-only and hybrid, prints the top-5 for each, and tags the TL
SECTIONAL ELEVATION chunk (``QZ-ENG-310-0000-EXC-DWG-TL-600-0000002``)
where it lands.

Usage:
    .venv/Scripts/python.exe scripts/smoke_hybrid_retrieval.py

This is SQLite-path only. The PostgreSQL hybrid leg (added by Alembic
0003) is not exercised here — there is no Postgres on this host.
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional

# Repo on sys.path so app.* imports work from a top-level script run.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# Point DATA_DIR at the live DB before importing the store so the
# default db_path resolution picks it up.
os.environ.setdefault("DATA_DIR", os.path.join(_REPO, "data"))

from app.core.rag.embeddings import get_embedder  # noqa: E402
from app.core.rag.vector_store import VectorStore, Chunk  # noqa: E402


LIVE_DB = os.path.join(_REPO, "data", "rag", "vectors.db")
PROJECT_ID = "drive_archive"
TL_MARKER = "TL-600-0000002"

PROBES = {
    "Q2": "SECTIONAL ELEVATION drawing for telecom infrastructure",
    "Q5": "Manhole spacing requirements for telecom ducts on the the client project project",
}


def _print_results(label: str, results: List[Chunk]) -> Optional[int]:
    """Print top-5 results; return the 1-based rank of the TL chunk if
    found in the list (top-N as printed), else None."""
    print(f"  [{label}]")
    tl_rank = None
    for i, c in enumerate(results, 1):
        is_tl = TL_MARKER in (c.text or "") or TL_MARKER in (c.chunk_id or "")
        flag = "  <-- TL chunk" if is_tl else ""
        if is_tl and tl_rank is None:
            tl_rank = i
        snippet = (c.text or "").replace("\n", " ")[:120]
        print(f"    {i:>2}. score={c.score!r:>8} doc={c.doc_id} :: {snippet}{flag}")
    if tl_rank is None:
        print(f"    [TL chunk NOT in top-{len(results)}]")
    else:
        print(f"    [TL chunk at rank {tl_rank}]")
    return tl_rank


def main() -> int:
    if not os.path.exists(LIVE_DB):
        print(f"ERROR: live DB not found at {LIVE_DB}")
        return 2

    print(f"Live DB: {LIVE_DB}")
    print(f"Project: {PROJECT_ID}")
    print()

    embedder = get_embedder()
    store = VectorStore(db_path=LIVE_DB, dim=embedder.dim)
    total = store.count(PROJECT_ID)
    print(f"Indexed chunks for project: {total}")
    print()

    for label, query in PROBES.items():
        print(f"=== {label}: {query!r} ===")
        q_vec = embedder.encode([query])[0]

        # Semantic-only (RAG_HYBRID_SEARCH=false or query_text not passed).
        os.environ["RAG_HYBRID_SEARCH"] = "false"
        sem = store.search(PROJECT_ID, q_vec, k=5, query_text=query)
        sem_rank = _print_results("semantic-only", sem)

        # Hybrid (env on, query_text supplied).
        os.environ["RAG_HYBRID_SEARCH"] = "true"
        hyb = store.search(PROJECT_ID, q_vec, k=5, query_text=query)
        hyb_rank = _print_results("hybrid", hyb)

        print()
        print(f"  TL chunk rank: semantic={sem_rank}  hybrid={hyb_rank}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
