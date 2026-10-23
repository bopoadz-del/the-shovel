"""Read-only probe: hybrid (BM25+vector+RRF) vs vector-only on the
live drive_archive corpus. Same 5 questions as the original probe.

Each query runs TWICE:
  (a) hybrid path with query_text passed to store.search()
  (b) vector-only path with query_text=None

Reports per-query top-20 side by side, the rank of the known-correct
chunk in each leg, and a one-line HYBRID BETTER / SAME / WORSE verdict.

Also prints the pre-rerun corpus baseline + test-fix status as a
header (to close the operator's outstanding sign-off), plus peak
RAM after the probe finishes (Render-starter envelope check).
"""
from __future__ import annotations

import os
import sqlite3
import sys

os.environ.setdefault("RAG_EMBEDDING_MODEL", "minishlab/potion-base-8M")
os.environ["RAG_HYBRID_SEARCH"] = "true"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psutil  # noqa: E402

from app.core.rag.embeddings import get_embedder  # noqa: E402
from app.core.rag.vector_store import get_store  # noqa: E402

PROJECT = "drive_archive"
TL_DOC = "d0c00004"        # TL-600-0000002 SECTIONAL ELEVATION
TL_CHUNK = 0
PRC951_DOC = "d0c00005"    # PRC-951_Design Reviews & Acceptance.pdf
TRENCH_DOC = "9c116493"    # AB-2023-101 Vol 2 - Specification (SIGNED)
TRENCH_CHUNKS = {1115, 1116}  # chunks containing literal "trench width"
TOP_K = 20

QUERIES = {
    "Q1": "What is the EXC drawing-number format used on the the client project project?",
    "Q2": "What does the SECTIONAL ELEVATION telecom drawing show?",
    "Q3": "What is the procedure for design review acceptance under PRC-951?",
    "Q4": "What is the payable trench width specification for the water supply pipe?",
    "Q5": "Manhole spacing requirements for telecom ducts on the the client project project?",
}

# Pre-rerun corpus baseline + test-fix status (recorded earlier this
# session — surfaced here so the probe report stands alone).
PRE_RERUN_BASELINE = "139,949 chunks / 2,924 distinct docs / 0 duplicate (doc_id, chunk_index) rows"
TEST_STATUS = (
    "tests/test_rag_injection.py::test_retrieve_drops_noise_before_top_k — GREEN "
    "(fake_search signature extended with query_text=None; 20/20 in file pass)"
)


def main() -> None:
    proc = psutil.Process(os.getpid())

    embedder = get_embedder()
    store = get_store(dim=embedder.dim)

    con = sqlite3.connect("data/rag/vectors.db")
    BS = chr(92)

    def doc_name(doc_id: str) -> str:
        row = con.execute(
            "SELECT text FROM chunks WHERE project_id=? AND doc_id=? "
            "ORDER BY chunk_index LIMIT 1",
            (PROJECT, doc_id),
        ).fetchone()
        if not row:
            return "?"
        t = row[0]
        if t.startswith("[source: "):
            end = t.find("]")
            if end > 0:
                src = t[9:end]
                tail = src.rsplit(BS, 1)[-1].rsplit("/", 1)[-1]
                return tail[:50]
        return "?"

    def is_exc_doc(doc_id: str) -> bool:
        return "EXC" in doc_name(doc_id).upper()

    def known_correct_rank(results, label: str):
        for i, c in enumerate(results, 1):
            if label in ("Q2", "Q5"):
                if c.doc_id == TL_DOC and c.chunk_index == TL_CHUNK:
                    return i
            elif label == "Q3":
                if c.doc_id == PRC951_DOC:
                    return i
            elif label == "Q4":
                if c.doc_id == TRENCH_DOC and c.chunk_index in TRENCH_CHUNKS:
                    return i
            elif label == "Q1":
                if is_exc_doc(c.doc_id):
                    return i
        return None

    def verdict(v_rank, h_rank) -> str:
        if v_rank == h_rank:
            tag = "SAME"
        elif h_rank is None:
            tag = "WORSE"
        elif v_rank is None:
            tag = "BETTER"
        elif h_rank < v_rank:
            tag = "BETTER"
        elif h_rank > v_rank:
            tag = "WORSE"
        else:
            tag = "SAME"
        return f"HYBRID {tag} (vector-only rank={v_rank}, hybrid rank={h_rank})"

    n_chunks = con.execute(
        "SELECT COUNT(*) FROM chunks WHERE project_id=?", (PROJECT,)
    ).fetchone()[0]
    n_docs = con.execute(
        "SELECT COUNT(DISTINCT doc_id) FROM chunks WHERE project_id=?", (PROJECT,)
    ).fetchone()[0]
    dup_count = len(con.execute(
        "SELECT doc_id, chunk_index FROM chunks WHERE project_id=? "
        "GROUP BY doc_id, chunk_index HAVING COUNT(*) > 1 LIMIT 1",
        (PROJECT,),
    ).fetchall())

    print("=" * 105)
    print("PROBE REPORT — hybrid vs vector-only, k=20, no threshold")
    print("=" * 105)
    print(f"  Pre-rerun baseline:       {PRE_RERUN_BASELINE}")
    print(f"  Post-rerun (now):         {n_chunks:,} chunks / {n_docs:,} docs / "
          f"{'0' if dup_count == 0 else dup_count} duplicates")
    print(f"  Test status:              {TEST_STATUS}")
    print(f"  Verdict targets:          Q1=EXC-named doc; Q2/Q5=TL-600 chunk 0 ({TL_DOC}); "
          f"Q3=PRC-951 doc ({PRC951_DOC}); Q4=Vol 2 chunks 1115/1116 ({TRENCH_DOC})")
    print()

    verdicts = {}
    for label, q in QUERIES.items():
        print("=" * 105)
        print(f"{label}: {q}")
        print("=" * 105)
        qvec = embedder.encode([q])[0]

        v_only = store.search(PROJECT, qvec, k=TOP_K, query_text=None)
        hybrid = store.search(PROJECT, qvec, k=TOP_K, query_text=q)

        v_rank = known_correct_rank(v_only, label)
        h_rank = known_correct_rank(hybrid, label)
        v = verdict(v_rank, h_rank)
        verdicts[label] = v

        print()
        print(f"{'rk':>3} | {'VECTOR-ONLY':<60} || {'HYBRID':<60}")
        print(f"{'':>3} | {'doc':<10} {'ch':>5} {'score':>7}  {'filename':<32} || "
              f"{'doc':<10} {'ch':>5} {'score':>7}  {'filename':<32}")
        print("-" * 145)
        for i in range(TOP_K):
            vc = v_only[i] if i < len(v_only) else None
            hc = hybrid[i] if i < len(hybrid) else None

            def fmt(c):
                if c is None:
                    return " " * 60
                mark = " *" if known_correct_rank([c], label) == 1 else "  "
                return f"{c.doc_id:<10}{mark}{c.chunk_index:>4} {(c.score or 0):7.4f}  {doc_name(c.doc_id):<32}"

            print(f"{i+1:>3} | {fmt(vc)} || {fmt(hc)}")
        print()
        print(f"  ==> {v}")
        print()

    # Summary verdict block
    print("=" * 105)
    print("PER-QUERY SUMMARY")
    print("=" * 105)
    for label, v in verdicts.items():
        print(f"  {label}: {v}")
    n_better = sum(1 for v in verdicts.values() if "BETTER" in v)
    n_same = sum(1 for v in verdicts.values() if "SAME" in v)
    n_worse = sum(1 for v in verdicts.values() if "WORSE" in v)
    print(f"  TOTAL: HYBRID BETTER on {n_better}/5,  SAME on {n_same}/5,  WORSE on {n_worse}/5")

    # Peak RAM (Windows: memory_info().peak_wset is the high-water mark)
    mi = proc.memory_info()
    peak_mb = (getattr(mi, "peak_wset", None) or mi.rss) / (1024 * 1024)
    current_mb = mi.rss / (1024 * 1024)
    print()
    print("=" * 105)
    print("PEAK RAM")
    print("=" * 105)
    print(f"  process peak working set: {peak_mb:.1f} MB")
    print(f"  process current RSS:      {current_mb:.1f} MB")
    print(f"  Render starter envelope:  512 MB  (BM25 over-fetch budget consideration)")


if __name__ == "__main__":
    main()
