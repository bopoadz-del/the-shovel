"""Read-only retrieval probe against the full 139,949-chunk drive_archive."""
from __future__ import annotations

import os
import re
import sqlite3
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_DATA_DIR = os.path.abspath(os.environ.setdefault("DATA_DIR", os.path.join(_REPO, "data")))
VECTORS_DB = os.path.join(_DATA_DIR, "rag", "vectors.db")

os.environ.setdefault("RAG_EMBEDDING_MODEL", "minishlab/potion-base-8M")
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from app.core.rag.embeddings import get_embedder
from app.core.rag.vector_store import get_store

PROJECT = "drive_archive"
TARGET_DOC = "d0c00004"  # TL-600-0000002-D.pdf
TARGET_CHUNK = 0

QUERIES_ALL = {
    "Q1": "What is the EXC drawing-number format used on the the client project project?",
    "Q2": "What does the SECTIONAL ELEVATION telecom drawing show?",
    "Q3": "What is the procedure for design review acceptance under PRC-951?",
    "Q4": "What is the payable trench width specification for the water supply pipe?",
    "Q5": "Manhole spacing requirements for telecom ducts on the the client project project?",
}

embedder = get_embedder()
store = get_store(dim=embedder.dim)

con = sqlite3.connect(VECTORS_DB)
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
            return tail[:55]
    return "?"


print("=" * 100)
print("PROBE 1: Q2 / Q5 retrieval, k=20 raw, against full 139,949-chunk index")
print("=" * 100)
for label in ("Q2", "Q5"):
    q = QUERIES_ALL[label]
    print(f"\n--- {label}: {q}")
    qvec = embedder.encode([q])[0]
    raw = store.search(PROJECT, qvec, k=80)
    target_rank = None
    print("  rank doc       chunk  score    filename")
    for rank, c in enumerate(raw, 1):
        marker = ""
        if c.doc_id == TARGET_DOC and c.chunk_index == TARGET_CHUNK:
            marker = "   <== TL chunk 0"
            target_rank = rank
        if rank <= 20 or marker:
            print(
                f"  #{rank:>3} {c.doc_id} {c.chunk_index:>5} "
                f"{c.score:7.4f}  {doc_name(c.doc_id)}{marker}"
            )
    if target_rank is None:
        print("  TL chunk 0 NOT in top-80")
    else:
        print(f"  TL chunk 0 at RANK {target_rank}")

print("\n" + "=" * 100)
print("PROBE 2: TL chunk cosine with/without [source: ...] prefix")
print("=" * 100)
row = con.execute(
    "SELECT text FROM chunks WHERE project_id=? AND doc_id=? AND chunk_index=?",
    (PROJECT, TARGET_DOC, TARGET_CHUNK),
).fetchone()
full = row[0]
stripped = re.sub(r"^\[source:[^\]]+\]\s*", "", full)


def cosine(a, b) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


emb_full = embedder.encode([full])[0]
emb_strip = embedder.encode([stripped])[0]
for label in ("Q5", "Q2"):
    qvec = embedder.encode([QUERIES_ALL[label]])[0]
    cf = cosine(qvec, emb_full)
    cs = cosine(qvec, emb_strip)
    print(f"  {label} vs TL chunk WITH prefix:    {cf:.4f}")
    print(f"  {label} vs TL chunk WITHOUT prefix: {cs:.4f}   delta={cs-cf:+.4f}")

print("\n" + "=" * 100)
print("PROBE 3: RAG_CONFIDENCE_THRESHOLD")
print("=" * 100)
print(f"  env RAG_CONFIDENCE_THRESHOLD: {os.getenv('RAG_CONFIDENCE_THRESHOLD')!r}")
print('  app default (rag/inject.py):  "0.4"')
print(f"  effective at chat-inject path: {os.getenv('RAG_CONFIDENCE_THRESHOLD') or '0.4'}")
print("  retrieve_with_filter does NOT apply threshold; rag_inject does.")
print("  Test calls retrieve_with_filter directly -> 0 candidates dropped by threshold.")

print("\n" + "=" * 100)
print("PROBE 4: All 5 questions, retrieval-only k=5, full corpus")
print("=" * 100)
for label in sorted(QUERIES_ALL):
    q = QUERIES_ALL[label]
    print(f"\n--- {label}: {q}")
    qvec = embedder.encode([q])[0]
    raw = store.search(PROJECT, qvec, k=80)
    tl_rank = None
    for rank, c in enumerate(raw, 1):
        if c.doc_id == TARGET_DOC and c.chunk_index == TARGET_CHUNK:
            tl_rank = rank
            break
    print("  rank doc       chunk  score    filename")
    for i, c in enumerate(raw[:5], 1):
        print(f"  #{i:>3} {c.doc_id} {c.chunk_index:>5} {c.score:7.4f}  {doc_name(c.doc_id)}")
    tlmsg = f"TL chunk 0 at rank {tl_rank}" if tl_rank else "TL chunk 0 NOT in top-80"
    print(f"  {tlmsg}")
