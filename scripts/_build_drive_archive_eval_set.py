#!/usr/bin/env python3
"""Build a drive_archive-anchored eval set for the pilot 3-way comparison.

For each candidate (query, expected, source_doc_id, source_chunk_index):
  1. Pull source chunk text from vectors.db and verify every `expected`
     substring is literally present in that chunk's text (case-insensitive).
  2. Run `retrieve_with_filter` against drive_archive and confirm at least
     one of the top-K retrieved chunks contains the expected token.

Drops candidates that fail (1) or (2); writes the survivors to
`drive_archive_eval_set.json` in the operator-specified directory.

Standalone — no LLM calls. Safe to run repeatedly.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("RAG_EMBEDDING_MODEL", "minishlab/potion-base-8M")
os.environ.setdefault("RAG_HYBRID_SEARCH", "true")

from app.core.rag.retriever import retrieve_with_filter  # noqa: E402

PROJECT = "drive_archive"
VECTORS_DB = Path("data/rag/vectors.db")
OUT_PATH = Path("data/learning/adapters/20260613-174725/drive_archive_eval_set.json")


@dataclass
class Candidate:
    id: str
    category: str  # "boq" | "drawing" | "contract"
    query: str
    expected: List[str]
    forbidden: List[str]
    source_doc_id: str
    source_chunk_index: int


# All anchors below were hand-picked from chunk inspection of:
#   - a3eda25f  the client - Infra-1 - Demolition BOQ.pdf  (BOQ rates)
#   - d0c00008  the client Demolition Vol 3 Drawings.pdf  (drawing sheet titles)
#   - d0c00006  the client project Infra-1 Vol 3 Drawings (6 of 7)  (electrical drawings)
#   - 9c116493  the client project Infra-1 Vol 2 Specification (4 of 11)  (spec clauses)
#   - d0c00007  the client project Infra-1 Vol 1 Conditions of Contract  (clauses)
CANDIDATES: List[Candidate] = [
    # ---------------- BOQ (the client Demolition BOQ, doc a3eda25f) -----------------
    Candidate(
        id="DA01",
        category="boq",
        query="In the the client Infrastructure Package 1 Demolition BOQ, what is the unit rate (SAR) for breakout and removal of existing carriageway including road surface markings, CESMM4 ref D 599.5?",
        expected=["31.00"],
        forbidden=[],
        source_doc_id="a3eda25f",
        source_chunk_index=9,
    ),
    Candidate(
        id="DA02",
        category="boq",
        query="What is the unit rate for breakout and removal of existing chain link fence per CESMM4 reference D 549.2 in the the client Demolition BOQ?",
        expected=["80.00"],
        forbidden=[],
        source_doc_id="a3eda25f",
        source_chunk_index=10,
    ),
    Candidate(
        id="DA03",
        category="boq",
        query="In the the client Demolition BOQ, what is the unit rate for protection of an existing 600mm diameter waste water pipeline at average depth 5-6m (CESMM4 ref D 999.46)?",
        expected=["10,317"],
        forbidden=[],
        source_doc_id="a3eda25f",
        source_chunk_index=40,
    ),
    Candidate(
        id="DA04",
        category="boq",
        query="What is the unit rate for general site clearance (CESMM4 ref D110, unit ha) in the the client Infrastructure Package 1 Demolition BOQ?",
        expected=["186,328"],
        forbidden=[],
        source_doc_id="a3eda25f",
        source_chunk_index=1,
    ),
    Candidate(
        id="DA05",
        category="boq",
        query="In the the client Demolition BOQ, what is the unit rate for breakout and remove existing low voltage cables and ducts (CESMM4 ref D 999.3)?",
        expected=["23.00"],
        forbidden=[],
        source_doc_id="a3eda25f",
        source_chunk_index=15,
    ),
    Candidate(
        id="DA06",
        category="boq",
        query="What is the unit rate for waste water manhole 1200mm diameter at depth 2.5-3.0m (CESMM4 ref D 999.29) in the the client Demolition BOQ temporary diversion section?",
        expected=["23,607"],
        forbidden=[],
        source_doc_id="a3eda25f",
        source_chunk_index=30,
    ),

    # ---------------- Drawings (EXC drawings index) ---------------------------
    Candidate(
        id="DA07",
        category="drawing",
        query="In the EXC Volume 3 drawings for the the client QRR detour, drawing QZ-ENG-310-0000-EXC-DWG-TM-200-0014151 — what is the drawing title and what is its revision?",
        expected=["EXAMPLE RING ROAD DETOUR ROAD LAYOUT", "B"],
        forbidden=[],
        source_doc_id="d0c00008",
        source_chunk_index=23,
    ),
    Candidate(
        id="DA08",
        category="drawing",
        query="On the EXC the client project Infrastructure Package 1 electrical drawings, what is the title of drawing QZ-ENG-310-0000-EXC-DWG-EL-600-3101501?",
        expected=["STANDARD INSTALLATION DETAILS"],
        forbidden=[],
        source_doc_id="d0c00006",
        source_chunk_index=850,
    ),
    Candidate(
        id="DA09",
        category="drawing",
        query="In the EXC the client project Infrastructure Package 1 Volume 3 electrical drawings (6 of 7), what does revision letter 'B' correspond to in the revision history block — 'DETAILED DESIGN ISSUE' or 'TENDER ADDENDUM'?",
        expected=["TENDER ADDENDUM"],
        forbidden=[],
        source_doc_id="d0c00006",
        source_chunk_index=891,
    ),
    Candidate(
        id="DA10",
        category="drawing",
        query="What discipline does the EXC drawing prefix 'WS-' refer to in the the client Volume 3 Drawings schedule (e.g. drawing QZ-ENG-310-0000-EXC-DWG-WS-200-0005001)?",
        expected=["Water Supply"],
        forbidden=[],
        source_doc_id="d0c00008",
        source_chunk_index=98,
    ),

    # ---------------- the client Contract (Vol 1 Conditions, Vol 2 Spec) ---------
    Candidate(
        id="DA11",
        category="contract",
        query="Per clause 1.4.1 of the AB-2023-101 Vol 1 Conditions of Contract for the client project Infrastructure Package 1, what law governs the Contract?",
        expected=["Kingdom"],
        forbidden=[],
        source_doc_id="d0c00007",
        source_chunk_index=135,
    ),
    Candidate(
        id="DA12",
        category="contract",
        query="In AB-2023-101 Vol 1 Conditions of Contract clause 1.6 (Contract Agreement), within how many days of the effective date of a Letter of Award must the Parties enter into a Contract Agreement?",
        expected=["90 days"],
        forbidden=[],
        source_doc_id="d0c00007",
        source_chunk_index=138,
    ),
    Candidate(
        id="DA13",
        category="contract",
        query="In AB-2023-101 Vol 2 Specification (4 of 11), what ambient temperature range must all field instruments be designed to operate over (section 3.4 environmental conditions)?",
        expected=["0 to 60"],
        forbidden=[],
        source_doc_id="9c116493",
        source_chunk_index=63,
    ),
    Candidate(
        id="DA14",
        category="contract",
        query="In AB-2023-101 Vol 2 Specification section 3, what ingress protection (IP) rating is required for control panels installed in controlled control-room environments?",
        expected=["IP 42"],
        forbidden=[],
        source_doc_id="9c116493",
        source_chunk_index=64,
    ),
]


def _load_chunk_text(conn: sqlite3.Connection, doc_id: str, chunk_index: int) -> str:
    cur = conn.cursor()
    cur.execute(
        "SELECT text FROM chunks WHERE project_id=? AND doc_id=? AND chunk_index=?",
        (PROJECT, doc_id, chunk_index),
    )
    r = cur.fetchone()
    return r[0] if r else ""


def _retrieval_hits_token(query: str, expected: List[str], k: int = 5) -> tuple:
    """Return (hits_token, retrieved_doc_ids, top_score, n_chunks).
    hits_token is True iff at least one retrieved chunk contains every expected token
    (case-insensitive). We require ALL expected substrings together so the eval signal
    is the same shape as `_score` (which requires every expected match)."""
    chunks, _ = retrieve_with_filter(query, PROJECT, k=k)
    if not chunks:
        return False, [], 0.0, 0
    docs = [c.doc_id for c in chunks]
    top = max((c.score or 0) for c in chunks)
    for c in chunks:
        tl = (c.text or "").lower()
        if all(e.lower() in tl for e in expected):
            return True, docs, top, len(chunks)
    return False, docs, top, len(chunks)


def main() -> int:
    conn = sqlite3.connect(str(VECTORS_DB))
    survivors: List[dict] = []
    rejects: List[dict] = []
    for cand in CANDIDATES:
        text = _load_chunk_text(conn, cand.source_doc_id, cand.source_chunk_index)
        if not text:
            rejects.append({"id": cand.id, "reason": "source chunk not found"})
            continue

        tl = text.lower()
        missing = [e for e in cand.expected if e.lower() not in tl]
        if missing:
            rejects.append({
                "id": cand.id,
                "reason": f"expected token(s) not in source chunk: {missing}",
                "source_excerpt": text[:300],
            })
            continue

        hit, docs, top, nc = _retrieval_hits_token(cand.query, cand.expected, k=5)
        if not hit:
            rejects.append({
                "id": cand.id,
                "reason": "retrieval k=5 did not surface a chunk containing the expected token",
                "retrieved_docs": docs,
                "n_chunks": nc,
                "top_score": round(top, 4),
                "source_excerpt": text[:300],
            })
            continue

        # Build the saved entry
        excerpt = text.strip().replace("\r", " ").replace("\n", " ")
        excerpt = " ".join(excerpt.split())[:300]
        survivors.append({
            "id": cand.id,
            "category": cand.category,
            "query": cand.query,
            "expected": cand.expected,
            "forbidden": cand.forbidden,
            "source_doc_id": cand.source_doc_id,
            "source_chunk_index": cand.source_chunk_index,
            "source_excerpt": excerpt,
            "retrieval_top_score": round(top, 4),
            "retrieval_n_chunks": nc,
            "retrieval_top_doc_ids": docs[:5],
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "project_id": PROJECT,
        "k": 5,
        "n_candidates": len(CANDIDATES),
        "n_survivors": len(survivors),
        "n_rejected": len(rejects),
        "rejected": rejects,
        "queries": survivors,
    }
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Wrote {OUT_PATH}")
    print(f"  survivors: {len(survivors)} / {len(CANDIDATES)}")
    if rejects:
        print(f"  rejected:")
        for r in rejects:
            print(f"    {r['id']}: {r['reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
