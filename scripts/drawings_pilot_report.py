"""Build the quality report for the drawings-only pilot.

Reads:
  - data/logs/drive_indexer_audit_drawings_pilot.jsonl
  - the in-memory vector store at drive_archive_drawings_test (already populated)

Prints per-drawing rows + aggregates + 3 retrieval queries.
"""
from __future__ import annotations
import json
import os
import re
import sys
from collections import defaultdict

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

AUDIT = os.path.join(_REPO_ROOT, r"data\logs\drive_indexer_audit_drawings_pilot.jsonl")
PROJECT_ID = "drive_archive_drawings_test"

RE = re.compile(r"QZ-ENG-\d+-\d+-EXC-DWG-([A-Z]{2,4})-(\d{3})-", re.IGNORECASE)
EXC_RE = re.compile(r"QZ-ENG-\d+-\d+-EXC-DWG-[A-Z]{2,4}-\d{3}-\d+", re.IGNORECASE)


def disc_of(name: str):
    m = RE.search(name)
    if m:
        return m.group(1).upper(), m.group(2)
    return "?", "?"


def fetch_first_chunk_and_titleblock(project_id: str, doc_id: str, doc_path: str):
    """Return (first_chunk_text, titleblock_chunk_text_or_None) via SQL."""
    from app.core.rag.retriever import get_store, get_embedder
    embedder = get_embedder()
    store = get_store(dim=embedder.dim)
    # Direct SQL — the in-process VectorStore uses sqlite, exposed via _conn.
    cur = store._conn.execute(
        "SELECT chunk_index, text FROM chunks "
        "WHERE project_id = ? AND doc_id = ? "
        "ORDER BY chunk_index ASC",
        (project_id, doc_id),
    )
    rows = cur.fetchall()
    if not rows:
        return None, None
    first_chunk = rows[0]["text"]

    fname = os.path.basename(doc_path)
    exc_match = EXC_RE.search(fname)
    titleblock = None
    if exc_match:
        exc = exc_match.group(0).lower()
        for r in rows:
            if exc in (r["text"] or "").lower():
                titleblock = r["text"]
                break

    return first_chunk, titleblock


def main() -> int:
    if not os.path.exists(AUDIT):
        print(f"missing audit: {AUDIT}", file=sys.stderr)
        return 2
    rows = []
    with open(AUDIT, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    print(f"=== AUDIT ROWS: {len(rows)} ===\n")
    for r in rows:
        name = os.path.basename(r["path"])
        d, s = disc_of(name)
        print(
            f"-- {name}\n"
            f"   disc={d} series={s} extractor={r.get('extractor_used')} "
            f"block_status={r.get('block_status')} n_chunks={r.get('n_chunks')} "
            f"extract_chars={r.get('extract_chars')} pages_ocrd={r.get('pages_ocrd')} "
            f"drawing_qto_deferred={r.get('drawing_qto_deferred')} "
            f"is_drawing={r.get('is_drawing')} elapsed_ms={r.get('elapsed_ms')} "
            f"error={r.get('error')}"
        )

    print(f"\n=== PER-DOC RETRIEVED CHUNKS ===\n")
    for r in rows:
        if not r.get("n_chunks"):
            print(f"-- {os.path.basename(r['path'])}: ZERO chunks; skipping chunk dump.\n")
            continue
        first, tb = fetch_first_chunk_and_titleblock(PROJECT_ID, r["doc_id"], r["path"])
        print(f"-- {os.path.basename(r['path'])} (doc_id={r['doc_id']}):")
        if first:
            snip = first[:400].replace("\n", " | ")
            print(f"   FIRST CHUNK[:400]: {snip}")
        else:
            print(f"   FIRST CHUNK[:400]: <not retrievable via path query>")
        if tb:
            snip = tb[:400].replace("\n", " | ")
            print(f"   TITLE-BLOCK CHUNK[:400]: {snip}")
        else:
            print(f"   TITLE-BLOCK CHUNK[:400]: <no chunk contained EXC code from filename>")
        print()

    # Aggregates
    print(f"\n=== AGGREGATES ===")
    nz = sum(1 for r in rows if (r.get("n_chunks") or 0) >= 1)
    zero = sum(1 for r in rows if (r.get("n_chunks") or 0) == 0)
    ocrd = sum(1 for r in rows if (r.get("pages_ocrd") or 0) > 0)
    text_only = sum(1 for r in rows if (r.get("pages_ocrd") or 0) == 0 and (r.get("n_chunks") or 0) > 0)
    total_chars = sum(r.get("extract_chars") or 0 for r in rows)
    total_chunks = sum(r.get("n_chunks") or 0 for r in rows)
    avg_chunks = total_chunks / max(len(rows), 1)
    print(f"  drawings with >=1 chunk:  {nz} / {len(rows)}")
    print(f"  drawings with zero chunk: {zero} / {len(rows)}")
    print(f"  pages_ocrd > 0:           {ocrd} / {len(rows)} (OCR fired)")
    print(f"  text-layer only:          {text_only} / {len(rows)}")
    print(f"  total chars extracted:    {total_chars}")
    print(f"  total chunks:             {total_chunks}")
    print(f"  avg chunks per drawing:   {avg_chunks:.2f}")

    # Retrieval check
    print(f"\n=== RETRIEVAL CHECK (project={PROJECT_ID}) ===")
    from app.core.rag.retriever import retrieve
    queries = [
        "Security drawing SE-200",
        "potable water supply layout",
        "drawing revision date for traffic management",
    ]
    for q in queries:
        print(f"\nQUERY: {q}")
        hits = retrieve(q, PROJECT_ID, k=3)
        if not hits:
            print("  <no hits>")
            continue
        for h in hits:
            snip = (h.text or "")[:150].replace("\n", " | ")
            print(f"  doc_id={h.doc_id} score={getattr(h,'score',None)} snip={snip}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
