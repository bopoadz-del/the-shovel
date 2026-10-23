#!/usr/bin/env python3
"""Embedder selection benchmark for the RAG rebuild (P0c).

Scores candidate embedders locally on two evaluation sets using the fixed
namespace-aware vector store:

  1. Recall subset: seeded sample of N questions from
     data/learning/training_scenarios_drive_archive_v2.jsonl. Each row carries
     its ground-truth source chunk in the ``context`` field, so the benchmark
     indexes exactly the chunks it evaluates against.

  2. Fresh-upload wins: the 12 single-chunk project-specific notes from
     scripts/rag_fresh_upload_eval.py. Each note states a unique fact whose
     topic overlaps general knowledge; PASS = the fresh doc is top-1 (top-3
     tracked as a softer signal).

For every candidate a fresh throwaway namespace is created, the eval corpus is
embedded into it, and the queries are run. The old ``chunks`` table is never
touched. Mixed-model contamination is structurally impossible because the
namespace identity columns are stamped and verified at startup.

Usage:
  python scripts/benchmark_embedders.py \
      --embedders minishlab/potion-base-8M \
      --embedders sentence-transformers/all-MiniLM-L6-v2 \
      --embedders BAAI/bge-small-en-v1.5 \
      --recall-sample 20 --fresh-upload --out data/learning/rag_audit/embedder_benchmark.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Repo paths
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

RECALL_PATH = REPO_ROOT / "data" / "learning" / "training_scenarios_drive_archive_v2.jsonl"

# Imported from rag_fresh_upload_eval.py to keep the benchmark in sync.
_FRESH_UPLOAD_CASES = [
    ("P-217-pier-mix-design.txt",
     "Mix design note - Pier P-217. Structural concrete for Pier P-217 is "
     "grade C50/60 with cement content 385 kg/m3 CEM I 42.5N, w/c ratio 0.38, "
     "target slump 180 mm. Trial mix approved by the Engineer on rev C.",
     "What is the cement content for the Pier P-217 concrete mix?"),
    ("road-R9-street-lighting.txt",
     "Street lighting layout - Road R-9. Lighting poles on Road R-9 are "
     "6.0 m galvanised steel columns fitted with 45 W LED luminaires at "
     "36 m spacing, control by astronomical timer plus photocell override.",
     "What wattage LED fixture is specified for the Road R-9 lighting poles?"),
    ("contract-eot-notice-period.txt",
     "Contract administration note. This project is under FIDIC Yellow Book "
     "2017. The Contractor shall give notice of a claim for extension of "
     "time within 21 days of becoming aware of the event, reduced from the "
     "standard 28 days by Particular Condition PC-20.2.",
     "Within how many days must the Contractor give notice of an EOT claim "
     "on this project?"),
    ("sewer-S4-manhole-spacing.txt",
     "Drainage design criteria - Sewer line S-4. Maximum manhole spacing on "
     "sewer line S-4 is 42 m for pipes up to DN400, and 55 m for DN500 and "
     "above, per the approved drainage design basis rev 2.",
     "What is the maximum manhole spacing on sewer line S-4?"),
    ("design-review-status-codes.txt",
     "Document control procedure - design review. Design submittals on this "
     "project are returned with status CODE-A (proceed), CODE-B (proceed "
     "with comments), or CODE-C (revise and resubmit). No other status "
     "codes are used on this project.",
     "What design review status codes are used on this project?"),
    ("boq-excavation-rate.txt",
     "BOQ rate note. The contract rate for excavation in ordinary soil "
     "(item 2.1.1) is SAR 27.50 per m3 including disposal within 5 km. Rock "
     "excavation (item 2.1.4) is SAR 96.00 per m3.",
     "What is the contract rate per m3 for excavation in ordinary soil?"),
    ("rebar-cover-substructure.txt",
     "Durability requirements. Minimum concrete cover to reinforcement for "
     "substructure elements in contact with soil is 75 mm; superstructure "
     "elements are 40 mm, all per the approved durability plan.",
     "What is the minimum concrete cover for substructure elements?"),
    ("asphalt-wearing-course.txt",
     "Pavement design - flexible pavement. The wearing course is 60 mm "
     "polymer-modified asphalt (PMB 76-10), over a 90 mm binder course and "
     "300 mm granular subbase.",
     "What thickness and type is the asphalt wearing course?"),
    ("hse-permit-to-work.txt",
     "HSE plan extract. Toolbox talks are held daily at 06:30 before shift "
     "start. Permit-to-work validity on this project is 12 hours maximum, "
     "and hot work permits require a dedicated fire watch for 60 minutes "
     "after completion.",
     "How long is a permit-to-work valid on this project?"),
    ("mv-substation-soak-test.txt",
     "Commissioning requirements - MV substation. Energisation is followed "
     "by a 5-day soak test at not less than 80 percent load before taking-"
     "over. Trip testing must be witnessed by the Engineer.",
     "How long is the soak test required for the MV substation?"),
    ("evm-cpi-threshold.txt",
     "Project controls procedure. Earned value is reported monthly. A CPI "
     "below 0.92 or an SPI below 0.90 for two consecutive periods triggers "
     "a mandatory corrective action plan to the PMO.",
     "What CPI threshold triggers a corrective action plan?"),
    ("mass-concrete-curing.txt",
     "Concrete works method statement extract. Mass concrete pours are "
     "cured for 14 days minimum using wet hessian and polythene; the "
     "temperature differential across the section shall not exceed 20 C.",
     "How many days must mass concrete pours be cured on this project?"),
]


def _slug(name: str) -> str:
    """Short filesystem/SQLite-safe identifier for an embedder name."""
    return re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_")[:40]


def _load_recall_items(n: int, seed: int) -> List[Dict[str, Any]]:
    """Load N seeded items from the v2 training scenarios file."""
    if not RECALL_PATH.exists():
        raise FileNotFoundError(f"Recall eval file not found: {RECALL_PATH}")
    rows = []
    with RECALL_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            src = row.get("source") or ""
            parts = src.split(":")
            if len(parts) == 3 and parts[0] == "drive_archive":
                rows.append({
                    "query": row["instruction"],
                    "doc_id": parts[1],
                    "chunk_index": int(parts[2]),
                    "text": row.get("context", ""),
                })
    random.Random(seed).shuffle(rows)
    return rows[:n]


def _index_items(store, items: List[Dict[str, Any]], embedder) -> int:
    """Index a list of recall items, grouping by doc_id."""
    from app.core.rag.retriever import chunk_text

    indexed = 0
    by_doc: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for it in items:
        key = (it["project_id"], it["doc_id"])
        by_doc.setdefault(key, []).append(it)

    for (project_id, doc_id), doc_items in by_doc.items():
        doc_items.sort(key=lambda x: x["chunk_index"])
        chunks = [it["text"] for it in doc_items]
        # Re-chunk long contexts so the indexed shape matches production,
        # but preserve chunk_index alignment by indexing each item as a
        # single-chunk document when the text is short enough.
        if all(len(c) <= 512 for c in chunks):
            embeddings = embedder.encode(chunks)
        else:
            # Fall back to chunk_text and index as one doc with multiple chunks.
            flat_chunks: List[str] = []
            for c in chunks:
                flat_chunks.extend(chunk_text(c))
            chunks = flat_chunks
            embeddings = embedder.encode(chunks)
        store.upsert_chunks(project_id, doc_id, chunks, embeddings)
        indexed += len(chunks)
    return indexed


def _run_recall_eval(store, embedder, items: List[Dict[str, Any]], k: int) -> Dict[str, Any]:
    """Run recall@K over the indexed items."""
    found_doc = 0
    found_chunk = 0
    details: List[Dict[str, Any]] = []
    for it in items:
        qvec = embedder.encode_queries([it["query"]])[0]
        chunks = store.search(it["project_id"], qvec, k=k)
        fdoc = any(c.doc_id == it["doc_id"] for c in chunks)
        fchunk = any(
            c.doc_id == it["doc_id"] and c.chunk_index == it["chunk_index"]
            for c in chunks
        )
        found_doc += fdoc
        found_chunk += fchunk
        details.append({
            "query": it["query"][:120],
            "doc_id": it["doc_id"],
            "chunk_index": it["chunk_index"],
            "found_doc": fdoc,
            "found_chunk": fchunk,
        })
    n = len(items)
    return {
        "n": n,
        "k": k,
        "doc_recall_at_k": found_doc / n if n else 0.0,
        "chunk_recall_at_k": found_chunk / n if n else 0.0,
        "details": details,
    }


def _run_fresh_upload_eval(store, embedder, k: int) -> Dict[str, Any]:
    """Run fresh-upload-wins eval locally."""
    project_id = "bench_fresh_upload"
    top1_wins = 0
    top3_wins = 0
    results = []
    for fname, text, query in _FRESH_UPLOAD_CASES:
        doc_id = f"doc_{fname}"
        chunks = [text]
        embeddings = embedder.encode(chunks)
        store.upsert_chunks(project_id, doc_id, chunks, embeddings)

        qvec = embedder.encode_queries([query])[0]
        retrieved = store.search(project_id, qvec, k=k)
        fresh_ranks = [
            j for j, c in enumerate(retrieved, 1)
            if c.doc_id == doc_id or text[:60].lower() in (c.text or "").lower()
        ]
        t1 = bool(fresh_ranks and fresh_ranks[0] == 1)
        t3 = bool(fresh_ranks and fresh_ranks[0] <= 3)
        top1_wins += t1
        top3_wins += t3
        results.append({
            "file": fname,
            "query": query,
            "fresh_rank": fresh_ranks[0] if fresh_ranks else None,
            "top1_win": t1,
            "top3_win": t3,
        })
    n = len(_FRESH_UPLOAD_CASES)
    return {
        "n": n,
        "k": k,
        "top1_wins": top1_wins,
        "top3_wins": top3_wins,
        "top1_rate": top1_wins / n if n else 0.0,
        "top3_rate": top3_wins / n if n else 0.0,
        "results": results,
    }


def _benchmark_one(
    model_name: str,
    recall_items: List[Dict[str, Any]],
    run_fresh: bool,
    recall_k: int,
    fresh_k: int,
) -> Dict[str, Any]:
    """Benchmark one embedder in a throwaway namespace."""
    from app.core.rag import embeddings as _emb
    from app.core.rag import vector_store as _vs

    slug = _slug(model_name)
    namespace = f"bench_{slug}"
    db_path = tempfile.mktemp(suffix=f"_{slug}.db")

    os.environ["RAG_EMBEDDING_MODEL"] = model_name
    os.environ["RAG_VECTOR_NAMESPACE"] = namespace
    _emb.reset_embedder_cache()
    _vs.reset_store_cache()

    t0 = time.monotonic()
    try:
        embedder = _emb.get_embedder(model_name=model_name)
        identity = embedder.identity
        store = _vs.VectorStore(
            db_path=db_path,
            dim=identity["dim"],
            namespace=namespace,
            model_name=model_name,
        )
        load_sec = time.monotonic() - t0

        # Index recall corpus.
        t_index_0 = time.monotonic()
        indexed_count = _index_items(store, recall_items, embedder)
        index_sec = time.monotonic() - t_index_0

        # Run evals.
        recall_result = _run_recall_eval(store, embedder, recall_items, k=recall_k)
        fresh_result = None
        if run_fresh:
            fresh_result = _run_fresh_upload_eval(store, embedder, k=fresh_k)

        return {
            "model": model_name,
            "identity": identity,
            "namespace": namespace,
            "load_seconds": round(load_sec, 2),
            "index_seconds": round(index_sec, 2),
            "indexed_chunks": indexed_count,
            "recall": recall_result,
            "fresh_upload": fresh_result,
        }
    finally:
        _emb.reset_embedder_cache()
        _vs.reset_store_cache()
        try:
            os.unlink(db_path)
        except OSError:
            pass


def _render_markdown(results: List[Dict[str, Any]], recall_k: int, fresh_k: int) -> str:
    lines = [
        "# EMBEDDER_DECISION.md",
        "",
        "Embedder selection benchmark for the Phase-1 RAG rebuild.",
        "Generated by `scripts/benchmark_embedders.py`.",
        "",
        "## Method",
        "",
        "- **Recall subset**: seeded sample from "
        "`data/learning/training_scenarios_drive_archive_v2.jsonl`. Each row's "
        "ground-truth chunk is indexed under its doc_id/chunk_index, then the "
        "question is queried at the same embedder dimension.",
        f"- **Recall metric**: doc/chunk recall@{recall_k}.",
        "- **Fresh-upload wins**: the 12 single-chunk project-specific notes from "
        "`scripts/rag_fresh_upload_eval.py`. PASS = fresh doc is top-1 (top-3 tracked).",
        f"- **Fresh metric**: top-1 / top-3 fresh wins out of {len(_FRESH_UPLOAD_CASES)}.",
        "- Each candidate runs in its own throwaway namespace on the fixed "
        "namespace-aware vector store; the legacy `chunks` table is untouched.",
        "",
        "## Results",
        "",
        "| Model | Dim | Load (s) | Index (s) | doc@{} | chunk@{} | fresh top-1 | fresh top-3 |".format(recall_k, recall_k),
        "|-------|-----|----------|-----------|--------|----------|-------------|-------------|",
    ]
    for r in results:
        if "error" in r:
            lines.append(
                f"| {r['model']} | - | - | - | ERROR | ERROR | ERROR | ERROR |"
            )
            continue
        identity = r["identity"]
        recall = r["recall"]
        fresh = r.get("fresh_upload") or {}
        lines.append(
            "| {model} | {dim} | {load:.1f} | {idx:.1f} | {doc:.2f} | {chunk:.2f} | {t1}/{n1} | {t3}/{n3} |".format(
                model=identity["model"],
                dim=identity["dim"],
                load=r["load_seconds"],
                idx=r["index_seconds"],
                doc=recall["doc_recall_at_k"],
                chunk=recall["chunk_recall_at_k"],
                t1=fresh.get("top1_wins", "-"),
                n1=fresh.get("n", "-"),
                t3=fresh.get("top3_wins", "-"),
                n3=fresh.get("n", "-"),
            )
        )
    lines.extend([
        "",
        "## Recommendation",
        "",
        "(To be filled by the owner after reviewing the numbers above.)",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="embedder selection benchmark")
    ap.add_argument("--embedders", action="append", required=True,
                    help="candidate model name (repeatable)")
    ap.add_argument("--recall-sample", type=int, default=20,
                    help="number of recall questions to sample")
    ap.add_argument("--recall-seed", type=int, default=42)
    ap.add_argument("--recall-k", type=int, default=5)
    ap.add_argument("--fresh-upload", action="store_true",
                    help="also run the 12-case fresh-upload eval")
    ap.add_argument("--fresh-k", type=int, default=5)
    ap.add_argument("--out", default="data/learning/rag_audit/embedder_benchmark.json",
                    help="JSON artifact path")
    ap.add_argument("--md", default="EMBEDDER_DECISION.md",
                    help="Markdown report path")
    args = ap.parse_args()

    if not RECALL_PATH.exists():
        print(f"[error] recall eval file not found: {RECALL_PATH}", file=sys.stderr)
        return 1

    recall_items = _load_recall_items(args.recall_sample, args.recall_seed)
    # Add a synthetic project_id for the recall subset.
    for it in recall_items:
        it["project_id"] = "bench_recall"
    print(f"loaded {len(recall_items)} recall items (seed={args.recall_seed})")

    results: List[Dict[str, Any]] = []
    for model_name in args.embedders:
        print(f"\n=== benchmarking {model_name} ===")
        try:
            result = _benchmark_one(
                model_name,
                recall_items,
                run_fresh=args.fresh_upload,
                recall_k=args.recall_k,
                fresh_k=args.fresh_k,
            )
            results.append(result)
            rec = result["recall"]
            print(f"  dim={result['identity']['dim']}  "
                  f"load={result['load_seconds']:.1f}s  "
                  f"index={result['index_seconds']:.1f}s  "
                  f"doc@{args.recall_k}={rec['doc_recall_at_k']:.2f}  "
                  f"chunk@{args.recall_k}={rec['chunk_recall_at_k']:.2f}")
            if result.get("fresh_upload"):
                fr = result["fresh_upload"]
                print(f"  fresh top-1={fr['top1_wins']}/{fr['n']}  "
                      f"top-3={fr['top3_wins']}/{fr['n']}")
        except Exception as exc:
            print(f"[error] {model_name} failed: {exc}", file=sys.stderr)
            results.append({"model": model_name, "error": str(exc)})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump({
            "recall_sample": args.recall_sample,
            "recall_seed": args.recall_seed,
            "recall_k": args.recall_k,
            "fresh_upload": args.fresh_upload,
            "fresh_k": args.fresh_k,
            "results": results,
        }, fh, indent=2)
    print(f"\njson artifact: {out_path}")

    md = _render_markdown(results, args.recall_k, args.fresh_k)
    md_path = Path(args.md)
    md_path.write_text(md, encoding="utf-8")
    print(f"markdown report: {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
