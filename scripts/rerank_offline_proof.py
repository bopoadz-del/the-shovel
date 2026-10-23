"""Offline reranker proof: does a cross-encoder lift live recall@5?

Run this BEFORE ever enabling RAG_RERANKER with a new model. It pulls top-K
from the LIVE /v1/rag/search (read-only, no platform changes) for the same
seeded ground-truth sample `rag_recall_eval.py` uses, reranks the candidates
locally with the given cross-encoder, and prints baseline vs reranked vs
ceiling recall. Enable the flag only on a measured lift.

Verdict already on record (docs/rag-reranker.md): the default
cross-encoder/ms-marco-MiniLM-L-6-v2 measured NEGATIVE on drive_archive
(2026-08-02, doc-recall@5 47% -> 43%, artifacts in
data/learning/rag_audit/rerank_proof.json).

Usage:
  python scripts/rerank_offline_proof.py                     # defaults below
  python scripts/rerank_offline_proof.py --model BAAI/bge-reranker-base \
      --sample 60 --seed 42 --base https://the-fork-jn3t.onrender.com

Auth: FORK_API_KEY / FORK_TOKEN / FORK_EMAIL+FORK_PASSWORD (same as
rag_recall_eval.py).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx

from scripts.rag_recall_eval import _auth_header, _load_sample, _search

OUT = "data/learning/rag_audit/rerank_proof.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default=os.getenv("FORK_BASE_URL",
                                                "https://the-fork-jn3t.onrender.com"))
    ap.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--sample", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k-fetch", type=int, default=50)
    ap.add_argument("--k-eval", type=int, default=5)
    ap.add_argument("--project", default="drive_archive")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    items = _load_sample(args.sample, args.seed)
    headers = _auth_header(args.base)

    print(f"fetching top-{args.k_fetch} for {len(items)} queries from {args.base} ...",
          flush=True)
    fetched = []
    t0 = time.monotonic()
    with httpx.Client() as client:
        for i, it in enumerate(items, 1):
            try:
                resp = _search(client, args.base, headers, it["query"],
                               args.k_fetch, project=args.project)
            except Exception as exc:  # noqa: BLE001
                print(f"{i:>3} ERROR {str(exc)[:100]}", flush=True)
                continue
            fetched.append({"item": it, "chunks": [
                {"doc_id": c["doc_id"], "chunk_index": c["chunk_index"],
                 "text": c.get("text") or ""}
                for c in (resp.get("chunks") or [])]})
            if i % 10 == 0:
                print(f"{i:>3}/{len(items)} fetched ({time.monotonic()-t0:.0f}s)",
                      flush=True)

    print(f"fetched {len(fetched)} result-sets. loading {args.model} ...", flush=True)
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(args.model, max_length=512)
    print("reranking...", flush=True)

    base_doc = base_chunk = rr_doc = rr_chunk = ceil_doc = ceil_chunk = 0
    details = []
    t1 = time.monotonic()
    for row in fetched:
        it, chunks = row["item"], row["chunks"]
        gt_doc, gt_ci = it["doc_id"], it["chunk_index"]

        def hit(subset, doc=gt_doc, ci=gt_ci):
            d = any(c["doc_id"] == doc for c in subset)
            ch = any(c["doc_id"] == doc and c["chunk_index"] == ci
                     for c in subset)
            return d, ch

        bd, bc = hit(chunks[:args.k_eval])
        cd, cc = hit(chunks)
        if chunks:
            scores = ce.predict([(it["query"], c["text"][:2000]) for c in chunks],
                                show_progress_bar=False)
            order = sorted(range(len(chunks)), key=lambda j: -float(scores[j]))
            rr_top = [chunks[j] for j in order[:args.k_eval]]
        else:
            rr_top = []
        rd, rc = hit(rr_top)

        base_doc += bd; base_chunk += bc
        rr_doc += rd; rr_chunk += rc
        ceil_doc += cd; ceil_chunk += cc
        details.append({"query": it["query"][:90], "base@k": bd,
                        "rerank@k": rd, "in_pool": cd})

    n = len(fetched)
    if not n:
        print("no results fetched — aborting", file=sys.stderr)
        return 1
    summary = {
        "model": args.model, "n": n, "k_fetch": args.k_fetch,
        "k_eval": args.k_eval, "seed": args.seed, "project": args.project,
        f"baseline_doc_recall_at_{args.k_eval}": round(base_doc / n, 4),
        f"reranked_doc_recall_at_{args.k_eval}": round(rr_doc / n, 4),
        f"ceiling_doc_recall_at_{args.k_fetch}": round(ceil_doc / n, 4),
        f"baseline_chunk_recall_at_{args.k_eval}": round(base_chunk / n, 4),
        f"reranked_chunk_recall_at_{args.k_eval}": round(rr_chunk / n, 4),
        f"ceiling_chunk_recall_at_{args.k_fetch}": round(ceil_chunk / n, 4),
        "rerank_seconds_total": round(time.monotonic() - t1, 1),
        "details": details,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)

    print("=" * 60)
    print(f"baseline  doc-recall@{args.k_eval}: {base_doc}/{n} = {base_doc/n:.0%}")
    print(f"RERANKED  doc-recall@{args.k_eval}: {rr_doc}/{n} = {rr_doc/n:.0%}")
    print(f"ceiling   doc-recall@{args.k_fetch}: {ceil_doc}/{n} = {ceil_doc/n:.0%}")
    print(f"results -> {args.out}")
    verdict = "LIFT — candidate may proceed to live trial" \
        if rr_doc > base_doc else "NO LIFT — do not enable this model"
    print(f"verdict: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
