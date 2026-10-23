#!/usr/bin/env python
"""Retrieval recall@K eval against a live Fork instance (RAG_AUDIT re-run).

Measures the two numbers the 2026-06-23 RAG_AUDIT.md baselined:
  - doc recall@K:   ground-truth doc_id appears in the top-K chunks
  - chunk recall@K: ground-truth (doc_id, chunk_index) appears exactly

Two modes, both read-only against ``POST /v1/rag/search``:

  --baseline30   re-run the EXACT 30 queries stored in
                 data/learning/rag_audit/sample_retrieval_recall_projects_folder.json
                 (paired comparison with the June-23 run — same questions,
                 same ground truth, so the delta is attributable to the
                 retrieval stack, not sampling noise).

  --sample N     seeded sample of N questions from
                 data/learning/training_scenarios_drive_archive_v2.jsonl
                 (the audit's canonical in-distribution set; every row's
                 source chunk is known-present in the corpus). Use
                 --seed to make the sample reproducible.

Auth (either, same as fork_cli): FORK_API_KEY / FORK_TOKEN /
FORK_EMAIL+FORK_PASSWORD. Target: FORK_BASE_URL (default prod).

Usage:
  python scripts/rag_recall_eval.py --baseline30
  python scripts/rag_recall_eval.py --sample 100 --seed 42
Artifacts are written incrementally to data/learning/rag_audit/ so a
killed run keeps its finished rows.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import httpx

DEFAULT_BASE = "https://the-fork-jn3t.onrender.com"
BASELINE_30 = "data/learning/rag_audit/sample_retrieval_recall_projects_folder.json"
QUESTIONS_V2 = "data/learning/training_scenarios_drive_archive_v2.jsonl"
# The audit measures the master corpus directly. On prod it lives under
# project_id "projects_folder"; on a local box the same corpus is stored
# under "drive_archive" — override with --project (default unchanged).
PROJECT = "projects_folder"


def _auth_header(base: str) -> dict:
    tok = os.getenv("FORK_TOKEN") or os.getenv("FORK_API_KEY")
    if not tok:
        email, pw = os.getenv("FORK_EMAIL"), os.getenv("FORK_PASSWORD")
        if not (email and pw):
            sys.exit("No credentials: set FORK_API_KEY / FORK_TOKEN / FORK_EMAIL+FORK_PASSWORD")
        r = httpx.post(f"{base}/v1/users/login",
                       json={"email": email, "password": pw}, timeout=30)
        r.raise_for_status()
        tok = r.json()["token"]
        print(f"[auth] logged in as {email}", file=sys.stderr)
    return {"Authorization": f"Bearer {tok}"}


def _search(client: httpx.Client, base: str, headers: dict,
            query: str, k: int, project: str = PROJECT) -> dict:
    # Retry 5xx with a long backoff: a burst of searches over the 110k-chunk
    # corpus can starve the single-worker box until Render's health check
    # times out and restarts it (observed live 2026-07-05, event
    # server_failed/unhealthy). Give the instance time to come back rather
    # than burning the rest of the sample as errors.
    last = None
    for attempt in range(4):
        if attempt:
            time.sleep(25 * attempt)
        try:
            r = client.post(f"{base}/v1/rag/search", headers=headers,
                            json={"query": query, "project_id": project, "k": k},
                            timeout=60)
        except httpx.HTTPError as exc:
            last = exc
            continue
        if r.status_code >= 500:
            last = RuntimeError(f"HTTP {r.status_code}")
            continue
        r.raise_for_status()
        return r.json()
    raise last  # type: ignore[misc]


def _load_baseline30() -> list[dict]:
    with open(BASELINE_30, encoding="utf-8") as fh:
        details = json.load(fh)["details"]
    return [{"query": d["query"], "doc_id": d["doc_id"],
             "chunk_index": d["chunk_index"],
             "baseline_found_doc": d["found_doc"],
             "baseline_found_chunk": d["found_chunk"]} for d in details]


def _load_sample(n: int, seed: int) -> list[dict]:
    rows = []
    with open(QUESTIONS_V2, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            src = row.get("source") or ""
            parts = src.split(":")
            # source format: drive_archive:<doc_id>:<chunk_index>
            if len(parts) == 3 and parts[0] == "drive_archive":
                rows.append({"query": row["instruction"],
                             "doc_id": parts[1],
                             "chunk_index": int(parts[2])})
    random.Random(seed).shuffle(rows)
    return rows[:n]


def main() -> int:
    ap = argparse.ArgumentParser(description="recall@K eval vs live /v1/rag/search")
    ap.add_argument("--baseline30", action="store_true",
                    help="re-run the June-23 30-question paired set")
    ap.add_argument("--sample", type=int, default=0,
                    help="seeded fresh sample size from the v2 training set")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--delay", type=float, default=2.0,
                    help="seconds to sleep between queries (protects the "
                         "single-worker box's health check)")
    ap.add_argument("--start", type=int, default=0,
                    help="run only items[start:end] of the (seeded) draw — "
                         "lets one 100-question sample split across tool-"
                         "timeout windows; artifacts are merged by the caller")
    ap.add_argument("--end", type=int, default=0, help="see --start (0 = all)")
    ap.add_argument("--base", default=os.getenv("FORK_BASE_URL") or DEFAULT_BASE)
    ap.add_argument("--out", help="artifact path (default derived from mode)")
    ap.add_argument("--project", default=PROJECT,
                    help="project id holding the master corpus (prod: "
                         "projects_folder; local boxes: drive_archive)")
    args = ap.parse_args()

    if args.baseline30:
        items = _load_baseline30()
        out = args.out or "data/learning/rag_audit/v2_baseline30_rerun.json"
        mode = "baseline30"
    elif args.sample:
        items = _load_sample(args.sample, args.seed)
        out = args.out or f"data/learning/rag_audit/v2_sample{args.sample}_seed{args.seed}.json"
        mode = f"sample{args.sample}/seed{args.seed}"
    else:
        ap.error("pick --baseline30 or --sample N")
        return 2

    if args.start or args.end:
        lo, hi = args.start, (args.end or len(items))
        items = items[lo:hi]
        out = out.replace(".json", f"_{lo}-{hi}.json")
        mode += f"[{lo}:{hi}]"

    headers = _auth_header(args.base)
    print(f"mode={mode}  n={len(items)}  k={args.k}  project={args.project}  base={args.base}")
    print("=" * 76)

    details, found_doc, found_chunk, top1 = [], 0, 0, []
    t0 = time.monotonic()
    with httpx.Client() as client:
        for i, it in enumerate(items, 1):
            if i > 1 and args.delay:
                time.sleep(args.delay)
            try:
                resp = _search(client, args.base, headers, it["query"], args.k,
                               project=args.project)
            except Exception as exc:  # noqa: BLE001
                print(f"{i:>3}  ERROR {str(exc)[:120]}")
                details.append({**it, "error": str(exc)[:300]})
                continue
            chunks = resp.get("chunks") or []
            fdoc = any(c["doc_id"] == it["doc_id"] for c in chunks)
            fchunk = any(c["doc_id"] == it["doc_id"]
                         and c["chunk_index"] == it["chunk_index"] for c in chunks)
            found_doc += fdoc
            found_chunk += fchunk
            if chunks:
                top1.append(float(chunks[0].get("score") or 0.0))
            details.append({**it, "found_doc": fdoc, "found_chunk": fchunk,
                            "count": len(chunks),
                            "top_scores": [round(float(c.get("score") or 0.0), 6)
                                           for c in chunks]})
            if i % 10 == 0 or i == len(items):
                n_done = len(details)
                print(f"{i:>3}/{len(items)}  doc@{args.k}={found_doc}/{n_done}"
                      f"  chunk@{args.k}={found_chunk}/{n_done}"
                      f"  ({time.monotonic()-t0:.0f}s)")
                # incremental write so a killed run keeps finished rows
                summary = {
                    "mode": mode, "sample_size": n_done, "k": args.k,
                    "project_id": args.project, "base": args.base,
                    f"doc_recall_at_{args.k}": round(found_doc / n_done, 4),
                    f"chunk_recall_at_{args.k}": round(found_chunk / n_done, 4),
                    "avg_top_score": round(sum(top1) / len(top1), 6) if top1 else None,
                    "details": details,
                }
                with open(out, "w", encoding="utf-8") as fh:
                    json.dump(summary, fh, indent=1)

    print("=" * 76)
    n = len(details)
    print(f"doc recall@{args.k}:   {found_doc}/{n} = {found_doc/n:.0%}")
    print(f"chunk recall@{args.k}: {found_chunk}/{n} = {found_chunk/n:.0%}")
    if top1:
        print(f"avg top-1 score:  {sum(top1)/len(top1):.4f}")
    print(f"artifact: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
