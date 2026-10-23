#!/usr/bin/env python
"""Calc-KB-intact eval (RAG_AUDIT_V3 guard rail).

The GK-contamination knobs (RAG_GK_SCORE_MARGIN / RAG_OWN_DOC_BOOST /
RAG_GK_TOPK_CAP, see app/core/rag/retriever.py) are intent-gated: they must
apply to lookup-shaped retrieval only and be bypassed for CALC_KB_INTENTS,
because calculation/standards features DEPEND on the curated GK corpus
winning (unit tables, CESMM classes, FIDIC clause references).

This runner asserts that contract. Three queries whose correct grounding is
a curated GK note are sent to ``POST /v1/rag/search`` scoped to the master
corpus project (that is how chat merges GK into a project conversation):

  1. mass concrete equilibrium time / thermal control limits  -> construction_kb.md
  2. CESMM4 work classes for civil BOQs                       -> boq_units_of_measurement.md
  3. FIDIC 2017 claim notice time bar 20.2                    -> fidic_2017_administration.md

PASS (per case) = a chunk belonging to the GK project is at rank 1.
GK chunks are identified by doc_id membership: the runner lists the GK
project's documents via ``GET /v1/projects/{gk}/documents`` first. A text
signature fallback covers deployments where that listing is unavailable.

Each query runs twice:
  * intent="calculation"  — the guarded path; MUST stay 3/3 under every
    knob config (the knobs must not apply);
  * intent unset          — informational column: what a calc-shaped query
    would get if the caller failed to classify it (knobs DO apply).

Auth/target env: FORK_API_KEY / FORK_TOKEN / FORK_EMAIL+FORK_PASSWORD,
FORK_BASE_URL (default prod).

Usage:
  python scripts/rag_calc_intact_eval.py --project drive_archive \
      --gk-project training_material --out evals/rag_audit_v3/cfg0_calc.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import httpx

DEFAULT_BASE = "https://the-fork-jn3t.onrender.com"

# (label, query, text signature of the expected GK note — fallback GK marker)
CASES = [
    ("mass_concrete_thermal",
     "mass concrete equilibrium time thermal control limits",
     "mass concrete"),
    ("cesmm4_work_classes",
     "CESMM4 work classes for civil BOQs",
     "cesmm4"),
    ("fidic_2017_time_bar",
     "FIDIC 2017 claim notice time bar 20.2",
     "20.2"),
]


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


def _gk_doc_ids(client: httpx.Client, gk_project: str) -> set[str]:
    try:
        r = client.get(f"/v1/projects/{gk_project}/documents")
        r.raise_for_status()
        body = r.json()
        docs = body if isinstance(body, list) else (
            body.get("documents") or body.get("items") or [])
        ids = {str(d["id"]) for d in docs if isinstance(d, dict) and d.get("id")}
        if ids:
            return ids
    except Exception as exc:  # noqa: BLE001 — fall back to text signatures
        print(f"[warn] GK doc listing failed ({exc}); using text-signature "
              f"fallback only", file=sys.stderr)
    return set()


def main() -> int:
    ap = argparse.ArgumentParser(description="calc-KB-intact eval vs /v1/rag/search")
    ap.add_argument("--base", default=os.getenv("FORK_BASE_URL") or DEFAULT_BASE)
    ap.add_argument("--project", default="projects_folder",
                    help="master-corpus project id (prod: projects_folder; "
                         "local boxes: drive_archive)")
    ap.add_argument("--gk-project", default="training_material",
                    help="general-knowledge project id (prod: training_material)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--delay", type=float, default=0.2)
    ap.add_argument("--out", help="artifact path (JSON)")
    args = ap.parse_args()

    headers = _auth_header(args.base)
    client = httpx.Client(base_url=args.base, headers=headers, timeout=120)
    gk_ids = _gk_doc_ids(client, args.gk_project)
    print(f"project={args.project}  gk={args.gk_project} "
          f"({len(gk_ids)} GK docs)  base={args.base}")
    print("=" * 76)

    def is_gk(chunk: dict, signature: str) -> bool:
        if gk_ids and str(chunk.get("doc_id")) in gk_ids:
            return True
        # training_material is a virtual alias project, so doc listing may 404.
        # Its chunk_id is stamped with the project prefix (e.g. "training_material:<doc_id>:<idx>").
        cid = (chunk.get("chunk_id") or "")
        if cid.startswith(f"{args.gk_project}:"):
            return True
        return signature.lower() in (chunk.get("text") or "").lower()

    results = []
    calc_pass = 0
    for label, query, signature in CASES:
        row: dict = {"case": label, "query": query}
        for intent in ("calculation", None):
            time.sleep(args.delay)
            body = {"query": query, "project_id": args.project, "k": args.k}
            if intent:
                body["intent"] = intent
            r = client.post("/v1/rag/search", json=body)
            r.raise_for_status()
            chunks = r.json().get("chunks") or []
            gk_ranks = [i for i, c in enumerate(chunks, 1) if is_gk(c, signature)]
            top = chunks[0] if chunks else None
            col = "with_intent" if intent else "no_intent"
            row[col] = {
                "gk_top1": bool(gk_ranks and gk_ranks[0] == 1),
                "gk_ranks": gk_ranks,
                "top1": {"doc_id": top["doc_id"], "score": round(float(top["score"]), 4),
                         "text_head": (top["text"] or "")[:80]} if top else None,
            }
        results.append(row)
        ok = row["with_intent"]["gk_top1"]
        calc_pass += ok
        print(f"{label:<24} intent=calculation: "
              f"{'PASS' if ok else 'FAIL'} (gk_ranks={row['with_intent']['gk_ranks']})"
              f"   no-intent gk_top1={row['no_intent']['gk_top1']}")

    print("=" * 76)
    print(f"calc-intact: {calc_pass}/{len(CASES)}")
    summary = {"project_id": args.project, "gk_project": args.gk_project,
               "base": args.base, "calc_intact": f"{calc_pass}/{len(CASES)}",
               "all_pass": calc_pass == len(CASES), "results": results}
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=1)
        print(f"artifact: {args.out}")
    return 0 if calc_pass == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
