#!/usr/bin/env python
"""Fresh-upload-wins eval (RAG_AUDIT_V2, new eval class).

Question being measured: when a project has its OWN freshly uploaded document
and the master/GK corpus contains topically similar material, does
project-scoped retrieval rank the fresh upload's chunks first?

Method (read-mostly; the only mutation is one clearly named eval project plus
small .txt uploads into it — projects must never be deleted, see the schema's
ON DELETE CASCADE on RAG chunks):

  1. create project "RAG Audit V2 - Fresh Upload Eval" (or reuse by name)
  2. upload N small .txt notes; each states a project-specific fact whose
     TOPIC deliberately overlaps the GK/master corpus (lighting wattage,
     FIDIC notice period, design-review statuses, BOQ rates...) but whose
     VALUE is unique to the project
  3. wait for eager indexing
  4. for each case, POST /v1/rag/search with the eval project's id and ask
     for the fact; PASS = a chunk from the uploaded doc is top-1
     (top-3 tracked as a softer signal)

Auth/target env: FORK_API_KEY / FORK_TOKEN / FORK_EMAIL+FORK_PASSWORD,
FORK_BASE_URL (default prod).

Usage:
  python scripts/rag_fresh_upload_eval.py            # full run
  python scripts/rag_fresh_upload_eval.py --query-only --project <id>
                                                     # re-query an existing run
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time

import httpx

DEFAULT_BASE = "https://the-fork-jn3t.onrender.com"
PROJECT_NAME = "FIXTURE — Fresh Upload Eval"
OUT = "data/learning/rag_audit/v2_fresh_upload_wins.json"


def _resolve_project_id(client: httpx.Client, name: str) -> str | None:
    """Look up a fixture project by canonical name; return None if not found."""
    r = client.get("/v1/projects")
    r.raise_for_status()
    for p in r.json().get("projects", []) or []:
        if p.get("name") == name:
            return p["id"]
    return None

# (filename, document text, retrieval query)
# Topics collide with GK/master content on purpose; values are project-unique.
CASES = [
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


def main() -> int:
    ap = argparse.ArgumentParser(description="fresh-upload-wins retrieval eval")
    ap.add_argument("--base", default=os.getenv("FORK_BASE_URL") or DEFAULT_BASE)
    ap.add_argument("--project", help="existing eval project id (skip create)")
    ap.add_argument("--query-only", action="store_true",
                    help="skip create+upload; just re-run the queries")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--wait", type=int, default=90,
                    help="seconds to allow eager indexing before querying")
    ap.add_argument("--delay", type=float, default=3.0)
    ap.add_argument("--out", default=OUT, help="artifact path")
    args = ap.parse_args()

    headers = _auth_header(args.base)
    client = httpx.Client(base_url=args.base, headers=headers, timeout=120)

    project_id = args.project
    created_here = False
    uploaded: dict[str, str] = {}  # filename -> document id

    if not project_id:
        resolved = _resolve_project_id(client, PROJECT_NAME)
        if resolved:
            project_id = resolved
            print(f"[setup] resolved {PROJECT_NAME} -> {project_id}")
        elif args.query_only:
            sys.exit(f"[error] query-only but project {PROJECT_NAME!r} not found")
        else:
            r = client.post("/v1/projects", json={"name": PROJECT_NAME})
            r.raise_for_status()
            project_id = r.json()["id"]
            created_here = True
            print(f"[setup] created project {project_id} ({PROJECT_NAME})")

    if not args.query_only:
        existing_names = {
            d["original_name"]
            for d in (client.get(f"/v1/projects/{project_id}/documents")
                      .raise_for_status().json().get("documents") or [])
        }
        for fname, text, _q in CASES:
            if fname in existing_names:
                print(f"[skip] {fname} already uploaded")
                continue
            r = client.post(
                f"/v1/projects/{project_id}/documents",
                files={"file": (fname, io.BytesIO(text.encode("utf-8")), "text/plain")},
            )
            r.raise_for_status()
            doc = r.json()
            doc_id = doc.get("id") or (doc.get("document") or {}).get("id")
            uploaded[fname] = str(doc_id)
            print(f"[upload] {fname} -> doc {doc_id}")
            time.sleep(1.0)
        if created_here or uploaded:
            print(f"[setup] waiting {args.wait}s for eager indexing...")
            time.sleep(args.wait)

    print(f"project={project_id}  k={args.k}  cases={len(CASES)}")
    print("=" * 76)

    results, top1_wins, top3_wins = [], 0, 0
    known_ids = set(uploaded.values())
    for i, (fname, _text, query) in enumerate(CASES, 1):
        if i > 1 and args.delay:
            time.sleep(args.delay)
        r = client.post("/v1/rag/search",
                        json={"query": query, "project_id": project_id,
                              "k": args.k})
        if r.status_code >= 500:
            time.sleep(30)
            r = client.post("/v1/rag/search",
                            json={"query": query, "project_id": project_id,
                                  "k": args.k})
        r.raise_for_status()
        chunks = r.json().get("chunks") or []

        def _is_fresh(c: dict) -> bool:
            # match by known doc id when we have it, else by a distinctive
            # slice of the note text having landed in the chunk (uploads are
            # single-chunk notes; chunk text may carry a [source: ...] prefix,
            # so substring containment, not startswith)
            if known_ids and c["doc_id"] in known_ids:
                return True
            return _text[:60].lower() in (c["text"] or "").lower()
        fresh_ranks = [j for j, c in enumerate(chunks, 1) if _is_fresh(c)]
        t1 = bool(fresh_ranks and fresh_ranks[0] == 1)
        t3 = bool(fresh_ranks and fresh_ranks[0] <= 3)
        top1_wins += t1
        top3_wins += t3
        results.append({
            "file": fname, "query": query,
            "fresh_rank": fresh_ranks[0] if fresh_ranks else None,
            "top1_win": t1, "top3_win": t3,
            "top_docs": [{"doc_id": c["doc_id"], "chunk_index": c["chunk_index"],
                          "score": round(float(c.get("score") or 0), 4),
                          "text_head": (c["text"] or "")[:90]}
                         for c in chunks[:3]],
        })
        mark = "WIN " if t1 else ("top3" if t3 else "LOSS")
        print(f"case{i:>2}  {mark}  fresh_rank={fresh_ranks[0] if fresh_ranks else '-'}"
              f"  {fname}")

    print("=" * 76)
    n = len(CASES)
    print(f"top-1 fresh wins: {top1_wins}/{n}")
    print(f"top-3 fresh wins: {top3_wins}/{n}")
    summary = {"project_id": project_id, "n_cases": n,
               "top1_wins": top1_wins, "top3_wins": top3_wins,
               "uploaded_docs": uploaded, "results": results}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print(f"artifact: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
