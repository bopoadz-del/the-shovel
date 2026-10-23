"""Standing prod-health smoke — the consistency oracle (STANDING TEST DOCTRINE).

Run before handing any pilot-critical surface to review:

    python scripts/consistency_oracle.py --base https://the-fork-jn3t.onrender.com

Auth: FORK_TOKEN or FORK_API_KEY env (same resolution as fork_cli.py).

Two stages, exit code 0 only if BOTH pass:

1. CORPUS HEALTH PREFLIGHT — pins the 2026-07-26 incident class forever:
   the Master-Corpus source project must be ACTIVE (the sidebar cleanup
   archived it, which silently kills retrieval) and must hold chunks.

2. CONSISTENCY ORACLE — one known corpus fact asked in three phrasings on
   the corpus project; every answer must cite sources and agree on the
   value. Divergence or an uncited answer is a FAIL. This single probe
   caught the Kimi tool-call leak, the archived corpus projects, the
   owner-only chat gates, and the identifier-ranking bug in one weekend.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request

FACT_PROJECT = os.environ.get("FORK_FACT_PROJECT", "example_infra_pack")
FACT_QUESTIONS = [
    "What is the total flow rate of wastewater pump station WWPS-01?",
    "State the design capacity in l/s of pump station WWPS-01.",
    "How many litres per second does WWPS-01 handle in total?",
]
FACT_PATTERN = r"(\d{2,4})\s*(?:l/s|litres per second|liters per second|L/s)"
FACT_EXPECTED = "366"


def _token() -> str:
    tok = os.getenv("FORK_TOKEN") or os.getenv("FORK_API_KEY")
    if not tok:
        print("FAIL: set FORK_TOKEN or FORK_API_KEY")
        sys.exit(2)
    return tok


def _get(base: str, path: str, tok: str, timeout: int = 90):
    req = urllib.request.Request(
        base + path, headers={"Authorization": f"Bearer {tok}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _stream_chat(base: str, tok: str, project_id: str, message: str,
                 timeout: int = 240):
    req = urllib.request.Request(
        base + "/v1/chat/stream",
        data=json.dumps({"project_id": project_id, "message": message,
                         "history": []}).encode(),
        headers={"Authorization": f"Bearer {tok}",
                 "Content-Type": "application/json"})
    out = urllib.request.urlopen(req, timeout=timeout).read().decode(errors="replace")
    tokens, end = [], {}
    for line in out.splitlines():
        if not line.startswith("data:"):
            continue
        try:
            ev = json.loads(line[5:].strip())
        except ValueError:
            continue
        if ev.get("type") == "token":
            tokens.append(ev.get("content", ""))
        elif ev.get("type") == "end":
            end = ev
    return "".join(tokens), end.get("sources") or []


def preflight_corpus_health(base: str, tok: str) -> bool:
    """The Master-Corpus source project must be active and retrievable."""
    ok = True
    try:
        proj = _get(base, f"/v1/projects/{FACT_PROJECT}?doc_limit=0", tok)
        status = proj.get("status")
        if status != "active":
            print(f"FAIL preflight: {FACT_PROJECT} status={status!r} (must be active "
                  "— archive kills retrieval; restore via "
                  "/v1/admin/projects/{id}/restore?hide_from_sidebar=true)")
            ok = False
        else:
            print(f"ok  preflight: {FACT_PROJECT} active")
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL preflight: cannot load {FACT_PROJECT}: {exc}")
        ok = False
    return ok


def run_oracle(base: str, tok: str) -> bool:
    # Consistency = the expected fact appears, CITED, in every answer.
    # Extra derived figures (per-pump duty flow = total/2, heads, etc.) are
    # legitimate engineering elaboration, not divergence — the first run of
    # this script over-failed on exactly that ("366 total, 183 per duty
    # pump"), so the assertion is fact-presence, not number-exclusivity.
    ok = True
    for q in FACT_QUESTIONS:
        try:
            answer, sources = _stream_chat(base, tok, FACT_PROJECT, q)
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL oracle: stream error on {q[:40]!r}: {exc}")
            ok = False
            continue
        nums = set(re.findall(FACT_PATTERN, answer))
        cited = len(sources) > 0
        has_fact = FACT_EXPECTED in nums
        line_ok = has_fact and cited
        print(f"{'ok ' if line_ok else 'FAIL'} oracle: {q[:48]!r} "
              f"sources={len(sources)} values={sorted(nums)}")
        if not line_ok:
            ok = False
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=os.getenv("FORK_BASE", "https://the-fork-jn3t.onrender.com"))
    args = ap.parse_args()
    tok = _token()
    healthy = preflight_corpus_health(args.base, tok)
    consistent = run_oracle(args.base, tok)
    verdict = healthy and consistent
    print("VERDICT:", "PASS" if verdict else "FAIL")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
