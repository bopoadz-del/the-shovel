#!/usr/bin/env python
"""One-command eval battery (S5 — PILOT_READINESS T6).

Runs the three pre-pilot evidence sweeps in sequence against a live Fork
instance and writes a combined summary:

  1. feature_matrix_sweep.py   — 68-run feature matrix + FEATURE_MATRIX.md
  2. golden_set_gate.py        — 28-query golden set; THE pilot gate
                                 (>= 90% PASS -> exit 0)
  3. rag_recall_eval.py        — retrieval recall@K (--baseline30), only when
                                 its data files exist locally (they live in
                                 gitignored data/learning/ on the operator's
                                 machine); skipped honestly otherwise.

Auth (env only, identical to the underlying scripts — values are never
printed or stored): FORK_TOKEN or FORK_API_KEY, or FORK_EMAIL+FORK_PASSWORD.
Target from FORK_BASE_URL (default prod).

Usage:
  python scripts/run_eval_battery.py --dry-run     # validate all, no network
  python scripts/run_eval_battery.py               # full battery
  python scripts/run_eval_battery.py --skip recall # e.g. no local data files
  python scripts/run_eval_battery.py --delay 30    # be gentle to a free box

The battery is kill-safe because every underlying sweep is: results append
to *.jsonl as they happen and each script's --start / --report can resume or
regenerate. Exit code: 0 iff every stage that RAN succeeded, where success
means exit 0 of the underlying script (for the golden set that IS the >= 90%
gate). A skipped stage never fails the battery but is named in the summary.

Artifacts:
  feature_matrix_results.jsonl / FEATURE_MATRIX.md / review_pack/
  golden_set_results.jsonl / GOLDEN_SET_REPORT.md
  data/learning/rag_audit/*  (recall)
  EVAL_BATTERY_SUMMARY.md    (this script)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RECALL_BASELINE = ROOT / "data/learning/rag_audit/sample_retrieval_recall_projects_folder.json"
SUMMARY_MD = ROOT / "EVAL_BATTERY_SUMMARY.md"

STAGES = ("matrix", "golden", "recall")


def _run(cmd: list, label: str) -> int:
    print(f"\n=== [{label}] {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=str(ROOT))


def _jsonl_counts(path: Path, key: str = "verdict") -> dict:
    """Last-wins verdict counts from a sweep's incremental jsonl."""
    if not path.exists():
        return {}
    rows: dict = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = (rec.get("action") or rec.get("id"),
                   rec.get("prompt_index"), rec.get("run_index"))
            rows[rid] = rec
    counts: dict = {}
    for rec in rows.values():
        v = rec.get(key) or "?"
        counts[v] = counts.get(v, 0) + 1
    return counts


def _fmt_counts(counts: dict) -> str:
    # " · " separator, not " | " — these strings land in a markdown table.
    if not counts:
        return "no results file"
    order = ("PASS", "PARTIAL", "FAIL", "BLOCKED")
    parts = [f"{k} {counts[k]}" for k in order if k in counts]
    parts += [f"{k} {v}" for k, v in sorted(counts.items()) if k not in order]
    return " · ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description="run the full pre-pilot eval battery")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate every stage without network calls")
    ap.add_argument("--skip", action="append", choices=STAGES, default=[],
                    help="skip a stage (repeatable)")
    ap.add_argument("--delay", type=float, default=None,
                    help="seconds between requests, passed to each sweep")
    ap.add_argument("--sample", type=int, default=0,
                    help="ALSO run a seeded recall sample of N (besides --baseline30)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    py = sys.executable
    results: dict = {}          # stage -> (status, detail)

    # 1) feature matrix ------------------------------------------------------
    if "matrix" in args.skip:
        results["matrix"] = ("SKIPPED", "by --skip")
    else:
        cmd = [py, "scripts/feature_matrix_sweep.py"]
        if args.dry_run:
            cmd.append("--dry-run")
        if args.delay is not None:
            cmd += ["--delay", str(args.delay)]
        rc = _run(cmd, "feature matrix")
        if rc == 0 and not args.dry_run:
            _run([py, "scripts/feature_matrix_sweep.py", "--report"],
                 "feature matrix report")
        results["matrix"] = ("OK" if rc == 0 else f"EXIT {rc}",
                             _fmt_counts(_jsonl_counts(ROOT / "feature_matrix_results.jsonl")))

    # 2) golden set (the gate) ----------------------------------------------
    if "golden" in args.skip:
        results["golden"] = ("SKIPPED", "by --skip")
    else:
        cmd = [py, "scripts/golden_set_gate.py"]
        if args.dry_run:
            cmd.append("--dry-run")
        if args.delay is not None:
            cmd += ["--delay", str(args.delay)]
        rc = _run(cmd, "golden set gate")
        results["golden"] = ("OK" if rc == 0 else f"EXIT {rc} (below the >=90% bar?)",
                             _fmt_counts(_jsonl_counts(ROOT / "golden_set_results.jsonl")))

    # 3) recall (only with local data) --------------------------------------
    if "recall" in args.skip:
        results["recall"] = ("SKIPPED", "by --skip")
    elif not RECALL_BASELINE.exists():
        results["recall"] = ("SKIPPED",
                             f"data file missing: {RECALL_BASELINE.relative_to(ROOT)} "
                             "(gitignored — run on the operator machine)")
    elif args.dry_run:
        results["recall"] = ("OK", "dry-run: data files present")
    else:
        rc = _run([py, "scripts/rag_recall_eval.py", "--baseline30"], "recall baseline30")
        if args.sample:
            rc2 = _run([py, "scripts/rag_recall_eval.py",
                        "--sample", str(args.sample), "--seed", str(args.seed)],
                       "recall sample")
            rc = rc or rc2
        results["recall"] = ("OK" if rc == 0 else f"EXIT {rc}",
                             "see data/learning/rag_audit/")

    # summary ----------------------------------------------------------------
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    mode = "DRY-RUN" if args.dry_run else "LIVE"
    base = os.getenv("FORK_BASE_URL") or "https://the-fork-jn3t.onrender.com"
    lines = [
        "# Eval battery summary",
        "",
        f"Generated {ts} — mode {mode} — target {base}",
        "",
        "| stage | status | headline |",
        "|---|---|---|",
    ]
    for stage in STAGES:
        status, detail = results[stage]
        lines.append(f"| {stage} | {status} | {detail} |")
    lines += [
        "",
        "Details: FEATURE_MATRIX.md (per-feature verdicts + triage classes, "
        "5 = NO_OUTPUT), GOLDEN_SET_REPORT.md (the gate), "
        "data/learning/rag_audit/ (recall).",
        "",
    ]
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"wrote {SUMMARY_MD}")

    failed = [s for s, (status, _) in results.items()
              if status not in ("OK",) and not status.startswith("SKIPPED")]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
