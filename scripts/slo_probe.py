"""SLO probe — measure the client-facing availability and response-time
objective and write a sanitized verdict.

Reads a JSON file of probe samples (one ``{"status": <int>, "latency_ms":
<float>}`` object per attempt), computes availability and latency percentiles,
compares them against SLO targets read from the environment, and writes a
one-line JSON verdict. No SLO threshold is hardcoded here: the numbers are the
owner's (due 17 Nov per the client-deployment schedule), so an unset target is
report-only, never a page.

Exit code: 0 when every configured target is met (or none is configured), 1
when a configured target is breached. The caller pages on the failed job.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from typing import Any, Dict, List


def _percentile(sorted_values: List[float], pct: float) -> float:
    """Nearest-rank percentile of an already-sorted, non-empty list."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = max(1, int(round(pct / 100.0 * len(sorted_values))))
    return sorted_values[rank - 1]


def _float_env(name: str) -> float | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        raise SystemExit(f"{name} is not a number: {raw!r}")


def evaluate(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Availability + latency percentiles + breaches, given probe samples.

    Zero samples is an explicit ``NO_SAMPLES`` verdict (the probe collected
    nothing -- the service was unreachable or curl produced no response),
    never a crash, so the caller still receives a report and pages on it.
    """
    avail_target = _float_env("SLO_AVAILABILITY")
    p95_target = _float_env("SLO_LATENCY_P95_MS")
    targets_configured = bool(avail_target is not None or p95_target is not None)

    if not samples:
        return {
            "samples": 0,
            "ok": 0,
            "availability": 0.0,
            "latency_p50_ms": 0.0,
            "latency_p95_ms": 0.0,
            "targets_configured": targets_configured,
            "breaches": [],
            "verdict": "NO_SAMPLES",
        }

    total = len(samples)
    ok = sum(1 for s in samples if int(s.get("status") or 0) == 200)
    availability = ok / total
    latencies = sorted(float(s.get("latency_ms") or 0.0) for s in samples)

    breaches: List[str] = []
    if avail_target is not None and availability < avail_target:
        breaches.append(f"availability {availability:.5f} below target {avail_target:.5f}")
    p95 = _percentile(latencies, 95.0)
    if p95_target is not None and p95 > p95_target:
        breaches.append(f"p95 latency {p95:.1f}ms above target {p95_target:.1f}ms")

    return {
        "samples": total,
        "ok": ok,
        "availability": round(availability, 5),
        "latency_p50_ms": round(_percentile(latencies, 50.0), 1),
        "latency_p95_ms": round(p95, 1),
        "targets_configured": targets_configured,
        "breaches": breaches,
        "verdict": "FAIL" if breaches else "PASS",
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SLO probe verdict")
    parser.add_argument("--json", required=True, help="probe samples JSON file")
    parser.add_argument("--out", help="write the verdict JSON here")
    args = parser.parse_args(argv)

    with open(args.json, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if isinstance(payload, dict):
        samples = payload.get("samples") or []
    else:
        samples = payload

    try:
        result = evaluate(samples)
    except ValueError as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 1

    line = json.dumps(result, sort_keys=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(line + "\n")
    print(line)

    # No targets configured is report-only (owner hasn't set the SLO yet);
    # zero samples pages regardless, because the probe could not reach the
    # service at all.
    if result["verdict"] == "NO_SAMPLES" or result["breaches"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
