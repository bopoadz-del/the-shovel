"""Unit tests for the SLO probe verdict (scripts/slo_probe.py).

The mutation gate is two inputs per decision: each branch that decides a
breach is exercised at least twice — one input that breaches and one that
does not — so a mutant that flips the comparison dies.
"""
from __future__ import annotations

import json

import pytest

from scripts import slo_probe


def _samples(*statuses: int, latency_ms: float = 50.0) -> list[dict]:
    return [{"status": s, "latency_ms": latency_ms} for s in statuses]


def test_availability_below_target_is_a_breach(monkeypatch):
    monkeypatch.setenv("SLO_AVAILABILITY", "0.999")
    monkeypatch.delenv("SLO_LATENCY_P95_MS", raising=False)
    # 2 of 3 are 200 -> 0.667, below 0.999.
    out = slo_probe.evaluate(_samples(200, 200, 503))
    assert out["verdict"] == "FAIL"
    assert any("availability" in b for b in out["breaches"])


def test_availability_at_target_is_not_a_breach(monkeypatch):
    monkeypatch.setenv("SLO_AVAILABILITY", "0.9")
    monkeypatch.delenv("SLO_LATENCY_P95_MS", raising=False)
    # 9 of 10 are 200 -> 0.9, meets the target.
    out = slo_probe.evaluate(_samples(*([200] * 9 + [503])))
    assert out["verdict"] == "PASS"
    assert out["breaches"] == []


def test_p95_latency_above_target_is_a_breach(monkeypatch):
    monkeypatch.delenv("SLO_AVAILABILITY", raising=False)
    monkeypatch.setenv("SLO_LATENCY_P95_MS", "100")
    out = slo_probe.evaluate([
        {"status": 200, "latency_ms": 90},
        {"status": 200, "latency_ms": 150},  # p95 = 150 > 100
    ])
    assert out["verdict"] == "FAIL"
    assert any("p95 latency" in b for b in out["breaches"])


def test_p95_latency_within_target_is_not_a_breach(monkeypatch):
    monkeypatch.delenv("SLO_AVAILABILITY", raising=False)
    monkeypatch.setenv("SLO_LATENCY_P95_MS", "200")
    out = slo_probe.evaluate([
        {"status": 200, "latency_ms": 90},
        {"status": 200, "latency_ms": 150},
    ])
    assert out["verdict"] == "PASS"
    assert out["breaches"] == []


def test_no_targets_is_report_only_pass(monkeypatch):
    monkeypatch.delenv("SLO_AVAILABILITY", raising=False)
    monkeypatch.delenv("SLO_LATENCY_P95_MS", raising=False)
    out = slo_probe.evaluate(_samples(200, 503))
    assert out["targets_configured"] is False
    assert out["verdict"] == "PASS"


def test_percentiles_are_computed_from_sorted_latencies(monkeypatch):
    monkeypatch.delenv("SLO_AVAILABILITY", raising=False)
    monkeypatch.delenv("SLO_LATENCY_P95_MS", raising=False)
    out = slo_probe.evaluate([
        {"status": 200, "latency_ms": 10},
        {"status": 200, "latency_ms": 20},
        {"status": 200, "latency_ms": 30},
        {"status": 200, "latency_ms": 40},
    ])
    # nearest-rank: rank = round(p/100 * n); p50 -> rank 2 of 4 = 20ms,
    # p95 -> rank 4 of 4 = 40ms.
    assert out["latency_p50_ms"] == 20.0
    assert out["latency_p95_ms"] == 40.0
    assert out["availability"] == 1.0


def test_malformed_target_exits(monkeypatch):
    monkeypatch.setenv("SLO_AVAILABILITY", "not-a-number")
    monkeypatch.delenv("SLO_LATENCY_P95_MS", raising=False)
    with pytest.raises(SystemExit):
        slo_probe.evaluate(_samples(200))


def test_main_writes_verdict_json(monkeypatch, tmp_path):
    monkeypatch.delenv("SLO_AVAILABILITY", raising=False)
    monkeypatch.delenv("SLO_LATENCY_P95_MS", raising=False)
    samples = tmp_path / "samples.json"
    out = tmp_path / "report.json"
    samples.write_text(json.dumps({"samples": _samples(200, 200)}))
    rc = slo_probe.main(["--json", str(samples), "--out", str(out)])
    assert rc == 0
    verdict = json.loads(out.read_text())
    assert verdict["samples"] == 2
    assert verdict["availability"] == 1.0
