"""Unit tests for the SLO probe verdict (scripts/slo_probe.py).

Each decision is parametrized with at least two distinct inputs so the
mutation gate's two-input lint and the CI-M mutants both hold: one case
that breaches and one that does not, so a flipped comparison dies.
"""
from __future__ import annotations

import json

import pytest

from scripts import slo_probe


def _samples(*statuses: int, latency_ms: float = 50.0) -> list[dict]:
    return [{"status": s, "latency_ms": latency_ms} for s in statuses]


@pytest.mark.parametrize(
    "statuses, target, expect_fail",
    [
        # 2 of 3 are 200 -> 0.667, below 0.999.
        ([200, 200, 503], "0.999", True),
        # 9 of 10 are 200 -> 0.9, meets 0.9.
        ([200] * 9 + [503], "0.9", False),
        # all 200 -> 1.0, meets 0.99.
        ([200, 200, 200], "0.99", False),
    ],
)
def test_availability_breach(monkeypatch, statuses, target, expect_fail):
    monkeypatch.setenv("SLO_AVAILABILITY", target)
    monkeypatch.delenv("SLO_LATENCY_P95_MS", raising=False)
    out = slo_probe.evaluate(_samples(*statuses))
    assert out["verdict"] == ("FAIL" if expect_fail else "PASS")
    assert bool(out["breaches"]) is expect_fail


@pytest.mark.parametrize(
    "latencies, target_ms, expect_fail",
    [
        # p95 = 150ms, above a 100ms target.
        ([90.0, 150.0], "100", True),
        # p95 = 150ms, within a 200ms target.
        ([90.0, 150.0], "200", False),
        # p95 = 40ms, within a 500ms target.
        ([10.0, 20.0, 40.0], "500", False),
    ],
)
def test_p95_breach(monkeypatch, latencies, target_ms, expect_fail):
    monkeypatch.delenv("SLO_AVAILABILITY", raising=False)
    monkeypatch.setenv("SLO_LATENCY_P95_MS", target_ms)
    samples = [{"status": 200, "latency_ms": x} for x in latencies]
    out = slo_probe.evaluate(samples)
    assert out["verdict"] == ("FAIL" if expect_fail else "PASS")
    assert bool(out["breaches"]) is expect_fail


@pytest.mark.parametrize(
    "statuses",
    [
        [200, 503],          # mixed
        [200, 200, 200],     # all good
    ],
)
def test_no_targets_is_report_only_pass(monkeypatch, statuses):
    monkeypatch.delenv("SLO_AVAILABILITY", raising=False)
    monkeypatch.delenv("SLO_LATENCY_P95_MS", raising=False)
    out = slo_probe.evaluate(_samples(*statuses))
    assert out["targets_configured"] is False
    assert out["verdict"] == "PASS"


@pytest.mark.parametrize(
    "latencies, expected_p50, expected_p95",
    [
        # nearest-rank: p50 -> rank 2 of 4 = 20; p95 -> rank 4 of 4 = 40.
        ([10.0, 20.0, 30.0, 40.0], 20.0, 40.0),
        # single value: both percentiles are that value.
        ([55.0], 55.0, 55.0),
        # three values: p50 -> rank 2 = 30; p95 -> rank 3 = 90.
        ([10.0, 30.0, 90.0], 30.0, 90.0),
    ],
)
def test_percentiles(monkeypatch, latencies, expected_p50, expected_p95):
    monkeypatch.delenv("SLO_AVAILABILITY", raising=False)
    monkeypatch.delenv("SLO_LATENCY_P95_MS", raising=False)
    samples = [{"status": 200, "latency_ms": x} for x in latencies]
    out = slo_probe.evaluate(samples)
    assert out["latency_p50_ms"] == expected_p50
    assert out["latency_p95_ms"] == expected_p95


def test_malformed_target_exits(monkeypatch):
    monkeypatch.setenv("SLO_AVAILABILITY", "not-a-number")
    monkeypatch.delenv("SLO_LATENCY_P95_MS", raising=False)
    with pytest.raises(SystemExit):
        slo_probe.evaluate(_samples(200))


@pytest.mark.parametrize(
    "statuses",
    [
        [200, 200],
        [200, 503],
    ],
)
def test_main_writes_verdict_json(monkeypatch, tmp_path, statuses):
    monkeypatch.delenv("SLO_AVAILABILITY", raising=False)
    monkeypatch.delenv("SLO_LATENCY_P95_MS", raising=False)
    samples = tmp_path / "samples.json"
    out = tmp_path / "report.json"
    samples.write_text(json.dumps({"samples": _samples(*statuses)}))
    rc = slo_probe.main(["--json", str(samples), "--out", str(out)])
    assert rc == 0
    verdict = json.loads(out.read_text())
    assert verdict["samples"] == len(statuses)
    assert verdict["availability"] == round(sum(1 for s in statuses if s == 200) / len(statuses), 5)


@pytest.mark.parametrize(
    "samples",
    [
        [],                                            # no samples -> NO_SAMPLES
        [{"status": 200, "latency_ms": 50.0}],        # one sample -> not NO_SAMPLES
    ],
)
def test_evaluate_empty_vs_present_samples_verdict(monkeypatch, samples):
    monkeypatch.delenv("SLO_AVAILABILITY", raising=False)
    monkeypatch.delenv("SLO_LATENCY_P95_MS", raising=False)
    out = slo_probe.evaluate(samples)
    assert out["verdict"] == ("NO_SAMPLES" if not samples else "PASS")
    assert out["samples"] == len(samples)


def test_main_zero_samples_writes_no_samples_report_and_exits_nonzero(
    monkeypatch, tmp_path,
):
    monkeypatch.delenv("SLO_AVAILABILITY", raising=False)
    monkeypatch.delenv("SLO_LATENCY_P95_MS", raising=False)
    samples = tmp_path / "samples.json"
    out = tmp_path / "report.json"
    samples.write_text(json.dumps({"samples": []}))
    rc = slo_probe.main(["--json", str(samples), "--out", str(out)])
    assert rc == 1
    verdict = json.loads(out.read_text())
    assert verdict["verdict"] == "NO_SAMPLES"
    assert verdict["samples"] == 0
    assert verdict["breaches"] == []
