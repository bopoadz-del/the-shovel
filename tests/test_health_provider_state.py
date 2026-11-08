"""M9: serving is a state, not a window (ORDERS_07NOV_0815Z).

During the 19 Oct outage every provider's latest outcome was exhausted, but the
failures aged past the 300 s annotation window and the box "recovered" to
serving without a single real ok. These pin the state rule: a fresh box or a
latest-ok outcome serves; age never resurrects a dead state; only a real
record_provider_result('ok') does.
"""
from __future__ import annotations

import pytest

from app.core import health_probes as hp


class _Clock:
    def __init__(self, start=1_000_000.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    with hp._PROVIDER_LOCK:
        hp._PROVIDER_LAST.clear()
        hp._PROVIDER_COUNTS.update({"billing_402": 0, "rate_429": 0})
    hp._LAST_SERVED_AT = 0.0
    fake = _Clock()
    monkeypatch.setattr(hp.time, "time", fake)
    yield fake


def test_fresh_box_is_serving():
    assert hp.provider_snapshot()["serving"] is True
    assert hp.provider_snapshot()["providers"] == {}


def test_exhausted_providers_stay_not_serving_after_two_hours(clock):
    hp.record_provider_result("primary", "exhausted", 402)
    hp.record_provider_result("fallback", "exhausted", 429)
    assert hp.provider_snapshot()["serving"] is False
    clock.advance(2 * 3600)
    snap = hp.provider_snapshot()
    # Both latest outcomes are still exhausted; two quiet hours change nothing.
    assert snap["serving"] is False


def test_a_real_ok_restores_serving_immediately(clock):
    hp.record_provider_result("primary", "exhausted", 402)
    hp.record_provider_result("fallback", "exhausted", 429)
    clock.advance(2 * 3600)
    hp.record_provider_result("primary", "ok")
    assert hp.provider_snapshot()["serving"] is True


def test_one_fallback_error_keeps_serving_when_primary_latest_ok(clock):
    hp.record_provider_result("primary", "ok")
    hp.record_provider_result("fallback", "error")
    assert hp.provider_snapshot()["serving"] is True


def test_ready_and_health_reflect_not_serving(clock):
    """(b) integration: an all-exhausted LLM is not-ready on /ready (503) and
    degraded on /health (still 200 for liveness)."""
    from fastapi.testclient import TestClient

    from app.main import app

    hp.record_provider_result("primary", "exhausted", 402)
    hp.record_provider_result("fallback", "exhausted", 429)

    with TestClient(app) as client:
        health_resp = client.get("/health")
        ready_resp = client.get("/ready")

    assert health_resp.status_code == 200  # liveness never trips
    health_body = health_resp.json()
    assert health_body["checks"]["llm"]["serving"] is False
    assert health_body["status"] == "degraded"

    assert ready_resp.status_code == 503  # readiness fails closed
    assert ready_resp.json()["ready"] is False


def test_reset_provider_state_clears_recorded_outcomes(clock):
    """reset_provider_state() must actually clear state: after two exhausted
    records, a reset returns the box to the fresh (serving, empty) state."""
    hp.record_provider_result("primary", "exhausted", 402)
    hp.record_provider_result("fallback", "exhausted", 429)
    assert hp.provider_snapshot()["serving"] is False

    hp.reset_provider_state()
    snap = hp.provider_snapshot()
    assert snap["serving"] is True
    assert snap["providers"] == {}
    assert snap["counts"]["billing_402"] == 0
    assert snap["counts"]["rate_429"] == 0
