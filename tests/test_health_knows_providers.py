"""PR-B: /health reflects real provider outcomes, not just key presence.

During the 19 Oct outage keys were present (primary_ready/fallback_ready true)
while every call returned 402/429, so /health stayed green. These tests pin the
recorder + snapshot that let health tell "serving" from "configured".
"""
from __future__ import annotations

import time

import pytest

from app.core import health_probes as hp


@pytest.fixture(autouse=True)
def _reset_provider_state():
    with hp._PROVIDER_LOCK:
        hp._PROVIDER_LAST.clear()
        hp._PROVIDER_COUNTS.update({"billing_402": 0, "rate_429": 0})
    hp._LAST_SERVED_AT = 0.0
    yield


def test_no_data_is_serving():
    """A quiet box with no recorded calls must not read as down."""
    snap = hp.provider_snapshot()
    assert snap["serving"] is True
    assert snap["providers"] == {}


def test_success_marks_serving_and_records_provider():
    hp.record_provider_result("primary", "ok")
    snap = hp.provider_snapshot()
    assert snap["serving"] is True
    assert snap["providers"]["primary"]["status"] == "ok"
    assert snap["last_served_age_s"] is not None


def test_all_providers_exhausted_is_not_serving():
    """Primary 402 + fallback 429, no success -> not serving (the outage)."""
    hp.record_provider_result("primary", "exhausted", 402)
    hp.record_provider_result("fallback", "exhausted", 429)
    snap = hp.provider_snapshot()
    assert snap["serving"] is False
    assert snap["counts"]["billing_402"] == 1
    assert snap["counts"]["rate_429"] == 1
    assert snap["providers"]["primary"]["http"] == 402


def test_primary_down_but_fallback_served_is_serving():
    """Primary 402 then fallback ok -> serving true (fallback covered it)."""
    hp.record_provider_result("primary", "exhausted", 402)
    hp.record_provider_result("fallback", "ok")
    snap = hp.provider_snapshot()
    assert snap["serving"] is True
    assert snap["counts"]["billing_402"] == 1


def test_recent_failure_after_stale_success_is_not_serving():
    """A success older than the window no longer counts; a recent failure wins."""
    hp.record_provider_result("primary", "ok")
    # Age the success past the staleness window.
    with hp._PROVIDER_LOCK:
        hp._PROVIDER_LAST["primary"]["at"] -= hp._PROVIDER_STALE_S + 10
    hp._LAST_SERVED_AT -= hp._PROVIDER_STALE_S + 10
    hp.record_provider_result("fallback", "exhausted", 429)
    snap = hp.provider_snapshot()
    assert snap["serving"] is False


def test_probe_llm_includes_providers_and_serving():
    hp.record_provider_result("primary", "ok")
    out = hp.probe_llm()
    assert "serving" in out and "providers" in out
    assert out["serving"] is True


def test_health_endpoint_degrades_when_not_serving(monkeypatch):
    """Integration seam (the 19 Oct outage): /health must flip status to
    'degraded' when the model cannot be served, while STILL returning 200 so
    the liveness probe does not restart a live service mid-incident.

    Failing-first: on main (before PR-B wired llm.serving into the status) a
    not-serving LLM left status 'healthy'.
    """
    from fastapi.testclient import TestClient

    from app.main import app

    # Force not-serving: both roles exhausted, no recent success.
    hp.record_provider_result("primary", "exhausted", 402)
    hp.record_provider_result("fallback", "exhausted", 429)

    with TestClient(app) as c:
        resp = c.get("/health")

    assert resp.status_code == 200  # liveness never trips on degraded
    body = resp.json()
    # Pin that it is the LLM, not the DB, driving the degrade.
    assert body["checks"]["database"]["ok"] is True
    assert body["checks"]["llm"]["serving"] is False
    assert body["status"] == "degraded"


def test_health_endpoint_healthy_when_serving(monkeypatch):
    """Control: a served-recently LLM + ok DB reads 'healthy'."""
    from fastapi.testclient import TestClient

    from app.main import app

    hp.record_provider_result("primary", "ok")

    with TestClient(app) as c:
        resp = c.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["checks"]["llm"]["serving"] is True
    if body["checks"]["database"]["ok"]:
        assert body["status"] == "healthy"
