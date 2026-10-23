"""Health endpoints report EVALUATED capability, not hardcoded/seeded state
(audit §8.5/§8.6).

Before: /health returned a hardcoded ``status:"healthy"`` with no DB/embedder
probe, and /v1/system/health seeded every provider at 100% reliability so a
never-called (or down) provider read "excellent". These tests pin the truthful
behaviour: real DB probe, honest degraded status, a fail-closed /ready gate,
and "unknown" for unproven providers.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core import health_probes


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


# ── /health — evaluated, always 200 (liveness), truthful status ──────────────

def test_health_reports_evaluated_checks(client):
    body = client.get("/health").json()
    assert "checks" in body
    assert "database" in body["checks"] and "embedder" in body["checks"]
    assert set(body["checks"]["database"]) >= {"ok", "latency_ms", "error"}


def test_health_healthy_when_db_ok(client):
    body = client.get("/health").json()
    assert body["checks"]["database"]["ok"] is True
    assert body["status"] == "healthy"


def test_health_degraded_but_200_when_db_down(client, monkeypatch):
    # A DB blip must surface as degraded, NOT a non-200 that restarts the live
    # service (Render's healthCheckPath is /health = liveness).
    monkeypatch.setattr("app.routers.health.probe_database",
                        lambda *a, **k: {"ok": False, "latency_ms": 1, "error": "down"})
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "degraded"
    assert r.json()["checks"]["database"]["ok"] is False


# ── /ready — fail-closed readiness gate ──────────────────────────────────────

def test_ready_200_when_db_ok(client):
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["ready"] is True


def test_ready_503_when_db_down(client, monkeypatch):
    monkeypatch.setattr("app.routers.health.probe_database",
                        lambda *a, **k: {"ok": False, "latency_ms": 1, "error": "down"})
    r = client.get("/ready")
    assert r.status_code == 503
    assert r.json()["ready"] is False


# ── probes are non-raising ───────────────────────────────────────────────────

def test_probe_database_never_raises(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("connection refused")
    monkeypatch.setattr("app.core.db.SessionLocal", boom)
    out = health_probes.probe_database()
    assert out["ok"] is False and "connection refused" in out["error"]


def test_probe_embedder_is_nonloading(client):
    # Reports loaded state; never triggers a model load.
    out = health_probes.probe_embedder()
    assert set(out) >= {"loaded", "identity"}
    assert isinstance(out["loaded"], bool)


# ── §8.6 — providers with no observed calls report "unknown", not excellent ──

def test_system_health_unproven_providers_are_unknown(client):
    d = client.get("/v1/system/health").json()
    lb = d["leaderboard"]["leaderboard"] if isinstance(d["leaderboard"], dict) else d["leaderboard"]
    # A fresh monitoring block has zero observed calls for every provider.
    for entry in lb:
        if entry["total_calls"] == 0:
            assert entry["status"] == "unknown", entry
            assert entry["reliability_score"] is None
            assert entry["recommendation"] == "unproven"


def test_system_health_overall_not_falsely_healthy_without_calls(client, monkeypatch):
    # The monitoring block is a process-wide singleton: a call recorded by an
    # earlier test in this process (order varies under xdist) would make this
    # read "healthy". Start from a monitor with no recorded calls -- the
    # condition this test is about.
    from collections import defaultdict, deque

    from app.dependencies import MONITORING_AVAILABLE, get_monitoring_block
    if MONITORING_AVAILABLE:
        block = get_monitoring_block()
        for attr in ("latency_history", "error_history", "uptime_history"):
            monkeypatch.setattr(block, attr, defaultdict(lambda: deque(maxlen=block.metrics_window)))
        monkeypatch.setattr(block, "leaderboard_cache", None)
        monkeypatch.setattr(block, "last_leaderboard_update", 0)
    d = client.get("/v1/system/health").json()
    # With no real calls recorded, overall must not claim healthy.
    assert d["overall_status"] != "healthy"
