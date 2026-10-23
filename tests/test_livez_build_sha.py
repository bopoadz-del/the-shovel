"""GET /livez reports the same deployed build_sha as /health, with no DB call.

The image already injects RENDER_GIT_COMMIT, GIT_SHA, or SOURCE_VERSION.
/health reads those through _deployed_build_sha. /livez must reuse that
helper. A liveness probe that opens a session would wake a scale-to-zero
database on every poll.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app

_SYNTHETIC_SHA = "synthetic-sha-not-a-commit"


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _pin_sha(monkeypatch):
    """One synthetic value, and no higher-priority key hiding it."""
    monkeypatch.delenv("RENDER_GIT_COMMIT", raising=False)
    monkeypatch.delenv("SOURCE_VERSION", raising=False)
    monkeypatch.setenv("GIT_SHA", _SYNTHETIC_SHA)


def test_livez_build_sha_matches_health_for_the_same_env(client, monkeypatch):
    _pin_sha(monkeypatch)
    live = client.get("/livez")
    health = client.get("/health")
    assert live.status_code == 200
    assert health.status_code == 200
    live_body = live.json()
    assert live_body["status"] == "alive"
    assert "timestamp" in live_body
    assert live_body["build_sha"] == health.json()["build_sha"] == _SYNTHETIC_SHA


def test_livez_returns_build_sha_when_database_probes_raise(client, monkeypatch):
    """Session, probes, and engine connect all raise. /livez still answers."""
    _pin_sha(monkeypatch)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("database touched from /livez")

    with (
        patch("app.routers.health.probe_database", side_effect=_boom),
        patch("app.routers.health.probe_corpus_chunks", side_effect=_boom),
        patch("app.core.db.SessionLocal", side_effect=_boom),
        patch("app.core.db.get_engine", side_effect=_boom),
    ):
        live = client.get("/livez")
    assert live.status_code == 200, live.text
    body = live.json()
    assert body["status"] == "alive"
    assert "timestamp" in body
    assert body["build_sha"] == _SYNTHETIC_SHA
