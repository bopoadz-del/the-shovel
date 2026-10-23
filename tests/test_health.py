"""Health endpoint tests."""

import os
import pytest
from fastapi.testclient import TestClient

from app.core import redis_client
from tests._dev_key import DEV_KEY


@pytest.fixture(autouse=True)
def _reset_redis(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    redis_client.reset_for_tests()
    yield
    redis_client.reset_for_tests()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    from app.main import app
    with TestClient(app, headers={"Authorization": "Bearer " + DEV_KEY}) as client:
        yield client


def test_health_sync_legacy(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "redis" not in data


def test_health_payload_includes_build_sha(client: TestClient):
    """Deploy verification reads build_sha; the key must always be present."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "build_sha" in data
    assert data["build_sha"] is None or isinstance(data["build_sha"], str)


def test_health_build_sha_from_render_git_commit(client: TestClient, monkeypatch):
    monkeypatch.delenv("GIT_SHA", raising=False)
    monkeypatch.delenv("SOURCE_VERSION", raising=False)
    monkeypatch.setenv("RENDER_GIT_COMMIT", "abc123def456")
    data = client.get("/health").json()
    assert data["build_sha"] == "abc123def456"


def test_health_build_sha_null_when_unset(client: TestClient, monkeypatch):
    monkeypatch.delenv("RENDER_GIT_COMMIT", raising=False)
    monkeypatch.delenv("GIT_SHA", raising=False)
    monkeypatch.delenv("SOURCE_VERSION", raising=False)
    data = client.get("/health").json()
    assert "build_sha" in data
    assert data["build_sha"] is None


def test_health_build_sha_falls_back_to_git_sha(client: TestClient, monkeypatch):
    monkeypatch.delenv("RENDER_GIT_COMMIT", raising=False)
    monkeypatch.delenv("SOURCE_VERSION", raising=False)
    monkeypatch.setenv("GIT_SHA", "fedcba987654")
    data = client.get("/health").json()
    assert data["build_sha"] == "fedcba987654"


def test_health_v1_async_includes_redis(client: TestClient):
    response = client.get("/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "observability" in data
    assert "block_metrics" in data
    assert "redis" in data
    assert data["redis"] == {"connected": False, "latency_ms": None}


def test_v1_system_health_fallback_when_monitoring_unavailable(client: TestClient, monkeypatch):
    """Regression test: full_health() must await health_v1() when monitoring is off."""
    monkeypatch.setattr("app.routers.health.MONITORING_AVAILABLE", False)
    response = client.get("/v1/system/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "observability" in data
    assert "block_metrics" in data
    assert "redis" in data
