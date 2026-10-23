"""Regression: endpoints that were previously unauthenticated now require
credentials (the memory cache and monitoring routers)."""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests._dev_key import DEV_KEY

H = {"Authorization": "Bearer " + DEV_KEY}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_unauthenticated_requests_are_rejected(client):
    """No credentials → 401/403 on the memory and monitoring endpoints."""
    for path in ("/v1/memory/stats", "/v1/leaderboard", "/v1/recommend", "/v1/predict"):
        assert client.get(path).status_code in (401, 403), path
    assert client.post("/v1/memory/get", json={"key": "x"}).status_code in (401, 403)
    assert client.post("/v1/metrics/record", json={}).status_code in (401, 403)


def test_memory_flush_and_keys_require_admin(client):
    """Destructive/enumerating memory actions are admin-only. the dev key
    authenticates but carries no admin role, so flush/keys are forbidden while
    ordinary actions still pass the gate."""
    assert client.post("/v1/memory/flush", json={}, headers=H).status_code == 403
    assert client.post("/v1/memory/keys", json={}, headers=H).status_code == 403
    assert client.post(
        "/v1/memory/get", json={"key": "x"}, headers=H
    ).status_code not in (401, 403)


def test_authenticated_requests_pass_the_auth_gate(client):
    """A valid key passes the gate — the response is not an auth rejection."""
    for path in ("/v1/memory/stats", "/v1/leaderboard"):
        assert client.get(path, headers=H).status_code not in (401, 403), path
