"""Tests for admin diagnostic endpoints.

Task 14 — dead-letter queue: scans Redis ``arq:result:*`` keys and returns
failed arq jobs. Tests mock the shared async Redis client so they run
without a real Redis server.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from app.core import redis_client
from app.dependencies import require_api_key
from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def admin_auth():
    app.dependency_overrides[require_api_key] = lambda: {
        "user_id": "test-admin",
        "role": "admin",
    }
    yield
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def _reset_redis():
    redis_client.reset_for_tests()
    yield
    redis_client.reset_for_tests()


def test_dead_letter_redis_unavailable(client, admin_auth):
    """No Redis client → empty list with a graceful error flag."""
    resp = client.get("/v1/admin/dead-letter")
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"dead_letter": [], "error": "Redis not available"}


def test_dead_letter_returns_only_failed_jobs(client, admin_auth):
    """Successful arq results are skipped; only failed jobs are returned."""
    mock_client = AsyncMock()
    mock_client.keys = AsyncMock(
        return_value=["arq:result:job-failed", "arq:result:job-ok"]
    )
    mock_client.get = AsyncMock(
        side_effect=[
            json.dumps({"f": "bad_func", "a": ["arg1"], "e": "boom", "t": 12345}),
            json.dumps({"f": "good_func", "a": [], "t": 12346}),
        ]
    )
    redis_client._async_client = mock_client

    resp = client.get("/v1/admin/dead-letter")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["dead_letter"] == [
        {
            "job_id": "job-failed",
            "function": "bad_func",
            "args": ["arg1"],
            "error": "boom",
            "timestamp": 12345,
        }
    ]
    assert "error" not in body


def test_dead_letter_skips_json_decode_error(client, admin_auth):
    """A malformed result value is skipped; other failed jobs still return."""
    mock_client = AsyncMock()
    mock_client.keys = AsyncMock(
        return_value=["arq:result:bad-json", "arq:result:job-failed"]
    )
    mock_client.get = AsyncMock(
        side_effect=[
            "not valid json",
            json.dumps({"f": "bad_func", "a": ["arg1"], "e": "boom", "t": 12345}),
        ]
    )
    redis_client._async_client = mock_client

    resp = client.get("/v1/admin/dead-letter")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["dead_letter"] == [
        {
            "job_id": "job-failed",
            "function": "bad_func",
            "args": ["arg1"],
            "error": "boom",
            "timestamp": 12345,
        }
    ]


def test_dead_letter_redis_scan_error(client, admin_auth):
    """Transient Redis failure during scan returns graceful empty response."""
    mock_client = AsyncMock()
    mock_client.keys = AsyncMock(side_effect=ConnectionError("Connection reset"))
    redis_client._async_client = mock_client

    resp = client.get("/v1/admin/dead-letter")
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"dead_letter": [], "error": "Redis scan failed"}


def test_dead_letter_requires_admin(client):
    """Non-admin callers are rejected by the existing admin gate."""
    app.dependency_overrides[require_api_key] = lambda: {
        "user_id": "test-user",
        "role": "user",
    }
    try:
        resp = client.get("/v1/admin/dead-letter")
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 403
