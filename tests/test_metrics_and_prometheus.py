"""PR #98 — metrics auth + Prometheus endpoint contract tests.

Two endpoints, two contracts:

- ``/v1/metrics`` returns per-block execution data — admin-only.
  Anonymous callers get 401; admin callers get a JSON snapshot dict.
- ``/metrics`` is Prometheus text exposition — unauth (scrapers
  don't auth) and intentionally exposes a narrower counter set.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.dependencies import require_api_key


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_v1_metrics_requires_auth(client):
    """No auth header -> 401 Unauthorized. Pre-PR-98 this returned 200 +
    leaked per-block execution counts to anonymous callers."""
    resp = client.get("/v1/metrics")
    assert resp.status_code in (401, 403), (
        f"/v1/metrics must reject unauthenticated calls; got {resp.status_code}. "
        "Pre-PR-98 this endpoint was in the unauth allowlist and leaked "
        "per-block execution counts to anonymous callers."
    )


def test_v1_metrics_returns_snapshot_for_admin(client):
    """With admin auth, /v1/metrics returns the block_metrics snapshot dict."""
    app.dependency_overrides[require_api_key] = lambda: {
        "user_id": "test-admin", "role": "admin",
    }
    try:
        resp = client.get("/v1/metrics")
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, dict)  # snapshot shape — keyed by block name


def test_v1_metrics_rejects_non_admin(client):
    """Authenticated non-admin users must not access /v1/metrics."""
    app.dependency_overrides[require_api_key] = lambda: {
        "user_id": "test-user", "role": "user",
    }
    try:
        resp = client.get("/v1/metrics")
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 403


def test_prometheus_endpoint_returns_text_format(client):
    """/metrics returns Prometheus text format with the right content type."""
    # First hit something that goes through the middleware so the counter
    # is registered with at least one labeled increment.
    client.get("/health")
    resp = client.get("/metrics")
    assert resp.status_code == 200, resp.text
    ct = resp.headers.get("content-type", "")
    assert "text/plain" in ct, f"expected Prometheus text format, got {ct!r}"
    body = resp.text
    # The counter should be exposed by the exposition format.
    assert "the_fork_requests_total" in body, (
        f"prometheus exposition is missing the seed counter; got:\n{body[:500]}"
    )
    # Standard Prometheus exposition includes the HELP + TYPE lines.
    assert "# HELP the_fork_requests_total" in body
    assert "# TYPE the_fork_requests_total counter" in body
