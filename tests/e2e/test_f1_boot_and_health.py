"""F1 — the app boots and answers its health endpoints.

The first thing a stranger following the README does, and the first thing
Render does on every deploy. If this fails nothing else matters.

Deliberately asserts on the CONTENT of the health payload, not just the 200.
A health endpoint that returns 200 while the app is broken is worse than one
that fails: it is what makes a bad deploy look green.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_the_app_imports_and_mounts_its_routes():
    """A boot failure usually shows up here first, as an import error."""
    assert len(app.routes) > 50, f"only {len(app.routes)} routes — router wiring is broken"


@pytest.mark.parametrize("path", ["/health", "/v1/health"])
def test_health_returns_200_without_credentials(client, path):
    """Health must be reachable unauthenticated or the platform's own probe
    cannot use it. This is the ONE place an open endpoint is correct."""
    response = client.get(path)
    assert response.status_code == 200, f"{path} -> {response.status_code}"


def test_health_reports_an_actual_status(client):
    """Not just a 200 — a payload a deploy check can read."""
    payload = client.get("/health").json()
    assert isinstance(payload, dict), f"health returned {type(payload)}"
    assert payload, "health returned an empty body"
    status_ish = {k.lower(): v for k, v in payload.items()}
    assert any(k in status_ish for k in ("status", "ok", "healthy", "state")), (
        f"health payload carries no status field: {sorted(payload)}"
    )


def test_the_root_path_is_reachable(client):
    """The README tells a stranger to open the base URL."""
    assert client.get("/").status_code == 200


def test_openapi_is_served(client):
    """A stranger integrating against this needs the schema, and a broken
    route signature surfaces here as a 500 rather than at request time."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert schema.get("paths"), "OpenAPI schema has no paths"
