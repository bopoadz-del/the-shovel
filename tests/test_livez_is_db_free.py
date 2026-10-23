"""`/livez` must never touch the database.

The database moved to Neon (2026-08-09), which bills by compute-time and
suspends when idle. Render polls `healthCheckPath` continuously, so that
endpoint must not run a query — otherwise every probe wakes Neon's compute
and holds it awake around the clock, defeating scale-to-zero and re-creating
the always-on bill that motivated the move.

`render.yaml` points `healthCheckPath` at `/livez`. These tests pin both
halves of that contract: the endpoint does no I/O, and the blueprint still
points at it. A future "let's make livez more useful" change fails here
rather than showing up as a surprise invoice.
"""
from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_livez_returns_200_without_probing_the_database(client):
    # Any DB probe would go through probe_database; make it explode so a
    # call cannot pass silently.
    def _boom():  # pragma: no cover - only runs if the contract breaks
        raise AssertionError("/livez probed the database — it must be DB-free")

    with patch("app.routers.health.probe_database", side_effect=_boom) as probe:
        resp = client.get("/livez")
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "alive"
    assert probe.call_count == 0


def test_livez_stays_up_when_the_database_is_unreachable(client):
    """The point of a liveness probe: a dead database must not restart the app."""
    with patch("app.routers.health.probe_database",
               return_value={"ok": False, "error": "connection refused"}):
        assert client.get("/livez").status_code == 200
        # /ready is the fail-closed counterpart and MUST go 503 on the same state.
        assert client.get("/ready").status_code == 503


def test_render_blueprint_health_check_points_at_the_db_free_endpoint():
    blueprint = (REPO / "render.yaml").read_text(encoding="utf-8")
    paths = re.findall(r"^\s*healthCheckPath:\s*(\S+)", blueprint, re.MULTILINE)
    assert paths, "render.yaml declares no healthCheckPath"
    for path in paths:
        assert path == "/livez", (
            f"healthCheckPath is {path!r}; Render polls it continuously and "
            "/health and /ready both run a real SELECT 1, which would keep "
            "Neon's compute awake permanently. Use /livez."
        )
