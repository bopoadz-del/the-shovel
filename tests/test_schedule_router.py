"""Smoke tests for the roadmap-named ``/v1/schedule`` router.

The router is a thin shim over existing schedule capabilities; these tests only
prove the endpoints are wired and return success envelopes.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests._dev_key import DEV_KEY
from tests import _wbs_fakes

H = {"Authorization": "Bearer " + DEV_KEY}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_schedule_cpm_endpoint(client):
    payload = {
        "activities": [
            {"id": "A", "duration": 3, "predecessors": []},
            {"id": "B", "duration": 5, "predecessors": [{"predecessor_id": "A"}]},
        ]
    }
    r = client.post("/v1/schedule/cpm", json=payload, headers=H)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["status"] == "success"
    assert data["project_duration"] == 8


def test_schedule_manpower_endpoint(client):
    payload = {
        "activities": [
            {
                "id": "A",
                "duration": 5,
                "predecessors": [],
                "resources": [{"trade": "labor", "count": 4}],
            }
        ]
    }
    r = client.post("/v1/schedule/manpower", json=payload, headers=H)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["status"] == "success"
    assert data["by_trade_totals"]["labor"] > 0


def test_schedule_fasttrack_endpoint(client):
    payload = {
        "activities": [
            {"id": "A", "duration": 10, "predecessors": []},
            {"id": "B", "duration": 8, "predecessors": [{"predecessor_id": "A"}]},
        ],
        "reductions": {"A": 3},
        "total_delay_days": 10,
    }
    r = client.post("/v1/schedule/fasttrack", json=payload, headers=H)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["status"] == "success"
    assert data["days_saved"] >= 0
    assert data["scenarios"]


def test_schedule_generate_endpoint(client, monkeypatch):
    _wbs_fakes.install(monkeypatch, min_activities=30)
    payload = {"brief": "small office building", "target_count": 30, "project_type": "building"}
    r = client.post("/v1/schedule/generate", json=payload, headers=H)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["status"] == "success"
    assert data["actual_count"] >= 30
