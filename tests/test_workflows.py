"""Tests for saved workflows — Roadmap V2 · Epic 7."""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests._dev_key import DEV_KEY

H = {"Authorization": "Bearer " + DEV_KEY}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


STEPS = [{"block": "formula_executor", "params": {}, "label": "Compute"}]


def test_save_list_get_delete_workflow(client):
    r = client.post("/v1/workflows", headers=H,
                     json={"name": "Tender Review", "steps": STEPS})
    assert r.status_code == 201, r.text
    wid = r.json()["id"]
    assert r.json()["name"] == "Tender Review"

    r = client.get("/v1/workflows", headers=H)
    assert wid in [w["id"] for w in r.json()["workflows"]]

    r = client.get(f"/v1/workflows/{wid}", headers=H)
    assert r.status_code == 200
    assert r.json()["steps"] == STEPS

    r = client.delete(f"/v1/workflows/{wid}", headers=H)
    assert r.status_code == 200
    assert client.get(f"/v1/workflows/{wid}", headers=H).status_code == 404


def test_save_rejects_empty_workflow(client):
    assert client.post("/v1/workflows", headers=H,
                       json={"name": "Empty", "steps": []}).status_code == 400
    assert client.post("/v1/workflows", headers=H,
                       json={"name": "", "steps": STEPS}).status_code == 400


def test_workflow_is_project_scoped(client):
    proj = client.post("/v1/projects", json={"name": "WF Project"}, headers=H).json()
    client.post("/v1/workflows", headers=H,
                json={"name": "Scoped", "steps": STEPS, "project_id": proj["id"]})
    r = client.get(f"/v1/workflows?project_id={proj['id']}", headers=H)
    assert len(r.json()["workflows"]) == 1
    assert r.json()["workflows"][0]["name"] == "Scoped"


def test_run_saved_workflow(client):
    wid = client.post("/v1/workflows", headers=H,
                       json={"name": "Runnable", "steps": STEPS}).json()["id"]
    r = client.post(f"/v1/workflows/{wid}/run", headers=H,
                    json={"initial_input": "2 + 2"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["workflow_id"] == wid
    assert "run" in body  # carries the chain execution result


def test_run_missing_workflow_404(client):
    assert client.post("/v1/workflows/nope1234/run", headers=H,
                       json={}).status_code == 404
