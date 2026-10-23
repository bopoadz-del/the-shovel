"""Tests for the project router — Reasoning Engine Plan 6."""

import json

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.routers import project as project_router
from app.core.session_store import InMemorySessionStore
from tests._dev_key import DEV_KEY

_HEADERS = {"Authorization": "Bearer " + DEV_KEY}


@pytest.fixture
def client(monkeypatch):
    # Inject a fresh in-memory store and a scripted mock reasoner so the
    # route is tested without DeepSeek.
    store = InMemorySessionStore()
    monkeypatch.setattr(project_router, "_store", store)

    from app.blocks.project_reasoner import ProjectReasonerBlock

    class _MockReasoner(ProjectReasonerBlock):
        async def _call_llm(self, prompt):
            # first call = plan JSON, second = answer
            if not getattr(self, "_called", False):
                self._called = True
                return json.dumps({"understanding": "u",
                                   "steps": [{"type": "compute_cpm"}]})
            return "Project duration is 10 days."

    monkeypatch.setattr(project_router, "_reasoner_factory",
                        lambda: _MockReasoner())
    return TestClient(app)


def test_ask_creates_session_and_returns_answer(client):
    resp = client.post("/v1/project/ask", json={
        "session_id": "p1",
        "request": "what is the duration?",
        "activities": [
            {"id": "A", "duration": 3, "predecessors": []},
            {"id": "B", "duration": 5, "predecessors": [{"predecessor_id": "A"}]},
            {"id": "C", "duration": 2, "predecessors": [{"predecessor_id": "B"}]},
        ],
    }, headers=_HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Project duration is 10 days."
    assert body["session_id"] == "p1"


@pytest.mark.asyncio
async def test_ask_persists_session_across_calls(client):
    payload = {"session_id": "p2", "request": "go",
               "activities": [{"id": "A", "duration": 3, "predecessors": []}]}
    client.post("/v1/project/ask", json=payload, headers=_HEADERS)
    # second call without activities — the session must still hold them
    resp = client.post("/v1/project/ask",
                        json={"session_id": "p2", "request": "again"},
                        headers=_HEADERS)
    assert resp.status_code == 200
    stored = await project_router._store.get("p2")
    assert stored is not None
    assert stored.data["activities"][0]["id"] == "A"


def test_ask_rejects_empty_request(client):
    resp = client.post("/v1/project/ask",
                        json={"session_id": "p3", "request": "  "},
                        headers=_HEADERS)
    assert resp.status_code == 422 or resp.json().get("status") == "error"


def test_ask_rejects_oversized_activity_list(client):
    # More than 5000 activities must be rejected to bound session memory.
    big = [{"id": f"A{i}", "duration": 1, "predecessors": []}
           for i in range(5001)]
    resp = client.post("/v1/project/ask", json={
        "session_id": "p_big",
        "request": "go",
        "activities": big,
    }, headers=_HEADERS)
    assert resp.status_code == 422


def test_ask_accepts_activity_list_at_the_cap(client):
    # Exactly 5000 activities is within the cap and must be accepted.
    at_cap = [{"id": f"A{i}", "duration": 1, "predecessors": []}
              for i in range(5000)]
    resp = client.post("/v1/project/ask", json={
        "session_id": "p_cap",
        "request": "go",
        "activities": at_cap,
    }, headers=_HEADERS)
    assert resp.status_code == 200


def test_project_route_is_mounted():
    paths = {r.path for r in app.routes}
    assert "/v1/project/ask" in paths


def test_main_initialises_a_shared_store():
    # app/main.py must put a SessionStore on app.state at startup.
    with TestClient(app):              # triggers the lifespan startup
        from app.core.session_store import SessionStore
        assert isinstance(app.state.project_store, SessionStore)
