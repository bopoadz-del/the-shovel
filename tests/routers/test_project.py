"""Focused async session-store tests for /v1/project/ask.

Avoids the full app lifespan by mounting only the project router and mocking
its external dependencies.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.dependencies import require_user
from app.routers import project as project_router
from app.core.session_store import InMemorySessionStore
from app.schemas.project_session import ProjectSession
from tests._fixture_values import STUB_KEY_07

_HEADERS = {"Authorization": "Bearer " + STUB_KEY_07}


class _MockReasoner:
    """Lightweight stand-in for ProjectReasonerBlock."""

    async def process(self, inputs):
        return {
            "status": "success",
            "answer": "mock answer",
            "understanding": "u",
            "plan": None,
            "execution": None,
            "sources": [],
        }


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)

    app = FastAPI()
    app.include_router(project_router.router)
    app.dependency_overrides[require_user] = lambda: {
        "user_id": "test-user", "role": "user", "email": "test@local"
    }

    store = InMemorySessionStore()
    monkeypatch.setattr(project_router, "_store", store)
    monkeypatch.setattr(project_router, "_reasoner_factory", lambda: _MockReasoner())

    # Tenant gate + alias resolution are not under test here.
    monkeypatch.setattr(
        "app.core.projects.get_project",
        lambda project_id, user_id=None, include_admin_approved=False: {"id": project_id},
    )
    monkeypatch.setattr("app.core.projects._master_corpus_source", lambda project_id: None)

    return TestClient(app)


def test_ask_creates_session_and_returns_answer(client):
    resp = client.post("/v1/project/ask", json={
        "session_id": "p1",
        "request": "what is the duration?",
        "activities": [
            {"id": "A", "duration": 3, "predecessors": []},
        ],
    }, headers=_HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "mock answer"
    assert body["session_id"] == "p1"


@pytest.mark.asyncio
async def test_ask_persists_session_across_calls(client):
    payload = {
        "session_id": "p2",
        "request": "go",
        "activities": [{"id": "A", "duration": 3, "predecessors": []}],
    }
    client.post("/v1/project/ask", json=payload, headers=_HEADERS)
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
    big = [{"id": f"A{i}", "duration": 1, "predecessors": []}
           for i in range(5001)]
    resp = client.post("/v1/project/ask", json={
        "session_id": "p_big",
        "request": "go",
        "activities": big,
    }, headers=_HEADERS)
    assert resp.status_code == 422


def test_ask_accepts_activity_list_at_the_cap(client):
    at_cap = [{"id": f"A{i}", "duration": 1, "predecessors": []}
              for i in range(5000)]
    resp = client.post("/v1/project/ask", json={
        "session_id": "p_cap",
        "request": "go",
        "activities": at_cap,
    }, headers=_HEADERS)
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_ask_rejects_session_owned_by_another_user(client, monkeypatch):
    await project_router._store.save(
        ProjectSession.new("owned-session", user_id="other-user")
    )
    resp = client.post("/v1/project/ask",
                       json={"session_id": "owned-session", "request": "hi"},
                       headers=_HEADERS)
    assert resp.status_code == 404
