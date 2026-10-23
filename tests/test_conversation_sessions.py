"""Multi-session chat history (Quarry standalone parity, follow-up to PR #101).

Two contracts:
1. ``append_message`` stamps a NULL conversation title from the FIRST user
   message, so session lists read like the design ("pipe specs...") instead
   of raw ids. Later messages never overwrite the title.
2. ``GET /v1/projects/{id}/conversations`` lists the project's sessions,
   newest-first, for the sidebar CHAT HISTORY section.
"""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient


def _reload(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    import app.core.agent_memory as _mod
    importlib.reload(_mod)
    return _mod


# ── title stamping ───────────────────────────────────────────────────────────

def test_first_user_message_stamps_title(tmp_path, monkeypatch):
    am = _reload(tmp_path, monkeypatch)
    am.get_or_create_conversation("c1", "scout", project_id="p1")
    am.append_message("c1", "user", "What type of storm water pipes are specified for Infra P1?")
    am.append_message("c1", "assistant", "uPVC Class C per the BOQ.")
    am.append_message("c1", "user", "and the manhole spacing?")
    conv = am.get_conversation("c1")
    assert conv["title"] == "What type of storm water pipes are specified for Infra P1?"


def test_title_truncated_to_80_chars(tmp_path, monkeypatch):
    am = _reload(tmp_path, monkeypatch)
    am.get_or_create_conversation("c2", "scout", project_id="p1")
    am.append_message("c2", "user", "x" * 300)
    assert len(am.get_conversation("c2")["title"]) == 80


def test_assistant_message_never_stamps_title(tmp_path, monkeypatch):
    am = _reload(tmp_path, monkeypatch)
    am.get_or_create_conversation("c3", "scout", project_id="p1")
    am.append_message("c3", "assistant", "Hello, how can I help?")
    assert am.get_conversation("c3")["title"] is None


# ── sessions endpoint ────────────────────────────────────────────────────────

@pytest.fixture
def client_and_memory(tmp_path, monkeypatch):
    am = _reload(tmp_path, monkeypatch)
    from app.main import app
    from app.dependencies import require_user

    app.dependency_overrides[require_user] = lambda: {
        "user_id": "u1", "role": "user",
    }
    from app.routers import projects as projects_router
    monkeypatch.setattr(
        projects_router.store, "get_project",
        lambda pid, **_k: {"id": pid, "name": "P", "user_id": "u1"},
    )
    monkeypatch.setattr(
        projects_router.store, "can_access_project", lambda *_a, **_k: True,
    )
    monkeypatch.setattr(
        projects_router.store, "storage_project_id", lambda pid: pid,
    )
    monkeypatch.setattr(
        projects_router.store, "project_owner", lambda _pid: "u1",
    )
    monkeypatch.setattr(
        projects_router.store, "_master_corpus_source", lambda _pid: None
    )
    with TestClient(app) as client:
        yield client, am
    app.dependency_overrides.clear()


def test_list_project_conversations_newest_first(client_and_memory):
    client, am = client_and_memory
    am.get_or_create_conversation("ws-p1", "project-assistant", project_id="p1")
    am.append_message("ws-p1", "user", "BOQ review for the demolition package")
    am.get_or_create_conversation("ws-p1-2", "project-assistant", project_id="p1")
    am.append_message("ws-p1-2", "user", "pipe specs")
    am.get_or_create_conversation("ws-other", "project-assistant", project_id="other")

    res = client.get("/v1/projects/p1/conversations")
    assert res.status_code == 200
    sessions = res.json()["conversations"]
    ids = [s["id"] for s in sessions]
    assert ids == ["ws-p1-2", "ws-p1"]  # newest first
    assert sessions[0]["title"] == "pipe specs"
    assert "updated_at" in sessions[0]
    assert "ws-other" not in ids
