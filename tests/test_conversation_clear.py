"""Tests for the conversation-clear endpoint.

POST /v1/projects/{project_id}/conversations/{conversation_id}/clear
wipes messages + agent_facts for the conversation but preserves the
conversation row so the React composer doesn't need to remount and the
agent_name / project_id metadata stays stable.

Operator-actionable escape hatch for a thread poisoned by prior
hallucinated assistant turns (the "fake CPM table" failure mode).
"""
from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core import agent_memory
from tests._dev_key import DEV_KEY


H = {"Authorization": "Bearer " + DEV_KEY}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _new_project(client, name="Clear Test"):
    r = client.post("/v1/projects", json={"name": name, "client": "ACME"}, headers=H)
    assert r.status_code == 201, r.text
    return r.json()


def _seed_conversation(conversation_id: str, project_id: str, agent_name: str = "project-assistant") -> None:
    agent_memory.get_or_create_conversation(conversation_id, agent_name, project_id)
    agent_memory.append_message(conversation_id, "user", "Generate a 250-activity schedule")
    agent_memory.append_message(conversation_id, "assistant", "Here is the fake table | A | B |")
    agent_memory.append_message(conversation_id, "user", "Now manpower histogram")
    agent_memory.append_message(conversation_id, "assistant", "[hallucinated histogram]")


# ── happy path ──────────────────────────────────────────────────────────────


def test_clear_workspace_conversation_wipes_messages(client):
    proj = _new_project(client, "Clear Happy Path")
    pid = proj["id"]
    cid = f"ws-{pid}"
    _seed_conversation(cid, pid)
    assert len(agent_memory.get_messages(cid)) == 4

    r = client.post(
        f"/v1/projects/{pid}/conversations/{cid}/clear", headers=H,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "cleared"
    assert body["conversation_id"] == cid
    assert body["messages"] == 4
    # No prior facts seeded but the field must still be present + numeric.
    assert isinstance(body["facts"], int)

    # Messages gone, conversation row preserved (so the agent runtime
    # finds it on the next chat turn without recreating).
    assert agent_memory.get_messages(cid) == []
    conv = agent_memory.get_conversation(cid)
    assert conv is not None, "conversation row must be preserved"
    assert conv["agent_name"] == "project-assistant"


# ── idempotency ─────────────────────────────────────────────────────────────


def test_clear_is_idempotent(client):
    """A second clear on an already-empty conversation must return 200
    with zero counts. Frontend can retry without surprise behaviour."""
    proj = _new_project(client, "Idempotent Clear")
    pid = proj["id"]
    cid = f"ws-{pid}"
    _seed_conversation(cid, pid)
    client.post(f"/v1/projects/{pid}/conversations/{cid}/clear", headers=H)

    r = client.post(f"/v1/projects/{pid}/conversations/{cid}/clear", headers=H)
    assert r.status_code == 200
    body = r.json()
    assert body["messages"] == 0
    assert body["facts"] == 0


def test_clear_nonexistent_conversation_returns_zero(client):
    """The cleanest UX: clearing a conversation that doesn't exist
    yet (e.g. a new project with no chat turns) returns 200 / zeros
    instead of 404. The conversation_id is still derived from the
    project, so no data leak."""
    proj = _new_project(client, "Empty Project Clear")
    pid = proj["id"]
    cid = f"ws-{pid}"

    r = client.post(f"/v1/projects/{pid}/conversations/{cid}/clear", headers=H)
    assert r.status_code == 200
    assert r.json()["messages"] == 0


# ── isolation / security ────────────────────────────────────────────────────


def test_clear_rejects_workspace_cid_for_different_project(client):
    """ws-{other_pid} on a project that doesn't own it: 404. Prevents
    one project's chat from being cleared by clicking elsewhere."""
    a = _new_project(client, "Owner A")
    b = _new_project(client, "Owner B")
    foreign_cid = f"ws-{b['id']}"

    r = client.post(f"/v1/projects/{a['id']}/conversations/{foreign_cid}/clear", headers=H)
    assert r.status_code == 404


def test_clear_rejects_nonworkspace_cid_belonging_to_other_project(client):
    """A non-workspace conversation_id (any string not starting with
    ws-) that was stored against a DIFFERENT project must also be
    rejected as 404."""
    a = _new_project(client, "X Holder")
    b = _new_project(client, "X Foreigner")
    foreign_cid = f"free-form-{uuid.uuid4().hex}"
    _seed_conversation(foreign_cid, b["id"])

    r = client.post(f"/v1/projects/{a['id']}/conversations/{foreign_cid}/clear", headers=H)
    assert r.status_code == 404
    # Data for project B was NOT touched.
    assert len(agent_memory.get_messages(foreign_cid)) == 4


def test_clear_requires_auth(client):
    proj = _new_project(client, "Auth Required")
    cid = f"ws-{proj['id']}"
    r = client.post(f"/v1/projects/{proj['id']}/conversations/{cid}/clear")
    assert r.status_code in (401, 403)


def test_clear_unknown_project_404(client):
    r = client.post(
        "/v1/projects/does-not-exist/conversations/ws-does-not-exist/clear",
        headers=H,
    )
    assert r.status_code == 404


# ── agent_memory.clear_conversation unit-level invariants ───────────────────


def test_clear_preserves_conversation_row_and_bumps_updated_at(client, monkeypatch, tmp_path):
    """clear_conversation must leave the conversation row in place and
    update its updated_at so the UI can detect the clear."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    # Reset the module cache so it re-reads DATA_DIR.
    agent_memory._initialized = False

    cid = "unit-clear-test"
    agent_memory.get_or_create_conversation(cid, "project-assistant", "proj-x")
    agent_memory.append_message(cid, "user", "hi")
    before = agent_memory.get_conversation(cid)
    assert before is not None

    cleared = agent_memory.clear_conversation(cid)
    assert cleared["messages"] == 1

    after = agent_memory.get_conversation(cid)
    assert after is not None, "row must be preserved"
    # updated_at should have advanced (or at least been re-stamped).
    assert after["updated_at"] >= before["updated_at"]
