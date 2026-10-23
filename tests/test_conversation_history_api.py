"""Tests for GET /v1/agents/conversations/{conversation_id}/messages.

Stream D Part 1 — expose persisted chat history for the workspace.

Uses TestClient(app) with a real JWT and a monkeypatched Agent._call_llm so
no network calls are made.  DATA_DIR is NOT isolated (same pattern as
test_agents_router_c6.py) because the HTTP layer needs the real users/projects
tables; uuid suffixes prevent cross-run collisions.
"""

import uuid

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.agents.runtime import Agent
from tests._fixture_values import PW_06, STUB_TOKEN_02

# ── module-level run-id ────────────────────────────────────────────────────────

_RUN = uuid.uuid4().hex[:8]


# ── fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _stub_llm_key(monkeypatch):
    """Agent.chat() guards on the provider API key *before* _call_llm is reached.

    These tests monkeypatch _call_llm so no network call ever happens, but the
    guard still fires when the key is unset (e.g. in CI, where conftest's
    load_dotenv finds no .env key).  A placeholder satisfies the guard.

    Deliberately a per-file fixture, not in conftest.py: a global key would
    un-skip the live DEEPSEEK acceptance tests (their skipif is evaluated at
    collection time) and make them run against the real API with a fake key.
    """
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_02)


# ── helpers ────────────────────────────────────────────────────────────────────

def _register_and_login(client, suffix: str) -> dict:
    email = f"histapi-{suffix}-{_RUN}@x.com"
    client.post("/v1/users/register", json={"email": email, "password": PW_06})
    token = client.post(
        "/v1/users/login", json={"email": email, "password": PW_06}
    ).json()["token"]
    return {"Authorization": f"Bearer {token}"}


def _fake_llm(text: str = "LLM reply"):
    """Return a scripted _call_llm that resolves with no tool calls."""
    async def _inner(self, messages, api_key, project_id=None, **kwargs):
        return {
            "status": "success",
            "choice": {"message": {"content": text, "tool_calls": []}},
            "raw": {},
        }
    return _inner


# ── tests ──────────────────────────────────────────────────────────────────────

def test_history_empty_for_new_conversation(client):
    """GET an owned-but-unused ws-{pid} conversation returns 200 + empty list.

    The conversation row never existed, but the caller owns the project, so the
    access check passes and an empty history is served.
    """
    headers = _register_and_login(client, "empty")

    proj_resp = client.post(
        "/v1/projects", json={"name": f"EmptyConv-{_RUN}"}, headers=headers
    )
    assert proj_resp.status_code in (200, 201), proj_resp.text
    pid = proj_resp.json()["id"]
    cid = f"ws-{pid}"

    resp = client.get(f"/v1/agents/conversations/{cid}/messages", headers=headers)

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["conversation_id"] == cid
    assert data["messages"] == []


def test_history_returns_persisted_turns(client, monkeypatch):
    """After a POST chat, GET should return the user + assistant turns oldest-first."""
    monkeypatch.setattr(Agent, "_call_llm", _fake_llm("mocked answer"))
    monkeypatch.setattr(
        "app.agents.runtime.project_is_rag_ready", lambda _pid: True
    )
    headers = _register_and_login(client, "persist")

    # Create a project so the workspace conversation has an owner
    proj_resp = client.post(
        "/v1/projects", json={"name": f"HistTest-{_RUN}"}, headers=headers
    )
    assert proj_resp.status_code in (200, 201), proj_resp.text
    pid = proj_resp.json()["id"]
    cid = f"ws-{pid}"

    # Send a chat message via the /chat endpoint (not streaming) so the turn is persisted
    chat_resp = client.post(
        "/v1/agents/project-assistant/chat",
        json={
            "message": "hello history",
            "project_id": pid,
            "conversation_id": cid,
        },
        headers=headers,
    )
    assert chat_resp.status_code == 200, chat_resp.text

    # GET the history
    hist_resp = client.get(f"/v1/agents/conversations/{cid}/messages", headers=headers)
    assert hist_resp.status_code == 200, hist_resp.text
    data = hist_resp.json()
    assert data["conversation_id"] == cid

    msgs = data["messages"]
    assert len(msgs) >= 2, f"Expected at least 2 messages, got: {msgs}"

    roles = [m["role"] for m in msgs]
    # Oldest-first: user comes before assistant
    assert roles[0] == "user"
    assert roles[-1] == "assistant"

    # Content check. Coverage honesty may append "N of M project documents
    # indexed", so match the mocked body as a substring, not list equality.
    contents = [m["content"] for m in msgs]
    assert "hello history" in contents
    assert any("mocked answer" in (c or "") for c in contents)


def test_history_cross_tenant_404(client, monkeypatch):
    """User B cannot read User A's workspace conversation — should get 404."""
    monkeypatch.setattr(Agent, "_call_llm", _fake_llm("tenant reply"))

    headers_a = _register_and_login(client, "tenantA")
    headers_b = _register_and_login(client, "tenantB")

    # User A creates a project and chats, creating ws-{pid} tied to A's project_id
    proj_resp = client.post(
        "/v1/projects", json={"name": f"A-Project-{_RUN}"}, headers=headers_a
    )
    assert proj_resp.status_code in (200, 201), proj_resp.text
    pid = proj_resp.json()["id"]
    cid = f"ws-{pid}"

    chat_resp = client.post(
        "/v1/agents/project-assistant/chat",
        json={
            "message": "secret message",
            "project_id": pid,
            "conversation_id": cid,
        },
        headers=headers_a,
    )
    assert chat_resp.status_code == 200, chat_resp.text

    # Verify the conversation row has project_id set (sanity-check our assumption)
    from app.core import agent_memory as am
    conv = am.get_conversation(cid)
    assert conv is not None, "Conversation row should exist after chat"
    assert conv["project_id"] == pid, (
        f"Expected project_id={pid!r}, got {conv['project_id']!r}"
    )

    # User B tries to GET User A's conversation — must be 404
    resp_b = client.get(f"/v1/agents/conversations/{cid}/messages", headers=headers_b)
    assert resp_b.status_code == 404, (
        f"Expected 404 for cross-tenant access, got {resp_b.status_code}: {resp_b.text}"
    )


# ── cross-tenant data-leak regression (NULL-project_id attack) ──────────────────

def test_attacker_cannot_precreate_victim_conversation(client, monkeypatch):
    """An attacker must NOT be able to write to a victim's ws-{pid} conversation.

    The original exploit: POST /chat with conversation_id=ws-{victim_pid} and
    NO project_id in the body → get_or_create_conversation creates the row with
    project_id=NULL and the chat endpoint skips its ownership check.

    The fix derives ownership from the ws- id, so the write path now rejects
    this with 404 before the agent ever runs.
    """
    monkeypatch.setattr(Agent, "_call_llm", _fake_llm("attacker reply"))

    headers_victim = _register_and_login(client, "victim-pre")
    headers_attacker = _register_and_login(client, "attacker-pre")

    # Victim creates a project — project id is visible in workspace URLs.
    proj_resp = client.post(
        "/v1/projects", json={"name": f"Victim-Pre-{_RUN}"}, headers=headers_victim
    )
    assert proj_resp.status_code in (200, 201), proj_resp.text
    pid = proj_resp.json()["id"]

    # Attacker POSTs chat targeting the victim's ws-{pid} with NO project_id.
    resp = client.post(
        "/v1/agents/project-assistant/chat",
        json={
            "message": "precreate victim conversation",
            "conversation_id": f"ws-{pid}",
            # deliberately no project_id
        },
        headers=headers_attacker,
    )
    assert resp.status_code == 404, (
        f"Expected 404 — attacker must not write victim's conversation, "
        f"got {resp.status_code}: {resp.text}"
    )

    # The conversation row must NOT have been created.
    from app.core import agent_memory as am
    assert am.get_conversation(f"ws-{pid}") is None, (
        "Attacker should not have been able to create the conversation row"
    )


def test_attacker_cannot_read_victim_conversation_even_if_precreated(client, monkeypatch):
    """Worst case: a ws-{victim_pid} row already exists with project_id=NULL.

    Even then, the attacker GET must be 404 (ownership is derived from the
    ws- id, not the spoofable stored row), while the victim's own GET is 200.
    """
    monkeypatch.setattr(Agent, "_call_llm", _fake_llm("victim reply"))

    headers_victim = _register_and_login(client, "victim-read")
    headers_attacker = _register_and_login(client, "attacker-read")

    proj_resp = client.post(
        "/v1/projects", json={"name": f"Victim-Read-{_RUN}"}, headers=headers_victim
    )
    assert proj_resp.status_code in (200, 201), proj_resp.text
    pid = proj_resp.json()["id"]
    cid = f"ws-{pid}"

    # Simulate the worst case: a NULL-project_id row already exists.
    from app.core import agent_memory as am
    am.get_or_create_conversation(cid, "project-assistant", project_id=None)
    am.append_message(cid, "user", "victim secret")
    am.append_message(cid, "assistant", "victim secret reply")

    # Attacker GET — must be 404 (no ownership of pid).
    resp_attacker = client.get(
        f"/v1/agents/conversations/{cid}/messages", headers=headers_attacker
    )
    assert resp_attacker.status_code == 404, (
        f"Expected 404 — attacker must not read victim's conversation, "
        f"got {resp_attacker.status_code}: {resp_attacker.text}"
    )

    # Victim GET — must be 200 and see their own messages.
    resp_victim = client.get(
        f"/v1/agents/conversations/{cid}/messages", headers=headers_victim
    )
    assert resp_victim.status_code == 200, resp_victim.text
    contents = [m["content"] for m in resp_victim.json()["messages"]]
    assert "victim secret" in contents


def test_new_chat_suffixed_ws_id_is_owned_by_project_owner(client):
    """UI 'New chat' mints ws-{pid}-{Date.now()}. That must not 404 the owner.

    _enforce_conversation_access used to treat everything after 'ws-' as the
    project id, so ws-{pid}-1740000000000 looked up a nonexistent project and
    the workspace showed 'Something went wrong' on the first send.
    """
    headers = _register_and_login(client, "newchat-owner")
    proj = client.post(
        "/v1/projects", json={"name": f"NewChat-{_RUN}"}, headers=headers
    )
    assert proj.status_code in (200, 201), proj.text
    pid = proj.json()["id"]
    cid = f"ws-{pid}-1740000000000"

    resp = client.get(
        f"/v1/agents/conversations/{cid}/messages", headers=headers
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["conversation_id"] == cid
    assert resp.json()["messages"] == []


def test_new_chat_suffixed_ws_id_still_404s_for_an_attacker(client):
    """A suffix must not let an attacker skip the project-ownership check."""
    victim = _register_and_login(client, "newchat-victim")
    attacker = _register_and_login(client, "newchat-attacker")
    proj = client.post(
        "/v1/projects", json={"name": f"NewChatVictim-{_RUN}"}, headers=victim
    )
    assert proj.status_code in (200, 201), proj.text
    pid = proj.json()["id"]
    cid = f"ws-{pid}-1740000000000"

    resp = client.get(
        f"/v1/agents/conversations/{cid}/messages", headers=attacker
    )
    assert resp.status_code == 404, resp.text


def test_workspace_project_candidates_covers_legacy_and_new_chat():
    from app.routers.agents import _workspace_project_candidates

    assert _workspace_project_candidates("ws-5942b19b") == ["5942b19b"]
    assert _workspace_project_candidates("ws-5942b19b-1740000000000") == [
        "5942b19b-1740000000000",
        "5942b19b",
    ]
    assert _workspace_project_candidates("ws-master_corpus-1") == [
        "master_corpus-1",
        "master_corpus",
    ]
    assert _workspace_project_candidates("hr-user-default") == []
