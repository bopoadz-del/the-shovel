"""Phase C6 router tests — project/conversation wiring, delegation, reasoning toolkits.

Uses TestClient(app) with a real JWT (registered user) and a monkeypatched
Agent._call_llm so no network calls are made.
"""

import importlib
import uuid

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.agents.runtime import Agent, load_agents, AGENT_REGISTRY
from tests._fixture_values import PW_06, STUB_TOKEN_02

# ── Helpers ───────────────────────────────────────────────────────────────────

_RUN = uuid.uuid4().hex[:8]


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


def _register_and_login(client, suffix):
    email = f"c6-{suffix}-{_RUN}@x.com"
    client.post("/v1/users/register", json={"email": email, "password": PW_06})
    token = client.post("/v1/users/login", json={"email": email, "password": PW_06}).json()["token"]
    return {"Authorization": f"Bearer {token}"}


def _llm_ok(text="Hello from the mock LLM."):
    """Scripted _call_llm that returns a clean final answer (no tool calls)."""
    async def fake(self, messages, api_key, project_id=None, **kwargs):
        return {
            "status": "success",
            "choice": {"message": {"content": text, "tool_calls": []}},
            "raw": {},
        }
    return fake


def _isolate_memory(tmp_path, monkeypatch):
    """Redirect DATA_DIR to a temp dir and reload only memory-related stores.

    Note: This redirects ALL SQLite stores (users, projects, agent_memory) to
    tmp_path.  For HTTP-level tests that call /v1/users/register and
    /v1/projects we must NOT call this, because those endpoints rely on the
    users/projects tables being pre-initialised.  Use it only for pure-Python
    unit tests that never hit the HTTP layer.
    """
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    import app.core.agent_memory as _am
    import app.core.projects as _proj
    import app.core.project_memory as _pm
    import app.core.users as _users
    importlib.reload(_proj)
    importlib.reload(_pm)
    importlib.reload(_am)
    importlib.reload(_users)
    return _am


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_agent_chat_basic(client, monkeypatch):
    """POST /v1/agents/smart-orchestrator/chat with just a message returns 200."""
    monkeypatch.setattr(Agent, "_call_llm", _llm_ok())
    headers = _register_and_login(client, "basic")

    resp = client.post(
        "/v1/agents/smart-orchestrator/chat",
        json={"message": "hi"},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "success"
    assert "answer" in data


def test_agent_chat_accepts_project_id_for_owned_project(client, monkeypatch):
    """POST with project_id owned by caller returns 200."""
    monkeypatch.setattr(Agent, "_call_llm", _llm_ok())
    headers = _register_and_login(client, "owner")

    # Create a project
    proj_resp = client.post("/v1/projects", json={"name": "C6 Test Project"}, headers=headers)
    assert proj_resp.status_code in (200, 201), proj_resp.text
    pid = proj_resp.json()["id"]

    resp = client.post(
        "/v1/agents/smart-orchestrator/chat",
        json={"message": "what is this project?", "project_id": pid},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "success"


def test_agent_chat_rejects_unowned_project(client, monkeypatch):
    """POST with project_id owned by user A, called by user B → 404."""
    monkeypatch.setattr(Agent, "_call_llm", _llm_ok())

    headers_a = _register_and_login(client, "ownA")
    headers_b = _register_and_login(client, "ownB")

    # User A creates a project
    proj_resp = client.post("/v1/projects", json={"name": "A's Project"}, headers=headers_a)
    assert proj_resp.status_code in (200, 201), proj_resp.text
    pid = proj_resp.json()["id"]

    # User B tries to chat with it
    resp = client.post(
        "/v1/agents/smart-orchestrator/chat",
        json={"message": "hi", "project_id": pid},
        headers=headers_b,
    )
    assert resp.status_code == 404, resp.text


def test_agent_chat_persists_conversation(client, monkeypatch):
    """POST with conversation_id stores messages in agent_memory."""
    from app.core import agent_memory as am
    monkeypatch.setattr(Agent, "_call_llm", _llm_ok("remembered answer"))
    headers = _register_and_login(client, "conv")
    conv_id = f"conv-{_RUN}"

    resp = client.post(
        "/v1/agents/smart-orchestrator/chat",
        json={"message": "remember this", "conversation_id": conv_id},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    # conversation_id echoed back
    assert data.get("conversation_id") == conv_id

    # Check messages were persisted
    msgs = am.get_messages(conv_id)
    roles = [(m["role"], m["content"]) for m in msgs]
    assert ("user", "remember this") in roles
    assert ("assistant", "remembered answer") in roles


def test_orchestrator_can_delegate():
    """smart-orchestrator and heavy-reasoning have can_delegate=True; bim-analyst does not."""
    load_agents()
    assert AGENT_REGISTRY["smart-orchestrator"].can_delegate is True
    assert AGENT_REGISTRY["heavy-reasoning"].can_delegate is True
    assert AGENT_REGISTRY["bim-analyst"].can_delegate is False


def test_reasoning_blocks_broadened():
    """construction-pm, contracts-manager, safety-officer, smart-orchestrator
    all carry sympy_reasoning and formula_executor_v2 (the LLM-driven
    code-gen successor to formula_executor v1, which PR #72 deleted).
    """
    load_agents()
    for agent_name in ("construction-pm", "contracts-manager", "safety-officer", "smart-orchestrator"):
        agent = AGENT_REGISTRY[agent_name]
        assert "sympy_reasoning" in agent.allowed_blocks, (
            f"{agent_name} missing sympy_reasoning"
        )
        assert "formula_executor_v2" in agent.allowed_blocks, (
            f"{agent_name} missing formula_executor_v2 "
            f"(allowed_blocks={agent.allowed_blocks!r})"
        )


def test_project_assistant_loads():
    """project-assistant loads with can_delegate=True and the full
    construction toolkit so the UI's primary chat agent can actually
    call generate_wbs / boq_processor / drawing_qto / spec_analyzer
    instead of describing what it could do in prose. sympy_reasoning
    + formula_executor_v2 are kept for calculation paths
    (PR #72 deleted formula_executor v1; v2 is the LLM-driven
    code-gen replacement)."""
    agents = load_agents()

    # New agent is present
    assert "project-assistant" in agents, "project-assistant not in AGENT_REGISTRY"

    pa = agents["project-assistant"]
    assert pa.can_delegate is True, "project-assistant.can_delegate should be True"
    required = {
        "sympy_reasoning", "formula_executor_v2",
        "construction",       # exposes generate_wbs synthetic tool
        "boq_processor", "drawing_qto", "spec_analyzer",
        "validation_pipeline", "recommendation_template",
        "historical_benchmark",
    }
    missing = required - set(pa.allowed_blocks)
    assert not missing, (
        f"project-assistant is missing construction tools: {sorted(missing)}. "
        f"Got allowed_blocks={pa.allowed_blocks!r}"
    )

    # Total count: supervision-proposal (2026-08-15) makes 15
    assert len(agents) == 15, f"Expected 15 agents total, got {len(agents)}: {sorted(agents)}"

    # All 13 original agents still load without error
    _ORIGINAL_AGENTS = {
        "bim-analyst",
        "construction-pm",
        "contracts-manager",
        "document-analyst",
        "document-ingestion",
        "external-mcp",
        "heavy-reasoning",
        "learning",
        "quantity-surveyor",
        "safety-officer",
        "self-coding",
        "smart-orchestrator",
        "validation",
    }
    for name in _ORIGINAL_AGENTS:
        assert name in agents, f"Original agent '{name}' missing after adding project-assistant"
