"""Nothing the app saved by itself enters a prompt unless someone verified it.

project_facts and agent_facts carry a mark (verified_by, verified_at). A row
without it is kept and listed, but the two prompt readers skip it:
``project_memory.build_memory_context`` (the project facts block on every
chat and agent turn) and the agent's "Known facts (you remembered):" block.
The memory route, where the user types a fact in, is the one path that saves
a fact as verified, under that user's id.
"""

import importlib

import pytest
from fastapi.testclient import TestClient

from app.agents.runtime import Agent
from tests._dev_key import DEV_KEY

H = {"Authorization": "Bearer " + DEV_KEY}


def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    import app.core.agent_memory as _am
    import app.core.projects as _proj
    import app.core.project_memory as _pm
    importlib.reload(_proj)
    importlib.reload(_pm)
    importlib.reload(_am)
    return _proj, _pm, _am


def _capture(box):
    async def fake(self, messages, api_key, project_id=None, **kwargs):
        box["messages"] = messages
        return {"status": "success", "choice": {"message": {"content": "ok"}}, "raw": {}}
    return fake


def _project_with_anchor(store) -> str:
    """A project with one verified fact, so a turn with no corpus still reaches
    the model (the empty-project gate passes on verified project context)."""
    pid = store.create_project("P")["id"]
    store.set_fact(pid, "employer", "Harbour Board", verified_by="u1")
    return pid


def _system_text(box) -> str:
    return "\n".join(m["content"] for m in box["messages"] if m["role"] == "system")


# ── the store: default not verified, every write sets the mark afresh ───────


@pytest.mark.parametrize("key,value", [("contract_value", "4,200,000"), ("ld_rate", "0.05%/day")])
def test_project_fact_defaults_to_not_verified(tmp_path, monkeypatch, key, value):
    store, _pm, _am = _isolate(tmp_path, monkeypatch)
    pid = store.create_project("P")["id"]
    fact = store.set_fact(pid, key, value)
    assert (fact["verified"], fact["verified_by"], fact["verified_at"]) == (False, None, None)
    assert [f["value"] for f in store.list_facts(pid)] == [value]
    assert store.list_facts(pid, verified_only=True) == []


@pytest.mark.parametrize("who", ["u1", "reviewer-7"])
def test_project_fact_verified_records_who_and_when(tmp_path, monkeypatch, who):
    store, _pm, _am = _isolate(tmp_path, monkeypatch)
    pid = store.create_project("P")["id"]
    fact = store.set_fact(pid, "employer", "Authority", verified_by=who)
    assert fact["verified"] is True and fact["verified_by"] == who
    assert fact["verified_at"] == fact["updated_at"]
    assert [f["key"] for f in store.list_facts(pid, verified_only=True)] == ["employer"]


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_verifier_is_not_a_verification(tmp_path, monkeypatch, blank):
    store, _pm, am = _isolate(tmp_path, monkeypatch)
    pid = store.create_project("P")["id"]
    assert store.set_fact(pid, "k", "v", verified_by=blank)["verified"] is False
    assert store.list_facts(pid, verified_only=True) == []
    assert am.set_agent_fact("a", "k", "v", project_id=pid, verified_by=blank)["verified"] is False
    assert am.list_agent_facts("a", pid, verified_only=True) == []


def test_app_rewrite_of_a_verified_fact_clears_the_mark(tmp_path, monkeypatch):
    """A new value nobody confirmed is not verified, even over a verified row."""
    store, _pm, am = _isolate(tmp_path, monkeypatch)
    pid = store.create_project("P")["id"]
    store.set_fact(pid, "contract_value", "4,200,000", verified_by="u1")
    rewritten = store.set_fact(pid, "contract_value", "5,000,000")
    assert (rewritten["value"], rewritten["verified"], rewritten["verified_at"]) == ("5,000,000", False, None)
    assert store.list_facts(pid, verified_only=True) == []

    am.set_agent_fact("a", "site", "Plot 7", project_id=pid, verified_by="u1")
    again = am.set_agent_fact("a", "site", "Plot 9", project_id=pid)
    assert (again["value"], again["verified"]) == ("Plot 9", False)
    assert am.list_agent_facts("a", pid, verified_only=True) == []


def test_search_facts_honours_verified_only(tmp_path, monkeypatch):
    store, _pm, _am = _isolate(tmp_path, monkeypatch)
    pid = store.create_project("P")["id"]
    store.set_fact(pid, "door_width", "0.90 m", verified_by="u1")
    store.set_fact(pid, "door_height", "2.40 m")
    assert {f["key"] for f in store.search_facts(pid, "door")} == {"door_width", "door_height"}
    assert [f["key"] for f in store.search_facts(pid, "door", verified_only=True)] == ["door_width"]


# ── the prompt readers ──────────────────────────────────────────────────────


def test_memory_context_carries_only_verified_facts(tmp_path, monkeypatch):
    store, pm, _am = _isolate(tmp_path, monkeypatch)
    pid = store.create_project("P")["id"]
    store.set_fact(pid, "contract_value", "4,200,000")
    store.set_fact(pid, "completion_date", "2027-03-01")
    assert pm.build_memory_context(pid, "contract value") == ""
    assert pm.build_memory_context(pid) == ""

    store.set_fact(pid, "completion_date", "2027-03-01", verified_by="u1")
    ctx = pm.build_memory_context(pid, "contract value")
    assert "2027-03-01" in ctx
    assert "4,200,000" not in ctx


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["480 per m3", "Thursday pour"])
async def test_agent_turn_reads_no_unverified_remembered_fact(tmp_path, monkeypatch, value):
    store, _pm, am = _isolate(tmp_path, monkeypatch)
    pid = _project_with_anchor(store)
    am.set_agent_fact("test-agent", "remembered", value, project_id=pid)
    store.set_fact(pid, "noted", value)
    box = {}
    monkeypatch.setattr(Agent, "_call_llm", _capture(box))
    agent = Agent(name="test-agent", description="t", system_prompt="You are a test agent.",
                  allowed_blocks=[], can_delegate=False)
    await agent.chat("what do you know?", api_key="k", project_id=pid)
    text = _system_text(box)
    assert value not in text
    assert "Known facts (you remembered)" not in text
    assert "Harbour Board" in text


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["Plot 7, North Gate", "Plot 12, East Gate"])
async def test_agent_turn_reads_a_verified_remembered_fact(tmp_path, monkeypatch, value):
    store, _pm, am = _isolate(tmp_path, monkeypatch)
    pid = _project_with_anchor(store)
    am.set_agent_fact("test-agent", "site", value, project_id=pid, verified_by="u1")
    am.set_agent_fact("test-agent", "guess", "not this one", project_id=pid)
    box = {}
    monkeypatch.setattr(Agent, "_call_llm", _capture(box))
    agent = Agent(name="test-agent", description="t", system_prompt="You are a test agent.",
                  allowed_blocks=[], can_delegate=False)
    await agent.chat("where is the site?", api_key="k", project_id=pid)
    text = _system_text(box)
    assert f"- site: {value}" in text
    assert "not this one" not in text


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["410 USD/m2", "Block B on Thursday"])
async def test_model_remembered_fact_is_stored_but_not_read_back(tmp_path, monkeypatch, value):
    """remember_fact is the model saving a value by itself: kept, not prompt material."""
    store, _pm, am = _isolate(tmp_path, monkeypatch)
    pid = _project_with_anchor(store)
    agent = Agent(name="test-agent", description="t", system_prompt="You are a test agent.",
                  allowed_blocks=[], can_delegate=False)
    call = {"id": "r1", "function": {"name": "remember_fact",
                                     "arguments": '{"key": "rate", "value": "%s"}' % value}}
    assert (await agent._run_tool_call(call, project_id=pid))["ok"] is True
    stored = am.list_agent_facts("test-agent", pid)
    assert [(f["value"], f["verified"]) for f in stored] == [(value, False)]

    box = {}
    monkeypatch.setattr(Agent, "_call_llm", _capture(box))
    await agent.chat("what is the rate?", api_key="k", project_id=pid)
    assert value not in _system_text(box)


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["4,200,000", "2027-03-01"])
async def test_unverified_facts_alone_do_not_open_an_empty_project(tmp_path, monkeypatch, value):
    """No corpus and only unverified facts: nothing to answer from, so the
    empty-project gate answers and the model is not called with them."""
    store, _pm, _am = _isolate(tmp_path, monkeypatch)
    pid = store.create_project("P")["id"]
    store.set_fact(pid, "remembered", value)
    box = {}
    monkeypatch.setattr(Agent, "_call_llm", _capture(box))
    agent = Agent(name="test-agent", description="t", system_prompt="You are a test agent.",
                  allowed_blocks=[], can_delegate=False)
    out = await agent.chat("what is the remembered value?", api_key="k", project_id=pid)
    assert box == {}
    assert value not in out["answer"]


# ── the memory route: the user's own entry is the verified one ──────────────


@pytest.fixture(scope="module")
def client():
    from app.main import app
    with TestClient(app) as c:
        yield c


@pytest.mark.parametrize("key,value", [("ld_rate", "0.1%/day"), ("employer", "Ports Authority")])
def test_user_entered_fact_is_verified_by_that_user(client, key, value):
    pid = client.post("/v1/projects", json={"name": "Verified Route"}, headers=H).json()["id"]
    saved = client.post(f"/v1/projects/{pid}/memory", json={"key": key, "value": value}, headers=H)
    assert saved.status_code == 201, saved.text
    body = saved.json()
    assert body["verified"] is True and body["verified_by"] and body["verified_at"]

    from app.core import projects as store
    store.set_fact(pid, "app_saved", "unconfirmed")
    listed = {f["key"]: f for f in client.get(f"/v1/projects/{pid}/memory", headers=H).json()["facts"]}
    # The user sees both rows, and which one is confirmed.
    assert (listed[key]["verified"], listed["app_saved"]["verified"]) == (True, False)

    from app.core.project_memory import build_memory_context
    ctx = build_memory_context(pid)
    assert value in ctx and "unconfirmed" not in ctx
