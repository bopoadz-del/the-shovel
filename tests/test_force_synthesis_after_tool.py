"""After a successful deliverable (non-search) tool result, the next LLM call
must be tool-free (force_synthesis). This stops the tool-loop and, on Groq,
avoids the large-context tool_use_failed(400)/429 -> slow-fallback hang, since a
tool-free request can neither emit a mis-parsed tool call nor loop.

Regression guard for the commissioning-hang fix.
"""
from __future__ import annotations

import asyncio
from unittest.mock import patch, AsyncMock

import pytest

from app.agents.runtime import Agent
from tests.conftest import requires_construction_kit
from tests._fixture_values import STUB_TOKEN_02


@pytest.fixture(autouse=True)
def _stub_llm_key(monkeypatch):
    """Agent.chat() guards on the provider api key *before* _call_llm is
    reached. These tests monkeypatch _call_llm so no network call ever
    happens, but the guard still fires when the key is unset (e.g. in CI,
    where conftest's load_dotenv finds no .env key). Pin the provider to
    deepseek so the guard checks DEEPSEEK_API_KEY regardless
    of a developer's LLM_PROVIDER, then satisfy it with a placeholder.

    Deliberately a per-file fixture, not in conftest.py: a global key would
    un-skip the live DEEPSEEK acceptance tests (their skipif is evaluated at
    collection time) and make them run against the real API with a fake key.
    """
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_02)
    # Remaining commissioning predispatch would draft the checklist before
    # iter-0 and force synthesis immediately. These tests pin the older
    # "model calls the tool, THEN we lock" path.
    monkeypatch.setenv("AGENT_COMMISSIONING_PREDISPATCH", "0")


def _pa_agent():
    return Agent(
        name="project-assistant", description="pa",
        system_prompt="x", allowed_blocks=["construction"],
    )


def _run(coro):
    return asyncio.run(coro)


def _drain(agen):
    async def go():
        return [ev async for ev in agen]
    return _run(go())


def _script_two_calls(with_tools_seen):
    """Fake _call_llm: 1st call returns a commissioning tool_call, 2nd returns a
    final answer. Records the with_tools kwarg of every call."""
    state = {"n": 0}

    async def fake(_self, messages, api_key, **kwargs):
        with_tools_seen.append(kwargs.get("with_tools", True))
        state["n"] += 1
        if state["n"] == 1:
            return {"status": "success", "choice": {"message": {
                "role": "assistant", "content": "",
                "tool_calls": [{"id": "c1", "type": "function", "function": {
                    "name": "commissioning_checklist",
                    "arguments": '{"systems":["electrical"]}'}}]}}}
        return {"status": "success", "choice": {"message": {
            "role": "assistant", "content": "Here is the commissioning checklist."}}}

    return fake


async def _fake_tool_ok(self, tool_call, **kwargs):
    return {"name": "commissioning_checklist", "ok": True,
            "result": {"status": "success", "checklists_by_system": {"electrical": []}}}


async def _fake_tool_search(self, tool_call, **kwargs):
    return {"name": "search_project_documents", "ok": True,
            "result": {"status": "success", "results": []}}


def test_force_synthesis_after_deliverable_tool(monkeypatch):
    """A successful commissioning_checklist result -> 2nd call is tool-free."""
    agent = _pa_agent()
    seen: list = []
    with patch.object(Agent, "_call_llm", _script_two_calls(seen)), \
         patch.object(Agent, "_run_tool_call", _fake_tool_ok):
        events = _drain(agent.chat_stream(
            user_message="Generate a commissioning checklist for electrical.",
            history=[], project_id=None, conversation_id=None, user_id=None,
        ))
    assert seen == [True, False], seen  # 1st call has tools, synthesis does not
    assert any(e.get("type") == "token" for e in events)


def test_empty_sympy_metadata_does_not_force_synthesis():
    """Leftover-hat L7: sympy with no boq_data returns formula metadata only.
    That must NOT disarm tools, or formula_executor_v2 never runs."""
    from app.agents.runtime import _should_force_synthesis
    rec = {
        "name": "sympy_reasoning",
        "ok": True,
        "result": {
            "status": "success",
            "variances": [],
            "cost_impacts": [],
            "formulas": {"variance_pct": "x"},
        },
    }
    assert _should_force_synthesis(rec) is False


def test_search_result_does_not_force_synthesis(monkeypatch):
    """A search result is exploratory — it must NOT force synthesis, so the
    model can keep working (2nd call still has tools)."""
    agent = _pa_agent()
    seen: list = []
    with patch.object(Agent, "_call_llm", _script_two_calls(seen)), \
         patch.object(Agent, "_run_tool_call", _fake_tool_search):
        _drain(agent.chat_stream(
            user_message="What does the spec say about concrete cover?",
            history=[], project_id=None, conversation_id=None, user_id=None,
        ))
    assert seen == [True, True], seen  # search does not disable tools


def test_force_synthesis_env_disable(monkeypatch):
    """AGENT_FORCE_SYNTHESIS=0 restores the old always-offer-tools behaviour."""
    monkeypatch.setenv("AGENT_FORCE_SYNTHESIS", "0")
    agent = _pa_agent()
    seen: list = []
    with patch.object(Agent, "_call_llm", _script_two_calls(seen)), \
         patch.object(Agent, "_run_tool_call", _fake_tool_ok):
        _drain(agent.chat_stream(
            user_message="Generate a commissioning checklist for electrical.",
            history=[], project_id=None, conversation_id=None, user_id=None,
        ))
    assert seen == [True, True], seen  # kill-switch keeps tools on both calls


@requires_construction_kit
def test_remaining_commissioning_locks_synthesis_immediately(monkeypatch):
    """Live M3: remaining already drafted the checklist — do not offer tools."""
    monkeypatch.delenv("AGENT_COMMISSIONING_PREDISPATCH", raising=False)
    agent = _pa_agent()
    seen: list = []

    async def fake(_self, messages, api_key, **kwargs):
        seen.append(kwargs.get("with_tools", True))
        return {"status": "success", "choice": {"message": {
            "role": "assistant",
            "content": "Here is the commissioning checklist.",
        }}}

    with patch.object(Agent, "_call_llm", fake):
        events = _drain(agent.chat_stream(
            user_message=(
                "Generate a commissioning checklist for the PWPS-09 "
                "reservoir before the first wet test."
            ),
            history=[], project_id=None, conversation_id=None, user_id=None,
        ))
    assert seen == [False], seen
    assert any(e.get("type") == "token" for e in events)
