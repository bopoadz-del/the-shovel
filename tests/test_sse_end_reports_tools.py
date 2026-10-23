"""The SSE ``end`` event must name the tools the turn actually ran.

Reported symptom: a self-coding answer produced the right number
(31.62 USD/ft2) and said "Tool: code" in its prose, while the SSE reported
``tools=[]``. The prose and the stream disagreed because only the prose carried
the information -- every ``end`` event emitted iterations/model/sources/exports
and no tools field at all, so the UI read an ABSENT key as an empty list.

The runtime always knew: the TIMING log line prints
``tools=['formula_executor_v2']`` for the same turn. These tests pin the field
onto the machine-readable contract so the badge cannot silently disagree with
the answer again.
"""
from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from app.agents.runtime import Agent
from tests._fixture_values import STUB_TOKEN_02


@pytest.fixture(autouse=True)
def _stub_llm_key(monkeypatch):
    """chat_stream guards on the provider key before _call_llm is reached, and
    these tests never make a network call. Pin the provider so the guard checks
    a key we control regardless of the developer's LLM_PROVIDER."""
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_02)


def _agent():
    return Agent(
        name="project-assistant", description="pa",
        system_prompt="x", allowed_blocks=["construction"],
    )


def _drain(agen):
    async def go():
        return [ev async for ev in agen]
    return asyncio.run(go())


def _end_of(events):
    ends = [e for e in events if e.get("type") == "end"]
    assert len(ends) == 1, f"expected exactly one end event, got {len(ends)}"
    return ends[0]


def _script(tool_name: str | None, arguments: str = "{}"):
    """Fake _call_llm: optionally one tool call, then a final answer."""
    state = {"n": 0}

    async def fake(_self, messages, api_key, **kwargs):
        state["n"] += 1
        if state["n"] == 1 and tool_name:
            return {"status": "success", "choice": {"message": {
                "role": "assistant", "content": "",
                "tool_calls": [{"id": "c1", "type": "function", "function": {
                    "name": tool_name, "arguments": arguments}}]}}}
        return {"status": "success", "choice": {"message": {
            "role": "assistant",
            "content": "The unit rate is 31.62 USD/ft2. Tool: code"}}}

    return fake


def _fake_tool(name: str):
    async def run(self, tool_call, **kwargs):
        return {"name": name, "ok": True,
                "result": {"status": "success", "value": 31.62}}
    return run


def test_end_event_names_the_tool_that_ran():
    """The exact reported case: formula_executor_v2 ran, so it must be named."""
    agent = _agent()
    with patch.object(Agent, "_call_llm", _script("formula_executor_v2")), \
         patch.object(Agent, "_run_tool_call", _fake_tool("formula_executor_v2")):
        events = _drain(agent.chat_stream(
            user_message="Work out the unit rate per square foot.",
            history=[], project_id=None, conversation_id=None, user_id=None,
        ))
    end = _end_of(events)
    assert "tools" in end, "end event dropped the tools field entirely"
    assert end["tools"] == ["formula_executor_v2"], end["tools"]


def test_end_event_tools_matches_the_emitted_tool_call_events():
    """The badge and the per-call events must describe the same turn.

    Comparing against the tool_call events rather than a hardcoded name is what
    catches a future path that streams a call but forgets to record it.
    """
    agent = _agent()
    with patch.object(Agent, "_call_llm", _script("construction")), \
         patch.object(Agent, "_run_tool_call", _fake_tool("construction")):
        events = _drain(agent.chat_stream(
            user_message="Generate a commissioning checklist for electrical.",
            history=[], project_id=None, conversation_id=None, user_id=None,
        ))
    streamed = [e.get("tool") or e.get("name")
                for e in events if e.get("type") == "tool_call"]
    assert streamed, "no tool_call events were emitted at all"
    assert _end_of(events)["tools"] == streamed


def test_end_event_reports_empty_list_when_no_tool_ran():
    """A tool-free turn must report ``tools: []`` -- present and empty.

    An absent key and an empty list render identically in the UI, which is
    exactly how the original bug hid: there was no way to tell "ran nothing"
    apart from "forgot to say".
    """
    agent = _agent()
    with patch.object(Agent, "_call_llm", _script(None)):
        events = _drain(agent.chat_stream(
            user_message="Say hello.",
            history=[], project_id=None, conversation_id=None, user_id=None,
        ))
    end = _end_of(events)
    assert end["tools"] == []
    assert not [e for e in events if e.get("type") == "tool_call"]


def test_capability_answer_early_exit_still_emits_tools():
    """The short-circuit exits must carry the field too -- and must not crash.

    Two `end` events (unindexed-project and capability-answer) sit ABOVE the
    point where the tool accumulator was first declared, so referencing it there
    raised UnboundLocalError on paths the happy-path tests never walk. This
    drives the capability short-circuit, which returns before any LLM call.
    """
    agent = _agent()
    from app.agents.runtime import _is_capability_request
    question = "What tools do you have?"
    assert _is_capability_request(question), (
        "test premise broken: this no longer takes the capability short-circuit"
    )
    # No _call_llm patch: this path must return before any LLM call is made.
    events = _drain(agent.chat_stream(
        user_message=question,
        history=[], project_id=None, conversation_id=None, user_id=None,
    ))
    end = _end_of(events)
    assert end["tools"] == []
    assert end["iterations"] == 0


# ── the heavy-reasoning stream is a SEPARATE end event ─────────────────────

def test_heavy_reasoning_end_event_names_its_tools_in_call_order():
    """`/v1/chat/stream`'s heavy path builds its own end event in chat.py, not
    through chat_stream, so it needed the same field added separately.

    It already emitted `tools_used`, but SORTED and de-duped into a set --
    which loses the order the tools actually ran in. The agent stream reports
    call order, so a UI reading one key across both paths would have seen the
    same turn described two different ways. `tools` is call-ordered on both.
    """
    import asyncio
    import json as _json

    from app.routers import chat as chat_router

    class _FakeAgent:
        async def chat(self, **kwargs):
            on_event = kwargs.get("on_event")
            if on_event:
                await on_event("tool_call", {"tool": "sympy_reasoning"})
            return {
                "status": "success",
                "answer": "The unit rate is 31.62 USD/ft2.",
                "iterations": 2,
                # Deliberately NOT alphabetical, and with a repeat: sorted()
                # would render this as [construction_calc, sympy_reasoning].
                "tool_calls": [
                    {"name": "sympy_reasoning"},
                    {"name": "construction_calc"},
                    {"name": "sympy_reasoning"},
                ],
            }

    import app.agents as agents_mod
    real_get_agent = agents_mod.get_agent
    agents_mod.get_agent = lambda _n: _FakeAgent()
    try:
        async def drain():
            out = []
            async for chunk in chat_router._stream_from_heavy_reasoning(
                user_message="work out the rate",
                project_id=None,
                user_id=None,
                history=[],
                session_id="t1",
            ):
                out.append(chunk)
            return out

        chunks = asyncio.run(drain())
    finally:
        agents_mod.get_agent = real_get_agent

    ends = []
    for c in chunks:
        for line in c.splitlines():
            if line.startswith("data: "):
                ev = _json.loads(line[6:])
                if ev.get("type") == "end":
                    ends.append(ev)
    assert len(ends) == 1, f"expected one end event, got {len(ends)}"
    end = ends[0]
    assert "tools" in end, "heavy-reasoning end event has no tools field"
    assert end["tools"] == ["sympy_reasoning", "construction_calc"], end["tools"]
    # The legacy key stays for older consumers, sorted as it always was.
    assert end["tools_used"] == ["construction_calc", "sympy_reasoning"]


def test_predefined_end_event_names_the_container_action():
    """The predefined path dispatches a container action directly, so nothing
    ever populated `tools` -- this is the path a live payment_certificate turn
    took, and it came back with the key absent.

    The action IS the tool that ran, so naming it is what makes the badge
    agree with the answer.
    """
    import asyncio
    import json as _json

    from app.routers import chat as chat_router
    import app.core.predefined_reasoning as pr

    async def fake_run_workflow(action, context, session):
        return {
            "handled": True,
            "answer": "Interim Payment Certificate - Error: no contract value.",
            "plan_steps": [],
            "exports": [],
        }

    real = pr.run_workflow
    pr.run_workflow = fake_run_workflow
    try:
        async def drain():
            out = []
            async for chunk in chat_router._stream_from_predefined(
                action="payment_certificate",
                user_message="Issue the interim payment certificate.",
                project_id=None,
                user_id=None,
                session_id="t1",
                document_ids=[],
            ):
                out.append(chunk)
            return out

        chunks = asyncio.run(drain())
    finally:
        pr.run_workflow = real

    ends = []
    for c in chunks:
        for line in c.splitlines():
            if line.startswith("data: "):
                ev = _json.loads(line[6:])
                if ev.get("type") == "end":
                    ends.append(ev)
    assert len(ends) == 1, f"expected one end event, got {len(ends)}"
    end = ends[0]
    assert end["mode"] == "predefined"
    assert end["workflow"] == "payment_certificate"
    assert end["tools"] == ["payment_certificate"], end
