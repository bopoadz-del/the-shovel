"""A streamed answer that ends on a promise to search is not an answer either.

#454 taught the NON-streamed branch of ``_chat_stream_impl`` that a
first-person promise to search is a dead end, not a final answer. The
streamed-synthesis branch -- the one production actually takes, because
``SYNTHESIS_STREAMING`` is set on the-fork -- never learned it. It retried
only on an EMPTY stream, and a promise is not empty.

Measured live on ``1594b32``, with #454 AND #455 both deployed, in a fresh
thread on theshovel.ai / Master Corpus. Battery test D1, "Who is the
Engineer's Representative on this project?", ended on, verbatim:

    I don't have the Engineer's Representative's name in the retrieved
    excerpts. Let me search the Contract Data and Schedules volume more
    specifically for that appointment.

Nothing followed. ``_SEARCH_PROMISE_TAIL_RE`` matches that string, so the
non-streamed branch would have forced a retry. The streamed branch shipped
it as the answer.

Two branches, one rule, one implementation. These pin both halves: the
promise is never flushed to the client, and the retry runs.

Harness copied from test_synthesis_streaming.py -- a deliverable tool call
on the first _call_llm sets force_synthesis, and the next iteration takes
the streamed branch.
"""
from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from app.agents.runtime import _EMPTY_RESPONSE_FALLBACK, Agent
from tests._fixture_values import STUB_TOKEN_04

_DEEPSEEK_CFG = {
    "provider": "deepseek",
    "url": "https://api.deepseek.com/v1/chat/completions",
    "env_key": "DEEPSEEK_API_KEY",
    "default_model": "deepseek-chat",
}

LIVE_D1 = (
    "I don't have the Engineer's Representative's name in the retrieved "
    "excerpts. Let me search the Contract Data and Schedules volume more "
    "specifically for that appointment."
)
LIVE_E1 = "I'm searching the executed contract volume for the Delay Damages figure."
REAL_ANSWER = "The Engineer's Representative is Project Signatory (AB-2023-101, Vol 1)."


@pytest.fixture(autouse=True)
def _no_predispatch(monkeypatch):
    monkeypatch.setenv("AGENT_COMMISSIONING_PREDISPATCH", "0")


@pytest.fixture
def deepseek_streaming(monkeypatch):
    monkeypatch.setattr("app.agents.runtime._llm_config", lambda: dict(_DEEPSEEK_CFG))
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_04)
    monkeypatch.setenv("SYNTHESIS_STREAMING", "1")
    monkeypatch.setenv("CHAT_STREAM_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("CHAT_STREAM_HEARTBEAT_SECONDS", "20")
    monkeypatch.setenv("FORCED_RETRY_MIN_SECONDS", "1")


def _pa_agent():
    return Agent(name="project-assistant", description="pa",
                 system_prompt="x", allowed_blocks=["construction"])


def _drain(agen):
    async def go():
        return [ev async for ev in agen]
    return asyncio.run(go())


def _tool_then(retry_text):
    """First call -> deliverable tool_call; every later call -> `retry_text`."""
    state = {"n": 0, "with_tools": []}

    async def fake(_self, messages, api_key, **kwargs):
        state["n"] += 1
        state["with_tools"].append(kwargs.get("with_tools", True))
        if state["n"] == 1:
            return {"status": "success", "choice": {"message": {
                "role": "assistant", "content": "",
                "tool_calls": [{"id": "c1", "type": "function", "function": {
                    "name": "commissioning_checklist",
                    "arguments": '{"systems":["electrical"]}'}}]}}}
        return {"status": "success", "choice": {"message": {
            "role": "assistant", "content": retry_text}}}

    fake.state = state
    return fake


async def _tool_ok(self, tool_call, **kwargs):
    return {"name": "commissioning_checklist", "ok": True,
            "result": {"status": "success", "checklists_by_system": {"electrical": []}}}


def _mk_stream(text):
    async def fake_stream(self, messages, api_key, **kwargs):
        if text:
            yield text
    return fake_stream


def _run(agent, streamed, retry_text):
    call_llm = _tool_then(retry_text)
    with patch.object(Agent, "_call_llm", call_llm), \
         patch.object(Agent, "_run_tool_call", _tool_ok), \
         patch.object(Agent, "_stream_synthesis", _mk_stream(streamed)):
        events = _drain(agent.chat_stream(
            user_message="Who is the Engineer's Representative on this project?",
            history=[], project_id=None, conversation_id=None, user_id=None,
        ))
    return events, call_llm.state


def _tokens(events):
    return "".join(e.get("content", "") for e in events if e.get("type") == "token")


# ── the live failure ─────────────────────────────────────────────────────────


def test_the_promise_is_never_flushed_to_the_client(deepseek_streaming):
    """D1, verbatim. The user must not see the promise at all."""
    events, _ = _run(_pa_agent(), LIVE_D1, REAL_ANSWER)
    shown = _tokens(events)
    assert "Let me search the Contract Data" not in shown, shown
    assert "I don't have the Engineer's Representative's name" not in shown, shown


def test_the_retry_answer_is_what_the_user_sees(deepseek_streaming):
    events, state = _run(_pa_agent(), LIVE_D1, REAL_ANSWER)
    assert "Project Signatory" in _tokens(events)
    assert state["n"] >= 2, "the forced retry never ran"
    assert False in state["with_tools"], "the retry must disable tools"


def test_e1_shape_is_caught_too(deepseek_streaming):
    """The other live string of the same class."""
    events, _ = _run(_pa_agent(), LIVE_E1, "Delay Damages are SAR 1,754,504.46/day.")
    shown = _tokens(events)
    assert "I'm searching the executed contract volume" not in shown
    assert "1,754,504.46" in shown


def test_a_retry_that_also_promises_falls_back(deepseek_streaming):
    """Two promises in a row must not ship the second one either."""
    events, _ = _run(_pa_agent(), LIVE_D1, LIVE_E1)
    shown = _tokens(events)
    assert "I'm searching the executed contract volume" not in shown
    assert _EMPTY_RESPONSE_FALLBACK.strip() in shown


# ── what must NOT change ─────────────────────────────────────────────────────


def test_a_real_streamed_answer_is_untouched(deepseek_streaming):
    """No retry, no held tokens, for an answer that answers."""
    events, state = _run(_pa_agent(), REAL_ANSWER, "SHOULD NOT BE USED")
    shown = _tokens(events)
    assert "Project Signatory" in shown
    assert "SHOULD NOT BE USED" not in shown
    assert state["n"] == 1, "synthesis should not have gone through _call_llm"


def test_an_answer_that_merely_mentions_searching_is_not_a_promise(deepseek_streaming):
    """'Search results show...' is a finding, not a dead end."""
    answer = "Search results show the Engineer's Representative is Project Signatory."
    events, state = _run(_pa_agent(), answer, "SHOULD NOT BE USED")
    assert "Project Signatory" in _tokens(events)
    assert state["n"] == 1


def test_an_empty_stream_still_retries(deepseek_streaming):
    """The behaviour this branch already had must survive."""
    events, state = _run(_pa_agent(), "", REAL_ANSWER)
    assert "Project Signatory" in _tokens(events)
    assert state["n"] >= 2
