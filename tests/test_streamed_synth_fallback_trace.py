"""The streamed-synthesis fallback names the rule that replaced the reply.

When the streamed synthesis is unusable, ``_chat_stream_impl`` makes one
no-tools retry. If that retry is unusable too, the user is shown
``_EMPTY_RESPONSE_FALLBACK`` (which the answer post-processor may then
reword). The non-streamed forced retry logs that
decision by rule name (``forced retry unusable reason=<rule>`` /
``forced retry failed reason=provider_error``). The streamed-synthesis branch
made the same decision through ``_final_text_needs_forced_retry``, which logs
nothing. So a live turn could end on "I was unable to generate a response"
and leave no line saying which rule threw the model's reply away.

These tests drive the real streamed-synthesis path. A deliverable tool call
sets force_synthesis, the stream returns a search promise, and the retry's
reply is the input under test. The text is invented; no live answer is
quoted.
"""
from __future__ import annotations

import asyncio
import logging
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

WHERE = "chat_stream streamed-synth"
PROMISE_A = "Let me search the piling drawings again for the anchor length."
PROMISE_B = "I am searching the geotechnical report for the allowable bearing value."
ANSWER_A = "The anchor length is 9.5 m below cut-off level [1]."
ANSWER_B = "Allowable bearing is 180 kPa at 1.5 m depth [2]."


@pytest.fixture(autouse=True)
def _no_predispatch(monkeypatch):
    monkeypatch.setenv("AGENT_COMMISSIONING_PREDISPATCH", "0")


@pytest.fixture
def streaming(monkeypatch):
    monkeypatch.setattr("app.agents.runtime._llm_config", lambda: dict(_DEEPSEEK_CFG))
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_04)
    monkeypatch.setenv("SYNTHESIS_STREAMING", "1")
    monkeypatch.setenv("CHAT_STREAM_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("CHAT_STREAM_HEARTBEAT_SECONDS", "20")
    monkeypatch.setenv("FORCED_RETRY_MIN_SECONDS", "1")


def _agent():
    return Agent(name="project-assistant", description="pa",
                 system_prompt="x", allowed_blocks=["construction"])


def _tool_then(retry):
    """First call -> deliverable tool_call; later calls -> ``retry``.

    ``retry`` is the reply text, or None for a provider error."""
    state = {"n": 0}

    async def fake(_self, messages, api_key, **kwargs):
        state["n"] += 1
        if state["n"] == 1:
            return {"status": "success", "choice": {"message": {
                "role": "assistant", "content": "",
                "tool_calls": [{"id": "c1", "type": "function", "function": {
                    "name": "commissioning_checklist",
                    "arguments": '{"systems":["electrical"]}'}}]}}}
        if retry is None:
            return {"status": "error", "error": "upstream 503"}
        return {"status": "success", "choice": {"message": {
            "role": "assistant", "content": retry}}}

    fake.state = state
    return fake


async def _tool_ok(self, tool_call, **kwargs):
    return {"name": "commissioning_checklist", "ok": True,
            "result": {"status": "success", "checklists_by_system": {"electrical": []}}}


def _stream(text):
    async def fake_stream(self, messages, api_key, **kwargs):
        if text:
            yield text
    return fake_stream


def _run(streamed, retry, caplog):
    call_llm = _tool_then(retry)
    caplog.set_level(logging.INFO)
    with patch.object(Agent, "_call_llm", call_llm), \
         patch.object(Agent, "_run_tool_call", _tool_ok), \
         patch.object(Agent, "_stream_synthesis", _stream(streamed)):
        async def go():
            return [ev async for ev in _agent().chat_stream(
                user_message="What does the project say about the piles?",
                history=[], project_id=None, conversation_id=None, user_id=None,
            )]
        events = asyncio.run(go())
    shown = "".join(e.get("content", "") for e in events if e.get("type") == "token")
    lines = [r.getMessage() for r in caplog.records]
    return shown, lines, call_llm.state


# ── the retry is unusable: the fallback is shown AND the rule is named ──────


@pytest.mark.parametrize("streamed,retry,reason", [
    (PROMISE_A, PROMISE_B, "search_promise"),
    (PROMISE_B, PROMISE_A, "search_promise"),
    (PROMISE_A, "", "empty"),
    (PROMISE_B, "   \n ", "empty"),
])
def test_an_unusable_retry_names_its_rule(streaming, caplog, streamed, retry, reason):
    shown, lines, state = _run(streamed, retry, caplog)
    assert state["n"] >= 2, "the forced retry never ran"
    # Neither unusable reply reaches the user. (What replaces the fallback
    # line afterwards belongs to the answer post-processor, not to this.)
    assert streamed not in shown, shown
    assert not retry.strip() or retry not in shown, shown
    raw = f" raw={len(retry)}c"
    assert f"{WHERE}: forced retry unusable reason={reason}{raw}; fallback shown" in lines, lines


@pytest.mark.parametrize("streamed", [PROMISE_A, PROMISE_B])
def test_a_provider_error_on_the_retry_is_named_as_such(streaming, caplog, streamed):
    shown, lines, _ = _run(streamed, None, caplog)
    assert _EMPTY_RESPONSE_FALLBACK.strip() in shown, shown
    assert f"{WHERE}: forced retry failed reason=provider_error; fallback shown" in lines, lines
    assert not any("forced retry unusable reason=" in m for m in lines), lines


# ── the retry answers: no fallback, no unusable trace ───────────────────────


@pytest.mark.parametrize("streamed,retry,needle", [
    (PROMISE_A, ANSWER_A, "9.5 m"),
    (PROMISE_B, ANSWER_B, "180 kPa"),
])
def test_a_usable_retry_is_shown_and_not_traced_as_unusable(streaming, caplog, streamed, retry, needle):
    shown, lines, state = _run(streamed, retry, caplog)
    assert state["n"] >= 2
    assert needle in shown, shown
    assert _EMPTY_RESPONSE_FALLBACK.strip() not in shown
    assert not any("forced retry unusable" in m or "forced retry failed" in m for m in lines), lines


@pytest.mark.parametrize("answer,needle", [(ANSWER_A, "9.5 m"), (ANSWER_B, "180 kPa")])
def test_a_usable_stream_makes_no_retry_and_no_trace(streaming, caplog, answer, needle):
    shown, lines, state = _run(answer, "SHOULD NOT BE USED", caplog)
    assert needle in shown
    assert state["n"] == 1
    assert not any("forced retry" in m for m in lines), lines
