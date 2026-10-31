"""#49 post-hoc: pin the streamed-synthesis flush loop in _chat_stream_impl.

The post-hoc mutation sample on the landed #49 left flush-loop survivors. These
drive the real stream path (Agent.chat_stream -> _chat_stream_impl) and pin:
- M18/M19/M20: the cumulative-suffix bookkeeping (emitted_len / raw_emitted
  advance; emit only the NEW suffix). If any is dropped, each flush re-emits the
  whole accumulated prefix, so early text arrives at the client many times.
- M7/M11: the _MIN_FLUSH_CHARS gate batches a no-newline stream into chunks; if
  it is 0, every tiny delta flushes on its own.

Tests reference app.agents.core.streaming so the CI-M selection picks them up
for changes to that module.
"""
from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

import app.agents.core.streaming as _streaming  # noqa: F401 -- CI-M selection
from app.agents.runtime import Agent
from tests._fixture_values import STUB_TOKEN_04


_DEEPSEEK_CFG = {
    "provider": "deepseek",
    "url": "https://api.deepseek.com/v1/chat/completions",
    "env_key": "DEEPSEEK_API_KEY",
    "default_model": "deepseek-chat",
}


@pytest.fixture(autouse=True)
def _no_predispatch(monkeypatch):
    monkeypatch.setenv("AGENT_COMMISSIONING_PREDISPATCH", "0")


@pytest.fixture
def deepseek_streaming(monkeypatch):
    monkeypatch.setattr("app.agents.runtime._llm_config", lambda: dict(_DEEPSEEK_CFG))
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_04)
    monkeypatch.setenv("SYNTHESIS_STREAMING", "1")


def _pa_agent():
    return Agent(name="project-assistant", description="pa",
                 system_prompt="x", allowed_blocks=["construction"])


def _tool_then_final():
    state = {"n": 0}

    async def fake(_self, messages, api_key, **kwargs):
        state["n"] += 1
        if state["n"] == 1:
            return {"status": "success", "choice": {"message": {
                "role": "assistant", "content": "",
                "tool_calls": [{"id": "c1", "type": "function", "function": {
                    "name": "commissioning_checklist",
                    "arguments": '{"systems":["electrical"]}'}}]}}}
        return {"status": "success", "choice": {"message": {
            "role": "assistant", "content": "NON-STREAMED fallback answer."}}}

    fake.state = state
    return fake


async def _tool_ok(self, tool_call, **kwargs):
    return {"name": "commissioning_checklist", "ok": True,
            "result": {"status": "success", "checklists_by_system": {"electrical": []}}}


def _mk_stream(deltas):
    async def fake_stream(self, messages, api_key, **kwargs):
        for d in deltas:
            yield d
    return fake_stream


def _run_turn(agent):
    async def go():
        return [ev async for ev in agent.chat_stream(
            user_message="Generate a commissioning checklist for electrical.",
            history=[], project_id=None, conversation_id=None, user_id=None,
        )]
    return asyncio.run(go())


def _token_text(events):
    return "".join(e.get("content", "") for e in events if e.get("type") == "token")


# ── M18/M19/M20: no text is streamed to the client twice ────────────────────
@pytest.mark.parametrize("n", [40, 64])
def test_streamed_text_is_never_emitted_twice(deepseek_streaming, n):
    # Unique, newline-free markers. Correct bookkeeping emits each marker once;
    # dropping the emitted_len/raw_emitted advance re-emits the whole prefix
    # every flush, so an early marker would appear many times.
    deltas = [f"mark{i}word " for i in range(n)]
    with patch.object(Agent, "_call_llm", _tool_then_final()), \
         patch.object(Agent, "_run_tool_call", _tool_ok), \
         patch.object(Agent, "_stream_synthesis", _mk_stream(deltas)):
        events = _run_turn(_pa_agent())
    text = _token_text(events)
    assert "NON-STREAMED" not in text
    for marker in ("mark0word", "mark1word", f"mark{n - 1}word"):
        assert text.count(marker) == 1, (
            f"{marker!r} appears {text.count(marker)}x -- flushed text was duplicated"
        )
    assert events[-1]["type"] == "end"


# ── M7/M11: the min-flush gate batches a no-newline stream ──────────────────
@pytest.mark.parametrize("n", [160, 240])
def test_min_flush_batches_tiny_no_newline_deltas(deepseek_streaming, n):
    # Many 1-char, newline-free deltas. With the 24-char min-flush gate the
    # stable prefix is released in chunks, so there are far fewer token events
    # than deltas. Setting the gate to 0 flushes on nearly every delta.
    deltas = ["x" for _ in range(n)]
    with patch.object(Agent, "_call_llm", _tool_then_final()), \
         patch.object(Agent, "_run_tool_call", _tool_ok), \
         patch.object(Agent, "_stream_synthesis", _mk_stream(deltas)):
        events = _run_turn(_pa_agent())
    token_events = [e for e in events if e.get("type") == "token"]
    # Correct: ~ (n - margin) / 24 flushes, well under a quarter of n.
    assert len(token_events) < n // 4, (
        f"{len(token_events)} token events for {n} 1-char deltas -- min-flush gate not batching"
    )
    assert events[-1]["type"] == "end"
