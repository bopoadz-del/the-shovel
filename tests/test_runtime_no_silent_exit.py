"""``chat_stream`` must never exit silently (FOLLOW-UP #90).

The bug: when the LLM returned an empty content string (and no tool calls
and no DSML markup), the generator fell through with zero token events
and only an ``end`` event. The UI renders that as an empty assistant
bubble — no visible content, no error feedback.

The fix: every exit path emits either at least one ``token`` event OR a
structured ``error`` event before the closing ``end``. The chat_stream
wrapper guarantees this even if the inner generator escapes with an
unexpected exception.
"""
from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Dict, List
from unittest.mock import AsyncMock, patch

from app.agents.runtime import Agent, _EMPTY_RESPONSE_FALLBACK
from tests._fixture_values import STUB_TOKEN_01


def _agent() -> Agent:
    return Agent(
        name="silent-exit-test",
        description="silent-exit test",
        system_prompt="x",
        allowed_blocks=[],
    )


def _collect(gen: AsyncIterator[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drain an async generator into a list of events."""
    async def _run():
        return [event async for event in gen]
    return asyncio.run(_run())


def _mock_llm_empty_content() -> AsyncMock:
    """LLM returns success with empty content (the production failure mode)."""
    return AsyncMock(return_value={
        "status": "success",
        "choice": {"message": {"content": "", "tool_calls": []}},
    })


def _mock_llm_error() -> AsyncMock:
    return AsyncMock(return_value={"status": "error", "error": "boom"})


# ── Core regression: empty content must NOT yield a silent terminal ─────────


def test_empty_llm_content_emits_token_and_end(monkeypatch):
    """When the LLM returns empty content AND the forced retry also returns
    empty, the generator must substitute the fallback string AND yield at
    least one ``token`` event before ``end``."""
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_01)
    a = _agent()
    mock = _mock_llm_empty_content()
    with patch.object(a, "_call_llm", mock):
        events = _collect(a.chat_stream(user_message="hello"))

    types = [e["type"] for e in events]
    assert "token" in types, f"silent exit — no token events: {types}"
    assert "end" in types, f"no terminal end event: {types}"

    # The token content is the user-safe fallback, never raw exception text.
    token_text = "".join(e.get("content", "") for e in events if e["type"] == "token")
    assert token_text == _EMPTY_RESPONSE_FALLBACK


def test_llm_error_emits_structured_error(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_01)
    a = _agent()
    with patch.object(a, "_call_llm", _mock_llm_error()):
        events = _collect(a.chat_stream(user_message="hello"))

    types = [e["type"] for e in events]
    assert "error" in types, f"LLM error was swallowed silently: {types}"
    err_event = next(e for e in events if e["type"] == "error")
    assert "boom" in err_event["message"]


# ── Safety net: an exception inside the inner generator becomes error+end ──


def test_inner_generator_exception_becomes_error_event(monkeypatch):
    """A bug somewhere downstream that raises should NOT bubble up to the SSE
    consumer as a 500 — the wrapper must convert it into a clean error event."""
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_01)
    a = _agent()

    async def _boom(*_a, **_kw):
        raise RuntimeError("synthetic crash")

    with patch.object(a, "_call_llm", _boom):
        events = _collect(a.chat_stream(user_message="hello"))

    types = [e["type"] for e in events]
    assert "error" in types
    err_event = next(e for e in events if e["type"] == "error")
    assert "synthetic crash" in err_event["message"]
    # The fallback token also fires so the bubble isn't blank.
    assert any(e["type"] == "token" for e in events)


# ── Provider misconfigured: env-key missing → structured error ─────────────


def test_missing_provider_key_yields_error(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    a = _agent()
    events = _collect(a.chat_stream(user_message="hello"))
    types = [e["type"] for e in events]
    assert "error" in types
    err = next(e for e in events if e["type"] == "error")
    assert "DEEPSEEK_API_KEY" in err["message"]


# ── Wrapper invariant: terminal_emitted is always satisfied ───────────────


def test_every_run_emits_a_terminal_event(monkeypatch):
    """Regardless of the inner generator's outcome, the wrapper must always
    yield either ``end`` or ``error`` as the closing event class."""
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_01)
    a = _agent()
    with patch.object(a, "_call_llm", _mock_llm_empty_content()):
        events = _collect(a.chat_stream(user_message="hello"))
    terminals = {"end", "error"}
    assert any(e["type"] in terminals for e in events), (
        f"no end or error event: {[e['type'] for e in events]}"
    )
