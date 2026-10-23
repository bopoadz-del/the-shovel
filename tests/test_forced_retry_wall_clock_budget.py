"""The forced no-tools retry must fit inside the turn's wall clock.

Live request 43e40b3a-e8f on the-fork @ 3e35389 (2026-08-31T01:34-01:38Z):

    01:36:41  TIMING chat_stream iter=0 call=126.8s status=success tools=final
    01:36:41  TIMING chat_stream EMPTY-FINAL raw=380c -> forced retry, cum=126.8s
    01:38:31  chat_stream: wall-clock deadline exceeded after 240.0s

There is no ``TIMING chat_stream forced-retry call=`` line for that request --
in seven days of logs there is none at all. The retry was awaited and never
returned: the producer task was cancelled inside it when the 240s turn
deadline fired. The user waited four minutes and got a timeout banner.

The arithmetic never worked. LLM_HTTP_TIMEOUT_SECONDS defaults to 150 PER
CALL and _call_llm walks a provider fallback ladder inside that, while
CHAT_STREAM_TIMEOUT_SECONDS caps the WHOLE TURN at 240. Any turn that needs
a second call -- which #454 made far more likely by dropping the
request-type gate on the forced retry -- can exceed the turn cap before the
second call has a chance to finish. The 200s default left ~40s after a hung
primary — often no budget for OpenRouter. 150s leaves ~90s.

Two rules, both pinned here:

1. A forced retry that cannot finish inside the remaining budget is not
   started. The fallback text is delivered immediately instead.
2. Every LLM call inside the turn is given the turn's deadline, so the
   fallback ladder cannot outlive it either.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator, Dict, List
from unittest.mock import patch

import pytest
from tests._fixture_values import STUB_TOKEN_01

from app.agents.runtime import (
    _EMPTY_RESPONSE_FALLBACK,
    _MIN_LLM_ATTEMPT_SECONDS,
    Agent,
    _forced_retry_min_seconds,
    _llm_http_timeout,
)

# The exact string that tripped the forced retry on the live request.
LIVE_PREAMBLE = "I'm searching the executed contract volume for that figure."


def _agent() -> Agent:
    return Agent(
        name="budget-test",
        description="budget test",
        system_prompt="x",
        allowed_blocks=[],
    )


def _collect(gen: AsyncIterator[Dict[str, Any]]) -> List[Dict[str, Any]]:
    async def _run():
        return [event async for event in gen]
    return asyncio.run(_run())


def _setup_provider(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_01)


def _ok(content: str) -> Dict[str, Any]:
    return {
        "status": "success",
        "choice": {"message": {"content": content, "role": "assistant"}},
        "raw": {"model": "test-model"},
    }


# ── the live shape ───────────────────────────────────────────────────────────


def test_forced_retry_is_skipped_when_the_turn_cannot_afford_it(monkeypatch):
    """43e40b3a-e8f: slow first call, unusable answer, no budget for a retry.

    The turn must end with the fallback text and a clean terminal event --
    not with the retry started and the whole turn cancelled inside it.
    """
    _setup_provider(monkeypatch)
    monkeypatch.setenv("CHAT_STREAM_TIMEOUT_SECONDS", "3")
    monkeypatch.setenv("CHAT_STREAM_HEARTBEAT_SECONDS", "10")
    monkeypatch.setenv("FORCED_RETRY_MIN_SECONDS", "2")

    a = _agent()
    calls: List[Dict[str, Any]] = []

    async def _slow_then_preamble(messages, *_a, **kw):
        calls.append({"with_tools": kw.get("with_tools", True)})
        if len(calls) == 1:
            await asyncio.sleep(1.5)   # 1.5s of a 3s turn spent
            return _ok(LIVE_PREAMBLE)
        return _ok("the retry answer")  # must never be reached

    t0 = time.monotonic()
    with patch.object(a, "_call_llm", _slow_then_preamble):
        events = _collect(a.chat_stream(user_message="what are the delay damages?"))
    elapsed = time.monotonic() - t0

    assert len(calls) == 1, (
        f"forced retry was started with too little budget: {calls}"
    )
    types = [e["type"] for e in events]
    assert "error" not in types, f"turn ended on an error: {events}"
    assert types[-1] == "end", f"no clean terminal event: {types}"

    text = "".join(e.get("content", "") for e in events if e["type"] == "token")
    assert text.strip() == _EMPTY_RESPONSE_FALLBACK.strip(), repr(text)
    # The preamble is what #454 stopped showing as an answer. Still stopped.
    assert LIVE_PREAMBLE not in text

    assert elapsed < 3.0, (
        f"turn ran to the wall clock ({elapsed:.1f}s) instead of answering early"
    )


def test_forced_retry_still_runs_when_there_is_budget(monkeypatch):
    """The guard must not disable the retry -- only the doomed ones."""
    _setup_provider(monkeypatch)
    monkeypatch.setenv("CHAT_STREAM_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("CHAT_STREAM_HEARTBEAT_SECONDS", "20")
    monkeypatch.setenv("FORCED_RETRY_MIN_SECONDS", "2")

    a = _agent()
    calls: List[Dict[str, Any]] = []

    async def _preamble_then_answer(messages, *_a, **kw):
        calls.append({"with_tools": kw.get("with_tools", True)})
        if len(calls) == 1:
            return _ok(LIVE_PREAMBLE)
        return _ok("Delay damages are capped at 10% of the contract sum.")

    with patch.object(a, "_call_llm", _preamble_then_answer):
        events = _collect(a.chat_stream(user_message="what are the delay damages?"))

    assert len(calls) == 2, f"forced retry did not run: {calls}"
    assert calls[1]["with_tools"] is False, "retry must disable tools"
    text = "".join(e.get("content", "") for e in events if e["type"] == "token")
    assert "10%" in text, repr(text)


def test_every_llm_call_in_the_turn_carries_the_deadline(monkeypatch):
    """_call_llm gets an absolute deadline strictly inside the turn's."""
    _setup_provider(monkeypatch)
    monkeypatch.setenv("CHAT_STREAM_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("CHAT_STREAM_HEARTBEAT_SECONDS", "20")

    a = _agent()
    seen: List[Any] = []
    turn_deadline_floor = time.monotonic() + 30.0

    async def _capture(messages, *_a, **kw):
        seen.append(kw.get("deadline"))
        return _ok("a plain answer")

    with patch.object(a, "_call_llm", _capture):
        _collect(a.chat_stream(user_message="hello"))

    assert seen, "no LLM call was made"
    for d in seen:
        assert d is not None, "an LLM call was made with no deadline"
        assert d < turn_deadline_floor, (
            "LLM deadline must leave head-room before the turn deadline"
        )


# ── _call_llm's own budget enforcement ───────────────────────────────────────


def test_call_llm_refuses_an_attempt_it_cannot_finish(monkeypatch):
    """A deadline already past means no HTTP request is even opened."""
    _setup_provider(monkeypatch)
    a = _agent()

    opened: List[Any] = []

    class _Boom:
        def __init__(self, *args, **kwargs):
            opened.append(kwargs.get("timeout"))
            raise AssertionError("no HTTP client may be opened")

    with patch("app.agents.runtime.httpx.AsyncClient", _Boom):
        resp = asyncio.run(
            a._call_llm(
                [{"role": "user", "content": "hi"}],
                "key",
                deadline=time.monotonic() - 1.0,
            )
        )

    assert opened == [], opened
    assert resp["status"] == "error"
    assert "budget" in resp["error"].lower(), resp


def test_call_llm_caps_the_attempt_timeout_at_the_remaining_budget(monkeypatch):
    """The per-call timeout must shrink to what the turn has left.

    Without this the fallback ladder spends _llm_http_timeout() per hop
    (150s by default) against a 240s turn cap.
    """
    _setup_provider(monkeypatch)
    monkeypatch.setenv("LLM_HTTP_TIMEOUT_SECONDS", "150")
    a = _agent()

    seen: List[float] = []

    class _Capture:
        def __init__(self, *args, **kwargs):
            seen.append(kwargs.get("timeout"))
            raise RuntimeError("stop here")

    with patch("app.agents.runtime.httpx.AsyncClient", _Capture):
        asyncio.run(
            a._call_llm(
                [{"role": "user", "content": "hi"}],
                "key",
                deadline=time.monotonic() + 40.0,
            )
        )

    assert seen, "no attempt was made"
    assert all(t is not None for t in seen), seen
    assert max(seen) <= 40.0, f"attempt timeout exceeded the budget: {seen}"
    assert min(seen) >= _MIN_LLM_ATTEMPT_SECONDS, seen


def test_call_llm_without_a_deadline_is_unchanged(monkeypatch):
    """No deadline (chat(), tools, internal callers) => the configured timeout."""
    _setup_provider(monkeypatch)
    monkeypatch.delenv("LLM_HTTP_TIMEOUT_SECONDS", raising=False)
    a = _agent()

    seen: List[float] = []

    class _Capture:
        def __init__(self, *args, **kwargs):
            seen.append(kwargs.get("timeout"))
            raise RuntimeError("stop here")

    with patch("app.agents.runtime.httpx.AsyncClient", _Capture):
        asyncio.run(a._call_llm([{"role": "user", "content": "hi"}], "key"))

    assert seen and seen[0] == 150.0, seen


def test_llm_http_timeout_default_is_150(monkeypatch):
    """A5: hung DeepSeek at 200s burned the 240s turn; 150s leaves failover room."""
    monkeypatch.delenv("LLM_HTTP_TIMEOUT_SECONDS", raising=False)
    assert _llm_http_timeout() == 150.0


def test_llm_http_timeout_garbage_falls_back_to_150(monkeypatch):
    monkeypatch.setenv("LLM_HTTP_TIMEOUT_SECONDS", "not-a-number")
    assert _llm_http_timeout() == 150.0


# ── naming the stalled component ─────────────────────────────────────────────


def test_wall_clock_error_names_what_it_was_waiting_on(monkeypatch):
    """The timeout message must say WHICH phase hung.

    Diagnosing 43e40b3a-e8f needed the ABSENCE of a log line. It should not
    have; the error itself should say it.
    """
    _setup_provider(monkeypatch)
    monkeypatch.setenv("CHAT_STREAM_TIMEOUT_SECONDS", "2")
    monkeypatch.setenv("CHAT_STREAM_HEARTBEAT_SECONDS", "1")

    a = _agent()

    async def _hang(*_a, **_kw):
        await asyncio.Future()

    with patch.object(a, "_call_llm", _hang):
        events = _collect(a.chat_stream(user_message="hello"))

    err = next(e for e in events if e["type"] == "error")
    msg = err["message"]
    assert "timeout" in msg.lower(), msg          # frontend banner mapping
    assert "llm-call iter=0" in msg, msg          # the named component


def test_wall_clock_error_names_the_forced_retry(monkeypatch):
    """The live stall, reproduced: hang INSIDE the forced retry."""
    _setup_provider(monkeypatch)
    monkeypatch.setenv("CHAT_STREAM_TIMEOUT_SECONDS", "3")
    monkeypatch.setenv("CHAT_STREAM_HEARTBEAT_SECONDS", "2")
    monkeypatch.setenv("FORCED_RETRY_MIN_SECONDS", "0")  # allow the doomed retry

    a = _agent()
    calls: List[int] = []

    async def _preamble_then_hang(*_a, **_kw):
        calls.append(1)
        if len(calls) == 1:
            return _ok(LIVE_PREAMBLE)
        await asyncio.Future()

    with patch.object(a, "_call_llm", _preamble_then_hang):
        events = _collect(a.chat_stream(user_message="what are the delay damages?"))

    err = next(e for e in events if e["type"] == "error")
    assert "forced-retry" in err["message"], err["message"]


# ── the knob ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("raw,expected", [("90", 90.0), ("0", 0.0), ("", 45.0)])
def test_forced_retry_min_seconds_env(monkeypatch, raw, expected):
    if raw:
        monkeypatch.setenv("FORCED_RETRY_MIN_SECONDS", raw)
    else:
        monkeypatch.delenv("FORCED_RETRY_MIN_SECONDS", raising=False)
    assert _forced_retry_min_seconds() == expected


def test_forced_retry_min_seconds_survives_garbage(monkeypatch):
    monkeypatch.setenv("FORCED_RETRY_MIN_SECONDS", "not-a-number")
    assert _forced_retry_min_seconds() == 45.0
