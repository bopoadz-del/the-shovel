"""6c pre-dispatch concurrency (ORDERS_31OCT_1605Z).

_predispatch_concurrent_stream runs the two routing hops (understand_intent +
select_agent_for_message) concurrently with each other AND with a speculative
project-assistant stream. On pass-through the speculative stream is committed;
on redirect it is cancelled before any client token and the routed path runs.

DESIGN NOTE (reported to the bus): emission is buffer-until-route, which
*guarantees* 1605Z item 2 ("cancel before any client token" on redirect) with no
race. The latency win is that generation overlaps routing, so the first token
lands at ~max(route, gen) instead of route+gen -- proven here by asserting the
first token arrives well under the sequential SUM (which also reddens the
gather->sequential mutant). It does NOT emit literally before route resolves;
doing so (optimistic emission) could leak a token on a redirect that resolves
slower than the first generated token, violating item 2.
"""
from __future__ import annotations

import asyncio
import time

import pytest

import app.routers.chat as chat
import app.agents.runtime as runtime
import app.agents as agents
import app.core.projects as _projects


class _FakeAgent:
    def __init__(self, name, tokens, delay=0.0):
        self.name = name
        self._tokens = tokens
        self._delay = delay
        self.started = False
        self.completed = False

    async def chat_stream(self, prompt, **kw):
        self.started = True
        try:
            await asyncio.sleep(self._delay)
            for t in self._tokens:
                yield {"type": "token", "content": t}
            yield {"type": "end", "content": "", "sources": [], "tools": []}
            self.completed = True
        except asyncio.CancelledError:
            raise


def _wire(monkeypatch, *, pa_agent, intent_result, select_result,
          intent_delay=0.0, select_delay=0.0, predefined=None):
    """Stub the routing hops + get_agent for the helper."""
    async def _intent(_prompt, has_documents=False):
        await asyncio.sleep(intent_delay)
        if isinstance(intent_result, Exception):
            raise intent_result
        return intent_result

    async def _select(_prompt, _agent):
        await asyncio.sleep(select_delay)
        if isinstance(select_result, Exception):
            raise select_result
        return select_result

    monkeypatch.setattr(chat, "understand_intent", _intent, raising=False)
    monkeypatch.setattr(runtime, "select_agent_for_message", _select, raising=False)
    monkeypatch.setattr(agents, "get_agent", lambda name: pa_agent)
    monkeypatch.setattr(chat, "_predefined_enabled", lambda: True)
    monkeypatch.setattr(chat, "_predefined_workflows", lambda: set(predefined or ()))
    monkeypatch.setattr(chat, "_message_vetoes_predefined", lambda _p: False)
    monkeypatch.setattr(_projects, "get_project_accessible", lambda *a, **k: {"id": "p"})


async def _collect(gen):
    """Drain the SSE generator; return [(t_since_start, raw_frame)]."""
    t0 = time.monotonic()
    out = []
    async for frame in gen:
        out.append((time.monotonic() - t0, frame))
    return out


@pytest.mark.parametrize("delay", [0.15, 0.20])
def test_passthrough_first_token_beats_the_sequential_sum(monkeypatch, delay):
    # Hops (each `delay`) and the speculative stream (`delay`) all run at once,
    # so the first token lands ~max(delay, delay)=delay, far under the sequential
    # sum 3*delay. The gather->sequential mutant pushes route to 2*delay and the
    # first token to ~2*delay, over the bound -> reddens.
    pa = _FakeAgent("project-assistant", ["Hi", " there"], delay=delay)
    _wire(monkeypatch, pa_agent=pa,
          intent_result={"action": None},
          select_result=(pa, {"requested": "project-assistant", "final": "project-assistant"}),
          intent_delay=delay, select_delay=delay)
    frames = asyncio.run(_collect(chat._predispatch_concurrent_stream(
        prompt="hello", history=[], project_id="p", user_id="u",
        conversation_id="c", attached_docs=[], document_ids=[], body={},
        session_id="s", rid="r")))
    first_tok = next((t for t, f in frames if '"type": "token"' in f), None)
    assert first_tok is not None, frames
    assert first_tok < delay * 2, (first_tok, delay)  # concurrency; sum would be ~3*delay


@pytest.mark.parametrize("final", ["heavy-reasoning", "self-coding"])
def test_redirect_cancels_the_speculative_stream_with_zero_client_tokens(monkeypatch, final):
    # select redirects; the speculative project-assistant stream (slow, so no
    # token before route resolves) must be cancelled with zero client tokens and
    # the routed agent answers. Removing spec_task.cancel() lets the speculative
    # stream run to completion (pa.completed True) -> reddens.
    pa = _FakeAgent("project-assistant", ["LEAK"], delay=0.5)      # slow: no token before route
    routed = _FakeAgent(final, ["routed-answer"], delay=0.0)
    monkeypatch.setattr(chat, "understand_intent", _aval({"action": None}), raising=False)
    monkeypatch.setattr(runtime, "select_agent_for_message",
                        _aval((routed, {"requested": "project-assistant", "final": final})), raising=False)
    monkeypatch.setattr(agents, "get_agent", lambda name: pa)
    monkeypatch.setattr(chat, "_predefined_enabled", lambda: True)
    monkeypatch.setattr(chat, "_predefined_workflows", lambda: set())
    monkeypatch.setattr(chat, "_message_vetoes_predefined", lambda _p: False)
    monkeypatch.setattr(_projects, "get_project_accessible", lambda *a, **k: {"id": "p"})

    frames = asyncio.run(_collect(chat._predispatch_concurrent_stream(
        prompt="build a 200-activity schedule", history=[], project_id="p", user_id="u",
        conversation_id="c", attached_docs=[], document_ids=[], body={},
        session_id="s", rid="r")))
    raw = "".join(f for _, f in frames)
    assert "LEAK" not in raw, "speculative stream leaked a client token on redirect"
    assert "routed-answer" in raw, "routed agent did not answer"
    assert pa.completed is False, "speculative stream was not cancelled (cancel removed?)"


@pytest.mark.parametrize("boom", [RuntimeError("route boom"), ValueError("bad route")])
def test_routing_exception_fails_open_to_the_speculative_stream(monkeypatch, boom):
    # Both hops raise; _route fails open to the requested agent (pass-through),
    # so the speculative stream commits and its tokens reach the client.
    pa = _FakeAgent("project-assistant", ["ok-answer"], delay=0.0)
    _wire(monkeypatch, pa_agent=pa, intent_result=boom, select_result=boom)
    frames = asyncio.run(_collect(chat._predispatch_concurrent_stream(
        prompt="what is a provisional sum?", history=[], project_id="p", user_id="u",
        conversation_id="c", attached_docs=[], document_ids=[], body={},
        session_id="s", rid="r")))
    raw = "".join(f for _, f in frames)
    assert "ok-answer" in raw, "fail-open did not commit the speculative stream"


def _aval(value):
    async def _f(*a, **k):
        return value
    return _f


@pytest.mark.parametrize("missing", [None, None])
def test_missing_agent_yields_an_error_frame_not_a_crash(monkeypatch, missing):
    # get_agent returns None -> the helper emits an error frame and returns.
    # Dropping the `if agent is None` guard (drop_rule) would call None.chat_stream
    # and raise -> this reddens it.
    monkeypatch.setattr(agents, "get_agent", lambda name: missing)
    monkeypatch.setattr(chat, "understand_intent", _aval({"action": None}), raising=False)
    monkeypatch.setattr(runtime, "select_agent_for_message",
                        _aval((object(), {"requested": "project-assistant", "final": "project-assistant"})),
                        raising=False)
    monkeypatch.setattr(chat, "_predefined_enabled", lambda: False)
    monkeypatch.setattr(chat, "_predefined_workflows", lambda: set())
    monkeypatch.setattr(chat, "_message_vetoes_predefined", lambda _p: False)
    monkeypatch.setattr(_projects, "get_project_accessible", lambda *a, **k: {"id": "p"})
    frames = asyncio.run(_collect(chat._predispatch_concurrent_stream(
        prompt="hi", history=[], project_id="p", user_id="u", conversation_id="c",
        attached_docs=[], document_ids=[], body={}, session_id="s", rid="r")))
    raw = "".join(f for _, f in frames)
    assert '"type": "error"' in raw and "temporarily unavailable" in raw
