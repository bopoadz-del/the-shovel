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
    def __init__(self, name, tokens, delay=0.0, raise_after=None):
        self.name = name
        self._tokens = tokens
        self._delay = delay
        self._raise_after = raise_after   # emit this many tokens, then raise
        self.started = False
        self.completed = False
        self.seen_pid = "<unset>"          # the project_id the helper passed in

    async def chat_stream(self, prompt, **kw):
        self.started = True
        self.seen_pid = kw.get("project_id", "<unset>")
        try:
            await asyncio.sleep(self._delay)
            for i, t in enumerate(self._tokens):
                if self._raise_after is not None and i >= self._raise_after:
                    raise RuntimeError("speculative provider blew up")
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


@pytest.mark.parametrize("prompt", ["hi", "hello there"])
def test_missing_agent_short_circuits_before_routing(monkeypatch, prompt):
    # get_agent returns None -> the helper emits ONE error frame and returns
    # WITHOUT running the routing hops or any stream. Dropping the `if agent is
    # None` guard (drop_rule) lets it fall through to _route (select runs) and a
    # None.chat_stream crash, so the guard-dropped output carries a route frame
    # and select was called. Asserting the sole error frame + select untouched
    # reddens the mutant.
    called = {"select": False}

    async def _select(_p, _a):
        called["select"] = True
        return object(), {"requested": "project-assistant", "final": "project-assistant"}

    monkeypatch.setattr(agents, "get_agent", lambda name: None)
    monkeypatch.setattr(chat, "understand_intent", _aval({"action": None}), raising=False)
    monkeypatch.setattr(runtime, "select_agent_for_message", _select, raising=False)
    monkeypatch.setattr(chat, "_predefined_enabled", lambda: True)
    monkeypatch.setattr(chat, "_predefined_workflows", lambda: set())
    monkeypatch.setattr(chat, "_message_vetoes_predefined", lambda _p: False)
    monkeypatch.setattr(_projects, "get_project_accessible", lambda *a, **k: {"id": "p"})
    frames = asyncio.run(_collect(chat._predispatch_concurrent_stream(
        prompt=prompt, history=[], project_id="p", user_id="u", conversation_id="c",
        attached_docs=[], document_ids=[], body={}, session_id="s", rid="r")))
    raws = [f for _, f in frames]
    assert len(raws) == 1, raws                       # ONLY the error frame
    assert '"type": "error"' in raws[0] and "temporarily unavailable" in raws[0]
    assert '"type": "route"' not in raws[0]
    assert called["select"] is False, "routing ran despite a missing agent"


@pytest.mark.parametrize("prompt", ["what is a provisional sum?", "define retention"])
def test_a_denied_project_resolves_the_stream_to_no_project(monkeypatch, prompt):
    # The tenant gate: when get_project_accessible RAISES, safe_pid falls to None
    # so the speculative/committed call runs with project_id=None (no leak into a
    # project the caller cannot see). The swallow_exception mutant turns that
    # `except -> safe_pid=None` into a no-op, leaving the raw pid -> the agent
    # would see 'proj-secret'. Asserting the agent saw None reddens it.
    pa = _FakeAgent("project-assistant", ["ok"], delay=0.0)

    def _boom(*a, **k):
        raise RuntimeError("tenant lookup failed")

    _wire(monkeypatch, pa_agent=pa, intent_result={"action": None},
          select_result=(pa, {"requested": "project-assistant", "final": "project-assistant"}))
    monkeypatch.setattr(_projects, "get_project_accessible", _boom)
    asyncio.run(_collect(chat._predispatch_concurrent_stream(
        prompt=prompt, history=[], project_id="proj-secret", user_id="u",
        conversation_id="c", attached_docs=[], document_ids=[], body={},
        session_id="s", rid="r")))
    assert pa.seen_pid is None, pa.seen_pid


@pytest.mark.parametrize("prompt", ["produce the schedule", "generate the wbs"])
def test_predefined_disabled_skips_the_intent_llm_entirely(monkeypatch, prompt):
    # With the predefined flag OFF, _intent_hop returns {action: None} WITHOUT
    # consulting understand_intent, so the turn stays pass-through even if the
    # (unused) intent model would have named a workflow. Dropping the
    # `if not _predefined_enabled(): return` guard (drop_rule) calls
    # understand_intent, gets 'generate_wbs' (a predefined workflow) and
    # redirects -> the spy fires and the answer is no longer the pass-through
    # one. Asserting the spy never fired AND the pass-through token shows reddens
    # the mutant.
    pa = _FakeAgent("project-assistant", ["PASS-THRU"], delay=0.0)
    consulted = {"intent": False}

    async def _intent_spy(_p, has_documents=False):
        consulted["intent"] = True
        return {"action": "generate_wbs", "params": {}}

    monkeypatch.setattr(agents, "get_agent", lambda name: pa)
    monkeypatch.setattr(chat, "understand_intent", _intent_spy, raising=False)
    monkeypatch.setattr(runtime, "select_agent_for_message",
                        _aval((pa, {"requested": "project-assistant", "final": "project-assistant"})),
                        raising=False)
    monkeypatch.setattr(chat, "_predefined_enabled", lambda: False)          # OFF
    monkeypatch.setattr(chat, "_predefined_workflows", lambda: {"generate_wbs"})
    monkeypatch.setattr(chat, "_message_vetoes_predefined", lambda _p: False)
    monkeypatch.setattr(_projects, "get_project_accessible", lambda *a, **k: {"id": "p"})
    frames = asyncio.run(_collect(chat._predispatch_concurrent_stream(
        prompt=prompt, history=[], project_id="p", user_id="u", conversation_id="c",
        attached_docs=[], document_ids=[], body={}, session_id="s", rid="r")))
    raw = "".join(f for _, f in frames)
    assert consulted["intent"] is False, "intent LLM was consulted while predefined was OFF"
    assert "PASS-THRU" in raw, "turn did not stay on the pass-through stream"


@pytest.mark.parametrize("prompt", ["what is a provisional sum?", "define float"])
def test_a_speculative_stream_that_raises_surfaces_an_error_frame(monkeypatch, prompt):
    # Pass-through, but the committed speculative stream RAISES after its first
    # token. _spec's `except Exception -> __spec_error__` must turn that into a
    # client error frame. The swallow_exception mutant drops that put, so the
    # client would see the one good token and a silent close with no error.
    # Asserting the error frame reddens the mutant.
    pa = _FakeAgent("project-assistant", ["partial", "never"], delay=0.0, raise_after=1)
    _wire(monkeypatch, pa_agent=pa, intent_result={"action": None},
          select_result=(pa, {"requested": "project-assistant", "final": "project-assistant"}))
    frames = asyncio.run(_collect(chat._predispatch_concurrent_stream(
        prompt=prompt, history=[], project_id="p", user_id="u", conversation_id="c",
        attached_docs=[], document_ids=[], body={}, session_id="s", rid="r")))
    raw = "".join(f for _, f in frames)
    assert "temporarily unavailable" in raw, "speculative failure did not surface an error frame"
