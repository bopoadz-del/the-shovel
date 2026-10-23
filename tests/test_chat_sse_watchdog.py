"""A TURN ENDS WITH AN ANSWER OR A REASON (owner's numbered item 5).

F-SILENT-1, from ``FLEET_OPS/artifacts/gate_battery_13b2bf7_2026-08-31.md``:

    a turn that ends with neither an answer nor an error (B6, first pass).
    119 seconds, three events (``start``, ``route``, ``start``), then
    nothing: no tokens, no ``end`` content, no error, HTTP 200.

B6's own event sequence is the fixture below, because the fence has to be
against the observed shape and not against a convenient one. The agent's
existing guard did not catch it: that guard begins inside
``Agent.chat_stream``, and B6's last event was the agent's own ``start``,
with the turn dying in the router's frame -- past ``route``.

Each test names the mutation it kills.
"""

import asyncio
import json

import pytest

from app.routers.chat_watchdog import (
    DEFAULT_WATCHDOG_SECONDS,
    SILENT_TURN_FALLBACK,
    event_type,
    frame,
    guarantee_terminal,
    watchdog_seconds,
)

#: The B6 turn, verbatim in shape: start, route, start, then nothing.
B6_FRAMES = [
    frame({"type": "start", "agent": "project-assistant"}),
    frame({"type": "route", "requested": "project-assistant",
           "final": "quantity-surveyor", "confidence": 0.82}),
    frame({"type": "start", "agent": "quantity-surveyor"}),
]


async def _gen(items, *, raises=None, hang=False):
    for item in items:
        yield item
    if hang:
        await asyncio.sleep(3600)
    if raises is not None:
        raise raises


async def _drain(agen):
    return [f async for f in agen]


def _types(frames):
    return [event_type(f) for f in frames]


# -- the pass-through cases ------------------------------------------------


@pytest.mark.asyncio
async def test_a_turn_that_ends_properly_is_not_touched():
    """Mutation killed: appending a synthetic end unconditionally, which
    would put a second `end` on every healthy turn."""
    src = B6_FRAMES + [frame({"type": "token", "content": "Rev B"}),
                       frame({"type": "end", "iterations": 1, "sources": []})]
    out = await _drain(guarantee_terminal(_gen(src)))
    assert out == src


@pytest.mark.asyncio
async def test_a_turn_that_ends_in_an_error_is_not_touched():
    src = B6_FRAMES + [frame({"type": "error", "message": "upstream refused"})]
    out = await _drain(guarantee_terminal(_gen(src)))
    assert out == src


# -- B6 --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_b6_shape_no_longer_ends_in_silence():
    """THE regression. start, route, start, nothing -> a named failure.

    Mutation killed: any change that lets the generator finish without a
    terminal frame.
    """
    out = await _drain(guarantee_terminal(_gen(list(B6_FRAMES)), request_id="rq-1"))
    assert out[:3] == B6_FRAMES
    assert _types(out) == ["start", "route", "start", "token", "error", "end"]

    body = json.loads(out[-2].split("data:", 1)[1].strip())
    # It names WHERE it stopped. "No answer and no error" is not actionable;
    # "ended after 'start'" says which frame to look at -- and 'start' is
    # exactly what B6 answered.
    assert "'start'" in body["message"]
    assert "platform fault" in body["message"]
    assert body["request_id"] == "rq-1"


@pytest.mark.asyncio
async def test_the_bubble_is_never_left_empty():
    """A silent turn renders as an empty assistant bubble. Mutation killed:
    emitting only the error, which the UI shows as a banner over nothing."""
    out = await _drain(guarantee_terminal(_gen(list(B6_FRAMES))))
    token = json.loads(out[3].split("data:", 1)[1].strip())
    assert token["content"] == SILENT_TURN_FALLBACK


@pytest.mark.asyncio
async def test_a_turn_that_already_said_something_gets_no_fallback_token():
    """Mutation killed: always prepending the fallback, which would append
    "did not produce an answer" to a turn that produced most of one."""
    src = B6_FRAMES + [frame({"type": "token", "content": "QZ-ENG-310"})]
    out = await _drain(guarantee_terminal(_gen(src)))
    assert _types(out) == ["start", "route", "start", "token", "error", "end"]
    assert SILENT_TURN_FALLBACK not in "".join(out)


# -- the producer blowing up ----------------------------------------------


@pytest.mark.asyncio
async def test_an_escaping_exception_becomes_a_named_terminal():
    out = await _drain(
        guarantee_terminal(_gen(list(B6_FRAMES), raises=RuntimeError("boom")))
    )
    assert _types(out)[-3:] == ["token", "error", "end"]
    body = json.loads(out[-2].split("data:", 1)[1].strip())
    assert "RuntimeError" in body["message"]


@pytest.mark.asyncio
async def test_the_exception_text_is_never_streamed_to_the_client():
    """Exception text carries infra detail -- the LLM tunnel URL has been in
    one. The class name says enough to triage; the server log has the rest.

    Mutation killed: interpolating the exception itself into the message.
    """
    secret = "https://tunnel.internal.example/llm?key=abc123"
    out = await _drain(
        guarantee_terminal(_gen(list(B6_FRAMES), raises=RuntimeError(secret)))
    )
    assert secret not in "".join(out)


@pytest.mark.asyncio
async def test_a_disconnected_client_is_not_answered():
    """Cancellation means nobody is listening. Mutation killed: catching
    CancelledError with the generic handler, which both invents a reply for
    an absent client and swallows the cancellation."""
    async def cancelling():
        yield B6_FRAMES[0]
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await _drain(guarantee_terminal(cancelling()))


# -- the wall clock --------------------------------------------------------


@pytest.mark.asyncio
async def test_a_hung_turn_is_ended_with_a_timeout_that_names_its_last_event():
    """B6 hung for 119 seconds. Mutation killed: dropping the wall clock, so
    a producer that never returns holds the connection until the proxy kills
    it -- which the client sees as a network error, not a platform fault."""
    out = await _drain(
        guarantee_terminal(_gen(list(B6_FRAMES), hang=True), timeout_s=0.05)
    )
    assert _types(out)[-3:] == ["token", "error", "end"]
    body = json.loads(out[-2].split("data:", 1)[1].strip())
    assert "timeout" in body["message"].lower()
    assert "'start'" in body["message"]


@pytest.mark.asyncio
async def test_the_deadline_is_absolute_and_not_reset_by_traffic():
    """Heartbeats keep a hung turn's queue busy. A per-read timeout would
    therefore never fire, which is the bug the agent-level guard already
    documented.

    Mutation killed: computing the deadline inside the loop.
    """
    async def chatty():
        for f in B6_FRAMES:
            yield f
        while True:
            await asyncio.sleep(0.01)
            yield frame({"type": "heartbeat"})

    out = await _drain(guarantee_terminal(chatty(), timeout_s=0.12))
    assert _types(out)[-1] == "end"
    body = json.loads(out[-2].split("data:", 1)[1].strip())
    assert "timeout" in body["message"].lower()


@pytest.mark.asyncio
async def test_a_wait_that_times_out_is_reported_as_a_timeout_whatever_the_clock_says():
    """The event loop's timer and ``clock()`` are different readings of time:
    asyncio fires a timer up to one clock tick early, and on Windows that
    tick is ~15 ms, so a wait bounded by the deadline can expire while
    ``deadline - clock()`` is still positive. A frozen clock makes that gap
    permanent and the outcome independent of the OS and of machine load.

    Mutation killed: retrying the read after the wait timed out. The timed-out
    ``wait_for`` already cancelled the producer, so the retry sees
    StopAsyncIteration and the turn is misreported as "ended with no answer
    and no error" instead of a timeout naming its last event.
    """
    frozen = lambda: 1000.0  # noqa: E731 - remaining never reaches zero
    out = await _drain(guarantee_terminal(
        _gen(list(B6_FRAMES), hang=True), timeout_s=0.05, clock=frozen,
    ))
    assert _types(out)[-3:] == ["token", "error", "end"]
    body = json.loads(out[-2].split("data:", 1)[1].strip())
    assert "timeout" in body["message"].lower()
    assert "'start'" in body["message"]


@pytest.mark.asyncio
async def test_the_boundary_watchdog_sits_above_the_agent_s_own():
    """The relationship IS the design: on the agent path the agent's
    structured timeout must be what the user sees, and this net only fires
    for the paths that have none.

    Mutation killed: lowering the default to or below 240, which would make
    every slow agent turn report the generic boundary message instead of the
    agent's named stall phase.
    """
    assert DEFAULT_WATCHDOG_SECONDS > 240.0


def test_a_bad_env_value_falls_back_instead_of_raising(monkeypatch):
    """A watchdog that raises is worse than no watchdog."""
    for bad in ("", "abc", "-5", "0"):
        monkeypatch.setenv("CHAT_SSE_WATCHDOG_SECONDS", bad)
        assert watchdog_seconds() == DEFAULT_WATCHDOG_SECONDS, bad
    monkeypatch.setenv("CHAT_SSE_WATCHDOG_SECONDS", "12.5")
    assert watchdog_seconds() == 12.5


@pytest.mark.asyncio
async def test_the_kill_switch_passes_everything_through(monkeypatch):
    monkeypatch.setenv("CHAT_SSE_WATCHDOG", "0")
    out = await _drain(guarantee_terminal(_gen(list(B6_FRAMES))))
    assert out == B6_FRAMES


# -- reading the frames ----------------------------------------------------


@pytest.mark.parametrize("raw,want", [
    ('data: {"type": "end"}\n\n', "end"),
    ('data: {"type":"token","content":"x"}\n\n', "token"),
    ("data: [DONE]\n\n", None),
    (": keep-alive\n\n", None),
    ('data: not json\n\n', None),
    ('event: ping\ndata: {"type": "heartbeat"}\n\n', "heartbeat"),
    ('data: {"content": "no type"}\n\n', None),
    ('data: ["a list"]\n\n', None),
    ("", None),
])
def test_a_frame_is_read_tolerantly(raw, want):
    """A keep-alive comment or another producer's frame must not be mistaken
    for a terminal event. Mutation killed: `json.loads` without a guard, or
    treating any frame as terminal."""
    assert event_type(raw) == want


@pytest.mark.asyncio
async def test_an_unparseable_frame_does_not_count_as_an_ending():
    """Mutation killed: returning a truthy type for junk, which would let a
    malformed frame satisfy the terminal requirement."""
    out = await _drain(guarantee_terminal(_gen(["data: not json\n\n"])))
    assert _types(out)[-1] == "end"


# -- the wiring ------------------------------------------------------------


def test_both_streaming_endpoints_are_wrapped():
    """Mutation killed: guarding one endpoint and not the other. B6 was
    measured on /v1/chat/stream; the unversioned path has the same shape and
    the same silence.
    """
    import app.routers.chat as chat_mod

    source = open(chat_mod.__file__, encoding="utf-8").read()
    # Hat activation sits INSIDE the terminal guard, so the call is
    # guarantee_terminal(with_hat_signals(with_answer_exit(event_stream()), ...)). The old
    # contiguous substring no longer matches; both wrappers must still
    # wrap both endpoints. The exit check sits innermost, around the
    # event stream itself.
    assert source.count("guarantee_terminal(") == 2
    assert source.count("with_hat_signals(with_answer_exit(event_stream())") == 2
    assert source.count("StreamingResponse(") == 2


def test_the_watchdog_never_catches_baseexception():
    """``asyncio.CancelledError`` is a BaseException, so ``except Exception``
    cannot swallow it and a disconnected client cancels cleanly. Widening the
    handler to BaseException would answer an absent client AND eat the
    cancellation -- and the propagation test above cannot see the difference
    until it is too late.

    Mutation killed: `except BaseException` in guarantee_terminal.
    """
    import ast

    import app.routers.chat_watchdog as mod

    tree = ast.parse(open(mod.__file__, encoding="utf-8").read())
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "guarantee_terminal"
    )
    caught = {
        h.type.id for h in
        [h for t in ast.walk(fn) if isinstance(t, ast.Try) for h in t.handlers]
        if isinstance(h.type, ast.Name)
    }
    assert "BaseException" not in caught, caught
    assert "Exception" in caught


# -- an `end` with no answer (order 0230Z item 9) --------------------------
#: LANE3: a driver turn ended with an empty answer and no mode; LANE4: the
#: stream ended with no final answer after a tool call. The producer DID send
#: `end` -- so the terminal guard above passed it -- but nothing to read.
EMPTY_AFTER_TOOL = [
    frame({"type": "start", "agent": "driver"}),
    frame({"type": "tool_call", "name": "construction_calc"}),
    frame({"type": "tool_result", "name": "construction_calc", "ok": True}),
    frame({"type": "end", "iterations": 2, "sources": [{"title": "s"}]}),
]


@pytest.mark.asyncio
async def test_an_end_with_no_answer_gets_the_fallback_and_a_named_error():
    """Mutation killed: treating any `end` as a finished turn."""
    out = await _drain(guarantee_terminal(_gen(EMPTY_AFTER_TOOL), request_id="r1"))
    kinds = _types(out)
    assert kinds[-3:] == ["token", "error", "end"], kinds
    assert json.loads(out[-3].split("data:", 1)[1])["content"] == SILENT_TURN_FALLBACK
    error = json.loads(out[-2].split("data:", 1)[1])
    assert "tool_result" in error["message"] and error["request_id"] == "r1"
    assert out[-1] == EMPTY_AFTER_TOOL[-1], "the producer's own end (its sources) is kept"


@pytest.mark.asyncio
async def test_an_end_that_carries_the_answer_is_not_touched():
    """Mutation killed: ignoring `content` on `end`, which the UI renders in
    place of the streamed tokens."""
    src = EMPTY_AFTER_TOOL[:-1] + [frame({"type": "end", "content": "Volume 24 m3.", "sources": []})]
    out = await _drain(guarantee_terminal(_gen(src)))
    assert out == src


@pytest.mark.asyncio
async def test_an_end_after_tokens_is_not_touched():
    src = EMPTY_AFTER_TOOL[:-1] + [frame({"type": "token", "content": "24 m3"}), EMPTY_AFTER_TOOL[-1]]
    out = await _drain(guarantee_terminal(_gen(src)))
    assert out == src
