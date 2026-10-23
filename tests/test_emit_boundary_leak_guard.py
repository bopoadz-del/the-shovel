"""No path may ship the platform's own context, including one I cannot name.

#457 put the context-leak check in the streamed-synthesis branch and in
_sanitize_final_text, and E1 came back clean on the deploy. It recurred once
on 87c7996 -- same shape, same wbs_id / retrieval brief / customer Drive
paths -- while the next run of the same question was clean, and the run after
that was clean too.

Intermittent means a path neither guard covers. Sampling seven-minute live
turns to find which one is a poor trade, so this sits where every event the
client sees passes through: chat_stream wraps _chat_stream_impl, and one
check at that seam cannot be bypassed by any branch inside it.
"""
from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from app.agents.runtime import (
    _TOOL_FORMAT_FALLBACK,
    Agent,
    _EmitLeakGuard,
    answer_contains_routing_preamble,
)

BRIEF = "AUTHORITATIVE REFERENCE CONTEXT \u2014 the material below was retrieved"
TELEMETRY = "[doc_id=d0c00002 chunk=11 score=2.199 src=Contract_Data.pdf]"


def _tokens(events):
    return "".join(
        e.get("content", "") for e in events
        if isinstance(e, dict) and e.get("type") == "token"
    )


def _end(events):
    for e in reversed(events):
        if isinstance(e, dict) and e.get("type") == "end":
            return e
    return None


# -- the guard on its own --------------------------------------------------


def test_a_clean_turn_passes_through_untouched():
    """Identity on the common path: same objects, same order, nothing added."""
    guard = _EmitLeakGuard()
    events = [
        {"type": "start", "agent": "pa"},
        {"type": "token", "content": "Delay Damages are 0.1% of the "},
        {"type": "token", "content": "Contract Price per calendar day."},
        {"type": "end", "content": "Delay Damages are 0.1%..."},
    ]
    out = [guard.check(e) for e in events]
    assert out == events
    assert guard.tripped is False


def test_a_marker_split_across_two_tokens_is_still_caught():
    """The reason this accumulates. Neither chunk contains the marker; the
    pair does.

    Mutation killed: checking each event in isolation.
    """
    guard = _EmitLeakGuard()
    assert guard.check({"type": "token", "content": '{"brief": "AUTHORI'}) is not None
    assert guard.check({"type": "token", "content": 'TATIVE REFERENCE CONTEXT'}) is None
    assert guard.tripped is True


def test_everything_after_the_trip_is_dropped():
    guard = _EmitLeakGuard()
    guard.check({"type": "token", "content": BRIEF})
    for content in (TELEMETRY, "X:\\Example Drive\\Contract Docs", "more of the brief"):
        assert guard.check({"type": "token", "content": content}) is None


def test_the_end_event_carries_the_fallback_not_the_leak():
    """Suppressing tokens alone would leave the turn blank and the leak still
    in the persisted `end` content."""
    guard = _EmitLeakGuard()
    guard.check({"type": "token", "content": BRIEF})
    end = guard.check({"type": "end", "content": BRIEF + TELEMETRY, "iterations": 3})
    assert end["content"] == _TOOL_FORMAT_FALLBACK
    assert end["iterations"] == 3, "the rest of the end event survives"


def test_a_leak_that_appears_only_in_the_end_event_is_caught():
    """Some paths never stream: they compute the answer and emit it once."""
    guard = _EmitLeakGuard()
    end = guard.check({"type": "end", "content": BRIEF})
    assert end["content"] == _TOOL_FORMAT_FALLBACK
    assert guard.tripped is True


def test_telemetry_alone_trips_it():
    guard = _EmitLeakGuard()
    assert guard.check({"type": "token", "content": TELEMETRY}) is None


@pytest.mark.parametrize(
    "event",
    [
        {"type": "start", "agent": "pa"},
        {"type": "tool_call", "tool": "boq_processor"},
        {"type": "error", "message": "boom"},
        {"type": "heartbeat"},
        {"type": "token"},
        {"type": "token", "content": ""},
        {"type": "token", "content": None},
        "not-a-dict",
    ],
)
def test_non_content_events_are_never_touched(event):
    """The guard reads token/end content and nothing else -- a tool_call whose
    args mention doc_id is not a leak to the user."""
    assert _EmitLeakGuard().check(event) is event


def test_a9_routing_preamble_is_stripped_and_jacobs_survives():
    """Live e24aee4: graft leak must not nuke the appointed firm."""
    leaked = (
        "The Engineer is APPOINTMENT — an excerpt below names the Engineer. "
        "That IS the answer. State the appointed firm. Do not say "
        "the identity is absent.\n\n"
        "EXAMPLAR (EXCON Example Limited)"
    )
    guard = _EmitLeakGuard()
    end = guard.check({"type": "end", "content": leaked, "iterations": 1})
    assert end["content"]
    assert "EXAMPLAR" in end["content"]
    assert "EXCON" in end["content"]
    assert not answer_contains_routing_preamble(end["content"])
    assert "That IS the answer" not in end["content"]
    assert "INTERNAL GUIDANCE" not in end["content"]
    assert "State the appointed firm" not in end["content"]
    assert guard.tripped is False


def test_a_real_answer_is_never_suppressed():
    guard = _EmitLeakGuard()
    for chunk in (
        "Under AB-2023-101, the daily rate for Delay Damages is 0.1% of the ",
        "Contract Price per calendar day (source: Contract_Data.pdf, chunks 10 & 11). ",
        "The retrieved excerpts do not contain the Contract Price in SAR.",
    ):
        assert guard.check({"type": "token", "content": chunk}) is not None
    assert guard.tripped is False


# -- and at the seam it actually sits on -----------------------------------


def _fake_impl(events):
    async def impl(self, *a, **kw):
        for e in events:
            yield e
    return impl


def _drain(agen):
    async def go():
        return [ev async for ev in agen]
    return asyncio.run(go())


def _run(events):
    agent = Agent(name="project-assistant", description="pa",
                  system_prompt="x", allowed_blocks=["construction"])
    with patch.object(Agent, "_chat_stream_impl", _fake_impl(events)):
        return _drain(agent.chat_stream(user_message="q", history=[]))


def test_a_leak_from_any_inner_path_is_stopped_at_the_boundary():
    """The point of the whole file: _chat_stream_impl is replaced wholesale,
    so this passes regardless of which branch inside it produced the leak --
    including the one I have not identified.

    Mutation killed: removing the guard from the chat_stream relay.
    """
    events = _run([
        {"type": "start", "agent": "pa"},
        {"type": "token", "content": '{"status": "success", "wbs_id": "wbs-c199246d", "brief": "'},
        {"type": "token", "content": BRIEF},
        {"type": "token", "content": TELEMETRY},
        {"type": "token", "content": "X:\\Example Drive\\Master Folder\\Contract Docs"},
        {"type": "end", "content": "the whole leak", "iterations": 2},
    ])
    text = _tokens(events)
    assert "AUTHORITATIVE" not in text
    assert "doc_id=" not in text
    assert "Example Drive" not in text
    end = _end(events)
    assert end is not None and end["content"] == _TOOL_FORMAT_FALLBACK


def test_a_clean_turn_through_the_boundary_is_unchanged():
    answer = "Delay Damages are 0.1% of the Contract Price per calendar day."
    events = _run([
        {"type": "start", "agent": "pa"},
        {"type": "token", "content": answer},
        {"type": "end", "content": answer, "iterations": 1},
    ])
    assert _tokens(events) == answer
    assert _end(events)["content"] == answer


def test_only_answer_text_is_inspected():
    """The guard reads token/end content and nothing else, by TYPE rather
    than by whether a "content" key happens to exist.

    A tool_result that quotes the retrieval brief is internal plumbing the
    client already receives; suppressing it would be this guard deciding
    what other event types may say. Any future event carrying `content` --
    a draft, a preview -- must pass through untouched rather than be
    silently filtered by a check written for answer text.

    Mutation killed: dropping the type check and relying on non-answer
    events happening not to have a "content" key today.
    """
    guard = _EmitLeakGuard()
    plumbing = {"type": "tool_result", "tool": "search", "content": BRIEF}
    assert guard.check(plumbing) is plumbing
    assert guard.tripped is False
    # And a real answer after it is still judged on its own merits.
    assert guard.check({"type": "token", "content": "The rate is 0.1%."}) is not None
    assert guard.check({"type": "token", "content": BRIEF}) is None
