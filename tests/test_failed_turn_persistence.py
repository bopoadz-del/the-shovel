"""A failed turn must leave a trace in the conversation, not a silent gap.

LIVE 2026-08-02, operator's own thread (project Chad):

    user:      "the client project package 1"
    assistant: <nothing — ever>
    user:      "Backfilling specification"

Two consecutive user turns with no reply between them. Not a rendering
glitch: `append_message(conversation_id, "user", ...)` runs BEFORE the LLM
call, while the assistant message is appended only on the SUCCESS paths. The
UI's error bubble is local React state, so it vanishes on reload — leaving a
question that looks ignored, with nothing to indicate a failure happened.

The feature sweep reproduced the mechanism independently: `document_metadata`
returned `kimi HTTP 400: "tokenization failed"` -> error event -> nothing
persisted.

This is worse than a wrong answer. A wrong answer can be challenged; a
missing one reads as the platform having chosen not to respond.
"""
from __future__ import annotations

import pytest

from app.core import agent_memory
from app.agents import runtime as rt


@pytest.fixture
def captured(monkeypatch):
    """Record every append_message call instead of touching a database."""
    calls: list[tuple[str, str, str]] = []
    # runtime.py imports agent_memory at FUNCTION scope, so patch the SOURCE
    # module — patching `rt.agent_memory` fails with AttributeError because
    # that name does not exist on the runtime module.
    monkeypatch.setattr(
        agent_memory,
        "append_message",
        lambda cid, role, content: calls.append((cid, role, content)),
    )
    return calls


def test_failed_turn_is_recorded(captured):
    rt._persist_failed_turn("conv_1")

    assert len(captured) == 1
    cid, role, content = captured[0]
    assert cid == "conv_1"
    assert role == "assistant"
    assert content == rt._FAILED_TURN_MESSAGE


def test_no_conversation_id_is_a_noop(captured):
    """Anonymous / unsaved turns have nothing to append to."""
    rt._persist_failed_turn(None)
    rt._persist_failed_turn("")

    assert captured == []


def test_bookkeeping_failure_never_masks_the_original_error(monkeypatch):
    """If persistence itself dies, the caller must still report the real error.

    The caller is mid-failure already; raising here would replace a useful
    provider message with a database stacktrace.
    """
    def _boom(*_a, **_k):
        raise RuntimeError("db is down")

    monkeypatch.setattr(agent_memory, "append_message", _boom)

    rt._persist_failed_turn("conv_1")  # must not raise


def test_marker_reads_as_a_failure_not_as_content():
    """It re-enters the model's context next turn, so it must not look like
    an answer the model could continue from."""
    msg = rt._FAILED_TURN_MESSAGE

    assert msg.startswith("[") and msg.endswith("]")
    assert "failed" in msg.lower()
    assert "nothing was generated" in msg.lower()
    # short enough that it cannot dominate the next turn's context
    assert len(msg) < 200


def test_marker_is_not_scrubbed_away_by_history_cleaning():
    """_scrub_history strips hallucinated WBS/BOQ tables. The failure marker
    must survive it, or reloaded history loses the trace again."""
    turns = [
        {"role": "user", "content": "the client project package 1"},
        {"role": "assistant", "content": rt._FAILED_TURN_MESSAGE},
    ]

    scrubbed = rt._scrub_history(turns)

    assert scrubbed[1]["content"] == rt._FAILED_TURN_MESSAGE
