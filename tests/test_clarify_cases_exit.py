"""The clarify-before-choosing exit WIRING (the 15:03Z "all sources" fold).

``clarify_cases.clarifying_question`` is the pure detector; these tests pin the
caller that turns a silently-picking draft into the clarifying question at the
answer exit (``runtime._apply_clarify_cases``) and records the state on the
turn (``runtime._record_clarifying_question``). Structural only: the passages
are synthetic and name no real question, document, structure or value. Every
decision is parametrized with two distinct inputs.
"""

from __future__ import annotations

import pytest

from app.agents import answer_exit
from app.agents.runtime import _apply_clarify_cases, _record_clarifying_question


@pytest.fixture
def turn():
    turn, token = answer_exit.begin_turn()
    yield turn
    answer_exit.reset_turn(token)


def _rag(*chunks: tuple[str, str]) -> dict:
    """A retrieval system message holding one marked chunk per (doc, text)."""
    parts = []
    for i, (doc, text) in enumerate(chunks):
        parts.append(f"[doc_id={doc} chunk={i} src={doc}.pdf]{text}")
    return {"content": "\n".join(parts)}


def _messages(question: str) -> list[dict]:
    return [{"role": "user", "content": question}]


@pytest.mark.parametrize(
    "doc_a,doc_b",
    [
        ("The concrete cover is 40 mm.", "The backfill depth is 300 mm."),
        ("The pipe cover is 60 mm.", "The slab thickness is 150 mm."),
    ],
)
def test_kill_switch_passes_through(monkeypatch, turn, doc_a, doc_b):
    """OFF leaves the draft byte-for-byte and records no turn state, whatever
    the evidence holds."""
    monkeypatch.setenv("CLARIFY_CASES", "0")
    rag = _rag(("a", doc_a), ("b", doc_b))
    text = "The value is 40 mm."
    assert _apply_clarify_cases(text, rag, _messages("what is the required value?")) == text
    assert turn.questions == []
    assert turn.awaiting_inputs == []


@pytest.mark.parametrize(
    "doc_a,doc_b,value_a,value_b",
    [
        ("The concrete cover is 40 mm.", "The backfill depth is 300 mm.", "40 mm", "300 mm"),
        ("The pipe cover is 60 mm.", "The slab thickness is 150 mm.", "60 mm", "150 mm"),
    ],
)
def test_two_differing_cases_silent_pick_asks(turn, doc_a, doc_b, value_a, value_b):
    rag = _rag(("a", doc_a), ("b", doc_b))
    text = f"The required value is {value_a}."
    out = _apply_clarify_cases(text, rag, _messages("what is the required value?"))
    assert out is not None
    assert out != text
    assert "Which one is meant?" in out
    assert value_a in out and value_b in out
    assert out in turn.questions
    assert "which_case" in turn.awaiting_inputs


@pytest.mark.parametrize(
    "question",
    [
        "what is the concrete cover?",   # names the case; only that case survives
        "what is the backfill depth?",   # names the other case
    ],
)
def test_question_names_case_no_ask(turn, question):
    rag = _rag(("a", "The concrete cover is 40 mm."), ("b", "The backfill depth is 300 mm."))
    text = "The value is 40 mm."
    assert _apply_clarify_cases(text, rag, _messages(question)) == text
    assert turn.questions == []
    assert turn.awaiting_inputs == []


@pytest.mark.parametrize(
    "doc_a,question",
    [
        ("The concrete cover is 40 mm.", "what is the concrete cover?"),
        ("The backfill depth is 300 mm.", "what is the backfill depth?"),
    ],
)
def test_single_value_no_ask(turn, doc_a, question):
    rag = _rag(("a", doc_a))
    text = "The value is 40 mm."
    assert _apply_clarify_cases(text, rag, _messages(question)) == text
    assert turn.awaiting_inputs == []


@pytest.mark.parametrize(
    "rag",
    [
        {},
        {"content": ""},
    ],
)
def test_no_evidence_passes_through(turn, rag):
    text = "The value is 40 mm."
    assert _apply_clarify_cases(text, rag, _messages("what is the value?")) == text
    assert turn.awaiting_inputs == []


@pytest.mark.parametrize(
    "messages",
    [
        None,
        [],
    ],
)
def test_no_question_passes_through(turn, messages):
    rag = _rag(("a", "The concrete cover is 40 mm."), ("b", "The backfill depth is 300 mm."))
    text = "The value is 40 mm."
    assert _apply_clarify_cases(text, rag, messages) == text
    assert turn.awaiting_inputs == []


@pytest.mark.parametrize(
    "text",
    [
        "The concrete cover is 40 mm and the backfill depth is 300 mm.",
        "Concrete cover is 40 mm; backfill depth is 300 mm.",
    ],
)
def test_draft_commits_to_both_no_ask(turn, text):
    """A draft already answering both structures side by side is never a
    silent pick, so the exit leaves it (the both-values / side-by-side duty)."""
    rag = _rag(("a", "The concrete cover is 40 mm."), ("b", "The backfill depth is 300 mm."))
    assert _apply_clarify_cases(text, rag, _messages("what is the required value?")) == text
    assert turn.awaiting_inputs == []


@pytest.mark.parametrize(
    "question",
    [
        "Which one is meant?",
        "Retrieved context supports more than one answer.",
    ],
)
def test_record_clarifying_question_with_no_turn_is_silent(question):
    """With no turn open, recording the question does nothing rather than
    falling through to ``None.questions`` and raising."""
    answer_exit._TURN.set(None)
    _record_clarifying_question(question)
    assert answer_exit.current_turn() is None
