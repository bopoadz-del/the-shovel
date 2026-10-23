"""A short follow-up must retrieve with the conversation's context.

THE LIVE FAILURE THIS CLOSES (2026-08-02, operator's own session):

    user: "what is the backfilling specs"   -> correct, grounded answer
    user: "layers thickness ?"              -> "I do not have the backfilling
                                               layer thickness in this
                                               reference context."

Retrieval ran on the raw current message, so the second turn searched the
whole corpus on two words and returned generic front-matter — drawing
disclaimers, ITP requirements, Variation Order paperwork. The model then
correctly reported it had nothing. Meanwhile the answer was sitting in a
drawing ("SAND AS BACKFILL MATERIAL PLACED ... IN 150mm") that a
keyword-rich query surfaced on the first try.

The expansion is DETERMINISTIC (no LLM rewrite) and flag-gated:
RAG_FOLLOWUP_CONTEXT.

2026-08-17 — DEFAULT FLIPPED TO ON. This fix shipped dormant, and a dormant
fix fixes nothing: the same failure mode recurred across an entire operator
session of thin follow-ups ("Backfilling above foundations ?", "geotechnical
standards what does it contain ??", "Saudi buiding code"), where consecutive
turns about the same corpus contradicted each other because a two-or-three-word
query carries too little signal to retrieve on. The flag remains as a kill
switch (RAG_FOLLOWUP_CONTEXT=0), but the safe state is enabled.
"""
from __future__ import annotations

import pytest

from app.core.rag.inject import build_retrieval_query


def _hist(*turns):
    return [{"role": r, "content": c} for r, c in turns]


LIVE_HISTORY = _hist(
    ("user", "what is the backfilling specs"),
    ("assistant", "Backfill shall be placed in layers, moisturized and compacted "
                  "to not less than 95% Maximum Dry Density."),
)


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv("RAG_FOLLOWUP_CONTEXT", "1")


@pytest.fixture
def flag_off(monkeypatch):
    """The kill switch must be set EXPLICITLY now — unset means enabled."""
    monkeypatch.setenv("RAG_FOLLOWUP_CONTEXT", "0")


# ── the flag is a real kill switch ───────────────────────────────────────────

def test_the_kill_switch_restores_the_raw_query(flag_off):
    """RAG_FOLLOWUP_CONTEXT=0 must be byte-identical to the pre-feature path."""
    assert build_retrieval_query("layers thickness ?", LIVE_HISTORY) == "layers thickness ?"


def test_expansion_is_on_when_the_variable_is_unset(monkeypatch):
    """The whole point of the 2026-08-17 change: an operator who configures
    nothing gets the fix, not the failure it was written to close."""
    monkeypatch.delenv("RAG_FOLLOWUP_CONTEXT", raising=False)
    assert build_retrieval_query("layers thickness ?", LIVE_HISTORY) != "layers thickness ?"


def test_an_unrecognised_value_keeps_the_safe_state(monkeypatch):
    """A typo'd value must not silently revert retrieval to the broken
    behaviour — only an explicit falsy value disables."""
    monkeypatch.setenv("RAG_FOLLOWUP_CONTEXT", "maybe")
    assert build_retrieval_query("layers thickness ?", LIVE_HISTORY) != "layers thickness ?"


# ── the live failure ─────────────────────────────────────────────────────────

def test_the_live_followup_regains_its_subject(flag_on):
    """'layers thickness ?' must carry 'backfilling' into retrieval."""
    q = build_retrieval_query("layers thickness ?", LIVE_HISTORY)

    assert "backfilling" in q.lower()
    assert "layers thickness" in q          # the live message is preserved
    assert q != "layers thickness ?"


def test_assistant_turns_are_never_used_as_context(flag_on):
    """Only prior USER turns.

    Feeding a previous ANSWER back into retrieval is how one wrong answer
    entrenches itself for the rest of a conversation.
    """
    poisoned = _hist(
        ("user", "hi"),
        ("assistant", "The pier cement content is 385 kg/m3 CEM I 42.5N"),
    )
    q = build_retrieval_query("and the cover ?", poisoned)

    assert "385" not in q
    assert "CEM" not in q


# ── it must not touch queries that can stand alone ───────────────────────────

def test_a_rich_question_is_left_alone(flag_on):
    rich = "what is the compacted layer thickness for trench backfill under roads"
    assert build_retrieval_query(rich, LIVE_HISTORY) == rich


def test_no_history_is_a_noop(flag_on):
    assert build_retrieval_query("layers thickness ?", []) == "layers thickness ?"
    assert build_retrieval_query("layers thickness ?", None) == "layers thickness ?"


def test_history_with_no_user_turns_is_a_noop(flag_on):
    only_assistant = _hist(("assistant", "some earlier answer about concrete"))
    assert build_retrieval_query("layers thickness ?", only_assistant) == "layers thickness ?"


def test_stopword_only_history_does_not_produce_a_junk_query(flag_on):
    """Borrowing 'is it the' would make retrieval worse, not better."""
    junk = _hist(("user", "is it the ?"))
    assert build_retrieval_query("thickness ?", junk) == "thickness ?"


def test_the_same_message_repeated_is_not_doubled(flag_on):
    """A user re-sending the identical text must not self-concatenate."""
    repeated = _hist(("user", "layers thickness ?"))
    assert build_retrieval_query("layers thickness ?", repeated) == "layers thickness ?"


def test_expansion_is_bounded(flag_on):
    """A long conversation cannot grow the query without limit."""
    long_hist = _hist(*[("user", "excavation " * 200) for _ in range(6)])
    q = build_retrieval_query("thickness ?", long_hist)
    assert len(q) < 600


def test_only_the_two_most_recent_user_turns_are_borrowed(flag_on):
    hist = _hist(
        ("user", "tell me about asphalt paving temperature"),
        ("user", "now the drainage manhole spacing"),
        ("user", "and the backfilling compaction"),
    )
    q = build_retrieval_query("thickness ?", hist).lower()

    assert "backfilling" in q and "drainage" in q
    assert "asphalt" not in q       # the third-oldest turn is dropped
