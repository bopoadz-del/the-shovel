"""Tests for the clarify-before-choosing exit decision (PR 2 part b).

Each of the seven bounds from the design note is a test. The detector is
pure, so these run against recorded strings, not a retriever mock.
"""

from __future__ import annotations

from app.agents.clarify_cases import (
    Case,
    clarifying_question,
    enabled,
)


def _cases(*rows: tuple[str, str, str]) -> list[Case]:
    return [Case(subject=s, value=v, source=src) for s, v, src in rows]


def test_enabled_by_default(monkeypatch):
    monkeypatch.delenv("CLARIFY_CASES", raising=False)
    assert enabled() is True


def test_kill_switch_off(monkeypatch):
    monkeypatch.setenv("CLARIFY_CASES", "0")
    assert enabled() is False
    q = clarifying_question(
        "what is the cover?",
        "The cover is 40 mm.",
        _cases(("concrete cover", "40 mm", "spec.pdf")),
    )
    assert q is None


# Bound 2 -- same value in every case: answer, never ask.
def test_bound2_same_value_no_ask():
    q = clarifying_question(
        "what is the cover?",
        "The cover is 40 mm.",
        _cases(
            ("concrete cover", "40 mm", "a.pdf"),
            ("concrete cover", "40 mm", "b.pdf"),
        ),
    )
    assert q is None


# Bound 3 -- the question names its case: no ask.
def test_bound3_question_names_case():
    q = clarifying_question(
        "what is the concrete cover to reinforcement?",
        "The cover is 40 mm.",
        _cases(
            ("concrete cover to reinforcement", "40 mm", "a.pdf"),
            ("concrete cover to foundation", "50 mm", "b.pdf"),
        ),
    )
    assert q is None


# Bound 3 -- the question names the value: no ask.
def test_bound3_question_names_value():
    q = clarifying_question(
        "is the cover 40 mm?",
        "The cover is 40 mm.",
        _cases(
            ("concrete cover", "40 mm", "a.pdf"),
            ("backfill", "98%", "b.pdf"),
        ),
    )
    assert q is None


# Bound 4 -- documents state one thing plainly (one case): no invented ask.
def test_bound4_single_case_no_ask():
    q = clarifying_question(
        "what is the cover?",
        "The cover is 40 mm.",
        _cases(("concrete cover", "40 mm", "a.pdf")),
    )
    assert q is None


# Bound 5 -- the question lists only the cases found, each with its source.
def test_bound5_lists_only_found_cases():
    q = clarifying_question(
        "what is the figure?",
        "The figure is 40 mm.",
        _cases(
            ("concrete cover", "40 mm", "a.pdf"),
            ("backfill depth", "300 mm", "b.pdf"),
        ),
    )
    assert q is not None
    assert "40 mm" in q and "a.pdf" in q
    assert "300 mm" in q and "b.pdf" in q
    assert "50 mm" not in q  # not invented


# Bound 6 -- one question, never a chain; "all" is the caller's next turn.
def test_bound6_one_question_not_a_chain():
    q = clarifying_question(
        "what is the figure?",
        "The figure is 40 mm.",
        _cases(
            ("concrete cover", "40 mm", "a.pdf"),
            ("backfill depth", "300 mm", "b.pdf"),
        ),
    )
    assert q is not None
    assert q.count("Which one is meant?") == 1
    assert q.count("?") == 1


# Bound 7 -- a merely broad question with one value is answered, not bounced.
def test_bound7_broad_question_answered():
    q = clarifying_question(
        "summarise the specification figures",
        "The specification gives a cover of 40 mm.",
        _cases(
            ("concrete cover", "40 mm", "a.pdf"),
            ("concrete cover", "40 mm", "b.pdf"),
        ),
    )
    assert q is None


# Case 2 -- different subjects, different values, question names none: ASK.
def test_case2_different_subjects_asks():
    q = clarifying_question(
        "what is the required value?",
        "The required value is 40 mm.",
        _cases(
            ("concrete cover", "40 mm", "a.pdf"),
            ("backfill depth", "300 mm", "b.pdf"),
        ),
    )
    assert q is not None
    assert "Which one is meant?" in q


# Case 3 -- same subject, different values: the both-values statement (#453),
# not this mechanism.
def test_case3_same_subject_different_values_no_ask():
    q = clarifying_question(
        "what is the concrete cover?",
        "The concrete cover is 40 mm.",
        _cases(
            ("concrete cover", "40 mm", "a.pdf"),
            ("concrete cover", "50 mm", "b.pdf"),
        ),
    )
    assert q is None


# The draft must have silently picked exactly ONE value to trigger.
def test_draft_commits_to_none_no_ask():
    q = clarifying_question(
        "what is the figure?",
        "See the documents.",
        _cases(
            ("concrete cover", "40 mm", "a.pdf"),
            ("backfill depth", "300 mm", "b.pdf"),
        ),
    )
    assert q is None


def test_draft_commits_to_both_no_ask():
    q = clarifying_question(
        "what is the figure?",
        "The figures are 40 mm and 300 mm.",
        _cases(
            ("concrete cover", "40 mm", "a.pdf"),
            ("backfill depth", "300 mm", "b.pdf"),
        ),
    )
    assert q is None


# Clause ids compare as strings, not floats (8.8.1 vs 8.8.2 is a conflict,
# not the same number).
def test_clause_ids_are_strings():
    q = clarifying_question(
        "which clause applies?",
        "Clause 8.8.1 applies.",
        _cases(
            ("design", "8.8.1", "a.pdf"),
            ("construction", "8.8.2", "b.pdf"),
        ),
    )
    assert q is not None
    assert "8.8.1" in q and "8.8.2" in q


def test_clause_ids_same_value_no_ask():
    q = clarifying_question(
        "which clause applies?",
        "Clause 8.8.1 applies.",
        _cases(
            ("design", "8.8.1", "a.pdf"),
            ("construction", "8.8.1", "b.pdf"),
        ),
    )
    assert q is None


# Same subject conservatively; two different structures never collapse.
def test_different_structures_never_collapse():
    q = clarifying_question(
        "what is the cover?",
        "The cover is 40 mm.",
        _cases(
            ("concrete cover to reinforcement", "40 mm", "a.pdf"),
            ("concrete cover to foundation", "50 mm", "b.pdf"),
        ),
    )
    assert q is not None
