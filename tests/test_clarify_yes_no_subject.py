"""A yes/no ask names its thing, so the clarifier narrows to it.

"is the <thing> higher than <figure>?" used to read no subject at all: the
subject readers only knew "what/which" openings. With a measure noun in the
ask, every measure in the handed evidence then counted as an alternative
answer, and the clarifying question listed all of them. Each test builds its
own input; every decision is checked from both sides.
"""
from __future__ import annotations

import pytest

from app.agents.clarify_cases import (
    Case,
    _question_asks,
    _yes_no_subject,
    clarifying_question,
)


def _cases(*rows: tuple[str, str, str]) -> list[Case]:
    return [Case(subject=s, value=v, source=src) for s, v, src in rows]


# --- the reader --------------------------------------------------------

@pytest.mark.parametrize(
    ("ask", "subject"),
    [
        ("is the hire rate of tower cranes above 9 eur/day", "hire rate of tower cranes"),
        ("are the pump pressures lower than the limit", "pump pressures"),
    ],
)
def test_subject_ends_at_a_comparison_word(ask, subject):
    assert _yes_no_subject(ask) == subject


@pytest.mark.parametrize(
    ("ask", "subject"),
    [
        ("is the screed depth 65 mm", "screed depth"),
        ("was a slab thickness 0.25 m", "slab thickness"),
    ],
)
def test_subject_ends_at_a_figure(ask, subject):
    assert _yes_no_subject(ask) == subject


def test_not_a_yes_no_opening_reads_nothing():
    assert _yes_no_subject("what is the screed depth") is None
    assert _yes_no_subject("list the pump pressures") is None


@pytest.mark.parametrize(
    ("ask", "head", "qualifier"),
    [
        ("Is the hire rate of tower cranes higher or lower than 9 EUR/day?", "rate", "hire"),
        ("Is the screed depth more than 65 mm?", "depth", "screed"),
    ],
)
def test_yes_no_ask_gets_a_head_and_qualifiers(ask, head, qualifier):
    kinds, heads, quals = _question_asks(ask)
    assert "measure" in kinds
    assert heads == {head}
    assert qualifier in quals


# --- the decision: ask, or stand as written ------------------------------

def test_unrelated_measures_are_not_alternatives_to_a_yes_no_ask():
    # The evidence holds many measures of OTHER things; the draft commits
    # to one figure. None of them is a hire rate, so there is nothing to
    # choose between and the answer stands.
    cases = _cases(
        ("storage room temperature", "21 °C", "code_a.pdf"),
        ("waste collection allowance", "9 kg", "code_a.pdf"),
        ("anchor bolt diameter", "16 mm", "rates_b.md"),
        ("design life", "40 years", "spec_c.pdf"),
        ("gas leakage", "0.2 %", "spec_c.pdf"),
    )
    draft = "The figure found is 9 kg, which is lower."
    assert clarifying_question(
        "Is the hire rate of tower cranes higher or lower than 9 EUR/day?",
        draft, cases,
    ) is None


def test_same_thing_in_two_places_still_asks_on_a_yes_no_ask():
    # Two different structures each state a screed depth; the draft picked
    # one. The ask names "screed depth", which both cases are, so the
    # clarifier still offers both.
    cases = _cases(
        ("podium screed depth", "65 mm", "spec_a.pdf"),
        ("roof screed depth", "90 mm", "spec_b.pdf"),
    )
    q = clarifying_question(
        "Is the screed depth more than 50 mm?",
        "The screed depth is 65 mm.",
        cases,
    )
    assert q is not None
    assert "65 mm" in q and "90 mm" in q


def test_yes_no_ask_drops_a_different_thing_but_keeps_the_named_one():
    cases = _cases(
        ("podium screed depth", "65 mm", "spec_a.pdf"),
        ("roof screed depth", "90 mm", "spec_b.pdf"),
        ("trench depth", "1.8 m", "spec_c.pdf"),
    )
    q = clarifying_question(
        "Is the screed depth more than 50 mm?",
        "The screed depth is 65 mm.",
        cases,
    )
    assert q is not None
    assert "1.8 m" not in q
