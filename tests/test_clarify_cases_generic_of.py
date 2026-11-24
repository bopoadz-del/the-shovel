"""A generic value word with an ``of`` object names its thing in the object.

"What is the quantity of <thing>?" asks about <thing>, the way "what class
of <thing>" does. Read as a bare generic ask ("what is the value?") it
narrowed nothing, so every figure in the evidence became an option of the
clarifying question, whatever thing it was about.
"""
from __future__ import annotations

import pytest

from app.agents.clarify_cases import (
    Case,
    _question_asks,
    _subject_past_generic,
    clarifying_question,
)


def _cases(*rows: tuple[str, str, str]) -> list[Case]:
    return [Case(subject=s, value=v, source=src) for s, v, src in rows]


@pytest.mark.parametrize(
    "question,head,qualifier",
    [
        ("what is the billed quantity of glazed wall tiling?", "tiling", "glazed"),
        ("what is the number of bored piles under the raft?", "piles", "bored"),
        ("what is the value of the amount of granular bedding?", "bedding", "granular"),
    ],
)
def test_generic_word_with_of_object_names_the_object(question, head, qualifier):
    asks = _question_asks(question)
    assert asks is not None
    assert head in asks[1]
    assert qualifier in asks[2]


@pytest.mark.parametrize(
    "question",
    [
        "what is the value?",
        "what is the total quantity?",
        "what is the quantity of?",
    ],
)
def test_generic_word_without_of_object_stays_generic(question):
    asks = _question_asks(question)
    assert asks is None or asks[1] == set()


@pytest.mark.parametrize(
    "phrase,expected",
    [
        ("billed quantity of glazed wall tiling", "glazed wall tiling"),
        ("amount of the retention money", "the retention money"),
        ("slab thickness", "slab thickness"),
        ("cover to reinforcement", "cover to reinforcement"),
    ],
)
def test_subject_past_generic(phrase, expected):
    assert _subject_past_generic(phrase) == expected


@pytest.mark.parametrize(
    "question,rows",
    [
        (
            "what is the billed quantity of glazed wall tiling?",
            (
                ("bulk excavation", "500 m3", "bill.xlsx"),
                ("Pool with Jacuzzi", "2 No", "schedule.xlsx"),
            ),
        ),
        (
            "what is the number of bored piles under the raft?",
            (
                ("precast kerbs", "320 m", "bill.pdf"),
                ("chilled water pipe", "45 m", "mep.xlsx"),
            ),
        ),
    ],
)
def test_cases_about_other_things_do_not_become_options(question, rows):
    """Figures about other things are not alternatives to the asked one.

    Before the object was read, the ask narrowed nothing and these cases
    became a clarifying question about unrelated rows."""
    draft = f"The figure is {rows[0][1]}."
    assert clarifying_question(question, draft, _cases(*rows)) is None


@pytest.mark.parametrize(
    "question,rows,other",
    [
        (
            "what is the quantity of bedding?",
            (
                ("granular pipe bedding", "120 m3", "bill.xlsx"),
                ("concrete pipe bedding", "40 m3", "bill.xlsx"),
            ),
            ("bulk excavation", "500 m3", "bill.xlsx"),
        ),
        (
            "what is the number of piles?",
            (
                ("bored piles", "84 No", "piling.pdf"),
                ("sheet piles", "210 No", "temporary works.pdf"),
            ),
            ("precast kerbs", "320 m", "bill.pdf"),
        ),
    ],
)
def test_cases_about_the_asked_thing_still_ask(question, rows, other):
    """Two different cases of the asked thing still ask which one, and the
    question lists only those two, not the row about another thing."""
    draft = f"The figure is {rows[0][1]}."
    q = clarifying_question(question, draft, _cases(*rows, other))
    assert q is not None
    assert rows[0][1] in q and rows[1][1] in q
    assert other[1] not in q
