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
    _before_than,
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
        # A generic head followed by a preposition other than "of" names
        # no object: the phrase stays whole.
        ("quantity in the bill", "quantity in the bill"),
        ("amount for retention money", "amount for retention money"),
        # A trailing "of" with nothing after it is no object either.
        ("billed quantity of", "billed quantity of"),
        ("amount of", "amount of"),
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


# A comparison ends the asked subject. "the quantity of joints wider than
# 20 mm" asks about joints; read past the comparison, the head became the
# comparative ("wider"), which no case names, so every case was dropped and
# the clarifier went silent. A pronoun object ("the quantity of them") is
# not a thing either: it narrows nothing, as a bare generic ask does.


@pytest.mark.parametrize(
    "question,head,qualifier,comparative",
    [
        ("what is the quantity of movement joints wider than 20 mm?",
         "joints", "movement", "wider"),
        ("what is the number of bored piles longer or shorter than 18 m?",
         "piles", "bored", "longer"),
    ],
)
def test_comparison_ends_the_subject(question, head, qualifier, comparative):
    asks = _question_asks(question)
    assert asks is not None
    assert asks[1] == {head}
    assert qualifier in asks[2]
    assert comparative not in asks[1] | asks[2]


@pytest.mark.parametrize(
    "question,head,kept",
    [
        # A comparative with no "than" is an ordinary modifier of the thing.
        ("what is the quantity of thicker screed?", "screed", "thicker"),
        # "or" with a third word that is not "than" is a list, not a pair.
        ("what is the number of piles or pile caps?", "caps", "piles"),
    ],
)
def test_no_than_no_cut(question, head, kept):
    asks = _question_asks(question)
    assert asks is not None
    assert asks[1] == {head}
    assert kept in asks[2]


@pytest.mark.parametrize(
    "question",
    ["what is the quantity of them?", "what is the amount of us?"],
)
def test_pronoun_object_narrows_nothing(question):
    asks = _question_asks(question)
    assert asks is None or asks[1] == set()


@pytest.mark.parametrize(
    "words,i,expected",
    [
        (["joints", "wider", "than", "20"], 1, True),
        (["piles", "longer", "or", "shorter", "than"], 1, True),
        (["piles", "longer", "and", "shorter", "than"], 1, True),
        (["joints", "wider", "than"], 0, False),
        (["piles", "longer", "nor", "shorter", "than"], 1, False),
        (["piles", "longer", "or", "shorter"], 1, False),
        (["than"], 0, False),
        ([], 0, False),
    ],
)
def test_before_than(words, i, expected):
    assert _before_than(words, i) is expected


@pytest.mark.parametrize(
    "question,rows,other",
    [
        (
            "what is the quantity of movement joints wider than 20 mm?",
            (
                ("movement joints in slab", "140 m", "bill.xlsx"),
                ("movement joints in screed", "65 m", "finishes.pdf"),
            ),
            ("precast kerbs", "320 m", "bill.pdf"),
        ),
        (
            "what is the number of piles longer or shorter than 18 m?",
            (
                ("bored piles", "84 No", "piling.pdf"),
                ("sheet piles", "210 No", "temporary works.pdf"),
            ),
            ("chilled water pipe", "45 m", "mep.xlsx"),
        ),
    ],
)
def test_comparison_ask_still_asks_between_the_asked_things(question, rows, other):
    """With the comparison read as the subject, no case matched and the
    question was dropped; now the two cases of the asked thing are offered
    and the row about another thing is not."""
    draft = f"The figure is {rows[0][1]}."
    q = clarifying_question(question, draft, _cases(*rows, other))
    assert q is not None
    assert rows[0][1] in q and rows[1][1] in q
    assert other[1] not in q
