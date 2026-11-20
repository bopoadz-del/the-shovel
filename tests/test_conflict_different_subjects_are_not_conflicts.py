"""Two passages stating values for DIFFERENT subjects are never a conflict.

The owner's ruling: a value for blinding concrete and a value for structural
concrete are values of two different things, so the answer never says "the
documents differ"; the turn asks which case is meant. This module builds its
own passages for each test (no fixture file, no case id), and pins the other
half too: one subject worded differently must still get the both-values
statement.
"""

from __future__ import annotations

import pytest

from app.agents.clarify_cases import clarifying_question, cases_from_passages
from app.agents.evidence_conflicts import detect


def _rows(*pairs: tuple[str, str]) -> list[dict[str, str]]:
    return [{"doc_name": doc, "page_or_section": "NOTES", "snippet": text} for doc, text in pairs]


# --- the conflict detector: different subjects, never a conflict -----------

@pytest.mark.parametrize(
    "a, b, question",
    [
        # The qualifying word sits next to the asked word in each passage.
        ("Minimum cover to blinding concrete shall be 50 mm.",
         "Minimum cover to structural concrete shall be 40 mm.",
         "What is the minimum concrete cover?"),
        # A different asked word, the same shape.
        ("The blinding layer thickness is 75 mm.",
         "The foundation layer thickness is 300 mm.",
         "What layer thickness applies?"),
    ],
)
def test_two_subjects_are_not_a_conflict(a, b, question):
    assert detect(f"One of them applies ({a.split()[0]}).", _rows(("Drawings", a), ("Spec", b)),
                  question) == []


def test_one_subject_worded_differently_is_still_a_conflict():
    """Only ONE side adding its own words keeps it one thing: a longer sentence
    or another verb never turns a real discrepancy into two subjects."""
    rows = _rows(
        ("Drawings", "Minimum cover to concrete shall be 50 mm."),
        ("Spec", "Spacers give a minimum cover to concrete of 40 mm throughout, as fixed on site."),
    )
    found = detect("The minimum cover is 50 mm (Drawings).", rows,
                   "What is the minimum concrete cover?")
    assert len(found) == 1
    assert {v["value"] for v in found[0]["values"]} == {"50 mm", "40 mm"}


def test_a_prepositional_qualifier_still_separates_the_subjects():
    """'cover to the foundation' vs 'cover to the slab': the qualifier follows
    a preposition, so the passages still name two subjects."""
    rows = _rows(
        ("Drawings", "Concrete cover to the foundation shall be 50 mm."),
        ("Spec", "Concrete cover to the slab shall be 40 mm."),
    )
    assert detect("The cover is 50 mm (Drawings).", rows, "What is the concrete cover?") == []


# --- the case-2 question: it must actually fire on those passages ----------

def test_the_turn_asks_which_subject_when_the_question_names_none():
    passages = [
        ("Drawings", "BLINDING CONCRETE: Grade C20 with 20 mm aggregate."),
        ("Spec", "STRUCTURAL CONCRETE: Grade C40 with 20 mm aggregate."),
    ]
    question = clarifying_question(
        "What concrete grade applies?",
        "Grade C20 applies (Drawings).",
        cases_from_passages(passages),
    )
    assert question is not None
    assert "Which one is meant?" in question
    # Only cases actually found, each with its own source.
    assert "Drawings" in question and "Spec" in question


def test_no_question_when_the_question_names_its_subject():
    passages = [
        ("Drawings", "BLINDING CONCRETE: Grade C20 with 20 mm aggregate."),
        ("Spec", "STRUCTURAL CONCRETE: Grade C40 with 20 mm aggregate."),
    ]
    assert clarifying_question(
        "What concrete grade applies to the structural concrete?",
        "Grade C20 applies (Drawings).",
        cases_from_passages(passages),
    ) is None


def test_a_case_whose_phrase_ends_in_the_value_label_still_answers_the_ask():
    """The passage's own phrase ends in the value's label ('Grade'), so its
    head noun is the label; the thing it names is still a word of the ask."""
    passages = [
        ("Drawings", "BLINDING CONCRETE: Grade C20 with 20 mm aggregate."),
        ("Spec", "STRUCTURAL CONCRETE: Grade C40 with 20 mm aggregate."),
    ]
    cases = cases_from_passages(passages)
    assert cases, "the reader must read both grades"
    question = clarifying_question(
        "What concrete grade applies?", "Grade C20 applies (Drawings).", cases)
    assert question is not None


def test_a_case_that_names_none_of_the_asked_words_is_dropped():
    passages = [
        ("Drawings", "BLINDING CONCRETE: Grade C20 with 20 mm aggregate."),
        ("Spec", "EXCAVATION: Depth 2.5 m to formation."),
    ]
    assert clarifying_question(
        "What concrete grade applies?", "Grade C20 applies (Drawings).",
        cases_from_passages(passages),
    ) is None
