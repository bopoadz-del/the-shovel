"""A knowledge formula is a calculation request when the ask's question
asks for its result, even with no calculation verb (D5, stage 1).

"<givens>. What is the <result>?" states the figures and names the result.
The question may hold only: the entry's title words, words the givens
already introduced, units and framing words. A question about anything
else, or an ask with no question and no calculation verb, selects nothing.

Every entry, title, variable and figure is invented here; the KB file is
this test's own (CONSTRUCTION_KB_FILE). Expected values are hand
calculations. Each decision has two inputs.
"""
from __future__ import annotations

import json

import pytest

from app.blocks import _knowledge as kb


def _entry(eid, title, expr, variables, unit):
    return {
        "id": eid, "type": "formula", "title": title, "expression": expr,
        "statement": title, "domain": "construction.test",
        "variables": variables, "thresholds": {"unit": unit},
        "credibility_tier": 3, "needs_review": False,
        "provenance": {"source": "invented test reference",
                       "confidence": "site_experience",
                       "verified_against_standard": False},
    }


ENTRIES = [
    _entry("t.kiln_soak", "Kiln soak duration", "40 * Z",
           {"Z": {"desc": "Wall thickness of the kiln", "unit": "meter"}}, "hour"),
    _entry("t.barge_trips", "Barge trip count", "ceiling(cargo / hold)",
           {"cargo": {"desc": "Total cargo to move", "unit": "tonne"},
            "hold": {"desc": "Capacity of one barge hold", "unit": "tonne"}}, "trip"),
]


@pytest.fixture(autouse=True)
def _own_kb(tmp_path, monkeypatch):
    path = tmp_path / "kb.json"
    path.write_text(json.dumps({"schema_version": "1", "kb_version": "exam",
                                "entries": ENTRIES}), encoding="utf-8")
    monkeypatch.setenv("CONSTRUCTION_KB_FILE", str(path))
    yield


def _select(text):
    return kb.formula_completed_by_user_text(text)


# -- givens, then a question that names only the result: selected ----------
@pytest.mark.parametrize("text,rule,values", [
    ("Kiln wall 0.6 m thick. What is the kiln soak duration?",
     "t.kiln_soak", {"Z": 0.6}),                               # 40 x 0.6 = 24 h
    ("Barge job: cargo = 950 t, hold = 300 t. What will the barge trip count be?",
     "t.barge_trips", {"cargo": 950.0, "hold": 300.0}),        # ceil(3.17) = 4
])
def test_question_for_the_result_is_a_calculation_request(text, rule, values):
    assert _select(text) == (rule, values), "stage=select: a result question was not read as a calculation"


# -- no question and no calculation verb: nothing ----------------------------
@pytest.mark.parametrize("text", [
    "Kiln soak duration, kiln wall 0.6 m thick.",
    "Barge trip count: cargo = 950 t, hold = 300 t",
])
def test_no_question_and_no_verb_selects_nothing(text):
    assert _select(text) is None, "stage=select: a statement was evaluated"


# -- the question asks about something else: nothing -------------------------
@pytest.mark.parametrize("text", [
    "Kiln wall 0.6 m thick. What does the kiln soak duration rule say?",
    "Cargo = 950 t, hold = 300 t. Who signs the barge trip count sheet?",
])
def test_question_about_something_else_selects_nothing(text):
    assert _select(text) is None, "stage=select: a question about the rule was evaluated"


# -- a question word the givens introduced is part of the result question ----
@pytest.mark.parametrize("text,rule,values", [
    ("Furnace kiln, wall 0.6 m thick. What is the furnace kiln soak duration?",
     "t.kiln_soak", {"Z": 0.6}),
    ("River convoy: cargo = 950 t, hold = 300 t. What is the river convoy barge trip count?",
     "t.barge_trips", {"cargo": 950.0, "hold": 300.0}),
])
def test_question_may_reuse_the_givens_words(text, rule, values):
    assert _select(text) == (rule, values), "stage=select: the givens' own subject vetoed the question"


@pytest.mark.parametrize("text", [
    "Kiln wall 0.6 m thick. What is the furnace kiln soak duration?",
    "Cargo = 950 t, hold = 300 t. What is the river convoy barge trip count?",
])
def test_question_word_nowhere_in_the_givens_selects_nothing(text):
    assert _select(text) is None, "stage=select: an unstated subject was accepted"


# -- a unit or a figure in the question is not an extra subject --------------
@pytest.mark.parametrize("text,rule,values", [
    ("Kiln wall 0.6 m thick. What is the kiln soak duration in hours?",
     "t.kiln_soak", {"Z": 0.6}),
    ("Cargo and hold in tonnes. What is the barge trip count for cargo = 1200 t, hold = 300 t?",
     "t.barge_trips", {"cargo": 1200.0, "hold": 300.0}),       # ceil(4.0) = 4
])
def test_units_and_figures_in_the_question_are_allowed(text, rule, values):
    assert _select(text) == (rule, values), "stage=select: a unit or figure vetoed the question"


# -- the question must name the result itself ---------------------------------
@pytest.mark.parametrize("text", [
    "Kiln soak duration: wall 0.6 m thick. What is it?",
    "Barge trip count: cargo = 950 t, hold = 300 t. What then?",
])
def test_question_naming_no_title_word_selects_nothing(text):
    assert _select(text) is None, "stage=select: a question that names nothing was evaluated"


# -- every question must ask only for the result ------------------------------
@pytest.mark.parametrize("text,expect", [
    ("Kiln wall 0.6 m thick. What is the kiln soak duration? Is the kiln safe?", None),
    ("Kiln wall 0.6 m thick. What is the kiln soak duration? Which kiln soak duration?",
     ("t.kiln_soak", {"Z": 0.6})),
])
def test_every_question_sentence_is_read(text, expect):
    assert _select(text) == expect, "stage=select: a second question was not read"


# -- the calculation verb still selects without a question --------------------
@pytest.mark.parametrize("text,rule,values", [
    ("Calculate the kiln soak duration, wall 0.6 m", "t.kiln_soak", {"Z": 0.6}),
    ("Work out barge trip count: cargo = 950 t, hold = 300 t",
     "t.barge_trips", {"cargo": 950.0, "hold": 300.0}),
])
def test_calculation_verb_still_selects(text, rule, values):
    assert _select(text) == (rule, values), "stage=select: the verb request was lost"
