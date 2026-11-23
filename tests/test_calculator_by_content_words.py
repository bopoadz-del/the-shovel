"""An ask that describes one calculator without naming it reaches that calculator.

``calculator_name_from_text`` finds a calculator by its written name. An ask
that writes no name but types the inputs and asks for the outputs of exactly
one formula used to get None, and the predispatch then ran
``Unknown calculation 'None'``. The last rule reads the words each formula
declares about itself (name, display name, input labels, output names) and
takes a calculator only when it clearly leads every other one.

Every ask here is invented; each decision has two inputs.
"""
from __future__ import annotations

import pytest

from app.lib import construction_formulas_structural_rc
from app.lib.construction_formulas import (
    _CONTENT_WORD_LEAD,
    _calculator_by_content_words,
    _content_words,
    calculator_name_from_text,
)


# Decision 1: inputs and outputs of one formula, no name written -> that formula.
@pytest.mark.parametrize("ask, expected", [
    ("Climbing panel of 45 m2, 6 m high, shape factor 1.2 and wind at 28 m/s. "
     "What force and overturning moment act on it?", "wind_load_on_formwork"),
    ("We must shift 9,000 t; a 40 t crane at 70 % utilisation, 15-minute "
     "cycle, 2 shifts of 10 h, 22 working days. How many cranes?",
     "crane_planning"),
])
def test_ask_describing_one_formula_reaches_it(ask, expected):
    assert calculator_name_from_text(ask) == expected


# Decision 2: the same descriptions with no figure -> nothing to compute.
@pytest.mark.parametrize("ask", [
    "What force and overturning moment does wind put on a climbing panel "
    "with a shape factor?",
    "How many cranes do we need given crane utilisation, cycle, shifts and "
    "working days?",
])
def test_description_without_a_figure_is_not_routed(ask):
    assert _calculator_by_content_words(ask) is None


# Decision 3: a document question with figures in it describes no formula.
@pytest.mark.parametrize("ask", [
    "What does clause 14.3 of the contract say about the 28 day notice?",
    "Which drawing revision C02 shows the level 3 slab openings?",
])
def test_document_question_with_figures_is_not_routed(ask):
    assert _calculator_by_content_words(ask) is None


# Decision 4: a lead narrower than the rule's margin is a guess -> None.
@pytest.mark.parametrize("ask", [
    # "wind" + "pressure" fit both wind calculators; no clear lead.
    "Wind 30 m/s: what pressure?",
    # "cranes" + "capacity" fit every crane calculator.
    "Two cranes, capacity 50 t each: enough?",
])
def test_narrow_lead_is_not_a_pick(ask):
    assert _calculator_by_content_words(ask) is None


def test_lead_margin_is_more_than_one_word():
    # One shared word apart is a coin toss between two formulas.
    assert _CONTENT_WORD_LEAD >= 2


# Decision 5: a word of two letters or a stop word is not a content word, so
# it can never add to a calculator's score.
@pytest.mark.parametrize("text, expected", [
    ("the panel of a crane", {"panel", "crane"}),
    ("what is it for the wind", {"wind"}),
])
def test_short_and_stop_words_are_not_content_words(text, expected):
    assert _content_words(text) == expected


# Decision 6: a blank ask names nothing, before any name rule is asked; a
# rule that would claim anything it is shown is never reached.
@pytest.mark.parametrize("ask", ["", " \t "])
def test_blank_ask_names_nothing_before_any_rule(ask, monkeypatch):
    monkeypatch.setattr(
        construction_formulas_structural_rc,
        "looks_like_slab_thickness_min_ask",
        lambda _text: True,
    )
    assert calculator_name_from_text(ask) is None
