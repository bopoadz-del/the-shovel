"""Exam for BUILD 2 last gap: a knowledge-base formula the ask names AND
supplies is selected and bound without a model.

Every entry, title, variable and figure below is invented here; the KB file
is this test's own (CONSTRUCTION_KB_FILE). Expected values are hand
calculations, never the product's output. Each decision has two inputs.

Selection rule under test (mirrors construction_formulas.
formula_completed_by_user_text for the registry):
  * the ask is an explicit calculation request;
  * every content word of the entry's title is in the ask;
  * every declared variable binds to exactly one figure: by its name
    (``name = 4`` / ``name of 4``) or, failing that, by the variable's
    declared unit when it is the only unbound variable of that unit;
  * any tie between entries, any missing or ambiguous variable -> None.
Failures name the stage: select, bind or evaluate.
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
    _entry("t.ramp_rise", "Ramp rise check", "run * grade / 100",
           {"run": {"desc": "Horizontal run of the ramp", "unit": "meter"},
            "grade": {"desc": "Grade of the ramp", "unit": "percent"}}, "meter"),
    # Two entries sharing every title word: naming them is no selection.
    _entry("t.silo_fill_a", "Silo fill time", "V / 2",
           {"V": {"desc": "Silo volume", "unit": "cubic meter"}}, "hour"),
    _entry("t.silo_fill_b", "Silo fill time", "V / 3",
           {"V": {"desc": "Silo volume", "unit": "cubic meter"}}, "hour"),
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


# -- select + bind by unit (single variable): two inputs -------------------
@pytest.mark.parametrize("text,value,hand", [
    ("Calculate the kiln soak duration for a 0.6 m wall", 0.6, 24.0),
    ("Work out kiln soak duration, wall thickness 1.25 m", 1.25, 50.0),
])
def test_single_variable_binds_by_unit(text, value, hand):
    got = _select(text)
    assert got is not None, "stage=select: named entry with its input not selected"
    rule_id, values = got
    assert rule_id == "t.kiln_soak", f"stage=select: picked {rule_id}"
    assert values == {"Z": pytest.approx(value)}, f"stage=bind: {values}"
    res = kb.evaluate(rule_id, **values)
    assert res["result"] == pytest.approx(hand), f"stage=evaluate: {res}"


# -- bind by variable name when two variables share a unit: two inputs ----
@pytest.mark.parametrize("text,cargo,hold,hand", [
    ("Calculate the barge trip count: cargo = 950 t, hold = 300 t", 950, 300, 4),
    ("How many? barge trip count with hold of 120 tonne and cargo of 600 tonne",
     600, 120, 5),
])
def test_same_unit_variables_bind_by_name(text, cargo, hold, hand):
    got = _select(text)
    assert got is not None, "stage=select"
    rule_id, values = got
    assert rule_id == "t.barge_trips"
    assert values == {"cargo": pytest.approx(cargo), "hold": pytest.approx(hold)}, \
        f"stage=bind: {values}"
    assert kb.evaluate(rule_id, **values)["result"] == pytest.approx(hand), "stage=evaluate"


# -- mixed units bind by unit in any order: two inputs ---------------------
@pytest.mark.parametrize("text,run,grade,hand", [
    ("Compute the ramp rise check for 8% over 25 m", 25, 8, 2.0),
    ("Calculate ramp rise check: 40 m run at 5 %", 40, 5, 2.0),
])
def test_distinct_units_bind_in_any_order(text, run, grade, hand):
    got = _select(text)
    assert got is not None, "stage=select"
    rule_id, values = got
    assert values == {"run": pytest.approx(run), "grade": pytest.approx(grade)}, \
        f"stage=bind: {values}"
    assert kb.evaluate(rule_id, **values)["result"] == pytest.approx(hand)


# -- must NOT select ---------------------------------------------------------
@pytest.mark.parametrize("text", [
    # same-unit variables, no names: which is cargo? not guessed
    "Calculate the barge trip count for 950 t and 300 t",
    "Work out barge trip count, 600 tonne and 120 tonne",
])
def test_ambiguous_same_unit_figures_are_not_guessed(text):
    assert _select(text) is None, "stage=bind: ambiguous figures were bound"


@pytest.mark.parametrize("text", [
    # inputs bind but the entry is not named
    "Calculate the duration for a 0.6 m wall",
    "Work out how long a 1.25 m thick wall takes",
])
def test_unnamed_entry_is_not_selected(text):
    assert _select(text) is None, "stage=select: an unnamed entry was selected"


@pytest.mark.parametrize("text", [
    # named, an input missing
    "Calculate the ramp rise check for a 25 m run",
    "Calculate the kiln soak duration",
])
def test_missing_input_is_not_selected(text):
    assert _select(text) is None, "stage=bind: a missing input was filled"


@pytest.mark.parametrize("text", [
    "Calculate the silo fill time for 120 m3",
    "Work out silo fill time, 90 cubic meter",
])
def test_two_entries_at_the_same_strength_is_no_selection(text):
    assert _select(text) is None, "stage=select: a tie was broken by guess"


@pytest.mark.parametrize("text", [
    # not a calculation request: a question about the rule
    "What does the kiln soak duration rule say for 0.6 m walls?",
    "Explain the barge trip count rule: cargo = 950 t, hold = 300 t",
])
def test_not_a_calculation_request(text):
    assert _select(text) is None, "stage=select: a non-calculation ask was bound"
