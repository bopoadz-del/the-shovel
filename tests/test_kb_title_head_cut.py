"""D5 selector, the "named" gate: an ask names a knowledge formula by the
head of its title, the words before the title's first qualifying
preposition. What follows ("of a member", "to empty a sump", "per shift")
qualifies the result, and an ask may word it differently or leave it out.

Every entry, title, variable and figure below is invented here; the KB file
is this test's own (CONSTRUCTION_KB_FILE). Expected values are hand
calculations, never the product's output. Each decision has two inputs.
Failures name the stage: head, select or bind.
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
    _entry("t.cable_sag", "Elastic sag of a guyed cable span", "L / 50",
           {"L": {"desc": "Span length", "unit": "meter"}}, "meter"),
    _entry("t.pump_hours", "Pump hours to empty a flooded sump", "V / 40",
           {"V": {"desc": "Water volume", "unit": "cubic meter"}}, "hour"),
    _entry("t.crate_stacks", "Number of crate stacks on a pallet deck", "ceiling(H / 1.5)",
           {"H": {"desc": "Total height of crates to stack", "unit": "meter"}}, "stack"),
]


@pytest.fixture(autouse=True)
def _own_kb(tmp_path, monkeypatch):
    path = tmp_path / "kb.json"
    path.write_text(json.dumps({"schema_version": "1", "kb_version": "exam",
                                "entries": ENTRIES}), encoding="utf-8")
    monkeypatch.setenv("CONSTRUCTION_KB_FILE", str(path))
    yield


# -- the head stops at the first qualifying preposition: two inputs each ---
@pytest.mark.parametrize("title,head", [
    ("Elastic sag of a guyed cable span", ["elastic", "sag"]),
    ("Pump hours to empty a flooded sump", ["pump", "hours"]),
    ("Grout take per metre of borehole", ["grout", "take"]),
    ("Kerb length from plan perimeter", ["kerb", "length"]),
])
def test_head_is_the_words_before_the_first_preposition(title, head):
    assert kb._title_head_words(title) == head, "stage=head"


# -- a counting head word runs on past "of": two inputs --------------------
@pytest.mark.parametrize("title,head", [
    ("Number of crate stacks on a pallet deck", ["number", "crate", "stacks"]),
    ("Amount of curing water for a slab", ["amount", "curing", "water"]),
])
def test_counting_head_runs_past_of(title, head):
    assert kb._title_head_words(title) == head, "stage=head"


# -- the ask names the head and words the qualifier its own way ------------
@pytest.mark.parametrize("text,rule,values", [
    ("Calculate the elastic sag for a 300 m span", "t.cable_sag", {"L": 300.0}),
    ("How many pump hours for 1,200 m3 of water?", "t.pump_hours", {"V": 1200.0}),
])
def test_ask_naming_only_the_head_selects_the_entry(text, rule, values):
    got = kb.formula_completed_by_user_text(text)
    assert got is not None, "stage=select: head named, entry not selected"
    assert got[0] == rule, f"stage=select: picked {got[0]}"
    assert got[1] == {k: pytest.approx(v) for k, v in values.items()}, f"stage=bind: {got[1]}"


# -- a head word missing from the ask is still no selection: two inputs ----
@pytest.mark.parametrize("text", [
    "Calculate the sag for a 300 m span",
    "How many hours to empty 1,200 m3 of water?",
])
def test_missing_head_word_is_not_selected(text):
    assert kb.formula_completed_by_user_text(text) is None, "stage=select"


# -- a counting head is asked by its question phrase plus the thing counted -
@pytest.mark.parametrize("text,selected", [
    ("How many crate stacks for 12 m of crates?", True),
    ("How many stacks for 12 m of crates?", False),
])
def test_counting_head_needs_the_thing_counted(text, selected):
    got = kb.formula_completed_by_user_text(text)
    if selected:
        assert got is not None and got[0] == "t.crate_stacks", f"stage=select: {got}"
    else:
        assert got is None, f"stage=select: {got}"
