"""A knowledge formula is named by the head of its title, not by every
word of the title.

A title names its result first and then how it is worked: "X from A and
B", "X (A vs B)"; a bracketed aside anywhere is left out. The ask names the entry when it carries X; it need not
repeat the method words. A counting head ("Number of X") is asked as
"how many X". Among named entries whose inputs all bind, the one with the
most title words in the ask wins, then the one with fewest missing; a
tie is no selection.

Every entry, title, variable and figure is invented here; the KB file is
this test's own (CONSTRUCTION_KB_FILE). Expected values are hand
calculations. Each decision has two inputs. Failures name the stage.
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
    _entry("t.tide_lag", "Tide lag from basin depth and channel width",
           "depth * 2 + width / 10",
           {"depth": {"desc": "Basin depth", "unit": "meter"},
            "width": {"desc": "Channel width", "unit": "percent"}}, "minute"),
    _entry("t.hoist_margin", "Hoist margin (rating vs load)", "rating / load",
           {"rating": {"desc": "Rated capacity", "unit": "tonne"},
            "load": {"desc": "Lifted load", "unit": "meter"}}, "ratio"),
    _entry("t.crate_tiers", "Number of stacked crate tiers", "ceiling(S / K)",
           {"S": {"desc": "Stack height", "unit": "meter"},
            "K": {"desc": "Crate height", "unit": "percent"}}, "tier"),
    _entry("t.span_drift", "Gross (laden) span drift from deck load", "load / 4",
           {"load": {"desc": "Deck load", "unit": "tonne"}}, "millimeter"),
    # Same head; the second adds a condition the ask may or may not name.
    _entry("t.grout_cure", "Grout cure time from slab depth", "depth * 3",
           {"depth": {"desc": "Slab depth", "unit": "meter"}}, "hour"),
    _entry("t.grout_cure_cold", "Grout cure time from slab depth in frost weather",
           "depth * 5",
           {"depth": {"desc": "Slab depth", "unit": "meter"}}, "hour"),
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


def _evaluated(text, rule):
    got = _select(text)
    assert got is not None, "stage=select: the entry the ask names was not selected"
    rule_id, values = got
    assert rule_id == rule, f"stage=select: picked {rule_id}"
    return kb.evaluate(rule_id, **values)["result"]


# -- the words after "from" describe the method; the ask need not say them --
@pytest.mark.parametrize("text,hand", [
    ("Calculate the tide lag: 4 m deep basin, 30 % opening", 11.0),
    ("Work out tide lag for 2.5 m and 50 %", 10.0),
])
def test_head_before_from_names_the_entry(text, hand):
    assert _evaluated(text, "t.tide_lag") == pytest.approx(hand), "stage=evaluate"


# -- the words in brackets describe the inputs; the ask need not say them --
@pytest.mark.parametrize("text,hand", [
    ("Compute the hoist margin, 12 t against 4 m", 3.0),
    ("Calculate hoist margin with 9 tonne over 6 m", 1.5),
])
def test_head_before_bracket_names_the_entry(text, hand):
    assert _evaluated(text, "t.hoist_margin") == pytest.approx(hand), "stage=evaluate"


# -- a bracketed aside inside the head is left out, not a cut ---------------
@pytest.mark.parametrize("text,hand", [
    ("Calculate the gross span drift for 8 t", 2.0),
    ("Work out gross span drift, 10 t on the deck", 2.5),
])
def test_aside_inside_the_head_is_left_out(text, hand):
    assert _evaluated(text, "t.span_drift") == pytest.approx(hand), "stage=evaluate"


# -- a counting head is asked as "how many" ----------------------------------
@pytest.mark.parametrize("text,hand", [
    ("A 7 m stack of crates at 20 %. How many stacked crate tiers?", 1.0),
    ("How many crate tiers are stacked: 3 m high, 2 % each", 2.0),
])
def test_counting_head_is_asked_as_how_many(text, hand):
    assert _evaluated(text, "t.crate_tiers") == pytest.approx(hand), "stage=evaluate"


@pytest.mark.parametrize("text", [
    # the counting word is neither written nor asked: not this entry
    "Calculate the stacked crate tiers for a 7 m stack at 20 %",
    "Work out crate tiers stacked 3 m high, 2 % each",
])
def test_counting_head_still_needs_its_question(text):
    assert _select(text) is None, "stage=select: a counting head was dropped unasked"


# -- a head word missing is still no selection --------------------------------
@pytest.mark.parametrize("text", [
    "Calculate the lag for a 4 m deep basin and 30 % opening",
    "Compute the margin, 12 t against 4 m",
    # the word before the aside alone is not the head
    "Calculate the gross load for 8 t",
])
def test_missing_head_word_is_not_selected(text):
    assert _select(text) is None, "stage=select: an entry was selected without its head"


# -- among entries sharing a head, the ask's own words decide ----------------
@pytest.mark.parametrize("text,rule,hand", [
    # names only the head: the entry with no missing words wins
    ("Calculate the grout cure time for a 2 m slab", "t.grout_cure", 6.0),
    # names the condition too: the entry carrying it wins
    ("Calculate grout cure time in frost weather for a 2 m slab",
     "t.grout_cure_cold", 10.0),
])
def test_shared_head_is_decided_by_the_title_words_asked(text, rule, hand):
    assert _evaluated(text, rule) == pytest.approx(hand), "stage=evaluate"
