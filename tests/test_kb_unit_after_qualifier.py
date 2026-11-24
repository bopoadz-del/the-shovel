"""A unit written after one qualifier word (``working days``) is the
figure's unit.

``210 working days`` used to leave the figure with no unit, because
neither ``working days`` nor ``working`` is a unit, so an ask that typed
every input never bound a knowledge entry declared in days and hours.
The second word is read only when it spells a unit name: a word that is
also an English function word (``at``, ``in``, ``a``, ``as``) stays a
word. Each test builds its own entry and ask; no shipped entry or exam
text is read.
"""
from __future__ import annotations

import pytest

from app.blocks import _knowledge as K


def _units(text: str) -> list:
    return [None if f["unit"] is None else str(f["unit"]) for f in K._figures(text)]


def _entry(**units: str) -> dict:
    return {"id": "t.entry", "type": "formula",
            "variables": {n: {"unit": u, "desc": ""} for n, u in units.items()}}


@pytest.mark.parametrize("text,units", [
    ("finish in 45 calendar days", ["day"]),
    ("finish in 45 calendar widgets", [None]),
])
def test_a_spelled_unit_after_a_qualifier_is_read(text, units):
    assert _units(text) == units


@pytest.mark.parametrize("text,units", [
    ("a pit of 12 cubic meter", ["meter ** 3"]),
    ("a pit of 12 square meter", ["meter ** 2"]),
])
def test_a_two_word_unit_still_wins_over_its_second_word(text, units):
    assert _units(text) == units


@pytest.mark.parametrize("text,units", [
    ("we placed 7 bollards", [None]),
    ("we placed 7 hours", ["hour"]),
])
def test_a_single_word_that_is_no_unit_leaves_no_unit(text, units):
    assert _units(text) == units


@pytest.mark.parametrize("text,units", [
    ("lay 6 courses at", [None]),
    ("lay 6 courses weeks", ["week"]),
    ("lay 6 more mins", ["minute"]),
])
def test_the_first_reading_decides(text, units):
    # "at" first reads as the technical-atmosphere symbol; "mins" first
    # reads as minutes (its second reading is milli-inch)
    assert _units(text) == units


@pytest.mark.parametrize("text,units", [
    ("about 3 times as", [None]),
    ("about 3 extra seconds", ["second"]),
])
def test_a_prefixed_reading_is_a_word(text, units):
    # "as" reads only as atto-second
    assert _units(text) == units


@pytest.mark.parametrize("text,units", [
    ("make 4 trips in", [None]),
    ("make 4 trips a", [None]),
    ("make 4 overtime week", ["week"]),
])
def test_a_bare_symbol_is_a_word_unless_it_is_also_the_name(text, units):
    # "in" is only the inch symbol, "a" only the year symbol; "week" is
    # both the symbol and the name of the week
    assert _units(text) == units


@pytest.mark.parametrize("text,expected", [
    ("move 3,600 m3 within 18 working days at 10 working hours per day",
     {"total_quantity": 3600.0, "working_days": 18.0, "hours_per_day": 10.0}),
    ("move 3,600 m3 within 18 working days at 10 working shifts per day",
     None),
])
def test_qualified_inputs_complete_an_entry(text, expected):
    entry = _entry(total_quantity="meter**3", working_days="day",
                   hours_per_day="hour")
    assert K.bind_entry_variables(entry, text) == expected
