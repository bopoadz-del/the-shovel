"""A unit written once after a list of figures is every listed figure's
unit, and a word after a unit is not read as a second unit.

``20, 28 and 36 minutes`` used to give the first two figures no unit, so
a knowledge entry declared in minutes could not bind them. Separately,
``36 minutes at`` used to read as minute times technical atmosphere
(``at`` is that unit's symbol), so even the last figure lost its unit.
A two-word reading is now taken only when its second word spells a unit,
and a unitless figure joined to the next by a bare comma, ``and``, ``or``
or ``x`` takes that figure's unit. Each test builds its own entry and
ask; no shipped entry or exam text is read.
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
    ("laps of 20, 28 and 36 minutes", ["minute", "minute", "minute"]),
    ("laps of 20, 28 and 36 widgets", [None, None, None]),
])
def test_a_trailing_unit_is_shared_across_a_list(text, units):
    assert _units(text) == units


@pytest.mark.parametrize("text,units", [
    ("either 4 or 5 hours", ["hour", "hour"]),
    ("either 4 cranes or 5 hours", [None, "hour"]),
])
def test_a_word_of_its_own_ends_the_list(text, units):
    assert _units(text) == units


@pytest.mark.parametrize("text,units", [
    ("we waited 9, then 12 hours", [None, "hour"]),
    ("we waited 9, 12 hours", ["hour", "hour"]),
])
def test_only_a_bare_joiner_shares_the_unit(text, units):
    assert _units(text) == units


@pytest.mark.parametrize("text,units", [
    ("a lap of 36 minutes at the pit", ["minute"]),
    ("a lap of 36 minutes in the pit", ["minute"]),
])
def test_a_symbol_only_word_after_a_unit_is_a_word(text, units):
    assert _units(text) == units


@pytest.mark.parametrize("text,units", [
    ("a pit of 12 cubic meter at the gate", ["meter ** 3"]),
    ("a pit of 12 square meter at the gate", ["meter ** 2"]),
])
def test_a_spelled_two_word_unit_still_wins(text, units):
    assert _units(text) == units


@pytest.mark.parametrize("text,expected", [
    ("a lap of 36 minutes at 50 m3 per load",
     {"t_cycle": 36.0, "v_bucket": 50.0}),
    ("a lap of 36 widgets at 50 m3 per load", None),
])
def test_the_unit_before_a_word_binds_an_entry(text, expected):
    entry = _entry(t_cycle="minute", v_bucket="meter**3")
    assert K.bind_entry_variables(entry, text) == expected


@pytest.mark.parametrize("text,expected", [
    ("a load of 50 m3 and laps of 20 and 36 minutes", None),
    ("a load of 50 m3 and a lap of 36 minutes", {"t_cycle": 36.0, "v_bucket": 50.0}),
])
def test_a_listed_unit_counts_every_listed_figure(text, expected):
    # Two minute figures for one minute input is ambiguous: no guess.
    entry = _entry(t_cycle="minute", v_bucket="meter**3")
    assert K.bind_entry_variables(entry, text) == expected
