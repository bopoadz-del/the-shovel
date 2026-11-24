"""Knowledge formula inputs that share a unit bind from the ask's own words.

A knowledge entry with two inputs of one unit (two forces in kN, two
distances in km) used to bind nothing, so an ask that typed every figure
never reached the entry's evaluation. Each test builds its own entry and
ask; no shipped entry or exam text is read.
"""
from __future__ import annotations

import pytest

from app.blocks import _knowledge as K


def _entry(**units: str) -> dict:
    return {"id": "t.entry", "type": "formula",
            "variables": {n: {"unit": u, "desc": ""} for n, u in units.items()}}


@pytest.mark.parametrize("text,unit,value", [
    ("the crane travels at 35 km/hr on site", "kilometer/hour", 35.0),
    ("the plant delivers 12 m3/hr of mix", "meter**3/hour", 12.0),
])
def test_a_unit_written_over_a_slash_is_one_unit(text, unit, value):
    entry = _entry(rate=unit)
    assert K.bind_entry_variables(entry, text) == {"rate": value}


@pytest.mark.parametrize("text,units", [
    ("a panel 20 x 30 cm thick", ["centimeter", "centimeter"]),
    ("a panel 20 and 30 cm thick", [None, "centimeter"]),
])
def test_a_dimension_pair_shares_the_unit_written_after_it(text, units):
    got = [None if f["unit"] is None else str(f["unit"]) for f in K._figures(text)]
    assert got == units


@pytest.mark.parametrize("text,expected", [
    ("a 40 kN push from the jack against a 25 kN pull on the tie",
     {"push_force": 40.0, "pull_force": 25.0}),
    ("a 40 kN pull on the tie against a 25 kN push from the jack",
     {"push_force": 25.0, "pull_force": 40.0}),
])
def test_same_unit_inputs_bind_by_their_own_name_word(text, expected):
    entry = _entry(push_force="kN", pull_force="kN")
    assert K.bind_entry_variables(entry, text) == expected


@pytest.mark.parametrize("text,expected", [
    ("a plate 15 x 15 cm", {"plate_width": 15.0, "plate_depth": 15.0}),
    ("a plate 15 x 20 cm", None),
])
def test_unnamed_same_unit_inputs_bind_only_when_their_figures_agree(text, expected):
    entry = _entry(plate_width="cm", plate_depth="cm")
    assert K.bind_entry_variables(entry, text) == expected


@pytest.mark.parametrize("text,expected", [
    ("push and pull of 40 kN, and 25 kN elsewhere", None),
    ("push of 40 kN, and pull of 25 kN elsewhere",
     {"push_force": 40.0, "pull_force": 25.0}),
])
def test_two_names_nearest_one_figure_bind_nothing(text, expected):
    entry = _entry(push_force="kN", pull_force="kN")
    assert K.bind_entry_variables(entry, text) == expected


@pytest.mark.parametrize("text,expected", [
    ("a 40 kN push, a 25 kN pull and a 10 kN load",  None),
    # each name has one nearest figure; the spare figure alone refuses it
    ("a 40 kN push and a 25 kN pull, 10 kN spare", None),
    ("a 40 kN push and a 25 kN pull", {"push_force": 40.0, "pull_force": 25.0}),
])
def test_more_figures_than_inputs_of_a_unit_bind_nothing(text, expected):
    entry = _entry(push_force="kN", pull_force="kN")
    assert K.bind_entry_variables(entry, text) == expected


@pytest.mark.parametrize("text,expected", [
    ("a 40 kN push", None),
    ("a 40 kN push and a 6 m reach", {"push_force": 40.0, "reach": 6.0}),
])
def test_a_single_input_of_its_unit_still_needs_its_one_figure(text, expected):
    entry = _entry(push_force="kN", reach="m")
    assert K.bind_entry_variables(entry, text) == expected


@pytest.mark.parametrize("text", ["", "   "])
def test_empty_unit_text_is_no_unit(text):
    assert K._parse_unit(text) is None


@pytest.mark.parametrize("entry", [
    {"id": "t.entry", "type": "formula", "variables": {}},
    {"id": "t.entry", "type": "formula"},
])
def test_an_entry_with_no_inputs_binds_nothing(entry):
    assert K.bind_entry_variables(entry, "a 40 kN push and a 6 m reach") is None
