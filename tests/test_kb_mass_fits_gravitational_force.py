"""A figure written in a mass unit binds an input declared in the force
unit that is that mass's weight under standard gravity (kg for
kilogram_force), with the number taken as written. Every other
dimension mismatch still binds nothing.

Each test builds its own entry and ask; two inputs per decision.
"""
from __future__ import annotations

import pytest

from app.blocks import _knowledge as kb


def _entry(**units):
    return {"variables": {n: {"unit": u} for n, u in units.items()}}


def _u(text):
    return kb._parse_unit(text)


# -- _unit_fits: the decision itself -----------------------------------

@pytest.mark.parametrize(
    "written, declared",
    [
        ("kg", "kilogram_force"),
        ("lb", "force_pound"),
        ("g", "gram_force"),
        ("kg/mm", "kilogram_force/millimeter"),
    ],
)
def test_mass_fits_its_own_weight_unit(written, declared):
    assert kb._unit_fits(_u(written), _u(declared)) is True


@pytest.mark.parametrize(
    "written, declared",
    [
        ("t", "kilogram_force"),   # a mass that would need a factor
        ("kg", "newton"),          # a force that is not a gravitational unit
        ("kg", "force_pound"),     # a different system's weight unit
        ("kg/m", "kilogram_force/millimeter"),  # a per-length factor
    ],
)
def test_mass_that_needs_a_factor_does_not_fit(written, declared):
    assert kb._unit_fits(_u(written), _u(declared)) is False


def test_non_mass_unit_does_not_fit_a_force():
    assert kb._unit_fits(_u("mm"), _u("kilogram_force")) is False
    assert kb._unit_fits(_u("kN"), _u("kilogram_force")) is False


def test_same_unit_still_fits_and_missing_unit_does_not():
    assert kb._unit_fits(_u("mm"), _u("millimeter")) is True
    assert kb._unit_fits(None, _u("kilogram_force")) is False
    assert kb._unit_fits(_u("kg"), None) is False


# -- bind_entry_variables: the binder uses it ----------------------------

def test_kg_figure_binds_a_kgf_input_with_the_number_as_written():
    entry = _entry(p_fail="kilogram_force", defl="millimeter")
    got = kb.bind_entry_variables(entry, "a load of 870 kg moved it 3.5 mm")
    assert got == {"p_fail": 870.0, "defl": 3.5}
    got = kb.bind_entry_variables(entry, "it moved 1.25 mm under 2410 kg")
    assert got == {"p_fail": 2410.0, "defl": 1.25}


def test_tonne_figure_does_not_bind_a_kgf_input():
    entry = _entry(p_fail="kilogram_force", defl="millimeter")
    assert kb.bind_entry_variables(entry, "a load of 4 t moved it 3.5 mm") is None
    entry_n = _entry(p_fail="newton", defl="millimeter")
    assert kb.bind_entry_variables(entry_n, "a load of 870 kg moved it 3.5 mm") is None


def test_two_mass_figures_for_one_weight_input_stay_ambiguous():
    entry = _entry(p_fail="kilogram_force", defl="millimeter")
    assert kb.bind_entry_variables(entry, "870 kg or 910 kg moved it 3.5 mm") is None
    got = kb.bind_entry_variables(entry, "910 kg moved it 3.5 mm")
    assert got == {"p_fail": 910.0, "defl": 3.5}
