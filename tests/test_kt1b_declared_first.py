"""K-T1b (declared-first): per DECISION_MAIN_0150Z ruling A, the DECLARED
(parseable) unit is authoritative in describe_calculation_params; the
registry name->unit default (formula_registry.NAME_UNIT_DEFAULTS) is a LAST
RESORT consulted only when a parameter declares no unit.

Intended behaviour change (stated line by line in the PR): for six params the
declared unit now wins over the old name default --
  temperature_history_c: degC (was °C)
  time_days / time_constant_days: days (was d)
  ultimate_shrinkage_microstrain: ratio (was µε)
  floor_count: count (was floors)
  cost_per_area.area: m2 (declaration fixed from the mis-declared 'ratio')
The exit-B expected strings for those params were updated to match; no other
test was touched. Display prettification (°C, µε) is a renderer concern and is
NOT introduced here.

Full-dotted imports so CI-M selects this file for the construction_formulas
mutants (the describe_calculation_params reorder in particular).
"""
from __future__ import annotations

import pytest

import app.lib.construction_formulas as cf
import app.lib.formula_registry as fr

# (formula, param, declared unit now resolved by describe_calculation_params)
DECLARED = [
    ("concrete_maturity_strength", "temperature_history_c", "degC"),
    ("concrete_shrinkage", "time_days", "days"),
    ("concrete_shrinkage", "time_constant_days", "days"),
    ("concrete_shrinkage", "ultimate_shrinkage_microstrain", "ratio"),
    ("dewatering_uplift_check", "floor_count", "count"),
    ("cost_per_area", "area", "m2"),
]


@pytest.mark.parametrize("formula, param, unit", DECLARED)
def test_describe_resolves_the_declared_unit(formula, param, unit):
    spec = fr.get(formula)
    assert spec is not None, f"{formula!r} not registered"
    rows = {r["name"]: r for r in cf.describe_calculation_params(spec.fn, name=formula)}
    assert param in rows, (formula, param)
    assert rows[param]["unit"] == unit, (formula, param, rows[param]["unit"])


@pytest.mark.parametrize("formula", ["concrete_shrinkage", "dewatering_uplift_check"])
def test_name_default_not_consulted_for_a_declared_param(monkeypatch, formula):
    """Declared-first: describe reaches the name->unit fallback only when a param
    declares no unit, so it is never consulted for a declared param. The mutant
    (fallback-first) consults it for every non-currency declared param -> the
    recorded list is non-empty -> reddens."""
    spec = fr.get(formula)
    declared = {n: (p.unit or "") for n, p in fr.parameters(spec).items()}
    called: list[str] = []
    real = cf._unit_from_name
    monkeypatch.setattr(cf, "_unit_from_name",
                        lambda name, _r=real: (called.append(name), _r(name))[1])
    cf.describe_calculation_params(spec.fn, name=formula)
    consulted_despite_declared = [k for k in called if declared.get(k, "") not in ("",)]
    assert consulted_despite_declared == [], (formula, consulted_despite_declared)
