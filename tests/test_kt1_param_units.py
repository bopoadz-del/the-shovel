"""K-T1a: every formula input declares a unit (no ``-``), so the model binds
arguments from the registry declarations and never from a duplicated hand
table. These inputs were open ``:unit`` gaps in
``tests/fixtures/registry_schema_gaps.json`` before this change; every unit is
one the base converter parses (see app.agents.base.formulas
.construction_formulas_planning._FAMILIES). ``fall_arrest_force.g`` stays an
honest gap -- the converter has no acceleration family -- so it keeps its
baseline entry rather than take an invented unit.

Guard (mutant-killable): revert any one of these units to ``"-"`` in its
``Param(...)`` declaration and the matching case reddens.
"""
from __future__ import annotations

import pytest

from app.lib import formula_registry as fr

# (formula, param, expected_unit) -- the exact inputs that were ``-`` before,
# with the unit their declaration now carries (semantics / the schema rule:
# a dimensionless numeric input is ``ratio`` or ``count``, never ``-``).
FILLED_UNITS = [
    ("bim_clash_tolerance", "lod", "count"),
    ("concrete_curing_time", "gain_a", "ratio"),
    ("concrete_curing_time", "gain_b", "ratio"),
    ("concrete_maturity_strength", "gain_a", "ratio"),
    ("concrete_maturity_strength", "gain_b", "ratio"),
    ("concrete_shrinkage", "ultimate_shrinkage_microstrain", "ratio"),
    ("crane_planning", "shifts_per_day", "count"),
    ("critical_path_float", "early_start", "days"),
    ("critical_path_float", "early_finish", "days"),
    ("critical_path_float", "late_start", "days"),
    ("critical_path_float", "late_finish", "days"),
    ("electrical_installation_sequence", "num_floors", "count"),
    ("plumbing_flow_programme", "num_floors", "count"),
    ("productivity_manpower_duration", "daily_production", "count/day"),
    ("productivity_manpower_duration", "manpower", "count"),
    ("productivity_manpower_duration", "quantity", "count"),
    ("productivity_manpower_duration", "quantity_executed", "count"),
    ("productivity_manpower_duration", "remaining_qty", "count"),
    ("progress_quantity", "actual_qty", "count"),
    ("progress_quantity", "planned_qty", "count"),
    ("progress_quantity", "total_qty", "count"),
    ("scaffold_load_capacity", "safety_factor", "ratio"),
    ("supervision_ratio", "piping_dia_inch", "in"),
    ("wind_load_on_formwork", "shape_factor", "ratio"),
]


@pytest.mark.parametrize("formula, param, expected_unit", FILLED_UNITS)
def test_filled_input_declares_its_unit(formula, param, expected_unit):
    spec = fr.get(formula)
    assert spec is not None, f"formula {formula!r} is not registered"
    p = spec.params.get(param)
    assert p is not None, f"{formula!r} declares no input {param!r}"
    assert p.unit not in ("", "-"), f"{formula}.{param} still has no unit"
    assert p.unit == expected_unit, (
        f"{formula}.{param} unit {p.unit!r} != expected {expected_unit!r}"
    )


@pytest.mark.parametrize("formula, param, _unit", FILLED_UNITS)
def test_filled_input_is_not_a_schema_gap(formula, param, _unit):
    # The declaration closes the ``:unit`` gap schema_gaps() would otherwise
    # report -- the mechanism the shrunk fixture now relies on.
    assert f"formula:{formula}.{param}:unit" not in set(fr.schema_gaps())
