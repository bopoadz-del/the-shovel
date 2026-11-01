"""K-T1b (fallback), behaviour-preserving relocation: the name -> unit override
table moved from construction_formulas._PARAM_UNIT_OVERRIDE to the registry
(formula_registry.NAME_UNIT_DEFAULTS) + name_unit_default(); _unit_from_name
delegates to it. This is a PURE MOVE -- _unit_from_name resolves exactly what it
did before, and the describe_calculation_params consumer is unchanged.

The further step 2330Z asked for -- making the fallback last-resort
(declared-first) -- is NOT behaviour-preserving yet: several params declare a
unit that DIFFERS from the name default (temperature_history_c degC vs °C;
time_days / time_constant_days days vs d; ultimate_shrinkage_microstrain ratio
vs µε; floor_count count vs floors; cost_per_area.area ratio vs m2), and the
exit-B tests pin the name-default spelling. That reconciliation is an owner
decision (DECISION_MAIN) and ships separately; this slice only relocates.

Full-dotted imports so CI-M selects this file for the construction_formulas /
formula_registry mutants.
"""
from __future__ import annotations

import pytest

import app.lib.construction_formulas as cf
import app.lib.formula_registry as fr


@pytest.mark.parametrize("name, expected", [
    ("pv", "currency"),
    ("rate_percent", "%"),
    ("column_diameter_mm", "mm"),
    ("floor_count", "floors"),
    ("not-a-known-name", ""),
])
def test_name_unit_default_lookup(name, expected):
    assert fr.name_unit_default(name) == expected


@pytest.mark.parametrize("name, expected", [
    ("pv", "currency"),            # registry default (override) path
    ("contract_amount", "currency"),
    ("unknown_name_zzz", ""),      # no default, no suffix -> ""
])
def test_unit_from_name_prefers_the_registry_default(name, expected):
    # Delegation: a name in NAME_UNIT_DEFAULTS resolves to that default.
    # drop_rule on the `if override:` guard -> the default is skipped and the
    # suffix path returns "" for these names -> reddens.
    assert cf._unit_from_name(name) == expected


@pytest.mark.parametrize("name, expected", [
    ("beam_span_m", "m"),          # suffix path still works (no default entry)
    ("load_kn", "kN"),
])
def test_unit_from_name_falls_through_to_the_suffix(name, expected):
    assert cf._unit_from_name(name) == expected


def test_the_override_table_moved_off_construction_formulas():
    assert not hasattr(cf, "_PARAM_UNIT_OVERRIDE")
    assert isinstance(fr.NAME_UNIT_DEFAULTS, dict) and fr.NAME_UNIT_DEFAULTS
