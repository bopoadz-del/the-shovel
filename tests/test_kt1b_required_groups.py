"""K-T1b (groups): the alternative-required input sets that lived in the
central ``_REQUIRED_GROUPS`` hand table are now declared on each formula
(``@formula(required_groups=...)``) and read from the registry, so the
requirement lives with the kernel. Behaviour-preserving: ``_missing_required``
consults the same sets, keyed from the formula's own declaration.

The sets are formula-level (not a single per-``Param`` label) because an input
can belong to two groups -- ``delay_damages_daily.amount_per_day`` satisfies
both the rate group and the contract-amount group.

Mutants:
- drop the group branch in ``_missing_required`` -> a formula given nothing no
  longer reports its group leads -> ``test_groups_drive_missing`` reddens
  (every grouped input carries a signature default, so without the groups
  nothing is reported missing).
- declare a wrong/empty ``required_groups`` on a formula ->
  ``test_each_formula_declares_its_required_groups`` reddens.
"""
from __future__ import annotations

import pytest

import app.lib.construction_formulas as cf
from app.lib import formula_registry as fr

# Snapshot of the sets as they stood in the pre-migration hand table.
GROUPS = {
    "calculate_evm": (("pv", "bcws"), ("ev", "bcwp"), ("ac", "acwp")),
    "delay_damages_daily": (("rate_percent", "amount_per_day"),
                            ("contract_amount", "amount_per_day")),
    "beam_shear_simple": (("udl_w_kn_m", "central_point_load_kn"), ("span_m",)),
    "concrete_volume": (("length_m", "diameter_m", "top_width_m"),),
}


@pytest.mark.parametrize("name, expected", list(GROUPS.items()))
def test_each_formula_declares_its_required_groups(name, expected):
    spec = fr.get(name)
    assert spec is not None, f"{name!r} is not registered"
    assert spec.required_groups == expected


@pytest.mark.parametrize("name, groups", list(GROUPS.items()))
def test_groups_drive_missing(name, groups):
    """Supplying one alias per group removes exactly those group leads from the
    missing set -- isolating the group mechanism from any always-required
    inputs (constant across both calls)."""
    fn = fr.get(name).fn
    leads = {g[0] for g in groups}
    none_given = set(cf._missing_required(fn, {}, name=name))
    satisfied = set(cf._missing_required(fn, {lead: 1 for lead in leads}, name=name))
    assert leads <= none_given, (name, leads, none_given)
    assert not (leads & satisfied), (name, leads, satisfied)


def test_overlapping_alias_satisfies_both_groups():
    # amount_per_day is in BOTH of delay_damages_daily's groups; giving it alone
    # clears both -- only a formula-level set (not one label per Param) can.
    fn = fr.get("delay_damages_daily").fn
    missing = set(cf._missing_required(fn, {"amount_per_day": 100},
                                       name="delay_damages_daily"))
    assert "rate_percent" not in missing and "contract_amount" not in missing, missing


@pytest.mark.parametrize("name, groups", list(GROUPS.items()))
def test_describe_marks_group_members_required(name, groups):
    """describe_calculation_params marks every group member required (they carry
    signature defaults, so only the group loop makes them required). Dropping
    that loop -> they read optional -> this reddens."""
    fn = fr.get(name).fn
    rows = {r["name"]: r for r in cf.describe_calculation_params(fn, name=name)}
    for group in groups:
        for member in group:
            if member in rows:  # a member may be a non-signature alias
                assert rows[member]["required"] is True, (name, member)


def test_the_hand_table_is_gone():
    assert not hasattr(cf, "_REQUIRED_GROUPS")
