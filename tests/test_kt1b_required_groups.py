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
import app.lib.formula_registry as fr  # full dotted path so CI-M selects this
                                       # file for formula_registry mutants

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


# The required_groups change edits formula() and _register, so CI-M mutates
# their (pre-existing) registration guards too. These exercise them directly.

@pytest.mark.parametrize("bad_owner", ["not-an-owner", ""])
def test_formula_rejects_an_unknown_owner(bad_owner):
    """formula() raises on an owner outside OWNERS -- kills the owner-check
    drop_rule and the raise->pass swallow_exception on formula()."""
    with pytest.raises(ValueError):
        fr.formula(owner=bad_owner, description="d", display_name="D",
                   inputs={}, outputs={})(lambda: None)


@pytest.mark.parametrize("probe", ["kt1b_dupe_probe_a", "kt1b_dupe_probe_b"])
def test_register_rejects_a_duplicate_name(probe):
    """Registering a second, different function under a name already taken
    raises -- kills the duplicate-name drop_rule and swallow_exception on
    _register. The probe is removed afterwards so the registry is unchanged."""
    deco = lambda fn: fr.formula(owner="base", description="d", display_name="D",
                                 inputs={}, outputs={}, name=probe)(fn)
    try:
        deco(lambda: 1)
        with pytest.raises(ValueError):
            deco(lambda: 2)  # a different function under the same name
    finally:
        fr._REGISTRY.pop(probe, None)
