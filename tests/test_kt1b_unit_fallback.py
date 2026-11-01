"""K-T1b (fallback): the name -> unit override table moved from
construction_formulas._PARAM_UNIT_OVERRIDE to the registry
(formula_registry.NAME_UNIT_DEFAULTS) as a LAST-RESORT default, consulted only
when a parameter declares no unit. describe_calculation_params now prefers the
declared unit (declared-first) and reaches the name fallback only as a last
resort -- so the fallback is never consulted for a parameter that declares a
unit, and it can be deleted once declarations cover every name.

Behaviour-preserving: where a parameter both declares a unit AND has a name
default, the two AGREE, so preferring the declared unit changes no resolved
unit (test_declared_unit_agrees_with_the_name_default).

Mutant (fallback consulted first): revert the describe_calculation_params line
to `_unit_from_name(key) or declared.get(key, "")` and the fallback is consulted
for parameters that already declare a unit -> test_fallback_not_consulted_...
reddens (it records the keys _unit_from_name is called with).

Full-dotted imports so CI-M selects this file for the construction_formulas /
formula_registry mutants.
"""
from __future__ import annotations

import pytest

import app.lib.construction_formulas as cf
import app.lib.formula_registry as fr
from app.lib.formula_registry import all_specs, parameters


# Every registered input that both declares a real unit AND has a name default.
_OVERLAP = sorted({
    (spec.name, name)
    for spec in all_specs()
    for name, p in parameters(spec).items()
    if name in fr.NAME_UNIT_DEFAULTS and (p.unit or "") not in ("", "-")
})


@pytest.mark.parametrize("formula, name", _OVERLAP or [("_none", "_none")])
def test_declared_unit_agrees_with_the_name_default(formula, name):
    if formula == "_none":
        pytest.skip("no declared param overlaps the name defaults")
    p = parameters(fr.get(formula))[name]
    # They agree -> declared-first resolves the SAME unit the fallback would,
    # so the reorder is behaviour-preserving and the default is redundant here.
    assert p.unit == fr.name_unit_default(name), (formula, name, p.unit)


@pytest.mark.parametrize("formula", ["concrete_volume", "beam_shear_simple", "calculate_evm"])
def test_fallback_not_consulted_when_a_param_declares_a_unit(monkeypatch, formula):
    spec = fr.get(formula)
    assert spec is not None, f"{formula!r} not registered"
    declared = {n: (p.unit or "") for n, p in parameters(spec).items()}
    called: list[str] = []
    real = cf._unit_from_name
    monkeypatch.setattr(cf, "_unit_from_name",
                        lambda name, _r=real: (called.append(name), _r(name))[1])
    cf.describe_calculation_params(spec.fn, name=formula)
    # The fallback must not be consulted for any parameter that declares a unit
    # (declared-first). The mutant (fallback-first) consults it for every
    # non-currency declared param -> this list is non-empty -> reddens.
    consulted_despite_declared = [k for k in called if declared.get(k, "") not in ("",)]
    assert consulted_despite_declared == [], (formula, consulted_despite_declared)


@pytest.mark.parametrize("name, expected", [
    ("pv", "currency"),
    ("rate_percent", "%"),
    ("column_diameter_mm", "mm"),
    ("not-a-known-name", ""),
])
def test_name_unit_default_lookup(name, expected):
    assert fr.name_unit_default(name) == expected


def test_the_override_table_moved_off_construction_formulas():
    assert not hasattr(cf, "_PARAM_UNIT_OVERRIDE")
