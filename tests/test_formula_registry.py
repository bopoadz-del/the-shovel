"""Every formula declares its owner, a description, inputs and outputs.

The registry is built from those declarations; a turn wearing a hat is offered
base formulas plus that hat's own.
"""
from __future__ import annotations

import pytest

from app.lib import formula_registry as fr


def test_every_calculator_is_a_declared_formula():
    from app.lib.construction_formulas import CALCULATORS

    declared = {s.name for s in fr.all_specs()}
    assert set(CALCULATORS) == declared


def test_every_formula_has_an_owner_a_description_inputs_and_outputs():
    bad = [s.name for s in fr.all_specs()
           if s.owner not in fr.OWNERS or not s.description.strip() or s.outputs is None or s.inputs is None]
    assert bad == []
    assert all(s.outputs for s in fr.all_specs()), "a formula declares no outputs"


def test_a_declaration_without_owner_or_description_is_refused():
    with pytest.raises(ValueError):
        fr.formula(owner="", description="x", display_name="X", inputs={}, outputs={})
    with pytest.raises(ValueError):
        fr.formula(owner="base", description="  ", display_name="X", inputs={}, outputs={})
    with pytest.raises(ValueError):
        fr.formula(owner="general", description="x", display_name="X", inputs={}, outputs={})


def test_a_hat_is_offered_base_plus_its_own_formulas_only():
    design = {s.name for s in fr.specs_for("design")}
    safety = {s.name for s in fr.specs_for("safety")}
    base = {s.name for s in fr.specs_for(None)}
    owned_by_design = {s.name for s in fr.all_specs() if s.owner == "design"}
    assert base and base <= design and base <= safety
    assert owned_by_design <= design
    assert not (owned_by_design & safety)


def test_tolerance_check_is_a_base_formula():
    from app.lib.construction_formulas import run_calculation

    spec = fr.get("tolerance_check")
    assert spec is not None and spec.owner == "base"
    out = run_calculation("tolerance_check", {"measured": 10.4, "specified": 10.0, "tolerance_plus": 0.5})
    res = out.get("result", out)
    assert res["within_tolerance"] is True and abs(res["margin"] - 0.1) < 1e-9
    out = run_calculation("tolerance_check", {"measured": 9.3, "specified": 10.0,
                                               "tolerance_plus": 0.5, "tolerance_minus": 0.5})
    assert out.get("result", out)["within_tolerance"] is False
