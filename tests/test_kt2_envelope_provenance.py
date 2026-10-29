"""K-T2: every run_calculation envelope carries provenance + standard; kernels are 'certified'."""
from __future__ import annotations

import pytest

from app.lib import construction_formulas as cf
from app.lib import formula_registry as fr

SPECS = fr.all_specs()


@pytest.mark.parametrize("spec", SPECS, ids=[s.name for s in SPECS])
def test_every_registered_formula_gets_provenance_and_standard(spec):
    out = cf.envelope_provenance(spec.name, {}, spec.fn)
    assert out["provenance"] in {"kernel", "reference_table"}
    assert isinstance(out["standard"], str) and out["standard"]
    assert out["certified"] is (out["provenance"] == "kernel")


@pytest.mark.parametrize("name,params", [
    ("concrete_volume", {"length_m": 10, "width_m": 5, "thickness_m": 0.2}),
    ("concrete_volume", {"length_m": 30, "width_m": 20, "thickness_m": 1.5}),
])
def test_run_calculation_envelope_is_certified_kernel(name, params):
    env = cf.run_calculation(name, params)
    assert env["status"] == "success", env
    assert env["provenance"] == "kernel"
    assert env["certified"] is True
    assert env["standard"]


@pytest.mark.parametrize("name", ["made_up_calc", "another_unknown_formula"])
def test_unregistered_name_is_improvised_never_certified(name):
    out = cf.envelope_provenance(name, {"value": 1})
    assert out == {"provenance": "improvised", "standard": "none declared", "certified": False}


@pytest.mark.parametrize("key,val", [("standard", "BS EN 1992-1-1"), ("code", "ACI 318-19")])
def test_standard_stated_by_the_result_wins(key, val):
    spec = SPECS[0]
    assert cf.envelope_provenance(spec.name, {key: val}, spec.fn)["standard"] == val


@pytest.mark.parametrize("module,expect", [
    ("app.agents.design.formulas.construction_formulas_reference_tables", "reference_table"),
    ("app.agents.base.formulas.construction_formulas_quantities", "kernel"),
])
def test_reference_table_module_is_not_certified(module, expect):
    spec = SPECS[0]
    def fake(): ...
    fake.__module__ = module
    out = cf.envelope_provenance(spec.name, {}, fake)
    assert out["provenance"] == expect
    assert out["certified"] is (expect == "kernel")
