"""Arguments are bound by the registry's own declarations (F-CORE-UNJAM step
5: argument binding), against a synthetic formula only."""
from __future__ import annotations

import pytest

from app.lib import construction_formulas as cf
from app.lib import formula_registry as fr

NAME = "syn_value_check"


@pytest.fixture(autouse=True)
def synthetic():
    def syn_value_check(planned_value: float, earned_value: float, contract_sum: float = 0.0,
                        offset_m: float = 0.0):
        return {"ratio": earned_value / planned_value, "contract_sum": contract_sum, "offset_m": offset_m}
    fr._REGISTRY.pop(NAME, None)
    fr.formula(owner="base", description="Synthetic value check.", display_name="Synthetic value check",
               inputs={"planned_value": fr.Param(unit="currency", lo=0, hi=1e13, label="planned value"),
                       "earned_value": fr.Param(unit="currency", lo=0, hi=1e13, label="earned value"),
                       "contract_sum": fr.Param(unit="currency", lo=0, hi=1e13, label="contract sum"),
                       "offset_m": fr.Param(unit="m", lo=-100, hi=100, label="level offset")},
               outputs={"ratio": "ratio"})(syn_value_check)
    cf.CALCULATORS[NAME] = syn_value_check
    yield
    fr._REGISTRY.pop(NAME, None)
    cf.CALCULATORS.pop(NAME, None)


def _run(params):
    return cf.run_calculation(NAME, params)


@pytest.mark.parametrize("params,field,value", [
    ({"planned value": 100, "earned value": 50}, "ratio", 0.5),             # label as key
    ({"Planned Value": 100, "Earned-Value": 25}, "ratio", 0.25),             # case and separators
    ({"PV": 200, "EV": 50}, "ratio", 0.25),                                  # label initials
    ({"planned_value": "2 million", "earned_value": "1 million"}, "ratio", 0.5),  # magnitude words
    ({"planned_value": 10, "earned_value": 5, "contract_sum": "SAR 1.5 bn"}, "contract_sum", 1.5e9),
    ({"planned_value": 10, "earned_value": 5, "offset_m": "-3 m"}, "offset_m", -3.0),  # sign kept, m is metres
])
def test_inputs_bind_by_their_declarations(params, field, value):
    out = _run(dict(params))
    assert out["status"] == "success", out
    assert out["result"][field] == pytest.approx(value)


def test_a_magnitude_word_never_reads_a_unit_as_a_multiplier():
    assert fr._with_magnitude("12 m") == "12 m"
    assert fr._with_magnitude("5 km") == "5 km"
    assert fr._with_magnitude("2 kN") == "2 kN"
    assert fr._with_magnitude("2 million") == 2_000_000


def test_a_key_that_could_name_two_inputs_is_left_alone():
    # "value" is in two labels: it binds to neither, so the calculator asks.
    out = _run({"value": 100})
    assert out["status"] != "success"


def test_a_magnitude_word_scales_amounts_never_engineering_figures():
    def syn_load(load_kn: float, budget: float = 0.0):
        return {"load_kn": load_kn, "budget": budget}
    fr._REGISTRY.pop("syn_load", None)
    fr.formula(owner="base", description="Synthetic load.", display_name="Synthetic load",
               inputs={"load_kn": fr.Param(unit="kN", lo=0, hi=1e6, label="load"),
                       "budget": fr.Param(unit="currency", lo=0, hi=1e13, label="budget")},
               outputs={"load_kn": "kN"})(syn_load)
    try:
        bound = fr.bind_declared("syn_load", {"load_kn": "2 MN", "budget": "2 mn"})
        assert bound["load_kn"] == "2 MN"      # a meganewton is left to unit conversion
        assert bound["budget"] == 2_000_000    # money takes the magnitude
    finally:
        fr._REGISTRY.pop("syn_load", None)


def test_a_one_character_symbol_is_left_to_the_symbol_binder():
    bound = fr.bind_declared(NAME, {"P": 5, "planned_value": 1, "earned_value": 1})
    assert "P" in bound and bound["planned_value"] == 1
