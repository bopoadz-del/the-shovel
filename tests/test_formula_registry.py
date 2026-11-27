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


# ── inputs that declare what they are "per" (Param.per) ──────────────────────
# A consumption per unit of work is per the quantity; a price per resource
# unit is per what the consumption yields. Where both inputs state a unit the
# registry converts a denominator of the same kind and refuses another kind.

def _chain(values):
    return fr.convert_written_units("resource_cost_line", dict(values))


@pytest.mark.parametrize("values,key,expect", [
    # litres consumed, price per m3: the price becomes a price per litre
    ({"quantity": "640 m2", "consumption_per_unit": "22 L/m2", "price_per_resource_unit": "310 per m3"},
     "price_per_resource_unit", 0.31),
    # kg consumed, price per tonne: a price per kg
    ({"quantity": "100 m3", "consumption_per_unit": "35 kg/m3", "price_per_resource_unit": "2800 AED/t"},
     "price_per_resource_unit", 2.8),
    # quantity in ft2, consumption per m2: a consumption per ft2
    ({"quantity": "640 ft2", "consumption_per_unit": "0.022 m3/m2", "price_per_resource_unit": 310},
     "consumption_per_unit", 0.022 * 0.09290304),
    # quantity in m, consumption per mm: a consumption per m
    ({"quantity": "12 m", "consumption_per_unit": "0.5 kg/mm", "price_per_resource_unit": 3},
     "consumption_per_unit", 500.0),
])
def test_a_chained_denominator_of_the_same_kind_is_converted(values, key, expect):
    out, conversions, rejected = _chain(values)
    assert rejected == []
    assert out[key] == pytest.approx(expect)
    chained = [c for c in conversions if c["parameter"] == key and c["to_unit"] != c["from_unit"]]
    assert len(chained) == 1 and chained[0]["value_out"] == pytest.approx(expect)


@pytest.mark.parametrize("values,key,per,dims", [
    ({"quantity": "640 m3", "consumption_per_unit": "0.022 m3/m2", "price_per_resource_unit": 310},
     "consumption_per_unit", "quantity", ("area", "volume")),
    ({"quantity": "640 m2", "consumption_per_unit": "0.022 m3/m2", "price_per_resource_unit": "310 AED/kg"},
     "price_per_resource_unit", "consumption_per_unit", ("mass", "volume")),
    ({"quantity": "40 m", "consumption_per_unit": "3 kg/m", "price_per_resource_unit": "9 per m3"},
     "price_per_resource_unit", "consumption_per_unit", ("volume", "mass")),
    ({"quantity": "50 m2", "consumption_per_unit": "12.5 nr/m2", "price_per_resource_unit": "0.8 per kg"},
     "price_per_resource_unit", "consumption_per_unit", ("mass", "count")),
])
def test_a_chained_denominator_of_another_kind_is_refused(values, key, per, dims):
    out, _, rejected = _chain(values)
    assert [r["parameter"] for r in rejected] == [key]
    row = rejected[0]
    assert row["reason"] == "unit_chain" and row["per"] == per
    assert (row["given_dimension"], row["per_dimension"]) == dims
    assert out[key] == values[key]  # left as supplied, so it is not also missing


@pytest.mark.parametrize("values", [
    # a bare figure on either side: nothing stated to check, nothing converted
    {"quantity": 640, "consumption_per_unit": "22 L/ft2", "price_per_resource_unit": 310},
    {"quantity": "640 m3", "consumption_per_unit": 0.022, "price_per_resource_unit": "310 AED/kg"},
    # the same unit on both sides: no conversion
    {"quantity": "640 m2", "consumption_per_unit": "0.022 m3/m2", "price_per_resource_unit": "310 per m3"},
    # a count of pieces priced per piece
    {"quantity": "50 m2", "consumption_per_unit": "12.5 nr/m2", "price_per_resource_unit": "0.8 per nr"},
])
def test_the_chain_does_not_fire_without_two_stated_units_of_different_scale(values):
    out, conversions, rejected = _chain(values)
    assert rejected == []
    assert all(c["from_unit"] == c["to_unit"] for c in conversions)


def test_the_chain_skips_an_input_already_refused():
    # the consumption itself is refused: no second row for the price's chain
    out, _, rejected = _chain({"quantity": "640 m2", "consumption_per_unit": "5 kN/m2",
                               "price_per_resource_unit": "310 AED/kg"})
    assert [r["parameter"] for r in rejected] == ["consumption_per_unit"]
    out, _, rejected = _chain({"quantity": "640 kN", "consumption_per_unit": "0.022 m3/kg",
                               "price_per_resource_unit": "310 AED/L"})
    assert [r["parameter"] for r in rejected] == ["quantity"]


@pytest.mark.parametrize("value,parts", [
    ("22 L/m2", ("L", "m2")), ("640 m2", ("m2", "")), ("310 AED/m3", ("AED", "m3")),
    (640, None), ("40 zz", None),
])
def test_stated_parts_split_what_is_measured_from_what_it_is_per(value, parts):
    assert fr._stated_parts(value) == parts


def test_stated_parts_reads_an_implied_currency():
    head, tail = fr._stated_parts("310 per m3")
    assert tail == "m3" and head
    assert fr._stated_parts("9 per t")[1] == "t"


@pytest.mark.parametrize("declared,written,chained,fills", [
    ("count", "m3", False, True),
    ("count/d", "m2/d", False, True),
    ("count/d", "m2/kN", False, False),          # a force is not the declared time
    ("count/count", "m3/m2", True, True),
    ("count/count", "m3/m2", False, False),      # a denominator needs a declared per
    ("currency/count", "AED/m3", True, True),
    ("currency/count", "AED/m3", False, False),
    ("currency/count", "AED/kN", True, False),   # a force is not counted
    ("count/count", "m3", True, False),          # parts differ
    ("count", "count", True, False),             # equal is not a fill
])
def test_a_measure_fills_a_count_place_only_where_it_may(declared, written, chained, fills):
    from app.agents.base.formulas.construction_formulas_planning import parse_unit

    assert fr._measure_fills_count(parse_unit(declared), parse_unit(written), chained) is fills
    assert fr._measure_fills_count(None, parse_unit(written), chained) is False


def test_an_undeclared_input_keeps_refusing_a_price_per_measure():
    # resource_line_cost's material rate declares no per: a price per kg or
    # per m3 is still refused, whatever the quantity's unit
    for rate, reason in (("45 AED/kg", "wrong_unit"), ("45 per m3", "unknown_unit")):
        _, _, rejected = fr.convert_written_units(
            "resource_line_cost", {"quantity": "500 m2", "daily_output": "20 m2/d",
                                   "day_rate": 800, "material_rate_per_unit": rate})
        assert [(r["parameter"], r["reason"]) for r in rejected] == [("material_rate_per_unit", reason)]


def test_a_per_that_names_no_input_is_a_schema_gap(monkeypatch):
    def calc(a: float, b: float) -> dict:
        return {}

    good = fr.FormulaSpec(name="chain_ok", fn=calc, owner="base", description="x", display_name="X",
                          params={"a": fr.Param("count", 0, 1, label="a"),
                                  "b": fr.Param("count/count", 0, 1, label="b", per="a")})
    bad = fr.FormulaSpec(name="chain_bad", fn=calc, owner="base", description="x", display_name="X",
                         params={"a": fr.Param("count", 0, 1, label="a", per="a"),
                                 "b": fr.Param("count/count", 0, 1, label="b", per="c")})
    monkeypatch.setattr(fr, "all_specs", lambda: [good, bad])
    gaps = [g for g in fr.schema_gaps() if g.startswith("formula:")]
    assert gaps == ["formula:chain_bad.a:per", "formula:chain_bad.b:per"]
    assert fr.parameters(good)["b"].per == "a" and fr.parameters(good)["a"].per == ""
