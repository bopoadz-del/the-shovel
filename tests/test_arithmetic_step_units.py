"""The arithmetic step reads its units through the one unit table: a fraction
is scaled in a product or quotient, a count is a plain number in a product,
and a length unit to a power the table names is written as that unit -- so
the next formula on the sheet can read what the step wrote."""
import math

import pytest

from app.agents.base.formulas import arithmetic
from app.agents.base.formulas.arithmetic import arithmetic_step
from app.agents.driver import sheet as ws


# --- _as_plain ------------------------------------------------------------------

@pytest.mark.parametrize("value,unit,plain", [(10, "%", 0.1), (5, "percent", 0.05), (2, "‰", 0.002),
                                              (0.25, "fraction", 0.25)])
def test_as_plain_scales_a_fraction_by_the_table(value, unit, plain):
    got, composes, reading = arithmetic._as_plain(value, unit, "multiply")
    assert got == pytest.approx(plain) and composes == "-"
    assert reading == f"{value:g} {unit} = {plain:g}"
    got, composes, _ = arithmetic._as_plain(value, unit, "divide")
    assert got == pytest.approx(plain) and composes == "-"


@pytest.mark.parametrize("value,unit", [(40, "nr"), (7, "count")])
def test_as_plain_drops_a_count_from_a_product_only(value, unit):
    assert arithmetic._as_plain(value, unit, "multiply") == (value, "-", "")
    # a quotient keeps it: "per nr" is what the rate is per
    assert arithmetic._as_plain(value, unit, "divide") == (value, unit, "")


@pytest.mark.parametrize("unit", ["m", "m3", "currency", "-", "nr*m", "kg/m"])
def test_as_plain_leaves_any_other_unit_as_given(unit):
    for op in ("multiply", "divide"):
        assert arithmetic._as_plain(3.5, unit, op) == (3.5, unit, "")


# --- _length_power / _powered_name ------------------------------------------------

@pytest.mark.parametrize("unit,expect", [("m", ("m", 1)), ("mm2", ("mm", 2)), ("ft3", ("ft", 3)),
                                         ("m4", ("m", 4)), (" cm ", ("cm", 1))])
def test_length_power_reads_a_powered_length(unit, expect):
    assert arithmetic._length_power(unit) == expect


@pytest.mark.parametrize("unit", ["kg", "nr", "%", "-", "m5", "ha", "currency", "m*m", "", "l", "mm1x"])
def test_length_power_is_none_for_anything_else(unit):
    # m5 is not in the table; ha and l are an area and a volume with no base length
    assert arithmetic._length_power(unit) is None


@pytest.mark.parametrize("base,power,name", [("mm", 2, "mm2"), ("m", 3, "m3"), ("ft", 1, "ft")])
def test_powered_name_names_what_the_table_knows(base, power, name):
    assert arithmetic._powered_name(base, power) == name


@pytest.mark.parametrize("base,power", [("m", 5), ("mm", 6), ("m", 0), ("m", -1)])
def test_powered_name_is_none_past_the_table(base, power):
    assert arithmetic._powered_name(base, power) is None


# --- _compose ---------------------------------------------------------------------

@pytest.mark.parametrize("a,b,op,unit", [
    ("mm", "mm", "multiply", "mm2"), ("m", "m2", "multiply", "m3"), ("M", "m", "multiply", "M2"),
    ("m3", "m", "multiply", "m4"), ("m3", "m", "divide", "m2"), ("ft3", "ft2", "divide", "ft"),
    ("m2", "m2", "divide", "-"),
])
def test_compose_reduces_one_length_unit_to_a_power(a, b, op, unit):
    assert arithmetic._compose(a, b, op) == unit


@pytest.mark.parametrize("a,b,op,unit", [
    ("m", "mm", "multiply", "m*mm"),      # two length units: not one power
    ("m2", "m3", "multiply", "m2*m3"),    # power 5 is not in the table
    ("m", "m3", "divide", "m/m3"),        # a negative power is not named
    ("currency", "m2", "divide", "currency/m2"),
    ("kg", "kg", "multiply", "kg*kg"),    # not a length
])
def test_compose_keeps_the_written_composition_otherwise(a, b, op, unit):
    assert arithmetic._compose(a, b, op) == unit


# --- arithmetic_step --------------------------------------------------------------

@pytest.mark.parametrize("amount,pct,expect", [(1_250_000, 10, 125_000), (20_000_000, 5, 1_000_000)])
def test_a_percent_of_an_amount_is_the_amount_times_the_fraction(amount, pct, expect):
    out = arithmetic_step("multiply", amount, pct, "currency", "%")
    assert out["result"] == pytest.approx(expect) and out["result_unit"] == "currency"
    assert out["method"].endswith(f"(taking {pct:g} % = {pct / 100:g})")
    # either order
    assert arithmetic_step("multiply", pct, amount, "%", "currency")["result"] == pytest.approx(expect)


@pytest.mark.parametrize("amount,pct", [(500, 10), (90, 4.5)])
def test_dividing_by_a_percent_divides_by_the_fraction(amount, pct):
    out = arithmetic_step("divide", amount, pct, "currency", "%")
    assert out["result"] == pytest.approx(amount / (pct / 100)) and out["result_unit"] == "currency"


def test_two_percents_in_one_product_are_both_read():
    out = arithmetic_step("multiply", 50, 20, "%", "%")
    assert out["result"] == pytest.approx(0.1) and out["result_unit"] == "-"
    assert "50 % = 0.5; 20 % = 0.2" in out["method"]


@pytest.mark.parametrize("a,b", [(10, 5), (2.5, 0.5)])
def test_percents_add_as_percents(a, b):
    # a sum is not a product: the percent is kept, and nothing is scaled
    out = arithmetic_step("add", a, b, "%", "%")
    assert out["result"] == pytest.approx(a + b) and out["result_unit"] == "%"
    assert "taking" not in out["method"]


@pytest.mark.parametrize("unit", ["%", "nr"])
def test_a_fraction_or_count_is_still_not_added_to_an_amount(unit):
    assert "convert" in arithmetic_step("add", 100, 10, "currency", unit)["error"]


@pytest.mark.parametrize("bars,length", [(40, 3.92), (12, 6.0)])
def test_a_count_times_a_length_is_a_length(bars, length):
    out = arithmetic_step("multiply", bars, length, "nr", "m")
    assert out["result"] == pytest.approx(bars * length) and out["result_unit"] == "m"
    assert "taking" not in out["method"]


@pytest.mark.parametrize("a,b", [(12, 12), (16, 20)])
def test_a_length_times_the_same_length_unit_is_an_area(a, b):
    out = arithmetic_step("multiply", a, b, "mm", "mm")
    assert out["result"] == pytest.approx(a * b) and out["result_unit"] == "mm2"


def test_a_percent_divisor_that_is_zero_is_refused():
    assert "zero" in arithmetic_step("divide", 5, 0, "currency", "%")["error"]


# --- the public path: a step's unit is read by the next formula on the sheet -----

def _sheet(givens, plan):
    s = ws.new_sheet("t-units", "q")
    s, problems = ws.apply_update(s, {"givens": givens, "plan": plan})
    assert problems == []
    return s


@pytest.mark.parametrize("bars,length,dia", [(40, 3.92, 12), (25, 6.0, 16)])
def test_a_count_times_length_step_feeds_a_mass_formula(bars, length, dia):
    s = _sheet([{"id": "g1", "value": bars, "unit": "nr"}, {"id": "g2", "value": length, "unit": "m"},
                {"id": "g3", "value": dia, "unit": "mm"}],
               [{"id": "s1", "entry": "formula:arithmetic_step",
                 "inputs": {"operator": "multiply", "a": "g1", "b": "g2"}},
                {"id": "s2", "entry": "formula:rebar_weight", "output": "total_mass_kg",
                 "inputs": {"bar_diameter_mm": "g3", "total_length_m": "s1"}}])
    s = ws.run_step(ws.run_step(s, "s1", "c1"), "s2", "c2")
    steps = {st["id"]: st for st in s["steps"]}
    assert steps["s2"]["status"] == "done", steps["s2"].get("error")
    unit_mass = math.pi / 4 * (dia / 1000) ** 2 * 7850
    assert steps["s2"]["result"]["value"] == pytest.approx(unit_mass * bars * length, rel=1e-3)


@pytest.mark.parametrize("amount,pct", [(1_125_000, 5), (800_000, 10)])
def test_a_percent_step_then_a_subtraction_gives_the_net(amount, pct):
    s = _sheet([{"id": "g1", "value": amount, "unit": "currency"}, {"id": "g2", "value": pct, "unit": "%"}],
               [{"id": "s1", "entry": "formula:arithmetic_step",
                 "inputs": {"operator": "multiply", "a": "g1", "b": "g2"}},
                {"id": "s2", "entry": "formula:arithmetic_step",
                 "inputs": {"operator": "subtract", "a": "g1", "b": "s1"}}])
    s = ws.run_step(ws.run_step(s, "s1", "c1"), "s2", "c2")
    steps = {st["id"]: st for st in s["steps"]}
    assert steps["s1"]["result"]["value"] == pytest.approx(amount * pct / 100)
    assert steps["s2"]["result"] == {"name": "result", "value": pytest.approx(amount * (1 - pct / 100)),
                                     "unit": "currency"}


# --- a value written with its unit inside it (the unit field is its sibling) ------

def _calc(**inputs):
    from app.lib.construction_formulas import run_calculation

    return run_calculation("arithmetic_step", inputs)


@pytest.mark.parametrize("a,unit,number,b,b_unit,op,want", [
    ("10.32256 d", "d", 10.32256, 2, "count", "divide", 5.16128),
    ("-2.5 m", "m", -2.5, "-4.1 m", "m", "subtract", 1.6),
])
def test_a_value_written_in_its_unit_fields_unit_is_that_number(a, unit, number, b, b_unit, op, want):
    out = _calc(operator=op, a=a, b=b, a_unit=unit, b_unit=b_unit)
    assert out["status"] == "success", out
    assert out["result"]["result"] == pytest.approx(want)
    assert out["result"]["method"].startswith(f"{number:g} {unit} ")
    assert not out.get("conversions")


@pytest.mark.parametrize("a,unit,converted", [("3 ft", "m", 0.9144), ("500 mm", "m", 0.5)])
def test_a_value_written_in_another_unit_of_the_same_kind_is_converted(a, unit, converted):
    out = _calc(operator="add", a=a, b=1, a_unit=unit, b_unit=unit)
    assert out["status"] == "success", out
    assert out["result"]["result"] == pytest.approx(converted + 1)
    assert out["result"]["result_unit"] == unit
    conv = out["conversions"][0]
    assert conv["parameter"] == "a" and conv["to_unit"] == unit
    assert conv["value_out"] == pytest.approx(converted)


@pytest.mark.parametrize("b,unit,given_dim", [("2 kg", "m", "mass"), ("2 m2", "nr", "area")])
def test_a_value_written_in_another_kind_is_refused_naming_both(b, unit, given_dim):
    out = _calc(operator="add", a=1, b=b, a_unit=unit, b_unit=unit)
    assert out["status"] == "error" and "result" not in out
    rejected = out["rejected"][0]
    assert rejected["parameter"] == "b" and rejected["reason"] == "wrong_unit"
    assert rejected["given_dimension"] == given_dim
    assert b in out["error"] and unit in out["error"]


@pytest.mark.parametrize("sibling", ["", "-", None])
@pytest.mark.parametrize("a,unit,number", [("10.32256 d", "d", 10.32256), ("7.5 h", "h", 7.5)])
def test_a_blank_unit_field_takes_the_unit_written_in_the_value(sibling, a, unit, number):
    inputs = {"operator": "divide", "a": a, "b": 3, "b_unit": "count"}
    if sibling is not None:
        inputs["a_unit"] = sibling
    out = _calc(**inputs)
    assert out["status"] == "success", out
    assert out["result"]["result"] == pytest.approx(number / 3)
    assert out["result"]["result_unit"] == f"{unit}/count"


@pytest.mark.parametrize("a,unit", [(4, "d"), ("6.25", "m")])
def test_a_bare_value_keeps_the_unit_field_as_given(a, unit):
    out = _calc(operator="multiply", a=a, b=2, a_unit=unit, b_unit="-")
    assert out["status"] == "success", out
    assert out["result"]["result"] == pytest.approx(float(a) * 2)
    assert out["result"]["result_unit"] == unit and not out.get("conversions")


def _one_step(givens, inputs):
    s = _sheet(givens, [{"id": "s1", "entry": "formula:arithmetic_step", "inputs": inputs}])
    return ws.run_step(s, "s1", "c1")["steps"][0]


@pytest.mark.parametrize("unit_field", [{"a_unit": "d"}, {"a_unit": "-"}, {}])
@pytest.mark.parametrize("a,unit,number", [("10.32256 d", "d", 10.32256), ("-2.5 m", "m", -2.5)])
def test_a_sheet_literal_written_with_its_unit_runs(unit_field, a, unit, number):
    if unit_field.get("a_unit") not in (None, "-"):
        unit_field = {"a_unit": unit}
    step = _one_step([], {"operator": "divide", "a": a, "b": 2, "b_unit": "count", **unit_field})
    assert step["status"] == "done", step.get("error")
    assert step["result"] == {"name": "result", "value": pytest.approx(number / 2), "unit": f"{unit}/count"}


@pytest.mark.parametrize("value,unit", [(10.32256, "d"), (42.0, "m")])
def test_a_sheet_given_with_its_unit_runs(value, unit):
    step = _one_step([{"id": "g1", "value": value, "unit": unit}],
                     {"operator": "divide", "a": "g1", "b": 4, "b_unit": "count"})
    assert step["status"] == "done", step.get("error")
    assert step["result"]["value"] == pytest.approx(value / 4)


@pytest.mark.parametrize("a,unit", [("2 kg", "m"), ("3 m2", "d")])
def test_a_sheet_literal_of_another_kind_fails_the_step(a, unit):
    step = _one_step([], {"operator": "add", "a": a, "b": 1, "a_unit": unit, "b_unit": unit})
    assert step["status"] == "failed" and "result" not in step
    assert step["error"] == "first value: wrong_unit"
