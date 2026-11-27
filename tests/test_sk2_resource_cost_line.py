"""S-K2 item 7: resource-based cost line = qty x consumption x waste x cited price.

Every expected value is worked out here from the method on inputs this file
builds itself; nothing is read back from the engine's arithmetic. Each
decision is checked on two inputs (mutation gate).
"""
from __future__ import annotations

import pytest

from app.agents.hats.quantities.formulas.construction_formulas_quantities import (
    resource_cost_line,
)

# Two seeded lines: (quantity, consumption per unit, price per resource unit).
LINES = [
    (240.0, 0.35, 410.0),
    (1_850.0, 12.5, 3.2),
]


def _base(q, c, p):
    return {"quantity": q, "consumption_per_unit": c, "price_per_resource_unit": p}


@pytest.mark.parametrize("q,c,p", LINES)
def test_line_without_waste_or_markup_is_qty_times_consumption_times_price(q, c, p):
    out = resource_cost_line(**_base(q, c, p))
    assert out["resource_quantity"] == pytest.approx(q * c)
    assert out["resource_cost"] == pytest.approx(round(q * c * p, 2))
    assert out["line_total"] == pytest.approx(round(q * c * p, 2))
    assert out["waste_factor"] is None
    assert out["markup_amount"] is None


@pytest.mark.parametrize("q,c,p", LINES)
@pytest.mark.parametrize("waste", [5.0, 12.5])
def test_waste_multiplies_the_resource_quantity(q, c, p, waste):
    out = resource_cost_line(**_base(q, c, p), waste_percent=waste)
    factor = 1 + waste / 100
    assert out["waste_factor"] == pytest.approx(factor)
    assert out["resource_quantity_net"] == pytest.approx(q * c)
    assert out["resource_quantity"] == pytest.approx(q * c * factor)
    assert out["resource_cost"] == pytest.approx(round(q * c * factor * p, 2))


@pytest.mark.parametrize("q,c,p", LINES)
def test_absent_waste_is_said_not_assumed(q, c, p):
    out = resource_cost_line(**_base(q, c, p))
    assert any("No waste allowance supplied" in w for w in out["working"])
    with_waste = resource_cost_line(**_base(q, c, p), waste_percent=5.0)
    assert not any("No waste allowance supplied" in w for w in with_waste["working"])


@pytest.mark.parametrize("q,c,p", LINES)
def test_no_default_markup(q, c, p):
    out = resource_cost_line(**_base(q, c, p), waste_percent=5.0)
    assert out["markup_amount"] is None
    assert out["line_total"] == out["resource_cost"]
    assert any("no default markup" in w for w in out["working"])


@pytest.mark.parametrize("q,c,p", LINES)
@pytest.mark.parametrize("markup", [0.0, 15.0])
def test_supplied_markup_is_added_on_the_resource_cost(q, c, p, markup):
    out = resource_cost_line(**_base(q, c, p), waste_percent=5.0,
                             markup_percent=markup)
    cost = q * c * 1.05 * p
    assert out["markup_amount"] == pytest.approx(round(cost * markup / 100, 2))
    assert out["line_total"] == pytest.approx(round(cost * (1 + markup / 100), 2))
    assert not any("no default markup" in w for w in out["working"])


@pytest.mark.parametrize("absent", ["quantity", "consumption_per_unit",
                                    "price_per_resource_unit"])
@pytest.mark.parametrize("q,c,p", LINES)
def test_a_missing_project_fact_is_named_not_defaulted(absent, q, c, p):
    args = _base(q, c, p)
    args[absent] = None
    out = resource_cost_line(**args)
    assert out["missing"] == [absent]
    assert "line_total" not in out


@pytest.mark.parametrize("negative", ["quantity", "consumption_per_unit",
                                      "price_per_resource_unit",
                                      "waste_percent", "markup_percent"])
@pytest.mark.parametrize("value", [-1.0, -0.01])
def test_negative_inputs_are_refused(negative, value):
    args = {**_base(*LINES[0]), "waste_percent": 5.0, "markup_percent": 10.0}
    args[negative] = value
    out = resource_cost_line(**args)
    assert "error" in out
    assert "line_total" not in out


@pytest.mark.parametrize("q,c,p", LINES)
def test_zero_inputs_are_allowed(q, c, p):
    out = resource_cost_line(quantity=0.0, consumption_per_unit=c,
                             price_per_resource_unit=p, waste_percent=0.0)
    assert out["line_total"] == 0.0
    assert out["waste_factor"] == 1.0


@pytest.mark.parametrize("q,c,p", LINES)
def test_working_prints_each_step(q, c, p):
    out = resource_cost_line(**_base(q, c, p), waste_percent=5.0,
                             markup_percent=10.0)
    text = " | ".join(out["working"])
    for step in ("Net resource", "With waste", "Resource cost", "Markup"):
        assert step in text
    assert f"{round(q * c * 1.05 * p * 1.1, 2):,.2f}" in text


@pytest.mark.parametrize("q,c,p", LINES)
def test_registered_and_run_through_the_registry(q, c, p):
    from app.lib.construction_formulas import CALCULATORS

    fn = CALCULATORS["resource_cost_line"]
    assert fn(**_base(q, c, p))["line_total"] == pytest.approx(round(q * c * p, 2))


def test_registry_module_reexports_the_hat_formula():
    from app.lib import construction_formulas_quantities as lib

    assert lib.resource_cost_line is resource_cost_line


_FIGURE = r"(-?[\d,]*\.?\d+)"


def _figures(line: str) -> list[float]:
    import re
    return [float(t.replace(",", "")) for t in re.findall(_FIGURE, line)]


@pytest.mark.parametrize("q,c,p", LINES + [(2_500_000.0, 1.0, 0.125),
                                           (1_234_567.0, 0.004, 17.375)])
def test_printed_figures_multiply_to_the_printed_result(q, c, p):
    # The working is read as a person reads it: the figures as printed, not
    # the engine's floats. Each step must reconcile on its own figures.
    out = resource_cost_line(**_base(q, c, p), waste_percent=7.5)
    net, waste, cost = (next(w for w in out["working"] if w.startswith(head))
                        for head in ("Net resource", "With waste", "Resource cost"))
    a, b, r = _figures(net)
    assert (a, b) == pytest.approx((q, c)) and r == pytest.approx(a * b)
    _, n, f, g = _figures(waste)
    assert n == pytest.approx(r) and g == pytest.approx(n * f)
    g2, price, total = _figures(cost)
    assert g2 == pytest.approx(g) and price == pytest.approx(p)
    assert total == pytest.approx(g2 * price, abs=0.005)
    assert "e+" not in " ".join(out["working"]).lower()


@pytest.mark.parametrize("q,c,p", [(20_000_000_000.0, 1.0, 0.5),
                                   (500.0, 0.00004, 90.0)])
def test_a_figure_beyond_ten_digits_or_below_a_ten_thousandth_prints_in_full(q, c, p):
    # Ten significant figures in ``g`` form switch to an exponent at 1e10 and
    # below 1e-4; the working must still print every figure positionally so a
    # reader can multiply them.
    import re
    out = resource_cost_line(**_base(q, c, p))
    text = " ".join(out["working"])
    assert not re.search(r"\de[+-]?\d", text.lower())
    net = next(w for w in out["working"] if w.startswith("Net resource"))
    a, b, r = _figures(net)
    assert (a, b) == pytest.approx((q, c)) and r == pytest.approx(q * c)



# --- Units on the rate and the price, every printed step reconciles, and no
# --- negative zero. Inputs built here; expected values worked by hand.

def _chat(params):
    import asyncio

    from app.agents.base.tools import handle_construction_calc
    from app.agents.core.tool_registry import ToolCall

    call = ToolCall(agent=None, name="construction_calc",
                    args={"calculation": "resource_cost_line", "params": dict(params)},
                    user_message="Calculate resource_cost_line", history=[])
    return asyncio.run(handle_construction_calc(call))["result"]


@pytest.mark.parametrize("q,rate,price,expect_qty,expect_total", [
    # 640 m2 x 0.022 m3/m2 = 14.08 m3; x 310 per m3 = 4,364.80
    (640, "0.022 m3/m2", "310 per m3", 14.08, 4_364.80),
    # 95 m2 x 1.08 m2/m2 = 102.6 m2; x 42.5 AED/m2 = 4,360.50
    (95, "1.08 m2/m2", "42.5 AED/m2", 102.6, 4_360.50),
    # 12 m3 x 35 kg/m3 = 420 kg; x 3.2 per kg = 1,344.00
    (12, "35 kg/m3", "3.2 per kg", 420.0, 1_344.00),
    # 50 m2 x 12.5 nr/m2 = 625 nr; bare price 0.8 = 500.00
    (50, "12.5 nr/m2", 0.8, 625.0, 500.00),
])
def test_a_rate_and_price_stated_with_their_units_run_through_the_chat_door(
        q, rate, price, expect_qty, expect_total):
    result = _chat({"quantity": q, "consumption_per_unit": rate,
                    "price_per_resource_unit": price})
    assert result.get("status") == "success", result
    body = result["result"]
    assert body["resource_quantity"] == pytest.approx(expect_qty)
    assert body["line_total"] == pytest.approx(expect_total)
    carried = {c["parameter"]: c for c in result.get("conversions") or []}
    assert carried["consumption_per_unit"]["value_out"] == pytest.approx(
        expect_qty / q)


@pytest.mark.parametrize("field,value", [
    ("consumption_per_unit", "5 kN/m2"),      # a force is not a counted measure
    ("consumption_per_unit", "0.022 m3/s"),   # nor is a time in the denominator
    ("consumption_per_unit", "0.4 m3"),       # one part where two are declared
    ("price_per_resource_unit", "310 per kN"),
    ("price_per_resource_unit", "310 kg/m3"),  # not a currency at all
])
def test_a_unit_that_is_not_a_counted_measure_is_still_refused(field, value):
    params = {"quantity": 640, "consumption_per_unit": 0.022,
              "price_per_resource_unit": 310}
    params[field] = value
    result = _chat(params)
    assert result.get("needs_input") is True, result
    assert "line_total" not in (result.get("result") or {})
    # the refusal names the kind of figure wanted, never a bare "count" unit
    assert " in count" not in result["error"] and "in per count" not in result["error"]


@pytest.mark.parametrize("q,c,p,waste", [
    (1_000_000.0, 1.0, 1_000.0, 3.3333333),
    (1_000.0, 1.0, 7.0, 7.123456),
    (3_217.75, 0.0137, 412.9, 2.718281828459),
    # a net whose eleventh digit would move the printed gross by one
    (1_234_567.891, 0.0137, 3.1, 3.3333333),
    (987_654.321, 0.0137, 2.0, 9.87654),
])
def test_a_waste_percent_with_many_decimals_still_reconciles(q, c, p, waste):
    from decimal import Decimal

    out = resource_cost_line(**_base(q, c, p), waste_percent=waste)
    step = next(w for w in out["working"] if w.startswith("With waste"))
    _, n, f, g = _figures(step)
    # the factor is written in full: 1 + waste / 100, every digit
    assert Decimal(step.split(" x ")[1].split(" =")[0]) == 1 + Decimal(repr(waste)) / 100
    # exact on the printed figures: printed net x printed factor, written to
    # ten significant figures, is the printed gross digit for digit
    import re
    net_s, f_s, g_s = (t.replace(",", "") for t in
                       re.findall(r"[\d,]*\.?\d+", step.split("% = ")[1]))
    assert Decimal(f"{Decimal(net_s) * Decimal(f_s):.10g}") == Decimal(g_s), step
    assert g == pytest.approx(n * f, rel=1e-9)
    cost = next(w for w in out["working"] if w.startswith("Resource cost"))
    g2, price, total = _figures(cost)
    assert g2 == g and total == pytest.approx(g2 * price, abs=0.005)


@pytest.mark.parametrize("zero_field", ["quantity", "consumption_per_unit",
                                        "price_per_resource_unit"])
@pytest.mark.parametrize("q,c,p", LINES)
def test_a_negative_zero_prints_and_returns_zero(zero_field, q, c, p):
    import math

    args = {**_base(q, c, p), "waste_percent": -0.0, "markup_percent": -0.0}
    args[zero_field] = -0.0
    out = resource_cost_line(**args)
    text = " | ".join(out["working"])
    assert "-0" not in text, text
    for key in ("resource_quantity", "resource_cost", "line_total", "markup_amount"):
        assert math.copysign(1.0, out[key]) == 1.0, (key, out[key])
    assert out["line_total"] == 0.0


@pytest.mark.parametrize("zero_field", ["quantity", "price_per_resource_unit"])
def test_a_negative_zero_through_the_registry_is_zero(zero_field):
    import math

    from app.lib.construction_formulas import run_calculation

    args = _base(*LINES[1])
    args[zero_field] = -0.0
    res = run_calculation("resource_cost_line", args)
    body = res.get("result", res)
    assert math.copysign(1.0, body["line_total"]) == 1.0, res
    assert "-0" not in " ".join(body["working"])


# ── the question a refused unit gets (app.lib.source_labels.input_question) ──
# A counted input has no unit of its own to name, so the sentence names its
# kind; every other input still names its unit. Through the registry.

@pytest.mark.parametrize("calculation,inputs,question", [
    # counted: a count per count, a price per count
    ("resource_cost_line", {"quantity": 100, "consumption_per_unit": "3 kg", "price_per_resource_unit": 40},
     "The resource consumed per unit of work is a count per count, and 3 kg is a mass, so it does not convert. "
     "What resource consumed per unit of work should I use?"),
    ("resource_cost_line", {"quantity": 100, "consumption_per_unit": 0.5, "price_per_resource_unit": "40 zz"},
     "I do not recognise the unit in 40 zz; the cited price per resource unit is a price per count. "
     "What cited price per resource unit should I use?"),
    # not counted: the unit is still named
    ("wind_pressure", {"wind_speed": "40 kg"},
     "The wind speed is a length per time in m/s, and 40 kg is a mass, so it does not convert. "
     "What wind speed should I use?"),
    ("wind_pressure", {"wind_speed": "40 zz"},
     "I do not recognise the unit in 40 zz; the wind speed is in m/s. What wind speed should I use?"),
    ("excavation_volume", {"length_m": "40 kg", "width_m": 2, "depth_m": 1},
     "The length is a length in m, and 40 kg is a mass, so it does not convert. What length should I use?"),
])
def test_a_refused_unit_is_asked_for_in_the_words_of_its_kind(calculation, inputs, question):
    from app.lib.construction_formulas import run_calculation

    env = run_calculation(calculation, dict(inputs))
    assert env["status"] == "error" and env["question"] == question


@pytest.mark.parametrize("unit,dimension,in_unit", [
    ("count", "count", ""), ("count/count", "count/count", ""), ("currency/count", "currency/count", ""),
    ("m", "", " in m"), ("m/s", "", " in m/s"), ("%", "", " in %"), ("zz", "", " in zz"), ("", "", " in "),
])
def test_only_a_unit_that_counts_something_is_named_by_its_kind(unit, dimension, in_unit):
    from app.lib import source_labels

    assert source_labels._counted_dimension(unit) == dimension
    assert source_labels._in_unit(unit) == in_unit


@pytest.mark.parametrize("rejected,missing,question", [
    ([], ["quantity"],
     "The Resource cost line calculator needs the quantity of work. What quantity of work should I use?"),
    ([{"parameter": "quantity", "value": "x", "reason": "not_a_number"}], ["price_per_resource_unit"],
     "The quantity of work needs a number, not “x”. The Resource cost line calculator also needs the cited "
     "price per resource unit. What values should I use for the quantity of work and cited price per resource unit?"),
    ([], ["quantity", "price_per_resource_unit"],
     "The Resource cost line calculator needs the quantity of work and cited price per resource unit. "
     "What values should I use for the quantity of work and cited price per resource unit?"),
])
def test_an_input_the_calculator_still_needs_is_named_in_the_question(rejected, missing, question):
    from app.lib import source_labels

    assert source_labels.input_question("resource_cost_line", rejected, missing=missing) == question

