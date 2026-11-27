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



# ── a rate and a price whose units must chain (Param.per) ────────────────────
# Through the chat tool door: the same kind converts and the converted figure
# is what is multiplied; another kind is refused with a question naming both.

@pytest.mark.parametrize("q,rate,price,expect_total", [
    # 640 m2 x 22 L/m2 = 14,080 L; 310 per m3 = 0.31 per L -> 4,364.80
    ("640 m2", "22 L/m2", "310 per m3", 4_364.80),
    # 100 m3 x 35 kg/m3 = 3,500 kg; 2,800 per t = 2.8 per kg -> 9,800.00
    ("100 m3", "35 kg/m3", "2800 per t", 9_800.00),
    # 640 ft2 = 59.4579 m2; x 0.022 m3/m2 x 310 per m3 -> 405.50
    ("640 ft2", "0.022 m3/m2", "310 per m3", 405.50),
])
def test_units_that_chain_are_converted_before_they_are_multiplied(q, rate, price, expect_total):
    result = _chat({"quantity": q, "consumption_per_unit": rate, "price_per_resource_unit": price})
    assert result.get("status") == "success", result
    assert result["result"]["line_total"] == pytest.approx(expect_total, abs=0.005)


@pytest.mark.parametrize("q,rate,price,question", [
    ("640 m3", "0.022 m3/m2", "310 per m3",
     "The resource consumed per unit of work (0.022 m3/m2) is per m², and the quantity of work (640 m3) "
     "is in m³: an area does not convert to a volume, so the two do not go together. "
     "What resource consumed per unit of work should I use?"),
    ("640 m2", "0.022 m3/m2", "310 AED/kg",
     "The cited price per resource unit (310 AED/kg) is per kg, and the resource consumed per unit of work "
     "(0.022 m3/m2) is in m³: a mass does not convert to a volume, so the two do not go together. "
     "What cited price per resource unit should I use?"),
])
def test_units_that_do_not_chain_are_refused_with_both_named(q, rate, price, question):
    result = _chat({"quantity": q, "consumption_per_unit": rate, "price_per_resource_unit": price})
    assert result.get("needs_input") is True, result
    assert "line_total" not in (result.get("result") or {})
    assert result["error"] == question


@pytest.mark.parametrize("rate,question", [
    ("45 AED/kg", "The material rate per unit is a price per count, and 45 AED/kg is a price per mass, "
                  "so it does not convert. What material rate per unit should I use?"),
    ("45 per m3", "I do not recognise the unit in 45 per m3; the material rate per unit is a price per count. "
                  "What material rate per unit should I use?"),
])
def test_resource_line_cost_still_refuses_a_material_rate_per_measure(rate, question):
    from app.lib.construction_formulas import run_calculation

    env = run_calculation("resource_line_cost", {"quantity": "500 m2", "daily_output": "20 m2/d",
                                                 "day_rate": 800, "material_rate_per_unit": rate})
    assert env["status"] == "error" and env["question"] == question


@pytest.mark.parametrize("q,c,p,waste,gross_step,total", [
    (68_712, 7.688, 2_457.14, 11.28,
     "With waste 11.28% = 528,257.856 x 1.1128 = 587,845.3422", 1_444_418_304.13),
    (90_284.8, 38, 9_708.09, 17.77, None, 39_225_339_017.48),
])
def test_the_gross_is_the_printed_net_times_the_factor(q, c, p, waste, gross_step, total):
    out = resource_cost_line(**_base(q, c, p), waste_percent=waste)
    if gross_step:
        assert gross_step in out["working"]
    assert out["line_total"] == total


# --- Money is rounded once and the printed lines add up: cost + markup =
# --- total as printed and as returned, over a grid of ordinary inputs.

def _money(text):
    from decimal import Decimal
    return Decimal(text.replace(",", ""))


@pytest.mark.parametrize("q", [1, 3, 7.5, 640])
@pytest.mark.parametrize("c", [12.5, 0.35, 22])
@pytest.mark.parametrize("p", [3.25, 42.5, 0.31, 2457.14])
@pytest.mark.parametrize("m", [10, 7.5, 12.5])
def test_the_printed_markup_line_adds_up(q, c, p, m):
    import re
    from decimal import Decimal
    out = resource_cost_line(q, c, p, markup_percent=m)
    line = out["working"][-1]
    cost, markup, total = (_money(x) for x in re.search(
        r"= ([\d,]+\.\d\d) \+ ([\d,]+\.\d\d) = ([\d,]+\.\d\d)$", line).groups())
    assert cost + markup == total
    assert (Decimal(repr(out["resource_cost"])), Decimal(repr(out["markup_amount"])),
            Decimal(repr(out["line_total"]))) == (cost, markup, total)


@pytest.mark.parametrize("args,line", [
    ((3, 12.5, 3.25, None, 10), "line total = 121.88 + 12.19 = 134.07"),
    ((1, 12.5, 42.5, None, 10), "line total = 531.25 + 53.13 = 584.38"),
])
def test_the_markup_is_taken_on_the_rounded_cost(args, line):
    assert resource_cost_line(*args)["working"][-1].endswith(line)


@pytest.mark.parametrize("args,gross_line,total", [
    ((600_150, 41.8517, 1, 6), "= 25,117,297.75 x 1.06 = 26,624,335.61", 26_624_335.61),
    ((68_712, 7.688, 2_457.14, 11.28), "= 528,257.856 x 1.1128 = 587,845.3422", 1_444_418_304.13),
    ((90_284.8, 38, 9_708.09, 17.77), None, 39_225_339_017.48),
])
def test_the_gross_and_the_total_on_large_and_tie_inputs(args, gross_line, total):
    out = resource_cost_line(*args)
    if gross_line:
        assert out["working"][1].endswith(gross_line)
    assert out["line_total"] == total


# --- Through the chat door: a converted price is printed as it is used, a
# --- unit written straight after the figure is kept, and a refusal quotes
# --- what the user wrote.

def _printed_step_reconciles(out):
    import re
    line = next(s for s in out["working"] if s.startswith("Resource cost"))
    gross, price, cost = (_money(x) for x in re.search(
        r"= ([\d,.]+) x price ([\d,.]+) = ([\d,]+\.\d\d)$", line).groups())
    from decimal import ROUND_HALF_UP
    return (gross * price).quantize(_money("0.01"), rounding=ROUND_HALF_UP) == cost


@pytest.mark.parametrize("params", [
    {"quantity": "1000000 m2", "consumption_per_unit": "1 m3/m2", "price_per_resource_unit": "95 per yd3"},
    {"quantity": "3333333 m2", "consumption_per_unit": "0.7 m3/m2",
     "price_per_resource_unit": "95 USD/yd3", "waste_percent": 7.5},
])
def test_a_converted_price_is_printed_as_it_is_used(params):
    out = _chat(params)
    assert out["status"] == "success", out
    assert _printed_step_reconciles(out["result"])


@pytest.mark.parametrize("price,total", [
    ("310/m3", 4_364.80),
    ("155/m3", 2_182.40),
    ("0.31/L", 4_364.80),
])
def test_a_price_unit_written_after_a_slash_is_kept(price, total):
    out = _chat({"quantity": "640 m2", "consumption_per_unit": "22 L/m2",
                 "price_per_resource_unit": price})
    assert out["status"] == "success", out
    assert out["result"]["line_total"] == total


@pytest.mark.parametrize("rate,price,quoted,never", [
    ("22 L/m2", "-310 per m3", "-310 per m3", "-0.31"),
    ("0.022 m3/m2", "900000000 per L", "900,000,000 per L", "900,000,000,000"),
    ("22 L/m2", "-310/m3", "(-310/m3)", "-0.31"),
])
def test_a_refusal_after_a_conversion_quotes_the_users_figure(rate, price, quoted, never):
    out = _chat({"quantity": "640 m2", "consumption_per_unit": rate,
                 "price_per_resource_unit": price})
    assert out["status"] == "error"
    assert quoted in out["question"]
    assert never not in out["question"]


@pytest.mark.parametrize("rate", ["22/m2", "0.5/m3"])
def test_a_rate_whose_unit_cannot_be_read_is_refused_by_name(rate):
    out = _chat({"quantity": "640 m2", "consumption_per_unit": rate,
                 "price_per_resource_unit": "310 per m3"})
    assert out["status"] == "error"
    assert "resource consumed per unit of work" in out["question"]


# --- the question quotes a converted input as figure + the unit written with it
@pytest.mark.parametrize("written,shown", [
    ([150.0, "%"], "(150%)"),
    ([-310.0, "per m3"], "(-310 per m3)"),
    ([310.0, "/m3"], "(310/m3)"),
    ([2500.5, "kg"], "(2500.5 kg)"),
])
def test_an_out_of_range_question_shows_the_written_figure_and_unit(written, shown):
    from app.lib import source_labels
    row = {"parameter": "price_per_resource_unit", "label": "price", "unit": "currency/count",
           "value": "as typed", "reason": "out_of_range", "lo": 0, "hi": 1e9,
           "kind": "number", "written": written}
    question = source_labels.input_question("resource_cost_line", [row])
    assert f" given {shown} is outside" in question
    assert "as typed" not in question


@pytest.mark.parametrize("value,shown", [(-0.31, "(-0.31 "), (2e9, "(2,000,000,000 ")])
def test_an_out_of_range_question_without_a_written_pair_shows_the_checked_value(value, shown):
    from app.lib import source_labels
    row = {"parameter": "price_per_resource_unit", "label": "price", "unit": "currency/count",
           "value": value, "reason": "out_of_range", "lo": 0, "hi": 1e9, "kind": "number"}
    assert shown in source_labels.input_question("resource_cost_line", [row])


# --- a converted input is stated in the working before the steps that use it
@pytest.mark.parametrize("conversions,lines", [
    ([{"parameter": "price_per_resource_unit", "value_in": 310.0, "from_unit": "currency/m3",
       "to_unit": "currency/L", "value_out": 0.31}],
     ["Converted cited price per resource unit: 310 per m³ = 0.31 per L"]),
    ([{"parameter": "consumption_per_unit", "value_in": 0.022, "from_unit": "m3/m2",
       "to_unit": "m3/ft2", "value_out": 0.0020438668800000003},
      {"parameter": "quantity", "value_in": 640.0, "from_unit": "ft2", "to_unit": "ft2",
       "value_out": 640.0}],
     ["Converted resource consumed per unit of work: 0.022 m³/m² = 0.00204386688 m³/ft²"]),
    ([{"parameter": "price_per_resource_unit", "value_in": 2800.0, "from_unit": "currency/t",
       "to_unit": "currency/kg", "value_out": 2.8000000000000003}],
     ["Converted cited price per resource unit: 2,800 per t = 2.8 per kg"]),
    ([{"parameter": "quantity", "value_in": 5.0, "from_unit": "m2", "to_unit": "m2",
       "value_out": 5.0}, "not a record"], []),
    ([], []),
])
def test_conversion_lines_state_each_converted_figure_with_both_units(conversions, lines):
    from app.lib import source_labels
    assert source_labels.conversion_lines("resource_cost_line", conversions) == lines


@pytest.mark.parametrize("params,first", [
    ({"quantity": "640 m2", "consumption_per_unit": "22 L/m2", "price_per_resource_unit": "310 per m3"},
     "Converted cited price per resource unit: 310 per m³ = 0.31 per L"),
    ({"quantity": "100 m3", "consumption_per_unit": "35 kg/m3", "price_per_resource_unit": "2800 per t"},
     "Converted cited price per resource unit: 2,800 per t = 2.8 per kg"),
    ({"quantity": 640, "consumption_per_unit": 22, "price_per_resource_unit": 0.31},
     "Net resource = quantity 640 x consumption 22 = 14,080"),
    ({"quantity": "640 m2", "consumption_per_unit": "22 L/m2", "price_per_resource_unit": "0.31 per L"},
     "Net resource = quantity 640 x consumption 22 = 14,080"),
])
def test_the_envelope_states_each_conversion_the_working_used(params, first):
    from app.lib.construction_formulas import run_calculation
    env = run_calculation("resource_cost_line", params)
    assert env["status"] == "success", env
    shown = env.get("converted_inputs") or []
    if first.startswith("Converted "):
        assert shown == [first]
    else:
        assert shown == [] and "converted_inputs" not in env
    # the result itself stays the figures of the calculation alone
    assert not any(line.startswith("Converted ") for line in env["result"]["working"])


# --- the public entry takes its inputs as an object, or as an object written in JSON
@pytest.mark.parametrize("params,total,shown", [
    ('{"quantity": "640 m2", "consumption_per_unit": "22 L/m2", "price_per_resource_unit": "310 per m3"}',
     4364.8, ["Converted cited price per resource unit: 310 per m³ = 0.31 per L"]),
    ('{"quantity": 100, "consumption_per_unit": 35, "price_per_resource_unit": 2.8}', 9800.0, None),
])
def test_inputs_written_as_a_json_object_are_read_through_the_public_entry(params, total, shown):
    from app.lib.construction_formulas import run_calculation
    env = run_calculation("resource_cost_line", params)
    assert env["status"] == "success", env
    assert env["result"]["line_total"] == total
    assert env.get("converted_inputs") == shown


@pytest.mark.parametrize("params", ["not an object", 42, 3.5])
def test_inputs_that_are_not_an_object_are_refused_as_such(params):
    from app.lib.construction_formulas import run_calculation
    env = run_calculation("resource_cost_line", params)
    assert env == {"status": "error", "error": "params must be an object of keyword arguments."}


# ── a counted noun names what is counted (formula_registry._noun_fills_count) ──
# "12.5 blocks per m2" for a count per count, "3.20 per block" or "3.20 each"
# for a price per count: the noun stands in the count place, as an
# unrecognised word after a plain count does. Through the public entry.

@pytest.mark.parametrize("consumption,price,total", [
    ("12.5 blocks per m2", "3.20 each", 1590 * 12.5 * 3.2),
    ("12.5 blocks/m2", "3.20 per block", 1590 * 12.5 * 3.2),
    ("6 bags per m2", "21 per bag", 1590 * 6 * 21),
    ("4 sheets/m2", "17.5 ea", 1590 * 4 * 17.5),
    ("2 panels per m2", "40 pcs", 1590 * 2 * 40),
])
def test_a_counted_noun_fills_the_count_place(consumption, price, total):
    from app.lib.construction_formulas import run_calculation

    env = run_calculation("resource_cost_line", {"quantity": "1590 m2", "consumption_per_unit": consumption,
                                                 "price_per_resource_unit": price})
    assert env["status"] == "success", env
    assert env["result"]["line_total"] == pytest.approx(total), env


@pytest.mark.parametrize("inputs,refused", [
    # a noun beside a unit of another kind: kN is a force, not a measure of work
    ({"consumption_per_unit": "6 bags per kN", "price_per_resource_unit": "21 per bag"}, "consumption_per_unit"),
    # a part that is not a plain word is not a noun
    ({"consumption_per_unit": "6 m33/m2", "price_per_resource_unit": "21 per bag"}, "consumption_per_unit"),
    # a noun alone does not fill a count per count
    ({"consumption_per_unit": "6 bags", "price_per_resource_unit": "21 per bag"}, "consumption_per_unit"),
    # a word that is not a count mark does not make a price per count
    ({"consumption_per_unit": "6 bags per m2", "price_per_resource_unit": "21 zz"}, "price_per_resource_unit"),
    # a price per volume against a consumption counted in bags
    ({"consumption_per_unit": "6 bags per m2", "price_per_resource_unit": "21 per m3"}, "price_per_resource_unit"),
    # a consumption per bag against a quantity in m2
    ({"consumption_per_unit": "6 blocks per bag", "price_per_resource_unit": "21 per block"}, "consumption_per_unit"),
])
def test_a_counted_noun_does_not_fill_what_it_cannot(inputs, refused):
    from app.lib.construction_formulas import run_calculation

    env = run_calculation("resource_cost_line", {"quantity": "1590 m2", **inputs})
    assert env["status"] != "success", env
    assert refused in str(env), env


@pytest.mark.parametrize("value", ["5000 per block", "5000 each"])
def test_a_counted_noun_does_not_fill_a_place_that_is_not_a_count(value):
    from app.lib.construction_formulas import run_calculation

    env = run_calculation("resource_line_cost", {"quantity": "500 m2", "daily_output": "20 m2/d",
                                                 "day_rate": value, "material_rate_per_unit": 30})
    assert env["status"] != "success", env
    assert "day_rate" in str(env), env


@pytest.mark.parametrize("unit,chained,fills", [
    ("blocks/m2", True, True),
    ("blocks/m2", False, False),   # a measure in a denominator needs per
    ("m2/blocks", True, True),
    ("m2/blocks", False, False),   # a noun in a denominator needs per
    ("m3/m2", True, False),        # no noun: a measure fill, not this rule
    ("blocks", True, False),       # parts do not match
    ("blocks/kN", True, False),
    ("bl0cks/m2", True, False),
])
def test_noun_fills_count_reads_each_part(unit, chained, fills):
    from app.agents.base.formulas.construction_formulas_planning import parse_unit
    from app.lib.formula_registry import _noun_fills_count

    assert _noun_fills_count(parse_unit("count/count"), unit, chained) is fills
    assert _noun_fills_count(None, unit, chained) is False


@pytest.mark.parametrize("quantity,consumption,refused", [
    # a quantity counted in a noun, against a consumption per area: refused
    ("1590 bricks", "12.5 blocks per m2", True),
    ("40 panels", "3 sheets/m2", True),
    # a noun against a noun: nothing to convert, computed
    ("1590 bricks", "12.5 blocks per brick", False),
    # a token that is not a plain word is no noun: nothing stated to check
    ("1590 m33", "12.5 blocks per m2", False),
    ("40 x9", "3 sheets/m2", False),
])
def test_a_counted_noun_on_the_quantity_is_checked_as_a_count(quantity, consumption, refused):
    from app.lib.construction_formulas import run_calculation

    env = run_calculation("resource_cost_line", {"quantity": quantity, "consumption_per_unit": consumption,
                                                 "price_per_resource_unit": "3.20 per block"})
    assert (env["status"] != "success") is refused, env


@pytest.mark.parametrize("value,parts", [
    ("12 blocks per m2", ("blocks", "m2")),
    ("3 per block", ("currency", "block")),
    ("1590 bricks", ("bricks", "")),
    ("6 m33/m2", None),
    ("6 bags per kN/m", None),
    (6, None),
])
def test_counted_parts_read_a_noun_in_a_place(value, parts):
    from app.lib.formula_registry import _counted_parts, _stated_parts

    assert _counted_parts(value) == parts
    if parts is not None:
        assert _stated_parts(value) is None   # the stated reading is unchanged


@pytest.mark.parametrize("unit,mark", [
    ("each", "nr"), ("ea", "nr"), ("pcs", "nr"), ("nr", "nr"),
    ("m3", None),       # a unit of another kind is not a count mark
    ("blocks", None),   # a counted noun is read elsewhere, not as a mark
    ("zz", None),
])
def test_count_mark_reads_a_written_count_unit(unit, mark):
    from app.lib.formula_registry import _count_mark

    assert _count_mark(unit) == mark
