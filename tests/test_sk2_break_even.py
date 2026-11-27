"""S-K2 item 6: break-even with its domain conditions.

Every expected value is worked out here from the method, on inputs this
file builds itself -- no import of the engine's arithmetic:

    contribution  = price - variable cost per unit
    Q*            = fixed costs / contribution      (only while contribution > 0)
    revenue at Q* = Q* x price
    profit at q   = q x contribution - fixed costs
    margin        = (q - Q*) / q

At or below a zero contribution there is no break-even point and the
answer says so; none of the three amounts has a default.
"""
from __future__ import annotations

import math
import random

import pytest

from app.agents.hats.commercial.formulas.construction_formulas_additions import (
    break_even_point,
)
from app.lib import formula_registry as fr

_SEEDS = (7, 41)


def _cases(seed: int):
    rng = random.Random(seed)
    out = []
    for _ in range(20):
        fixed = float(rng.randint(10_000, 5_000_000))
        var = round(rng.uniform(1, 900), 2)
        price = round(var + rng.uniform(0.5, 600), 2)
        out.append((fixed, price, var))
    return out


@pytest.mark.parametrize("seed", _SEEDS)
def test_break_even_follows_the_method(seed):
    for fixed, price, var in _cases(seed):
        r = break_even_point(fixed, price, var)
        units = fixed / (price - var)
        assert r["break_even_units"] == pytest.approx(units, rel=1e-6, abs=1e-4)
        assert r["break_even_revenue"] == pytest.approx(units * price, abs=0.01)
        assert r["contribution_per_unit"] == pytest.approx(price - var, rel=1e-12)
        assert r["contribution_ratio"] == pytest.approx((price - var) / price, abs=1e-6)
        assert r["units_to_cover_fixed_costs"] == math.ceil(units - 1e-9)


@pytest.mark.parametrize(
    "fixed, price, var, whole",
    [(1_000, 30, 10, 50), (1_001, 30, 10, 51)],
)
def test_whole_units_cover_the_fixed_costs(fixed, price, var, whole):
    r = break_even_point(fixed, price, var)
    assert r["units_to_cover_fixed_costs"] == whole
    assert whole * (price - var) >= fixed
    assert (whole - 1) * (price - var) < fixed


@pytest.mark.parametrize("price, var", [(40, 40), (25, 40)])
def test_no_break_even_when_price_does_not_exceed_variable_cost(price, var):
    r = break_even_point(50_000, price, var)
    assert r["break_even_units"] is None and r["break_even_revenue"] is None
    assert "no break-even" in r["domain"]
    assert any("no break-even point" in line for line in r["working"])


@pytest.mark.parametrize("price, var", [(40.01, 40), (90, 15)])
def test_a_positive_contribution_has_a_break_even(price, var):
    r = break_even_point(50_000, price, var)
    assert r["break_even_units"] is not None
    assert "domain" not in r


@pytest.mark.parametrize("fixed, price, var, q", [(60_000, 50, 20, 3_000), (9_000, 12, 9, 7_500)])
def test_margin_of_safety_and_profit_above_break_even(fixed, price, var, q):
    r = break_even_point(fixed, price, var, expected_units=q)
    units = fixed / (price - var)
    assert r["profit_at_expected"] == pytest.approx(q * (price - var) - fixed, abs=0.01)
    assert r["margin_of_safety_units"] == pytest.approx(q - units, abs=1e-4)
    assert r["margin_of_safety_percent"] == pytest.approx((q - units) / q * 100, abs=0.01)
    assert not any("shortfall" in line for line in r["working"])


@pytest.mark.parametrize("fixed, price, var, q", [(60_000, 50, 20, 1_500), (9_000, 12, 9, 1_000)])
def test_below_break_even_is_a_stated_shortfall(fixed, price, var, q):
    r = break_even_point(fixed, price, var, expected_units=q)
    assert r["profit_at_expected"] < 0
    assert r["margin_of_safety_percent"] < 0
    assert any("shortfall" in line for line in r["working"])


@pytest.mark.parametrize("fixed", [60_000, 1_250])
def test_zero_expected_quantity_gives_the_loss_without_a_margin(fixed):
    r = break_even_point(fixed, 50, 20, expected_units=0)
    assert r["profit_at_expected"] == pytest.approx(-fixed)
    assert "margin_of_safety_percent" not in r


@pytest.mark.parametrize("fixed", [60_000, 1_250])
def test_no_expected_quantity_reports_no_profit(fixed):
    r = break_even_point(fixed, 50, 20)
    assert "profit_at_expected" not in r and "margin_of_safety_units" not in r


@pytest.mark.parametrize("seed", _SEEDS)
def test_working_prints_the_figures_it_reports(seed):
    for fixed, price, var in _cases(seed)[:5]:
        r = break_even_point(fixed, price, var)
        text = "\n".join(r["working"])
        assert any(v == pytest.approx(fixed) for v in _printed(text))
        assert f"{r['break_even_revenue']:,.2f}" in text


@pytest.mark.parametrize(
    "name", ["fixed_costs", "price_per_unit", "variable_cost_per_unit"]
)
def test_each_amount_has_no_default(name):
    kw = {"fixed_costs": 1_000, "price_per_unit": 30, "variable_cost_per_unit": 10}
    kw[name] = None
    r = break_even_point(**kw)
    assert r["missing"] == [name]
    assert fr.parameters(fr.get("break_even_point"))[name].required


@pytest.mark.parametrize(
    "kw",
    [
        {"fixed_costs": -1, "price_per_unit": 30, "variable_cost_per_unit": 10},
        {"fixed_costs": 1, "price_per_unit": -30, "variable_cost_per_unit": 10},
        {"fixed_costs": 1, "price_per_unit": 30, "variable_cost_per_unit": -10},
        {"fixed_costs": 1, "price_per_unit": 30, "variable_cost_per_unit": 10,
         "expected_units": -5},
    ],
)
def test_a_negative_input_is_refused(kw):
    assert "must be >= 0" in break_even_point(**kw)["error"]


@pytest.mark.parametrize("fixed, q", [(1_000, 80), (24_000, 1_500)])
def test_the_registry_runs_the_formula(fixed, q):
    spec = fr.get("break_even_point")
    assert spec is not None and spec.owner == "commercial"
    r = spec.fn(fixed_costs=fixed, price_per_unit=30, variable_cost_per_unit=10,
                expected_units=q)
    assert r["break_even_units"] == pytest.approx(fixed / 20)


# ── a negative figure the calculator returned keeps its Source line ─────────

def _gated(inputs: dict, answer: str) -> str:
    import json

    from app.agents.citation_provenance import gate
    from app.lib.construction_formulas import run_calculation

    env = run_calculation("break_even_point", dict(inputs))
    args = {"calculation": "break_even_point", "params": inputs}
    msgs = [
        {"role": "user", "content": "Calculate the break-even point."},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "c1", "function": {"name": "construction_calc",
                                     "arguments": json.dumps(args)}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "construction_calc",
         "content": json.dumps(env, default=str)},
    ]
    return gate(answer, None, msgs)


@pytest.mark.parametrize("price,variable,stated", [
    # contribution = 15 - 18 = -3, stated with a minus sign
    (15.0, 18.0, "-3"),
    # contribution = 20 - 25 = -5, stated as a loss in words
    (20.0, 25.0, "a loss of 5"),
])
def test_a_negative_calculator_figure_keeps_the_source_line(price, variable, stated):
    out = _gated({"fixed_costs": 12.0, "price_per_unit": price,
                  "variable_cost_per_unit": variable},
                 f"The contribution per unit is {stated}.\nSource: Break-even point (platform calculator)")
    assert "Source: Break-even point" in out, out


@pytest.mark.parametrize("stated", ["7", "-7"])
def test_a_figure_the_calculator_did_not_return_gets_no_source_line(stated):
    out = _gated({"fixed_costs": 12.0, "price_per_unit": 15.0, "variable_cost_per_unit": 18.0},
                 f"The contribution per unit is {stated}.\nSource: Break-even point (platform calculator)")
    assert "Source: Break-even point" not in out, out


def _printed(line: str) -> list[float]:
    import re
    return [float(t.replace(",", "")) for t in re.findall(r"-?[\d,]*\.?\d+", line)]


@pytest.mark.parametrize("fixed,price,variable", [
    (1_000_000.0, 81.4, 47.0),
    (2_500_000.0, 0.125, 0.0375),
])
def test_printed_working_reconciles(fixed, price, variable):
    # Read the working as a person does: the figures as printed must give
    # the result printed after them, with no exponent and no cut price.
    out = break_even_point(fixed, price, variable)
    w = out["working"]
    assert "e+" not in " ".join(w).lower()
    p, v, c = _printed(w[0])
    assert (p, v) == pytest.approx((price, variable)) and c == pytest.approx(p - v)
    f, c2, units = _printed(w[1])[:3]
    assert units == pytest.approx(f / c2, rel=1e-9)
    u2, p2, revenue = _printed(w[2])[:3]
    assert revenue == pytest.approx(u2 * p2, abs=0.005)


@pytest.mark.parametrize("fixed,price,variable,q", [
    (1_000.0, 10.0, 12.0, 50.0),
    (40_000.0, 5.0, 5.0, 900.0),
])
def test_expected_volume_states_its_loss_when_there_is_no_break_even(fixed, price, variable, q):
    out = break_even_point(fixed, price, variable, expected_units=q)
    assert out["break_even_units"] is None
    assert out["profit_at_expected"] == pytest.approx(round(q * (price - variable) - fixed, 2))
    assert out["working"][-1].startswith(f"Profit at {q:g} units")


@pytest.mark.parametrize("fixed,price,variable", [(1_000.0, 10.0, 6.0), (7_200.0, 30.0, 12.0)])
def test_selling_exactly_the_break_even_quantity_is_not_a_shortfall(fixed, price, variable):
    units = fixed / (price - variable)
    at = break_even_point(fixed, price, variable, expected_units=units)
    assert at["margin_of_safety_units"] == 0
    assert "shortfall" not in at["working"][-1]
    below = break_even_point(fixed, price, variable, expected_units=units - 1)
    assert "shortfall" in below["working"][-1]


@pytest.mark.parametrize("stated", ["(3)", "a shortfall of 3"])
def test_other_written_negatives_keep_the_source_line(stated):
    out = _gated({"fixed_costs": 12.0, "price_per_unit": 15.0, "variable_cost_per_unit": 18.0},
                 f"The contribution per unit is {stated}.\nSource: Break-even point (platform calculator)")
    assert "Source: Break-even point" in out, out


@pytest.mark.parametrize("fixed,price,variable", [
    (20_000_000_000.0, 10.0, 5.0),
    (1_000.0, 0.00008, 0.00003),
])
def test_a_figure_beyond_ten_digits_or_below_a_ten_thousandth_prints_in_full(
        fixed, price, variable):
    # Ten significant figures in ``g`` form switch to an exponent at 1e10 and
    # below 1e-4; the working must still print every figure positionally.
    import re
    w = break_even_point(fixed, price, variable)["working"]
    assert not re.search(r"\de[+-]?\d", " ".join(w).lower())
    p, v, c = _printed(w[0])
    assert (p, v) == pytest.approx((price, variable)) and c == pytest.approx(p - v)
    f, c2, units = _printed(w[1])[:3]
    assert f == pytest.approx(fixed) and units == pytest.approx(f / c2, rel=1e-9)


def _cent_cases(seed: int):
    # Price and variable cost with cents, and fixed costs that are exactly a
    # whole number n of contributions, all built in whole cents.
    rng = random.Random(seed)
    for _ in range(300):
        var_c = rng.randint(1, 50_000)
        price_c = var_c + rng.randint(1, 50_000)
        n = rng.randint(1, 20_000)
        yield price_c / 100, var_c / 100, (price_c - var_c) * n / 100, n


@pytest.mark.parametrize("seed", _SEEDS)
def test_expected_equal_to_break_even_with_cents_is_no_shortfall(seed):
    for price, var, fixed, n in _cent_cases(seed):
        out = break_even_point(fixed, price, var, expected_units=n)
        assert out["units_to_cover_fixed_costs"] == n, (price, var, fixed, n)
        assert out["break_even_units"] == n
        assert out["margin_of_safety_percent"] == 0.0
        assert math.copysign(1, out["margin_of_safety_percent"]) == 1
        assert out["profit_at_expected"] == 0.0
        assert math.copysign(1, out["profit_at_expected"]) == 1
        assert "shortfall" not in out["working"][-1]
        assert out["working"][-1].endswith("= 0.00%")


@pytest.mark.parametrize("seed", _SEEDS)
def test_one_unit_below_break_even_with_cents_is_a_shortfall(seed):
    for price, var, fixed, n in _cent_cases(seed):
        if n < 2:
            continue
        out = break_even_point(fixed, price, var, expected_units=n - 1)
        assert out["working"][-1].endswith("(below break-even: a shortfall)")
        assert out["profit_at_expected"] < 0


@pytest.mark.parametrize("inputs", [
    dict(fixed_costs=3000, price_per_unit=0.30, variable_cost_per_unit=0.10,
         expected_units=15000),
    dict(fixed_costs=7000, price_per_unit=10.70, variable_cost_per_unit=3.70,
         expected_units=1000),
])
def test_public_path_break_even_with_cents_is_exact(inputs):
    from app.lib.construction_formulas import run_calculation

    env = run_calculation("break_even_point", dict(inputs))
    assert env["status"] == "success", env
    res = env["result"]
    assert res["units_to_cover_fixed_costs"] == inputs["expected_units"]
    assert res["margin_of_safety_percent"] == 0.0
    assert "shortfall" not in " ".join(res["working"])


@pytest.mark.parametrize("divisor, start", [(3, 2 ** 54), (7, 2 ** 55)])
def test_whole_units_come_from_the_exact_quotient_beyond_float_precision(divisor, start):
    # Past 2**53 the float quotient can round down onto a whole number; the
    # whole units to sell are the ceiling of the exact quotient (checked
    # against fractions, an independent exact oracle).
    from fractions import Fraction

    found = 0
    for fixed in range(start, start + 4000, 8):
        exact = math.ceil(Fraction(repr(float(fixed))) / divisor)
        if math.ceil(float(fixed) / divisor) == exact:
            continue
        out = break_even_point(fixed, divisor, 0)
        assert out["units_to_cover_fixed_costs"] == exact, fixed
        found += 1
    assert found > 0


# ── a positive result keeps its Source line whatever prose leads it ────────

def _gated_calc(calc: str, inputs: dict, name: str, answer: str) -> str:
    import json

    from app.agents.citation_provenance import gate
    from app.lib.construction_formulas import run_calculation

    env = run_calculation(calc, dict(inputs))
    args = {"calculation": calc, "params": inputs}
    msgs = [
        {"role": "user", "content": "Calculate it."},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "c1", "function": {"name": "construction_calc",
                                     "arguments": json.dumps(args)}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "construction_calc",
         "content": json.dumps(env, default=str)},
    ]
    return gate(f"{answer}\nSource: {name} (platform calculator)", None, msgs)


@pytest.mark.parametrize("calc,inputs,name,answer", [
    # q = 0.613 x V^2: 980.8 Pa at 40 m/s, 551.7 Pa at 30 m/s
    ("wind_pressure", {"wind_speed": 40}, "Wind pressure",
     "Velocity pressure - 980.8 Pa at 40 m/s."),
    ("wind_pressure", {"wind_speed": 30}, "Wind pressure",
     "Result:\n- 551.7 Pa velocity pressure at 30 m/s"),
    # Q* = 120,000 / (85 - 61) = 5,000; 1,000 / (90 - 70) = 50
    ("break_even_point", {"fixed_costs": 120000, "price_per_unit": 85,
                          "variable_cost_per_unit": 61}, "Break-even point",
     "The plant runs at a loss until 5,000 m3 a month."),
    ("break_even_point", {"fixed_costs": 1000, "price_per_unit": 90,
                          "variable_cost_per_unit": 70}, "Break-even point",
     "The break-even volume (50 units) is the floor."),
    ("break_even_point", {"fixed_costs": 120000, "price_per_unit": 85,
                          "variable_cost_per_unit": 61}, "Break-even point",
     "85 minus 61 leaves 24."),
    ("break_even_point", {"fixed_costs": 1000, "price_per_unit": 90,
                          "variable_cost_per_unit": 70}, "Break-even point",
     "Break-even volume - 50 units."),
])
def test_a_positive_result_after_a_dash_bracket_or_loss_word_keeps_the_source_line(
        calc, inputs, name, answer):
    out = _gated_calc(calc, inputs, name, answer)
    assert f"Source: {name}" in out, out


@pytest.mark.parametrize("price,variable,stated", [
    (15.0, 18.0, "–3"),                    # contribution -3, en dash
    (20.0, 25.0, "minus 5"),                    # contribution -5, in words
    (15.0, 18.0, "a deficit of 3"),
    (20.0, 25.0, "a shortfall of about 5"),     # two words between
])
def test_a_negative_stated_in_other_words_keeps_the_source_line(price, variable, stated):
    out = _gated({"fixed_costs": 12.0, "price_per_unit": price,
                  "variable_cost_per_unit": variable},
                 f"The contribution per unit is {stated}.\nSource: Break-even point (platform calculator)")
    assert "Source: Break-even point" in out, out


# ── the reported figures are the exact ones ────────────────────────────────

@pytest.mark.parametrize("price,variable", [
    (0.125, 0.0375),        # 0.0875: four places
    (0.125, 0.121),         # 0.004: below a cent
    (0.00008, 0.00003),     # 0.00005: below a ten-thousandth
])
def test_contribution_per_unit_is_not_cut_to_the_cent(price, variable):
    from fractions import Fraction
    exact = float(Fraction(repr(price)) - Fraction(repr(variable)))
    out = break_even_point(1000.0, price, variable)
    assert out["contribution_per_unit"] == exact
    assert out["contribution_ratio"] == pytest.approx(
        exact / price, abs=1e-6)


@pytest.mark.parametrize("fixed,price,variable", [
    (1_234_567_890.12, 85.0, 61.0),
    (120_000.0, 123_456_789.12, 61.0),
    # the float product of quantity and price lands a cent off the printed one
    (4_024_525_814.09, 9_960.54, 9_883.97),
])
def test_inputs_print_as_written_and_each_step_adds_up(fixed, price, variable):
    from decimal import ROUND_HALF_UP, Decimal
    w = break_even_point(fixed, price, variable)["working"]
    assert f"price {price:,.2f}".rstrip("0").rstrip(".") in w[0]
    assert f"fixed costs {fixed:,.2f}".rstrip("0").rstrip(".") in w[1]
    import re
    m = re.fullmatch(r"Break-even revenue = ([\d,.]+) x ([\d,.]+) = ([\d,.]+) \(.*",
                     w[2])
    u, p, revenue = (Decimal(g.replace(",", "")) for g in m.groups())
    assert p == Decimal(repr(price))
    assert (u * p).quantize(Decimal("0.01"), ROUND_HALF_UP) == revenue


@pytest.mark.parametrize("q", [0.5, 0.25])
def test_an_expected_quantity_below_one_still_has_a_margin(q):
    # Q* = 1,000 / (10 - 6) = 250
    out = break_even_point(1000.0, 10.0, 6.0, expected_units=q)
    assert out["margin_of_safety_units"] == pytest.approx(q - 250, abs=1e-4)
    assert out["margin_of_safety_percent"] == pytest.approx((q - 250) / q * 100, abs=0.01)


@pytest.mark.parametrize("q,profit", [(101.0, -267.75), (150.0, 87.5)])
def test_profit_keeps_its_cents(q, profit):
    # contribution = 10.75 - 3.50 = 7.25; profit = q x 7.25 - 1,000
    out = break_even_point(1000.0, 10.75, 3.5, expected_units=q)
    assert out["profit_at_expected"] == profit


@pytest.mark.parametrize("fixed,price,var,whole", [(1.1, 0.2, 0.1, 11), (2.2, 0.3, 0.1, 11)])
def test_whole_units_on_a_quotient_that_floats_just_above_a_whole(fixed, price, var, whole):
    # 2.2 / (0.3 - 0.1) is 11.000000000000002 in floats; 11 units cover it
    assert break_even_point(fixed, price, var)["units_to_cover_fixed_costs"] == whole


@pytest.mark.parametrize("fixed,word", [(20.0, "1 whole unit)"), (40.0, "2 whole units)")])
def test_whole_units_are_singular_for_one(fixed, word):
    # contribution = 30 - 10 = 20
    assert break_even_point(fixed, 30.0, 10.0)["working"][1].endswith(word)


# ── verdict at 7387c57 ─────────────────────────────────────────────────────
# 1. Every figure inside the declared ranges is worked exactly or refused in
#    plain words; 2. the gate matches a negative result on its size, with no
#    rule keyed on wording.

@pytest.mark.parametrize("inputs,units,revenue,revenue_line", [
    # contribution 0.01: Q* = 1e13 / 0.01 = 1e15; revenue = 1e15 x 1e12 = 1e27
    ({"fixed_costs": 1e13, "price_per_unit": 1e12, "variable_cost_per_unit": 999999999999.99},
     1e15, 1e27, "1,000,000,000,000,000 x 1,000,000,000,000 = 1,000,000,000,000,000,000,000,000,000.00"),
    # contribution 0.5: Q* = 2e13; revenue = 2e13 x 1e12 = 2e25
    ({"fixed_costs": 1e13, "price_per_unit": 1e12, "variable_cost_per_unit": 999999999999.5},
     2e13, 2e25, "20,000,000,000,000 x 1,000,000,000,000 = 20,000,000,000,000,000,000,000,000.00"),
])
def test_the_largest_declared_figures_are_worked_exactly_through_the_registry(inputs, units, revenue, revenue_line):
    from app.lib.construction_formulas import run_calculation

    env = run_calculation("break_even_point", dict(inputs))
    assert env["status"] == "success", env
    result = env["result"]
    assert result["break_even_units"] == units and result["break_even_revenue"] == revenue
    assert revenue_line in result["working"][-1]


@pytest.mark.parametrize("price", [1e-300, 5e-324])
def test_a_quantity_too_large_to_state_is_refused_in_plain_words(price):
    """Inside the declared ranges, but fixed costs / contribution is beyond
    any figure that can be written: a plain refusal, never an exception's
    name and never an infinite number."""
    from app.lib.construction_formulas import run_calculation

    env = run_calculation("break_even_point", {"fixed_costs": 1e13, "price_per_unit": price,
                                               "variable_cost_per_unit": 0})
    assert env["status"] == "error"
    said = env["error"]
    assert "too large to state" in said
    for raw in ("InvalidOperation", "decimal", "<class", "Traceback", "inf"):
        assert raw not in said
    direct = break_even_point(1e13, price, 0)
    assert set(direct) == {"error"}


@pytest.mark.parametrize("price,units", [(1e-290, 1e303), (1e-295, 1e308)])
def test_a_very_large_quantity_that_can_still_be_written_is_worked(price, units):
    out = break_even_point(1e13, price, 0)
    assert out["break_even_units"] == pytest.approx(units, rel=1e-12)
    assert out["break_even_revenue"] == pytest.approx(1e13, rel=1e-12)
    assert math.isfinite(out["break_even_units"]) and math.isfinite(out["break_even_revenue"])


@pytest.mark.parametrize("fixed,units,revenue,line", [
    # Q* = F / 3 does not end; the working prints the quantity a float holds
    # and the revenue is THAT figure times the price, so the line adds up.
    (1e15 + 1, 333333333333333.7, 1000000000000001.1, "333,333,333,333,333.7 x 3 = 1,000,000,000,000,001.10"),
    (1e15 + 4, 333333333333334.7, 1000000000000004.1, "333,333,333,333,334.7 x 3 = 1,000,000,000,000,004.10"),
])
def test_revenue_is_the_printed_quantity_times_the_price(fixed, units, revenue, line):
    out = break_even_point(fixed, 3, 0)
    assert out["break_even_units"] == units and out["break_even_revenue"] == revenue
    assert line in out["working"][-1]


@pytest.mark.parametrize("kw,field,value", [
    # margin of safety in units to four places: 1,000 - 10,000/15 = 333.3333
    ({"fixed_costs": 10000, "price_per_unit": 45, "variable_cost_per_unit": 30, "expected_units": 1000},
     "margin_of_safety_units", 333.3333),
    # 1 x 0.005 - 0 = 0.005: half up to the cent
    ({"fixed_costs": 0, "price_per_unit": 0.015, "variable_cost_per_unit": 0.01, "expected_units": 1},
     "profit_at_expected", 0.01),
    # 1e12 x (0.3 - 0.1) on the decimal figures: exactly 200,000,000,000
    ({"fixed_costs": 0, "price_per_unit": 0.3, "variable_cost_per_unit": 0.1, "expected_units": 1e12},
     "profit_at_expected", 200000000000.0),
    # the largest declared price and quantity: 1e12 x 1e12 - 1e13
    ({"fixed_costs": 1e13, "price_per_unit": 1e12, "variable_cost_per_unit": 0, "expected_units": 1e12},
     "profit_at_expected", 9.9999999999e23),
])
def test_reported_figures_are_exact_to_their_stated_places(kw, field, value):
    assert break_even_point(**kw)[field] == value


@pytest.mark.parametrize("fixed", [0.001, 0.004])
def test_a_loss_below_half_a_cent_prints_as_zero_not_minus_zero(fixed):
    out = break_even_point(fixed, 10, 4, expected_units=0)
    assert out["profit_at_expected"] == 0.0 and str(out["profit_at_expected"]) == "0.0"
    assert out["working"][-1].endswith("= 0.00")


@pytest.mark.parametrize("calc,inputs,name,answer", [
    # contribution = 40 - 45 = -5, in words the check has no list for
    ("break_even_point", {"fixed_costs": 10000, "price_per_unit": 40, "variable_cost_per_unit": 45},
     "Break-even point", "Each unit sold loses 5, so there is no break-even point."),
    ("break_even_point", {"fixed_costs": 10000, "price_per_unit": 40, "variable_cost_per_unit": 45},
     "Break-even point", "The contribution per unit is -5."),
    ("break_even_point", {"fixed_costs": 12, "price_per_unit": 15, "variable_cost_per_unit": 18},
     "Break-even point", "Every unit is 3 under water."),
    # profit = 1,500 x 30 - 60,000 = -15,000
    ("break_even_point", {"fixed_costs": 60000, "price_per_unit": 50, "variable_cost_per_unit": 20,
                          "expected_units": 1500}, "Break-even point", "Expect losses of 15,000 at that volume."),
    # balance = 1,000 - 1,500 = -500 m3: another calculator, same check
    ("cut_fill_balance", {"cut_volume_m3": 1000, "fill_volume_m3": 1500}, "Cut and fill balance",
     "You are short 500 m3 of fill."),
    ("cut_fill_balance", {"cut_volume_m3": 1000, "fill_volume_m3": 1500}, "Cut and fill balance",
     "There is a deficit of 500 m3."),
])
def test_a_negative_result_keeps_its_source_line_however_the_answer_words_it(calc, inputs, name, answer):
    out = _gated_calc(calc, inputs, name, answer)
    assert f"Source: {name}" in out, out


@pytest.mark.parametrize("calc,inputs,name,answer", [
    # a result of exactly zero is a figure like any other
    ("cut_fill_balance", {"cut_volume_m3": 1000, "fill_volume_m3": 1000}, "Cut and fill balance", "The net is 0 m3."),
    ("break_even_point", {"fixed_costs": 600, "price_per_unit": 10, "variable_cost_per_unit": 4,
                          "expected_units": 100}, "Break-even point", "At 100 units the profit is 0.00."),
])
def test_a_result_of_zero_keeps_its_source_line(calc, inputs, name, answer):
    out = _gated_calc(calc, inputs, name, answer)
    assert f"Source: {name}" in out, out


@pytest.mark.parametrize("calc,inputs,name,answer", [
    # -5 and -500 are the results: a different size is not the calculator's figure
    ("break_even_point", {"fixed_costs": 10000, "price_per_unit": 40, "variable_cost_per_unit": 45},
     "Break-even point", "Each unit sold loses 6."),
    ("cut_fill_balance", {"cut_volume_m3": 1000, "fill_volume_m3": 1500}, "Cut and fill balance",
     "You are short 480 m3 of fill."),
])
def test_a_size_the_calculator_did_not_return_gets_no_source_line(calc, inputs, name, answer):
    out = _gated_calc(calc, inputs, name, answer)
    assert f"Source: {name}" not in out, out

