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
        assert r["contribution_per_unit"] == pytest.approx(price - var, abs=0.005)
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


@pytest.mark.parametrize("stated", ["3", "+3", "a gain of 3"])
def test_a_figure_stated_with_the_wrong_sign_gets_no_source_line(stated):
    # contribution = 15 - 18 = -3; an answer giving it as positive misstates it
    out = _gated({"fixed_costs": 12.0, "price_per_unit": 15.0, "variable_cost_per_unit": 18.0},
                 f"The contribution per unit is {stated}.\nSource: Break-even point (platform calculator)")
    assert "Source: Break-even point" not in out, out


@pytest.mark.parametrize("stated", ["(3)", "a shortfall of 3"])
def test_other_written_negatives_keep_the_source_line(stated):
    out = _gated({"fixed_costs": 12.0, "price_per_unit": 15.0, "variable_cost_per_unit": 18.0},
                 f"The contribution per unit is {stated}.\nSource: Break-even point (platform calculator)")
    assert "Source: Break-even point" in out, out


@pytest.mark.parametrize("working", ["500 - 3 = 497", "1,203 − 3 = 1,200"])
def test_a_minus_after_a_figure_is_a_subtraction_not_a_negative(working):
    # contribution = 15 - 18 = -3; "x - 3" subtracts 3, it does not state -3
    out = _gated({"fixed_costs": 12.0, "price_per_unit": 15.0, "variable_cost_per_unit": 18.0},
                 f"Revenue then works out as {working}.\nSource: Break-even point (platform calculator)")
    assert "Source: Break-even point" not in out, out


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
