"""S-K2 item 9: overhead vs compounding markup with tender-supplied rates.

Every expected figure is worked by hand in the comment beside it, not taken
from the implementation. Two inputs per decision.
"""
from __future__ import annotations

import pytest

from app.agents.hats.commercial.formulas.construction_formulas_commercial import (
    _markup_basis,
    markup_build_up,
)
from app.lib.construction_formulas import run_calculation


@pytest.mark.parametrize("cost,rates,price", [
    # 1,000,000 x (1 + 10% + 5%) = 1,150,000
    (1_000_000, [10, 5], 1_150_000.0),
    # 80,000 x (1 + 8% + 4% + 3%) = 92,000
    (80_000, [8, 4, 3], 92_000.0),
])
def test_on_cost_sums_the_rates(cost, rates, price):
    r = markup_build_up(cost, rates, "on cost")
    assert r["basis"] == "on_cost"
    assert r["price"] == price
    assert r["price_on_cost"] == price


@pytest.mark.parametrize("cost,rates,price", [
    # 1,000,000 x 1.10 = 1,100,000; x 1.05 = 1,155,000
    (1_000_000, [10, 5], 1_155_000.0),
    # 80,000 x 1.08 = 86,400; x 1.04 = 89,856; x 1.03 = 92,551.68
    (80_000, [8, 4, 3], 92_551.68),
])
def test_compounding_applies_each_rate_to_the_running_total(cost, rates, price):
    r = markup_build_up(cost, rates, "compounding")
    assert r["basis"] == "compounding"
    assert r["price"] == price
    assert r["price_compounding"] == price


@pytest.mark.parametrize("basis,expected_other", [
    # chosen on cost -> the other (compounding) is 1,155,000
    ("on cost", 1_155_000.0),
    # chosen compounding -> the other (on cost) is 1,150,000
    ("compounding", 1_150_000.0),
])
def test_both_bases_and_their_difference_are_stated(basis, expected_other):
    r = markup_build_up(1_000_000, [10, 5], basis)
    other = r["price_compounding"] if r["basis"] == "on_cost" else r["price_on_cost"]
    assert other == expected_other
    # 1,155,000 - 1,150,000 = 5,000, whichever basis was chosen
    assert r["difference_between_bases"] == 5_000.0
    assert any(f"{expected_other:,.2f}" in line for line in r["working"]), r["working"]


@pytest.mark.parametrize("basis,markup,effective,margin", [
    # on cost: 200,000 x 20% = 40,000; 40,000/200,000 = 20%; 40,000/240,000 = 16.6667%
    ("on cost", 40_000.0, 20.0, 16.6667),
    # compounding: 200,000 x 1.125 x 1.075 = 241,875; markup 41,875 = 20.9375%;
    # 41,875 / 241,875 = 17.3127%
    ("compounding", 41_875.0, 20.9375, 17.3127),
])
def test_markup_on_cost_is_not_the_margin_on_price(basis, markup, effective, margin):
    r = markup_build_up(200_000, [12.5, 7.5], basis)
    assert r["markup_amount"] == markup
    assert r["effective_markup_percent"] == effective
    assert r["margin_on_price_percent"] == margin


@pytest.mark.parametrize("word,basis", [
    ("additive", "on_cost"), ("ON_COST", "on_cost"), ("on-cost", "on_cost"),
    ("compound", "compounding"), ("Cascading", "compounding"),
])
def test_basis_words_name_one_basis(word, basis):
    assert _markup_basis(word) == basis


@pytest.mark.parametrize("word", ["", "margin", None])
def test_a_basis_word_that_names_neither_is_refused(word):
    assert _markup_basis(word) == ""
    r = markup_build_up(1_000, [10], word)
    assert "error" in r and "on cost" in r["error"] and "compounding" in r["error"]


@pytest.mark.parametrize("rates", [[], ()])
def test_no_rate_is_refused(rates):
    assert "error" in markup_build_up(1_000, rates, "on cost")


@pytest.mark.parametrize("rates,named", [([10, -2], "2"), ([-1, 4, -3], "1, 3")])
def test_a_negative_rate_is_refused_by_number(rates, named):
    r = markup_build_up(1_000, rates, "compounding")
    assert "error" in r and named in r["error"]


@pytest.mark.parametrize("cost", [-1, -250_000])
def test_a_negative_cost_is_refused(cost):
    assert "error" in markup_build_up(cost, [10], "on cost")


@pytest.mark.parametrize("basis", ["on cost", "compounding"])
def test_a_single_rate_is_the_same_on_both_bases(basis):
    # A bare 10 (not a list): 5,000 x 1.10 = 5,500 on either basis.
    r = markup_build_up(5_000, 10, basis)
    assert r["price"] == 5_500.0
    assert r["difference_between_bases"] == 0.0


@pytest.mark.parametrize("basis", ["on cost", "compounding"])
def test_zero_cost_gives_zero_without_dividing_by_it(basis):
    r = markup_build_up(0, [10, 5], basis)
    assert r["price"] == 0.0
    assert r["effective_markup_percent"] == 0.0
    assert r["margin_on_price_percent"] == 0.0


@pytest.mark.parametrize("cost", [1_234_567_890, 987_654_321_000])
def test_large_figures_are_written_in_full(cost):
    r = markup_build_up(cost, [10, 5], "compounding")
    text = " ".join(r["working"])
    assert "e+" not in text
    assert f"{cost:,.2f}" in text


@pytest.mark.parametrize("basis,price", [("on cost", 1_150_000.0), ("compounding", 1_155_000.0)])
def test_through_the_live_dispatch_envelope(basis, price):
    out = run_calculation("markup_build_up",
                          {"cost": 1_000_000, "rates_percent": [10, 5], "basis": basis})
    assert out["status"] == "success", out
    assert out["result"]["price"] == price
