"""A figure written per a unit ("310 per m3") is a rate over that unit.

It fills the one open input that is a rate over that unit. It is never read
as an amount of that unit, a count, a percentage or a bare number for another
input. The rate is read from the unit each input declares, never from a list
of cases.

Each calculator here is built in the test, so no registered formula and no
product answer supplies an expected value. Every decision has two inputs.
"""
from __future__ import annotations

import pytest

from app.lib import construction_formulas as cf
from app.lib.construction_formulas import extract_calculation_params_from_text as read
from app.lib.formula_registry import FormulaSpec


def _valuation(gross_valuation, retention_percent=5.0):
    return gross_valuation * (1 - retention_percent / 100)


def _volume_count(concrete_m3, num_trucks):
    return concrete_m3 / num_trucks


def _priced(quantity_m3, price_per_m3):
    return quantity_m3 * price_per_m3


def _two_prices(water_price_per_m3, plant_cost_per_m3):
    return water_price_per_m3 + plant_cost_per_m3


@pytest.fixture
def declared(monkeypatch):
    """Declare the units the calculators above carry, the way the registry
    declares them, without touching the registry."""
    units = {
        "gross_valuation": "currency", "retention_percent": "%",
        "concrete_m3": "m3", "num_trucks": "count",
        "quantity_m3": "m3", "price_per_m3": "currency/m3",
        "water_price_per_m3": "currency/m3", "plant_cost_per_m3": "currency/m3",
    }

    real = cf._formula_spec

    def _spec(fn):
        code = getattr(fn, "__code__", None)
        args = code.co_varnames[:code.co_argcount] if code else ()
        if not args or any(k not in units for k in args):
            return real(fn)
        return FormulaSpec(
            name=fn.__name__, fn=fn, owner="test", description="",
            inputs={k: units[k] for k in args},
        )

    monkeypatch.setattr(cf, "_formula_spec", _spec)


@pytest.mark.parametrize("ask, figure", [("Concrete is 310 per m3.", 310), ("It costs 45 AED per m3.", 45)])
def test_a_rate_is_not_an_amount_or_a_bare_figure(declared, ask, figure):
    assert figure not in read(_valuation, ask).values()
    assert figure not in read(_volume_count, ask).values()


@pytest.mark.parametrize("figure", [310, 45.5])
def test_a_rate_fills_the_one_input_that_is_a_rate_over_its_unit(declared, figure):
    got = read(_priced, f"Pour 120 m3 at {figure} per m3.")
    assert got.get("price_per_m3") == pytest.approx(figure)
    assert got.get("quantity_m3") != pytest.approx(figure)


@pytest.mark.parametrize("figure", [310, 12])
def test_a_rate_with_two_inputs_it_could_fill_binds_neither(declared, figure):
    got = read(_two_prices, f"Allow {figure} per m3.")
    assert figure not in got.values()


@pytest.mark.parametrize("figure", [310, 75])
def test_a_label_before_a_rate_binds_only_a_rate_input(declared, figure):
    # "concrete" labels an amount in m3: a rate written after it is not that amount.
    assert "concrete_m3" not in read(_volume_count, f"Concrete is {figure} per m3.")
    # "price" labels a rate over m3: the same sentence shape fills it.
    assert read(_priced, f"Price is {figure} per m3.").get("price_per_m3") == pytest.approx(figure)


@pytest.mark.parametrize("raw, per, word", [
    ("310 per m3 here", "m3", ""),
    ("310 AED per m2 here", "m2", "AED"),
])
def test_the_per_unit_is_read_straight_after_the_figure(raw, per, word):
    got = cf._per_unit_after(raw, 3, cf._TEXT_UNIT_RE)
    assert got is not None and got[0] == per and got[1] == word
    assert cf._per_unit_after(raw.replace(" per ", " for the "), 3, cf._TEXT_UNIT_RE) is None


@pytest.mark.parametrize("raw, word", [("12 N/mm2 here", "N"), ("40 kg/m3 here", "kg")])
def test_a_compound_unit_written_after_a_figure_is_its_own_unit(raw, word):
    end = raw.index(" ")
    assert cf._per_unit_after(raw, end, cf._TEXT_UNIT_RE) is None
    spaced = raw.replace("/", " per ")
    got = cf._per_unit_after(spaced, end, cf._TEXT_UNIT_RE)
    assert got is not None and got[1] == word


@pytest.mark.parametrize("token, per", [("currency/m3", "m3"), ("currency/t", "t")])
def test_a_rate_over_a_unit_is_read_from_the_declared_unit(token, per):
    assert cf._rate_over(token, per, bare=True)
    assert not cf._rate_over(token.split("/")[1], per)


@pytest.mark.parametrize("token", ["kN/m2", "N/mm2"])
def test_a_rate_with_a_unit_of_its_own_needs_that_unit_written(token):
    assert cf._rate_over(token, "m2")
    assert not cf._rate_over(token, "m2", bare=True)
