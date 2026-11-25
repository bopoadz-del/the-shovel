"""Edges of the rule that a figure written per a unit is a rate over it.

Each case is where the rule must NOT fire, or the exact boundary of where it
does. Every calculator is built here, units are declared by the test, and
every decision is checked with two inputs.
"""
from __future__ import annotations

import pytest

from app.lib import construction_formulas as cf
from app.lib.construction_formulas import extract_calculation_params_from_text as read
from app.lib.formula_registry import FormulaSpec


def _priced(quantity_m3, price_per_m3):
    return quantity_m3 * price_per_m3


def _priced_text(quantity_m3, price_per_m3: str):
    return quantity_m3


def _priced_optional(quantity_m3, price_per_m3=0.0):
    return quantity_m3 * price_per_m3


def _two_prices(water_price_per_m3, plant_cost_per_m3):
    return water_price_per_m3 + plant_cost_per_m3


def _daily(area_m2, cost_per_day):
    return area_m2 * cost_per_day


def _loaded(area_m2, load_per_m2):
    return area_m2 * load_per_m2


def _unpriced(quantity_m3, unit_rate):
    return quantity_m3 * unit_rate


UNITS = {
    "quantity_m3": "m3", "price_per_m3": "currency/m3",
    "water_price_per_m3": "currency/m3", "plant_cost_per_m3": "currency/m3",
    "area_m2": "m2", "cost_per_day": "currency/day", "load_per_m2": "kN/m2",
    "unit_rate": "",
}


@pytest.fixture
def declared(monkeypatch):
    real = cf._formula_spec

    def _spec(fn):
        code = getattr(fn, "__code__", None)
        args = code.co_varnames[:code.co_argcount] if code else ()
        if not args or any(k not in UNITS for k in args):
            return real(fn)
        return FormulaSpec(name=fn.__name__, fn=fn, owner="test", description="",
                           inputs={k: UNITS[k] for k in args})

    monkeypatch.setattr(cf, "_formula_spec", _spec)


@pytest.mark.parametrize("glued", ["310 perm3 here", "40 perm2 here"])
def test_per_must_be_its_own_word(glued):
    end = glued.index(" ")
    assert cf._per_unit_after(glued, end, cf._TEXT_UNIT_RE) is None
    spaced = glued.replace("perm", "per m")
    assert cf._per_unit_after(spaced, end, cf._TEXT_UNIT_RE) is not None


@pytest.mark.parametrize("figure", [310, 45])
def test_a_rate_over_another_unit_does_not_fill_the_rate_input(declared, figure):
    assert read(_priced, f"Pour 120 m3 at {figure} per m2.").get("price_per_m3") != pytest.approx(figure)
    assert read(_priced, f"Pour 120 m3 at {figure} per m3.").get("price_per_m3") == pytest.approx(figure)


@pytest.mark.parametrize("figure", [310, 45])
def test_a_label_rate_fills_an_input_that_declares_no_unit(declared, figure):
    assert read(_unpriced, f"Unit rate is {figure} per m3.").get("unit_rate") == pytest.approx(figure)


@pytest.mark.parametrize("figure", [12, 30])
def test_a_figure_with_its_own_unit_written_per_a_unit_is_not_a_pure_rate(declared, figure):
    got = read(_daily, f"Lay {figure} m2 per day.")
    assert got.get("cost_per_day") != pytest.approx(figure)


@pytest.mark.parametrize("a, b", [(310, 290), (45, 60)])
def test_two_rates_over_the_same_unit_bind_neither(declared, a, b):
    got = read(_priced, f"Pour 120 m3 at {a} per m3 or {b} per m3.")
    assert got.get("price_per_m3") not in (pytest.approx(a), pytest.approx(b))


@pytest.mark.parametrize("figure", [310, 45])
def test_a_rate_input_typed_as_text_takes_no_figure(declared, figure):
    assert "price_per_m3" not in read(_priced_text, f"Pour 120 m3 at {figure} per m3.")
    assert read(_priced, f"Pour 120 m3 at {figure} per m3.").get("price_per_m3") == pytest.approx(figure)


@pytest.mark.parametrize("figure", [310, 45])
def test_an_optional_rate_input_is_not_filled_by_a_bare_rate(declared, figure):
    assert read(_priced_optional, f"Pour 120 m3 at {figure} per m3.").get("price_per_m3") != pytest.approx(figure)


@pytest.mark.parametrize("first, second", [(310, 290), (45, 60)])
def test_a_labelled_rate_is_not_overwritten_by_a_later_rate(declared, first, second):
    got = read(_priced, f"Price per m3 is {first}; pour 120 m3 at {second} per m3.")
    assert got.get("price_per_m3") == pytest.approx(first)


@pytest.mark.parametrize("figure", [5, 7.5])
def test_a_rate_input_with_a_unit_of_its_own_needs_that_unit_written(declared, figure):
    assert read(_loaded, f"Apply {figure} per m2.").get("load_per_m2") != pytest.approx(figure)


@pytest.mark.parametrize("figure", [310, 45])
def test_a_labelled_rate_is_not_read_again_for_another_rate_input(declared, figure):
    # The label takes the figure; the rate pass must not read it a second time.
    got = read(_two_prices, f"Water price per m3 is {figure} per m3.")
    assert got.get("water_price_per_m3") == pytest.approx(figure)
    assert got.get("plant_cost_per_m3") != pytest.approx(figure)
