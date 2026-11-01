"""K-T5b: a currency named in the free ask text flows to the currency label.

ORDERS_01NOV_0800Z: "... 60 days in AED" (no currency on the amount) must put
AED on the result. The binder reads the currency from the ORIGINAL ask, before
_normalize_ask strips a currency that sits in front of a figure; when the ask
names no currency the label stays absent -- never invented (K-T5 ruling ii).
"""
from __future__ import annotations

import pytest

import app.lib.construction_formulas as cf
import app.lib.formula_registry as fr


def _dd_fn():
    return fr.get("delay_damages_daily").fn


@pytest.mark.parametrize("ask, expected", [
    ("Delay damages at a flat 5,000 per day for 60 days in AED", "AED"),
    ("Delay damages at a flat 5,000 per day for 60 days in USD", "USD"),
])
def test_currency_named_in_free_ask_text_is_captured(ask, expected):
    found = cf.extract_calculation_params_from_text(_dd_fn(), ask)
    assert found.get("currency") == expected, found


@pytest.mark.parametrize("ask, before_figure, expected", [
    ("Delay damages of SAR 1,200 per day for 45 days", True, "SAR"),
    ("Delay damages of AED 2,000 per day for 30 days", True, "AED"),
])
def test_currency_in_front_of_a_figure_survives_the_strip(ask, before_figure, expected):
    # _normalize_ask removes a currency written in front of a figure; the label
    # is read from the original ask, so it is not lost.
    found = cf.extract_calculation_params_from_text(_dd_fn(), ask)
    assert found.get("currency") == expected, found


@pytest.mark.parametrize("ask", [
    "Delay damages at a flat 5,000 per day for 60 days",
    "Delay damages of 0.1% of the contract amount for 30 days",
])
def test_no_currency_in_the_ask_leaves_the_label_absent(ask):
    found = cf.extract_calculation_params_from_text(_dd_fn(), ask)
    assert found.get("currency") in (None, ""), found
