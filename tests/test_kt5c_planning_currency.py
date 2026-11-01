"""K-T5c: productivity-cost currency flows from the ask, never a SAR default.

ORDERS_01NOV_0931Z. construction_formulas_planning.py had two invented-SAR
defaults the 10-site sweep missed: the parse (`else "SAR"`) and the
format_productivity_cost_line signature default. Both now follow ruling (ii):
the currency is whatever the ask named, or none -- never an invented symbol.
The scan_hardwiring currency-literal tripwire that keeps these out is covered
by tests/test_scan_hardwiring.
"""
from __future__ import annotations

import pytest

import app.lib.construction_formulas_planning as planning


@pytest.mark.parametrize("ask, expected", [
    ("plaster 3,400 m2 at 42 m2 per gang-day, gang SAR 1,950 per day", "SAR"),
    ("plaster 1,000 m2 at 50 m2 per gang-day, gang AED 2,000 per day", "AED"),
])
def test_productivity_currency_flows_from_the_ask(ask, expected):
    *_rest, currency = planning.parse_productivity_cost_ask(ask)
    assert currency == expected, currency


@pytest.mark.parametrize("ask", [
    "plaster 3,400 m2 at 42 m2 per gang-day, gang 1,950 per day",
    "plaster 1,000 m2 at 50 m2 per gang-day, gang 2,000 per day",
])
def test_productivity_invents_no_currency_when_the_ask_names_none(ask):
    *_rest, currency = planning.parse_productivity_cost_ask(ask)
    assert currency == "", currency


@pytest.mark.parametrize("currency, shown", [("SAR", "SAR 1,000.00"), ("AED", "AED 1,000.00")])
def test_productivity_cost_line_carries_the_given_currency(currency, shown):
    line = planning.format_productivity_cost_line(
        {"duration": 2.0, "total_cost": 1000.0}, currency)
    assert shown in line, line


@pytest.mark.parametrize("cost", [1000.0, 2500.0])
def test_productivity_cost_line_omits_an_absent_currency(cost):
    line = planning.format_productivity_cost_line({"duration": 2.0, "total_cost": cost})
    assert f"{cost:,.2f}" in line, line
    assert "SAR" not in line and "  " not in line, line
