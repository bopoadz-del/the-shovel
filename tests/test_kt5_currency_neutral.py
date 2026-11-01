"""K-T5 (currency-neutral): no invented currency default in formula/runtime code.

Per ORDERS_01NOV_0515Z: every `or "SAR"` / `currency="SAR"` default is removed;
the currency flows from the composed project data / parsed input, and when none
is present the formula emits no invented currency (the figure carries the
currency the data gave, or none). The `SAR/t` unit label becomes `currency/t`.
Role-1 (project data feeding SAR as input) and role-4 (the currency parse
vocabulary) are untouched -- a SAR project still yields SAR output, which proves
the currency now flows from data rather than a code default.

Guard (mutant-killable): re-introduce a literal currency default (`or "SAR"`,
`currency = "SAR"`) in any of these paths and test_no_currency_default_literal
reddens. Behaviour: delay_damages_daily echoes the currency it is given and
invents none when none is given.
"""
from __future__ import annotations

import pathlib

import pytest

import app.agents.hats.contracts.formulas.construction_formulas_commercial as cont
import app.lib.construction_formulas_commercial as cfc

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_PATHS = [
    "app/lib/construction_formulas_commercial.py",
    "app/lib/construction_formulas.py",
    "app/agents/hats/contracts/formulas/construction_formulas_commercial.py",
    "app/agents/runtime.py",
]
# Code-level currency-default patterns (not comments/docstrings/parse vocab).
_BANNED = ['or "SAR"', "or 'SAR'", '= "SAR"', "= 'SAR'",
           '"SAR").upper()', "'SAR').upper()", '"SAR").strip()']


@pytest.mark.parametrize("rel", _PATHS)
def test_no_currency_default_literal(rel):
    src = (_ROOT / rel).read_text(encoding="utf-8")
    hits = [b for b in _BANNED if b in src]
    assert hits == [], f"{rel} still has a currency default: {hits}"


@pytest.mark.parametrize("currency, expected", [("EUR", "EUR"), ("usd", "USD")])
def test_delay_damages_echoes_the_given_currency(currency, expected):
    out = cont.delay_damages_daily(rate_percent=5.0, contract_amount=1_000_000.0,
                                   currency=currency)
    assert out.get("currency") == expected, out


@pytest.mark.parametrize("rate, base", [(5.0, 1_000_000.0), (2.5, 400_000.0)])
def test_delay_damages_invents_no_currency_when_none_given(rate, base):
    out = cont.delay_damages_daily(rate_percent=rate, contract_amount=base)
    # No invented "SAR": the field is empty (unspecified), never a code default.
    assert out.get("currency") == "", out
    assert "SAR" not in (out.get("note") or ""), out


# ── CI-M coverage for the commercial format functions the currency edit
#    touched: each exercises the function's first branch so the mutation gate's
#    drop_rule/constant_return on it is killed, and the currency it is given
#    (from data) flows into the rendered line. ───────────────────────────────

def _period_composed(milestones):
    return {"currency": "SAR", "amount": 1_000.0, "days": 30,
            "contract_amount": 1_000_000.0, "rates": [0.1],
            "milestones": milestones}


@pytest.mark.parametrize("milestones, label", [
    ([1, 2], "Milestone 1 and Milestone 2"),
    ([3], "Milestone 3"),
])
def test_period_line_milestone_branch(milestones, label):
    line = cfc.format_delay_damages_period_line(_period_composed(milestones))
    assert f"Combined delay damages for {label}" in line, line  # kills drop_rule
    assert "SAR 1,000.00" in line and "SAR 1,000,000.00" in line, line


def test_period_line_no_milestone_branch():
    line = cfc.format_delay_damages_period_line(_period_composed([]))
    assert line.startswith("Delay damages for 30 days are SAR 1,000.00"), line
    assert "Combined" not in line, line


def _stated_total_composed(waste_percent):
    return {"currency": "SAR", "count": 2, "length": 10.0, "width": 5.0,
            "depth": 0.2, "net_volume_m3": 20.0, "volume_with_waste_m3": 21.0,
            "waste_percent": waste_percent, "unit_rate": 100.0,
            "total_cost": 2_100.0}


@pytest.mark.parametrize("waste_percent", [5.0, 12.5])
def test_stated_total_follow_up_waste_branch(waste_percent):
    line = cfc.format_stated_total_follow_up_line(_stated_total_composed(waste_percent))
    assert f"With {waste_percent:g}% waste:" in line, line  # kills drop_rule
    assert "Volume:" not in line, line
    assert "SAR 2,100.00" in line, line  # kills constant_return


def test_stated_total_follow_up_no_waste_branch():
    line = cfc.format_stated_total_follow_up_line(_stated_total_composed(0.0))
    assert "Volume: 21.00 m³." in line, line
    assert "waste:" not in line, line


def _takeoff_composed(waste_percent):
    return {"currency": "SAR", "count": 2, "length": 10.0, "width": 5.0,
            "depth": 0.2, "net_volume_m3": 20.0, "volume_with_waste_m3": 21.0,
            "waste_percent": waste_percent, "contingency_percent": 0.0,
            "unit_rate": 100.0, "base_cost": 2_100.0, "total_cost": 2_100.0}


@pytest.mark.parametrize("waste_percent", [5.0, 12.5])
def test_user_priced_takeoff_waste_branch(waste_percent):
    line = cfc.format_user_priced_takeoff_line(_takeoff_composed(waste_percent))
    assert f"With {waste_percent:g}% waste:" in line, line  # kills drop_rule
    assert "At the instructed SAR" in line, line
