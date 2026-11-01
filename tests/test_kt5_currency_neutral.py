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
import app.lib.formula_registry as fr

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
    # Currency is a label, not an operand (K-T5 ruling ii): absent -> None, the
    # figure stands on its own, no "SAR" and no leading space in the line.
    assert out.get("currency") is None, out
    assert "SAR" not in (out.get("note") or ""), out
    assert "  " not in (out.get("note") or ""), out


@pytest.mark.parametrize("amount, expected", [
    ("SAR 1,200 per day", "SAR"),
    ("AED 5,000 per day", "AED"),
])
def test_currency_is_derived_from_a_money_amount_unit(amount, expected):
    # The currency of a money value is the answer's currency -- it flows from
    # the data, not a code default (kills the convert_written_units guard).
    out, _conv, _rej = fr.convert_written_units(
        "delay_damages_daily", {"amount_per_day": amount, "days": 10})
    assert out.get("currency") == expected, out


@pytest.mark.parametrize("amount", ["5,000 per day", "1,200 /day"])
def test_no_currency_is_invented_when_the_amount_carries_none(amount):
    out, _conv, _rej = fr.convert_written_units(
        "delay_damages_daily", {"amount_per_day": amount, "days": 10})
    assert out.get("currency") in (None, ""), out


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


# ── CI-M coverage for the two compose functions the currency edit touched.
#    Each exercises the intent-guard's FALSE path (the composer must decline an
#    ask of the wrong kind even when the operands are present), which the
#    mutation gate's drop_rule on that first guard removes, plus the TRUE path
#    where the currency flows from the Accepted Contract Amount in the data.
#    Excerpt rows mirror tests/test_set3_empty_turn_particulars_and_period_dd. ─

_CD_LABEL = "CONTRACT DATA particulars.\nCONTRACT DATA\n"
_CD_PCT = _CD_LABEL + (
    "14.2.1: | | Advance Payment: 10% of the Accepted Contract Amount | |\n"
    "1.1.1: | | Accepted Contract Amount: SAR 1,754,504,456.25 excluding VAT |\n"
)
_CD_DD = _CD_LABEL + (
    "8.8.1: | | Delay Damages (if applicable per Milestone): Milestone | Delay Damages\n"
    "|: | | Milestone 1 | 0.015% of the Contract Price per calendar day\n"
    "1.1.1: | | Accepted Contract Amount: SAR 1,754,504,456.25 excluding VAT |\n"
)


def test_percentage_compose_declines_a_non_money_ask_but_composes_the_money_one():
    # Names the Advance Payment but does not ask for it in money -> declined.
    assert cfc.compose_percentage_of_aca_from_excerpts(
        "What is the Advance Payment amount under the contract?", _CD_PCT) is None
    out = cfc.compose_percentage_of_aca_from_excerpts(
        "Calculate the Advance Payment in SAR.", _CD_PCT)
    assert out is not None and out.get("currency") == "SAR", out


def test_period_delay_compose_declines_a_non_delay_ask_but_composes_the_delay_one():
    # Names a milestone and a duration but is not a delay-damages ask -> declined.
    assert cfc.compose_delay_damages_over_period_from_excerpts(
        "Milestone 1 is 30 days late.", _CD_DD) is None
    out = cfc.compose_delay_damages_over_period_from_excerpts(
        "If Milestone 1 is 30 days late, what are the milestone delay damages?", _CD_DD)
    assert out is not None and out.get("currency") == "SAR", out
