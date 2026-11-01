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

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_PATHS = [
    "app/lib/construction_formulas_commercial.py",
    "app/lib/construction_formulas.py",
    "app/agents/hats/contracts/formulas/construction_formulas_commercial.py",
    "app/agents/runtime.py",
]
# Code-level currency-default patterns (not comments/docstrings/parse vocab).
_BANNED = ['or "SAR"', "or 'SAR'", 'currency="SAR"', 'currency = "SAR"',
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


def test_delay_damages_invents_no_currency_when_none_given():
    out = cont.delay_damages_daily(rate_percent=5.0, contract_amount=1_000_000.0)
    # No invented "SAR": the field is empty (unspecified), never a code default.
    assert out.get("currency") == "", out
    assert "SAR" not in (out.get("note") or ""), out
