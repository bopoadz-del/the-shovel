"""A delay-damages operand the user types wins over the project's stored figure.

Each decision is tested with two inputs. Expected figures are computed
from the operands each test constructs; none is copied from a product answer.
"""
from __future__ import annotations

import pytest

from app.lib import construction_formulas_commercial as cc


def _excerpts(aca: float, rate: float) -> str:
    """A Contract Data block holding a stored base and a stored daily rate."""
    return (
        "[doc_id=cd chunk=0] Contract Data | "
        f"Accepted Contract Amount (excluding VAT): SAR {aca:,.2f} | "
        f"Delay Damages: {rate:g}% of the Contract Price per calendar day"
    )


STORED_ACA = 1_754_504_456.25
STORED_RATE = 0.1


# ---- the parser: what the ask itself states -------------------------------

@pytest.mark.parametrize("ask, rate", [
    ("Contract 8,000,000, delay damages 0.05% per day, 20 days late. Total?", 0.05),
    ("Delay damages at 0.2% a day on a contract of 3,500,000, 10 days late?", 0.2),
])
def test_a_typed_daily_rate_is_read(ask, rate):
    assert cc.typed_delay_damages_operands(ask)[0] == pytest.approx(rate)


@pytest.mark.parametrize("ask", [
    "What are the delay damages for 20 days late?",
    "Delay damages under our contract, 15 days late, capped at 10%?",
])
def test_no_typed_rate_when_the_ask_states_none(ask):
    assert cc.typed_delay_damages_operands(ask)[0] is None


@pytest.mark.parametrize("ask, amount, currency", [
    ("Contract 8,000,000, delay damages 0.05% per day, 20 days late. Total?",
     8_000_000.0, ""),
    ("Contract sum AED 3,500,000 and delay damages 0.1% per day, 10 days late?",
     3_500_000.0, "AED"),
    ("contract value 12500000 USD, delay damages 0.05% per day, 4 days late",
     12_500_000.0, "USD"),
])
def test_a_typed_contract_base_is_read(ask, amount, currency):
    base = cc.typed_delay_damages_operands(ask)[1]
    assert base is not None
    assert base[0] == pytest.approx(amount)
    assert base[1] == currency


@pytest.mark.parametrize("ask", [
    "What are the delay damages under the contract for 20 days late?",
    "Contract 2019 edition: delay damages for 12 days late?",
])
def test_no_typed_base_when_the_ask_states_none(ask):
    assert cc.typed_delay_damages_operands(ask)[1] is None


# ---- the period compose ---------------------------------------------------

@pytest.mark.parametrize("amount, rate, days", [
    (8_000_000.0, 0.05, 20),
    (3_500_000.0, 0.2, 10),
])
def test_typed_operands_win_over_the_stored_contract_data(amount, rate, days):
    ask = (f"Contract {amount:,.0f}, delay damages {rate:g}% per day, "
           f"{days} days late. Total?")
    out = cc.compose_delay_damages_over_period_from_excerpts(
        ask, _excerpts(STORED_ACA, STORED_RATE),
    )
    assert out is not None
    assert out["contract_amount"] == pytest.approx(amount)
    assert out["rates"] == [pytest.approx(rate)]
    assert out["amount"] == pytest.approx(amount * rate / 100.0 * days)
    line = cc.format_delay_damages_period_line(out)
    assert f"{amount * rate / 100.0:,.2f} per day" in line
    assert f"{amount * rate / 100.0 * days:,.2f}" in line
    assert "Accepted Contract Amount" not in line


@pytest.mark.parametrize("ask, days", [
    ("Contract 8,000,000, delay damages 0.05% per day, 20 days late. Total?", 20),
    ("Contract 8,000,000, delay damages 0.05% per day, 7 days late. Total?", 7),
])
def test_typed_operands_compose_with_no_excerpts(ask, days):
    out = cc.compose_delay_damages_over_period_from_excerpts(ask, "")
    assert out is not None
    assert out["amount"] == pytest.approx(8_000_000.0 * 0.05 / 100.0 * days)


@pytest.mark.parametrize("amount", [8_000_000.0, 2_250_000.0])
def test_a_typed_base_alone_takes_the_stored_rate(amount):
    ask = f"Contract {amount:,.0f}: what are the delay damages for 20 days late?"
    out = cc.compose_delay_damages_over_period_from_excerpts(
        ask, _excerpts(STORED_ACA, STORED_RATE),
    )
    assert out is not None
    assert out["contract_amount"] == pytest.approx(amount)
    assert out["amount"] == pytest.approx(amount * STORED_RATE / 100.0 * 20)


@pytest.mark.parametrize("rate", [0.05, 0.3])
def test_a_typed_rate_alone_takes_the_stored_base(rate):
    ask = f"Delay damages at {rate:g}% per day, 20 days late: total?"
    out = cc.compose_delay_damages_over_period_from_excerpts(
        ask, _excerpts(STORED_ACA, STORED_RATE),
    )
    assert out is not None
    assert out["contract_amount"] == pytest.approx(STORED_ACA)
    assert out["amount"] == pytest.approx(round(STORED_ACA * rate / 100.0 * 20, 2))


@pytest.mark.parametrize("aca, rate", [(1_200_000.0, 0.1), (5_400_000.0, 0.05)])
def test_with_nothing_typed_the_stored_figures_still_compose(aca, rate):
    ask = "What are the delay damages for 20 days late?"
    out = cc.compose_delay_damages_over_period_from_excerpts(ask, _excerpts(aca, rate))
    assert out is not None
    assert out["contract_amount"] == pytest.approx(aca)
    assert out["amount"] == pytest.approx(aca * rate / 100.0 * 20)
    assert "Accepted Contract Amount" in cc.format_delay_damages_period_line(out)


# ---- the daily compose ----------------------------------------------------

@pytest.mark.parametrize("amount, rate", [(8_000_000.0, 0.05), (3_500_000.0, 0.2)])
def test_typed_operands_win_in_the_daily_compose(amount, rate):
    ask = (f"Calculate the delay damages in SAR per day: contract SAR "
           f"{amount:,.0f}, delay damages {rate:g}% per day.")
    out = cc.compose_delay_damages_daily_from_excerpts(
        ask, _excerpts(STORED_ACA, STORED_RATE),
    )
    assert out is not None
    assert out["daily_amount"] == pytest.approx(amount * rate / 100.0)
    line = cc.format_delay_damages_daily_line(out)
    assert "Accepted Contract Amount" not in line
    assert "Contract Data" not in line
