"""Formulas from app.lib.construction_formulas_commercial owned by the contracts hat.

Moved unchanged by scripts/move_formulas.py (F-DRIVER Phase A).
"""
from __future__ import annotations
from app.lib.formula_registry import Param, formula


@formula(
    owner='contracts',
    display_name='Delay damages per day',
    description="Daily delay damages from the contract's daily rate and the contract amount it applies to.",
    inputs={
        'rate_percent': Param('%', 0, 100, label='delay damages rate'),
        'contract_amount': Param('currency', 0, 1e13, label='contract amount'),
        'currency': Param('-', label='currency'),
    },
    outputs={'daily_amount': 'currency', 'rate_percent': '%', 'contract_amount': 'currency', 'currency': '-', 'per': '-'},
)
def delay_damages_daily(
    rate_percent: float = 0.0,
    contract_amount: float = 0.0,
    currency: str = "",
) -> dict:
    """Daily delay damages = rate% × Accepted Contract Amount / Contract Price.

    FIDIC Sub-Clause 8.8: the Contractor pays the rate stated in the
    Contract Data for every calendar day of delay. The live daily amount is 0.1% of
    the net ACA. Operands are parameters — this function does not invent
    a rate or an amount.
    """
    pct = float(rate_percent)
    base = float(contract_amount)
    if pct < 0 or base < 0:
        return {"error": "rate_percent and contract_amount must be >= 0."}
    # Empty-args / unbound class: defaults are 0, 0. Emitting
    # "0% of SAR 0.00" looks like a successful rate and poisons the
    # turn (live per-milestone delay-damages ask). A genuine 0% rate against a real base is
    # still 0/day — only a missing contract amount is unbound.
    if base <= 0:
        return {
            "error": (
                "delay_damages_daily needs a contract_amount "
                "(unbound delay-damages / empty rates)."
            )
        }
    # The contract's own currency, when given; none is assumed.
    cur = str(currency or "").strip().upper()
    daily = round(base * (pct / 100.0), 2)
    return {
        "daily_amount": daily,
        "rate_percent": pct,
        "contract_amount": round(base, 2),
        "currency": cur,
        "per": "calendar day",
        "standard": "FIDIC Sub-Clause 8.8 (rate × Accepted Contract Amount)",
        "note": (
            f"{pct:g}% of {(cur + ' ') if cur else ''}{base:,.2f} = "
            f"{(cur + ' ') if cur else ''}{daily:,.2f} per calendar day."
        ),
    }
