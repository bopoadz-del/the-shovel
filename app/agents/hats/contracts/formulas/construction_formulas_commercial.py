"""Formulas from app.lib.construction_formulas_commercial owned by the contracts hat.

Moved unchanged by scripts/move_formulas.py (F-DRIVER Phase A).
"""
from __future__ import annotations
from app.lib.formula_registry import Param, formula


@formula(
    owner='contracts',
    display_name='Delay damages per day',
    description=(
        "Daily delay damages: a flat amount per day, or the contract's rate as a "
        "percent of the contract amount; and their total over the days of delay when given."
    ),
    inputs={
        'rate_percent': Param('%', 0, 100, label='delay damages rate'),
        'contract_amount': Param('currency', 0, 1e13, label='contract amount'),
        'currency': Param('-', label='currency'),
        'amount_per_day': Param('currency/day', 0, 1e13, label='flat daily delay damages amount'),
        'days': Param('day', 0, 1e9, label='days of delay'),
    },
    outputs={'total_amount': 'currency', 'daily_amount': 'currency', 'rate_percent': '%',
             'contract_amount': 'currency', 'days': 'day', 'currency': '-', 'per': '-'},
    required_groups=(('rate_percent', 'amount_per_day'), ('contract_amount', 'amount_per_day')),
)
def delay_damages_daily(
    rate_percent: float = 0.0,
    contract_amount: float = 0.0,
    currency: str = "",
    amount_per_day: float = 0.0,
    days: float = 0.0,
) -> dict:
    """Daily delay damages, and their total over ``days`` when given.

    The contract states either a flat ``amount_per_day``, or (FIDIC
    Sub-Clause 8.8) a rate: rate% × Accepted Contract Amount / Contract
    Price per calendar day. Operands are parameters -- this function does
    not invent a rate, an amount or a period.
    """
    pct = float(rate_percent)
    base = float(contract_amount)
    if pct < 0 or base < 0:
        return {"error": "rate_percent and contract_amount must be >= 0."}
    flat = float(amount_per_day)
    n_days = float(days)
    if flat < 0 or n_days < 0:
        return {"error": "amount_per_day and days must be >= 0."}
    cur = (currency or "").strip().upper()

    def _money(x: float) -> str:
        # Currency is a label, not an operand: when the data carried none, the
        # figure stands on its own -- never an invented symbol (K-T5, ruling ii).
        return f"{cur} {x:,.2f}" if cur else f"{x:,.2f}"

    if flat > 0:
        daily = round(flat, 2)
        out = {
            "daily_amount": daily,
            "currency": cur or None,
            "per": "calendar day",
            "note": f"{_money(daily)} per calendar day.",
        }
    else:
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
        daily = round(base * (pct / 100.0), 2)
        out = {
            "daily_amount": daily,
            "rate_percent": pct,
            "contract_amount": round(base, 2),
            "currency": cur or None,
            "per": "calendar day",
            "standard": "FIDIC Sub-Clause 8.8 (rate × Accepted Contract Amount)",
            "note": (
                f"{pct:g}% of {_money(base)} = {_money(daily)} "
                f"per calendar day."
            ),
        }
    if n_days > 0:
        total = round(daily * n_days, 2)
        out["days"] = n_days
        out["total_amount"] = total
        out["note"] += f" Over {n_days:g} days: {_money(total)}."
    return out
