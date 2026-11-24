"""Formulas from app.lib.construction_formulas_additions owned by the commercial hat.

Moved unchanged by scripts/move_formulas.py (F-DRIVER Phase A).
"""
from __future__ import annotations

import math

from app.lib.formula_registry import Param, formula


@formula(
    owner='commercial',
    display_name='Interim payment',
    description='Net interim payment from a gross valuation and a retention percentage.',
    inputs={
        'gross_valuation': Param('currency', 0, 1e13, label='gross valuation', basis='period'),
        'retention_percent': Param('%', 0, 100, label='retention'),
    },
    outputs={'gross_valuation': 'currency', 'retention_percent': '%', 'retention_amount': 'currency', 'net_payment': 'currency'},
)
def calculate_interim_payment(
    gross_valuation: float | int,
    retention_percent: float | int,
) -> dict:
    """
    Calculate interim payment after retention deduction.

    FIDIC Red Book Clause 14.3: the Engineer issues an Interim Payment
    Certificate within 28 days of receiving the Statement. Retention is
    deducted per Clause 14.3 unless the limit (typically 5% of Accepted
    Contract Amount) has been reached.

    Args:
        gross_valuation: Certified gross amount for this period.
        retention_percent: Retention percentage stated by the contract. It has
            no default: a missing rate is an error that names the input.

    Returns:
        dict with gross, retention, net payment, and breakdown.
    """
    gross = float(gross_valuation)
    if gross < 0:
        return {"error": "gross_valuation must be >= 0."}
    if retention_percent is None:
        return {"error": "retention_percent is required (the contract's retention rate); "
                         "it has no default.", "missing": ["retention_percent"]}
    if not (0.0 <= float(retention_percent) <= 100.0):
        return {"error": "retention_percent must be between 0 and 100."}
    retention_rate = float(retention_percent) / 100.0
    retention_amount = gross * retention_rate
    net_payment = gross - retention_amount

    return {
        "gross_valuation": round(gross, 2),
        "retention_percent": float(retention_percent),
        "retention_amount": round(retention_amount, 2),
        "net_payment": round(net_payment, 2),
        "calculation": f"{gross} - ({gross} x {retention_percent}%) = {round(net_payment, 2)}",
        "standard": "FIDIC Red Book Clause 14.3",
        "note": (
            "Retention at the rate given; FIDIC cap is 5% of Accepted Contract Amount. "
            "Verify if retention cap has been reached on this project."
        ),
    }


def _amount(x: float) -> str:
    return f"{x:,.2f}"


@formula(
    owner='commercial',
    display_name='Break-even point',
    description=('Quantity and revenue at which income covers the fixed costs: fixed '
                 'costs divided by the contribution per unit (price less variable cost '
                 'per unit). Only defined while the price is above the variable cost; '
                 'with an expected quantity, also the margin of safety and the profit.'),
    inputs={
        'fixed_costs': Param('currency', 0, 1e13, label='fixed costs for the period'),
        'price_per_unit': Param('currency', 0, 1e12, label='selling price per unit'),
        'variable_cost_per_unit': Param('currency', 0, 1e12,
                                        label='variable cost per unit'),
        'expected_units': Param('count', 0, 1e12, label='expected quantity'),
    },
    outputs={'break_even_units': 'nr', 'break_even_revenue': 'currency',
             'contribution_per_unit': 'currency', 'working': '-'},
)
def break_even_point(
    fixed_costs: float,
    price_per_unit: float,
    variable_cost_per_unit: float,
    expected_units: float | None = None,
) -> dict:
    """Break-even with its domain conditions stated, never assumed.

    Q* = F / (p - v) holds only while the contribution per unit p - v is
    positive: at or below zero every extra unit loses (or makes nothing) and
    there is no break-even quantity, which is said rather than divided
    through to a negative or infinite answer. Units are whole, so the
    smallest whole quantity that covers the fixed costs is given beside the
    exact figure. None of the three amounts has a default.
    """
    supplied = {"fixed_costs": fixed_costs, "price_per_unit": price_per_unit,
                "variable_cost_per_unit": variable_cost_per_unit}
    missing = [k for k, v in supplied.items() if v is None]
    if missing:
        return {"error": ", ".join(missing) + " required; no default is assumed.",
                "missing": missing}
    fixed = float(fixed_costs)
    price = float(price_per_unit)
    var = float(variable_cost_per_unit)
    if min(fixed, price, var) < 0:
        return {"error": "fixed_costs, price_per_unit and variable_cost_per_unit "
                         "must be >= 0."}
    if expected_units is not None and float(expected_units) < 0:
        return {"error": "expected_units must be >= 0."}

    contribution = price - var
    working = [(f"Contribution per unit = price {_amount(price)} - variable cost "
                f"{_amount(var)} = {_amount(contribution)}")]
    if contribution <= 0:
        working.append("The price does not exceed the variable cost per unit, so no "
                       "quantity covers the fixed costs: there is no break-even point.")
        return {"break_even_units": None, "break_even_revenue": None,
                "contribution_per_unit": round(contribution, 2),
                "domain": "no break-even: price <= variable cost per unit",
                "working": working}

    units = fixed / contribution
    whole = math.ceil(units - 1e-9)
    revenue = units * price
    ratio = contribution / price
    working += [
        (f"Break-even quantity = fixed costs {_amount(fixed)} / {_amount(contribution)} "
         f"= {units:,.4g} units (sell {whole:,} whole units)"),
        (f"Break-even revenue = {units:,.4g} x {_amount(price)} = {_amount(revenue)} "
         f"(contribution ratio {ratio:.2%})"),
    ]
    out = {
        "break_even_units": round(units, 4),
        "units_to_cover_fixed_costs": whole,
        "break_even_revenue": round(revenue, 2),
        "contribution_per_unit": round(contribution, 2),
        "contribution_ratio": round(ratio, 6),
    }
    if expected_units is not None:
        q = float(expected_units)
        profit = q * contribution - fixed
        working.append(f"Profit at {q:,.4g} units = {q:,.4g} x {_amount(contribution)} "
                       f"- {_amount(fixed)} = {_amount(profit)}")
        out["profit_at_expected"] = round(profit, 2)
        if q > 0:
            margin = (q - units) / q
            out["margin_of_safety_units"] = round(q - units, 4)
            out["margin_of_safety_percent"] = round(margin * 100.0, 2)
            working.append(
                f"Margin of safety = ({q:,.4g} - {units:,.4g}) / {q:,.4g} = {margin:.2%}"
                + (" (below break-even: a shortfall)" if q < units else ""))
    out["working"] = working
    return out
