"""Formulas from app.lib.construction_formulas_additions owned by the commercial hat.

Moved unchanged by scripts/move_formulas.py (F-DRIVER Phase A).
"""
from __future__ import annotations

import math
from decimal import Decimal, ROUND_HALF_UP, localcontext

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


def _to_cent(x: float | Decimal) -> Decimal:
    """An amount to the cent, half up, on the decimal figure (no binary
    residue decides the last cent)."""
    d = x if isinstance(x, Decimal) else _as_written(x)
    return d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP) + 0


def _amount(x: float | Decimal) -> str:
    text = format(_to_cent(x), ",f")
    return "0.00" if text == "-0.00" else text


def _as_written(x: float) -> Decimal:
    """A figure as the caller wrote it (the shortest decimal that reads back
    as the same float), so sums and quotients of figures with cents carry no
    binary residue: 0.30 - 0.10 is 0.2, not 0.19999999999999998."""
    return Decimal(repr(float(x)))


def _figure(x: float | Decimal) -> str:
    """A working figure written out in full: thousands separators, no
    exponent and every digit the figure has, so an input prints as written
    and the figures a step prints give the result printed after it."""
    d = x if isinstance(x, Decimal) else _as_written(x)
    text = format(d.normalize(), ",f")
    return "0" if text in ("", "-0") else text


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

    with localcontext() as exact:
        # Exact decimal arithmetic on the figures as written, wide enough to
        # hold any amount a float can to the cent: no step is rounded and no
        # figure inside the declared ranges runs out of digits.
        exact.prec = _EXACT_DIGITS
        return _break_even_figures(fixed, price, var,
                                   None if expected_units is None else float(expected_units))


#: Digits the break-even arithmetic keeps. The largest float has 309 digits
#: before the point; a revenue is a quantity a float holds times a price of
#: at most 1e12, written to the cent. 400 covers every such figure whole.
_EXACT_DIGITS = 400


_TOO_LARGE_TO_STATE = ("The price is above the variable cost per unit by so little that the "
                       "break-even quantity for these fixed costs is too large to state. "
                       "Check the price and the variable cost per unit.")


def _stated(out: dict) -> dict:
    """The result as it may be handed on: every number in it can be written.
    A figure no float holds is refused in plain words, never returned as an
    infinite number."""
    if any(isinstance(v, float) and not math.isfinite(v) for v in out.values()):
        return {"error": _TOO_LARGE_TO_STATE}
    return out


def _break_even_figures(fixed: float, price: float, var: float, q: float | None) -> dict:
    """The break-even figures and their working, in exact decimal arithmetic
    on the figures as written: a residue would put the break-even a hair above
    a whole quantity and call an expected volume equal to it a shortfall. Run
    inside the caller's wide decimal context."""
    contribution_d = _as_written(price) - _as_written(var)
    contribution = float(contribution_d)
    working = [(f"Contribution per unit = price {_figure(price)} - variable cost "
                f"{_figure(var)} = {_figure(contribution_d)}")]
    q_d = None if q is None else _as_written(q)

    def _profit_at_expected(out: dict) -> None:
        # Stated on both sides of the domain condition: below it the
        # expected volume makes a loss, and that loss is the answer.
        if q is None:
            return
        profit_d = q_d * contribution_d - _as_written(fixed)
        working.append(f"Profit at {_figure(q)} units = {_figure(q)} x "
                       f"{_figure(contribution_d)} - {_figure(fixed)} = {_amount(profit_d)}")
        out["profit_at_expected"] = float(_to_cent(profit_d)) + 0.0

    if contribution <= 0:
        working.append("The price does not exceed the variable cost per unit, so no "
                       "quantity covers the fixed costs: there is no break-even point.")
        out = {"break_even_units": None, "break_even_revenue": None,
               "contribution_per_unit": contribution,
               "domain": "no break-even: price <= variable cost per unit"}
        _profit_at_expected(out)
        out["working"] = working
        return _stated(out)

    units_d = _as_written(fixed) / contribution_d
    units = float(units_d)
    if not math.isfinite(units):
        # The quantity is defined but beyond any figure that can be written:
        # said plainly, never returned as an infinite number.
        return {"error": _TOO_LARGE_TO_STATE}
    # The quantity as the working prints it; the revenue is that printed
    # figure times the price, so the step adds up as read.
    units_shown = _as_written(units)
    whole = math.ceil(units_d)
    revenue_d = units_shown * _as_written(price)
    ratio = float(contribution_d / _as_written(price))
    working += [
        (f"Break-even quantity = fixed costs {_figure(fixed)} / {_figure(contribution_d)} "
         f"= {_figure(units_shown)} units (sell {whole:,} whole "
         f"unit{'' if whole == 1 else 's'})"),
        (f"Break-even revenue = {_figure(units_shown)} x {_figure(price)} = "
         f"{_amount(revenue_d)} (contribution ratio {ratio:.2%})"),
    ]
    out = {
        "break_even_units": round(units, 4),
        "units_to_cover_fixed_costs": whole,
        "break_even_revenue": float(_to_cent(revenue_d)),
        "contribution_per_unit": contribution,
        "contribution_ratio": round(ratio, 6),
    }
    _profit_at_expected(out)
    if q is not None:
        if q > 0:
            # The percentage is worked in the decimal context: a break-even
            # quantity many times the expected one gives a ratio no float holds.
            percent = float((q_d - units_d) / q_d * 100)
            out["margin_of_safety_units"] = round(float(q_d - units_d), 4)
            if math.isfinite(percent):
                out["margin_of_safety_percent"] = round(percent, 2) + 0.0
                working.append(
                    f"Margin of safety = ({_figure(q)} - {_figure(units_shown)}) / {_figure(q)} "
                    f"= {percent:.2f}%"
                    + (" (below break-even: a shortfall)" if q_d < units_d else ""))
            else:
                # Defined, but beyond any percentage that can be written:
                # said in words, and no percentage is returned.
                working.append(
                    f"Margin of safety: the break-even quantity is so many times the expected "
                    f"{_figure(q)} units that the shortfall cannot be stated as a percentage.")
    out["working"] = working
    return _stated(out)
