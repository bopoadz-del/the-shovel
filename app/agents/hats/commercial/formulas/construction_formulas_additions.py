"""Formulas from app.lib.construction_formulas_additions owned by the commercial hat.

Moved unchanged by scripts/move_formulas.py (F-DRIVER Phase A).
"""
from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal, localcontext

from app.lib.formula_registry import Param, formula


def _cents(x: float | Decimal) -> Decimal:
    """An amount to the cent, half up, on the figure as written (the shortest
    decimal that reads back as the same number), so no binary residue decides
    the last cent. Never a negative zero."""
    d = x if isinstance(x, Decimal) else Decimal(repr(float(x)))
    return d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP) + 0


def _money(x: float | Decimal) -> str:
    return f"{_cents(x):,.2f}"


@formula(
    owner='commercial',
    display_name='Interim payment to date',
    standard='FIDIC Clause 14.3',
    description=('Amount due on an interim payment certificate: work done to date plus '
                 'materials on site, less retention at the contract rate up to its limit, '
                 'less the amount certified on earlier certificates.'),
    inputs={
        'work_value_to_date': Param('currency', 0, 1e13, label='value of work done to date',
                                    basis='cumulative'),
        'retention_percent': Param('%', 0, 100, label='retention'),
        'previously_certified': Param('currency', 0, 1e13,
                                      label='amount certified on earlier certificates',
                                      basis='cumulative'),
        'materials_on_site': Param('currency', 0, 1e13, label='materials on site',
                                   basis='cumulative'),
        'retention_limit': Param('currency', 0, 1e13, label='retention limit'),
    },
    outputs={'gross_to_date': 'currency', 'retention_to_date': 'currency',
             'net_to_date': 'currency', 'amount_due': 'currency', 'working': '-'},
)
def interim_payment_to_date(
    work_value_to_date: float,
    retention_percent: float,
    previously_certified: float,
    materials_on_site: float = 0.0,
    retention_limit: float | None = None,
) -> dict:
    """Interim certificate worked from the totals to date.

    Every amount is a total to date, so the amount due is what is owed now
    less what earlier certificates already certified -- not a period figure
    with retention taken again. Retention is held on the work done, at the
    contract's rate, until it reaches the contract's limit; materials on site
    are added after retention (FIDIC 14.3: retention at (c), plant and
    materials at (e)). The rate, the earlier certificates and the limit are
    the contract's figures: the rate and the earlier certificates have no
    default, and with no limit given none is applied and the working says so.
    A total of earlier certificates above the net to date gives a negative
    amount due, stated as such, never clipped to nil.
    """
    if retention_percent is None:
        return {"error": "retention_percent is required (the contract's retention rate); "
                         "it has no default.", "missing": ["retention_percent"]}
    if previously_certified is None:
        return {"error": "previously_certified is required (the total of earlier "
                         "certificates; 0 for the first); it has no default.",
                "missing": ["previously_certified"]}
    work = float(work_value_to_date)
    mos = float(materials_on_site or 0.0)
    prev = float(previously_certified)
    rate = float(retention_percent) + 0.0   # a rate of -0.0 is 0
    if min(work, mos, prev) < 0:
        return {"error": "work_value_to_date, materials_on_site and previously_certified "
                         "must be >= 0."}
    if not (0.0 <= rate <= 100.0):
        return {"error": "retention_percent must be between 0 and 100."}
    if retention_limit is not None and float(retention_limit) < 0:
        return {"error": "retention_limit must be >= 0."}

    # Amounts are money. Every amount is taken to the cent as it comes in
    # and every step is worked on those cent figures in exact decimal
    # arithmetic, so each line of the working adds up on the figures it
    # prints, and earlier certificates equal to the net to the cent --
    # however their total was added up -- give a due of exactly nil.
    with localcontext() as exact:
        exact.prec = 60
        work_c, mos_c, prev_c = _cents(work), _cents(mos), _cents(prev)
        limit_c = None if retention_limit is None else _cents(float(retention_limit))
        gross = work_c + mos_c
        retention_at_rate = _cents(work_c * Decimal(repr(rate)) / 100)
        capped = limit_c is not None and retention_at_rate > limit_c
        retention = limit_c if capped else retention_at_rate
        net = gross - retention
        due = net - prev_c

    working = [
        f"Gross to date = work done {_money(work_c)} + materials on site {_money(mos_c)} "
        f"= {_money(gross)}",
        f"Retention = {rate:g}% x work done {_money(work_c)} = {_money(retention_at_rate)}"
        + (f", held at the retention limit {_money(limit_c)}" if capped
           else (f" (limit {_money(limit_c)} not reached)"
                 if limit_c is not None else " (no retention limit given)")),
        f"Net to date = {_money(gross)} - {_money(retention)} = {_money(net)}",
        f"Due this certificate = {_money(net)} - earlier certificates {_money(prev_c)} "
        f"= {_money(due)}",
    ]
    out = {
        "gross_to_date": float(gross),
        "retention_to_date": float(retention),
        "retention_limit_reached": capped,
        "net_to_date": float(net),
        "previously_certified": float(prev_c),
        "amount_due": float(due),
        "working": working,
        "standard": "FIDIC Clause 14.3",
    }
    if due < 0:
        out["note"] = ("Earlier certificates exceed the net value to date: this "
                       "certificate is a deduction of " + _money(-due) + ".")
    return out


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
