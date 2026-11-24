"""Formulas from app.lib.construction_formulas_commercial owned by the commercial hat.

Moved unchanged by scripts/move_formulas.py (F-DRIVER Phase A).
"""
from __future__ import annotations
from app.lib.formula_registry import Param, formula


@formula(
    owner='commercial',
    display_name='Return on investment',
    description='Return on investment as a percentage of cost.',
    inputs={
        'gain': Param('currency', -1e13, 1e13, label='amount gained'),
        'cost': Param('currency', 0, 1e13, label='investment cost'),
    },
    outputs={'net_profit': '-', 'roi_percent': '%'},
)
def roi_calculator(gain: float, cost: float) -> dict:
    """Return on investment: ROI% = (gain - cost) / cost * 100."""
    g, c = float(gain), float(cost)
    if c == 0:
        return {"error": "cost must be non-zero to compute ROI"}
    net = g - c
    roi = net / c * 100.0
    return {
        "net_profit": round(net, 2),
        "roi_percent": round(roi, 2),
        "standard": "arithmetic",
        "note": f"ROI = (gain - cost)/cost*100 = ({g} - {c})/{c}*100 = {roi:.2f}%.",
    }


#: How a tender applies its percentage additions (overheads, profit, risk)
#: to the cost, keyed by the words a tender or a user writes for each.
#: ``on_cost``: every rate is a share of the cost, so the additions sum.
#: ``compounding``: every rate is a share of the running total so far, in
#: the order given, so each addition also earns the rates before it.
_MARKUP_BASIS_WORDS = {
    "on_cost": ("on cost", "oncost", "additive", "simple", "each on cost",
                "of cost", "summed", "sum"),
    "compounding": ("compounding", "compound", "compounded", "cascading",
                    "cascade", "cumulative", "on the running total",
                    "successive", "sequential"),
}


def _markup_basis(text: str) -> str:
    """The basis a written word names, or '' when it names neither."""
    words = " ".join(str(text or "").lower().replace("_", " ").replace("-", " ").split())
    for basis, names in _MARKUP_BASIS_WORDS.items():
        if words in names:
            return basis
    return ""


def _money(x: float) -> str:
    """A figure in full with two decimals, never as an exponent."""
    return f"{x:,.2f}"


@formula(
    owner='commercial',
    display_name='Markup build-up: on cost vs compounding',
    description=('Selling price from a cost and the tender-supplied percentage '
                 'additions (overheads, profit, risk) on the basis the tender '
                 'states: each rate on the cost (additions summed) or compounding '
                 '(each rate on the running total, in order). States both '
                 'results, the difference, and the margin the markup gives on '
                 'the selling price.'),
    inputs={
        'cost': Param('currency', 0, 1e13, label='cost before additions'),
        'rates_percent': Param('%', 0, 1000, label='tender percentage additions, in order'),
        'basis': Param('-', label='how the rates apply: on cost or compounding'),
    },
    outputs={'price': '-', 'markup_amount': '-', 'effective_markup_percent': '%',
             'margin_on_price_percent': '%', 'working': '-'},
)
def markup_build_up(cost: float, rates_percent: list, basis: str) -> dict:
    """Price = cost x (1 + sum r) on cost, or cost x product(1 + r) compounding.

    No rate and no basis has a default: both are the tender's. The answer
    states the other basis's price and the difference, so a reader sees what
    choosing the wrong one would cost, and the margin on the selling price,
    which a markup on cost is often mistaken for.
    """
    chosen = _markup_basis(basis)
    if not chosen:
        return {"error": ("basis must say how the tender applies its rates: "
                          "'on cost' (each rate on the cost, summed) or "
                          f"'compounding' (each rate on the running total); got {basis!r}.")}
    rates = [rates_percent] if not isinstance(rates_percent, (list, tuple)) else list(rates_percent)
    if not rates:
        return {"error": "At least one tender percentage addition is required."}
    rates = [float(r) for r in rates]
    bad = [i + 1 for i, r in enumerate(rates) if r < 0]
    if bad:
        return {"error": ("A percentage addition cannot be negative; it is for "
                          f"rate(s) {', '.join(map(str, bad))}.")}
    c = float(cost)
    if c < 0:
        return {"error": "cost cannot be negative."}

    on_cost = c * (1.0 + sum(rates) / 100.0)
    steps: list = []
    running = c
    for r in rates:
        add = running * r / 100.0
        steps.append((r, running, add, running + add))
        running += add
    compounding = running

    price = on_cost if chosen == "on_cost" else compounding
    other = compounding if chosen == "on_cost" else on_cost
    other_name = "compounding" if chosen == "on_cost" else "on cost"
    markup = price - c
    out: dict = {
        "basis": chosen,
        "price": round(price, 2),
        "markup_amount": round(markup, 2),
        "effective_markup_percent": round(markup / c * 100.0, 4) if c else 0.0,
        "margin_on_price_percent": round(markup / price * 100.0, 4) if price else 0.0,
        "price_on_cost": round(on_cost, 2),
        "price_compounding": round(compounding, 2),
        "difference_between_bases": round(compounding - on_cost, 2),
    }
    rate_list = " + ".join(f"{r:g}%" for r in rates)
    if chosen == "on_cost":
        working = [f"On cost: each rate is a share of the cost {_money(c)}; "
                   f"additions {rate_list} = {sum(rates):g}%.",
                   f"Price = {_money(c)} x (1 + {sum(rates):g}%) = {_money(on_cost)}."]
    else:
        working = ["Compounding: each rate is a share of the running total, in the order given."]
        working += [f"{_money(base)} + {r:g}% = {_money(base)} + {_money(add)} = {_money(total)}"
                    for r, base, add, total in steps]
        working.append(f"Price = {_money(compounding)}.")
    working += [
        (f"Markup {_money(markup)} = {out['effective_markup_percent']:g}% of cost, "
         f"which is {out['margin_on_price_percent']:g}% of the selling price (margin)."),
        (f"On the other basis ({other_name}) the price would be {_money(other)}; "
         f"the bases differ by {_money(abs(compounding - on_cost))}."),
    ]
    out["working"] = working
    return out
