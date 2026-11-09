"""Project-control formulas from the owner's planner template pack
(SPEC_09NOV_PROJECT_CONTROL_FORMULAS_AND_RATES.md).

Generic: variable names are the contract, not Excel cells. No sheet data, no
project names, no activity ids. Source: owner-supplied project-control
template, 09 Nov 2026; standard planner practice.
"""
from __future__ import annotations

import math

from app.lib.formula_registry import Param, formula


@formula(
    owner="planning",
    display_name="Crew sizing and productivity",
    description=(
        "Crews required, crew-days and man-hours for a quantity at a given "
        "productivity rate, plus the duration a stated crew size gives."
    ),
    inputs={
        "quantity": Param("count", 0, 1e12, label="quantity"),
        "duration_days": Param("days", 0, 1e9, label="duration in working days"),
        "rate_per_crew_day": Param("count/d", 0, 1e9, label="rate (units per crew-day)"),
        "hours_per_day": Param("h/d", 0, 24, label="working hours per day"),
        "crew_size": Param("count", 0, 1e6, label="crew size"),
    },
    outputs={
        "crews_required": "-",
        "crews_required_rounded_up": "-",
        "crew_days": "days",
        "man_hours": "h",
        "duration_for_crews": "days",
        "identity_check": "-",
    },
)
def crew_sizing(
    quantity: float,
    duration_days: float,
    rate_per_crew_day: float,
    hours_per_day: float = 10.0,
    crew_size: float | None = None,
) -> dict:
    """Productivity / crew sizing.

    crews_required = Q / (D * R); crew_days = Q / R; man_hours = crew_days * h.
    Fractional crews are reported rounded UP with the rounding shown, and the
    identity crews * D * R = Q is checked. ``crew_size`` (N) gives
    duration_for_crews = Q / (R * N). R, D and h are inputs with no rate
    baked in; the template's 10-hour day is only a suggested default.
    """
    q, d, r, h = float(quantity), float(duration_days), float(rate_per_crew_day), float(hours_per_day)
    if min(q, d, r, h) <= 0:
        return {"error": "quantity, duration_days, rate_per_crew_day and hours_per_day must all be > 0."}
    crews = q / (d * r)
    crews_up = math.ceil(crews)
    crew_days = q / r
    man_hours = crew_days * h
    note = (
        f"crews = {q} / ({d} x {r}) = {crews:.3f}; rounded UP to {crews_up}. "
        f"crew-days = {q} / {r} = {crew_days:.2f}; man-hours = crew-days x {h} = {man_hours:.1f}. "
        f"Identity check: {crews_up} x {d} x {r} = {crews_up * d * r:.2f} (stated Q = {q})."
    )
    result: dict = {
        "crews_required": round(crews, 3),
        "crews_required_rounded_up": crews_up,
        "crew_days": round(crew_days, 2),
        "man_hours": round(man_hours, 1),
        "identity_check": round(crews_up * d * r, 2),
        "standard": "owner-supplied project-control template, 09 Nov 2026; standard planner practice",
        "note": note,
    }
    if crew_size is not None and float(crew_size) > 0:
        result["duration_for_crews"] = round(q / (r * float(crew_size)), 2)
    else:
        result["duration_for_crews"] = None
    return result


@formula(
    owner="planning",
    display_name="Progress KPI (SPI and variance)",
    description=(
        "Planned and actual percent complete, the progress variance and a "
        "progress-based SPI (earned by schedule, not EVM cost-based)."
    ),
    inputs={
        "planned_cum": Param("count", 0, 1e12, label="planned cumulative to date"),
        "actual_cum": Param("count", 0, 1e12, label="actual cumulative to date"),
        "planned_total": Param("count", 0, 1e12, label="planned total at finish"),
    },
    outputs={
        "planned_pct": "%",
        "actual_pct": "%",
        "variance_pct": "%",
        "spi_progress": "-",
    },
)
def progress_kpi(
    planned_cum: float,
    actual_cum: float,
    planned_total: float,
) -> dict:
    """Planned vs actual + KPI.

    Percent complete is against the baseline total (planned_total). The SPI
    here is progress-based (actual_cum / planned_cum), which is what these
    jobs use - it is stated as such, never an EVM cost-based CPI/SPI.
    """
    pc, ac, tot = float(planned_cum), float(actual_cum), float(planned_total)
    if tot <= 0:
        return {"error": "planned_total must be > 0."}
    planned_pct = pc / tot * 100
    actual_pct = ac / tot * 100
    variance = actual_pct - planned_pct
    spi = ac / pc if pc > 0 else None
    note = (
        f"planned = {pc} / {tot} = {planned_pct:.1f}%; actual = {ac} / {tot} = "
        f"{actual_pct:.1f}%; variance = {variance:+.1f}%. SPI (progress-based) = "
        f"{ac} / {pc} = {spi:.3f}." if spi is not None else
        f"planned = {pc} / {tot} = {planned_pct:.1f}%; actual = {ac} / {tot} = "
        f"{actual_pct:.1f}%; variance = {variance:+.1f}%. SPI undefined (planned_cum = 0)."
    )
    return {
        "planned_pct": round(planned_pct, 1),
        "actual_pct": round(actual_pct, 1),
        "variance_pct": round(variance, 1),
        "spi_progress": round(spi, 3) if spi is not None else None,
        "standard": "progress-based SPI (planner template, not EVM cost-based)",
        "note": note,
    }


@formula(
    owner="planning",
    display_name="Contractor cash flow (monthly)",
    description=(
        "Advance, retention, certified valuations, net cash and the peak "
        "funding requirement from a monthly value and cost series."
    ),
    inputs={
        "monthly_value": Param("currency", 0, 1e12, label="projected monthly work value (series)"),
        "monthly_cost": Param("currency", 0, 1e12, label="monthly cost (series)"),
        "advance_pct": Param("ratio", 0, 1, label="advance % (suggested 10%)"),
        "retention_pct": Param("ratio", 0, 1, label="retention % (suggested 5%)"),
        "payment_lag_days": Param("days", 0, 365, label="payment lag days (suggested 15)"),
    },
    outputs={
        "contract_value": "currency",
        "advance_payment": "currency",
        "retention_total": "currency",
        "peak_funding_requirement": "currency",
        "check_advance_recovery": "currency",
        "check_retention": "currency",
        "check_valuation": "currency",
    },
)
def cash_flow(
    monthly_value: list,
    monthly_cost: list,
    advance_pct: float = 0.10,
    retention_pct: float = 0.05,
    payment_lag_days: float = 15.0,
) -> dict:
    """Contractor cash flow.

    advance paid at month 0, recovered pro-rata; retention deducted monthly
    and released in full at completion (last month); payments arrive lagged by
    ceil(payment_lag_days / 30) months. advance_pct / retention_pct / lag are
    inputs (10% / 5% / 15 days are only suggested defaults).
    """
    V = [float(v) for v in (monthly_value or [])]
    C = [float(c) for c in (monthly_cost or [])]
    if not V:
        return {"error": "monthly_value must be a non-empty series."}
    if len(C) != len(V):
        return {"error": "monthly_cost must match monthly_value length."}
    ap, rp = float(advance_pct), float(retention_pct)
    lag = math.ceil(float(payment_lag_days) / 30.0)
    contract_value = sum(V)
    advance_payment = ap * contract_value
    retention_total = rp * contract_value
    advance_recovery = [-ap * v for v in V]
    retention_deduction = [-rp * v for v in V]
    certified = [v + ar + rd for v, ar, rd in zip(V, advance_recovery, retention_deduction)]
    certified[-1] += retention_total  # release at completion
    cash_in = [0.0] * len(V)
    for m in range(len(V)):
        if m - lag >= 0:
            cash_in[m] = certified[m - lag]
    net = [cash_in[m] - C[m] for m in range(len(V))]
    cum = [0.0] * len(V)
    running = advance_payment
    for m in range(len(V)):
        running += net[m]
        cum[m] = running
    peak = max(0.0, -min(cum))
    return {
        "contract_value": round(contract_value, 2),
        "advance_payment": round(advance_payment, 2),
        "retention_total": round(retention_total, 2),
        "peak_funding_requirement": round(peak, 2),
        "check_advance_recovery": round(sum(advance_recovery), 2),
        "check_retention": round(sum(retention_deduction), 2),
        "check_valuation": round(sum(certified), 2),
        "standard": "owner-supplied project-control template, 09 Nov 2026; standard planner practice",
        "note": (
            f"contract value = {contract_value:,.2f}; advance {ap:.0%} = "
            f"{advance_payment:,.2f} at month 0; retention {rp:.0%} = "
            f"{retention_total:,.2f} released at completion; payment lag "
            f"{lag} month(s); peak funding = {peak:,.2f}."
        ),
    }


@formula(
    owner="planning",
    display_name="Manhours S-curve",
    description=(
        "Planned vs actual manhours percent complete and a productivity index "
        "(progress earned per manhour)."
    ),
    inputs={
        "planned_mh": Param("count", 0, 1e12, label="planned manhours (series)"),
        "actual_mh": Param("count", 0, 1e12, label="actual manhours to date (series)"),
    },
    outputs={
        "cum_planned_pct": "%",
        "cum_actual_pct": "%",
        "productivity_index": "-",
    },
)
def manhours_scurve(planned_mh: list, actual_mh: list) -> dict:
    """Manhours S-curve. Actual is normalised to the PLANNED total, and the
    productivity index is progress earned per manhour."""
    planned = [float(x) for x in (planned_mh or [])]
    actual = [float(x) for x in (actual_mh or [])]
    planned_total = sum(planned)
    if planned_total <= 0:
        return {"error": "planned_mh must sum to more than 0."}
    n = min(len(planned), len(actual))
    cum_planned = sum(planned[:n])
    cum_actual = sum(actual[:n])
    cum_planned_pct = cum_planned / planned_total * 100
    cum_actual_pct = cum_actual / planned_total * 100
    prod = (cum_planned_pct / cum_actual_pct) if cum_actual_pct > 0 else None
    note = (
        f"planned {cum_planned:,.0f} / {planned_total:,.0f} = {cum_planned_pct:.1f}%; "
        f"actual {cum_actual:,.0f} / {planned_total:,.0f} = {cum_actual_pct:.1f}%; "
        + (f"productivity index = {cum_planned_pct:.1f} / {cum_actual_pct:.1f} = {prod:.3f}."
           if prod is not None else "productivity index undefined (no actual manhours).")
    )
    return {
        "cum_planned_pct": round(cum_planned_pct, 1),
        "cum_actual_pct": round(cum_actual_pct, 1),
        "productivity_index": round(prod, 3) if prod is not None else None,
        "standard": "owner-supplied project-control template, 09 Nov 2026; standard planner practice",
        "note": note,
    }


@formula(
    owner="planning",
    display_name="Floor-cycle progress weighting",
    description=(
        "Weighted progress across floor/zone elements from completed and total "
        "units per element."
    ),
    inputs={
        "completed_units": Param("count", 0, 1e12, label="completed units (series)"),
        "total_units": Param("count", 0, 1e12, label="total units (series)"),
        "weights": Param("count", 0, 1e12, label="element weights (series, input)"),
    },
    outputs={
        "weighted_progress": "-",
        "element_progress": "-",
    },
)
def floor_cycle_progress(
    completed_units: list,
    total_units: list,
    weights: list,
) -> dict:
    """Floor-cycle weighting. element_progress = completed/total per element;
    weighted_progress = SUM(element_progress * weight) / SUM(weight). Weights are
    inputs (the template used ~0.9 of an activity complete at a stage)."""
    comp = [float(x) for x in (completed_units or [])]
    tot = [float(x) for x in (total_units or [])]
    wt = [float(x) for x in (weights or [])]
    if not (len(comp) == len(tot) == len(wt)) or not comp:
        return {"error": "completed_units, total_units and weights must be non-empty and equal length."}
    elements = []
    num = 0.0
    den = 0.0
    for c, t, w in zip(comp, tot, wt):
        if t <= 0:
            return {"error": "total_units must be > 0 for every element."}
        p = c / t
        elements.append(round(p, 3))
        num += p * w
        den += w
    weighted = num / den if den > 0 else None
    note = (
        f"element progress = {elements}; weighted progress = "
        + (f"{weighted:.3f}." if weighted is not None else "undefined (weights sum to 0).")
    )
    return {
        "weighted_progress": round(weighted, 3) if weighted is not None else None,
        "element_progress": elements,
        "standard": "owner-supplied project-control template, 09 Nov 2026; standard planner practice",
        "note": note,
    }

