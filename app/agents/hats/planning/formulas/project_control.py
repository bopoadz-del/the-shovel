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
        "hours_per_day": Param("h/d", 0, 24, label="hours per day"),
        "crew_size": Param("count", 0, 1e6, label="crew size (optional, for duration)"),
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
    """A4 productivity / crew sizing.

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
    """A2 planned vs actual + KPI.

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
