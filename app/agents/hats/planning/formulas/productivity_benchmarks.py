"""Productivity benchmark table (SPEC_09NOV_PROJECT_CONTROL_FORMULAS_AND_RATES,
section B). Owner-supplied planner template, UNVERIFIED.

Rate = units per crew-day on a 10-hour day. GENERAL tier, MEDIUM confidence:
these are never a project figure; a project question gets its rate from the
project's own documents or asks for it.
"""
from __future__ import annotations

from app.lib.formula_registry import Param, formula
from app.lib.productivity_norms import TEMPLATE_PROVENANCE, benchmark_row

# The rows live in ONE place, app/lib/productivity_norms.py (D1), which the
# schedule path also reads; this calculator keeps no copy of its own.
_BENCHMARK_PROVENANCE = TEMPLATE_PROVENANCE


@formula(
    owner="planning",
    display_name="Productivity benchmark",
    description=(
        "Indicative productivity rate (units per crew-day) for a finishing "
        "trade, from the owner-supplied planner template. Unverified."
    ),
    inputs={
        "work_item": Param("-", label="work item"),
    },
    outputs={
        "work_item": "-",
        "unit": "-",
        "crew": "-",
        "rate_per_crew_day": "count/d",
        "provenance": "-",
        "tier": "-",
        "confidence": "-",
        "flag": "-",
    },
)
def productivity_benchmark(work_item: str) -> dict:
    """Look up the benchmark rate for a finishing-trade work item.

    GENERAL tier / MEDIUM confidence: the returned provenance says
    "owner-supplied planner template, unverified", and an answer using this
    rate must say so rather than present it as a project figure.
    """
    row = benchmark_row(work_item)
    if row is None:
        return {
            "work_item": work_item,
            "rate_per_crew_day": None,
            "provenance": _BENCHMARK_PROVENANCE,
            "tier": "GENERAL",
            "confidence": "MEDIUM",
            "flag": "no benchmark entry for this work item; use the project's own rate",
        }
    unit, crew, rate, flag = row["unit"], row["crew"], row["rate"], row["flag"]
    return {
        "work_item": work_item,
        "unit": unit,
        "crew": crew,
        "rate_per_crew_day": rate,
        "provenance": _BENCHMARK_PROVENANCE,
        "tier": "GENERAL",
        "confidence": "MEDIUM",
        "flag": flag,
        "note": f"{work_item}: {rate} {unit} / crew-day ({crew}). {_BENCHMARK_PROVENANCE}."
                + (f" FLAG: {flag}." if flag else ""),
    }
