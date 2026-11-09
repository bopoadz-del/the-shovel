"""Productivity benchmark table (SPEC_09NOV_PROJECT_CONTROL_FORMULAS_AND_RATES,
section B). Owner-supplied planner template, UNVERIFIED.

Rate = units per crew-day on a 10-hour day. GENERAL tier, MEDIUM confidence:
these are never a project figure; a project question gets its rate from the
project's own documents or asks for it.
"""
from __future__ import annotations

from app.lib.formula_registry import Param, formula

_BENCHMARK_PROVENANCE = (
    "owner-supplied planner template (GCC residential tower, finishing trades), unverified"
)

# (work item, unit, crew, rate per crew-day, flag note)
_PRODUCTIVITY_BENCHMARKS = [
    ("blockwork walls", "m2", "mason + helper (1:1)", 12.0, ""),
    ("internal wall plaster", "m2", "mason + helper", 20.0, ""),
    ("ceiling plaster", "m2", "mason", 20.0, ""),
    ("wall paint primer and stucco", "m2", "painter", 45.0, "painting rate at the low end of the usual GCC range"),
    ("emulsion paint walls final coat", "m2", "painter", 45.0, "painting rate at the low end of the usual GCC range"),
    ("acrylic emulsion ceilings", "m2", "painter", 45.0, "painting rate at the low end of the usual GCC range"),
    ("ceramic tiles floors", "m2", "mason (tiler)", 40.0, ""),
    ("ceramic tiles walls", "m2", "mason (tiler)", 40.0, "wall tiling at 40 is optimistic vs floor; flag"),
    ("ceiling finishes false ceiling", "m2", "mason", 40.0, ""),
    ("wall finishes common areas", "m2", "mason", 40.0, ""),
    ("floor finishes common areas", "m2", "mason", 40.0, ""),
    ("epoxy paint floors", "m2", "painter", 40.0, "painting rate at the low end of the usual GCC range"),
    ("epoxy paint ceilings", "m2", "painter", 40.0, "painting rate at the low end of the usual GCC range"),
    ("ceiling painting common areas", "m2", "painter", 40.0, "painting rate at the low end of the usual GCC range"),
]

_BENCHMARK_KEYS = {item: (unit, crew, rate, flag) for item, unit, crew, rate, flag in _PRODUCTIVITY_BENCHMARKS}


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
        "flag": "-",
    },
)
def productivity_benchmark(work_item: str) -> dict:
    """Look up the benchmark rate for a finishing-trade work item.

    GENERAL tier / MEDIUM confidence: the returned provenance says
    "owner-supplied planner template, unverified", and an answer using this
    rate must say so rather than present it as a project figure.
    """
    key = (work_item or "").strip().lower()
    if key not in _BENCHMARK_KEYS:
        return {
            "work_item": work_item,
            "rate_per_crew_day": None,
            "provenance": _BENCHMARK_PROVENANCE,
            "flag": "no benchmark entry for this work item; use the project's own rate",
        }
    unit, crew, rate, flag = _BENCHMARK_KEYS[key]
    return {
        "work_item": work_item,
        "unit": unit,
        "crew": crew,
        "rate_per_crew_day": rate,
        "provenance": _BENCHMARK_PROVENANCE,
        "flag": flag,
        "note": f"{work_item}: {rate} {unit} / crew-day ({crew}). {_BENCHMARK_PROVENANCE}."
                + (f" FLAG: {flag}." if flag else ""),
    }
