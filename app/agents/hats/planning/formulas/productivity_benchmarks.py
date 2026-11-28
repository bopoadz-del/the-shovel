"""Productivity benchmark table (SPEC_09NOV_PROJECT_CONTROL_FORMULAS_AND_RATES,
section B). Owner-supplied planner template, UNVERIFIED.

Rate = units per crew-day on a 10-hour day. GENERAL tier, MEDIUM confidence:
these are never a project figure; a project question gets its rate from the
project's own documents or asks for it.
"""
from __future__ import annotations

import math
import re
from typing import Any

from app.agents.base.formulas.arithmetic import BARE_NUMBER
from app.lib.formula_registry import Param, formula

_BENCHMARK_PROVENANCE = (
    "benchmark, unverified, owner-supplied planner template "
    "(GCC residential tower, finishing trades)"
)

#: The planner template's rows live in the shipped knowledge base, one entry
#: per work item (``productivity.<work item>``), each with its unit, crew,
#: rate, flag and the exclusions the template states, if any. That is the one copy of the
#: figures; the catalogue offers each row as its own entry. This tuple names
#: the rows the planner template holds, in the template's order.
ROW_IDS = (
    "productivity.blockwork_walls",
    "productivity.internal_wall_plaster",
    "productivity.ceiling_plaster",
    "productivity.wall_paint_primer_and_stucco",
    "productivity.emulsion_paint_walls_final_coat",
    "productivity.acrylic_emulsion_ceilings",
    "productivity.ceramic_tiles_floors",
    "productivity.ceramic_tiles_walls",
    "productivity.ceiling_finishes_false_ceiling",
    "productivity.wall_finishes_common_areas",
    "productivity.floor_finishes_common_areas",
    "productivity.epoxy_paint_floors",
    "productivity.epoxy_paint_ceilings",
    "productivity.ceiling_painting_common_areas",
)


def template_rows() -> list:
    """The planner template's rows as ``(work item, unit, crew, rate per
    crew-day, flag note, not covered)``, read from the shipped knowledge
    base. A named row the knowledge base does not hold raises: the table is
    never served with a row silently missing."""
    from app.blocks import _knowledge

    out = []
    for rid in ROW_IDS:
        entry = _knowledge.shipped_entry(rid)
        if entry is None:
            raise LookupError(f"planner template row {rid} is not in the knowledge base")
        t = entry["thresholds"]
        out.append((t["work_item"], t["unit"], t["crew"], float(t["rate_per_crew_day"]),
                    t.get("flag") or "", tuple(t.get("not_covered") or ())))
    return out


#: What a row reports when its source states no exclusions.
_EXCLUSIONS_NOT_STATED = "not stated by the planner template"


def _normalise_work_item(text: str) -> str:
    """Canonical lookup form of a work item.

    Case, punctuation, hyphens, parentheses and whitespace do not
    distinguish one name from another, so "Blockwork (walls)" and
    "blockwork walls" reach the same row. Applied to both the stored keys
    and the lookup key; crew strings and flag notes stay as written.
    """
    s = str(text or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


_BENCHMARK_KEYS = {
    _normalise_work_item(item): (unit, crew, rate, flag)
    for item, unit, crew, rate, flag, _not_covered in template_rows()
}


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
    key = _normalise_work_item(work_item)
    if key not in _BENCHMARK_KEYS:
        return {
            "work_item": work_item,
            "rate_per_crew_day": None,
            "provenance": _BENCHMARK_PROVENANCE,
            "tier": "GENERAL",
            "confidence": "MEDIUM",
            "flag": "no benchmark entry for this work item; use the project's own rate",
        }
    unit, crew, rate, flag = _BENCHMARK_KEYS[key]
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


def _same_unit(a: Any, b: Any) -> bool:
    """Two unit words name the same unit, as the platform's unit converter
    reads them: the same dimension at the same scale (m², M2, sqm and
    m2 are one unit; ft2 is an area but not this one). A word the converter
    does not know, or a value that is not text, is not a unit."""
    from app.agents.base.formulas.construction_formulas_planning import parse_unit

    ua, ub = parse_unit(a), parse_unit(b)
    if ua is None or ub is None:
        return False
    return (ua.dimension, ua.scale, ua.offset) == (ub.dimension, ub.scale, ub.offset)


@formula(
    owner="planning",
    display_name="Production duration from a planner template row",
    description=(
        "Crew-days to do a quantity of one finishing-trade work item at the "
        "planner template's rate: duration = quantity / rate per crew-day. "
        "The work item is one template row, named exactly as the row is "
        "(see the productivity.* knowledge entries for each row's unit and "
        "crew). The quantity is a bare number and its unit goes in "
        "quantity_unit; a quantity written with a unit of its own is refused "
        "and asked for again. The quantity's unit must be the row's "
        "unit; a quantity in another unit is refused, naming both. Unverified "
        "owner-supplied template: never a project figure."
    ),
    inputs={
        "work_item": Param("-", label="planner template row (work item, as the row names it)"),
        # A bare number: its unit is read from quantity_unit only, so the
        # row-unit check below reads the unit the figure is in. A unit
        # written in the value itself is refused at the front door, not
        # dropped (the same declaration the arithmetic step's values use).
        "quantity": Param(BARE_NUMBER, lo=0.0, hi=1e12, above_lo=True,
                          label="quantity of the work item"),
        "quantity_unit": Param("-", label="unit the quantity is measured in"),
    },
    outputs={
        "duration_crew_days": "d",
        "work_item": "-",
        "unit": "-",
        "crew": "-",
        "rate_per_crew_day": "count/d",
        "not_covered": "-",
        "provenance": "-",
    },
    main_output="duration_crew_days",
)
def crew_days_from_rate_row(work_item: str, quantity: float, quantity_unit: str) -> dict:
    """Duration of one template row's work: the row the caller names, its
    unit checked against the quantity's. No row is guessed from a
    description: a name that is not a row is refused with the rows listed."""
    if not isinstance(work_item, str):
        raise ValueError(f"the work item is the name of one planner template row, not {work_item!r}")
    if not math.isfinite(float(quantity)):
        raise ValueError(f"the quantity of the work item is not a finite number ({quantity!r})")
    key = _normalise_work_item(work_item)
    if key not in _BENCHMARK_KEYS:
        rows = ", ".join(item for item, *_ in template_rows())
        raise ValueError(f"{work_item!r} is not a planner template row; the rows are: {rows}")
    unit, crew, rate, _flag = _BENCHMARK_KEYS[key]
    if not _same_unit(quantity_unit, unit):
        raise ValueError(
            f"row {key!r} is measured in {unit}; the quantity is in "
            f"{quantity_unit or 'no unit'}, so this row does not apply to it")
    not_covered = next(nc for item, _u, _c, _r, _f, nc in template_rows()
                       if _normalise_work_item(item) == key)
    return {
        "duration_crew_days": float(quantity) / rate,
        "work_item": key,
        "unit": unit,
        "crew": crew,
        "rate_per_crew_day": rate,
        # Only exclusions the source states; when it states none, that is
        # said, never an empty list a reader could take as "covers all".
        "not_covered": list(not_covered) or _EXCLUSIONS_NOT_STATED,
        "provenance": _BENCHMARK_PROVENANCE,
    }
