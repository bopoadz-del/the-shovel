"""The ONE production-rate table (D1) — kernel DATA (S-1b; ruling ORDERS_28OCT_1045Z).

Every output rate the product holds lives in this module, and every reader
reads it here: the schedule path (``norm_rate_for``, called by the brief and
BOQ schedule builders) and the planning-hat ``productivity_benchmark``
calculator (``benchmark_row``). Two tiers, never mixed silently:

  * ``_NORMS`` — PUBLISHED rows, each with a real ``citation`` (publication,
    edition, table/item). Empty until the publication is transcribed.
  * ``_TEMPLATE_RATES`` — the owner-supplied planner template (units per crew
    on a 10-hour day). UNVERIFIED: every rate read from it carries that
    provenance into the activity's duration basis, so a schedule says which
    durations rest on the template. A published row for the same trade and
    unit always wins over a template row.

The published tier follows the 1045Z ruling:

  * REAL CITATION PER ROW. A row stays only if it cites a real published norm
    — a ``citation`` naming the publication, edition and table/item. A
    self-declared "generic planning norm" is NOT a citation. A row without one
    is removed. This is the same bar the Store ST-3 rows were held to.
  * EMPTY UNTIL SOURCED. No verifiable published figures are shipped in code,
    so ``_NORMS`` is EMPTY. The table is meant to be populated as a Store DATA
    item whose rows each carry provenance — never hand-written defaults.

Both tiers:

  * UNIT PER ROW, CHECKED. Each row carries the unit its rate is measured in;
    a quantity whose unit does not match the row's unit yields no rate (a m2
    rate is never applied to a m3 quantity); a quantity with no unit is a
    mismatch, not a free pass.
  * WORD-BOUNDARY TRADE MATCH. Every word of a keyword must be a whole word of
    the activity name, so "concrete" does not match inside an unrelated word.
    The most specific (most-word) matching keyword wins.

Published row schema: ``{"keywords": [str, ...], "rate": "<n> <unit> per
gang-day", "unit": "<unit>", "citation": "<publication, edition, table/item>"}``.
The ``rate`` text is the exact form the kernel ``productivity_manpower_duration``
parses; paired with a matching-unit quantity it yields a duration in days.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

# EMPTY until rows with real published provenance exist (1045Z ruling). Do NOT
# add a row without a real ``citation`` and a ``unit`` — an uncited row is a
# self-declaration and is rejected. Populate this as a sourced Store data item.
_NORMS: List[Dict[str, object]] = []

TEMPLATE_PROVENANCE = (
    "benchmark, unverified, owner-supplied planner template "
    "(GCC residential tower, finishing trades)"
)
TEMPLATE_SHIFT_HOURS = 10

# The owner-supplied planner template (SPEC_09NOV_PROJECT_CONTROL_FORMULAS_AND_
# RATES, section B), moved here from the planning hat so the schedule path and
# the calculator read ONE copy. Rate = units per crew-day on a 10-hour day.
# ``keywords`` are the word sets a schedule activity name must contain for the
# row to apply (each word on a word boundary, any order, plural/-ing allowed).
_TEMPLATE_RATES: List[Dict[str, object]] = [
    {"work_item": "blockwork walls", "unit": "m2", "crew": "mason + helper (1:1)", "rate": 12.0,
     "flag": "", "keywords": ["blockwork", "block work", "block wall"]},
    {"work_item": "internal wall plaster", "unit": "m2", "crew": "mason + helper", "rate": 20.0,
     "flag": "", "keywords": ["wall plaster"]},
    {"work_item": "ceiling plaster", "unit": "m2", "crew": "mason", "rate": 20.0,
     "flag": "", "keywords": ["ceiling plaster"]},
    {"work_item": "wall paint primer and stucco", "unit": "m2", "crew": "painter", "rate": 45.0,
     "flag": "painting rate at the low end of the usual GCC range", "keywords": ["primer", "stucco"]},
    {"work_item": "emulsion paint walls final coat", "unit": "m2", "crew": "painter", "rate": 45.0,
     "flag": "painting rate at the low end of the usual GCC range", "keywords": ["emulsion wall"]},
    {"work_item": "acrylic emulsion ceilings", "unit": "m2", "crew": "painter", "rate": 45.0,
     "flag": "painting rate at the low end of the usual GCC range", "keywords": ["emulsion ceiling"]},
    {"work_item": "ceramic tiles floors", "unit": "m2", "crew": "mason (tiler)", "rate": 40.0,
     "flag": "", "keywords": ["ceramic floor", "floor tile", "floor tiling"]},
    {"work_item": "ceramic tiles walls", "unit": "m2", "crew": "mason (tiler)", "rate": 40.0,
     "flag": "wall tiling at 40 is optimistic vs floor; flag",
     "keywords": ["ceramic wall", "wall tile", "wall tiling"]},
    {"work_item": "ceiling finishes false ceiling", "unit": "m2", "crew": "mason", "rate": 40.0,
     "flag": "", "keywords": ["false ceiling"]},
    {"work_item": "wall finishes common areas", "unit": "m2", "crew": "mason", "rate": 40.0,
     "flag": "", "keywords": ["wall finish"]},
    {"work_item": "floor finishes common areas", "unit": "m2", "crew": "mason", "rate": 40.0,
     "flag": "", "keywords": ["floor finish"]},
    {"work_item": "epoxy paint floors", "unit": "m2", "crew": "painter", "rate": 40.0,
     "flag": "painting rate at the low end of the usual GCC range", "keywords": ["epoxy floor"]},
    {"work_item": "epoxy paint ceilings", "unit": "m2", "crew": "painter", "rate": 40.0,
     "flag": "painting rate at the low end of the usual GCC range", "keywords": ["epoxy ceiling"]},
    {"work_item": "ceiling painting common areas", "unit": "m2", "crew": "painter", "rate": 40.0,
     "flag": "painting rate at the low end of the usual GCC range",
     "keywords": ["ceiling paint", "ceiling painting"]},
]


def _norm_unit(unit: Optional[str]) -> str:
    """Normalise a unit token for comparison (m²->m2, sqm->m2, blank-safe)."""
    u = (unit or "").strip().lower().replace("²", "2").replace("³", "3")
    u = u.replace(" ", "")
    return {"sqm": "m2", "cubicm": "m3", "cum": "m3", "number": "nr", "no": "nr", "no.": "nr", "nos": "nr"}.get(u, u)


def _row_is_cited(row: Dict[str, object]) -> bool:
    """A row may ship only with a real citation AND a unit (1045Z)."""
    return bool(str(row.get("citation") or "").strip()) and bool(str(row.get("unit") or "").strip())


def normalise_work_item(text: str) -> str:
    """Canonical lookup form of a work item: case, punctuation, hyphens,
    parentheses and whitespace do not distinguish one name from another."""
    t = str(text or "").lower()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return " ".join(t.split())


def benchmark_row(work_item: str) -> Optional[Dict[str, object]]:
    """The template row whose work item IS ``work_item`` (normalised), or None."""
    key = normalise_work_item(work_item)
    for row in _TEMPLATE_RATES:
        if normalise_work_item(str(row["work_item"])) == key:
            return row
    return None


def _keyword_words_match(keyword: str, name: str) -> bool:
    """Every word of ``keyword`` is a whole word of ``name`` (any order). A
    name word may carry a plural or -ing ending the keyword word lacks."""
    words = keyword.lower().split()
    if not words:
        return False
    return all(
        re.search(r"\b" + re.escape(w) + r"(?:s|es|ing)?\b", name) for w in words
    )


def _best_match(rows: List[Dict[str, object]], name: str, want_unit: str) -> Optional[Dict[str, object]]:
    """The unit-matching row whose matched keyword has the most words (the
    most specific trade wins); ties go to the earlier row. None if no row."""
    best: Optional[Dict[str, object]] = None
    best_len = 0
    for row in rows:
        row_unit = _norm_unit(str(row.get("unit") or ""))
        # The quantity's unit must be PRESENT and match the row's (1045Z/1130Z):
        # an absent or unverified quantity unit is a mismatch, not a free pass.
        if row_unit and want_unit != row_unit:
            continue
        for kw in row.get("keywords", []):  # type: ignore[union-attr]
            kw_l = str(kw).lower().strip()
            n = len(kw_l.split())
            if n > best_len and _keyword_words_match(kw_l, name):
                best, best_len = row, n
    return best


def norm_rate_for(activity_name: str, quantity_unit: Optional[str] = None) -> Optional[Dict[str, str]]:
    """Return ``{"rate", "unit", "source", "tier"}`` for the production rate
    whose trade matches ``activity_name`` AND whose unit ``quantity_unit``
    matches, or ``None`` when no row applies. A published (cited) row wins;
    only when none applies is the unverified template read, and its
    ``source`` says so. An uncited published-tier row is never used.
    """
    name = (activity_name or "").lower().strip()
    if not name:
        return None
    want_unit = _norm_unit(quantity_unit)
    cited = _best_match([r for r in _NORMS if _row_is_cited(r)], name, want_unit)
    if cited is not None:
        return {
            "rate": str(cited["rate"]),
            "unit": str(cited.get("unit") or ""),
            "source": str(cited["citation"]),
            "tier": "published",
        }
    row = _best_match(_TEMPLATE_RATES, name, want_unit)
    if row is None:
        return None
    unit = str(row["unit"])
    return {
        "rate": f"{float(row['rate']):g} {unit} per gang-day",  # type: ignore[arg-type]
        "unit": unit,
        "source": (
            f"{TEMPLATE_PROVENANCE}; {row['work_item']}, crew {row['crew']}, "
            f"{TEMPLATE_SHIFT_HOURS}-hour crew-day"
        ),
        "tier": "unverified template",
    }
