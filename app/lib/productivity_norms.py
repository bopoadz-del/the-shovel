"""Sourced productivity-norm table — kernel DATA (S-1b; ruling ORDERS_28OCT_1045Z).

When a brief/BOQ states a *quantity* for an activity but no productivity
*rate*, the schedule estimator may read a per-trade productivity norm from this
table so the activity gets a kernel-computed duration instead of UNKNOWN. The
1045Z ruling is strict about what may live here:

  * REAL CITATION PER ROW. A row stays only if it cites a real published norm
    — a ``citation`` naming the publication, edition and table/item. A
    self-declared "generic planning norm" is NOT a citation. A row without one
    is removed. This is the same bar the Store ST-3 rows were held to.
  * EMPTY UNTIL SOURCED. No verifiable published figures are shipped in code,
    so ``_NORMS`` is EMPTY. ``norm_rate_for`` therefore returns ``None`` and
    the schedule path yields UNKNOWN (the honest 0845Z fallback) until sourced
    rows exist. The table is meant to be populated as a Store DATA item whose
    rows each carry provenance — never hand-written, self-declared defaults.
  * UNIT PER ROW, CHECKED. Each row carries the unit its rate is measured in;
    a quantity whose unit does not match the row's unit yields UNKNOWN (a m2
    norm is never applied to a m3 quantity).
  * WORD-BOUNDARY TRADE MATCH. Keywords match on word boundaries, so "concrete"
    does not match inside an unrelated word.

Row schema (when populated): ``{"keywords": [str, ...], "rate": "<n> <unit> per
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


def _norm_unit(unit: Optional[str]) -> str:
    """Normalise a unit token for comparison (m²->m2, sqm->m2, blank-safe)."""
    u = (unit or "").strip().lower().replace("²", "2").replace("³", "3")
    u = u.replace(" ", "")
    return {"sqm": "m2", "cubicm": "m3", "cum": "m3", "number": "nr", "no": "nr", "no.": "nr", "nos": "nr"}.get(u, u)


def _row_is_cited(row: Dict[str, object]) -> bool:
    """A row may ship only with a real citation AND a unit (1045Z)."""
    return bool(str(row.get("citation") or "").strip()) and bool(str(row.get("unit") or "").strip())


def norm_rate_for(activity_name: str, quantity_unit: Optional[str] = None) -> Optional[Dict[str, str]]:
    """Return ``{"rate", "unit", "source"}`` for the sourced productivity norm
    whose trade matches ``activity_name`` AND whose unit matches
    ``quantity_unit`` (when given), or ``None`` when no cited, unit-matching row
    applies. With an empty ``_NORMS`` it always returns ``None`` — the caller
    then leaves the activity UNKNOWN. Matching is word-boundary; a row with no
    real citation is never used even if present.
    """
    name = (activity_name or "").lower().strip()
    if not name:
        return None
    want_unit = _norm_unit(quantity_unit)
    for row in _NORMS:
        if not _row_is_cited(row):
            continue  # uncited rows never apply
        row_unit = _norm_unit(str(row.get("unit") or ""))
        if want_unit and row_unit and want_unit != row_unit:
            continue  # unit mismatch -> not applicable (caller -> UNKNOWN)
        for kw in row.get("keywords", []):  # type: ignore[union-attr]
            kw_l = str(kw).lower().strip()
            if kw_l and re.search(r"\b" + re.escape(kw_l) + r"\b", name):
                return {
                    "rate": str(row["rate"]),
                    "unit": str(row.get("unit") or ""),
                    "source": str(row["citation"]),
                }
    return None
