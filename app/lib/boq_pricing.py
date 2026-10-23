"""Price an UNPRICED Bill of Quantities from the typed rate-card.

This is the pure, FastAPI-free core of the platform's "price an unpriced BOQ"
capability. Given a list of boq_processor line items (description / quantity /
unit, with NO usable rate because the source BOQ is unpriced), it classifies
each line, normalises its unit, and looks up a median rate from the deployed
rate-card (``app/lib/data/boq_rate_card.json``):

    exact (work_category + unit)
      -> category median (any unit for that category)
        -> asset median (any category, any unit)
          -> NO RATE (flagged, never invented)

``categorize`` and ``norm_unit`` are ported faithfully from the validated
prototype ``scripts/price_boq.py`` so this endpoint prices identically to it.
Every priced line records its ``rate_source`` and the sample size ``n`` behind
the rate, so confidence stays visible -- and lines with no comparable rate are
kept and flagged (rate 0, ``rate_source == "NO RATE"``), never dropped and
never assigned an invented number.
"""
from __future__ import annotations

import json
import os
import re
import statistics as st
from typing import Any

# ── rate-card data (loaded once, path relative to this module) ───────────────
_CARD_PATH = os.path.join(os.path.dirname(__file__), "data", "boq_rate_card.json")


def _load_card() -> dict[str, Any]:
    if not os.path.exists(_CARD_PATH):
        raise RuntimeError(
            f"Rate-card data not found at {_CARD_PATH}. The priced-BOQ feature "
            "requires app/lib/data/boq_rate_card.json to be deployed."
        )
    try:
        with open(_CARD_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:  # noqa: BLE001 - surface a clear, actionable error
        raise RuntimeError(f"Rate-card data at {_CARD_PATH} is unreadable: {exc}")
    cards = data.get("cards")
    if not isinstance(cards, dict) or not cards:
        raise RuntimeError(
            f"Rate-card data at {_CARD_PATH} has no 'cards' section."
        )
    return data


_CARD = _load_card()


# ── categorisation + unit normalisation (ported from scripts/price_boq.py) ───
CATS = [
    ("Earthworks/Excavation", r"excavat|earthwork|backfill|\bfill\b|grading|clearing|disposal|dewater|trench"),
    ("Demolition", r"demolit|dismantl|\bbreak|removal|salvage|strip"),
    ("Piling/Foundations", r"pil(e|ing)|bored|caisson|footing|foundation"),
    ("Concrete", r"concrete(?!\s*block)|\brcc\b|blinding|screed|\bslab|grade \d|c\d{2}\b|pour"),
    ("Reinforcement", r"reinforc(?!ed\s+concrete)|rebar|steel bar|\bmesh|bending"),
    ("Formwork", r"formwork|shutter|falsework"),
    ("Structural Steel", r"structural steel|steelwork|\bfabricat|purlin|truss|bracing"),
    ("Masonry/Blockwork", r"block|brick|masonry|thermal ?stone"),
    ("Pipework/Drainage", r"\bpipe|sewer|foul|drainage|manhole|culvert|gully|gravity|rising main|\bgrp\b|vitrified"),
    ("Roads/Paving", r"asphalt|paving|\broad|kerb|pavement|interlock|sub-?base|road base"),
    ("Windows/Doors/Facade", r"window|\bdoor|aluminum|aluminium|glazing|glass|curtain wall|facade|cladding|shopfront"),
    ("Finishes", r"plaster|render|paint|\btile|floor(ing)?|ceiling|gypsum|skirting|coving|wallpaper|screed"),
    ("Waterproofing/Insulation", r"waterproof|membrane|insulat|damp proof|tanking"),
    ("Electrical (MEP)", r"cable|switchgear|transformer|busduct|electric|lighting|\bpanel|\bmv\b|\bhv\b|\blv\b|earthing|containment"),
    ("Mechanical/HVAC (MEP)", r"chiller|\bpump|hvac|cooling|\bcrah|\bcdu\b|ducting|\bahu\b|ventilat|refrigerant|mechanical"),
    ("Fire Protection", r"fire (extinguish|blanket|protec|detec|alarm|fighting)|sprinkler|\bfe-\d"),
    ("Sanitary/Accessories", r"toilet|sanitary|\bwc\b|mirror|towel|holder|\bbin\b|basin|accessor|\btray\b"),
    ("Landscape/Softscape", r"\bplant|\btree|palm|softscape|landscape|irrigat|shrub|turf|topsoil"),
    ("Preliminaries/General", r"prelim|mobiliz|insurance|provisional|general item|attendance|supervision|overhead"),
]


def norm_unit(u: Any) -> str:
    """Canonicalise a unit token (handles Italian mq/mc/ml, sqm, cum, nos, etc)."""
    u = str(u).strip().lower().replace("²", "2").replace("³", "3").replace(" ", "")
    m = {"sqm": "m2", "sq.m": "m2", "m^2": "m2", "mq": "m2", "mq.": "m2", "mc": "m3",
         "cum": "m3", "cu.m": "m3", "m^3": "m3",
         "l.m": "m", "lm": "m", "ml": "m", "ml.": "m", "rm": "m", "r.m": "m", "lin.m": "m", "linearmeter": "m",
         "nos": "no", "nr": "no", "no.": "no", "each": "no", "ea": "no", "pcs": "no", "pce": "no", "pc": "no",
         "l.s": "ls", "lumpsum": "ls", "sum": "ls", "item": "ls", "lot": "ls"}
    return m.get(u, u) or "?"


_CATS_RX = [(name, re.compile(pat)) for name, pat in CATS]


def categorize(desc: Any) -> str:
    """Classify a line-item description into one of the ~19 work categories.

    The category whose pattern matches EARLIEST wins, not the first one in list
    order. A BOQ line names its work first and its substrate second -- "plaster
    render to blockwork walls", "waterproofing membrane to wet area floors" --
    so first-in-list matching let a trailing substrate keyword decide the trade
    and price the line as the wrong one. Measured 2026-08-15: plaster priced as
    blockwork (88/m2) and waterproofing priced as finishes (38/m2).

    List order remains the tie-break, so two categories matching at the same
    offset resolve exactly as they did before.
    """
    d = str(desc).lower()
    best: tuple[tuple[int, int], str] | None = None
    for idx, (name, rx) in enumerate(_CATS_RX):
        m = rx.search(d)
        if m is None:
            continue
        key = (m.start(), idx)
        if best is None or key < best[0]:
            best = (key, name)
    return best[1] if best else "Other/Uncategorized"


def _num(x: Any) -> float | None:
    """Coerce a possibly-dirty numeric cell to float, or None."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x) if x == x else None  # reject NaN
    s = re.sub(r"[,\sA-Za-z$]", "", str(x))
    try:
        v = float(s)
        return v if v == v else None
    except Exception:
        return None


# ── rate-card lookup (ported logic: exact -> category -> asset -> NO RATE) ────
def available_assets() -> dict[str, list[str]]:
    """Return ``asset_type -> [currency, ...]`` for every card in the rate-card,
    so the endpoint can validate requests and report the valid options."""
    return {asset: list(ccys.keys()) for asset, ccys in _CARD["cards"].items()}


def _build_lookup(asset: str, ccy: str) -> tuple[dict[tuple[str, str], tuple[float, int]],
                                                  dict[str, list[tuple[float, int, str]]],
                                                  list[tuple[float, int, str]]]:
    """Build (exact, by-category, by-asset) rate structures for (asset, ccy).

    Returns empty structures when the asset/currency is not in the card -- the
    caller then flags every line NO RATE rather than raising, so an unknown
    combination degrades gracefully (endpoint-level validation is separate)."""
    rows = _CARD["cards"].get(asset, {}).get(ccy, [])
    exact: dict[tuple[str, str], tuple[float, int, str | None]] = {}
    bycat: dict[str, list[tuple[float, int, str]]] = {}
    byasset: list[tuple[float, int, str]] = []
    for r in rows:
        median = float(r["median"])
        n = int(r["n"])
        # cell-level provenance: observed cells have no source field; cells
        # transcribed from published references carry it, and the label
        # surfaces all the way into the priced workbook's Rate Basis column.
        exact[(r["cat"], r["unit"])] = (median, n, r.get("source"))
        bycat.setdefault(r["cat"], []).append((median, n, r["unit"]))
        byasset.append((median, n, r["unit"]))
    return exact, bycat, byasset


# Unit commensurability (F10, phase-3 campaign). A fallback median is only
# meaningful when it is built from rates measured the same WAY as the line
# being priced: an AED 37,250 lump-sum "plumbing works" rate must never price
# a per-metre pipe run just because both classify as Pipework/Drainage. Units
# outside this map (%, months, '?') have no class -- lines carrying them keep
# the legacy any-unit pools, because commensurability cannot be judged.
_UNIT_CLASS: dict[str, str] = {
    "m": "linear",
    "m2": "area",
    "m3": "volume",
    "kg": "mass", "t": "mass", "mt": "mass", "ton": "mass", "tonne": "mass",
    "no": "count", "set": "count", "pair": "count", "bay": "count",
    "ls": "lump",
}


def _unit_class(unit: str) -> str | None:
    return _UNIT_CLASS.get(unit)


def _lookup(cat, unit, exact, bycat, byasset):
    """Resolve a rate for (category, unit). Returns (rate, n, rate_source).

    rate_source is one of: 'exact (cat+unit)', 'fallback (category median)',
    'weak (asset median)', 'NO RATE'. NO RATE yields rate None -- never an
    invented number.

    Fallback tiers are unit-commensurable: the category and asset medians are
    computed only over rates whose unit belongs to the same class (linear /
    area / volume / mass / count / lump) as the line being priced. A tier with
    no commensurable rates is skipped -- NO RATE is more honest than a number
    with the wrong dimensions."""
    if (cat, unit) in exact:
        m, n, src = exact[(cat, unit)]
        if src:
            return m, n, f"published reference ({src})"
        return m, n, "exact (cat+unit)"
    cls = _unit_class(unit)

    def _pool(vals):
        if cls is None:
            return vals
        return [v for v in vals if _unit_class(v[2]) == cls]

    pool = _pool(bycat.get(cat, []))
    if pool:
        m = st.median([v for v, _, _ in pool])
        n = sum(n for _, n, _ in pool)
        return round(m, 2), n, "fallback (category median)"
    pool = _pool(byasset)
    if pool:
        m = st.median([v for v, _, _ in pool])
        return round(m, 2), sum(n for _, n, _ in pool), "weak (asset median)"
    return None, 0, "NO RATE"


def price_line_items(
    line_items: list[dict[str, Any]],
    asset_type: str,
    currency: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Price a list of UNPRICED boq_processor line items from the rate-card.

    ``line_items``: boq_processor output dicts (keys: description, quantity,
    unit; item_key/section optional). The source rate is 0/blank because the
    BOQ is unpriced -- any source rate is DISCARDED and re-derived from the card.

    Returns ``(priced_items, summary)``:
      - ``priced_items``: [{item_no, description, unit, qty, rate, rate_source,
        n, work_category, section}]. Lines with no comparable rate are kept with
        rate 0 and rate_source 'NO RATE' (flagged, never dropped, never invented).
      - ``summary``: {total, exact, fallback, no_rate, grand_total, asset_type,
        currency}. ``grand_total`` is the estimated qty*rate sum over priced lines.

    This never raises on an unknown/empty (asset_type, currency): the lookup is
    empty and every line comes back NO RATE. Request-level validation of the
    asset/currency lives in the endpoint (via ``available_assets``)."""
    exact, bycat, byasset = _build_lookup(asset_type, currency)

    priced: list[dict[str, Any]] = []
    n_exact = n_fallback = n_norate = 0
    grand_total = 0.0

    for i, it in enumerate(line_items, 1):
        desc = it.get("description") or ""
        qty = _num(it.get("quantity"))
        if qty is None:
            qty = 0.0
        unit = norm_unit(it.get("unit")) if it.get("unit") else "?"
        cat = categorize(desc)
        rate, n, source = _lookup(cat, unit, exact, bycat, byasset)

        if rate is None:
            n_norate += 1
            item_rate = 0.0
            rate_source = "NO RATE"
            n = 0
        else:
            item_rate = rate
            rate_source = source
            if source.startswith("exact"):
                n_exact += 1
            else:
                n_fallback += 1
            grand_total += qty * item_rate

        priced.append({
            "item_no": it.get("item_key") or str(i),
            "description": desc,
            "unit": it.get("unit") or unit,
            "qty": qty,
            "rate": round(item_rate, 2),
            "rate_source": rate_source,
            "n": n,
            "work_category": cat,
            "section": it.get("section") or "",
        })

    summary = {
        "total": len(priced),
        "exact": n_exact,
        "fallback": n_fallback,
        "no_rate": n_norate,
        "grand_total": round(grand_total, 2),
        "asset_type": asset_type,
        "currency": currency,
    }
    return priced, summary
