"""Derive a construction schedule from a priced BOQ — quantities become man-hours.

Why this exists
---------------
``/export/schedule-from-document`` accepted a BOQ and produced a schedule whose
activities came from a canned per-project-type template: measured 2026-08-17,
a BOQ listing structural steel truss, composite cladding, chilled-water piping
and epoxy flooring yielded 46 activities of which ZERO referenced that scope.
The document route mines equipment lead times and target milestones; a BOQ
carries neither, so it contributed nothing.

A BOQ is the one document that states the actual scope WITH quantities, which
is exactly what a duration needs.

Man-hours are the unit
----------------------
Planning is done in MAN-HOURS, not "crew-days" (operator correction,
2026-08-17: "everything is in manhour"). A norm of 0.8 mh/m2 is a property of
the work; a crew-day figure silently bundles crew size and shift length into
one number, so it cannot be compared between projects, cannot be re-planned
when the crew changes, and cannot produce an honest manpower histogram. So:

    total_manhours  = quantity x manhours_per_unit
    duration_days   = ceil(total_manhours / (crew_size x hours_per_day))

The histogram and S-curve then fall out of the same man-hours rather than
being derived separately — the programme and the resource curve agree by
construction, which is the whole point of a cost/resource-loaded schedule.

Process, not facts
------------------
Man-hour norms are PROJECT variables — they move with crew, site, height,
access and repetition — so nothing here carries a built-in rate. They are
supplied by the caller, sourced in order: the project's own records, the
project facts store, a cited published reference, or the operator. A category
with no supplied norm is REFUSED by name, because a programme resting on an
invented output looks exactly like a real one until it slips.

Output shape matches ``generate_wbs`` activities exactly (id/name/duration_days/
predecessors/resources/wbs_phase), so the existing CPM, cost-loading bridge and
Excel writers consume it unchanged.
"""

from __future__ import annotations

import logging
import math
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.lib.boq_pricing import categorize

logger = logging.getLogger(__name__)

# Standard shift. Overridable per call — some projects run 10s, Ramadan runs 6.
DEFAULT_HOURS_PER_DAY = 8.0
DEFAULT_CREW_SIZE = 4

# Construction order for the categorizer's work packages. A schedule is a
# sequence claim, and this is the only place that claim is made.
CONSTRUCTION_SEQUENCE: Tuple[str, ...] = (
    "Preliminaries/General",
    "Demolition",
    "Earthworks/Excavation",
    "Piling/Foundations",
    "Formwork",
    "Reinforcement",
    "Concrete",
    "Structural Steel",
    "Masonry/Blockwork",
    "Waterproofing/Insulation",
    "Windows/Doors/Facade",
    "Mechanical/HVAC (MEP)",
    "Electrical (MEP)",
    "Fire Protection",
    "Pipework/Drainage",
    "Sanitary/Accessories",
    "Finishes",
    "Roads/Paving",
    "Landscape/Softscape",
    "Other/Uncategorized",
)

_TRADE_OF = {
    "Earthworks/Excavation": "earthworks",
    "Demolition": "demolition",
    "Piling/Foundations": "piling",
    "Concrete": "concrete",
    "Reinforcement": "steelfixing",
    "Formwork": "carpentry",
    "Structural Steel": "steel_erection",
    "Masonry/Blockwork": "masonry",
    "Pipework/Drainage": "plumbing",
    "Roads/Paving": "paving",
    "Windows/Doors/Facade": "facade",
    "Finishes": "finishes",
    "Waterproofing/Insulation": "waterproofing",
    "Electrical (MEP)": "electrical",
    "Mechanical/HVAC (MEP)": "hvac",
    "Fire Protection": "fire",
    "Sanitary/Accessories": "sanitary",
    "Landscape/Softscape": "landscaping",
    "Preliminaries/General": "management",
    "Other/Uncategorized": "general",
}

# Substructure markers. The pricing categorizer answers "what trade is this?",
# never "which part of the building?" — "Reinforced concrete C40 to raft
# foundation" and "...to columns" are both plain Concrete. Scheduled as one
# package in BOQ order, a raft listed second lands AFTER the columns it
# carries: a wrong programme, not a cosmetic one.
#
# Plurals are the norm in a real BOQ ("to foundations", "pile caps"), and a
# \b-anchored singular silently fails on every one of them.
_SUBSTRUCTURE_RX = re.compile(
    r"\b(foundations?|footings?|pile[\s-]?caps?|piling|rafts?|blinding|"
    r"substructure|ground\s+beams?|pad\s+bases?|tie\s+beams?|lean\s+concrete|"
    r"underground|below\s+ground)\b",
    re.IGNORECASE,
)

# Trades that recur between substructure and superstructure and so need
# splitting; finishes and facade have no substructure phase.
_STAGED_CATEGORIES = frozenset({
    "Concrete", "Reinforcement", "Formwork", "Waterproofing/Insulation",
    "Masonry/Blockwork",
})

# Categories that ARE substructure/enabling by definition — no keyword needed.
# Live 2026-08-17: without Earthworks here, "Bulk excavation to reduced level"
# sorted into the superstructure block and scheduled on day 32, AFTER the
# foundations it digs for. Demolition and site clearance are the same class:
# they precede everything, always.
_ALWAYS_SUBSTRUCTURE = frozenset({
    "Piling/Foundations", "Earthworks/Excavation", "Demolition",
})

# Enabling operations that precede the reinforcement they are poured under.
# Blinding is categorised as Concrete (it is), but a blinding layer is struck
# before rebar is fixed, not after it — trade order alone puts it last.
_ENABLING_RX = re.compile(r"(blinding|lean\s+concrete|sub[\s-]?base|"
                          r"levelling\s+course)", re.IGNORECASE)

# Vertical/applied-after work inside the substructure: tanking to retaining
# walls, protection boards to raft sides. These follow the pour, unlike the
# horizontal membrane below it.
_AFTER_POUR_RX = re.compile(
    r"(retaining\s+wall|to\s+walls?\b|wall\s+face|vertical|"
    r"tanking\s+to\s+(?:walls?|sides?)|sides?\s+of|external\s+face|"
    r"protection\s+board)",
    re.IGNORECASE,
)

SUBSTRUCTURE = "substructure"
SUPERSTRUCTURE = "superstructure"


def element_stage(description: Any, category: str) -> str:
    """``substructure`` for below-ground work, else ``superstructure``."""
    if category in _ALWAYS_SUBSTRUCTURE:
        return SUBSTRUCTURE
    if category in _STAGED_CATEGORIES and _SUBSTRUCTURE_RX.search(str(description)):
        return SUBSTRUCTURE
    return SUPERSTRUCTURE


class MissingProductivity(Exception):
    """No man-hour norm was supplied for a work category present in the BOQ.

    Carries the categories and their units so the caller asks a precise
    question ("what is your blockwork norm, in man-hours per m2?") instead of
    reporting a generic failure.
    """

    def __init__(self, missing: Dict[str, str]) -> None:
        self.missing = missing
        detail = "; ".join(f"{cat} (man-hours per {unit})"
                           for cat, unit in sorted(missing.items()))
        super().__init__(
            "No man-hour norm supplied for: " + detail +
            ". Durations are derived from quantity x man-hours per unit, so "
            "these cannot be scheduled until the norms are given — supply them "
            "from the project's own records, a cited published norm, or the "
            "operator."
        )


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")[:40] or "item"


def _qty(item: Dict[str, Any]) -> float:
    for key in ("quantity", "total_quantity", "qty"):
        v = item.get(key)
        if isinstance(v, (int, float)) and v == v:
            return float(v)
        if isinstance(v, str):
            try:
                return float(re.sub(r"[,\s]", "", v))
            except ValueError:
                continue
    return 0.0


def manhours(quantity: float, manhours_per_unit: float) -> float:
    """Total man-hours for a measured quantity."""
    if manhours_per_unit <= 0:
        raise ValueError("manhours_per_unit must be > 0")
    return quantity * manhours_per_unit


def duration_days(
    total_manhours: float,
    crew_size: int = DEFAULT_CREW_SIZE,
    hours_per_day: float = DEFAULT_HOURS_PER_DAY,
) -> int:
    """Working days to burn ``total_manhours`` with ``crew_size`` on shift.

    Always at least one day: a measurable item still occupies the crew for a
    shift, and a zero-day activity makes the CPM degenerate.
    """
    if crew_size <= 0:
        raise ValueError("crew_size must be > 0")
    if hours_per_day <= 0:
        raise ValueError("hours_per_day must be > 0")
    return max(1, math.ceil(total_manhours / (crew_size * hours_per_day)))


def from_daily_output(output_per_man_day: float,
                      hours_per_day: float = DEFAULT_HOURS_PER_DAY) -> float:
    """Convert a published 'units per tradesman per day' norm to man-hours/unit.

    Estimating handbooks publish daily outputs; planning needs man-hours. This
    is the one conversion, stated once, so a cited reference can be used
    without re-deriving it at each call site.
    """
    if output_per_man_day <= 0:
        raise ValueError("output_per_man_day must be > 0")
    return hours_per_day / output_per_man_day


def group_by_category(
    line_items: Iterable[Dict[str, Any]],
    overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Bucket BOQ lines into work packages using the shipped categorizer.

    ``categorize`` is shared with pricing and matches the EARLIEST pattern in
    the description, which is right for a rate lookup but can misplace a line
    for sequencing: "Chilled water piping DN200 insulated" classifies as
    Waterproofing/Insulation on the word "insulated". Rather than fork that
    taxonomy (and quietly change how rates resolve), a planner corrects
    individual lines through ``overrides``, keyed by ``item_key`` or exact
    description.
    """
    overrides = overrides or {}
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for item in line_items or []:
        desc = item.get("description") or item.get("item_key") or ""
        cat = (overrides.get(item.get("item_key") or "")
               or overrides.get(desc)
               or categorize(desc))
        grouped.setdefault(cat, []).append(item)
    return grouped


def uncategorized(line_items: Iterable[Dict[str, Any]],
                  overrides: Optional[Dict[str, str]] = None) -> List[str]:
    """Descriptions the classifier could not place, so they stay VISIBLE.

    An unrecognised line is still scheduled (Other package, last), but a
    planner must see what the taxonomy did not understand rather than find it
    at the end of the programme.
    """
    return [
        (i.get("description") or i.get("item_key") or "")
        for i in group_by_category(line_items, overrides).get("Other/Uncategorized", [])
    ]


def _norm_desc(text: Any) -> str:
    """Collapse a description to its work identity for aggregation."""
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def aggregate_lines(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge BOQ lines describing the SAME work into one measured item.

    A real BOQ measures the same work many times — per floor, per zone, per
    block — so scheduling one activity per line is not a programme: the live
    run of a 640-line BOQ produced 640 chained activities and an 18,400-day
    (73-year) duration, because 160 repeats of "RC C40 to columns" each became
    their own sequential 6-day task. Planners schedule WORK PACKAGES: the
    quantities add up, the activity is one.

    Quantities and costs are summed; ``line_count`` records how many BOQ lines
    the activity represents, so the aggregation stays visible rather than
    looking like a shorter BOQ.
    """
    merged: Dict[str, Dict[str, Any]] = {}
    for item in items or []:
        key = _norm_desc(item.get("description") or item.get("item_key"))
        cur = merged.get(key)
        if cur is None:
            cur = dict(item)
            cur["quantity"] = _qty(item)
            cur["line_count"] = 1
            cur["total_cost"] = float(item.get("total_cost") or 0)
            merged[key] = cur
            continue
        cur["quantity"] += _qty(item)
        cur["line_count"] += 1
        cur["total_cost"] += float(item.get("total_cost") or 0)
    return list(merged.values())


def activities_from_boq(
    line_items: Iterable[Dict[str, Any]],
    *,
    manhours_per_unit: Dict[str, float],
    crew_size: Optional[Dict[str, int]] = None,
    default_crew_size: int = DEFAULT_CREW_SIZE,
    hours_per_day: float = DEFAULT_HOURS_PER_DAY,
    min_quantity: float = 0.0,
    category_overrides: Optional[Dict[str, str]] = None,
    aggregate: bool = True,
) -> List[Dict[str, Any]]:
    """Turn priced BOQ lines into CPM-ready, man-hour-loaded activities.

    ``manhours_per_unit`` maps a work category to its norm IN THAT CATEGORY'S
    OWN UNIT (mh/m3 for concrete, mh/m2 for blockwork, mh/kg for rebar). Every
    category present in the BOQ must appear or :class:`MissingProductivity` is
    raised naming the gaps — nothing is defaulted.

    ``crew_size`` sets heads per category (default ``default_crew_size``);
    duration follows from man-hours and crew, so re-planning with a bigger
    gang is a crew change, not a new norm.

    Sequencing: substructure packages first, then superstructure, each in
    :data:`CONSTRUCTION_SEQUENCE` order; the first activity of a package
    depends on the last of the previous one (finish-to-start), and lines
    inside a package run back-to-back because they share a crew. Overlap is a
    planner's decision, not a derivation.
    """
    crew_size = crew_size or {}
    # Same work measured many times over is ONE activity (see aggregate_lines):
    # without this a 640-line BOQ becomes 640 chained tasks and a 73-year
    # programme. Pass aggregate=False for a line-by-line programme.
    line_items = aggregate_lines(line_items) if aggregate else list(line_items or [])
    grouped = group_by_category(line_items, category_overrides)
    grouped = {cat: [i for i in items if _qty(i) > min_quantity]
               for cat, items in grouped.items()}
    grouped = {cat: items for cat, items in grouped.items() if items}

    missing = {
        cat: (items[0].get("unit") or "unit")
        for cat, items in grouped.items()
        if not manhours_per_unit.get(cat)
    }
    if missing:
        raise MissingProductivity(missing)

    # Package key carries an ORDER RANK beside the category so enabling work
    # can be sequenced apart from its trade while still using that trade's
    # norm, crew and phase. Blinding is Concrete (it is), but it is struck
    # BEFORE the rebar fixed on top of it; ranking by trade alone scheduled it
    # after (live 2026-08-17).
    seq_index = {c: i for i, c in enumerate(CONSTRUCTION_SEQUENCE)}
    _enabling_rank = seq_index["Piling/Foundations"] + 0.5
    # Horizontal substructure waterproofing goes ON the blinding and UNDER the
    # raft — between the blinding and the steel fixed on top of it. Ranking it
    # by trade (Waterproofing sits after Concrete) scheduled the membrane after
    # the pour it is meant to protect (operator, 2026-08-17: "waterproofing is
    # after blinding and after raft"). Vertical/applied-after work — tanking to
    # retaining walls, protection boards to raft sides — keeps the post-pour
    # slot, which is the other half of that sentence.
    _membrane_rank = seq_index["Piling/Foundations"] + 0.7
    _after_pour_rank = seq_index["Concrete"] + 0.5

    staged: Dict[tuple, List[Dict[str, Any]]] = {}
    for cat, items in grouped.items():
        for item in items:
            desc = str(item.get("description") or item.get("item_key") or "")
            stage = element_stage(desc, cat)
            if _ENABLING_RX.search(desc):
                rank = _enabling_rank
            elif (stage == SUBSTRUCTURE
                  and cat == "Waterproofing/Insulation"):
                rank = (_after_pour_rank if _AFTER_POUR_RX.search(desc)
                        else _membrane_rank)
            else:
                rank = float(seq_index.get(cat, len(seq_index)))
            staged.setdefault((stage, rank, cat), []).append(item)

    ordered = sorted(
        staged, key=lambda k: (0 if k[0] == SUBSTRUCTURE else 1, k[1], k[2]),
    )

    activities: List[Dict[str, Any]] = []
    prev_package_tail: Optional[str] = None
    for p_idx, key in enumerate(ordered, start=1):
        stage, _rank, cat = key
        norm = float(manhours_per_unit[cat])
        crew = int(crew_size.get(cat, default_crew_size))
        trade = _TRADE_OF.get(cat, "general")
        phase = _slug(cat) if stage == SUPERSTRUCTURE else f"{_slug(cat)}_substructure"
        prev_in_package: Optional[str] = None
        for a_idx, item in enumerate(staged[key], start=1):
            qty = _qty(item)
            mh = manhours(qty, norm)
            act_id = f"{p_idx}.{a_idx}"
            preds = [prev_in_package] if prev_in_package else (
                [prev_package_tail] if prev_package_tail else [])
            activities.append({
                "id": act_id,
                "code": act_id,
                "name": (item.get("description") or item.get("item_key") or "Item")[:120],
                "duration_days": duration_days(mh, crew, hours_per_day),
                "predecessors": preds,
                # one entry per head: the manpower histogram sums these
                "resources": [trade] * crew,
                "crew_size": crew,
                "total_manhours": round(mh, 1),
                "wbs_phase": phase,
                # provenance — every number traceable to a BOQ line and a norm
                "boq": {
                    "item_key": item.get("item_key"),
                    "quantity": qty,
                    "unit": item.get("unit"),
                    "manhours_per_unit": norm,
                    "total_manhours": round(mh, 1),
                    "crew_size": crew,
                    "hours_per_day": hours_per_day,
                    "category": cat,
                    "stage": stage,
                    "line_count": int(item.get("line_count") or 1),
                    "unit_cost": item.get("unit_cost"),
                    "total_cost": item.get("total_cost"),
                },
            })
            prev_in_package = act_id
        prev_package_tail = prev_in_package
    return activities


def total_manhours(activities: Iterable[Dict[str, Any]]) -> float:
    """Programme man-hours — the figure the histogram and S-curve integrate to."""
    return round(sum(float(a.get("total_manhours") or 0) for a in activities), 1)


def schedule_basis(manhours_per_unit: Dict[str, float],
                   crew_size: Optional[Dict[str, int]] = None,
                   hours_per_day: float = DEFAULT_HOURS_PER_DAY,
                   default_crew_size: int = DEFAULT_CREW_SIZE) -> List[str]:
    """Human-readable basis — what every duration rests on."""
    crew_size = crew_size or {}
    out = [
        f"Man-hours = quantity x norm; duration = man-hours / (crew x "
        f"{hours_per_day:g} h shift), rounded up to whole working days "
        f"(minimum 1).",
    ]
    for cat in sorted(manhours_per_unit):
        crew = crew_size.get(cat, default_crew_size)
        out.append(f"{cat}: {manhours_per_unit[cat]:g} man-hours/unit, crew of {crew}")
    out.append("Work packages run substructure first, then superstructure, "
               "sequenced finish-to-start in construction order; lines within "
               "a package run consecutively (shared crew). Overlap between "
               "packages is a planning decision and is not assumed here.")
    return out


# ── leftover F1: high-level WBS from retrieved BOQ demolition rows ─────────
#
# generate_wbs is a template scheduler. Live F1 asked for a WBS over "the
# demolition and site clearance scope in this project's BOQ" and still got
# the building scaffold (Site Preparation / Substructure / Superstructure)
# because retrieved CESMM rows were never elected into the tree. These
# helpers parse measured rows that are already in the excerpts, filter to
# demolition / site-clearance, and build a high-level tree. They do not
# invent item codes or descriptions. Kill-switch: BOQ_SCOPE_WBS=0.

_BOQ_SCOPE_WBS_PHASE = "Demolition and Site Clearance"

_PIPE_BOQ_ROW_RE = re.compile(
    r"\|\s*(?P<code>[A-Za-z]\s*\d{2,4}(?:\.\d+)?)\s*"
    r"\|\s*(?P<desc>[^|\n]+?)\s*\|",
)
_INDEXER_BOQ_LINE_RE = re.compile(
    r"(?i)BOQ line item(?:[^—\n]*)?[—–-]\s*(?P<desc>.+?):\s*total quantity",
)
_CESMM_PROSE_ROW_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9])(?P<code>[A-Z]\s*\d{2,4}(?:\.\d+)?)\s+"
    r"(?P<desc>[A-Za-z][^|\n]{3,120})",
)
_SKIP_BOQ_DESC_RE = re.compile(
    r"(?i)\b(?:part\s+summary|grand\s+total|carried\s+forward|"
    r"total\s+this\s+page|brought\s+forward|item\s*$|description\s*$)\b|"
    r"^(?:is\s+a\b|rate\s+only\b)",
)
_TRAILING_MEASURED_RE = re.compile(
    r"\s+\d[\d,]*(?:\.\d+)?\s*"
    r"(?:ha|m2|m²|m3|m³|nr|nos|no\.?|m|lm|sum|item|kg|t)\b.*$",
    re.IGNORECASE,
)
_DEMO_SITE_CLEAR_ITEM_RE = re.compile(
    r"(?i)\b(?:demolit|site\s+clear|clearance|clearing|grubbing|"
    r"tree\s+remov|remov(?:al|e)\s+of\s+(?:trees?|fence|culvert|pavement|"
    r"carriageway)|breaking\s+out|chain\s+link|carriageway|sidewalk|"
    r"road\s+markings?)\b",
)
_CESMM_D_CODE_RE = re.compile(r"(?i)^D\d{2,4}(?:\.\d+)?$")
_DEMO_BOQ_NAME_RE = re.compile(
    r"(?i)demolit|site\s*clear|clearance|clearing",
)
_BOQ_SCOPE_TITLE_PHRASES = (
    "demolition and site clearance",
    "site clearance",
    "bill of quantities",
    "schedule of quantities",
)
_BOQ_SCOPE_FILENAME_TERMS = ("demolition", "clearance", "clearing", "boq")


def boq_scope_wbs_enabled() -> bool:
    """ON by default — live leftover F1 BOQ-grounded WBS.

    ``BOQ_SCOPE_WBS=0`` restores the building-template scaffold.
    """
    return (os.getenv("BOQ_SCOPE_WBS", "1") or "").strip().lower() not in (
        "0", "false", "no", "off",
    )


def _compact_cesmm_code(code: str) -> str:
    return re.sub(r"\s+", "", (code or "").strip()).upper()


def _clean_boq_description(desc: str) -> str:
    text = re.sub(r"\s+", " ", (desc or "").strip())
    text = _TRAILING_MEASURED_RE.sub("", text).strip(" |-")
    return text[:120]


def parse_boq_measured_rows(text: str) -> List[Dict[str, Any]]:
    """Extract CESMM / priced-bill rows from retrieved chunk text.

    Accepts pipe tables, indexer ``BOQ line item — …`` lines, and
    ``D110 General site clearance`` prose. Returns only rows that
    already appear in ``text`` — nothing is invented.
    """
    blob = text or ""
    if not blob.strip():
        return []
    try:
        from app.core.rag.vector_store import normalize_cesmm_item_codes
        blob = normalize_cesmm_item_codes(blob) or blob
    except Exception as exc:  # noqa: BLE001 — parse the raw excerpt if normalize is down
        logger.warning("CESMM normalize skipped; parsing raw excerpt: %s", exc)
    found: List[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    seen_codes: set[str] = set()

    def _add(code: str, desc: str) -> None:
        item_key = _compact_cesmm_code(code)
        description = _clean_boq_description(desc)
        if not description or _SKIP_BOQ_DESC_RE.search(description):
            return
        if item_key and not re.match(r"(?i)^[A-Z]\d{2,4}(?:\.\d+)?$", item_key):
            return
        # One package per CESMM code. A later "D529.3 is a Rate Only
        # item…" sentence is the same row, not a second work package.
        if item_key and item_key in seen_codes:
            return
        key = (item_key, _norm_desc(description))
        if not key[1] or key in seen:
            return
        seen.add(key)
        if item_key:
            seen_codes.add(item_key)
        found.append({
            "item_key": item_key or None,
            "description": description,
        })

    for match in _PIPE_BOQ_ROW_RE.finditer(blob):
        _add(match.group("code"), match.group("desc"))
    for match in _INDEXER_BOQ_LINE_RE.finditer(blob):
        _add("", match.group("desc"))
    for match in _CESMM_PROSE_ROW_RE.finditer(blob):
        _add(match.group("code"), match.group("desc"))
    return found


def item_is_demolition_or_site_clearance(item: Dict[str, Any]) -> bool:
    """True for CESMM class D or a demolition / site-clearance description."""
    code = _compact_cesmm_code(str(item.get("item_key") or ""))
    if _CESMM_D_CODE_RE.match(code):
        return True
    desc = str(item.get("description") or item.get("item_key") or "")
    if _DEMO_SITE_CLEAR_ITEM_RE.search(desc):
        return True
    return categorize(desc) == "Demolition"


def filter_demolition_site_clearance_items(
    items: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Keep retrieved demolition / site-clearance rows only."""
    out: List[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    seen_codes: set[str] = set()
    for item in items or []:
        if not isinstance(item, dict):
            continue
        if not item_is_demolition_or_site_clearance(item):
            continue
        code = _compact_cesmm_code(str(item.get("item_key") or ""))
        key = (
            code,
            _norm_desc(item.get("description") or item.get("item_key")),
        )
        if not key[1] or key in seen:
            continue
        if code and code in seen_codes:
            continue
        seen.add(key)
        if code:
            seen_codes.add(code)
        out.append(item)
    return out


def _chunk_looks_like_measured_boq(text: str, filename: str) -> bool:
    """True when the excerpt is a bill row, not contract prose about the bill."""
    try:
        from app.core.rag.retriever import document_is_a_bill_of_quantities
    except Exception:  # noqa: BLE001 — retrieval helper must not break WBS
        document_is_a_bill_of_quantities = None
    if document_is_a_bill_of_quantities and document_is_a_bill_of_quantities(filename):
        return True
    blob = text or ""
    if re.search(r"(?i)BOQ line item", blob):
        return True
    if blob.count("|") >= 4 and _CESMM_PROSE_ROW_RE.search(blob):
        return True
    # OCR / prose bills: no pipes and no recognised filename, but the
    # excerpt already carries CESMM D / demolition measured rows.
    rows = parse_boq_measured_rows(blob)
    return any(item_is_demolition_or_site_clearance(r) for r in rows)


def _resolve_rag_project_id(project_id: str) -> str:
    """Map the Master Corpus UI alias to the backing RAG project.

    Chunks are stored under ``MASTER_CORPUS_SOURCE_PROJECT_ID`` (the
    backing source project). Searching the alias ``master_corpus`` is empty.
    Listing helpers already remap; ``chunks_for_docs`` / ``retrieve_with_filter``
    do not.
    """
    try:
        from app.core.projects import _master_corpus_source
        return _master_corpus_source(project_id) or project_id
    except Exception:  # noqa: BLE001 — keep the caller's id
        return project_id


# A contract number (PREFIX-YEAR[-SEQ]: AB-2031-007, AB2031) in a BOQ-derived
# WBS answer. The WBS is built from bill rows; a contract-number cite means
# the answer leaned on a Conditions of Contract volume instead.
_CONTRACT_CITE_RE = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z]{2,}-?(?:19|20)\d{2}(?:-\d+)?(?![A-Za-z0-9])"
)
_WBS_REFUSE_RE = re.compile(
    r"(?i)\b(?:cannot|can\s*'?\s*t|could\s+not|unable\s+to|do\s+not\s+have|"
    r"don\s*'?\s*t\s+have|does\s+not\s+contain|do\s+not\s+contain|"
    r"excerpts?\s+do\s+not|not\s+(?:enough|sufficient)\s+(?:to\s+)?"
    r"(?:produce|generate|build)|won\s*'?\s*t\s+(?:produce|generate)|"
    r"cannot\s+produce|can\s*'?\s*t\s+produce|cannot\s+generate|"
    r"can\s*'?\s*t\s+generate|i\s+(?:cannot|can\s*'?\s*t|could\s+not))\b"
)
_WBS_HIERARCHY_RE = re.compile(r"(?m)^(?:### )?\d+(?:\.\d+)*\s+\S")
_WBS_CESMM_D_RE = re.compile(r"(?i)\bD\d{2,4}(?:\.\d+)?\b")
_WBS_TEMPLATE_RE = re.compile(
    r"(?i)template\s+scaffold|site\s+preparation|superstructure|"
    r"project_type\s+inferred:\s*building"
)
_WBS_DEMOLITION_SCOPE_RE = re.compile(r"(?i)\b(?:demolit|site\s+clear)\w*")


def _source_contract_recency(name: str) -> Optional[Tuple[int, int]]:
    """PREFIX-YEAR-SEQ sort key from a filename, or None when undated."""
    try:
        from app.core.rag.retriever import extract_contract_doc_ids
    except Exception as exc:  # noqa: BLE001
        logger.warning("contract-id parse unavailable for BOQ election: %s", exc)
        return None
    ids = extract_contract_doc_ids(name or "")
    if not ids:
        return None
    parts = ids[0].lower().split("-")
    year = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else -1
    seq = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else -1
    return (year, seq)


def _prefer_newer_contract_year_docs(
    docs: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """Stable-sort demolition bills so the later PREFIX-YEAR-SEQ is first."""
    dated = []
    undated = []
    for doc in docs or []:
        blob = f"{doc.get('original_name') or ''} {doc.get('file_path') or ''}"
        key = _source_contract_recency(blob)
        if key is None:
            undated.append(doc)
        else:
            dated.append((key, doc))
    dated.sort(key=lambda pair: pair[0], reverse=True)
    return [doc for _key, doc in dated] + undated


def _prefer_newer_contract_year_items(
    items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Keep measured rows from the newest dated bill when two years mixed.

    Undated fixture rows (no PREFIX-YEAR-SEQ in ``source``) stay. When a
    later-year bill is present, drop earlier-year siblings so an unnamed
    ask cannot elect the earlier contract once the later one's bill is in
    the pool.
    """
    dated_keys: List[Tuple[int, int]] = []
    for item in items or []:
        key = _source_contract_recency(str((item or {}).get("source") or ""))
        if key is not None:
            dated_keys.append(key)
    if not dated_keys:
        return list(items or [])
    best = max(dated_keys)
    kept: List[Dict[str, Any]] = []
    for item in items or []:
        key = _source_contract_recency(str((item or {}).get("source") or ""))
        if key is None or key == best:
            kept.append(item)
    return kept


def boq_wbs_answer_fails_wrong_contract(answer: str) -> bool:
    """True for the live leftover F1 FAIL: refuse and/or a contract-number cite.

    Soft battery ``must_any: clearance|trees|pavement`` marks PASS when
    the refuse merely mentions demolition. That is a FAIL.
    """
    blob = answer or ""
    cites_contract = bool(_CONTRACT_CITE_RE.search(blob))
    refuses = bool(_WBS_REFUSE_RE.search(blob))
    if cites_contract and refuses:
        return True
    if cites_contract:
        return True
    if refuses and not boq_wbs_answer_is_grounded(blob):
        return True
    return False


def boq_wbs_answer_is_grounded(answer: str) -> bool:
    """True for a demolition / site-clearance WBS from BOQ rows, not a refuse.

    Requires the demolition/site-clearance scope AND either a numbered
    hierarchy or CESMM D-codes. A mere mention of demolition is not enough.
    Template scaffold and contract-number cites fail.
    """
    blob = answer or ""
    if not blob.strip():
        return False
    if _CONTRACT_CITE_RE.search(blob):
        return False
    if _WBS_TEMPLATE_RE.search(blob):
        return False
    if _WBS_REFUSE_RE.search(blob) and not _WBS_CESMM_D_RE.search(blob):
        return False
    if not _WBS_DEMOLITION_SCOPE_RE.search(blob):
        return False
    return bool(_WBS_HIERARCHY_RE.search(blob) or _WBS_CESMM_D_RE.search(blob))


def _list_boq_scope_documents(project_id: str) -> List[Dict[str, str]]:
    """Bills of quantities already in the project document pool.

    Filename / title listing, not cosine top-k. Live leftover F1 after
    #524 still hit the building template because ``retrieve_with_filter``
    never returned the demolition BOQ from a 2700-doc corpus.
    """
    try:
        from app.core.projects import (
            documents_matching_filename_terms,
            documents_matching_title_phrase,
        )
        from app.core.rag.retriever import document_is_a_bill_of_quantities
    except Exception as exc:  # noqa: BLE001
        logger.warning("BOQ-scope WBS document listing unavailable: %s", exc)
        return []

    found: List[Dict[str, str]] = []
    seen: set[str] = set()

    def _add(doc: Dict[str, str]) -> None:
        doc_id = str((doc or {}).get("id") or "").strip()
        if not doc_id or doc_id in seen:
            return
        name = str(doc.get("original_name") or doc.get("file_path") or "")
        if not document_is_a_bill_of_quantities(name):
            return
        seen.add(doc_id)
        found.append({
            "id": doc_id,
            "original_name": str(doc.get("original_name") or ""),
            "file_path": str(doc.get("file_path") or ""),
        })

    for phrase in _BOQ_SCOPE_TITLE_PHRASES:
        try:
            for doc in documents_matching_title_phrase(project_id, phrase, limit=8):
                _add(doc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("BOQ-scope title listing failed: %s", exc)
    try:
        for doc in documents_matching_filename_terms(
            project_id,
            list(_BOQ_SCOPE_FILENAME_TERMS),
            min_terms=2,
            require_letter=False,
            limit=8,
        ):
            _add(doc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("BOQ-scope filename listing failed: %s", exc)

    def _demo_name(doc: Dict[str, str]) -> bool:
        blob = f"{doc.get('original_name') or ''} {doc.get('file_path') or ''}"
        return bool(_DEMO_BOQ_NAME_RE.search(blob))

    demo = [d for d in found if _demo_name(d)]
    # Prefer the demolition / site-clearance bill. A generic priced bill
    # is used only when no scoped bill is in the pool — we still only
    # keep rows the demolition filter accepts.
    # Unnamed Master Corpus leftover F1: when both years' demolition
    # bills are in the pool, the later package owns the ask. Arrival
    # order used to hand generate_wbs the earlier year's bill (or only
    # its Conditions of Contract prose) and synthesis then refused.
    pool = demo or found
    pool = _prefer_newer_contract_year_docs(pool)
    chosen = pool[:4]
    return chosen


def _items_from_chunk_text(
    text: str,
    filename: str,
    *,
    require_measured_gate: bool,
) -> List[Dict[str, Any]]:
    if require_measured_gate and not _chunk_looks_like_measured_boq(text, filename):
        return []
    out: List[Dict[str, Any]] = []
    for row in parse_boq_measured_rows(text):
        item = dict(row)
        if filename:
            item["source"] = filename
        out.append(item)
    return out


def _items_from_named_boq_docs(project_id: str) -> List[Dict[str, Any]]:
    """Parse measured rows from BOQ-named documents in the project pool."""
    docs = _list_boq_scope_documents(project_id)
    if not docs:
        return []
    source_id = _resolve_rag_project_id(project_id)
    try:
        from app.core.rag.vector_store import get_store
        store = get_store()
    except Exception as exc:  # noqa: BLE001
        logger.warning("BOQ-scope WBS store unavailable: %s", exc)
        return []
    names = {d["id"]: (d.get("original_name") or "") for d in docs}
    ids = [d["id"] for d in docs]
    collected: List[Dict[str, Any]] = []
    fetch_all = getattr(store, "doc_chunk_texts", None)
    if callable(fetch_all):
        try:
            by_doc = fetch_all(source_id, ids) or {}
        except Exception as exc:  # noqa: BLE001
            logger.warning("BOQ-scope WBS doc_chunk_texts failed: %s", exc)
            by_doc = {}
        for doc_id, texts in by_doc.items():
            filename = names.get(doc_id, "")
            for text in texts or []:
                collected.extend(
                    _items_from_chunk_text(
                        text or "", filename, require_measured_gate=False,
                    )
                )
        if collected:
            return collected
    fetch = getattr(store, "chunks_for_docs", None)
    if not callable(fetch):
        return collected
    try:
        hits = fetch(source_id, ids, k_per_doc=80) or []
    except Exception as exc:  # noqa: BLE001
        logger.warning("BOQ-scope WBS chunks_for_docs failed: %s", exc)
        return collected
    for chunk in hits:
        filename = (
            names.get(getattr(chunk, "doc_id", "") or "")
            or getattr(chunk, "source_name", "")
            or ""
        )
        collected.extend(
            _items_from_chunk_text(
                getattr(chunk, "text", "") or "",
                filename,
                require_measured_gate=False,
            )
        )
    return collected


def retrieve_boq_scope_items(query: str, project_id: str) -> List[Dict[str, Any]]:
    """Retrieve measured BOQ rows for a demolition / site-clearance WBS ask.

    Prefers chunks from BOQ-named documents already in the project pool
    (filename listing). Semantic ``retrieve_with_filter`` is a fallback
    only — on Master Corpus it often returns Conditions of Contract
    prose and never the demolition bill. Empty when there is no project,
    both paths fail, or no measured rows land. Does not invent items.
    """
    if not project_id or not (query or "").strip():
        return []
    collected: List[Dict[str, Any]] = []
    try:
        collected.extend(_items_from_named_boq_docs(project_id))
    except Exception as exc:  # noqa: BLE001
        logger.warning("BOQ-scope WBS named-doc retrieval failed: %s", exc)

    source_id = _resolve_rag_project_id(project_id)
    try:
        from app.core.rag.retriever import _doc_name_for_id, retrieve_with_filter
        chunks, _noise = retrieve_with_filter(query, source_id, k=8)
    except Exception as exc:  # noqa: BLE001 — named-doc rows still stand
        logger.warning("BOQ-scope WBS retrieval failed: %s", exc)
        chunks = []

    for chunk in chunks or []:
        text = getattr(chunk, "text", "") or ""
        filename = (
            getattr(chunk, "source_name", "")
            or _doc_name_for_id(getattr(chunk, "doc_id", "") or "")
        )
        collected.extend(
            _items_from_chunk_text(
                text, filename, require_measured_gate=True,
            )
        )
    # Elect the later-year bill BEFORE CESMM-code dedupe. Filtering first
    # would keep an earlier-year D110 and drop the later-year twin.
    return filter_demolition_site_clearance_items(
        _prefer_newer_contract_year_items(collected),
    )


def wbs_tree_from_boq_items(
    items: Iterable[Dict[str, Any]],
    *,
    phase: str = _BOQ_SCOPE_WBS_PHASE,
) -> Dict[str, str]:
    """High-level tree: one phase, one package per retrieved BOQ row.

    Package names keep the CESMM code when the row had one. No template
    packages are added.
    """
    scoped = filter_demolition_site_clearance_items(items)
    if not scoped:
        return {}
    tree: Dict[str, str] = {"1": phase}
    for idx, item in enumerate(scoped, start=1):
        code = _compact_cesmm_code(str(item.get("item_key") or ""))
        desc = _clean_boq_description(
            str(item.get("description") or item.get("item_key") or "")
        )
        name = f"{code} {desc}".strip() if code else desc
        if name:
            tree[f"1.{idx}"] = name
    return tree


def activities_from_boq_scope_outline(
    items: Iterable[Dict[str, Any]],
    *,
    phase: str = _BOQ_SCOPE_WBS_PHASE,
) -> List[Dict[str, Any]]:
    """One lightweight activity per retrieved row so CPM / cost-load can run.

    Durations are placeholders (1 day). This is a high-level scope WBS, not
    a productivity-derived programme — callers must say so.
    """
    scoped = filter_demolition_site_clearance_items(items)
    activities: List[Dict[str, Any]] = []
    prev: Optional[str] = None
    phase_slug = _slug(phase)
    for idx, item in enumerate(scoped, start=1):
        act_id = f"1.{idx}"
        code = _compact_cesmm_code(str(item.get("item_key") or ""))
        desc = _clean_boq_description(
            str(item.get("description") or item.get("item_key") or "Item")
        )
        name = f"{code} {desc}".strip() if code else desc
        activities.append({
            "id": act_id,
            "code": act_id,
            "name": name[:120] or f"Item {idx}",
            "duration_days": 1,
            "predecessors": [prev] if prev else [],
            "resources": ["demolition"],
            "wbs": act_id,
            "wbs_code": act_id,
            "wbs_phase": phase_slug,
            "boq": {
                "item_key": item.get("item_key") or code or None,
                "description": desc,
                "source": item.get("source"),
                "category": "Demolition",
            },
        })
        prev = act_id
    return activities


# ── learning the norms back from a built programme ─────────────────────────
#
# Operator, 2026-08-17: "the built programs and its manpower histogram should
# tell u the productivity per manhour". Exactly — a completed programme is a
# productivity record, not just a plan: each activity's man-hours (crew x
# duration x shift, or the resource assignment where one exists) divided by the
# quantity it delivered IS the norm for that work, measured on this project
# with these crews. Deriving it here puts the operator's OWN history first in
# the sourcing order, ahead of any published reference.

def activity_manhours(activity: Dict[str, Any],
                      hours_per_day: float = DEFAULT_HOURS_PER_DAY) -> Optional[float]:
    """Man-hours an activity consumed.

    Prefers an explicit figure (``total_manhours``, or a P6 resource
    assignment's ``target_qty``/``act_reg_qty``, which are already man-hours),
    and only then falls back to crew x duration x shift. Returns None when the
    activity carries neither — a silent zero would understate a norm and make
    the next programme optimistic.
    """
    for key in ("total_manhours", "actual_manhours", "target_qty", "act_reg_qty"):
        v = activity.get(key)
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    crew = activity.get("crew_size") or activity.get("manpower")
    if not crew:
        res = activity.get("resources")
        crew = len(res) if isinstance(res, list) else None
    dur = activity.get("duration_days") or activity.get("duration")
    if crew and dur:
        return float(crew) * float(dur) * hours_per_day
    return None


def norms_from_programme(
    activities: Iterable[Dict[str, Any]],
    quantities: Dict[str, float],
    *,
    hours_per_day: float = DEFAULT_HOURS_PER_DAY,
    category_overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Derive man-hours-per-unit per work category from a BUILT programme.

    ``quantities`` maps an activity id (or its exact name) to the quantity that
    activity delivered, in the category's own unit — the one thing a programme
    cannot know by itself.

    Returns ``{category: {manhours_per_unit, total_manhours, total_quantity,
    samples, activities}}``. The norm is quantity-weighted (total man-hours /
    total quantity), never a mean of ratios: a 4,000 m2 slab and a 20 m2
    landing must not carry equal weight.

    Activities with no quantity, or no recoverable man-hours, are skipped and
    counted rather than assumed — an unmeasured activity has no norm to teach.
    """
    overrides = category_overrides or {}
    acc: Dict[str, Dict[str, Any]] = {}
    for a in activities or []:
        key = a.get("id") or a.get("code") or a.get("name")
        qty = quantities.get(str(key))
        if qty is None:
            qty = quantities.get(str(a.get("name")))
        if not qty or qty <= 0:
            continue
        mh = activity_manhours(a, hours_per_day)
        if not mh:
            continue
        name = a.get("name") or ""
        cat = (overrides.get(str(key)) or overrides.get(name)
               or (a.get("boq") or {}).get("category") or categorize(name))
        bucket = acc.setdefault(cat, {"total_manhours": 0.0, "total_quantity": 0.0,
                                      "samples": 0, "activities": []})
        bucket["total_manhours"] += mh
        bucket["total_quantity"] += float(qty)
        bucket["samples"] += 1
        bucket["activities"].append(name)
    out: Dict[str, Dict[str, Any]] = {}
    for cat, b in acc.items():
        if b["total_quantity"] <= 0:
            continue
        out[cat] = {
            "manhours_per_unit": round(b["total_manhours"] / b["total_quantity"], 4),
            "total_manhours": round(b["total_manhours"], 1),
            "total_quantity": round(b["total_quantity"], 2),
            "samples": b["samples"],
            "activities": b["activities"],
            "basis": "measured from a built programme on this project",
        }
    return out
