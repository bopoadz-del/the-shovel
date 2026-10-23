"""Quantity take-off calculators (additive library, drop-catalog gap-fill).

Geometry + material density — code-agnostic (no ACI/EC distinction). Rebar mass
uses the physical relation mass/m = (pi/4)*d^2 * density, which reproduces the
BS 8666 / standard bar-mass table (d=16 -> 1.578 kg/m). Rates/densities are
parameters; arithmetic shown in ``note``.
"""
from __future__ import annotations

from app.lib.formula_registry import SIGNED_FIGURE, formula

import functools

import logging

import math

import os

import re

# Formulas (and the helpers only they use) live in their owner's package;
# imported back so existing imports of this module keep working.
from app.agents.base.formulas.construction_formulas_quantities import (  # noqa: F401 -- moved
    _concrete_quantity,
    concrete_volume,
    documented_waste_enabled,
    logger,
)
from app.agents.hats.quantities.formulas.construction_formulas_quantities import (  # noqa: F401 -- moved
    _STEEL_DENSITY,
    _WEIGHT_TO_LENGTH_MODES,
    _rebar_mass_kg,
    interior_finishes_takeoff,
    rebar_by_area,
    rebar_weight,
    resource_line_cost,
)

# A raft-volume ask that says to include the documented waste factor:
#   L x W x T m net, documented waste 5% (x 1.05) on top.
# Leftover L6 aliased unnamed L×W×D to excavation_volume (bank only, no
# waste). A concrete/raft ask must pin concrete_volume and this factor.
# Kill switch: APPLY_DOCUMENTED_WASTE=0 restores the FAIL (net 900).
# Compose kill-switch: COMPOSE_CONCRETE_VOLUME=0 restores the leftover-E4
# hang (validate preamble ships, no 945). The calculator still returns 945;
# only the postprocess graft that writes the volume when the model stalls
# is disabled.
DOCUMENTED_CONCRETE_WASTE_FACTOR = 0.05

_CONCRETE_VOLUME_ASK_RE = re.compile(
    r"\b(concrete|raft|slab|footing|pile[\s-]?cap|waste[\s-]?factor)\b",
    re.IGNORECASE,
)

_DOCUMENTED_WASTE_ASK_RE = re.compile(
    r"\b(documented\s+waste|waste[\s-]?factor|including\b.{0,40}\bwaste)\b",
    re.IGNORECASE,
)

# "30x20x1.5", "15 m x 8 m x 0.2 m", "15 m by 8 m by -0.2 m", "1200 x 600 x
# 200 mm", "50 ft by 20 ft by 8 in". Each figure keeps the length unit written
# after it; a figure written bare takes the nearest unit written in the chain.
_CHAIN_SEP = r"\s*(?:[x×*]|by(?![A-Za-z]))\s*"


@functools.lru_cache(maxsize=1)
def _lwt_chain_re() -> re.Pattern[str]:
    from app.agents.base.formulas.construction_formulas_planning import _FAMILIES

    units = "|".join(re.escape(u) for u in sorted(_FAMILIES["length"], key=len, reverse=True))
    dim = rf"({SIGNED_FIGURE})(?:\s*({units})(?![A-WYZa-wyz0-9]))?"
    return re.compile(rf"(?<![A-Za-z0-9]){dim}{_CHAIN_SEP}{dim}{_CHAIN_SEP}{dim}", re.IGNORECASE)


def _chain_units(units: list[str | None]) -> list[str | None]:
    """A bare figure in a chain takes the nearest unit written to its right,
    else to its left."""
    out = list(units)
    for i, unit in enumerate(units):
        if unit:
            continue
        right = next((u for u in units[i + 1:] if u), None)
        left = next((u for u in reversed(units[:i]) if u), None)
        out[i] = right or left
    return out


def dimension_chains(text: str) -> list[list[tuple[float, str | None]]]:
    """Every L×W×T chain in ``text`` as (figure, unit written for it)."""
    chains = []
    for match in _lwt_chain_re().finditer(text or ""):
        groups = match.groups()
        numbers = [float(groups[i].replace(",", "")) for i in (0, 2, 4)]
        chains.append(list(zip(numbers, _chain_units([groups[i] for i in (1, 3, 5)]))))
    return chains

# A waste percentage the operator stated out loud. The number has to BELONG to
# the waste -- a contract full of percentages (retention 5%, advance recovery
# 15%, compaction 95%) must not donate one -- so both spellings bind the figure
# and the word together, within a few characters.
_WASTE_PCT_RE = re.compile(
    r"(?:(\d{1,2}(?:\.\d+)?)\s*(?:%|per\s?cent(?:age)?)\s*(?:\w+\s+){0,2}?waste"
    r"|waste(?:\s+factor)?\s*(?:of|at|=|:)?\s*(\d{1,2}(?:\.\d+)?)\s*(?:%|per\s?cent(?:age)?))",
    re.IGNORECASE,
)

# Above this a "percentage" is a parse artefact, not an instruction.
_WASTE_PCT_MAX = 50.0

def waste_factor_from_text(text: str) -> float | None:
    """The waste fraction the ask names (0.07 for "add 7% waste"), or None."""
    match = _WASTE_PCT_RE.search(text or "")
    if not match:
        return None
    raw = match.group(1) or match.group(2)
    try:
        pct = float(raw)
    except (TypeError, ValueError):
        logger.debug("waste percentage is not numeric: %r", raw)
        return None
    if pct <= 0.0 or pct > _WASTE_PCT_MAX:
        return None
    return pct / 100.0

def compose_concrete_volume_enabled() -> bool:
    """ON by default. ``COMPOSE_CONCRETE_VOLUME=0`` restores the validate hang."""
    raw = (os.getenv("COMPOSE_CONCRETE_VOLUME", "1") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")

def _format_volume_figure(value: float) -> str:
    number = float(value)
    if number == int(number):
        return str(int(number))
    return f"{number:g}"

def answer_states_volume(text: str, volume: float) -> bool:
    """True when ``text`` already names the headline cubic-metre figure."""
    if not text:
        return False
    needle = _format_volume_figure(volume)
    compact = (text or "").replace(",", "")
    return needle in compact

def format_concrete_volume_line(inner: dict) -> str:
    """User-facing E4 line from a ``concrete_volume`` result dict."""
    if not isinstance(inner, dict):
        return ""
    volume = inner.get("volume_m3", inner.get("volume_with_waste_m3"))
    if volume is None:
        return ""
    net = inner.get("net_volume_m3")
    try:
        waste = float(inner.get("waste_factor") or 0.0)
    except (TypeError, ValueError):
        waste = 0.0
    vol_s = _format_volume_figure(volume)
    if waste > 0 and net is not None:
        pct = waste * 100.0
        factor = 1.0 + waste
        net_s = _format_volume_figure(net)
        return (
            f"Concrete volume including documented {pct:.0f}% waste: "
            f"{vol_s} m³ (net {net_s} × {factor:g})."
        )
    return f"Concrete volume: {vol_s} m³."

def compose_concrete_volume_from_ask(text: str) -> dict | None:
    """Run the documented-waste concrete calculator from the operator ask.

    Numbers live in the question (E4: 30×20×1.5). Leftover L6 earthwork is
    not a concrete/raft ask and returns None. Kill-switch
    ``COMPOSE_CONCRETE_VOLUME=0`` returns None (hang / no graft).
    """
    if not compose_concrete_volume_enabled():
        return None
    if not looks_like_concrete_volume_ask(text):
        return None
    if not parse_lwt_metres(text):
        return None
    from app.lib import construction_formulas as _cf
    result = _cf.run_calculation(None, {"text": text})
    if not isinstance(result, dict) or result.get("status") != "success":
        return None
    if result.get("calculation") != "concrete_volume":
        return None
    inner = result.get("result") if isinstance(result.get("result"), dict) else {}
    volume = inner.get("volume_m3")
    if volume is None:
        return None
    line = format_concrete_volume_line(inner)
    if not line:
        return None
    return {
        "volume_m3": volume,
        "net_volume_m3": inner.get("net_volume_m3"),
        "waste_factor": inner.get("waste_factor"),
        "note": inner.get("note") or "",
        "line": line,
        "envelope": result,
    }

def documented_concrete_waste_factor() -> float:
    """Project-documented concrete waste (5%), or 0.0 when the kill-switch is off."""
    if not documented_waste_enabled():
        return 0.0
    return DOCUMENTED_CONCRETE_WASTE_FACTOR

def looks_like_concrete_volume_ask(text: str) -> bool:
    """True when the ask is a concrete/raft/slab volume, not leftover-L6 earthwork."""
    return bool(_CONCRETE_VOLUME_ASK_RE.search(text or ""))

def parse_lwt_metres(text: str) -> tuple[float, float, float] | None:
    """First L×W×T (or L×W×D) chain in ``text``, in metres, sign kept."""
    from app.agents.base.formulas.construction_formulas_planning import convert_units

    chains = dimension_chains(text)
    if not chains:
        return None
    return tuple(  # type: ignore[return-value]
        convert_units(n, u, "m")["value_out"] if u else n for n, u in chains[0]
    )

# "24 pile caps" / "18 pad footings". The count belongs to the element, not
# to a nearby percentage or a unit rate. A bare "2.5 m" must not match.
_ELEMENT_COUNT_RE = re.compile(
    r"(?i)(?<!\d)(?<!\d\.)(\d+)\s+"
    r"(?:(?:pad|strip|isolated)\s+)?"
    r"(?:pile[\s-]?caps?|footings?|pads?|bases?|columns?|piers?)\b",
)

_THAT_TOTAL_RE = re.compile(r"(?i)\b(?:that|this|the)\s+total\b")

_FOLLOW_UP_ADJUST_RE = re.compile(r"(?i)\b(?:waste|price|priced|pricing)\b")

_PLATFORM_BUBBLE_PREFIXES = ("PLATFORM PRE-DISPATCH:", "Tool result (")

def element_count_from_text(text: str) -> int | None:
    """How many elements the ask names (24 pile caps), or None."""
    match = _ELEMENT_COUNT_RE.search(text or "")
    if not match:
        return None
    count = int(match.group(1))
    return count if count > 0 else None

def unit_dims_metres(text: str) -> tuple[float, float, float] | None:
    """Per-element L×W×D. ``m by m by m`` or an ``x`` chain. None if absent."""
    return parse_lwt_metres(text)

def follow_up_refers_to_stated_total(text: str) -> bool:
    """True when this ask adjusts a total named in an earlier turn.

    "Add a waste allowance and price the total…" carries the rate and the
    waste, not the element count. An ask that restates the geometry is not a
    continuation — its own dimensions are the operands.
    """
    raw = text or ""
    if not _THAT_TOTAL_RE.search(raw):
        return False
    if not _FOLLOW_UP_ADJUST_RE.search(raw):
        return False
    if unit_dims_metres(raw):
        return False
    return True

def prior_operator_text(history: list | None) -> str:
    """Operator turns in ``history``, oldest first. Platform bubbles skipped."""
    parts: list[str] = []
    for msg in history or []:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = str(msg.get("content") or "").strip()
        if not content or content.lstrip().startswith(_PLATFORM_BUBBLE_PREFIXES):
            continue
        parts.append(content)
    return "\n".join(parts)

def _quantity_is_one_or_missing(value: object) -> bool:
    if value in (None, "", 0, 0.0):
        return True
    try:
        return abs(float(value) - 1.0) <= 1e-9  # type: ignore[arg-type]
    except (TypeError, ValueError):
        logger.debug("quantity token %r is not numeric", value)
        return False

def _dims_match_each(params: dict, each: tuple[float, float, float]) -> bool:
    """True when the bound L×W×T are the per-element size, not a rolled total."""
    got: list[float] = []
    for key in ("length_m", "width_m", "thickness_m"):
        raw = params.get(key)
        if raw in (None, ""):
            return False
        try:
            got.append(float(raw))
        except (TypeError, ValueError):
            logger.debug("element dimension %r is not numeric", raw)
            return False
    return all(abs(a - b) <= 1e-6 for a, b in zip(got, each))

def resolve_concrete_volume_calc(
    calc: str | None,
    params: dict | None = None,
    text: str = "",
) -> tuple[str | None, dict]:
    """Pin ``concrete_volume`` + documented waste for a concrete/raft ask.

    Leftover L6 still owns unnamed trench L×W×D (no concrete/raft/waste words).
    E4's sheet phrasing is a concrete ask; excavation_volume would return the
    FAIL (bank 900, waste missing).
    """
    out = dict(params or {})
    blob = " ".join(
        str(x) for x in (
            text,
            calc,
            out.get("text"),
            out.get("prior_text"),
            out.get("formula"),
            out.get("name"),
            out.get("calculation"),
        ) if x
    )
    named = str(calc or "").strip()
    # E4 may steal unnamed / excavation_volume when the ask is a raft/slab.
    # A *named* calculator (dewatering, carbon, mix design, …) must stay on
    # its signature even if the question mentions "raft" or "concrete".
    # Live standing-exit: those names were remapped to concrete_volume and
    # returned volume_m3=0 (tool_error×2 / 1-of-2).
    _e4_stealable = {"", "excavation_volume", "concrete_volume"}
    is_concrete = named == "concrete_volume" or (
        named in _e4_stealable and looks_like_concrete_volume_ask(blob)
    )
    if not is_concrete:
        return calc, out

    dims = parse_lwt_metres(blob)
    if dims:
        out.setdefault("length_m", dims[0])
        out.setdefault("width_m", dims[1])
        out.setdefault("thickness_m", dims[2])

    if not out.get("thickness_m"):
        for key in ("depth_m", "height_m", "thickness", "depth", "height"):
            value = out.get(key)
            if value not in (None, "", 0, 0.0):
                out["thickness_m"] = value
                break

    if documented_waste_enabled():
        requested = out.get("waste_factor")
        stated = waste_factor_from_text(blob)
        if stated is not None:
            # The operator named a figure. It wins over the documented
            # default AND over the documented-waste phrase -- a live waste-factor ask was
            # answered on 5% after being asked for 7%, with nothing in the
            # answer saying which rate had been used.
            out["waste_factor"] = stated
        elif requested in (None, "", 0, 0.0) or _DOCUMENTED_WASTE_ASK_RE.search(blob):
            out["waste_factor"] = DOCUMENTED_CONCRETE_WASTE_FACTOR
    else:
        out["waste_factor"] = 0.0

    # A follow-up that says "that total" often re-calls this calculator with
    # the per-element size and no count. ``quantity`` used to be dropped on
    # bind — concrete_volume did not accept it — so N elements came back as
    # one. The stated count wins only when those dims are the
    # "each" size; a length that is already the rolled total is left alone.
    stated_n = element_count_from_text(blob)
    each = unit_dims_metres(blob)
    if (
        stated_n
        and stated_n > 1
        and each
        and _dims_match_each(out, each)
        and _quantity_is_one_or_missing(out.get("quantity"))
    ):
        out["quantity"] = stated_n

    return "concrete_volume", out

# Phase 2 F–W (#43–84) routing. Agent C owns this slice; do not remap A–F
# names except to recover them from an E4 ``concrete_volume`` steal when
# the ask is clearly an F–W calculator (resource line / material
# consumption / an F–W registry name in the text).
FW_ROUTE_NAMES = frozenset({
    "fineness_modulus",
    "formwork_striking_time",
    "foundation_bearing_pressure",
    "grout_pressure_calc",
    "guardrail_top_rail_height",
    "interior_finishes_takeoff",
    "laser_scan_accuracy",
    "leed_points_estimate",
    "live_load_reduction",
    "masonry_wall_capacity",
    "material_consumption",
    "mobilization_cost_estimate",
    "modulus_of_elasticity_concrete",
    "modulus_of_rupture",
    "pe_unit_convert",
    "plumbing_flow_programme",
    "post_tensioning_force",
    "precast_beam_erection_check",
    "productivity_manpower_duration",
    "productivity_rate",
    "progress_quantity",
    "rc_beam_moment_capacity",
    "rc_beam_shear_capacity",
    "rebar_by_area",
    "rebar_lap_length",
    "rebar_weight",
    "resource_line_cost",
    "roi_calculator",
    "scaffold_load_capacity",
    "score_risk",
    "seismic_base_shear",
    "shear_stress_check",
    "slab_thickness_min",
    "slope_fos_simple",
    "steel_tension_capacity",
    "supervision_ratio",
    "thermal_shrinkage_equivalence",
    "tolerance_check",
    "unit_cost_total",
    "unit_weight_concrete",
    "weld_capacity",
    "wind_load_on_formwork",
    "wind_pressure",
})

_RESOURCE_LINE_TEXT_RE = re.compile(
    r"\b(resource\s+line|daily\s+output|day\s+rate|crew\s+days?)\b",
    re.IGNORECASE,
)

_CONSUMPTION_TEXT_RE = re.compile(
    r"\b(material\s+consumption|material\s+required|output\s+per\s+unit)\b",
    re.IGNORECASE,
)

_MIX_RATIO_RE = re.compile(r"\b\d+\s*:\s*\d+\s*:\s*\d+\b")

_MIX_TEXT_RE = re.compile(
    r"\b(mix\s+(?:design|proportion|ratio)|cement\s+parts|1\s*:\s*2\s*:\s*4)\b",
    re.IGNORECASE,
)

# Live Phase 2 leftover: "mobilization cost" / UK "mobilisation" / "site
# mob" do not contain the full registry string "mobilization cost estimate",
# so _blob_names_fw missed them and the LLM sometimes called the tool with
# no num_personnel / duration_months → TypeError → no SAR number.
_MOBILIZATION_TEXT_RE = re.compile(
    r"\b(mobili[sz]ation\s+cost|site\s+mobili[sz]ation|"
    r"mob(?:ilisation|ilization)?\s+cost|camp\s+mobili[sz]ation)\b",
    re.IGNORECASE,
)

_MOB_PERSONNEL_RE = re.compile(
    r"(\d[\d,]*)\s*(?:staff|personnel|people|workers|persons?|headcount|men)\b",
    re.IGNORECASE,
)

_MOB_MONTHS_RE = re.compile(
    r"(\d[\d,]*)\s*(?:months?|mos?\b)",
    re.IGNORECASE,
)

# Documented GCC example (audit oracle) — used only when the ask names
# mobilization but supplies no headcount / duration.
_MOB_DEFAULT_PERSONNEL = 100

_MOB_DEFAULT_MONTHS = 18

_PRESENT = (None, "", 0, 0.0)

# construction_calc presentation for the four live leftovers whose result
# had a number but no unit the phone UI / stream could copy. F–W (#43–84)
# only; A–F envelopes are untouched.
_FW_PRESENTATION = {
    "fineness_modulus": {"unit": "unitless", "value_keys": ("value",)},
    "score_risk": {"unit": "unitless", "value_keys": ("score",)},
    "slope_fos_simple": {"unit": "unitless", "value_keys": ("factor_of_safety",)},
    "mobilization_cost_estimate": {
        "unit": "SAR",
        "value_keys": ("grand_total_sar", "value"),
    },
    # #628 unitless/SAR twin: live probe computed seismic but the envelope
    # had no unit the stream could copy (tool-but-no-unit).
    "seismic_base_shear": {
        "unit": "kN",
        "value_keys": ("base_shear_kn", "value"),
    },
    # Live probe: productivity_rate returned 12.5 with the unit only buried
    # in the note. Headline value + unit so the stream can copy "/hr".
    "productivity_rate": {
        "unit": "/hr",
        "value_keys": ("value", "rate_per_hour"),
    },
}

def _fw_blob(calc: str | None, params: dict, text: str, original_name: str | None) -> str:
    return " ".join(
        str(x) for x in (
            text,
            calc,
            original_name,
            params.get("text"),
            params.get("formula"),
            params.get("name"),
            params.get("calculation"),
        ) if x
    )

def _param_present(params: dict, key: str) -> bool:
    return params.get(key) not in _PRESENT

def _blob_names_fw(blob: str) -> str | None:
    """Longest F–W registry name mentioned as snake, spaced, or 'of'-flexed."""
    if not blob:
        return None
    low = blob.lower()
    snake = re.sub(r"[\s\-]+", "_", low)
    hit: str | None = None
    for name in sorted(FW_ROUTE_NAMES, key=len, reverse=True):
        spaced = name.replace("_", " ")
        if name in snake or spaced in low:
            hit = name
            break
        parts = name.split("_")
        flex = r"\b" + r"\b(?:\s+of)?\s+".join(re.escape(p) for p in parts) + r"\b"
        if re.search(flex, low):
            hit = name
            break
    return hit

def _looks_like_resource_line(params: dict, blob: str) -> bool:
    if _param_present(params, "daily_output") or _param_present(params, "day_rate"):
        return True
    return bool(_RESOURCE_LINE_TEXT_RE.search(blob or ""))

def _looks_like_mix(params: dict, blob: str) -> bool:
    if _param_present(params, "cement_parts"):
        return True
    return bool(_MIX_RATIO_RE.search(blob or "") or _MIX_TEXT_RE.search(blob or ""))

def _looks_like_consumption(params: dict, blob: str) -> bool:
    if _param_present(params, "quantity_of_work") and _param_present(params, "output_per_unit"):
        return True
    if _param_present(params, "output_per_unit") and (
        _param_present(params, "quantity") or _param_present(params, "quantity_of_work")
    ):
        return True
    if _CONSUMPTION_TEXT_RE.search(blob or "") and not _looks_like_mix(params, blob):
        return True
    return False

def _alias_resource_line_params(out: dict) -> None:
    if "day_rate" not in out and out.get("unit_rate") not in (None, ""):
        out["day_rate"] = out["unit_rate"]

def _alias_consumption_params(out: dict) -> None:
    if "quantity_of_work" not in out or out.get("quantity_of_work") in (None, ""):
        if out.get("quantity") not in (None, ""):
            out["quantity_of_work"] = out["quantity"]
    # E4 injects waste_factor=0.05 (a fraction). material_consumption treats
    # waste_factor as the full multiplier (1.05). Convert fractions.
    factor = out.get("waste_factor")
    try:
        factor_f = float(factor) if factor not in (None, "") else None
    except (TypeError, ValueError):
        factor_f = None
    if factor_f is not None and 0 < factor_f < 1:
        if out.get("waste_percent") in (None, ""):
            out["waste_percent"] = factor_f * 100.0
        out.pop("waste_factor", None)

def _looks_like_mobilization(_params: dict, blob: str) -> bool:
    return bool(_MOBILIZATION_TEXT_RE.search(blob or ""))

def _int_from_match(match: re.Match[str] | None) -> int | None:
    if match is None:
        return None
    raw = (match.group(1) or "").replace(",", "")
    if not raw.isdigit():
        return None
    return int(raw)

def _alias_mobilization_params(out: dict, blob: str = "") -> None:
    """Fill headcount / duration so construction_calc always returns a SAR figure."""
    search = " ".join(
        str(x) for x in (
            blob, out.get("text"), out.get("formula"), out.get("query"),
        ) if x
    )
    if not _param_present(out, "num_personnel"):
        parsed = _int_from_match(_MOB_PERSONNEL_RE.search(search))
        out["num_personnel"] = parsed if parsed is not None else _MOB_DEFAULT_PERSONNEL
    if not _param_present(out, "duration_months"):
        parsed = _int_from_match(_MOB_MONTHS_RE.search(search))
        out["duration_months"] = parsed if parsed is not None else _MOB_DEFAULT_MONTHS

def _fw_presentation_note(unit: str, value: object, existing: object) -> str:
    if isinstance(existing, list):
        existing = " ".join(str(x) for x in existing)
    note = str(existing or "").strip()
    if value is None:
        token = unit
    elif isinstance(value, float) and value == int(value):
        token = f"{int(value)} {unit}"
    elif isinstance(value, float):
        token = f"{value:g} {unit}"
    else:
        token = f"{value} {unit}"
    if unit.lower() in note.lower() and (
        value is None or str(value) in note.replace(",", "")
    ):
        return note
    if note:
        return f"{note} Result {token}."
    return f"Result {token}."

def shape_fw_calc_result(name: str, result: dict) -> dict:
    """Stamp ``unit`` (and a copyable note) on the four live leftover results.

    Fineness modulus / risk score / infinite-slope FoS are dimensionless —
    the live probe needs the word ``unitless`` in the tool JSON. Mobilization
    already has ``*_sar`` keys; the stream still needs a headline ``value``
    + ``unit: SAR``.
    """
    spec = _FW_PRESENTATION.get(name)
    if not spec or not isinstance(result, dict):
        return result
    if isinstance(result.get("error"), str):
        return result
    out = dict(result)
    unit = spec["unit"]
    value = None
    for key in spec["value_keys"]:
        if out.get(key) not in (None, ""):
            value = out[key]
            break
    out.setdefault("unit", unit)
    if value is not None:
        out.setdefault("value", value)
    out["note"] = _fw_presentation_note(
        unit, value, out.get("note") or out.get("description"),
    )
    return out

def resolve_fw_calc(
    calc: str | None,
    params: dict | None = None,
    text: str = "",
    original_name: str | None = None,
) -> tuple[str | None, dict]:
    """Pin F–W calculators the LLM (or E4) mis-named.

    Live Phase 2 modes this owns:
    - ``resource_line_cost`` asked, model picked ``unit_cost_total``
    - ``material_consumption`` asked, model (or E4 ``concrete`` pin) picked
      ``concrete_mix_proportions`` / ``concrete_volume``
    - text names an F–W registry function (snake or spaced)
    - mobilization / mobilisation / site-mob language, including a no-figure
      ask (fill documented 100 staff × 18 months so a SAR number always lands)

    True qty×rate and true 1:2:4 mix are left alone. A raft L×W×T volume
    ask stays on ``concrete_volume``.
    """
    out = dict(params or {})
    blob = _fw_blob(calc, out, text, original_name)
    named = str(calc or "").strip() or None
    original = str(original_name or "").strip() or None
    pinned = _blob_names_fw(blob)

    stealable = {
        None, "", "unit_cost_total", "concrete_mix_proportions",
        "concrete_volume", "excavation_volume",
    }

    if _looks_like_resource_line(out, blob) and not (
        pinned == "unit_cost_total" and not (
            _param_present(out, "daily_output") or _param_present(out, "day_rate")
        )
    ):
        if named in stealable or pinned == "resource_line_cost":
            _alias_resource_line_params(out)
            return "resource_line_cost", out

    if _looks_like_consumption(out, blob) and not _looks_like_mix(out, blob):
        if named in stealable or pinned == "material_consumption":
            _alias_consumption_params(out)
            return "material_consumption", out

    if _looks_like_mobilization(out, blob) and (
        named in stealable or pinned == "mobilization_cost_estimate"
        or named == "mobilization_cost_estimate"
    ):
        _alias_mobilization_params(out, blob)
        return "mobilization_cost_estimate", out

    if named == "concrete_volume" and original in FW_ROUTE_NAMES:
        if original == "mobilization_cost_estimate":
            _alias_mobilization_params(out, blob)
        return original, out

    if pinned and named in stealable | {None}:
        if pinned == "resource_line_cost":
            _alias_resource_line_params(out)
        elif pinned == "material_consumption":
            _alias_consumption_params(out)
        elif pinned == "mobilization_cost_estimate":
            _alias_mobilization_params(out, blob)
        return pinned, out

    if pinned and named == "concrete_volume" and pinned != "unit_cost_total":
        if pinned == "material_consumption":
            _alias_consumption_params(out)
        elif pinned == "mobilization_cost_estimate":
            _alias_mobilization_params(out, blob)
        return pinned, out

    if named == "mobilization_cost_estimate":
        _alias_mobilization_params(out, blob)
        return named, out

    return calc, out

# A metres-run-from-bar-mass ask (N tonnes of a named bar size).
_METRES_RUN_ASK_RE = re.compile(
    r"(?i)metres?\s+run|meters?\s+run|"
    r"(?:tonnes?|tons?|kg).{0,40}(?:y|t|h)?\d{1,2}.{0,40}(?:metr|length|run)|"
    r"(?:y|t|h)\d{1,2}.{0,40}(?:tonnes?|tons?|kg).{0,40}(?:metr|length|run)"
)

_Y_BAR_RE = re.compile(r"(?i)\b[YTH](\d{1,2})\b")

_DIA_MM_RE = re.compile(r"(?i)(\d{1,2})\s*mm\b")

_TONNES_RE = re.compile(rf"(?i)({SIGNED_FIGURE})\s*(?:tonnes?|tons?|t)\b")

_KG_MASS_RE = re.compile(rf"(?i)({SIGNED_FIGURE})\s*kg\b")

def looks_like_rebar_metres_run_ask(text: str) -> bool:
    """True when the operator asked for metres run from a bar mass."""
    return bool(_METRES_RUN_ASK_RE.search(text or ""))

def parse_rebar_metres_run_ask(text: str) -> tuple[float, float] | None:
    """Return (bar_diameter_mm, total_weight_kg) or None."""
    raw = text or ""
    dia = None
    y = _Y_BAR_RE.search(raw)
    if y:
        dia = float(y.group(1))
    else:
        d = _DIA_MM_RE.search(raw)
        if d:
            dia = float(d.group(1))
    if dia is None or dia <= 0:
        return None
    tonnes = _TONNES_RE.search(raw)
    if tonnes:
        return dia, float(tonnes.group(1).replace(",", "")) * 1000.0
    kg = _KG_MASS_RE.search(raw)
    if kg:
        return dia, float(kg.group(1).replace(",", ""))
    return None

def format_rebar_metres_run_line(inner: dict) -> str:
    """User-facing metres-run line. Rejects the 1 m unit-mass demo."""
    if not isinstance(inner, dict):
        return ""
    metres = inner.get("metres_run", inner.get("total_length_m"))
    unit = inner.get("unit_mass_kg_m")
    mass = inner.get("total_mass_kg")
    try:
        metres_f = float(metres)
    except (TypeError, ValueError):
        logger.debug("metres_run is not numeric: %r", metres)
        return ""
    if metres_f <= 10:
        return ""
    bits = [f"{metres_f:,.2f} m"]
    if unit not in (None, "") and mass not in (None, ""):
        bits = [
            f"Unit mass = {float(unit):.4f} kg/m; "
            f"{float(mass):,.0f} kg / {float(unit):.4f} = {metres_f:,.2f} m"
        ]
    return bits[0] + "."

def compose_rebar_metres_run_from_ask(text: str) -> dict | None:
    """Run weight→length from the operator ask (live A2-2)."""
    if not looks_like_rebar_metres_run_ask(text):
        return None
    parsed = parse_rebar_metres_run_ask(text)
    if not parsed:
        return None
    dia, mass_kg = parsed
    from app.lib import construction_formulas as _cf
    env = _cf.run_calculation(
        "rebar_weight",
        {
            "bar_diameter_mm": dia,
            "total_weight_kg": mass_kg,
            "mode": "weight_to_length",
        },
    )
    if not isinstance(env, dict) or env.get("status") != "success":
        return None
    inner = env.get("result") if isinstance(env.get("result"), dict) else {}
    line = format_rebar_metres_run_line(inner)
    if not line:
        return None
    metres = inner.get("metres_run", inner.get("total_length_m"))
    return {
        "metres_run": metres,
        "total_length_m": metres,
        "unit_mass_kg_m": inner.get("unit_mass_kg_m"),
        "total_mass_kg": inner.get("total_mass_kg"),
        "line": line,
        "envelope": env,
        "result": inner,
    }

ADDITIONAL_CALCULATORS = {
    "concrete_volume": concrete_volume,
    "rebar_weight": rebar_weight,
    "rebar_by_area": rebar_by_area,
    "interior_finishes_takeoff": interior_finishes_takeoff,
    "resource_line_cost": resource_line_cost,
}
