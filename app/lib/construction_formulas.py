"""
Construction Calculations — deterministic engineering formula library.

Single source of truth for construction calculations (deep foundations,
concrete technology, structural, crane planning, cost build-up, MEP, QC).
Sourced from SPEC-C900 course material. No project/company names.

DETERMINISTIC TOOLS, not a rate oracle: every cost build-up takes its unit
rates as PARAMETERS. The hardcoded defaults are indicative GCC fallbacks only —
callers should pass real rates from RAG (company priced BOQ -> GK) so the cost
answer stays grounded (see the-fork-rates-in-rag). The maths never changes; only
the inputs do.
"""
from __future__ import annotations

from app.lib.formula_registry import SIGNED_FIGURE, formula

import math

from dataclasses import dataclass, field

from typing import Any, Dict, Iterable, List, Optional, Tuple

import dataclasses as _dc

import inspect as _inspect

import json

import logging

import contextvars
import re

from app.agents.base.formulas.construction_formulas_planning import (
    convert_units as _convert_units,
    parse_unit as _parse_unit,
    text_unit_pattern as _text_unit_pattern,
)

# Formulas (and the helpers only they use) live in their owner's package;
# imported back so existing imports of this module keep working.
from app.agents.base.formulas.construction_formulas_shared import (  # noqa: F401 -- moved
    _is_aci_code,
)
from app.agents.base.formulas.construction_formulas import (  # noqa: F401 -- moved
    diaphragm_wall_panel_volume,
)
from app.agents.hats.design.formulas.construction_formulas import (  # noqa: F401 -- moved
    DewateringResult,
    _I_MM4_FLOOR,
    _second_moment_mm4,
    beam_deflection_cantilever_point_load,
    beam_deflection_cantilever_udl,
    beam_deflection_ss_point_load_midspan,
    beam_deflection_ss_udl,
    composite_column_design,
    concrete_mix_design_sg,
    concrete_mix_slip_form,
    dewatering_uplift_check,
    dewatering_well_point_spacing,
    foundation_bearing_pressure,
    modulus_of_elasticity_concrete,
    post_tensioning_force,
    precast_beam_erection_check,
    shear_stress_check,
    thermal_shrinkage_equivalence,
    unit_weight_concrete,
)
from app.agents.hats.planning.formulas.construction_formulas import (  # noqa: F401 -- moved
    electrical_installation_sequence,
    plumbing_flow_programme,
    supervision_ratio,
)
from app.agents.hats.procurement.formulas.construction_formulas import (  # noqa: F401 -- moved
    crane_planning,
)
from app.agents.hats.qaqc.formulas.construction_formulas import (  # noqa: F401 -- moved
    FormworkStrikingResult,
    concrete_maturity_strength,
    concrete_thermal_cracking_check,
    fineness_modulus,
    formwork_striking_time,
    grout_pressure_calc,
    modulus_of_rupture,
)
from app.agents.hats.quantities.formulas.construction_formulas import (  # noqa: F401 -- moved
    cost_buildup_concrete,
    cost_buildup_formwork,
    cost_buildup_rebar,
    crane_cost_estimate,
    mobilization_cost_estimate,
)
from app.agents.hats.safety.formulas.construction_formulas import (  # noqa: F401 -- moved
    wind_load_on_formwork,
)

logger = logging.getLogger(__name__)

def _build_calculator_registry() -> "Dict[str, Any]":
    """name -> function for every DECLARED formula (``@formula`` in
    ``app.lib.formula_registry``). A function is a calculator only when it
    declares its owner, description, inputs and outputs where it is defined;
    importing the formula modules is what registers them."""
    from app.lib import formula_registry

    return formula_registry.calculators()

CALCULATORS: Dict[str, Any] = _build_calculator_registry()

def available_calculations() -> List[str]:
    return sorted(CALCULATORS)

# Stem collisions: first two tokens are a document/schedule lookup, not a calc.
_FORMULA_NAME_STEM_COLLISIONS = frozenset({
    "critical path",
})

# Optional dests that error unless the incoming key is the real name.
# "sand" / "dune sand" must not suffix-bind onto dune_sand_pct (live mix
# table SGs were scored tool_error).
_EXPLICIT_BIND_ONLY = frozenset({
    "dune_sand_pct",
})

def _phrase_words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def calculators_named_by_display(text: str) -> List[str]:
    """Registry names whose display name the ask writes out in full.

    The display name is what the platform itself calls a calculator in its
    answers and Sources. One of three or more words is as specific as a
    registry id written out. A two-word one ("Concrete volume") is also a
    bill and take-off phrase, so it is left to the id rules.
    """
    hits, subsumed = _display_hits(text)
    return [name for name, phrase in hits.items()
            if len(phrase.split()) >= 3 and name not in subsumed]


def _display_hits(text: str) -> Tuple[Dict[str, str], set]:
    """({registry name: display phrase} for every calculator whose display
    name, two words or more, the ask writes out in full; the names among them
    written only as part of a longer display name the ask also writes --
    "Kestrel total" inside "Kestrel total net" names the longer one)."""
    words = _phrase_words(text)
    if not words:
        return {}, set()
    padded = f" {' '.join(words)} "
    hits = {name: phrase for name, phrase in _display_phrases() if phrase in padded}
    subsumed = {name for name, phrase in hits.items()
                if any(other != phrase and phrase in other for other in hits.values())}
    return {name: phrase.strip() for name, phrase in hits.items()}, subsumed


_DISPLAY_PHRASES: Dict[int, List[Tuple[str, str]]] = {}


def _display_phrases() -> List[Tuple[str, str]]:
    """(registry name, padded display-name phrase) for every calculator whose
    display name has three or more words. Built once per registry state:
    rebuilding it for every message ran on the event loop each turn."""
    from app.lib.formula_registry import all_specs

    specs = all_specs()
    key = len(specs)
    cached = _DISPLAY_PHRASES.get(key)
    if cached is None:
        cached = []
        for spec in specs:
            if spec.name not in CALCULATORS:
                continue
            phrase = _phrase_words(spec.display_name)
            if len(phrase) >= 2:
                cached.append((spec.name, f" {' '.join(phrase)} "))
        _DISPLAY_PHRASES.clear()
        _DISPLAY_PHRASES[key] = cached
    return cached


def calculator_name_from_text(text: str) -> Optional[str]:
    """Unique registry name implied by ``text``, or None if absent/ambiguous.

    Full underscore / spaced names beat 2-token stems so
    ``concrete mix design sg`` is not tied with ``concrete_mix_slip_form``
    and ``cost buildup concrete`` is not tied with ``cost_buildup_rebar``.
    A display name written out in full ranks next to a full registry name.
    A unique 3-token tail (``well point spacing``) also counts — live
    dewatering asks omit the leading ``dewatering_``.
    """
    raw = text or ""
    if not raw.strip():
        return None
    try:
        from app.lib.construction_formulas_structural_rc import (
            looks_like_slab_thickness_min_ask,
        )
        if looks_like_slab_thickness_min_ask(raw):
            return "slab_thickness_min"
    except Exception:  # noqa: BLE001 — name lookup must not break dispatch
        logger.exception("slab_thickness_min name check failed")
    underscored = raw.lower().replace("-", "_")
    spaced = raw.lower().replace("-", " ").replace("_", " ")
    full: List[str] = []
    triples: List[str] = []
    pairs: List[str] = []
    stems: List[str] = []
    for name in CALCULATORS:
        if len(name) < 6:
            continue
        tokens = [tok for tok in name.lower().split("_") if tok]
        spaced_name = name.replace("_", " ")
        if name.lower() in underscored or (
            len(tokens) >= 3 and spaced_name in spaced
        ):
            full.append(name)
            continue
        if len(tokens) >= 3:
            for i in range(len(tokens) - 2):
                chunk = " ".join(tokens[i:i + 3])
                if chunk in spaced:
                    triples.append(name)
                    break
            for i in range(len(tokens) - 1):
                chunk = " ".join(tokens[i:i + 2])
                if len(chunk) >= 10 and chunk in spaced:
                    pairs.append(name)
            stem = " ".join(tokens[:2])
            if (
                len(stem) >= 8
                and stem in spaced
                and stem not in _FORMULA_NAME_STEM_COLLISIONS
            ):
                stems.append(name)
    hits, subsumed = _display_hits(raw)
    display = [name for name, phrase in hits.items() if len(phrase.split()) >= 3]
    # A two-word display name is also a take-off or lookup phrase ("Concrete
    # volume"): it names the calculator only when no id rule does and the ask
    # gives a figure to compute with.
    two_word = ([name for name, phrase in hits.items() if len(phrase.split()) == 2]
                if re.search(r"\d", raw) else [])
    for group in (full, display, triples, pairs, stems, two_word):
        uniq = [name for name in dict.fromkeys(group) if name not in subsumed]
        if len(uniq) == 1:
            return uniq[0]
        if len(uniq) > 1:
            return None
    return None

# PMI (BCWS/BCWP/ACWP/BAC) and long PE-sheet names → calculate_evm kwargs.
# ``run_calculation`` keeps only exact signature names, so uppercase / long
# aliases were dropped and the tool reported missing PV/EV/AC.
_EVM_CALC_ALIASES: Dict[str, str] = {
    "pv": "pv",
    "planned_value": "pv",
    "bcws": "bcws",
    "ev": "ev",
    "earned_value": "ev",
    "bcwp": "bcwp",
    "ac": "ac",
    "actual_cost": "ac",
    "acwp": "acwp",
    "bac": "bac",
    "budget_at_completion": "bac",
}

# Symbols an engineer writes, and therefore what the model sends. Live on
# 9e6fe98 both of these came back as tool errors instead of figures:
#   "Unknown argument(s) for beam_deflection_cantilever_udl: E, I"
#   "Unknown argument(s) for modulus_of_elasticity_concrete: c"
#
# A rename alone would be worse than the error it replaces: E is quoted in GPa
# (200) where ``ec_mpa`` wants MPa (200 000), and I in m4 (2.0e-4) where
# ``i_mm4`` wants mm4 (2.0e8), so a bare alias computes a deflection a thousand
# times wrong and reports it with no complaint at all.
#
# The unit is therefore inferred from magnitude, and only across a gap that
# cannot occur in practice: a modulus below 1000 is GPa (concrete 20-40, steel
# 200 — nothing real sits between 1000 and 20 000 MPa), and a second moment
# below 1.0 is m4 (a 10 mm square bar is 833 mm4). Outside those windows the
# value is taken as already canonical, and ``_second_moment_mm4`` still refuses
# anything that is neither.
_SYMBOL_DEST: Dict[str, str] = {"e": "ec_mpa", "i": "i_mm4", "c": "code"}

_E_GPA_CEILING = 1000.0      # below this, E is GPa

_I_M4_CEILING = 1.0          # below this, I is m4

# "200 GPa", "2.0e-4 m4", "2.0e8 mm^4": the model writes the unit it read.
# The unit token must be allowed to CONTAIN digits ("m4", "mm^4", "cm4") but
# not to START with one, or it would eat the tail of the number. The first
# version had no digits in the class, so every I unit failed to match and the
# raw string fell through to a 253,125,000 mm deflection.
_SYMBOL_VALUE_RE = re.compile(
    r"^\s*([-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?)\s*"
    r"((?:[A-Za-z\u00b2\u00b3\u2074^/][A-Za-z0-9\u00b2\u00b3\u2074^/]*)?)\s*$"
)

_SYMBOL_UNIT: Dict[str, str] = {"e": "MPa", "i": "mm4"}

class _UnknownUnit(ValueError):
    """A symbol carried a unit this binder cannot scale. Refuse, never guess."""

def _canonical_from_symbol(symbol: str, value: Any) -> Any:
    """Scale a symbol's value onto its canonical parameter's unit.

    A bare number is scaled by magnitude (see the ceilings above). A string
    with a unit token is scaled by that unit; a unit the table does not know
    raises so the call errors instead of computing a figure that is wrong by
    orders of magnitude and saying nothing -- live, "200 GPa" passed through
    unconverted once and produced a 253,125,000,000 mm deflection.
    """
    if symbol == "c":
        return value
    target = _SYMBOL_UNIT[symbol]
    if isinstance(value, str):
        m = _SYMBOL_VALUE_RE.match(value)
        if not m:
            # A string with digits in it that does not parse is a value this
            # binder cannot scale. Passing it through is how a wrong figure
            # gets computed and reported without complaint; refuse instead.
            if any(ch.isdigit() for ch in value):
                raise _UnknownUnit(
                    f"{symbol.upper()}={value!r}: cannot read a number and a unit from it")
            return value
        number = float(m.group(1).replace(",", "."))
        unit = m.group(2)
        if unit:
            out = _convert_units(number, unit, target)
            if "value_out" not in out:
                raise _UnknownUnit(
                    f"{symbol.upper()}={value!r}: unit {unit!r} is not one this binder "
                    f"can scale to {target} ({out.get('error', '')})")
            return out["value_out"]
    else:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return value
    if symbol == "e" and 0 < number < _E_GPA_CEILING:
        return number * 1e3      # GPa -> MPa
    if symbol == "i" and 0 < number < _I_M4_CEILING:
        return number * 1e12     # m4 -> mm4
    return number

def _alias_physics_symbols(fn: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Bind E / I / c onto the destination the calculator actually accepts.

    Only when that calculator has the destination parameter, and only when the
    canonical key is absent — an explicit ``ec_mpa`` is the instruction and a
    symbol never overrides it. A symbol whose destination this calculator does
    not accept is left alone, so it still surfaces as an unknown argument.
    """
    try:
        accepted = set(_inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return params
    out = dict(params)
    for key in list(out):
        dest = _SYMBOL_DEST.get(str(key).strip().lower())
        if not dest or dest not in accepted or key == dest:
            continue
        if out.get(dest) not in (None, ""):
            out.pop(key, None)       # canonical key already given; drop the symbol
            continue
        out[dest] = _canonical_from_symbol(str(key).strip().lower(), out.pop(key))
    return out

def _alias_calculate_evm_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Bind case-insensitive PMI / PE names onto ``calculate_evm`` kwargs.

    Canonical keys already present win; aliases only fill holes so mixed
    ``pv`` + ``BCWP`` + ``acwp`` still resolve. The source key is removed
    once it has been applied: ``pv`` and ``bcws`` are both real parameters,
    so the generic binder cannot pick one, and leaving ``planned_value``
    in the dict would be reported as an unknown argument.
    """
    out = dict(params)
    consumed: List[str] = []
    for raw_key, val in params.items():
        if val is None or val == "":
            continue
        dest = _EVM_CALC_ALIASES.get(str(raw_key).strip().lower())
        if dest is None:
            continue
        if dest not in out or out[dest] in (None, ""):
            out[dest] = val
        if _snake_key(raw_key) != _snake_key(dest):
            consumed.append(str(raw_key))
    for key in consumed:
        out.pop(key, None)
    return out

# Units for empty / incomplete construction_calc kwargs. The model often
# calls with {} or a half-set; the error must name every required input
# with its unit so the retry can bind (same class as empty-args
# payment_certificate / B-calc FAILs).
_PARAM_UNITS: Dict[str, str] = {
    "bar_diameter_mm": "mm",
    "total_length_m": "m",
    "total_weight_kg": "kg",
    "total_mass_kg": "kg",
    "total_mass_t": "t",
    "quantity": "qty",
    "quantity_executed": "qty",
    "quantity_kg": "kg",
    "daily_production": "qty/day",
    "productivity": "qty/h",
    "productivity_rate": "qty/day",
    "crew_cost_per_day": "currency/day",
    "day_rate": "currency/day",
    "gang_cost_per_day": "currency/day",
    "man_hours": "h",
    "manpower": "persons",
    "working_hours": "h",
    "remaining_manhours": "h",
    "available_hours": "h",
    "remaining_qty": "qty",
    "remaining_days": "days",
    "material_price_per_t": "SAR/t",
}

_CALC_REQUIRED_HELP: Dict[str, str] = {
    "rebar_weight": (
        "rebar_weight needs bar_diameter_mm (mm) and either "
        "total_length_m (m) or total_weight_kg (kg) with mode=weight_to_length."
    ),
    "productivity_manpower_duration": (
        "productivity_manpower_duration needs paired inputs (no invented "
        "rates): quantity_executed (qty) + man_hours (h); "
        "quantity (qty) + productivity (qty/h); "
        "quantity (qty) + daily_production (qty/day); "
        "quantity (qty) + productivity_rate (qty/gang-day) + "
        "crew_cost_per_day (currency/day); "
        "manpower (persons) + working_hours (h); "
        "remaining_manhours (h) + available_hours (h); "
        "or remaining_qty (qty) + remaining_days (days)."
    ),
}

def _param_with_unit(name: str) -> str:
    unit = _PARAM_UNITS.get(name)
    return f"{name} ({unit})" if unit else name

def required_params_error(name: str, fn: Any) -> str:
    """Error text that names every required parameter with its unit."""
    canned = _CALC_REQUIRED_HELP.get(str(name or "").strip())
    if canned:
        return canned
    try:
        sig = _inspect.signature(fn)
    except (TypeError, ValueError):
        return f"{name} needs its documented inputs; none were usable."
    required = [
        k for k, p in sig.parameters.items()
        if p.default is _inspect.Parameter.empty
        and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        and k != "self"
    ]
    if not required:
        return f"{name} needs its documented inputs; none were usable."
    return f"{name} needs: " + ", ".join(_param_with_unit(k) for k in required) + "."

def _result_is_failure(result: Dict[str, Any]) -> bool:
    """Did a calculator report failure by RETURNING rather than raising?

    Keyed on a STRING "error" deliberately: a NUMERIC field named "error" is a
    tolerance or deviation -- data, not a failure signal -- and must not be
    mistaken for one. (Swept 2026-08-15 across all registered calculators: none
    currently returns a numeric "error".)
    """
    return isinstance(result.get("error"), str)

# Keys the model / container / tool envelope add beside real calculator kwargs.
# Flatten unwraps ``params`` / ``input`` then drops these so they never
# reach fn(**kwargs). ``text`` / ``formula`` / ``prior_text`` / ``query``
# stay available for the concrete-volume / waste-factor / F–W resolvers that run *before* bind,
# and are stripped at bind time so they are not unknown-argument errors.
_BIND_JUNK_KEYS = frozenset({
    "action", "calculation", "name", "calculator", "params",
    "block", "unit", "formula", "text", "ok", "status",
    "input", "project_id", "conversation_id", "user_id",
    "message", "history", "messages", "chat",
    "kwargs", "arguments", "variables", "values",
    "prior_text", "query",
})

_FLATTEN_NEST_KEYS = ("params", "input", "kwargs", "arguments", "variables", "values")

_E4_PASSTHROUGH_KEYS = frozenset({
    "text", "formula", "prior_text", "query",
})

# Longest-first unit suffixes stripped when matching volume ↔ volume_m3.
_BIND_UNIT_SUFFIXES: Tuple[Tuple[str, str], ...] = (
    ("_kgco2e_m3", "kgCO2e/m3"),
    ("_n_mm2", "N/mm2"),
    ("_kn_m3", "kN/m3"),
    ("_kn_m2", "kN/m2"),
    ("_kn_m", "kN.m"),
    ("_mm4", "mm4"),
    ("_mm2", "mm2"),
    ("_m2", "m2"),
    ("_m3", "m3"),
    ("_mm", "mm"),
    ("_mpa", "MPa"),
    ("_kpa", "kPa"),
    ("_kn", "kN"),
    ("_kg", "kg"),
    ("_percent", "%"),
    ("_pct", "%"),
    ("_days", "d"),
    ("_day", "d"),
    ("_deg", "deg"),
    ("_m", "m"),
    ("_s", "s"),
    ("_t", "t"),
)

# Incoming name (normalized) → candidate signature names. Only a candidate
# that is actually on *this* calculator is used, and only when unique.
# Includes the #636 PMI / PE aliases so calculate_evm binds without a
# special-case table of its own (Agent C shares this path), plus #639
# excavation/claim/bolt aliases and #652 standing-exit aliases.
_BIND_SEMANTIC_ALIASES: Dict[str, Tuple[str, ...]] = {
    "pv": ("pv", "bcws"),
    "planned_value": ("pv", "bcws"),
    "ev": ("ev", "bcwp"),
    "earned_value": ("ev", "bcwp"),
    "ac": ("ac", "acwp"),
    "actual_cost": ("ac", "acwp"),
    "budget_at_completion": ("bac",),
    "volume": ("volume_m3", "quantity_m3"),
    "vol": ("volume_m3", "quantity_m3"),
    "qty": ("quantity_m3", "quantity_kg", "quantity", "volume_m3"),
    "quantity": ("quantity_m3", "quantity_kg", "quantity", "quantity_of_work"),
    # Element count the model sends ("count": 24). Alias of ``quantity``
    # only — not floor_count / room_count. A calculator without ``quantity``
    # does not accept it.
    "count": ("quantity",),
    "quantity_t": ("quantity_kg",),
    "qty_t": ("quantity_kg",),
    "tonnes": ("quantity_kg",),
    "tons": ("quantity_kg",),
    "tonne": ("quantity_kg",),
    "ton": ("quantity_kg",),
    "mass": ("quantity_kg", "total_mass_kg"),
    "rebar_kg": ("quantity_kg",),
    "material_price": ("material_price_per_t",),
    "steel_price": ("material_price_per_t",),
    "rebar_price": ("material_price_per_t",),
    "unit_rate": ("material_price_per_t", "day_rate"),
    "waste": ("waste_pct",),
    "waste_percent": ("waste_pct",),
    "bidders": ("tenderers",),
    "bids": ("tenderers",),
    "suppliers": ("tenderers",),
    "submissions": ("tenderers",),
    "tenderer": ("tenderers",),
    "diameter": ("column_diameter_mm", "diameter_mm", "bolt_diameter_mm", "diameter_m"),
    "dia": ("column_diameter_mm", "diameter_mm", "bolt_diameter_mm", "diameter_m"),
    "excavation": ("excavation_bank_m3",),
    "excavation_bank": ("excavation_bank_m3",),
    "bank": ("excavation_bank_m3",),
    "bank_volume": ("excavation_bank_m3",),
    "structure": ("structure_volume_m3",),
    "structure_volume": ("structure_volume_m3",),
    "bolt_area": ("bolt_area_mm2",),
    "ab": ("bolt_area_mm2",),
    "ab_mm2": ("bolt_area_mm2",),
    "area": ("bolt_area_mm2", "area", "floor_area_m2", "formwork_area_m2", "area_m2"),
    "fnv": ("shear_strength_mpa",),
    "fub": ("shear_strength_mpa",),
    "gross": ("gross_valuation",),
    "valuation": ("gross_valuation",),
    "gross_value": ("gross_valuation",),
    "gross_amount": ("gross_valuation",),
    "certified_gross": ("gross_valuation",),
    "ipc": ("gross_valuation",),
    "amount": ("gross_valuation", "total_cost", "claimed_amount"),
    "cost": ("total_cost",),
    "price": ("material_price_per_t", "total_cost"),
    "total": ("total_cost",),
    "area_m2": ("area", "floor_area_m2", "area_m2"),
    "gfa": ("floor_area_m2",),
    "floor_area": ("floor_area_m2",),
    "claimed": ("claimed_amount",),
    "claim": ("claimed_amount",),
    "certified": ("certified_amount",),
    "cert": ("certified_amount",),
    "retention": ("retention_percent", "retention_rate"),
    "field": ("field_dry_density",),
    "fdd": ("field_dry_density",),
    "field_density": ("field_dry_density",),
    "mdd": ("max_dry_density",),
    "lab_density": ("max_dry_density",),
    "max_density": ("max_dry_density",),
    "axial": ("axial_load_kn",),
    "axial_load": ("axial_load_kn",),
    "load": ("axial_load_kn",),
    "n_ed": ("axial_load_kn",),
    "pu": ("axial_load_kn",),
    "column_dia": ("column_diameter_mm",),
    "col_dia": ("column_diameter_mm",),
    "w": ("udl_w_kn_m", "width_m", "seismic_weight_kn", "formwork_width_m"),
    "udl": ("udl_w_kn_m",),
    "udl_w": ("udl_w_kn_m",),
    "uniform_load": ("udl_w_kn_m",),
    "span": ("span_m",),
    "l": ("span_m", "length_m", "panel_length"),
    "as": ("steel_area_mm2",),
    "ast": ("steel_area_mm2",),
    "ag": ("gross_area_mm2",),
    "ae": ("net_area_mm2",),
    "anet": ("net_area_mm2",),
    "an": ("net_area_mm2",),
    "fy": ("fy_mpa",),
    "fu": ("fu_mpa",),
    "fc": ("fc_mpa", "fck_n_mm2"),
    "fck": ("fck_n_mm2", "fc_mpa"),
    "fm": ("masonry_strength_mpa",),
    "b": ("width_m", "width_mm"),
    "breadth": ("width_m",),
    "bw": ("width_mm",),
    "d": ("depth_m", "excavation_depth", "eff_depth_mm", "bar_diameter_mm"),
    "l0": ("base_live_load_kn_m2",),
    "at": ("tributary_area_m2", "area_m2"),
    "kll": ("kll",),
    "ll": ("live_load_kn_m2",),
    "live_load": ("live_load_kn_m2",),
    "slab": ("slab_thickness_m",),
    "slab_thickness": ("slab_thickness_m",),
    "sds": ("sds",),
    "ie": ("ie",),
    "h": ("height_m", "height_mm", "formwork_height_m"),
    "height": ("height_m", "depth_m", "height_mm", "formwork_height_m"),
    "height_m": ("height_m", "depth_m"),
    "width": ("width_mm", "formwork_width_m", "width_m"),
    "v": ("wind_velocity_m_s", "wind_speed_m_s"),
    "p": ("central_point_load_kn", "axial_load_kn", "point_load_kn"),
    "rate": ("rate_percent", "material_price_per_t"),
    "delay_rate": ("rate_percent",),
    "aca": ("contract_amount",),
    "accepted_contract_amount": ("contract_amount",),
    "contract_value": ("contract_amount",),
    "hw": ("water_depth",),
    "h_w": ("water_depth",),
    "water_table": ("water_depth",),
    "water_head": ("water_depth",),
    "groundwater": ("water_depth",),
    "gwl": ("water_depth",),
    "gw_depth": ("water_depth",),
    "uplift_head": ("water_depth",),
    "raft": ("raft_thickness",),
    "traft": ("raft_thickness",),
    "t_raft": ("raft_thickness",),
    "raft_t": ("raft_thickness",),
    "raft_thk": ("raft_thickness",),
    "raft_depth": ("raft_thickness",),
    "floors": ("floor_count", "num_floors"),
    "n_floors": ("floor_count", "num_floors"),
    "nfloors": ("floor_count", "num_floors"),
    "num_floors": ("floor_count", "num_floors"),
    "storeys": ("floor_count", "num_floors"),
    "stories": ("floor_count", "num_floors"),
    "storey": ("floor_count", "num_floors"),
    "story": ("floor_count", "num_floors"),
    "levels": ("floor_count", "num_floors"),
    "n_storeys": ("floor_count", "num_floors"),
    "no_of_floors": ("floor_count", "num_floors"),
    "number_of_floors": ("floor_count", "num_floors"),
    "t": ("time_days", "thickness_m", "wall_thickness", "raft_thickness", "thickness_mm", "slab_thickness_m"),
    "time": ("time_days",),
    "days": ("time_days",),
    "age": ("time_days",),
    "age_days": ("time_days",),
    "esh_ult": ("ultimate_shrinkage_microstrain",),
    "ultimate_shrinkage": ("ultimate_shrinkage_microstrain",),
    "wc": ("w_c_ratio",),
    "w_c": ("w_c_ratio",),
    "wc_ratio": ("w_c_ratio",),
    "water_cement": ("w_c_ratio",),
    "water_cement_ratio": ("w_c_ratio",),
    "wcr": ("w_c_ratio",),
    "temps": ("temperature_history_c",),
    "temperatures": ("temperature_history_c",),
    "temperature": ("temperature_history_c",),
    "temperature_history": ("temperature_history_c",),
    "hours": ("time_intervals_hours",),
    "intervals": ("time_intervals_hours",),
    "time_intervals": ("time_intervals_hours",),
    "dt": ("time_intervals_hours",),
    "length": ("length_m", "panel_length", "total_length_m"),
    "thickness": ("thickness_m", "wall_thickness", "raft_thickness"),
    "depth": ("depth_m", "excavation_depth"),
    "panel_l": ("panel_length",),
    "wall_t": ("wall_thickness",),
    "excavation_d": ("excavation_depth",),
    "bulking": ("bulking_factor",),
    "swell": ("bulking_factor",),
    "bulk": ("bulking_factor",),
    "cut": ("cut_volume_m3",),
    "cut_vol": ("cut_volume_m3",),
    "cut_volume": ("cut_volume_m3",),
    "fill": ("fill_volume_m3",),
    "fill_vol": ("fill_volume_m3",),
    "fill_volume": ("fill_volume_m3",),
    "permeability": ("soil_permeability_m_s",),
    "k": ("soil_permeability_m_s",),
    "k_m_s": ("soil_permeability_m_s",),
    "soil_k": ("soil_permeability_m_s",),
    "drawdown": ("required_drawdown_m",),
    "required_drawdown": ("required_drawdown_m",),
}

# Signature defaults that must NOT silently succeed (live wrong_number /
# tool_error). Any one name in a group satisfies the group.
_REQUIRED_GROUPS: Dict[str, Tuple[Tuple[str, ...], ...]] = {
    "calculate_evm": (
        ("pv", "bcws"),
        ("ev", "bcwp"),
        ("ac", "acwp"),
    ),
    # A flat daily amount stands in for both the rate and the contract
    # amount it would apply to.
    "delay_damages_daily": (
        ("rate_percent", "amount_per_day"),
        ("contract_amount", "amount_per_day"),
    ),
    "beam_shear_simple": (
        ("udl_w_kn_m", "central_point_load_kn"),
        ("span_m",),
    ),
    # A volume needs a plan dimension of some shape; with none it asks
    # rather than reporting 0 m3.
    "concrete_volume": (
        ("length_m", "diameter_m", "top_width_m"),
    ),
}

_PARAM_UNIT_OVERRIDE: Dict[str, str] = {
    "pv": "currency",
    "ev": "currency",
    "ac": "currency",
    "bcws": "currency",
    "bcwp": "currency",
    "acwp": "currency",
    "bac": "currency",
    "rate_percent": "%",
    "contract_amount": "currency",
    "water_depth": "m",
    "raft_thickness": "m",
    "floor_count": "floors",
    "floor_thickness": "m",
    "time_days": "d",
    "time_constant_days": "d",
    "ultimate_shrinkage_microstrain": "µε",
    "udl_w_kn_m": "kN/m",
    "central_point_load_kn": "kN",
    "quantity_m3": "m3",
    "w_c_ratio": "ratio",
    "floor_area_m2": "m2",
    "panel_length": "m",
    "wall_thickness": "m",
    "excavation_depth": "m",
    "total_cost": "currency",
    "area": "m2",
    "gross_valuation": "currency",
    "temperature_history_c": "°C",
    "time_intervals_hours": "h",
    "axial_load_kn": "kN",
    "column_diameter_mm": "mm",
}

def _snake_key(raw: Any) -> str:
    s = str(raw or "").strip().replace("-", "_").replace("/", "_")
    out: List[str] = []
    for i, ch in enumerate(s):
        if ch.isupper() and i and (s[i - 1].islower() or s[i - 1].isdigit()):
            out.append("_")
        out.append(ch.lower())
    return "".join(out).strip("_")

def _param_stem(name: str) -> str:
    n = _snake_key(name)
    changed = True
    while changed and n:
        changed = False
        for suf, _unit in _BIND_UNIT_SUFFIXES:
            if n.endswith(suf) and len(n) > len(suf):
                n = n[: -len(suf)]
                changed = True
                break
    return n.replace("_", "")

def _unit_from_name(name: str) -> str:
    if name in _PARAM_UNIT_OVERRIDE:
        return _PARAM_UNIT_OVERRIDE[name]
    n = _snake_key(name)
    for suf, unit in _BIND_UNIT_SUFFIXES:
        if n.endswith(suf):
            return unit
    return ""

def _ann_label(annotation: Any) -> str:
    if annotation is _inspect.Parameter.empty:
        return ""
    return getattr(annotation, "__name__", None) or str(annotation).replace("typing.", "")

def _is_junk_key(key: Any) -> bool:
    return _snake_key(key) in {_snake_key(j) for j in _BIND_JUNK_KEYS}

_POSITIONAL_KEY = "_positional"

_KV_ASSIGN_RE = re.compile(
    r"([A-Za-z_][\w]*(?:/[A-Za-z_][\w]*)*)\s*[:=]\s*"
    r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|"
    r"\"[^\"]*\"|'[^']*')",
)

_KV_JSON_START_RE = re.compile(
    r"([A-Za-z_][\w]*(?:/[A-Za-z_][\w]*)*)\s*[:=]\s*([\[{])",
)

_TONNE_INCOMING = frozenset({
    "quantity_t", "qty_t", "tonnes", "tons", "tonne", "ton",
})

def _percent_named_fraction(key: str, unit: Any) -> bool:
    """An input that takes a fraction while its name says percent."""
    words = _snake_key(key).split("_")
    return str(unit or "").strip().lower() == "fraction" and words[-1] in ("pct", "percent")

def _value_has_number(val: Any) -> bool:
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return True
    if isinstance(val, (list, tuple)):
        return any(_value_has_number(item) for item in val)
    if isinstance(val, dict):
        return any(_value_has_number(item) for item in val.values())
    return False

def _parse_kv_assignments(text: str) -> Dict[str, Any]:
    """Parse ``udl_w_kn_m=20, span_m=6`` / ``temps=[20, 22, 25]``.

    Live standing-exit probe prints and often *sends* params as assignment
    strings, not JSON objects. Only keep a parse that yielded a number so
    ``status: error`` prose is not treated as kwargs.
    """
    if not text or ("=" not in text and ":" not in text):
        return {}
    out: Dict[str, Any] = {}
    consumed: List[Tuple[int, int]] = []

    def _overlaps(span: Tuple[int, int]) -> bool:
        return any(span[0] < c[1] and c[0] < span[1] for c in consumed)

    decoder = json.JSONDecoder()
    for match in _KV_JSON_START_RE.finditer(text):
        if _overlaps(match.span()):
            continue
        try:
            obj, end = decoder.raw_decode(text, match.end() - 1)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        out[match.group(1)] = obj
        consumed.append((match.start(), end))

    for match in _KV_ASSIGN_RE.finditer(text):
        if _overlaps(match.span()):
            continue
        raw = match.group(2).strip().strip("\"'")
        number = raw.replace(",", "").strip()
        unit = re.match(r"^([+-]?\d+(?:\.\d+)?)(?:\s*[A-Za-zµμ/%²³³°]+)?$", number)
        if unit:
            token = unit.group(1)
            out[match.group(1)] = float(token) if "." in token else int(token)
        else:
            out[match.group(1)] = raw
        consumed.append(match.span())
    if not any(_value_has_number(val) for val in out.values()):
        return {}
    return out

def coerce_calc_params(raw: Any) -> Dict[str, Any]:
    """Accept a dict, JSON object/array, or ``k=v, k=v`` assignment string.

    A JSON array becomes ``{_positional: [...]}`` so run_calculation can
    zip it onto the signature (beam_shear [20, 6], tenderer list, maturity
    pair-of-lists). Never invent keys. Empty / undecodable is ``{}``.
    """
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, (list, tuple)):
        return {_POSITIONAL_KEY: list(raw)}
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        if text[0] in "{[":
            try:
                obj = json.loads(text)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                logger.debug("coerce_calc_params: undecodable params string (%s)", exc)
                obj = None
            if isinstance(obj, dict):
                return dict(obj)
            if isinstance(obj, list):
                return {_POSITIONAL_KEY: obj}
        parsed = _parse_kv_assignments(text)
        if parsed:
            return parsed
        return {}
    return {}

def _length_unit_factor(incoming: str, dest: str) -> Optional[float]:
    """Factor to convert ``incoming``'s unit to ``dest``'s, or None.

    Only when the two keys are the SAME quantity in different units --
    ``span_m`` onto ``span_mm``, ``load_kn`` onto ``load_n`` -- so nothing
    is converted across quantities. The factor is the base conversion tool's.
    """
    if "_" not in incoming or "_" not in dest:
        return None
    inc_stem, inc_unit = incoming.rsplit("_", 1)
    dest_stem, dest_unit = dest.rsplit("_", 1)
    if inc_stem != dest_stem or inc_unit == dest_unit:
        return None
    src, dst = _parse_unit(inc_unit), _parse_unit(dest_unit)
    if (src is None or dst is None or src.dimension != dst.dimension or src.needs or dst.needs
            or src.offset or dst.offset):
        return None
    out = _convert_units(1.0, inc_unit, dest_unit)
    return out.get("value_out")

def _scale_bound_value(incoming: str, dest: str, val: Any) -> Any:
    """quantity_t / tonnes → quantity_kg, and span_m → span_mm. Never invents a
    value that was absent.

    The length case is a live slab-thickness ask: the model called slab_thickness_min with
    ``span_m: 4.8`` against a ``span_mm`` parameter. The value bound unchanged,
    so 4.8 metres was read as 4.8 millimetres and the calculator returned
    ``min_thickness_mm: 0.2`` with status success. The answer then reported
    "200 mm" -- 0.2 read back as metres -- while its own working said
    "L/20 = 4800/20". A silently wrong unit is worse than a rejected call.
    """
    inc = _snake_key(incoming)
    if dest == "quantity_kg" and (
        inc in _TONNE_INCOMING or inc.endswith("_t")
    ):
        try:
            return float(val) * 1000.0
        except (TypeError, ValueError):
            logger.debug("quantity_t token %r is not numeric", val)
            return val
    factor = _length_unit_factor(inc, _snake_key(dest))
    if factor is not None:
        try:
            scaled = float(val) * factor
        except (TypeError, ValueError):
            logger.debug("unit-suffixed token %r is not numeric", val)
            return val
        logger.info("bind: converted %s=%s to %s=%s", inc, val, dest, scaled)
        return scaled
    return val

def _bind_sequence_params(
    fn: Any, values: List[Any], name: Optional[str] = None,
) -> Dict[str, Any]:
    """Zip a JSON/Python array onto this calculator's signature.

    List-of-dicts → unique list param (evaluate_tender tenderers).
    List-of-lists → list-typed params in signature order (maturity).
    Scalars → required-then-optional positional names (beam_shear [20, 6]).
    """
    seq = list(values or [])
    if not seq or fn is None:
        return {}
    try:
        sig = _inspect.signature(fn)
    except (TypeError, ValueError):
        logger.debug("positional bind: calculator has no inspectable signature")
        return {}
    dests = [
        key for key, param in sig.parameters.items()
        if param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
        and not key.startswith("_")
    ]
    list_dests = [
        key for key in dests
        if _annotation_wants_list(sig.parameters[key].annotation)
    ]
    if dests and all(isinstance(item, dict) for item in seq):
        calc = str(name or "").strip().lower()
        if calc == "evaluate_tender" or "tenderers" in list_dests:
            return {"tenderers": seq}
        if len(list_dests) == 1:
            return {list_dests[0]: seq}
        return {}
    if (
        list_dests
        and len(seq) == len(list_dests)
        and all(isinstance(item, (list, tuple)) for item in seq)
    ):
        return {key: list(item) for key, item in zip(list_dests, seq)}
    required = [
        key for key in dests
        if sig.parameters[key].default is sig.parameters[key].empty
    ]
    order = required + [key for key in dests if key not in required]
    return {key: val for key, val in zip(order, seq)}

def _pop_positional(params: Dict[str, Any]) -> Optional[List[Any]]:
    if not isinstance(params, dict):
        return None
    raw = params.pop(_POSITIONAL_KEY, None)
    if isinstance(raw, (list, tuple)):
        return list(raw)
    return None

def _flatten_calc_kwargs(params: Dict[str, Any]) -> Dict[str, Any]:
    """Merge nested ``params`` / ``input`` into top-level calculator kwargs.

    Live Phase-2 / #636 / DIR7 / #652: the model often puts calculator kwargs
    next to ``calculation`` *or* nests them under ``params`` or ``input``.
    Envelope keys (text, formula, input, project_id, …) are never copied
    through as calculator kwargs. ``text`` / ``formula`` are the E4
    exception — they survive flatten for the resolvers, then bind drops
    them. Nested keys lose to an explicit top-level of the same name.
    """
    if not isinstance(params, dict):
        return {}
    junk = {_snake_key(j) for j in _BIND_JUNK_KEYS}
    e4 = {_snake_key(j) for j in _E4_PASSTHROUGH_KEYS}
    out: Dict[str, Any] = {}
    for nest_key in _FLATTEN_NEST_KEYS:
        nested = coerce_calc_params(params.get(nest_key))
        if not nested:
            continue
        for key, val in nested.items():
            if val is None or val == "":
                continue
            if _snake_key(key) in junk:
                continue
            out[key] = val
    for key, val in params.items():
        if val is None or val == "":
            continue
        nk = _snake_key(key)
        if nk in junk and nk not in e4:
            continue
        out[key] = val
    return out

def _formula_spec(fn: Any) -> Any:
    """The registry spec for this calculator, when it is registered."""
    try:
        from app.lib import formula_registry

        for spec in formula_registry.all_specs():
            if spec.fn is fn or getattr(fn, "__wrapped__", None) is spec.fn:
                return spec
    except Exception:  # noqa: BLE001 -- the name suffix is the fallback
        logger.debug("formula registry unavailable for unit lookup", exc_info=True)
    return None


def _declared_units(fn: Any) -> Dict[str, str]:
    """Input units the formula declares where it is defined (@formula
    inputs), when it is a registered formula; '-' means unitless."""
    spec = _formula_spec(fn)
    if spec is None:
        return {}
    return {k: v for k, v in (spec.inputs or {}).items() if v and v != "-"}


def describe_calculation_params(
    fn: Any,
    name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Signature-derived expected params with units (name suffix / override)."""
    sig = _inspect.signature(fn)
    declared = _declared_units(fn)  # for parameters whose name carries no unit
    required_names: set[str] = set()
    if name:
        for group in _REQUIRED_GROUPS.get(str(name).strip().lower(), ()):
            required_names.update(group)
    rows: List[Dict[str, Any]] = []
    for key, param in sig.parameters.items():
        if param.kind not in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
            continue
        if key.startswith("_"):
            continue
        required = param.default is param.empty or key in required_names
        # A declared money unit ("currency/t") wins over a name suffix
        # ("_per_t" alone would read as tonnes).
        if str(declared.get(key, "")).startswith("currency"):
            unit = declared[key]
        else:
            unit = _unit_from_name(key) or declared.get(key, "")
        row: Dict[str, Any] = {
            "name": key,
            "required": required,
            "unit": unit,
            "annotation": _ann_label(param.annotation),
        }
        if param.default is not param.empty:
            row["default"] = param.default
        rows.append(row)
    return rows

def _unique_semantic_dest(incoming: str, accepted: Dict[str, str]) -> Optional[str]:
    hits = []
    for cand in _BIND_SEMANTIC_ALIASES.get(incoming, ()):
        dest = accepted.get(_snake_key(cand))
        if dest and dest not in hits:
            hits.append(dest)
    if len(hits) == 1:
        return hits[0]
    return None

def _unique_stem_dest(
    incoming: str,
    accepted: Dict[str, str],
    required: Optional[set] = None,
) -> Optional[str]:
    """Bind volume → volume_m3 / diameter → column_diameter_mm / water_depth_m → water_depth when unique.

    Suffix matches (sand → dune_sand_pct, cement → cement_sg) only land on
    *required* dests. Optional dests need an explicit name or semantic alias
    so a mix-design table SG cannot raise dune_sand_pct.
    """
    stem = _param_stem(incoming)
    if len(stem) < 3:
        return None
    exact = [
        dest for dest in accepted.values()
        if _param_stem(dest) == stem and dest not in _EXPLICIT_BIND_ONLY
    ]
    if len(exact) == 1:
        return exact[0]
    if exact:
        return None
    suffix = [
        dest for dest in accepted.values()
        if dest not in _EXPLICIT_BIND_ONLY
        and _param_stem(dest).endswith(stem)
        and len(_param_stem(dest)) > len(stem)
    ]
    if required is not None:
        suffix = [dest for dest in suffix if dest in required]
    if len(suffix) == 1:
        return suffix[0]
    return None

def _ignored_explicit_stem(incoming: str, accepted: Dict[str, str]) -> bool:
    """True when the only stem hit is an explicit-bind-only parameter.

    ``sand`` must not become ``dune_sand_pct`` (live mix-table SGs). It is
    also not an unknown argument: the call still computes from the keys
    that did bind. ``bogus_param`` matches nothing here and stays unknown.
    """
    stem = _param_stem(incoming)
    if len(stem) < 3:
        return False
    hits = []
    for dest in accepted.values():
        dest_stem = _param_stem(dest)
        if dest_stem == stem or (
            dest_stem.endswith(stem) and len(dest_stem) > len(stem)
        ):
            hits.append(dest)
    return bool(hits) and all(dest in _EXPLICIT_BIND_ONLY for dest in hits)

def _partition_bound_params(
    fn: Any, params: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """Map kwargs onto ``fn``. The list is unbound keys, in input order.

    A ``**kwargs`` calculator absorbs unbound keys. ``concrete_mix_slip_form``
    is the only one: extra kwargs are ignored so a probe that sends slump
    next to the name still returns the locked mix. Every other calculator
    leaves the key unbound so ``run_calculation`` can return a tool error
    instead of dropping it.
    """
    raw = _flatten_calc_kwargs(params or {})
    sig = _inspect.signature(fn)
    accepted_list = [
        key for key, param in sig.parameters.items()
        if param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
    ]
    accepted_norm = {_snake_key(k): k for k in accepted_list}
    has_var_kw = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
    required = {
        key for key, param in sig.parameters.items()
        if param.default is param.empty
        and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
    }

    bound: Dict[str, Any] = {}
    unknown: List[str] = []
    for key, val in raw.items():
        if val is None or val == "":
            continue
        if _is_junk_key(key):
            continue
        dest = accepted_norm.get(_snake_key(key))
        if dest in _EXPLICIT_BIND_ONLY and _snake_key(key) != _snake_key(dest):
            dest = None
        if dest is None:
            dest = _unique_semantic_dest(_snake_key(key), accepted_norm)
        if dest is None:
            dest = _unique_stem_dest(key, accepted_norm, required=required)
        if dest is None and _ignored_explicit_stem(key, accepted_norm):
            continue
        if dest is None:
            unknown.append(str(key))
            continue
        if isinstance(val, (list, tuple)) and len(val) == 1:
            val = val[0]
        scaled = _scale_bound_value(str(key), dest, val)
        if dest not in bound or bound[dest] in (None, ""):
            bound[dest] = scaled

    if has_var_kw:
        for key in unknown:
            bound.setdefault(key, raw.get(key))
        return bound, []
    return bound, unknown

def bind_calculation_params(fn: Any, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Map incoming kwargs onto ``fn``'s signature names.

    Shared with Agent C (#636 flattened the tool path + EVM aliases). This
    generalises that bind: case-insensitive keys, unit-suffix stems, and a
    small synonym table. Canonical names already present win.
    Shared with Agent C / #636 / #639 / #652.
    """
    # E / I / c onto ec_mpa / i_mm4 / code, HERE rather than at a call site:
    # the live tool resolves the calculator after run_calculation's early hook
    # ran with fn=None, so the alias was skipped and construction_calc looped
    # twelve times on the same unknown-argument envelope (beam deflection and
    # concrete modulus asks on 533f08c).
    # Every entry point passes through this function.
    params = _alias_physics_symbols(fn, params or {})
    bound, _unknown = _partition_bound_params(fn, params)
    return bound

_ASK_NUM = rf"({SIGNED_FIGURE})"

_NUM_UNIT_VALUE_RE = re.compile(
    r"^\s*([+\-]?\s*[\d,]+(?:\.\d+)?)\s*[A-Za-zµμ/%²³³°]+"
)

_PLAIN_NUM_RE = re.compile(r"^[+\-]?\s*[\d,]+(?:\.\d+)?\s*$")

_LWD_WORDS_RE = re.compile(
    rf"{_ASK_NUM}\s*m?\s*long\b.*?{_ASK_NUM}\s*m?\s*wide\b.*?"
    rf"{_ASK_NUM}\s*m?\s*(?:deep|thick)",
    re.IGNORECASE | re.DOTALL,
)

_FORMULA_TRIPLE_RE = re.compile(
    rf"{_ASK_NUM}\s*[*×x]\s*{_ASK_NUM}\s*[*×x]\s*{_ASK_NUM}",
    re.IGNORECASE,
)

def _ask_float(raw: str) -> float:
    return float(str(raw).replace(",", ""))

def _ask_present(params: Dict[str, Any], *keys: str) -> bool:
    return any(params.get(k) not in (None, "") for k in keys)

#: The user's own words for this turn, set where the turn starts
#: (app.agents.core.loop). Every calculator run reads its units from them, by
#: whatever path the call arrived: a model's paraphrase in ``text`` can drop
#: the unit the user wrote ("21,528 sq ft"), the user's turn cannot.
_CURRENT_ASK: contextvars.ContextVar[str] = contextvars.ContextVar("calc_current_ask", default="")


def set_current_ask(text: str):
    """This turn's user words for every calculator run in this context."""
    return _CURRENT_ASK.set(str(text or ""))


def _with_current_ask(text: str) -> str:
    ask = _CURRENT_ASK.get()
    if not ask or ask in (text or ""):
        return text or ""
    return f"{text} {ask}".strip()


def _ask_blob(params: Dict[str, Any]) -> str:
    return " ".join(
        str(x) for x in (
            params.get("text"),
            params.get("formula"),
            params.get("message"),
            params.get("query"),
        ) if x
    )

def _fill_lwd_from_text(text: str, out: Dict[str, Any], depth_key: str = "depth_m") -> None:
    """Bank / trench L×W×D from the ask. Does not invent numbers."""
    if _ask_present(out, "length_m") and _ask_present(out, "width_m") and _ask_present(
        out, depth_key, "height_m", "thickness_m",
    ):
        return
    dims = None
    try:
        from app.lib.construction_formulas_quantities import parse_lwt_metres
        dims = parse_lwt_metres(text)
    except (TypeError, ValueError, ImportError):
        dims = None
    if dims:
        out.setdefault("length_m", dims[0])
        out.setdefault("width_m", dims[1])
        out.setdefault(depth_key, dims[2])
        return
    match = _LWD_WORDS_RE.search(text or "")
    if match:
        out.setdefault("length_m", _ask_float(match.group(1)))
        out.setdefault("width_m", _ask_float(match.group(2)))
        out.setdefault(depth_key, _ask_float(match.group(3)))
        return
    match = _FORMULA_TRIPLE_RE.search(text or "")
    if match:
        out.setdefault("length_m", _ask_float(match.group(1)))
        out.setdefault("width_m", _ask_float(match.group(2)))
        out.setdefault(depth_key, _ask_float(match.group(3)))

def _extract_slab_thickness_from_ask(text: str, out: Dict[str, Any]) -> None:
    """Fill slab_thickness_min from 'spanning 4.8 m' / one-end continuous.

    The binder's span label does not see 'spanning'. Holes only — an
    explicit span_mm on the call wins.
    """
    try:
        from app.lib.construction_formulas_structural_rc import (
            slab_thickness_params_from_ask,
        )
    except Exception:  # noqa: BLE001 — extract must not break other calculators
        logger.exception("slab thickness ask extract unavailable")
        return
    found = slab_thickness_params_from_ask(text or "")
    for key, val in found.items():
        if out.get(key) in (None, ""):
            out[key] = val

def _extract_beam_shear_from_ask(text: str, out: Dict[str, Any]) -> None:
    if not _ask_present(out, "udl_w_kn_m"):
        match = re.search(
            rf"(?:udl(?:_w)?|w)\s*[=:]?\s*{_ASK_NUM}\s*(?:kN\s*/\s*m|kn/m)?",
            text, re.IGNORECASE,
        )
        if match is None:
            match = re.search(rf"{_ASK_NUM}\s*kN\s*/\s*m", text, re.IGNORECASE)
        if match:
            out["udl_w_kn_m"] = _ask_float(match.group(1))
    if not _ask_present(out, "span_m"):
        match = re.search(
            rf"(?:span|l)\s*[=:]?\s*{_ASK_NUM}\s*m?\b",
            text, re.IGNORECASE,
        )
        if match:
            out["span_m"] = _ask_float(match.group(1))

def _extract_dewatering_from_ask(text: str, out: Dict[str, Any]) -> None:
    if not _ask_present(out, "water_depth"):
        match = re.search(
            rf"{_ASK_NUM}\s*m(?:etre)?s?\s+(?:of\s+)?(?:water|head|groundwater|uplift)",
            text, re.IGNORECASE,
        )
        if match is None:
            match = re.search(
                rf"(?:water|head|hw|groundwater)(?:\s+depth)?(?:\s+of)?\s*[:=]?\s*{_ASK_NUM}",
                text, re.IGNORECASE,
            )
        if match:
            out["water_depth"] = _ask_float(match.group(1))
    if not _ask_present(out, "raft_thickness"):
        match = re.search(
            rf"{_ASK_NUM}\s*m(?:etre)?s?\s+(?:thick\s+)?raft",
            text, re.IGNORECASE,
        )
        if match is None:
            match = re.search(
                rf"raft(?:\s+thickness)?(?:\s+of)?\s*[:=]?\s*{_ASK_NUM}",
                text, re.IGNORECASE,
            )
        if match:
            out["raft_thickness"] = _ask_float(match.group(1))
    if not _ask_present(out, "floor_count"):
        match = re.search(
            rf"{_ASK_NUM}\s+(?:floors?|storeys?|stories|levels)\b",
            text, re.IGNORECASE,
        )
        if match:
            out["floor_count"] = int(_ask_float(match.group(1)))

def _extract_carbon_from_ask(text: str, out: Dict[str, Any]) -> None:
    if not _ask_present(out, "volume_m3"):
        match = re.search(rf"{_ASK_NUM}\s*m\s*3\b", text, re.IGNORECASE)
        if match is None:
            match = re.search(rf"{_ASK_NUM}\s*m³", text, re.IGNORECASE)
        if match:
            out["volume_m3"] = _ask_float(match.group(1))

def _extract_mix_from_ask(text: str, out: Dict[str, Any]) -> None:
    if _ask_present(out, "w_c_ratio"):
        return
    match = re.search(
        rf"(?:w\s*/\s*c|w[_/]?c(?:\s+ratio)?)\s*[:=]?\s*{_ASK_NUM}",
        text, re.IGNORECASE,
    )
    if match:
        out["w_c_ratio"] = _ask_float(match.group(1))

def _csv_floats(raw: str) -> List[float]:
    return [_ask_float(tok) for tok in re.findall(_ASK_NUM, raw or "")]

def _extract_maturity_from_ask(text: str, out: Dict[str, Any]) -> None:
    """CSV temps / hours from the ask. Does not invent a missing series."""
    if not _ask_present(out, "temperature_history_c"):
        match = re.search(
            rf"((?:{_ASK_NUM}\s*,\s*){{1,}}{_ASK_NUM})\s*"
            rf"(?:°\s*C|deg(?:rees)?(?:\s*C)?|\bC\b)",
            text, re.IGNORECASE,
        )
        if match is None:
            match = re.search(
                rf"(?:temps?|temperatures?|temperature_history(?:_c)?)\s*[:=]?\s*\[?\s*"
                rf"((?:{_ASK_NUM}\s*,\s*){{1,}}{_ASK_NUM})\s*\]?",
                text, re.IGNORECASE,
            )
        if match:
            nums = _csv_floats(match.group(1))
            if nums:
                out["temperature_history_c"] = nums
    if not _ask_present(out, "time_intervals_hours"):
        match = re.search(
            rf"((?:{_ASK_NUM}\s*,\s*){{1,}}{_ASK_NUM})\s*"
            rf"(?:hours?|hrs?|h)\b",
            text, re.IGNORECASE,
        )
        if match is None:
            match = re.search(
                rf"(?:hours?|intervals?|time_intervals(?:_hours)?|dt)\s*[:=]?\s*\[?\s*"
                rf"((?:{_ASK_NUM}\s*,\s*){{1,}}{_ASK_NUM})\s*\]?",
                text, re.IGNORECASE,
            )
        if match:
            nums = _csv_floats(match.group(1))
            if nums:
                out["time_intervals_hours"] = nums

def _extract_rebar_cost_from_ask(text: str, out: Dict[str, Any]) -> None:
    if not _ask_present(out, "quantity_kg"):
        match = re.search(rf"{_ASK_NUM}\s*kg\b", text, re.IGNORECASE)
        if match:
            out["quantity_kg"] = _ask_float(match.group(1))
        else:
            match = re.search(
                rf"{_ASK_NUM}\s*(?:tonnes?|tons?|t)\b",
                text, re.IGNORECASE,
            )
            if match:
                out["quantity_kg"] = _ask_float(match.group(1)) * 1000.0
    if not _ask_present(out, "material_price_per_t"):
        match = re.search(
            rf"{_ASK_NUM}\s*(?:SAR|USD|AED)?\s*/\s*t(?:onne)?s?\b",
            text, re.IGNORECASE,
        )
        if match:
            out["material_price_per_t"] = _ask_float(match.group(1))

_TENDER_ROW_RE = re.compile(
    r"(Bidder\s+[A-Za-z0-9]+|[A-Za-z][\w.\-]{1,30})\s*[—\-:]\s*"
    r"technical\s+(\d+(?:\.\d+)?)\s*,\s*commercial\s+(\d+(?:\.\d+)?)\s*,\s*"
    r"HSE\s+(\d+(?:\.\d+)?)(?:\s*,\s*local(?:\s+content)?\s+(\d+(?:\.\d+)?))?",
    re.IGNORECASE,
)

def _extract_tender_from_ask(text: str, out: Dict[str, Any]) -> None:
    if _ask_present(out, "tenderers", "bidders", "bids"):
        return
    rows: List[Dict[str, Any]] = []
    for match in _TENDER_ROW_RE.finditer(text or ""):
        local = match.group(5)
        rows.append({
            "name": match.group(1).strip(),
            "technical_score": _ask_float(match.group(2)),
            "commercial_score": _ask_float(match.group(3)),
            "hse_score": _ask_float(match.group(4)),
            "local_content_score": _ask_float(local) if local is not None else 0,
        })
    if rows:
        out["tenderers"] = rows

def _extract_calc_kwargs_from_ask(
    name: str, params: Dict[str, Any],
) -> Dict[str, Any]:
    """Fill missing kwargs from ``text`` / ``formula``. Never invent defaults."""
    out = dict(params or {})
    blob = _ask_blob(out)
    if not blob.strip():
        return out
    # Live predispatch often ships only {text: ask}. Assignment strings
    # (temps=[20,22,25], w/c=0.48, cut=5000) must become kwargs here —
    # coerce_calc_params only sees the params object, not the ask blob.
    for key, val in _parse_kv_assignments(blob).items():
        if out.get(key) in (None, ""):
            out[key] = val
    calc = str(name or "").strip().lower()
    if calc == "beam_shear_simple":
        _extract_beam_shear_from_ask(blob, out)
    elif calc == "dewatering_uplift_check":
        _extract_dewatering_from_ask(blob, out)
    elif calc == "excavation_volume":
        _fill_lwd_from_text(blob, out, "depth_m")
    elif calc == "concrete_volume":
        _fill_lwd_from_text(blob, out, "thickness_m")
    elif calc == "carbon_footprint_concrete":
        _extract_carbon_from_ask(blob, out)
    elif calc == "concrete_mix_design_sg":
        _extract_mix_from_ask(blob, out)
    elif calc == "concrete_maturity_strength":
        _extract_maturity_from_ask(blob, out)
    elif calc == "cost_buildup_rebar":
        _extract_rebar_cost_from_ask(blob, out)
    elif calc == "evaluate_tender":
        _extract_tender_from_ask(blob, out)
    elif calc == "slab_thickness_min":
        _extract_slab_thickness_from_ask(blob, out)
    elif calc == "diaphragm_wall_panel_volume":
        _fill_lwd_from_text(blob, out, "excavation_depth")
        if _ask_present(out, "length_m") and not _ask_present(out, "panel_length"):
            out["panel_length"] = out["length_m"]
        if _ask_present(out, "thickness_m") and not _ask_present(out, "wall_thickness"):
            out["wall_thickness"] = out["thickness_m"]
        elif _ask_present(out, "width_m") and not _ask_present(out, "wall_thickness"):
            out["wall_thickness"] = out["width_m"]
        if _ask_present(out, "depth_m") and not _ask_present(out, "excavation_depth"):
            out["excavation_depth"] = out["depth_m"]
    return out

def _annotation_wants_list(annotation: Any) -> bool:
    if annotation is _inspect.Parameter.empty:
        return False
    if isinstance(annotation, str):
        low = annotation.replace(" ", "").lower()
        return low.startswith("list[") or low == "list"
    origin = getattr(annotation, "__origin__", None)
    return origin in (list, List)

def _coerce_scalar(val: Any) -> Any:
    if isinstance(val, bool) or val is None or isinstance(val, (int, float)):
        return val
    if not isinstance(val, str):
        return val
    text = val.strip()
    if not text:
        return val
    if _PLAIN_NUM_RE.match(text):
        number = text.replace(",", "").replace(" ", "")
        return float(number) if "." in number else int(number)
    match = _NUM_UNIT_VALUE_RE.match(text)
    if match:
        number = match.group(1).replace(",", "").replace(" ", "")
        return float(number) if "." in number else int(number)
    return val

def _coerce_list(val: Any) -> Any:
    if isinstance(val, (list, tuple)):
        return [_coerce_scalar(item) for item in val]
    if isinstance(val, str):
        text = val.strip()
        if text.startswith("["):
            try:
                obj = json.loads(text)
            except (TypeError, ValueError, json.JSONDecodeError):
                obj = None
            if isinstance(obj, list):
                return [_coerce_scalar(item) for item in obj]
        parts = [part for part in re.split(r"[,\s]+", text) if part]
        if parts and all(
            _PLAIN_NUM_RE.match(part) or _NUM_UNIT_VALUE_RE.match(part)
            for part in parts
        ):
            return [_coerce_scalar(part) for part in parts]
    coerced = _coerce_scalar(val)
    if isinstance(coerced, (int, float)):
        return [coerced]
    return val

def _stated_percents(text: Any) -> frozenset:
    """Every figure the user's words write as a percentage ("150%",
    "12.5 %"), as numbers."""
    found = set()
    for match in re.finditer(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s*(?:%|per\s*cent\b|percent\b)", str(text or ""), re.I):
        try:
            found.add(round(float(match.group(1).replace(",", "")), 9))
        except ValueError:
            continue
    return frozenset(found)

def _coerce_bound_values(fn: Any, bound: Dict[str, Any], converted: frozenset = frozenset(),
                         stated_percents: frozenset = frozenset()) -> Dict[str, Any]:
    """Strip unit suffixes from numeric strings; wrap list-typed scalars.
    ``converted``: inputs the unit converter already put in their declared
    unit (150% already became 1.5); they are not re-read as percentages.
    ``stated_percents``: the figures the user wrote as percentages; a value
    that is one of them divided by 100 is already a fraction, whatever its
    declared range, and is not divided again."""
    try:
        sig = _inspect.signature(fn)
    except (TypeError, ValueError):
        return bound
    out = dict(bound)
    spec = _formula_spec(fn)
    declared = dict(spec.inputs or {}) if spec else {}
    for key, val in list(out.items()):
        param = sig.parameters.get(key)
        if param is None:
            continue
        if _annotation_wants_list(param.annotation):
            out[key] = _coerce_list(val)
            continue
        if isinstance(val, (list, tuple)) and len(val) == 1:
            val = val[0]
            out[key] = val
        ann = param.annotation
        ann_s = ann if isinstance(ann, str) else getattr(ann, "__name__", str(ann))
        if str(ann_s).lower() in ("str", "string"):
            continue
        if isinstance(val, str):
            out[key] = _coerce_scalar(val)
        # Live excavation ask-2 sent bulking=25 (percent) not 0.25.
        # Values in (1, 100] are percents; 1.25 stays a multiplier. An input
        # declared as a fraction whose name says it is a percentage
        # (waste_pct, markup_pct) reads a bare 10 as 10% the same way.
        if key in converted:
            continue
        if key in {"bulking_factor", "swell_factor"} or _percent_named_fraction(key, declared.get(key)):
            number = out[key]
            # A figure already inside the input's declared range is taken as
            # given (150% bound as 1.5 on a 0..2 fraction is not divided again).
            declared_param = spec.params.get(key) if spec else None
            hi = getattr(declared_param, "hi", None)
            in_range = hi is not None and isinstance(number, (int, float)) and float(number) <= float(hi)
            already_fraction = isinstance(number, (int, float)) and round(float(number) * 100.0, 9) in stated_percents
            if isinstance(number, (int, float)) and 1.0 < float(number) <= 100.0 and not in_range                     and not already_fraction:
                out[key] = float(number) / 100.0
    return out

# Thousand separators must be interior (1,000) — a trailing comma is list
# punctuation ("100, man_hours 50") and must not be eaten as part of the number.
# A figure: 1,250 / 0.2 / 5.4e9 / 5.4 x 10^9 / 5.4×10^9 (scientific forms
# as engineers write them).
_TEXT_NUM_RE = (r"[+-]?\d+(?:,\d{3})*(?:\.\d+)?"
                r"(?:\s*[x×*]\s*10\s*\^\s*[+-]?\d+|[eE][+-]?\d+)?")

# Any unit the base unit-conversion tool reads, written after a figure.
_TEXT_UNIT_RE = _text_unit_pattern()

_CODE_ACI_RE = re.compile(r"\b(?:aci|asce|aisc|tms)\b", re.IGNORECASE)

_CODE_EC_RE = re.compile(r"\b(?:eurocode|en\s*199\d)\b", re.IGNORECASE)

_UNIT_EQ = {
    "mpa": "mpa",
    "n/mm2": "mpa",
    "kpa": "kpa",
    "kn/m2": "kn/m2",
    "m2": "m2",
    "mm2": "mm2",
    "m3": "m3",
    "m": "m",
    "mm": "mm",
    "kn": "kn",
    "m/s": "m/s",
    "%": "%",
    "deg": "deg",
}

def _norm_label(raw: Any) -> str:
    return _snake_key(str(raw or "").replace(" ", "_").replace("/", "_"))

def _norm_unit_token(raw: Any) -> str:
    s = str(raw or "").strip().lower().replace("²", "2").replace("³", "3")
    s = s.replace(" ", "").replace(".", "")
    return _UNIT_EQ.get(s, s)


# A declared unit that is not written next to the figure. A bare number
# can bind to one of these, and only when it is the only one still missing.
_BARE_KINDS = frozenset({"", "-", "currency", "ratio", "count", "no", "nr"})


def _same_kind(written: str, declared: str) -> bool:
    """A written unit measures what the declared unit measures (kg and t
    are both a mass)."""
    try:
        from app.agents.base.formulas.construction_formulas_planning import parse_unit

        a, b = parse_unit(written), parse_unit(declared)
        if a is not None and b is not None:
            return a.dimension == b.dimension
    except Exception:  # noqa: BLE001 -- fall back to the token rule
        logger.debug("unit dimension lookup failed", exc_info=True)
    return binding_kind(written) == binding_kind(declared)


def binding_kind(unit: str) -> str:
    """How a figure has to be written to identify this declared unit.

    Empty means the unit has no token of its own, so the figure is a bare
    number. Callers that see two inputs of one kind bind neither.
    """
    norm = _norm_unit_token(unit)
    if norm in _BARE_KINDS:
        return ""
    return norm


def _figure_unit_pattern(extra: Iterable[str] = ()) -> str:
    """Units a figure may carry: every unit the conversion tool reads, plus
    this formula's own declared tokens."""
    tokens = []
    for token in extra:
        text = str(token or "").strip()
        if text and binding_kind(text) and text not in tokens:
            tokens.append(text)
    tokens.sort(key=len, reverse=True)
    own = "|".join(re.escape(token) for token in tokens)
    return f"(?:{_TEXT_UNIT_RE}" + (f"|(?:{own})(?![A-Za-z0-9])" if own else "") + ")"

def _parse_text_number(raw: str) -> Optional[float]:
    text = str(raw).replace(",", "").strip()
    sci = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)\s*[x×*]\s*10\s*\^\s*([+-]?\d+)", text)
    if sci:
        return float(sci.group(1)) * 10 ** int(sci.group(2))
    try:
        return float(text)
    except (TypeError, ValueError):
        logger.debug("engineer-ask token %r is not numeric", raw)
        return None

def _text_label_map(fn: Any) -> Dict[str, str]:
    """Normalized ask labels → unique signature dest for this calculator."""
    sig = _inspect.signature(fn)
    accepted_list = [
        key for key, param in sig.parameters.items()
        if param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
        and not key.startswith("_")
    ]
    accepted_norm = {_snake_key(k): k for k in accepted_list}
    out: Dict[str, str] = {}
    ambiguous: set[str] = set()

    def add(label: str, dest: str) -> None:
        key = _norm_label(label)
        if not key or key in ambiguous:
            return
        prev = out.get(key)
        if prev is None:
            out[key] = dest
        elif prev != dest:
            out.pop(key, None)
            ambiguous.add(key)

    for dest in accepted_list:
        add(dest, dest)
        add(dest.replace("_", " "), dest)
        if dest in _EXPLICIT_BIND_ONLY:
            continue
        stem = dest
        for suf, _unit in _BIND_UNIT_SUFFIXES:
            if stem.endswith(suf) and len(stem) > len(suf):
                stem = stem[: -len(suf)]
                add(stem, dest)
                break
        add(stem.replace("_", " "), dest)

    for incoming, dests in _BIND_SEMANTIC_ALIASES.items():
        dest = _unique_semantic_dest(_snake_key(incoming), accepted_norm)
        if dest:
            add(incoming, dest)
    # The registry's declared labels are the words a user reads: each names
    # its own input, over any word built from a signature name.
    spec = _formula_spec(fn)
    for dest, param in (spec.params.items() if spec else ()):
        key = _norm_label(param.label or "")
        if key and dest in accepted_list:
            out[key] = dest
            ambiguous.discard(key)
    return out

# English dimension adjectives written after the figure ("12 m long",
# "200 mm thick") and the dimension noun each names.
_DIMENSION_ADJECTIVES: Dict[str, str] = {
    "long": "length", "in length": "length",
    "wide": "width", "in width": "width", "broad": "width",
    "thick": "thickness", "in thickness": "thickness",
    "deep": "depth", "in depth": "depth",
    "high": "height", "tall": "height", "in height": "height",
    "in diameter": "diameter", "diameter": "diameter",
}


def _to_param_unit(num: float, unit: Optional[str], dest: str, declared: str = "") -> Any:
    """``num`` written in ``unit``, put into the unit ``dest`` declares (its
    declaration, else the unit its name spells) by the base conversion tool.

    A figure the tool cannot put into that unit on its own -- another kind
    of quantity, or labour that needs a crew size -- keeps its unit written
    ("5 kN"), so the input check converts it with what the call supplies, or
    asks. An input with no unit of its own takes the figure as written.
    """
    written = (unit or "").strip()
    target = (declared or "").strip() or _unit_from_name(dest)
    if not written or not target or target == "-" or _parse_unit(target) is None:
        return num
    out = _convert_units(num, written, target)
    if "value_out" in out:
        return out["value_out"]
    return f"{num!r} {written}"


_STANDARD_DESIGNATION_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:AASHTO|ACI|AISC|ANSI|AS|ASCE|ASHRAE|ASTM|BS(?:\s+EN)?|CIBSE|CIRIA|CSA|DIN|EN|IBC|"
    r"ICC|IS|ISO|NFPA|NZS|SASO|SBC|TMS|UBC|UFC)\s?[A-Z]?\d{2,5}(?:[-:.]\d+)*(?!\d|\.\d)",
)


def _first_open(rx: "re.Pattern[str]", raw: str, overlaps: Any) -> "Optional[re.Match[str]]":
    """The first match of ``rx`` in ``raw`` that no earlier binding consumed."""
    return next((m for m in rx.finditer(raw) if not overlaps(m.span())), None)


def _bind_grade_labels(fn: Any, grades: List[Tuple[Tuple[int, int], str, float, str]],
                       found: Dict[str, Any]) -> None:
    """Bind each material's one grade label to that material's one open input."""
    from app.lib import formula_registry

    spec = _formula_spec(fn)
    params = formula_registry.parameters(spec) if spec else {}
    by_family: Dict[str, List[Tuple[float, str]]] = {}
    for _span, family, strength, label in grades:
        if (strength, label) not in by_family.setdefault(family, []):
            by_family[family].append((strength, label))
    for family, labels in by_family.items():
        owners = [k for k, p in params.items() if p.grade == family and k not in found]
        if len(labels) == 1 and len(owners) == 1:
            strength, label = labels[0]
            found[owners[0]] = formula_registry.grade_value(params[owners[0]], strength, label)


#: Currency codes and symbols written in front of a figure ("USD 150000",
#: "$5,000"). Only these are dropped: three capitals before a number are
#: otherwise a standard ("ACI 318") or a code and stay as written.
_CURRENCY_CODES = ("USD", "EUR", "GBP", "SAR", "AED", "QAR", "KWD", "OMR", "BHD", "EGP", "JOD", "INR",
                   "PKR", "CNY", "JPY", "KRW", "AUD", "NZD", "CAD", "CHF", "SGD", "HKD", "ZAR", "TRY",
                   "MYR", "IDR", "THB", "PHP", "BRL", "MXN", "SEK", "NOK", "DKK", "PLN", "RUB", "NGN", "KES")
_CURRENCY_BEFORE_RE = re.compile(r"(?<![A-Za-z])(?:(?:" + "|".join(_CURRENCY_CODES) + r")\s*|[$€£¥]\s*)(?=[+-]?\d)")
_SPELLED_MAGNITUDE_RE = re.compile(r"(?<![\d.])([+-]?\d[\d,]*(?:\.\d+)?)\s*(thousand|million|billion)\b", re.IGNORECASE)
_SPELLED_MAGNITUDE = {"thousand": 1e3, "million": 1e6, "billion": 1e9}
_AREA_WORD_RE = (
    (re.compile(r"\b(?:sq\.?|square)\s*(?:ft|feet|foot)\b", re.IGNORECASE), "ft2"),
    (re.compile(r"\b(?:sq\.?|square)\s*(?:m|metres?|meters?)\b", re.IGNORECASE), "m2"),
)


def _normalize_ask(raw: str) -> str:
    """The ask as the binder reads it: a written minus is a sign, an area
    written in words is a unit, a spelled-out magnitude is part of its figure
    (it scales any figure: "1.5 million m"), and a currency code or symbol in
    front of a figure is dropped. Abbreviated magnitudes (k, mn, bn) are left
    as written: after an engineering figure "MN" is a meganewton."""
    text = raw.replace("−", "-")
    text = re.sub(r"(?i)\bminus\s+(?=\d)", "-", text)
    for rx, unit in _AREA_WORD_RE:
        text = rx.sub(unit, text)

    def _scale(m: "re.Match[str]") -> str:
        value = float(m.group(1).replace(",", "")) * _SPELLED_MAGNITUDE[m.group(2).lower()]
        return f"{int(value) if value.is_integer() else value}"
    text = _SPELLED_MAGNITUDE_RE.sub(_scale, text)
    return _CURRENCY_BEFORE_RE.sub("", text)


def _figure_before_label(pat: str, raw: str, overlaps: Any) -> Optional[Any]:
    """The first open figure written directly before ``pat`` (its unit)."""
    rx = re.compile(rf"(?<![\w.,])({_TEXT_NUM_RE})\s*{pat}(?![A-Za-z0-9/])", re.IGNORECASE)
    return _first_open(rx, raw, overlaps)


def extract_calculation_params_from_text(
    fn: Any,
    text: str,
) -> Dict[str, Any]:
    """Pull labeled engineering numbers out of ask text. Never invents.

    A labeled figure binds to that input. A figure written only as a unit
    binds when exactly one still-missing required input declares that unit.
    A bare number binds when exactly one still-missing required input
    declares no unit of its own. Two figures of one kind bind to neither.
    """
    raw = _normalize_ask(str(text or "").strip())
    if not raw or fn is None:
        return {}
    labels = _text_label_map(fn)
    if not labels:
        return {}
    found: Dict[str, Any] = {}
    consumed: List[Tuple[int, int]] = []

    def _overlaps(span: Tuple[int, int]) -> bool:
        return any(span[0] < c[1] and c[0] < span[1] for c in consumed)

    # A standard's designation ("ACI 318", "BS 8110") is never a figure. A
    # grade label ("C30", "S355", "B500B") states a strength for the input
    # of its own material only, and is never a figure for any other input.
    consumed.extend(m.span() for m in _STANDARD_DESIGNATION_RE.finditer(raw))
    from app.lib import formula_registry
    grades = [g for g in formula_registry.grade_labels_in(raw) if not _overlaps(g[0])]
    consumed.extend(g[0] for g in grades)
    spec = _formula_spec(fn)
    declared_units = dict(spec.inputs or {}) if spec else {}

    for label, dest in sorted(labels.items(), key=lambda kv: len(kv[0]), reverse=True):
        if dest in found:
            continue
        pat = r"[\s_]+".join(re.escape(part) for part in label.split("_") if part)
        compact = label.replace("_", "")
        if len(compact) == 1:
            rx = re.compile(
                rf"(?<![A-Za-z0-9]){pat}(?:\s*[=:]\s*|\s+(?:is|of|equals|=)\s+)({_TEXT_NUM_RE})"
                rf"(?:\s*({_TEXT_UNIT_RE}))?",
                re.IGNORECASE,
            )
            glued = re.compile(
                rf"(?<![A-Za-z0-9]){pat}({_TEXT_NUM_RE})(?![\dA-Za-z])"
                rf"(?:\s*({_TEXT_UNIT_RE}))?",
                re.IGNORECASE,
            )
            match = _first_open(rx, raw, _overlaps) or _first_open(glued, raw, _overlaps)
        else:
            rx = re.compile(
                rf"(?<![A-Za-z0-9]){pat}(?:\s*[=:]\s*|\s+(?:is|of|equals|at)\s+|\s+)({_TEXT_NUM_RE})"
                rf"(?:\s*({_TEXT_UNIT_RE}))?",
                re.IGNORECASE,
            )
            # A label written straight after a figure is that figure's unit
            # ("60 days at 5,000 per day"): it labels the figure before it,
            # never the next one. Bare symbols (one letter) are not units here.
            match = _figure_before_label(pat, raw, _overlaps) or _first_open(rx, raw, _overlaps)
            if match is None and len(compact) <= 4:
                glued = re.compile(
                    rf"(?<![A-Za-z0-9]){pat}({_TEXT_NUM_RE})(?![A-Za-z0-9])"
                    rf"(?:\s*({_TEXT_UNIT_RE}))?",
                    re.IGNORECASE,
                )
                match = _first_open(glued, raw, _overlaps)
        if match is None:
            continue
        num = _parse_text_number(match.group(1))
        if num is None:
            continue
        found[dest] = _to_param_unit(num, match.group(2) if match.lastindex and match.lastindex >= 2 else None,
                                     dest, declared_units.get(dest, ""))
        consumed.append(match.span())

    # A period or a comma-space is punctuation. A comma that is followed by
    # a digit is still this number (a thousands group). Digits glued to a
    # letter belong to a label, grade or code, except a
    # multiplication sign written as x ("10x5x0.25"). A hyphen after a digit
    # is a range ("10-12"), not the sign of the second figure.
    number = rf"(?<![\d,])(?<![A-WYZa-wyz])({_TEXT_NUM_RE})(?!\d)(?!,\d)"

    # Figure first, dimension after: "12 m long, 8 m wide and 200 mm thick".
    adjectives = "|".join(sorted((re.escape(a) for a in _DIMENSION_ADJECTIVES), key=len, reverse=True))
    trailing = re.compile(
        rf"{number}\s*({_TEXT_UNIT_RE})?\s+({adjectives})(?![A-Za-z])",
        re.IGNORECASE,
    )
    for match in trailing.finditer(raw):
        if _overlaps(match.span()):
            continue
        noun = _DIMENSION_ADJECTIVES[match.group(3).lower()]
        dest = labels.get(_norm_label(noun))
        if not dest or dest in found:
            continue
        num = _parse_text_number(match.group(1))
        if num is None:
            continue
        found[dest] = _to_param_unit(num, match.group(2), dest, declared_units.get(dest, ""))
        consumed.append(match.span())

    # Figure first, label after: "120,000 m3 of concrete", "6,000 t steel".
    # The label may be written by its last word when no other label ends
    # with that word ("steel" for structural steel), and the figure's unit
    # must be the kind of unit that input declares, so a figure never binds
    # to an input of another kind on a word alone.
    label_after = re.compile(
        rf"{number}\s*({_TEXT_UNIT_RE})\s+(?:of\s+)?([A-Za-z][A-Za-z ]{{0,48}})",
        re.IGNORECASE,
    )

    def _named(words: List[str]) -> Optional[str]:
        for n in range(min(3, len(words)), 0, -1):
            key = _norm_label(" ".join(words[:n]))
            if key in labels:
                return labels[key]
            tails = {dest for lab, dest in labels.items() if lab.endswith("_" + key)}
            if len(tails) == 1:
                return tails.pop()
        return None

    for match in label_after.finditer(raw):
        if _overlaps(match.span()):
            continue
        dest = _named(match.group(3).lower().split())
        if not dest or dest in found:
            continue
        declared = str(declared_units.get(dest) or _unit_from_name(dest) or "")
        if not declared or not _same_kind(match.group(2), declared):
            continue
        num = _parse_text_number(match.group(1))
        if num is None:
            continue
        found[dest] = _to_param_unit(num, match.group(2), dest, declared_units.get(dest, ""))
        consumed.append(match.span())

    accepted = {
        row["name"] for row in describe_calculation_params(fn)
    }
    if "code" in accepted and "code" not in found:
        aci = bool(_CODE_ACI_RE.search(raw))
        euro = bool(_CODE_EC_RE.search(raw))
        if aci and not euro:
            found["code"] = "aci"
        elif euro and not aci:
            found["code"] = "eurocode"

    _bind_grade_labels(fn, grades, found)

    rows = describe_calculation_params(fn, name=spec.name if spec else None)
    required_by_kind: Dict[str, List[str]] = {}
    required_tokens: Dict[str, str] = {}
    extra_units: List[str] = []
    for row in rows:
        declared = str(declared_units.get(row["name"]) or "").strip()
        token = str(
            (declared if declared != "-" else "") or row.get("unit") or _unit_from_name(row["name"])
            or ""
        ).strip()
        if token:
            extra_units.append(token)
        if not row.get("required") or row["name"] in found:
            continue
        # The formula's own declaration wins over a name suffix
        # (w_kn_m is declared kN/m; its name suffix reads as kN.m).
        kind = binding_kind(token)
        required_by_kind.setdefault(kind, []).append(row["name"])
        required_tokens[row["name"]] = token

    unit_pat = _figure_unit_pattern(extra_units)
    boundary = r"(?![A-Za-z0-9])"
    leftover_rx = re.compile(
        rf"{number}\s*({unit_pat}){boundary}",
        re.IGNORECASE,
    )
    leftovers: Dict[str, List[Tuple[float, str]]] = {}
    for match in leftover_rx.finditer(raw):
        if _overlaps(match.span()):
            continue
        num = _parse_text_number(match.group(1))
        unit = _norm_unit_token(match.group(2))
        if num is None or not unit:
            continue
        leftovers.setdefault(unit, []).append((num, match.group(2)))
        consumed.append(match.span())
    unplaced: List[Tuple[float, str]] = []
    for unit, figures in leftovers.items():
        dests = [name for name in (required_by_kind.get(unit) or []) if name not in found]
        if len(figures) == 1 and len(dests) == 1:
            found[dests[0]] = _to_param_unit(*figures[0], dests[0], required_tokens[dests[0]])
        else:
            unplaced.extend(figures)
    # A figure in another unit of the same kind of quantity ("15.75 ft" for
    # a span in mm) binds when it is the only one of that kind and exactly
    # one open input measures that kind.
    by_dimension: Dict[str, List[Tuple[float, str]]] = {}
    for num, written in unplaced:
        parsed = _parse_unit(written)
        if parsed is not None:
            by_dimension.setdefault(parsed.dimension, []).append((num, written))
    open_by_dimension: Dict[str, List[str]] = {}
    for name, token in required_tokens.items():
        parsed = _parse_unit(token) if name not in found else None
        if parsed is not None:
            open_by_dimension.setdefault(parsed.dimension, []).append(name)
    for dimension, figures in by_dimension.items():
        dests = open_by_dimension.get(dimension) or []
        if len(figures) == 1 and len(dests) == 1:
            found[dests[0]] = _to_param_unit(*figures[0], dests[0], required_tokens[dests[0]])

    bare_rx = re.compile(
        rf"{number}(?!\s*(?:{unit_pat}){boundary})",
        re.IGNORECASE,
    )
    bare_nums: List[float] = []
    for match in bare_rx.finditer(raw):
        if _overlaps(match.span()):
            continue
        num = _parse_text_number(match.group(1))
        if num is None:
            continue
        bare_nums.append(num)
        consumed.append(match.span())
    bare_dests = [name for name in (required_by_kind.get("") or []) if name not in found]
    if len(bare_nums) == 1 and len(bare_dests) == 1:
        found[bare_dests[0]] = bare_nums[0]
    return found

def _ask_figure_counts(values: Any) -> Dict[float, int]:
    counts: Dict[float, int] = {}
    for val in values:
        num = _operand_value(val)
        if isinstance(num, (int, float)):
            key = round(float(num), 9)
            counts[key] = counts.get(key, 0) + 1
    return counts


def _unspent_ask_figures(params: Dict[str, Any], extracted: Dict[str, Any], text: str) -> Dict[str, Any]:
    """The ask-text figures that may fill ``params``' empty inputs.

    One written figure fills one input. A figure the call already placed on
    an input is spent: the extractor reading it again by another label
    ("a daily rate of 5,000" -> rate_percent next to the call's
    amount_per_day 5000) would bind one figure twice. A figure written as
    often as it is placed is spent; a converted figure (not written as such)
    and a text value are never counted. Figures compare to 9 decimals, so a
    float sum (0.1 + 0.2) is the figure written (0.3).
    """
    written = _ask_figure_counts(re.findall(_TEXT_NUM_RE, text))
    placed = _ask_figure_counts(params.values())
    out: Dict[str, Any] = {}
    for key, val in extracted.items():
        # run_calculation drops empty ("" / None) inputs before this point,
        # so a key still present is a given value and is never overwritten.
        if key in params:
            continue
        num = _operand_value(val)
        if isinstance(num, (int, float)):
            fig = round(float(num), 9)
            if placed.get(fig, 0) >= written.get(fig, 0) > 0:
                continue
            placed[fig] = placed.get(fig, 0) + 1
        out[key] = val
    return out


def _missing_required(fn: Any, bound: Dict[str, Any], name: Optional[str] = None) -> List[str]:
    missing: List[str] = []
    groups = _REQUIRED_GROUPS.get(str(name or "").strip().lower())
    if groups:
        for group in groups:
            if not any(k in bound and bound[k] not in (None, "") for k in group):
                missing.append(group[0])
    else:
        for row in describe_calculation_params(fn, name=name):
            if row["required"] and row["name"] not in bound:
                missing.append(row["name"])
    return missing + [key for key in _declared_required(name)
                      if _is_blank(bound.get(key)) and key not in missing]


def _declared_required(name: Optional[str]) -> List[str]:
    """Inputs the formula's registry entry declares required even though the
    signature gives them a default: they are asked for, never defaulted."""
    from app.lib import formula_registry

    spec = formula_registry.get(str(name or "").strip()) if name else None
    if spec is None:
        return []
    return [key for key, row in formula_registry.parameters(spec).items() if row.required]


# Words that do not identify a formula. A display name's remaining words
# must all appear in the ask, so two formulas that share a noun are not
# the same match.
_DISPLAY_WORD_STOP = frozenset({
    "the", "for", "per", "and", "from", "with", "under", "of", "a", "an",
    "to", "on", "in", "by", "or", "as", "at", "into", "over", "than", "via",
    "its", "this", "that",
})
_CALC_SELECT_VERB_RE = re.compile(
    r"\b(?:calculate|compute|work out|how many|how much)\b",
    re.IGNORECASE,
)


def _display_content_words(display: str) -> List[str]:
    words = re.findall(r"[a-z0-9]+", (display or "").lower())
    return [w for w in words if len(w) >= 3 and w not in _DISPLAY_WORD_STOP]


def _words_all_present(text: str, words: List[str]) -> bool:
    low = (text or "").lower()
    return all(re.search(rf"\b{re.escape(word)}\b", low) for word in words)


def explicit_calculation_request(text: str) -> bool:
    """The user asked for a calculation, not for what a document states."""
    return _CALC_SELECT_VERB_RE.search(text or "") is not None


def message_names_registry_id(text: str) -> bool:
    """The ask contains a formula's registry id, not only its display name."""
    underscored = (text or "").lower().replace("-", "_")
    if not underscored.strip():
        return False
    for name in CALCULATORS:
        if len(name) >= 6 and name.lower() in underscored:
            return True
    return False


def named_formula_operands(text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Figures the ask already binds for the one formula it names."""
    name = calculator_name_from_text(text)
    if not name:
        return None
    fn = CALCULATORS.get(name)
    if fn is None:
        return None
    bound = extract_calculation_params_from_text(fn, text)
    numeric = {
        key: val for key, val in bound.items()
        if isinstance(val, (int, float)) and not isinstance(val, bool)
    }
    if not numeric:
        return None
    return name, numeric


def _operand_value(raw: Any) -> Any:
    """A number or a non-empty string a call passed, else None."""
    if isinstance(raw, bool) or raw is None:
        return None
    if isinstance(raw, (int, float)):
        return raw
    if isinstance(raw, str) and raw.strip():
        parsed = _parse_text_number(raw.strip())
        return parsed if parsed is not None else raw.strip()
    return None


def _is_signature_default(param: Any, value: Any) -> bool:
    if param is None or param.default is _inspect.Parameter.empty:
        return False
    default = param.default
    if isinstance(default, bool):
        return False
    if isinstance(value, (int, float)) and isinstance(default, (int, float)):
        return abs(float(value) - float(default)) <= 1e-9
    if isinstance(value, str) and isinstance(default, str):
        return value.strip().casefold() == default.strip().casefold()
    return False


def run_carries_operand(
    name: str,
    params: Optional[Dict[str, Any]] = None,
    text: str = "",
    stated: str = "",
) -> bool:
    """True when a call gives this calculator a figure it did not default.

    A passed value equal to the signature default is that default, not an
    operand -- unless the user's own words (``stated``) write that figure:
    then it is the user's figure, which happens to equal the default (lane 4:
    "cube 7 MPa" equals the striking-time default 7, and the run was refused
    as defaulted). A figure the ask text binds counts the same as a passed
    one. An unknown calculator is left to ``run_calculation`` to report.
    """
    written = {round(n, 9) for n in (_parse_text_number(m) for m in
               re.findall(_TEXT_NUM_RE, str(stated or ""))) if n is not None}
    fn = CALCULATORS.get(str(name or "").strip())
    if fn is None:
        return True
    try:
        signature = _inspect.signature(fn)
    except (TypeError, ValueError):
        return True
    raw = dict(params or {}) if isinstance(params, dict) else {}
    flat = _flatten_calc_kwargs(raw)
    junk = {_snake_key(j) for j in _BIND_JUNK_KEYS}
    for key, val in flat.items():
        if _snake_key(key) in junk:
            continue
        value = _operand_value(val)
        if value is None:
            continue
        if not _is_signature_default(signature.parameters.get(key), value):
            return True
        if isinstance(value, (int, float)) and round(float(value), 9) in written:
            return True
    blob = " ".join(
        str(part) for part in (
            text, raw.get("text"), raw.get("formula"), raw.get("message"),
        ) if isinstance(part, str) and part.strip()
    )
    if not blob:
        return False
    for key, val in extract_calculation_params_from_text(fn, blob).items():
        value = _operand_value(val)
        if value is None:
            continue
        if not _is_signature_default(signature.parameters.get(key), value):
            return True
    return False


def formula_completed_by_user_text(
    text: str,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """The one registered formula this ask both names and supplies.

    The display name's content words must all be in the ask, and every
    required input must already be bound from the ask by the same extractor
    the calculator uses. Several formulas at the same strength is no
    selection: the turn is not guessed. A formula the ask does not name
    is not selected, even when its inputs would bind.
    """
    raw = text or ""
    if not raw.strip() or _CALC_SELECT_VERB_RE.search(raw) is None:
        return None
    from app.lib.formula_registry import all_specs

    candidates: List[Tuple[str, Dict[str, Any], int, int]] = []
    for spec in all_specs():
        words = _display_content_words(spec.display_name)
        if not words or not _words_all_present(raw, words):
            continue
        bound = extract_calculation_params_from_text(spec.fn, raw)
        if _missing_required(spec.fn, bound, spec.name):
            continue
        numeric = {
            key: val for key, val in bound.items()
            if isinstance(val, (int, float)) and not isinstance(val, bool)
        }
        if not numeric:
            continue
        candidates.append((spec.name, bound, len(words), len(numeric)))
    if not candidates:
        return None
    best = max((row[2], row[3]) for row in candidates)
    tied = [row for row in candidates if (row[2], row[3]) == best]
    if len(tied) != 1:
        return None
    name, bound, _words, _count = tied[0]
    return name, bound


def calculation_headline(name: str, result: Any) -> Optional[Tuple[str, float]]:
    """First numeric declared output that is not also an input."""
    from app.lib.formula_registry import get

    spec = get(name)
    if spec is None or not isinstance(result, dict):
        return None
    declared_inputs = set(spec.inputs or {})
    for key in spec.outputs or {}:
        if key in declared_inputs:
            continue
        val = result.get(key)
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            continue
        return key, float(val)
    return None


def format_user_calculation_answer(
    name: str,
    supplied: Dict[str, Any],
    envelope: Dict[str, Any],
    user_text: str = "",
) -> str:
    """The calculator's result, its note, and a credit for the user's inputs.

    A non-zero signature default the user did not supply, echoed in the
    result, is named as the calculator's default. It is not described as
    a figure from the project documents.
    """
    from app.lib.formula_registry import get
    from app.lib.source_labels import (
        calculator_default_lines,
        calculator_label,
        input_phrase,
    )

    spec = get(name)
    result = envelope.get("result") if isinstance(envelope, dict) else None
    if spec is None or not isinstance(result, dict):
        return ""
    lines: List[str] = []
    headline = calculation_headline(name, result)
    if headline:
        key, val = headline
        unit = (spec.outputs or {}).get(key, "")
        lines.append(f"{spec.display_name}: {input_phrase(key, val, unit)}.")
    else:
        lines.append(f"{spec.display_name}.")
    note = result.get("note")
    if isinstance(note, str) and note.strip():
        lines.append(note.strip())
    notes = result.get("notes")
    if isinstance(notes, list):
        for item in notes:
            if str(item).strip():
                lines.append(str(item).strip())
    lines.extend(calculator_default_lines(
        name, result, user_text, stated_keys=set(supplied or {}),
    ))
    lines.append("Source: " + calculator_label(name, supplied))
    return "\n".join(lines)

def _format_param_label(row: Dict[str, Any]) -> str:
    name = row["name"]
    unit = row.get("unit") or ""
    if unit:
        name = f"{name} ({unit})"
    if row.get("required"):
        return name
    default = row.get("default")
    return f"{name} (default {default!r})"

def _bind_error_envelope(
    name: str,
    fn: Any,
    missing: List[str],
    extra: str = "",
) -> Dict[str, Any]:
    expected = describe_calculation_params(fn, name=name)
    by_name = {row["name"]: row for row in expected}
    missing_labels = [
        _format_param_label(by_name[m]) if m in by_name else m for m in missing
    ]
    expected_labels = [_format_param_label(row) for row in expected]
    if missing_labels:
        detail = "missing required " + ", ".join(missing_labels)
    else:
        detail = "could not bind arguments"
    expected_clause = ", ".join(expected_labels) if expected_labels else "(none)"
    msg = f"Bad parameters for {name}: {detail}. Expected: {expected_clause}."
    if extra:
        msg = f"{msg} ({extra})"
    # A required input the asked output needs but did not get -- blank, or
    # omitted when it is needed -- is an ASK, not a hard error: one status,
    # "needs_input", naming the field (item 7 ruling, ORDERS_31OCT_0405Z). A
    # bind failure with nothing to ask for (no missing field) stays "error".
    env: Dict[str, Any] = {
        "status": "needs_input" if missing else "error",
        "error": msg,
        "signature": f"{name}{_inspect.signature(fn)}",
        "expected_params": expected,
        "missing": missing,
    }
    if missing:
        env["needs_input"] = True
    return _with_input_question(env, name, missing)

def _accepts_param(fn: Any, param_name: str) -> bool:
    try:
        sig = _inspect.signature(fn)
    except (TypeError, ValueError):
        logger.debug(
            "signature unavailable for %s",
            getattr(fn, "__name__", type(fn).__name__),
        )
        return False
    param = sig.parameters.get(param_name)
    if param is None:
        return False
    return param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)

def _numbers_agree(left: Any, right: Any) -> bool:
    """True when both tokens are the same number, including ``\"24\"`` and 24."""
    a = _coerce_scalar(left)
    b = _coerce_scalar(right)
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= 1e-9
    return a == b

def _count_quantity_conflict(params: Dict[str, Any]) -> Optional[str]:
    """Error text when ``count`` and ``quantity`` are both set and differ.

    Equal values (including a numeric string) are one element count. A
    silent pick of either side is the bug: the model cannot tell which
    figure was used.
    """
    count_val: Any = None
    qty_val: Any = None
    seen_count = False
    seen_qty = False
    for key, val in params.items():
        if val is None or val == "":
            continue
        nk = _snake_key(key)
        if nk == "count":
            seen_count = True
            count_val = val
        elif nk == "quantity":
            seen_qty = True
            qty_val = val
    if not (seen_count and seen_qty):
        return None
    if _numbers_agree(count_val, qty_val):
        return None
    return (
        f"count and quantity differ: count={count_val!r}, quantity={qty_val!r}. "
        "Pass one element count, or the same value for both."
    )

def _unknown_arg_envelope(
    name: str, fn: Any, unknown: List[str],
) -> Dict[str, Any]:
    """Structured tool error for keys that did not bind. The turn does not raise."""
    expected = describe_calculation_params(fn, name=name)
    labels = ", ".join(_format_param_label(row) for row in expected) or "(none)"
    listed = ", ".join(unknown)
    return {
        "status": "error",
        "calculation": name,
        "error": (
            f"Unknown argument(s) for {name}: {listed}. "
            f"These were not applied. Expected: {labels}."
        ),
        "unknown": list(unknown),
        "signature": f"{name}{_inspect.signature(fn)}",
        "expected_params": expected,
    }

def _pop_conversion_context(fn: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Crew size, working hours per day and working days per week given for
    converting another input. A calculator that does not take one as an
    input of its own still uses it to convert, and it is not an unknown argument."""
    from app.agents.base.formulas.construction_formulas_planning import CONVERSION_CONTEXT
    from app.lib import formula_registry

    held: Dict[str, Any] = {}
    for key in list(params):
        canon = _snake_key(key)
        if canon not in CONVERSION_CONTEXT:
            continue
        held[canon] = params[key]
        if not _accepts_param(fn, canon):
            params.pop(key)
    return formula_registry.conversion_context(held)


def _with_input_question(env: Dict[str, Any], name: str, missing: List[str]) -> Dict[str, Any]:
    """The missing inputs asked for by their labels, next to the technical error."""
    from app.lib.source_labels import input_question

    if missing:
        env["question"] = input_question(name, [], missing)
    return env


def _input_rejection_envelope(
    name: str, rejected: List[Dict[str, Any]], missing: List[str],
) -> Dict[str, Any]:
    """The calculator was not run: a supplied value has the wrong type or
    lies outside the input's declared range. Either way the value the asked
    output needs is not usable, so this is an ASK, not a hard error: status
    "needs_input" with the question that names the inputs by their labels and
    carries the range/reason text (item 7 ruling, ORDERS_31OCT_0405Z/0705Z --
    out-of-range asks WITH its range named)."""
    from app.lib.source_labels import input_question

    question = input_question(name, rejected, missing)
    return {
        "status": "needs_input",
        "needs_input": True,
        "calculation": name,
        "error": question,
        "question": question,
        "rejected": rejected,
        "missing": list(missing),
    }


def _bare_figure(value: Any) -> Optional[float]:
    """A figure passed with no unit of its own, or None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and _PLAIN_NUM_RE.match(value):
        return _ask_float(value)
    return None


_LEADING_FIGURE_RE = re.compile(r"^\s*(\d+(?:,\d{3})*(?:\.\d+)?)(?=\s|[A-Za-z%/]|$)")


def _restore_written_signs(params: Dict[str, Any], user_words: str) -> Dict[str, Any]:
    """Every figure the caller sent -- a number, or a string that starts with
    one ("120 m2/day") -- keeps the sign the user wrote: when the user's own
    words write that figure only with a minus, the minus goes back, so the
    formula sees the user's input and refuses or uses it as written. The
    user's words are this turn's ask, never a paraphrase of it: a model that
    writes "120 m2/day" in its own text has not shown the user wrote 120."""
    if not user_words:
        return params
    out = dict(params)
    for key, val in params.items():
        if _is_junk_key(key) or isinstance(val, bool):
            continue
        if isinstance(val, (int, float)):
            if val > 0 and _written_only_negative(float(val), user_words):
                out[key] = -val
        elif isinstance(val, str):
            m = _LEADING_FIGURE_RE.match(val)
            figure = _ask_float(m.group(1)) if m else None
            if figure and figure > 0 and _written_only_negative(figure, user_words):
                out[key] = val[:m.start(1)] + "-" + val[m.start(1):]
    return out


_ZERO_WORD_RE = re.compile(r"(?<![\w.,])0+(?:\.0+)?(?![\d.,]*\d)|\b(?:zero|none|no|nil|without)\b", re.I)
_WORD_RE = re.compile(r"[a-z]+")
#: How many words from a zero word an input's own word may sit to be the
#: input that zero is written for ("0 cranes", "no riggers", "zero waste").
_ZERO_REACH_WORDS = 3


def _input_words(key: str, label: str) -> set:
    """The words that name an input: its declared label and its key, as stems
    (a trailing s dropped, so 'cranes' names num_cranes), ignoring words of
    one or two letters."""
    words = set(_WORD_RE.findall(f"{label} {key.replace('_', ' ')}".lower()))
    return {w[:-1] if w.endswith("s") and len(w) > 3 else w for w in words if len(w) > 2}


def _zero_written_for(key: str, label: str, user_words: str) -> bool:
    """True when the user's words write a zero for this input: a zero word
    that qualifies one of the input's own words -- before it in the same
    phrase ("0 cranes", "no cranes needed"), or after it joined by a colon,
    equals or "is" ("number of cranes: zero", "cranes = 0"). Punctuation
    ends a phrase, so "No rush: crane hire" writes no zero for the cranes."""
    names = _input_words(key, label)
    tokens = [m.group(0).lower() for m in re.finditer(r"[A-Za-z]+|\d+(?:[.,]\d+)*|[^\sA-Za-z\d]", user_words)]

    def stem(word: str) -> str:
        return word[:-1] if word.endswith("s") and len(word) > 3 else word

    def is_word(token: str) -> bool:
        return bool(re.fullmatch(r"[a-z]+|\d+(?:[.,]\d+)*", token))

    for i, token in enumerate(tokens):
        if not _ZERO_WORD_RE.fullmatch(token):
            continue
        # Before the input: the next few words of the same phrase.
        ahead = []
        for nxt in tokens[i + 1:]:
            if not is_word(nxt) or len(ahead) == _ZERO_REACH_WORDS:
                break
            ahead.append(stem(nxt))
        if names.intersection(ahead):
            return True
        # After the input, joined by ':', '=' or 'is'/'are'.
        j = i - 1
        while j >= 0 and tokens[j] in (":", "=", "is", "are"):
            j -= 1
        if j < i - 1 and j >= 0 and stem(tokens[j]) in names:
            return True
    return False


def _drop_unwritten_zeros(name: str, fn: Any, params: Dict[str, Any], user_words: str) -> Dict[str, Any]:
    """A zero the caller sent for an input whose zero the user's own words
    never write is not the user's input: it is left out, so the calculator
    asks for it (or uses its declared default) instead of refusing "0 was
    given" (LANE2: a crane hire cost refused a phantom crane count of 0).
    Judged per input, by its own declared label: a zero written for one input
    ("no riggers") never vouches for another (LANE4). Switches (inputs whose
    default is True/False) keep their value: 0 there is "off"."""
    if not user_words:
        return params
    try:
        signature = _inspect.signature(fn)
    except (TypeError, ValueError):
        return params
    from app.lib import formula_registry

    spec = formula_registry.get(name)
    declared = formula_registry.parameters(spec) if spec is not None else {}
    out = dict(params)
    for key, val in params.items():
        p = signature.parameters.get(key)
        if p is None or isinstance(p.default, bool) or isinstance(val, bool):
            continue
        if not ((isinstance(val, (int, float)) and val == 0) or (isinstance(val, str) and val.strip() in ("0", "0.0"))):
            continue
        label = getattr(declared.get(key), "label", "") or ""
        if not _zero_written_for(key, label, user_words):
            out.pop(key)
    return out


def _call_unit(params: Dict[str, Any]) -> Optional[str]:
    """The one unit a call states for its figures (``"unit": "ft"``)."""
    for source in [params] + [coerce_calc_params(params.get(k)) for k in _FLATTEN_NEST_KEYS]:
        unit = source.get("unit") if isinstance(source, dict) else None
        if isinstance(unit, str) and unit.strip() and _parse_unit(unit.strip()) is not None:
            return unit.strip()
    return None


def _units_written_for(figure: float, text: str) -> List[str]:
    """Every unit the ask writes for this figure, a chain's shared unit included.
    Read from the ask with its written units normalised ("sq ft" is ft2)."""
    from app.lib.construction_formulas_quantities import dimension_chains

    text = _normalize_ask(text or "")
    found = [unit for chain in dimension_chains(text) for num, unit in chain
             if unit and _numbers_agree(num, figure)]
    written = re.compile(rf"(?<![\w.,])({_TEXT_NUM_RE})\s*({_TEXT_UNIT_RE})(?![A-Za-z0-9])", re.IGNORECASE)
    for match in written.finditer(text or ""):
        num = _parse_text_number(match.group(1))
        if num is not None and _numbers_agree(num, figure):
            found.append(match.group(2))
    return found


def _words_written_after(figure: float, text: str) -> str:
    """The unit words the ask writes right after this figure, up to the end
    of the clause (at most six words); empty when the figure is not written
    or is written twice with different words."""
    found = set()
    num = re.compile(r"(?<![\w.,])(" + _TEXT_NUM_RE + r")(?![\d,.]*\d)")
    for match in num.finditer(text or ""):
        value = _parse_text_number(match.group(1))
        if value is None or not _numbers_agree(value, figure):
            continue
        rest = re.split(r"[.;:?!,()]|\s(?:and|for|with|to|over|in|on)\s", text[match.end():], maxsplit=1)[0]
        words = " ".join(rest.split()[:6]).strip()
        if words and re.match(r"[A-Za-z%/]", words):
            found.add(words)
    return next(iter(found)) if len(found) == 1 else ""


def _written_only_negative(figure: float, text: str) -> bool:
    """True when the ask writes this figure only with a minus sign ("-120 m2
    per day") -- so a positive value in the call dropped the user's sign."""
    num = re.compile(r"(?<![\w.,])(" + _TEXT_NUM_RE + r")(?![\d,.]*\d)")
    values = [_parse_text_number(m.group(1)) for m in num.finditer(text or "")]
    values = [v for v in values if v is not None]
    return (any(_numbers_agree(v, -figure) for v in values)
            and not any(_numbers_agree(v, figure) for v in values))


_RATE_WORDS = re.compile(r"[A-Za-z0-9]+(?:\s+[A-Za-z0-9]+)?\s*(?:/\s*|\s+per\s+)[A-Za-z]+(?:-[A-Za-z]+)?")


def _attach_written_units(name: str, fn: Any, bound: Dict[str, Any], caller: Dict[str, float],
                          text: str, call_unit: Optional[str]) -> Dict[str, Any]:
    """A bare figure the caller passed is in the unit the ask wrote for that
    figure, else in the call's own unit, when that unit measures what the
    input measures. The figure then goes through the input's unit conversion.
    Two different units written for one figure leave it as passed."""
    from app.lib import formula_registry

    spec = formula_registry.get(name)
    if spec is None:
        return bound
    declared = formula_registry.parameters(spec)
    out = dict(bound)
    for key, figure in caller.items():
        if isinstance(figure, (int, float)) and figure > 0 and _written_only_negative(figure, text):
            # The ask wrote this figure with a minus sign and the call dropped
            # it: the user's own figure goes to the formula, which refuses it
            # and asks -- a sign is never silently normalised away.
            for dest in _partition_bound_params(fn, {key: figure})[0]:
                if _numbers_agree(out.get(dest), figure):
                    out[dest] = -figure
            figure = -figure
        dests = list(_partition_bound_params(fn, {key: figure})[0])
        p = declared.get(dests[0]) if len(dests) == 1 else None
        if p is not None and p.kind == "text" and isinstance(out.get(p.name), (int, float))                 and not isinstance(out.get(p.name), bool) and _numbers_agree(out.get(p.name), figure):
            # A text input reads the unit words written on its value ("1,500
            # m2 per gang-day"); a bare figure gets the words the ask wrote
            # after that figure, so the formula reads the user's unit.
            words = _words_written_after(figure, text)
            if words:
                out[p.name] = f"{figure:g} {words}"
            continue
        if (p is None or p.kind not in formula_registry.NUMERIC_KINDS or not p.unit
                or not _numbers_agree(out.get(p.name), figure)):
            continue
        target = _parse_unit(p.unit)
        if target is None:
            continue

        def factor(unit: str) -> Optional[float]:
            parsed = _parse_unit(unit)
            if parsed is None or parsed.dimension != target.dimension:
                return None
            got = _convert_units(1.0, unit, p.unit).get("value_out")
            return round(got, 9) if isinstance(got, (int, float)) else None

        by_factor: Dict[float, str] = {}
        for unit in _units_written_for(figure, text):
            f = factor(unit)
            if f is not None:
                by_factor.setdefault(f, unit)
        unit = next(iter(by_factor.values())) if len(by_factor) == 1 else None
        if unit is None and call_unit and factor(call_unit) is not None and (
                not by_factor or factor(call_unit) in by_factor):
            unit = call_unit
        if unit and factor(unit) != 1.0:
            out[p.name] = f"{figure!r} {unit}"
        elif not unit and not by_factor and "/" not in str(p.unit):
            # Written as a rate ("10 lifts/h", "12 m2 per day") for an input
            # that is not one: the rate goes with the figure, so the unit
            # check asks instead of reading the rate as the input's unit
            # (lane 3: "10 lifts/h" ran as a 10 t crane).
            rate = _RATE_WORDS.match(_words_written_after(figure, text))
            if rate:
                out[p.name] = f"{figure!r} {rate.group(0).strip()}"
    return out


def _defaults_taken(fn: Any, params: Dict[str, Any], name: str) -> List[Dict[str, Any]]:
    """The optional inputs this run took from the formula's defaults: not
    given, with a real default (None means "not given" and is never a
    value). Rows as the hats' default disclosure writes them."""
    import inspect

    from app.lib.source_labels import parameter_label

    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        logger.debug("defaults: no signature for %s; nothing to disclose", name, exc_info=True)
        return []
    rows = []
    for key, p in signature.parameters.items():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) or key in params or key == "unit":
            continue
        if p.default is inspect.Parameter.empty or p.default is None or p.default == "":
            continue
        rows.append({"parameter": key, "value": p.default, "label": parameter_label(name, key)})
    return rows


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _blank_calc_kwargs(params: Dict[str, Any]) -> Tuple[str, ...]:
    """The calculator kwargs, top level or nested as _flatten_calc_kwargs
    reads them, that the caller sent as null or blank, in the caller's order."""
    nested = [coerce_calc_params(params.get(k)) for k in _FLATTEN_NEST_KEYS]
    return tuple(dict.fromkeys(str(key) for group in (params, *nested)
                               for key, val in group.items() if _is_blank(val)))


def _named_but_blank(name: str, blank: Tuple[str, ...], params: Dict[str, Any],
                     missing: List[str]) -> List[str]:
    """The inputs, figure or text, the caller sent as null or blank that
    nothing has filled since (the ask's own words may have) and that would
    otherwise run on a real default: each is asked for, so no default stands
    in for a value the user was meant to give. An input whose own default is
    None or "" loses nothing when left blank, so it is not asked for."""
    from app.lib import formula_registry

    spec = formula_registry.get(name)
    declared = formula_registry.parameters(spec) if spec is not None else {}
    return [key for key in blank
            if key in declared and not _is_blank(declared[key].default)
            and _is_blank(params.get(key)) and key not in missing]


def run_calculation(name: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run one whitelisted deterministic calculator by name with keyword params.

    Returns a standard envelope: {status, calculation, result, note} on success,
    or {status:error, error, ...} on a bad name / bad params. COST build-ups use
    the unit rates passed in ``params``; the built-in defaults are indicative GCC
    fallbacks only — pass the project's priced-BOQ rates (from RAG) for a firm
    cost so the answer stays grounded.
    """
    params = params or {}
    if not isinstance(params, dict):
        coerced = coerce_calc_params(params)
        if coerced:
            params = coerced
        elif params:
            return {"status": "error", "error": "params must be an object of keyword arguments."}
        else:
            params = {}
    positional = _pop_positional(params) if isinstance(params, dict) else None
    # Shared with Agent C / #636 / #639 / #652: nested ``params`` / ``input``
    # + top-level siblings (volume / BCWS / excavation_bank_m3 / water_depth_m)
    # must reach the calculator. Flatten before E4 so a nested concrete ask
    # still pins.
    call_unit = _call_unit(params)
    # Inputs the caller named but could not fill (null or blank): a value it
    # meant to give and did not have. Flatten drops them; they are asked
    # for below, never defaulted.
    caller_blank = _blank_calc_kwargs(params)
    params = _flatten_calc_kwargs(params)
    if positional is None:
        positional = _pop_positional(params)
    else:
        params.pop(_POSITIONAL_KEY, None)
    # A minus sign the user wrote survives the call, by any key and in any
    # form the caller chose (LANE2 -120: the model sent productivity=120,
    # or "120 m2/day", or wrote its own positive paraphrase in ``text``).
    params = _restore_written_signs(params, _CURRENT_ASK.get() or _ask_blob(params))
    caller_figures = {key: figure for key, val in params.items()
                      if not _is_junk_key(key) and (figure := _bare_figure(val)) is not None}
    # The keys the caller sent, before the ask's words are read for more: an
    # argument only the ask extractors produced is the server's own guess.
    caller_keys = frozenset(str(k) for k in params)
    ask_text = _ask_blob(params)
    # Live UI pack E4: a concrete/raft ask (or leftover-L6 excavation name
    # plus "documented waste factor" in ``text``) must apply the project's
    # 5% waste. Resolve before the name lookup so a missing calculation
    # still reaches concrete_volume when the ask carries the dims.
    original_name = name
    try:
        from app.lib.construction_formulas_quantities import (
            resolve_concrete_volume_calc,
        )
        name, params = resolve_concrete_volume_calc(name, params)
    except Exception:  # noqa: BLE001 — never break a non-concrete calc
        logger.warning(
            "swallowed %s in resolve_concrete_volume_calc() — continuing",
            "Exception", exc_info=True,
        )
    # Phase 2 F–W (#43–84): remap resource_line / material_consumption
    # mis-picks (and undo an E4 concrete_volume steal) after the volume
    # pin so a "concrete" word in a consumption ask is not left on E4.
    try:
        from app.lib.construction_formulas_quantities import resolve_fw_calc
        name, params = resolve_fw_calc(
            name, params, original_name=original_name,
        )
    except Exception:  # noqa: BLE001 — never break a non-F–W calc
        logger.warning(
            "swallowed %s in resolve_fw_calc() — continuing",
            "Exception", exc_info=True,
        )
    fn = CALCULATORS.get(name)
    if fn is None:
        # Recover only from the ask/params prose (text/formula/message/query).
        # Do not feed the unknown *name* into calculator_name_from_text —
        # "calculate_delay_damages" contains "delay damages" and would
        # silently bind delay_damages_daily, dropping the Unknown-calculation
        # envelope (and its ``available`` list) that the model needs to retry.
        blob = _ask_blob(params) if isinstance(params, dict) else ""
        recovered = calculator_name_from_text(blob) if str(blob).strip() else None
        if recovered:
            name = recovered
            fn = CALCULATORS.get(name)
    if fn is None:
        return {
            "status": "error",
            "error": f"Unknown calculation '{name}'.",
            "available": available_calculations(),
        }
    if positional:
        for key, val in _bind_sequence_params(fn, positional, name).items():
            if params.get(key) in (None, ""):
                params[key] = val
    # The registry's own declarations first: a key written as the input's
    # label or its name in words, and a figure with a magnitude word.
    from app.lib import formula_registry as _registry
    params = _registry.bind_declared(str(name), params)
    # Signature-derived bind (case-insensitive + unit-suffix synonyms).
    # Generalises #636's calculate_evm PMI aliases — do not re-add a
    # name-specific filter here (Agent C / DIR7 share this path).
    if str(name or "").strip().lower() == "calculate_evm":
        params = _alias_calculate_evm_params(params)
    # E / I / c -> ec_mpa / i_mm4 / code, with the unit scaled onto the
    # destination. Runs before binding so a symbol is bound, not reported
    # unknown; a calculator without the destination is untouched.
    try:
        params = _alias_physics_symbols(fn, params)
    except _UnknownUnit as exc:
        return {"status": "error", "calculation": str(name), "error": str(exc)}
    # Standing-exit B: model aliases / numbers live in ``text`` more often
    # than in kwargs. Fill holes only — never invent defaults.
    params = _extract_calc_kwargs_from_ask(str(name), params)
    # Named-calculator / predispatch often ships ``text`` only. Pull labeled
    # numbers (As1500, span 8m, W=10000 kN) into kwargs. Explicit keys win.
    # D7: extract never invents a figure that is not in the ask.
    text_blob = " ".join(
        str(x) for x in (
            params.get("text"), params.get("formula"), params.get("message"),
        ) if x not in (None, "")
    )
    if text_blob:
        # Only an input this calculator accepts spends a figure: a stray key
        # another extractor read ("damages: 45") is dropped later, unbound.
        inputs = {k: v for k, v in params.items() if _accepts_param(fn, k)}
        params.update(_unspent_ask_figures(
            inputs, extract_calculation_params_from_text(fn, text_blob), text_blob))
    # Re-run the symbol alias AFTER both text extractors. The ask sentence rides
    # along in ``text`` / ``query`` and both extractors above pull "E = 200",
    # "I = 2.0e-4" (and a shortened ``c``) back out of it as bare keys — after
    # the first alias at the top of this function had already run. Without this
    # second pass those re-extracted symbols reach _partition_bound_params and
    # are reported "Unknown argument(s): E, I" even when the caller sent perfect
    # ec_mpa / i_mm4 (the live beam-deflection ask 0/6; #741 and #743 both missed it because the
    # local tests attached no text). The alias is idempotent: a symbol whose
    # canonical destination is already set is dropped, so ec_mpa still wins.
    try:
        params = _alias_physics_symbols(fn, params)
    except _UnknownUnit as exc:
        return {"status": "error", "calculation": str(name), "error": str(exc)}
    if _accepts_param(fn, "quantity"):
        conflict = _count_quantity_conflict(params)
        if conflict:
            return {
                "status": "error",
                "calculation": name,
                "error": conflict,
                "signature": f"{name}{_inspect.signature(fn)}",
                "expected_params": describe_calculation_params(fn, name=str(name)),
            }
    from app.lib import formula_registry
    context = _pop_conversion_context(fn, params)
    params = _drop_unwritten_zeros(str(name), fn, params, _CURRENT_ASK.get())
    params, unknown = _partition_bound_params(fn, params)
    # An argument read out of the ask's words that this calculator does not
    # take is dropped, not an error: only an argument the caller sent can be
    # "unknown" (lane 5: "10 panels of 6 x 0.8 x 25 m" gave width_m and wall,
    # and a complete diaphragm-wall call failed 3/3).
    unknown = [key for key in unknown if key in caller_keys]
    if not _accepts_param(fn, "unit"):
        params = _attach_written_units(str(name), fn, params, caller_figures,
                                       _with_current_ask(ask_text), call_unit)
    params, conversions, unit_rejected = formula_registry.convert_written_units(
        str(name), params, context)
    params = _coerce_bound_values(fn, params, frozenset(c.get("parameter") for c in conversions),
                                  _stated_percents(ask_text))
    params, rejected = formula_registry.check_inputs(str(name), params)
    refused = {row["parameter"] for row in unit_rejected}
    rejected = unit_rejected + [row for row in rejected if row["parameter"] not in refused]
    missing = _missing_required(fn, params, name=str(name))
    missing += _named_but_blank(str(name), caller_blank, params, missing)
    if rejected and not unknown:
        return _input_rejection_envelope(str(name), rejected, missing)
    if unknown:
        if missing:
            env = _bind_error_envelope(str(name), fn, missing)
            # #649 canned help names paired alternatives (rebar length-or-mass,
            # productivity pairs) that the signature-required list omits.
            if str(name or "").strip() in _CALC_REQUIRED_HELP:
                env["error"] = required_params_error(name, fn)
            env["error"] = (
                f"{env['error']} Unknown argument(s) not applied: "
                f"{', '.join(unknown)}."
            )
            env["unknown"] = list(unknown)
            env["calculation"] = name
            return env
        return _unknown_arg_envelope(str(name), fn, unknown)
    if missing:
        env = _bind_error_envelope(str(name), fn, missing)
        # #649 canned help names paired alternatives (rebar length-or-mass,
        # productivity pairs) that the signature-required list omits.
        if str(name or "").strip() in _CALC_REQUIRED_HELP:
            env["error"] = required_params_error(name, fn)
            env["calculation"] = name
        return env
    try:
        result = fn(**params)
    except TypeError as exc:
        # Never a bare TypeError string alone — name the expected params
        # with units so the next call can bind (#639/#652 envelope + #649
        # required_params_error canned help). Keep inspect signature.
        still_missing = _missing_required(fn, params, name=str(name)) or [
            row["name"] for row in describe_calculation_params(fn, name=str(name))
            if row["required"]
        ]
        env = _bind_error_envelope(str(name), fn, still_missing, extra=str(exc))
        env["calculation"] = name
        env["cause"] = str(exc)
        if str(name or "").strip() in _CALC_REQUIRED_HELP:
            env["error"] = required_params_error(name, fn)
        return env
    except Exception as exc:  # noqa: BLE001 — never raise into the agent loop
        return {"status": "error", "error": f"{name} failed: {exc}"}

    if _dc.is_dataclass(result):
        result = _dc.asdict(result)
    elif not isinstance(result, dict):
        result = {"value": result}

    # Not every calculator signals failure by RAISING. Several (score_risk and
    # friends in app/core/construction_knowledge.py) validate their inputs and
    # RETURN {"error": "..."} instead. Without this, that reply was wrapped as
    # status:"success" -- an error envelope inside a green one, which the agent
    # tool loop reads as a working answer and narrates as a result.
    #
    if not _result_is_failure(result) and "defaults_used" not in result:
        # An optional input the caller did not give ran on its declared
        # default: the result says so, so the answer can disclose it (LANE2:
        # a tender ranked on default weights nobody stated).
        defaulted = _defaults_taken(fn, params, str(name))
        if defaulted:
            result = {**result, "defaults_used": defaulted}
    if not _result_is_failure(result) and isinstance(result.get("needs_input"), list) and result["needs_input"]:
        # A partial result: what the calculator could compute, and the inputs
        # it needed for the rest. Said in the input's own words; no value is
        # ever assumed for an input the caller did not give.
        result = dict(result)
        spec = formula_registry.get(str(name))
        params_spec = formula_registry.parameters(spec) if spec is not None else {}
        named = [(params_spec[p].label if p in params_spec and params_spec[p].label else str(p).replace("_", " "))
                 for p in result["needs_input"]]
        result["not_computed"] = (f"Not computed without the {', '.join(named)}: it was not given, "
                                  "and no value is assumed.")
    if _result_is_failure(result):
        if isinstance(result.get("needs_input"), list) and result["needs_input"]:
            env = _input_rejection_envelope(str(name), [], list(result["needs_input"]))
            # The calculator's own reason (e.g. "-120 was given") comes first:
            # the user sees why the value is asked for again.
            reason = str(result.get("error") or "").strip()
            if reason and reason not in str(env.get("error") or ""):
                env["error"] = f"{reason} {env.get('error') or ''}".strip()
            return env
        err = result["error"]
        if name in _CALC_REQUIRED_HELP and "needs" in err.lower():
            err = _CALC_REQUIRED_HELP[name]
        out = {
            "status": "error",
            "calculation": name,
            "error": err,
        }
        if isinstance(result.get("required"), list):
            out["required"] = result["required"]
        if isinstance(result.get("required_pairs"), list):
            out["required_pairs"] = result["required_pairs"]
        return out

    # Phase 2 F–W (#43–84) leftovers: stamp unit / unitless on the four
    # results the live probe scored as "number present, unit missing".
    try:
        from app.lib.construction_formulas_quantities import (
            FW_ROUTE_NAMES,
            shape_fw_calc_result,
        )
        if name in FW_ROUTE_NAMES:
            result = shape_fw_calc_result(name, result)
    except Exception:  # noqa: BLE001 — never break a successful calc
        logger.warning(
            "swallowed %s in shape_fw_calc_result() — continuing",
            "Exception", exc_info=True,
        )

    envelope = {
        "status": "success",
        "calculation": name,
        "result": result,
        "note": (
            "Deterministic engineering calculation. Any cost figure uses the unit "
            "rates provided (or indicative GCC defaults if none were given) — for a "
            "firm cost, supply the project's priced-BOQ / rate-schedule rates."
        ),
    }
    if conversions:
        envelope["conversions"] = conversions
    envelope.update(envelope_provenance(name, result, fn))
    return envelope


#: Result keys a formula uses to state the code / standard it applies.
_STANDARD_KEYS = ("standard", "code", "basis_standard", "reference", "source")


def envelope_provenance(name: str, result: Any, fn: Any = None) -> Dict[str, Any]:
    """``provenance`` / ``standard`` / ``certified`` for any calc envelope.

    provenance is ``kernel`` for a registered formula_registry kernel,
    ``reference_table`` for a kernel whose module is a reference-table module
    (indicative data, not a derivation), else ``improvised``. Only a kernel
    answer is labelled ``certified``. ``standard`` is the standard the result
    states (first of ``_STANDARD_KEYS``), else an explicit
    ``"none declared"`` -- every envelope carries the key.
    """
    from app.lib import formula_registry as _fr

    spec = _fr.get(str(name or ""))
    module = getattr(fn if fn is not None else getattr(spec, "fn", None), "__module__", "")
    if spec is None:
        provenance = "improvised"
    elif "reference_table" in module:
        provenance = "reference_table"
    else:
        provenance = "kernel"
    standard = ""
    if isinstance(result, dict):
        for key in _STANDARD_KEYS:
            val = result.get(key)
            if isinstance(val, str) and val.strip():
                standard = val.strip()
                break
    return {
        "provenance": provenance,
        "standard": standard or "none declared",
        "certified": provenance == "kernel",
    }
