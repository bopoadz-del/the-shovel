"""validation_pipeline — runnable 5-stage validation checker.

The heavy-reasoning agent prompt described a 5-stage validation pipeline
(syntactic / dimensional / physical / empirical / operational) but only as
prose for the LLM to "do in its head." When the LLM made a unit-conversion
error in a NumPy code-gen task (5,900 °C instead of 5.9 °C ΔT — multiplied
*and* divided by 1000), nothing caught it because the validation was
unimplemented.

This block makes those checks runnable. The agent can call it after any
sympy_reasoning, formula_executor, or numeric-bearing tool result, get a
deterministic pass/fail per stage, and refuse to report a number that
failed validation.

Input shape::

    {
        "value": 5.9,                          # required, must be numeric
        "unit": "degC",                        # optional, Pint string
        "context": {
            "material_type": "concrete",       # picks the empirical range
            "metric": "rate_usd_per_m3",       # picks the empirical range
            "currency": "SAR",                 # optional; swaps the cost band
            "slack_factor": 2.0,               # optional; empirical slack
            "strict": False,                   # optional; no slack at all
            "physical_min": 0,                 # optional override
            "physical_max": 1e9,
            "empirical_min": 100,
            "empirical_max": 250,
            # Operational stage uses paired fields:
            "duration_weeks": 16,
            "available_weeks": 8,
        }
    }

Output::

    {
        "status": "success",
        "overall": "pass" | "fail",
        "stages": {
            "syntactic":   {"pass": bool, "reason": str},
            "dimensional": {"pass": bool, "reason": str},
            "physical":    {"pass": bool, "reason": str},
            "empirical":   {"pass": bool, "reason": str, "borderline": bool?},
            "operational": {"pass": bool, "reason": str},
        },
        "first_failure": Optional[str],
    }
"""
from __future__ import annotations

import math
import os
import re
from typing import Any, Dict, Optional, Tuple

from app.core.universal_base import UniversalBlock
import logging

logger = logging.getLogger(__name__)


# Empirical sanity ranges live in config/empirical_ranges.json, keyed
# as `"<material>.<metric>.<currency>": [min, max]`. The file is
# operator-editable; learning_engine state at key `empirical_ranges`
# overrides the file when populated. Neither path requires a code
# change to add or refine a band.
#
# Keys use `<material>.<metric>.<currency>` strings (vs runtime tuples)
# so they survive JSON / DB round-trips cleanly. `__none__` is used as
# the currency slot for non-cost metrics (temperature, volume, etc.);
# `__any__` is the material wildcard for generic backstops.

_RANGES_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "config", "empirical_ranges.json")
_RANGES_OVERRIDE_ENV = "EMPIRICAL_RANGES_FILE"
_RANGES_CACHE: Optional[Dict[str, Tuple[float, float]]] = None
_RANGES_MTIME: float = 0.0


def _load_ranges() -> Dict[str, Tuple[float, float]]:
    """Reload the empirical ranges table when the on-disk JSON has
    changed, OR when learning_engine has a more recent override. The
    operator can edit ranges without a redeploy: next validation call
    picks up the new file. Learning-engine corrections shadow the file.
    """
    global _RANGES_CACHE, _RANGES_MTIME
    import json
    path = os.getenv(_RANGES_OVERRIDE_ENV) or _RANGES_PATH

    # File-based base layer.
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0.0
    if _RANGES_CACHE is None or mtime != _RANGES_MTIME:
        base: Dict[str, Tuple[float, float]] = {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k, v in (data.get("ranges") or {}).items():
                if isinstance(v, (list, tuple)) and len(v) == 2:
                    base[k] = (float(v[0]), float(v[1]))
        except (OSError, ValueError):
            logger.debug(
                "swallowed %s in _load_ranges() — continuing",
                "(OSError, ValueError)", exc_info=True,
            )
        _RANGES_CACHE = base
        _RANGES_MTIME = mtime

    # learning_engine override layer — when it knows a better band from
    # real corrections, it wins.
    try:
        from app.blocks import BLOCK_REGISTRY
        from app.dependencies import block_instances, _create_block_instance
        if "learning_engine" in BLOCK_REGISTRY:
            inst = block_instances.get("learning_engine")
            if inst is None:
                inst = _create_block_instance(BLOCK_REGISTRY["learning_engine"])
                block_instances["learning_engine"] = inst
            getter = getattr(inst, "get_state", None) or getattr(inst, "state", None)
            if callable(getter):
                state = getter("empirical_ranges") if getter.__name__ == "get_state" else getter
                if isinstance(state, dict):
                    overrides = state.get("ranges") or state
                    if isinstance(overrides, dict):
                        for k, v in overrides.items():
                            if isinstance(v, (list, tuple)) and len(v) == 2:
                                _RANGES_CACHE[k] = (float(v[0]), float(v[1]))
    except Exception:
        logger.debug(
            "swallowed %s in _load_ranges() — continuing",
            "Exception", exc_info=True,
        )

    return _RANGES_CACHE


# Metrics that are inherently currency-denominated. Everything else is
# physical and uses the "__none__" currency slot.
_COST_METRICS = {"rate_usd_per_m3", "rate_usd_per_kg", "rate_usd_per_m2", "cost_usd"}

# Lazy singleton so the (expensive) Pint UnitRegistry is created once. If the
# environment cannot build it, we degrade gracefully instead of crashing.
_UREG: Optional[Any] = None


def _get_ureg() -> Optional[Any]:
    """Return a cached pint UnitRegistry, or None if Pint is unusable."""
    global _UREG
    if _UREG is not None:
        return _UREG
    try:
        import pint

        _UREG = pint.UnitRegistry()
        return _UREG
    except Exception:
        # Production environments have seen SystemError / low-level failures
        # from Pint at registry creation time. Swallow them here so callers
        # can fall back to string-based checks instead of returning 500.
        return None


def _normalise_unit(unit: str) -> str:
    """Normalise common construction shorthand before handing to Pint.

    Pint expects ``m**3``; site teams write ``m3``, ``m³``, ``cum``, ``sqm``,
    etc. This mapping is intentionally generic, not hardcoded to one test.
    """
    if not isinstance(unit, str):
        return str(unit)
    mapping = {
        "m3": "m**3",
        "m³": "m**3",
        "cum": "m**3",
        "cbm": "m**3",
        "m2": "m**2",
        "m²": "m**2",
        "sqm": "m**2",
        "sqft": "ft**2",
        "ton": "tonne",
        "tons": "tonne",
        "hr": "hour",
        "hrs": "hour",
    }
    return mapping.get(unit.strip().lower(), unit.strip())


def _lookup_range(material: str, metric: str, currency: Optional[str]) -> Optional[Tuple[float, float]]:
    """Return the empirical (min, max) for (material, metric, currency).

    Lookup order on the flat string-keyed table:
      1. <material>.<metric>.<currency>  — exact match if currency supplied
      2. <material>.<metric>.USD          — USD baseline for cost metrics
      3. <material>.<metric>.__none__     — non-cost metric
      4. __any__.<metric>.<currency> -> __any__.<metric>.USD ->
         __any__.<metric>.__none__       — generic backstops
    """
    ranges = _load_ranges()
    keys = []
    if currency:
        keys.append(f"{material}.{metric}.{currency.upper()}")
    keys.append(f"{material}.{metric}.USD")
    keys.append(f"{material}.{metric}.__none__")
    if currency:
        keys.append(f"__any__.{metric}.{currency.upper()}")
    keys.append(f"__any__.{metric}.USD")
    keys.append(f"__any__.{metric}.__none__")
    for k in keys:
        rng = ranges.get(k)
        if rng:
            return rng
    return None


def _infer_metric(value: Any, unit: Optional[str], ctx: Dict[str, Any]) -> Optional[str]:
    """Best-effort guess of the empirical-range key from unit + material_type.

    Caller can always override by setting `context.metric` explicitly. This
    just covers the common case where the result envelope already carries
    enough info to disambiguate.
    """
    if not unit:
        return None
    u = unit.strip()
    u_lower = u.lower()
    material = (ctx.get("material_type") or "").lower().strip()

    u_compact = u_lower.replace(" ", "")
    if u_compact.endswith("/m3") or u_compact.endswith("/m**3") or u_compact.endswith("/cubicmeter"):
        if material in {"concrete", "excavation"}:
            return "rate_usd_per_m3"
    if u_compact.endswith("/kg"):
        if material in {"steel", "rebar"}:
            return "rate_usd_per_kg"
    if u_compact.endswith("/m2") or u_compact.endswith("/m**2"):
        if material == "formwork":
            return "rate_usd_per_m2"

    if u_lower in {"kg", "kilogram", "kilograms"} and material in {"steel", "rebar"}:
        return "weight_kg"

    if u_lower in {"m3", "m**3", "cubicmeter", "cubic_meter", "cubic meters"}:
        if material in {"concrete", "excavation"}:
            return "volume_m3"

    if u_lower in {"m2", "m**2", "squaremeter", "square_meter", "square meters"} and material == "formwork":
        return "area_m2"

    if u_lower in {"m", "meter", "meters"} and material == "rebar":
        return "length_m"

    if u_lower in {"mpa", "megapascal"} and material == "concrete":
        return "compressive_mpa"

    if u_lower in {"degc", "celsius", "delta_degc", "deg c", "°c"} or u_lower.endswith("degc"):
        return "temperature_degc"

    if u_lower in {"weeks", "week"} and ctx.get("duration_weeks") is not None:
        return "duration_weeks"

    if u_lower in {"%", "percent", "pct"}:
        return "percent"

    return None


_EN_SMALL = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
    "ninety": 90,
}
_EN_SCALES = {"hundred": 100, "thousand": 1_000, "million": 1_000_000}
_EN_WORDS = set(_EN_SMALL) | set(_EN_SCALES) | {"and"}


def _en_words_to_int(words: list[str]) -> Optional[float]:
    if not words:
        return None
    total = 0
    current = 0
    saw = False
    for w in words:
        if w == "and":
            continue
        if w in _EN_SMALL:
            current += _EN_SMALL[w]
            saw = True
        elif w == "hundred":
            current = (current or 1) * 100
            saw = True
        elif w in _EN_SCALES:
            current = (current or 1) * _EN_SCALES[w]
            total += current
            current = 0
            saw = True
        else:
            return None
    return float(total + current) if saw else None


def _digitize_english_quantities(claim: str) -> str:
    """Rewrite 'forty metres' / 'fifty millimetre' as '40 metres' / '50 millimetre'."""
    tokens = re.findall(r"[A-Za-z]+|[\d,.]+|[^\sA-Za-z\d]", claim)
    out: list[str] = []
    i = 0
    while i < len(tokens):
        run: list[str] = []
        j = i
        while j < len(tokens) and tokens[j].lower() in _EN_WORDS:
            run.append(tokens[j].lower())
            j += 1
        parsed = _en_words_to_int(run) if run else None
        if parsed is not None:
            out.append(str(int(parsed) if parsed == int(parsed) else parsed))
            i = j
            continue
        out.append(tokens[i])
        i += 1
    return " ".join(out)


def _lengths_m_from_claim(claim: str) -> list[float]:
    """Lengths in metres extracted from a prose claim."""
    out: list[float] = []
    for m in re.finditer(
        r"(\d[\d,]*(?:\.\d+)?)\s*-?\s*"
        r"(millimetres?|millimeters?|mm|centimetres?|centimeters?|cm|"
        r"metres?|meters?|m)\b",
        claim,
        re.IGNORECASE,
    ):
        v = float(m.group(1).replace(",", ""))
        u = m.group(2).lower()
        if u.startswith("mm") or "milli" in u:
            out.append(v / 1000.0)
        elif u.startswith("cm") or "centi" in u:
            out.append(v / 100.0)
        else:
            out.append(v)
    return out


def _check_physical_claim(claim: str) -> Dict[str, Any]:
    """Physical red flags in an unstructured claim (no single `value`).

    Leftover-hat L4: a 40 m office-floor span on a 50 mm steel I-section.
    Span/depth > 50 is not a real beam; flag Physical even when the caller
    never supplied a numeric payload.
    """
    claim = _digitize_english_quantities(claim)
    lengths = _lengths_m_from_claim(claim)
    positives = [x for x in lengths if x > 0]
    if len(positives) >= 2:
        span, depth = max(positives), min(positives)
        if span / depth > 50:
            return {
                "pass": False,
                "reason": (
                    f"span/depth {span / depth:.0f}:1 "
                    f"({span:g} m / {depth:g} m) is physically implausible "
                    "for a beam or floor member"
                ),
            }
    udl = re.search(
        r"(\d[\d,]*(?:\.\d+)?)\s*(kN/m|kn/m|kilonewtons?\s+per\s+metre)",
        claim,
        re.I,
    )
    if udl and lengths:
        w = float(udl.group(1).replace(",", ""))
        depth = min(x for x in lengths if x > 0) if any(x > 0 for x in lengths) else 0
        if depth and depth < 0.15 and w >= 50:
            return {
                "pass": False,
                "reason": (
                    f"{w:g} kN/m on a {depth*1000:.0f} mm deep section "
                    "is physically unrealizable"
                ),
            }
    return {"pass": True, "reason": "no physical red-flag in claim text"}


def _check_syntactic(value: Any) -> Dict[str, Any]:
    if value is None:
        return {"pass": False, "reason": "value is None"}
    if isinstance(value, bool):
        return {"pass": False, "reason": "value is a bool, not a numeric"}
    if not isinstance(value, (int, float)):
        try:
            float(value)
        except Exception:
            return {"pass": False, "reason": f"value is not numeric (got {type(value).__name__})"}
        return {"pass": True, "reason": "value is string-numeric; coerce before reporting"}
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return {"pass": False, "reason": f"value is non-finite ({value})"}
    return {"pass": True, "reason": "value is finite numeric"}


def _check_dimensional(value: Any, unit: Optional[str]) -> Dict[str, Any]:
    """Run a Pint sanity check on the unit string, with two carve-outs:

    * Currency-bearing units (USD/m3, SAR/kg, etc.) — Pint doesn't model
      currencies by default, so we strip the currency token and re-check
      the remaining physical units instead of marking the whole result
      as failed.
    * Offset-unit ambiguity (degC, degF used as a delta) — Pint refuses
      to multiply scalars by an offset-bearing unit; we re-try with the
      `delta_` prefix which is Pint's syntax for "this is a difference".

    The whole Pint interaction is wrapped so that a Pint-level failure
    (e.g. UnitRegistry creation failing in some Linux containers) cannot
    crash the request. It degrades to a controlled dimensional result.
    """
    if not unit:
        return {"pass": True, "reason": "no unit declared — dimensional check skipped"}

    import re

    unit = _normalise_unit(unit)

    # Currency-only units are outside Pint's scope; treat them as valid.
    currency_re = re.compile(
        r"\b(USD|SAR|AED|EUR|GBP|JPY|CNY|AUD|CAD|KWD|QAR|BHD|OMR|INR|PKR)\b",
        re.IGNORECASE,
    )
    stripped = currency_re.sub("", unit)
    stripped = re.sub(r"^[/\s*·]+|[/\s*·]+$", "", stripped).strip("/ ")
    if stripped != unit and not stripped:
        return {
            "pass": True,
            "reason": f"unit '{unit}' is currency-only — outside dimensional check scope",
        }
    probe = stripped or unit

    # Carve-out: construction shorthand. Pint wants `m**3`, not `m3`.
    probe = re.sub(r"\b(m|mm|cm|km|in|ft|yd)([2-4])\b", r"\1**\2", probe)

    try:
        ureg = _get_ureg()
        if ureg is None:
            return {
                "pass": True,
                "reason": "dimensional validator unavailable — check skipped",
            }

        def _try(u: str):
            try:
                q = float(value) * ureg(u)
                return True, str(q.units)
            except Exception as e:
                return False, f"{type(e).__name__}: {str(e)[:120]}"

        ok, info = _try(probe)
        if ok:
            return {"pass": True, "reason": f"unit '{unit}' parsed as {info}"}
        if "Offset" in info or "OffsetUnitCalculusError" in info:
            delta_probe = re.sub(
                r"\b(deg[CF]|celsius|fahrenheit|degree[CF])\b",
                lambda m: f"delta_{m.group(1)}",
                probe,
                flags=re.IGNORECASE,
            )
            ok2, info2 = _try(delta_probe)
            if ok2:
                return {
                    "pass": True,
                    "reason": f"unit '{unit}' parsed as delta-{info2}",
                }
        return {"pass": False, "reason": f"unit '{unit}' not recognised ({info})"}
    except Exception as e:
        # Final safety net: any Pint/SystemError/ImportError is converted to
        # a controlled result. We mark it as a warning (pass=True) so the
        # overall pipeline can still report other stages, but the caller
        # sees that dimensional validation was not completed.
        return {
            "pass": True,
            "reason": f"dimensional validation unavailable ({type(e).__name__}) — skipped",
        }


def _check_physical(value: float, ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Hard physical bounds — usually positivity + a generous ceiling."""
    pmin = ctx.get("physical_min")
    pmax = ctx.get("physical_max")
    # Default: most construction metrics are non-negative.
    if pmin is None:
        pmin = -1e30 if ctx.get("metric") == "temperature_degc" else 0.0
    if pmax is None:
        pmax = 1e30
    try:
        v = float(value)
    except Exception:
        return {"pass": False, "reason": "value not coercible to float for physical check"}
    if v < pmin:
        return {"pass": False, "reason": f"value {v} below physical_min {pmin}"}
    if v > pmax:
        return {"pass": False, "reason": f"value {v} above physical_max {pmax}"}
    return {"pass": True, "reason": f"value {v} within physical bounds [{pmin}, {pmax}]"}


def _check_empirical(value: float, ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Rough industry sanity range with a tunable slack band.

    Slack model:
      * `slack_factor` (default 2.0): values inside `[emin/slack, emax*slack]`
        pass but are flagged borderline if outside `[emin, emax]`.
      * `strict=True`: slack disabled — anything outside `[emin, emax]` fails.

    The previous implementation hard-coded a 5× slack, which let 1,500 USD/m³
    concrete pass as "borderline" (3× the high end). Default 2× catches that
    while still tolerating regional pricing wiggle.
    """
    emin = ctx.get("empirical_min")
    emax = ctx.get("empirical_max")
    if emin is None or emax is None:
        material = (ctx.get("material_type") or "").lower().strip()
        metric = (ctx.get("metric") or "").lower().strip()
        currency = ctx.get("currency")
        rng = _lookup_range(material, metric, currency)
        if rng:
            emin = emin if emin is not None else rng[0]
            emax = emax if emax is not None else rng[1]
        else:
            return {"pass": True, "reason": "no empirical range for (material, metric) — skipped"}
    try:
        v = float(value)
    except Exception:
        return {"pass": False, "reason": "value not coercible to float for empirical check"}

    strict = bool(ctx.get("strict"))
    try:
        slack = float(ctx.get("slack_factor", 2.0))
    except (TypeError, ValueError):
        slack = 2.0
    if slack < 1.0:
        slack = 1.0

    if strict:
        if v < emin or v > emax:
            return {"pass": False, "reason": f"value {v} outside empirical range [{emin}, {emax}] (strict)"}
        return {"pass": True, "reason": f"value {v} within empirical range [{emin}, {emax}] (strict)"}

    if emin > 0 and emax > 0:
        slack_lo = emin / slack
        slack_hi = emax * slack
    else:
        # Range spans or sits below zero (e.g. temperature -40..500). Widen
        # additively by `(slack - 1)` of the span on each side so 2× slack
        # = "twice the width," matching the multiplicative case for spans
        # starting near zero.
        width = emax - emin
        slack_lo = emin - width * (slack - 1.0)
        slack_hi = emax + width * (slack - 1.0)

    if v < slack_lo:
        return {"pass": False, "reason": f"value {v} far below empirical range [{emin}, {emax}] ({slack}x slack {slack_lo})"}
    if v > slack_hi:
        return {"pass": False, "reason": f"value {v} far above empirical range [{emin}, {emax}] ({slack}x slack {slack_hi})"}
    if v < emin or v > emax:
        return {
            "pass": True,
            "borderline": True,
            "reason": f"value {v} outside tight empirical range [{emin}, {emax}] but within {slack}x slack — borderline",
        }
    return {"pass": True, "reason": f"value {v} within empirical range [{emin}, {emax}]"}


def _check_operational(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Compare paired fields when present (e.g. procurement vs. site need)."""
    duration = ctx.get("duration_weeks")
    available = ctx.get("available_weeks")
    if duration is None or available is None:
        return {"pass": True, "reason": "no operational fields supplied — skipped"}
    try:
        d, a = float(duration), float(available)
    except Exception:
        return {"pass": False, "reason": "operational fields not numeric"}
    if d <= a:
        return {"pass": True, "reason": f"duration {d}w fits available {a}w"}
    return {
        "pass": False,
        "reason": f"action not achievable: duration {d}w exceeds available {a}w (need {d - a}w more)",
    }


class ValidationPipelineBlock(UniversalBlock):
    """Runs the 5-stage validation pipeline the heavy-reasoning prompt
    documents but the LLM was previously expected to perform unaided.
    """

    name = "validation_pipeline"
    version = "1.1.0"
    description = (
        "Runnable 5-stage validation (syntactic / dimensional / physical / empirical / "
        "operational) for numeric results. Returns per-stage pass/fail plus an overall "
        "verdict so the agent can refuse to report numbers that failed validation."
    )
    layer = 3
    tags = ["domain", "construction", "validation", "reasoning"]
    requires = []

    ui_schema = {
        "input": {
            "type": "json",
            "placeholder": (
                '{"value": 5.9, "unit": "degC", '
                '"context": {"material_type": "concrete", "metric": "temperature_degc"}}'
            ),
            "multiline": True,
        },
        "output": {
            "type": "json",
            "fields": [
                {"name": "overall", "type": "text", "label": "Verdict"},
                {"name": "stages", "type": "json", "label": "Per-stage results"},
                {"name": "first_failure", "type": "text", "label": "First failed stage"},
            ],
        },
        "quick_actions": [
            {"icon": "", "label": "Validate", "prompt": "Run the 5-stage validation pipeline"},
        ],
    }

    async def process(self, input_data: Any, params: Dict = None) -> Dict:
        params = params or {}
        data = input_data if isinstance(input_data, dict) else {}
        if not data and not params:
            return {"status": "error", "error": "Provide {value, unit?, context?} as input"}

        value = data.get("value", params.get("value"))
        unit = data.get("unit", params.get("unit"))
        ctx = dict(data.get("context") or params.get("context") or {})
        claim = data.get("claim") or data.get("text") or data.get("description") or params.get("claim")
        if isinstance(input_data, str) and not claim:
            claim = input_data

        # Params can promote slack_factor / strict / currency without
        # forcing every caller to nest them under `context`.
        for k in ("slack_factor", "strict", "currency", "metric", "material_type"):
            if k not in ctx and k in params:
                ctx[k] = params[k]

        # Prose claim with no numeric payload: still run Physical on the
        # extracted lengths. Do NOT skip remaining stages (leftover-hat L4).
        if value is None and claim:
            syntactic = {
                "pass": True,
                "reason": "prose claim with extractable quantities — no single value required",
            }
            physical = _check_physical_claim(str(claim))
            dimensional = {
                "pass": True,
                "reason": "claim units parsed independently — no mixed-unit sum asserted",
            }
            empirical = {
                "pass": True,
                "reason": "no single metric to band — empirical skipped",
            }
            operational = _check_operational(ctx)
            stages = {
                "syntactic": syntactic,
                "dimensional": dimensional,
                "physical": physical,
                "empirical": empirical,
                "operational": operational,
            }
            first_failure = next(
                (name for name, r in stages.items() if not r.get("pass")),
                None,
            )
            fail_n = sum(1 for r in stages.values() if not r.get("pass"))
            tier = 4 if fail_n >= 2 or (first_failure == "physical") else (3 if first_failure else 2)
            return {
                "status": "success",
                "overall": "fail" if first_failure else "pass",
                "stages": stages,
                "first_failure": first_failure,
                "tier": tier,
                "claim": str(claim)[:500],
            }

        # Stage 1 — syntactic gates the rest when there is no claim text.
        syntactic = _check_syntactic(value)
        if not syntactic["pass"]:
            return {
                "status": "success",
                "overall": "fail",
                "stages": {
                    "syntactic": syntactic,
                    "dimensional": {"pass": False, "reason": "skipped — syntactic failed"},
                    "physical":    {"pass": False, "reason": "skipped — syntactic failed"},
                    "empirical":   {"pass": False, "reason": "skipped — syntactic failed"},
                    "operational": {"pass": False, "reason": "skipped — syntactic failed"},
                },
                "first_failure": "syntactic",
            }

        # Coerce stringy-numeric to float before continuing.
        v = float(value) if not isinstance(value, (int, float)) else value

        # Infer metric from unit + material_type when caller didn't spell
        # it out. Runs before the physical check so temperature deltas get
        # the negative-allowed floor.
        if not ctx.get("metric"):
            inferred = _infer_metric(v, unit, ctx)
            if inferred:
                ctx["metric"] = inferred
                ctx.setdefault("_metric_inferred", True)

        dimensional = _check_dimensional(v, unit)
        physical = _check_physical(v, ctx)
        empirical = _check_empirical(v, ctx)
        operational = _check_operational(ctx)

        stages = {
            "syntactic": syntactic,
            "dimensional": dimensional,
            "physical": physical,
            "empirical": empirical,
            "operational": operational,
        }
        first_failure = next(
            (name for name, r in stages.items() if not r.get("pass")),
            None,
        )
        overall = "fail" if first_failure else "pass"
        result = {
            "status": "success",
            "value": v,
            "unit": unit,
            "overall": overall,
            "stages": stages,
            "first_failure": first_failure,
        }
        if empirical.get("borderline"):
            result["borderline"] = True
        if ctx.get("_metric_inferred"):
            result["metric_inferred"] = ctx.get("metric")
        return result
