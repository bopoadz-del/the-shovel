"""Defaulted inputs: omit them, the calculator still runs, and they are named.

A signature default is a code or instrument figure, not a required
measurement. The wrapper records every default Python applied and writes
it on the result so a caller can see what was not supplied.
"""
from __future__ import annotations

import functools
import inspect
import logging
import re
from dataclasses import asdict, is_dataclass, replace
from typing import Any, Callable, Dict, List

_LOG = logging.getLogger(__name__)

from app.lib.formula_registry import NUMERIC_KINDS

OWNER = "safety"

_NOT_DIMENSIONAL = frozenset({
    "", "-", "ratio", "count", "fraction", "%", "currency", "no", "nr",
})
_STRESS_UNITS = frozenset({
    "n/mm2", "n/mm²", "mpa", "kpa", "kn/m2", "kn/m²", "n/m2", "psi",
})
_HOUR_AGE_LABEL = re.compile(r"\b\d+\s*-?\s*h(?:ou)?rs?\b", re.I)


def _norm_unit(unit: str) -> str:
    return (unit or "").strip().lower().replace("²", "2").replace("³", "3")


def is_dimensional(unit: str, kind: str = "number") -> bool:
    """True when this input is a physical size, not a count, ratio or flag."""
    if kind not in NUMERIC_KINDS:
        return False
    return _norm_unit(unit) not in _NOT_DIMENSIONAL


def encodes_hour_age_on_stress(name: str, unit: str, label: str = "") -> bool:
    """A stress input whose label freezes a test age in hours.

    The identifier may still carry a CORE-coupled suffix; the words a user
    reads are the label, and those name the quantity.
    """
    if _norm_unit(unit) not in _STRESS_UNITS:
        return False
    return bool(_HOUR_AGE_LABEL.search(label or ""))


def _usable_default(value: Any) -> bool:
    if value is None or value == "":
        return False
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)) and float(value) == 0.0:
        return False
    return True


def applied_defaults(fn: Callable[..., Any], given: Dict[str, Any]) -> Dict[str, Any]:
    """Signature defaults applied because ``given`` did not supply them."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        _LOG.warning("signature unreadable for %s; no defaults named", getattr(fn, "__name__", fn), exc_info=True)
        return {}
    supplied = set(given or ())
    out: Dict[str, Any] = {}
    for name, param in sig.parameters.items():
        if param.kind not in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
            continue
        if param.default is inspect.Parameter.empty or name in supplied:
            continue
        if _usable_default(param.default):
            out[name] = param.default
    return out


def _rows(fn: Callable[..., Any], used: Dict[str, Any]) -> List[Dict[str, Any]]:
    spec = getattr(fn, "__formula__", None)
    name = spec.name if spec is not None else getattr(fn, "__name__", "")
    from app.lib.source_labels import parameter_label

    return [
        {"parameter": key, "value": value, "label": parameter_label(name, key)}
        for key, value in used.items()
    ]


def _attach(fn: Callable[..., Any], result: Any, used: Dict[str, Any]) -> Any:
    if not used:
        return result
    if is_dataclass(result) and not isinstance(result, type):
        result = asdict(result)
    if not isinstance(result, dict):
        result = {"value": result}
    if result.get("error") or result.get("status") in ("error", "out_of_range"):
        return result
    for key, value in used.items():
        result.setdefault(key, value)
    result["defaults_used"] = _rows(fn, used)
    spec = getattr(fn, "__formula__", None)
    calc = spec.name if spec is not None else getattr(fn, "__name__", "")
    from app.lib.source_labels import default_sentence

    sentences = [default_sentence(calc, key, value) for key, value in used.items()]
    notes = result.get("notes")
    if isinstance(notes, list):
        result["notes"] = list(notes) + sentences
    elif isinstance(result.get("note"), str):
        extra = " ".join(sentences)
        result["note"] = (result["note"] + " " + extra).strip() if result["note"] else extra
    elif sentences:
        result["defaults_note"] = " ".join(sentences)
    return result


def disclose_defaults(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Run ``fn`` and name every signature default that Python applied."""
    if getattr(fn, "_lane_defaults_wrapped", False):
        return fn
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        _LOG.warning("signature unreadable for %s; disclosure wrap skipped", getattr(fn, "__name__", fn), exc_info=True)
        return fn

    @functools.wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        bound = sig.bind_partial(*args, **kwargs)
        used = applied_defaults(fn, dict(bound.arguments))
        return _attach(fn, fn(*args, **kwargs), used)

    wrapped._lane_defaults_wrapped = True  # type: ignore[attr-defined]
    wrapped.__signature__ = sig  # type: ignore[attr-defined]
    return wrapped


def install_default_disclosure(owner: str = OWNER) -> None:
    """Wrap every registered formula of ``owner`` so defaults are named."""
    from app.lib import formula_registry

    for key, spec in list(formula_registry._REGISTRY.items()):
        if spec.owner != owner:
            continue
        if getattr(spec.fn, "_lane_defaults_wrapped", False):
            continue
        wrapped = disclose_defaults(spec.fn)
        formula_registry._REGISTRY[key] = replace(spec, fn=wrapped)
        wrapped.__formula__ = formula_registry._REGISTRY[key]  # type: ignore[attr-defined]
