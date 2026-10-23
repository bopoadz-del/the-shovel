"""Primary result field and formatter for every qaqc calculator.

Each successful result carries ``primary_result`` with value, unit and
label. ``format_result`` always writes that field first. Applied by
``qaqc_calculator`` to every registered formula in the package.
"""
from __future__ import annotations

import functools
import inspect
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Dict, Optional

from app.agents.hats.qaqc.validation import first_out_of_range


def _output_label(key: str, unit: str) -> str:
    from app.lib.source_labels import parameter_words

    words = parameter_words(key, unit).replace("_", " ").strip()
    return words or key.replace("_", " ")


def _as_mapping(raw: Any) -> Any:
    if is_dataclass(raw) and not isinstance(raw, type):
        return asdict(raw)
    return raw


def _infer_primary(spec: Any, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    value = result.get("value")
    unit = result.get("unit")
    if (
        value is not None
        and not isinstance(value, (bool, dict, list))
        and isinstance(unit, str)
        and unit.strip()
        and unit.strip().lower() != "currency"
    ):
        label = result.get("label") or (getattr(spec, "display_name", None) or "result")
        return {"value": value, "unit": unit, "label": label}

    inputs = set(getattr(spec, "inputs", None) or {})
    outputs = getattr(spec, "outputs", None) or {}
    for key, out_unit in outputs.items():
        if key in inputs:
            continue
        val = result.get(key)
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            continue
        return {
            "value": val,
            "unit": out_unit,
            "label": _output_label(key, out_unit),
        }
    return None


def attach_primary(spec: Any, raw: Any) -> Any:
    """Add ``primary_result`` to a mapping result. Scalars are left unchanged."""
    result = _as_mapping(raw)
    if not isinstance(result, dict):
        return raw
    if result.get("error") or result.get("status") == "out_of_range":
        return result
    existing = result.get("primary_result")
    if (
        isinstance(existing, dict)
        and existing.get("value") not in (None, "")
        and existing.get("label")
    ):
        return result
    primary = _infer_primary(spec, result)
    if primary is None:
        return result
    result["primary_result"] = primary
    result.setdefault("value", primary["value"])
    result.setdefault("unit", primary["unit"])
    result.setdefault("label", primary["label"])
    return result


def ensure_primary(spec: Any, raw: Any) -> Dict[str, Any]:
    """The same primary field the formatter reads, including a scalar result."""
    attached = attach_primary(spec, raw)
    if isinstance(attached, dict):
        if attached.get("primary_result") or attached.get("error"):
            return attached
        primary = _infer_primary(spec, attached)
        if primary is not None:
            attached = dict(attached)
            attached["primary_result"] = primary
        return attached
    outputs = getattr(spec, "outputs", None) or {}
    key, unit = next(iter(outputs.items()), ("value", "-"))
    label = _output_label(key, unit)
    return {
        key: attached,
        "value": attached,
        "unit": unit,
        "label": label,
        "primary_result": {"value": attached, "unit": unit or "-", "label": label},
    }


def format_result(result: Any) -> str:
    """Render a calculator result with the primary field on the first line."""
    data = result if isinstance(result, dict) else {"value": result}
    lines = []
    primary = data.get("primary_result")
    if isinstance(primary, dict) and primary.get("value") not in (None, ""):
        label = str(primary.get("label") or "result").strip()
        unit = str(primary.get("unit") or "").strip()
        value = primary["value"]
        if unit and unit != "-":
            lines.append(f"{label}: {value} {unit}")
        else:
            lines.append(f"{label}: {value}")
    note = data.get("note")
    if isinstance(note, str) and note.strip():
        lines.append(note.strip())
    notes = data.get("notes")
    if isinstance(notes, list):
        for item in notes:
            if str(item).strip():
                lines.append(str(item).strip())
    return "\n".join(lines)


def _bound_values(fn: Callable[..., Any], args: tuple, kwargs: dict) -> Dict[str, Any]:
    try:
        bound = inspect.signature(fn).bind_partial(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)
    except TypeError:
        return dict(kwargs)


def qaqc_calculator(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Hold every call to the declared ranges and stamp a primary result."""

    @functools.wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        spec = getattr(wrapped, "__formula__", None)
        if spec is not None:
            from app.lib.formula_registry import parameters

            rejected = first_out_of_range(parameters(spec), _bound_values(fn, args, kwargs))
            if rejected:
                return rejected
        return attach_primary(spec, fn(*args, **kwargs))

    return wrapped
