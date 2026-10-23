"""Shared input checks for every qaqc calculator.

A supplied figure is held to the range declared on its param. The same
rule applies to every calculator in the package: there is no per-formula
branch. A physical size is a param whose unit is a length. A specimen
size is a length used as a test-specimen dimension; its declared range
is the published concrete specimen sizes (EN 12390-1 / ASTM C39).
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

# Published concrete specimen sizes, millimetres: cubes 100/150/200,
# cylinders 100/150/200/250/300 (EN 12390-1 and ASTM C39).
STANDARD_SPECIMEN_MM = (100.0, 300.0)

_LENGTH_UNITS = frozenset({
    "mm", "m", "cm", "km", "um", "µm", "micrometer", "micrometre",
    "in", "inch", "inches", "ft", "yd",
})
_SPECIMEN_MARKERS = ("specimen", "cylinder", "cube")


def _norm_unit(unit: str) -> str:
    return (unit or "").strip().lower().replace(" ", "")


def is_physical_size(param: Any) -> bool:
    return _norm_unit(getattr(param, "unit", "")) in _LENGTH_UNITS


def is_specimen_size(param: Any) -> bool:
    if not is_physical_size(param):
        return False
    text = f"{getattr(param, 'name', '')} {getattr(param, 'label', '')}".lower()
    return any(marker in text for marker in _SPECIMEN_MARKERS)


def _in_range(param: Any, number: float) -> bool:
    lo = getattr(param, "lo", None)
    hi = getattr(param, "hi", None)
    return (lo is None or number >= lo) and (hi is None or number <= hi)


def _numbers(value: Any) -> list:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        return [
            float(item)
            for item in value
            if isinstance(item, (int, float)) and not isinstance(item, bool)
        ]
    return []


def out_of_range_result(param: Any, value: Any) -> Dict[str, Any]:
    """Structured rejection that states the declared range and asks to confirm."""
    label = (getattr(param, "label", None) or getattr(param, "name", None) or "value")
    label = str(label).strip() or "value"
    unit = getattr(param, "unit", "") or ""
    lo = getattr(param, "lo", None)
    hi = getattr(param, "hi", None)
    unit_bit = f" {unit}" if unit and unit not in ("-", "") else ""
    question = (
        f"{label} must be between {lo:g} and {hi:g}{unit_bit}. "
        f"Please confirm the {label}."
    )
    return {
        "status": "out_of_range",
        "error": question,
        "question": question,
        "parameter": getattr(param, "name", ""),
        "value": value,
        "lo": lo,
        "hi": hi,
        "unit": unit,
        "needs_confirmation": True,
    }


def first_out_of_range(
    params: Mapping[str, Any],
    values: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """The first supplied figure that lies outside its declared range."""
    for name, param in params.items():
        if name not in values or values[name] is None:
            continue
        if getattr(param, "lo", None) is None and getattr(param, "hi", None) is None:
            continue
        numbers = _numbers(values[name])
        if not numbers:
            continue
        kind = getattr(param, "kind", "") or ""
        for number in numbers:
            if not _in_range(param, number):
                shown = values[name] if kind == "series" else number
                return out_of_range_result(param, shown)
    return None
