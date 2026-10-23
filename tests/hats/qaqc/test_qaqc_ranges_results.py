"""Package rules for qaqc calculators: declared size ranges, out-of-range
confirmation, and a primary result on every calculator.

Synthetic figures only. No project, client or person names.
"""
from __future__ import annotations

import inspect

import pytest

from app.agents.hats.qaqc.results import ensure_primary, format_result
from app.agents.hats.qaqc.validation import (
    STANDARD_SPECIMEN_MM,
    is_physical_size,
    is_specimen_size,
)
from app.lib.formula_registry import NUMERIC_KINDS, all_specs, parameters

LANE = "qaqc"


def _lane_specs():
    return [spec for spec in all_specs() if spec.owner == LANE]


def _lane_params():
    return [
        (spec, name, p)
        for spec in _lane_specs()
        for name, p in parameters(spec).items()
    ]


def _ids(rows):
    return [f"{spec.name}.{name}" for spec, name, _p in rows]


def _synthetic(spec, skip=()):
    """In-range figures taken from each param's own schema, never a named case."""
    values = {}
    for name, p in parameters(spec).items():
        if name in skip:
            continue
        if p.kind == "series":
            lo = 0.0 if p.lo is None else p.lo
            hi = 10.0 if p.hi is None else p.hi
            mid = (lo + hi) / 2.0
            if mid <= 0 and hi > 0:
                mid = hi / 2.0
            values[name] = [mid, mid]
        elif p.kind == "integer":
            if p.lo is not None and p.hi is not None:
                values[name] = int((p.lo + p.hi) / 2.0)
            else:
                values[name] = int(p.default if p.default is not None else 1)
        elif p.kind in NUMERIC_KINDS:
            if p.lo is not None and p.hi is not None:
                values[name] = (p.lo + p.hi) / 2.0
            else:
                values[name] = 1.0
        elif p.kind == "flag":
            values[name] = False
        elif not p.required:
            continue
        else:
            values[name] = p.default if p.default not in (None, "") else "x"
    return values


def _outside(p):
    span = 1.0 if p.hi is None or p.lo is None else max(p.hi - p.lo, 1.0)
    return p.hi + span


PHYSICAL_SIZE_PARAMS = [
    (spec, name, p) for spec, name, p in _lane_params() if is_physical_size(p)
]
SPECIMEN_SIZE_PARAMS = [
    (spec, name, p) for spec, name, p in _lane_params() if is_specimen_size(p)
]


@pytest.mark.parametrize("spec,name,p", PHYSICAL_SIZE_PARAMS, ids=_ids(PHYSICAL_SIZE_PARAMS))
def test_every_physical_size_param_declares_a_typical_range(spec, name, p):
    assert p.lo is not None and p.hi is not None, f"{spec.name}.{name} has no typical range"
    assert p.lo < p.hi


@pytest.mark.parametrize("spec,name,p", SPECIMEN_SIZE_PARAMS, ids=_ids(SPECIMEN_SIZE_PARAMS))
def test_specimen_size_range_is_the_published_specimen_range(spec, name, p):
    assert (p.lo, p.hi) == STANDARD_SPECIMEN_MM


@pytest.mark.parametrize("spec,name,p", PHYSICAL_SIZE_PARAMS, ids=_ids(PHYSICAL_SIZE_PARAMS))
def test_size_outside_the_declared_range_states_the_range_and_asks_to_confirm(spec, name, p):
    values = _synthetic(spec, skip=(name,))
    values[name] = _outside(p)
    out = spec.fn(**values)
    assert isinstance(out, dict), f"{spec.name} accepted {_outside(p)} {p.unit} without a result"
    text = " ".join(
        str(out.get(key) or "") for key in ("error", "question")
    ).lower()
    assert "must be between" in text
    assert f"{p.lo:g}" in text.replace(",", "")
    assert f"{p.hi:g}" in text.replace(",", "")
    assert (p.unit or "").lower() in text
    assert "confirm" in text


@pytest.mark.parametrize("spec", _lane_specs(), ids=lambda s: s.name)
def test_every_registered_calculator_returns_a_primary_result(spec):
    values = _synthetic(spec)
    raw = spec.fn(**values)
    if isinstance(raw, dict) and raw.get("error"):
        pytest.fail(f"{spec.name} rejected valid synthetic inputs: {raw}")
    result = raw if isinstance(raw, dict) else ensure_primary(spec, raw)
    primary = result.get("primary_result")
    assert isinstance(primary, dict), f"{spec.name} returned no primary_result"
    assert primary.get("value") not in (None, "")
    assert primary.get("label")
    unit = primary.get("unit")
    assert unit is not None and str(unit).strip() != ""


@pytest.mark.parametrize("spec", _lane_specs(), ids=lambda s: s.name)
def test_result_formatter_emits_the_primary_result_first(spec):
    values = _synthetic(spec)
    raw = spec.fn(**values)
    if isinstance(raw, dict) and raw.get("error"):
        pytest.fail(f"{spec.name} rejected valid synthetic inputs: {raw}")
    result = raw if isinstance(raw, dict) else ensure_primary(spec, raw)
    primary = result["primary_result"]
    rendered = format_result(result)
    first = rendered.splitlines()[0]
    assert str(primary["label"]) in first
    assert str(primary["value"]) in first
    assert inspect.isfunction(format_result)
