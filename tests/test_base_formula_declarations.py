"""Every base formula states each input in plain words and a unit, and
reports a rejected input as data.

One rule for every formula owned by the base package, discovered from the
registry. A formula added later is covered without naming it here.

* every input has a plain label (words a person reads, not a parameter id);
* a count (a whole number, or an input whose label is "number of …") declares
  a count unit the shared converter reads;
* a factor whose declared range lies between 0 and 1 declares a fraction;
* any other declared unit is one the shared converter reads;
* a numeric input with no fixed dimension of its own may stay unmarked;
* a formula returns an error mapping. It does not raise.
"""
from __future__ import annotations

import ast
import inspect
import re

import pytest

from app.agents.base.formulas.construction_formulas_planning import parse_unit
from app.lib.formula_registry import NUMERIC_KINDS, all_specs, parameters

_PLAIN_LABEL = re.compile(r"[A-Za-z]")
_NUMBER_OF = re.compile(r"^number of\b", re.IGNORECASE)


def _base_specs():
    return [spec for spec in all_specs() if spec.owner == "base"]


def _inputs():
    return [
        (spec, name, param)
        for spec in _base_specs()
        for name, param in parameters(spec).items()
    ]


def _is_fraction_range(param) -> bool:
    return (
        param.kind in NUMERIC_KINDS
        and param.kind != "integer"
        and param.lo is not None
        and param.hi is not None
        and param.lo >= 0
        and param.hi <= 1
    )


def _is_count(name: str, param) -> bool:
    if param.kind not in NUMERIC_KINDS:
        return False
    if param.kind == "integer":
        return True
    return bool(_NUMBER_OF.match((param.label or "").strip()))


def _unmarked(unit: str) -> bool:
    return (unit or "").strip() in ("", "-")


def _function_raises(fn) -> bool:
    tree = ast.parse(inspect.getsource(fn))
    return any(isinstance(node, ast.Raise) for node in ast.walk(tree))


def _out_of_range_call(spec):
    kwargs = {}
    for name, param in parameters(spec).items():
        if param.kind == "text":
            kwargs[name] = "x"
        elif param.kind == "integer":
            kwargs[name] = -1
        elif param.kind in NUMERIC_KINDS:
            kwargs[name] = -1.0
        else:
            kwargs[name] = None
    return kwargs


@pytest.mark.parametrize(
    "spec,name,param",
    _inputs(),
    ids=[f"{spec.name}.{name}" for spec, name, _param in _inputs()],
)
def test_every_base_input_has_a_plain_label(spec, name, param):
    label = (param.label or "").strip()
    assert label, f"{spec.name}.{name} has no label"
    assert "_" not in label, f"{spec.name}.{name} label {label!r} is a parameter id"
    assert _PLAIN_LABEL.search(label), f"{spec.name}.{name} label {label!r} is not words"


@pytest.mark.parametrize(
    "spec,name,param",
    _inputs(),
    ids=[f"{spec.name}.{name}" for spec, name, _param in _inputs()],
)
def test_every_base_input_declares_a_unit_for_its_kind(spec, name, param):
    unit = (param.unit or "").strip()
    if param.kind not in NUMERIC_KINDS:
        return
    if _unmarked(unit):
        assert not _is_count(name, param), (
            f"{spec.name}.{name} is a count and declares no unit"
        )
        assert not _is_fraction_range(param), (
            f"{spec.name}.{name} is a fraction and declares no unit"
        )
        return
    parsed = parse_unit(unit)
    assert parsed is not None, f"{spec.name}.{name} declares {unit!r}, which the converter does not read"
    if _is_count(name, param):
        assert parsed.dimension == "count", f"{spec.name}.{name} unit {unit!r} is not a count"
    if _is_fraction_range(param):
        assert parsed.dimension == "fraction", f"{spec.name}.{name} unit {unit!r} is not a fraction"


@pytest.mark.parametrize("spec", _base_specs(), ids=lambda spec: spec.name)
def test_every_base_formula_returns_an_error_instead_of_raising(spec):
    assert not _function_raises(spec.fn), f"{spec.name} raises; an error is returned"
    try:
        result = spec.fn(**_out_of_range_call(spec))
    except Exception as exc:  # noqa: BLE001 — the assertion is that this does not happen
        pytest.fail(f"{spec.name} raised {type(exc).__name__}: {exc}")
    if isinstance(result, dict) and "error" in result:
        assert isinstance(result["error"], str) and result["error"].strip(), result
        assert not isinstance(result["error"], BaseException)
