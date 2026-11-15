"""Exam D1 checker proven in both directions, on invented envelopes and an
invented calculator (no registry calculator, no product output used as an
expected value)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tests.calc_entry_exam import checker as ck


# ---------------------------------------------------------------- rule M
def test_missing_passes_when_the_input_is_asked_for_by_name_or_label():
    asked = [
        {"status": "needs_input", "needs_input": ["trench_depth_m"]},
        {"status": "error", "error": "Bad parameters: missing required Panel height (m)."},
    ]
    assert ck.check_missing(asked[0], "trench_depth_m", "")[0]
    assert ck.check_missing(asked[1], "panel_h", "panel height")[0]


def test_missing_fails_when_run_anyway_or_input_not_named():
    ran = {"status": "success", "result": {"volume_m3": 12.5}}
    unnamed = {"status": "needs_input", "error": "more inputs needed"}
    with_number = {"status": "needs_input", "needs_input": ["x_m"], "result": {"y": 4.0}}
    assert not ck.check_missing(ran, "x_m", "")[0]
    assert not ck.check_missing(unnamed, "x_m", "x length")[0]
    assert not ck.check_missing(with_number, "x_m", "")[0]


# ---------------------------------------------------------------- rule U
def test_same_result_passes_when_equal_within_tolerance():
    base = {"status": "success", "result": {"a": 7.25, "b": {"c": 3}}}
    assert ck.check_same_result(base, {"status": "success", "result": {"a": 7.25, "b": {"c": 3.0}}}, "u")[0]
    assert ck.check_same_result(base, {"status": "success", "result": {"a": 7.2500000001, "b": {"c": 3}}}, "u")[0]


def test_same_result_fails_on_unscaled_value_or_refusal():
    base = {"status": "success", "result": {"a": 7.25}}
    assert not ck.check_same_result(base, {"status": "success", "result": {"a": 7250.0}}, "u")[0]
    assert not ck.check_same_result(base, {"status": "error", "error": "unit"}, "u")[0]
    assert not ck.check_same_result(base, {"status": "success", "result": {}}, "u")[0]


# ---------------------------------------------------------------- rule D
def test_disclosed_passes_when_every_default_is_named():
    env = {"status": "success", "result": {"v": 1, "defaults_used": [
        {"parameter": "loss_fraction", "value": 0.05}, {"parameter": "cover_mm", "value": 40}]}}
    assert ck.check_disclosed(env, {"loss_fraction", "cover_mm"})[0]
    assert ck.check_disclosed(env, {"cover_mm"})[0]


def test_disclosed_fails_when_a_default_is_silent():
    partial = {"status": "success", "result": {"v": 1, "defaults_used": [{"parameter": "cover_mm"}]}}
    none = {"status": "success", "result": {"v": 1}}
    assert not ck.check_disclosed(partial, {"loss_fraction", "cover_mm"})[0]
    assert not ck.check_disclosed(none, {"cover_mm"})[0]


# ---------------------------------------------------------------- the sweep, on an invented calculator
@dataclass(frozen=True)
class _P:
    kind: str
    unit: str
    lo: Any = None
    hi: Any = None
    label: str = ""
    required: bool = True


@dataclass
class _Spec:
    name: str
    fn: Any
    params: dict = field(default_factory=dict)
    required_groups: tuple = ()


def _slab(length_m, width_m, thick_mm, loss_fraction=0.05):
    return length_m * width_m * thick_mm / 1000 * (1 + loss_fraction)


_PARAMS = {
    "length_m": _P("number", "m", 1, 100, "length"),
    "width_m": _P("number", "m", 1, 100, "width"),
    "thick_mm": _P("number", "mm", 50, 600, "thickness"),
    "loss_fraction": _P("number", "fraction", 0, 0.2, required=False),
}


def _to_number(v: Any, declared: str) -> float:
    if isinstance(v, str):
        num, unit = v.split()
        num = float(num)
        if unit == declared:
            return num
        other, factor = ck.OTHER_UNIT[declared]
        assert unit == other
        return num / factor
    return float(v)


def _good_door(name, params):
    missing = [k for k, p in _PARAMS.items() if p.required and k not in params]
    if missing:
        return {"status": "needs_input", "needs_input": missing}
    vals = {k: _to_number(v, _PARAMS[k].unit) for k, v in params.items()}
    rows = [] if "loss_fraction" in vals else [{"parameter": "loss_fraction", "value": 0.05}]
    return {"status": "success", "result": {"volume_m3": _slab(**vals), "defaults_used": rows}}


def _bad_door(name, params):
    # Runs on a zero for a missing input, reads "X mm" as metres' number, and
    # never discloses the default: every rule must catch it.
    vals = {k: float(str(params.get(k, 0)).split()[0]) for k in ("length_m", "width_m", "thick_mm")}
    return {"status": "success", "result": {"volume_m3": _slab(**vals)}}


def _sweep(door):
    spec = _Spec("slab", _slab, _PARAMS)
    return ck.sweep([spec], lambda s: {k: _named(k, p) for k, p in s.params.items()}, door)


def _named(key, p):
    return type("PS", (), {"name": key, "kind": p.kind, "unit": p.unit, "lo": p.lo, "hi": p.hi,
                           "label": p.label, "required": p.required})()


def test_sweep_passes_a_door_that_keeps_all_three_rules():
    rep = _sweep(_good_door)
    assert rep["passed"] == rep["total"], [r for r in rep["rows"] if not r["passed"]]
    assert rep["by_rule"] == {"baseline": [1, 1], "M": [3, 3], "U": [3, 3], "D": [1, 1]}


def test_sweep_fails_a_door_that_breaks_each_rule():
    rep = _sweep(_bad_door)
    assert rep["by_rule"]["M"] == [0, 3]
    assert rep["by_rule"]["U"][0] == 0
    assert rep["by_rule"]["D"] == [0, 1]


def test_sweep_reports_the_stage_when_a_calculator_cannot_be_exercised():
    blocked = {**_PARAMS, "mix": _P("text", "-", label="mix")}

    def needs_mix(length_m, width_m, thick_mm, mix, loss_fraction=0.05):
        return 0.0

    refuses = _Spec("refuses", _slab, _PARAMS)
    rep = ck.sweep([_Spec("needs_mix", needs_mix, blocked), refuses],
                   lambda s: {k: _named(k, p) for k, p in s.params.items()},
                   lambda n, p: {"status": "error", "error": "refused"})
    stages = [(r["calc"], r["rule"], r["passed"]) for r in rep["rows"]]
    assert stages == [("needs_mix", "baseline", False), ("refuses", "baseline", False)]
