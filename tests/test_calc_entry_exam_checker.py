"""Exam D1 checker proven in both directions, on invented envelopes and an
invented calculator (no registry calculator, no product output used as an
expected value)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

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


#: invented module-level norm the slab reads (rule C must see it)
_SLAB_LAP = 1.08


def _slab(length_m, width_m, thick_mm, loss_fraction=0.05):
    return length_m * width_m * thick_mm / 1000 * (1 + loss_fraction) * _SLAB_LAP


def _beam(span_m, udl_kn_m, point_kn, load_factor=1.5, extra_knm=2.0):
    return load_factor * (udl_kn_m * span_m ** 2 / 8 + point_kn * span_m / 4) + extra_knm


@dataclass(frozen=True)
class _Calc:
    name: str
    fn: Any
    params: dict
    defaults: dict  # optional input -> the value the calculator falls back on
    formula: str = ""  # the calculator's written basis, constants included


# Two invented calculators with different units, different required inputs
# and a different number of fall-back defaults: every sweep decision below is
# taken on both.
_SLAB = _Calc("slab", _slab, {
    "length_m": _P("number", "m", 1, 100, "length"),
    "width_m": _P("number", "m", 1, 100, "width"),
    "thick_mm": _P("number", "mm", 50, 600, "thickness"),
    "loss_fraction": _P("number", "fraction", 0, 0.2, required=False),
}, {"loss_fraction": 0.05}, "L x W x t / 1000 x (1 + loss) x lap 1.08")
_BEAM = _Calc("beam", _beam, {
    "span_m": _P("number", "m", 2, 20, "span"),
    "udl_kn_m": _P("number", "kN/m", 1, 80, "distributed load"),
    "point_kn": _P("number", "kN", 0, 300, "point load"),
    "load_factor": _P("number", "-", 1, 2, required=False),
    "extra_knm": _P("number", "kNm", 0, 50, required=False),
}, {"load_factor": 1.5, "extra_knm": 2.0}, "M = lf (w L^2 / 8 + P L / 4) + extra")
_CALCS = [_SLAB, _BEAM]
_IDS = [c.name for c in _CALCS]


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


def _good_door(calc: _Calc):
    def door(name, params):
        missing = [k for k, p in calc.params.items() if p.required and k not in params]
        if missing:
            return {"status": "needs_input", "needs_input": missing}
        vals = {k: _to_number(v, calc.params[k].unit) for k, v in params.items()}
        rows = [{"parameter": k, "value": v} for k, v in calc.defaults.items() if k not in vals]
        return {"status": "success", "result": {"value": calc.fn(**vals), "defaults_used": rows,
                                                "formula": calc.formula}}
    return door


def _bad_door(calc: _Calc):
    # Runs on a zero for a missing input, reads "X mm" as a bare number, and
    # never discloses a default: every rule must catch it.
    def door(name, params):
        vals = {k: float(str(params.get(k, 0)).split()[0]) for k, p in calc.params.items() if p.required}
        return {"status": "success", "result": {"value": calc.fn(**vals)}}
    return door


def _sweep(calc: _Calc, door):
    spec = _Spec(calc.name, calc.fn, calc.params)
    return ck.sweep([spec], lambda s: {k: _named(k, p) for k, p in s.params.items()}, door)


def _named(key, p):
    return type("PS", (), {"name": key, "kind": p.kind, "unit": p.unit, "lo": p.lo, "hi": p.hi,
                           "label": p.label, "required": p.required})()


@pytest.mark.parametrize("calc", _CALCS, ids=_IDS)
def test_sweep_passes_a_door_that_keeps_all_three_rules(calc):
    rep = _sweep(calc, _good_door(calc))
    assert rep["passed"] == rep["total"], [r for r in rep["rows"] if not r["passed"]]
    # Both invented calculators have three required inputs, each in a unit the
    # exam's table can rewrite, at least one fall-back default and at least
    # one body constant.
    assert rep["by_rule"] == {"baseline": [1, 1], "M": [3, 3], "U": [3, 3], "D": [1, 1], "C": [1, 1]}


@pytest.mark.parametrize("calc", _CALCS, ids=_IDS)
def test_sweep_fails_a_door_that_breaks_each_rule(calc):
    rep = _sweep(calc, _bad_door(calc))
    assert rep["by_rule"]["M"] == [0, 3]
    assert rep["by_rule"]["U"] == [0, 3]
    assert rep["by_rule"]["D"] == [0, 1]
    assert rep["by_rule"]["C"] == [0, 1]


# ---------------------------------------------------------------- rule C
_RATE = 2.75  # invented module-level scalar


def _two_branch(a, b, mode=0):
    total = 0.0
    total += a
    lap = 1.15
    if mode:
        hidden = 3.3
        return total * hidden
    return (total * lap + b * _RATE) ** 2 / 1000


def _plain(x, y):
    scale = 1
    return x * y * scale + 10 - 0


def _expected_constants(fn, call):
    _, lines = ck.traced_lines(lambda: fn(**call), fn)
    return ck.named_constants(fn, lines)


def test_constants_found_on_the_lines_the_run_executed():
    took_main = _expected_constants(_two_branch, {"a": 2.0, "b": 3.0})
    took_side = _expected_constants(_two_branch, {"a": 2.0, "b": 3.0, "mode": 1})
    assert took_main == {"lap": 1.15, "_RATE": 2.75}
    assert took_side == {"hidden": 3.3}


def test_identities_scales_exponents_accumulators_and_inputs_are_not_constants():
    assert _expected_constants(_plain, {"x": 4.0, "y": 5.0}) == {}
    assert _expected_constants(_plain, {"x": 0.5, "y": 9.0}) == {}
    assert "total" not in ck.named_constants(_two_branch)  # accumulator, whole body
    assert "literal 2" not in ck.named_constants(_two_branch)  # exponent


def test_inline_literal_operand_is_a_constant():
    def stair(rise_mm, going_mm):
        return 2 * rise_mm + going_mm - 630

    def ramp(rise_m):
        return rise_m * 12.5

    assert _expected_constants(stair, {"rise_mm": 170.0, "going_mm": 280.0}) == {"literal 2": 2.0, "literal 630": 630.0}
    assert _expected_constants(ramp, {"rise_m": 0.4}) == {"literal 12.5": 12.5}


def test_stated_passes_by_row_name_row_value_or_written_basis():
    consts = {"lap": 1.15, "_RATE": 2.75}
    by_rows = {"status": "success", "result": {"v": 1, "assumptions": [
        {"name": "lap", "value": 9}, {"parameter": "x", "value": 2.75}]}}
    by_text = {"status": "success", "result": {"v": 1, "formula": "v = a x 1.15 + b x 2.75"}}
    by_note = {"status": "success", "note": "uses lap and rate", "result": {"v": 1}}
    assert ck.check_stated(by_rows, consts)[0]
    assert ck.check_stated(by_text, consts)[0]
    assert ck.check_stated(by_note, consts)[0]


def test_stated_fails_when_any_constant_is_unstated():
    consts = {"lap": 1.15, "literal 630": 630.0}
    half = {"status": "success", "result": {"v": 1, "formula": "v = a x 1.15"}}
    near = {"status": "success", "result": {"v": 1, "formula": "v = a x 1.155 + 6300"}}
    assert ck.check_stated(half, consts) == (False, "constant used unstated: literal 630")
    assert not ck.check_stated(near, consts)[0]
    assert not ck.check_stated({"status": "success", "result": {"v": 1}}, {"_RATE": 2.75})[0]


def test_sweep_fails_c_when_the_door_never_runs_the_body():
    # The door answers through another function than the registered one: the
    # registered body never runs, so rule C cannot be judged and must be
    # reported as failed, never passed silently.
    for calc in _CALCS:
        rep = ck.sweep([_Spec(calc.name, lambda **k: 0.0, calc.params)],
                       lambda s: {k: _named(k, p) for k, p in s.params.items()}, _good_door(calc))
        c_rows = [r for r in rep["rows"] if r["rule"] == "C"]
        assert c_rows and not c_rows[0]["passed"], rep["rows"]


@pytest.mark.parametrize("calc", _CALCS, ids=_IDS)
def test_sweep_fails_a_door_that_discloses_only_some_defaults(calc):
    good = _good_door(calc)
    first = next(iter(calc.defaults))

    def door(name, params):
        env = good(name, params)
        if env.get("status") == "success":
            env["result"]["defaults_used"] = [r for r in env["result"]["defaults_used"]
                                              if r["parameter"] == first][: len(calc.defaults) - 1]
        return env

    rep = _sweep(calc, door)
    assert rep["by_rule"]["D"] == [0, 1]
    assert rep["by_rule"]["M"] == [3, 3] and rep["by_rule"]["U"] == [3, 3]


@pytest.mark.parametrize("calc", _CALCS, ids=_IDS)
def test_sweep_reports_the_stage_when_a_calculator_cannot_be_exercised(calc):
    blocked = {**calc.params, "mix": _P("text", "-", label="mix")}

    def needs_text(*args, mix=None, **kwargs):
        return 0.0

    rep = ck.sweep([_Spec("needs_text", needs_text, blocked), _Spec("refuses", calc.fn, calc.params)],
                   lambda s: {k: _named(k, p) for k, p in s.params.items()},
                   lambda n, p: {"status": "error", "error": "refused"})
    stages = [(r["calc"], r["rule"], r["passed"]) for r in rep["rows"]]
    assert stages == [("needs_text", "baseline", False), ("refuses", "baseline", False)]
    assert "mix" in rep["rows"][0]["why"]
