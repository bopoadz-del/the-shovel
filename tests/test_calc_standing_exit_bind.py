"""Standing-exit bind path for the four live tip a8498b3 failures.

calculate_evm still reported tool_error (missing required) after #636 because
empty kwargs / nested input never became a named+unit bind error. delay_damages
defaults of 0 made a failed bind look like a successful 0 (wrong_number).
dewatering_uplift_check and concrete_shrinkage dropped unit-suffix / short
aliases, then TypeError'd or computed the wrong figure.

Formulas are the gated oracles. Bind is what this file pins.
"""
from __future__ import annotations

import json

import pytest

from app.lib.construction_formulas import (
    CALCULATORS,
    bind_calculation_params,
    describe_calculation_params,
    run_calculation,
)
from tests.test_construction_calc_tool import _agent, _call, _run


def _ok(name, params):
    env = run_calculation(name, params)
    assert env.get("status") == "success", (name, params, env)
    return env["result"]


def _err(name, params):
    env = run_calculation(name, params)
    assert env.get("status") == "error", (name, params, env)
    return env


def _tool(calculation, extra):
    agent = _agent(["construction"])
    tc = {
        "id": "c1",
        "function": {
            "name": "construction_calc",
            "arguments": json.dumps({"calculation": calculation, **extra}),
        },
    }
    return _run(agent._run_tool_call(tc))


def _assert_named_missing(env, required):
    """Empty/failed bind must name every required param AND its unit."""
    err = env["error"]
    assert "missing required" in err, err
    assert "TypeError" not in err
    expected = {row["name"]: row for row in env["expected_params"]}
    for name, unit in required:
        assert name in err, (name, err)
        assert f"{name} ({unit})" in err, (name, unit, err)
        assert name in env["missing"], env["missing"]
        assert expected[name]["unit"] == unit, expected[name]
        assert expected[name]["required"] is True, expected[name]


# ── calculate_evm ──────────────────────────────────────────────────────────

_EVM_ORACLE = dict(pv=500_000, ev=400_000, ac=450_000, bac=1_000_000)
_EVM_REQUIRED = (
    ("pv", "currency"),
    ("ev", "currency"),
    ("ac", "currency"),
)


def _assert_evm(inner):
    assert inner["CPI"] == 0.889
    assert inner["SPI"] == 0.8
    assert inner["EV"] == 400_000
    assert inner["PV"] == 500_000
    assert inner["AC"] == 450_000


def test_calculate_evm_empty_kwargs_names_missing_params_and_units():
    env = _err("calculate_evm", {})
    _assert_named_missing(env, _EVM_REQUIRED)


def test_calculate_evm_empty_construction_calc_names_missing_params_and_units():
    env = _tool("calculate_evm", {})
    assert env.get("ok") is False, env
    _assert_named_missing(env["result"], _EVM_REQUIRED)


@pytest.mark.parametrize(
    "params",
    (
        {"pv": 500_000, "ev": 400_000, "ac": 450_000, "bac": 1_000_000},
        {"bcws": 500_000, "bcwp": 400_000, "acwp": 450_000, "bac": 1_000_000},
        {"pv": 500_000, "bcwp": 400_000, "acwp": 450_000, "bac": 1_000_000},
        {"BCWS": 500_000, "BCWP": 400_000, "ACWP": 450_000, "BAC": 1_000_000},
        {
            "planned_value": 500_000,
            "earned_value": 400_000,
            "actual_cost": 450_000,
            "budget_at_completion": 1_000_000,
        },
    ),
)
def test_calculate_evm_pmi_and_pe_aliases_bind(params):
    _assert_evm(_ok("calculate_evm", params))


@pytest.mark.parametrize("shape", ("params", "input"))
def test_calculate_evm_nested_params_and_input_bind(shape):
    _assert_evm(_ok("calculate_evm", {shape: dict(_EVM_ORACLE)}))


@pytest.mark.parametrize("shape", ("top_level", "params", "input"))
def test_calculate_evm_construction_calc_three_shapes(shape):
    if shape == "top_level":
        extra = dict(_EVM_ORACLE)
    else:
        extra = {shape: dict(_EVM_ORACLE)}
    env = _tool("calculate_evm", extra)
    assert env.get("ok") is True, env
    _assert_evm(env["result"]["result"])


def test_calculate_evm_flatten_does_not_pass_envelope_as_kwargs():
    leaked = {
        **_EVM_ORACLE,
        "text": "compute evm",
        "formula": "not-a-kwarg",
        "project_id": "ws-evm",
        "input": {"noise": 1},
    }
    bound = bind_calculation_params(CALCULATORS["calculate_evm"], leaked)
    for key in ("text", "formula", "input", "project_id", "noise"):
        assert key not in bound, bound
    rejected = _err("calculate_evm", leaked)
    assert "noise" in rejected["error"]
    clean = {key: val for key, val in leaked.items() if key != "input"}
    _assert_evm(_ok("calculate_evm", clean))


# ── delay_damages_daily (unbound zeros were live wrong_number) ─────────────

_DD_RATE = 0.1
_DD_ACA = 1_754_504_456.25
_DD_DAILY = 1_754_504.46  # 0.1% × ACA, banker's-round half-up via round(x, 2)
_DD_REQUIRED = (
    ("rate_percent", "%"),
    ("contract_amount", "currency"),
)


def test_delay_damages_daily_formula_matches_rate_times_aca():
    """Harness: expected daily figure is rate% × Accepted Contract Amount."""
    expected = round(_DD_ACA * (_DD_RATE / 100.0), 2)
    assert expected == pytest.approx(_DD_DAILY, abs=0.005)
    inner = _ok("delay_damages_daily", {
        "rate_percent": _DD_RATE, "contract_amount": _DD_ACA, "currency": "SAR",
    })
    assert inner["daily_amount"] == pytest.approx(expected, abs=0.005)
    assert inner["daily_amount"] == pytest.approx(_DD_DAILY, abs=0.005)


def test_delay_damages_daily_empty_kwargs_is_missing_not_zero():
    """Unbound defaults of 0 used to succeed with daily_amount=0 (wrong_number)."""
    env = _err("delay_damages_daily", {})
    _assert_named_missing(env, _DD_REQUIRED)
    assert env.get("result") is None or env.get("result", {}).get("daily_amount") in (
        None, 0, 0.0,
    )


def test_delay_damages_daily_aliases_and_nests():
    aliased = _ok("delay_damages_daily", {"rate": _DD_RATE, "aca": _DD_ACA})
    assert aliased["daily_amount"] == pytest.approx(_DD_DAILY, abs=0.005)
    nested = _ok("delay_damages_daily", {
        "input": {"contract_value": _DD_ACA, "rate_percent": _DD_RATE},
    })
    assert nested["daily_amount"] == pytest.approx(_DD_DAILY, abs=0.005)


def test_delay_damages_daily_construction_calc_empty_and_bound():
    empty = _tool("delay_damages_daily", {})
    assert empty.get("ok") is False, empty
    _assert_named_missing(empty["result"], _DD_REQUIRED)
    bound = _call(_agent(["construction"]), "delay_damages_daily", {
        "rate_percent": _DD_RATE, "contract_amount": _DD_ACA,
    })
    assert bound["ok"] is True, bound
    assert bound["result"]["result"]["daily_amount"] == pytest.approx(
        _DD_DAILY, abs=0.005,
    )


# ── dewatering_uplift_check ────────────────────────────────────────────────

# FOS = (raft*γc + floors*t_floor*γc) / (hw*γw)
#      = (2.0*2.5 + 5*0.3*2.5) / 23 = 8.75 / 23 = 0.380434… → 0.380
_DEWATER_KW = dict(water_depth=23, raft_thickness=2.0, floor_count=5)
_DEWATER_REQUIRED = (
    ("water_depth", "m"),
    ("raft_thickness", "m"),
    ("floor_count", "floors"),
)


def test_dewatering_uplift_check_formula_matches_counter_over_uplift():
    raft_w = 2.0 * 2.5
    floor_w = 5 * 0.3 * 2.5
    expected_fos = (raft_w + floor_w) / 23.0
    inner = _ok("dewatering_uplift_check", _DEWATER_KW)
    assert inner["uplift_force_t_m2"] == pytest.approx(23.0)
    assert inner["counter_weight_t_m2"] == pytest.approx(8.75)
    assert inner["fos"] == pytest.approx(expected_fos, abs=0.001)
    assert inner["can_stop"] is False


def test_dewatering_uplift_check_empty_kwargs_names_missing_params_and_units():
    env = _err("dewatering_uplift_check", {})
    _assert_named_missing(env, _DEWATER_REQUIRED)


def test_dewatering_uplift_check_aliases_and_nests():
    aliased = _ok("dewatering_uplift_check", {
        "water_depth_m": 23, "raft_thickness_m": 2.0, "floors": 5,
    })
    assert aliased["fos"] == pytest.approx(0.380, abs=0.001)
    nested = _ok("dewatering_uplift_check", {"params": dict(_DEWATER_KW)})
    assert nested["fos"] == pytest.approx(0.380, abs=0.001)
    via_input = _ok("dewatering_uplift_check", {"input": dict(_DEWATER_KW)})
    assert via_input["fos"] == pytest.approx(0.380, abs=0.001)


def test_dewatering_uplift_check_construction_calc_three_shapes():
    for extra in (
        dict(_DEWATER_KW),
        {"params": dict(_DEWATER_KW)},
        {"input": dict(_DEWATER_KW)},
    ):
        env = _tool("dewatering_uplift_check", extra)
        assert env.get("ok") is True, (extra, env)
        assert env["result"]["result"]["fos"] == pytest.approx(0.380, abs=0.001)


# ── concrete_shrinkage (tool_error + wrong_number) ─────────────────────────

# ACI 209R moist-cured: esh = t/(f+t)*esh_ult ; f=35, esh_ult=780
# t=365 → 365/400*780 = 711.75 µε  (docs/formula-verification-table.md)
_SHRINK_T = 365.0
_SHRINK_F = 35.0
_SHRINK_ULT = 780.0
_SHRINK_ORACLE = _SHRINK_T / (_SHRINK_F + _SHRINK_T) * _SHRINK_ULT
_SHRINK_REQUIRED = (("time_days", "d"),)


def test_concrete_shrinkage_formula_is_aci209_hyperbolic():
    assert _SHRINK_ORACLE == pytest.approx(711.75, abs=0.005)
    inner = _ok("concrete_shrinkage", {"time_days": _SHRINK_T})
    assert inner["shrinkage_microstrain"] == pytest.approx(_SHRINK_ORACLE, abs=0.5)
    assert inner["shrinkage_microstrain"] == pytest.approx(711.75, abs=0.5)


def test_concrete_shrinkage_empty_kwargs_names_missing_params_and_units():
    env = _err("concrete_shrinkage", {})
    _assert_named_missing(env, _SHRINK_REQUIRED)


def test_concrete_shrinkage_aliases_and_nests():
    for params in (
        {"t": _SHRINK_T},
        {"time": _SHRINK_T},
        {"days": _SHRINK_T},
        {"input": {"t": _SHRINK_T}},
        {"params": {"time_days": _SHRINK_T}},
    ):
        inner = _ok("concrete_shrinkage", params)
        assert inner["shrinkage_microstrain"] == pytest.approx(711.75, abs=0.5), params


def test_concrete_shrinkage_construction_calc_empty_and_bound():
    empty = _tool("concrete_shrinkage", {})
    assert empty.get("ok") is False, empty
    _assert_named_missing(empty["result"], _SHRINK_REQUIRED)
    bound = _call(_agent(["construction"]), "concrete_shrinkage", {"t": _SHRINK_T})
    assert bound["ok"] is True, bound
    assert bound["result"]["result"]["shrinkage_microstrain"] == pytest.approx(
        711.75, abs=0.5,
    )


def test_bind_helpers_are_not_registered_calculators():
    assert "bind_calculation_params" not in CALCULATORS
    assert "describe_calculation_params" not in CALCULATORS
    assert "coerce_calc_params" not in CALCULATORS
    assert "extract_calculation_params_from_text" not in CALCULATORS


def test_describe_evm_marks_pv_ev_ac_required_with_currency():
    rows = describe_calculation_params(
        CALCULATORS["calculate_evm"], name="calculate_evm",
    )
    by_name = {r["name"]: r for r in rows}
    for key in ("pv", "ev", "ac", "bcws", "bcwp", "acwp"):
        assert by_name[key]["required"] is True
        assert by_name[key]["unit"] == "currency"
    assert by_name["bac"]["required"] is False
