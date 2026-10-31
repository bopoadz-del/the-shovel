"""Empty construction_calc tool args must still see the current user ask.

Live Phase-2 re-probe (tip 0a95d03 / #663): the model calls
``construction_calc`` with only ``{"action": "construction_calc"}`` —
no calculation name, no params, no text. #663 extract reads kwargs
``text`` / ``formula`` / ``message``, so bind never sees the labeled
engineer numbers (As1500, span 8m, W=10000 kN).

Owner D7: inject the current user turn as ``text`` before
``run_calculation`` / bind. Never invent a figure that is not in the
ask. Calculation name still resolves from the ask (intent / registry).
"""
from __future__ import annotations

import json

import pytest

from tests.test_construction_calc_tool import _agent, _run


def _empty_call(user_message, extra=None):
    """Live shape: action only (or caller extra), plus the user turn."""
    agent = _agent(["construction"])
    tc = {
        "id": "c1",
        "function": {
            "name": "construction_calc",
            "arguments": json.dumps(extra if extra is not None else {
                "action": "construction_calc",
            }),
        },
    }
    return _run(agent._run_tool_call(tc, user_message=user_message))


# ── labeled engineer asks (same oracles as #663 / DIR7) ────────────────────

_RC_ASK = (
    "rc_beam_moment_capacity design ACI As1500, fy420, b300, d550, fc30"
)
_RC_PARTIAL = "rc_beam_moment_capacity As1500 fy420"
_STEEL_ASK = (
    "steel_tension_capacity AISC Ag=3000, Ae=2550 mm2, Fy=345, Fu=450"
)
_STEEL_PARTIAL = "steel_tension_capacity Fy=345 Fu=450"
_PT_ASK = (
    "post_tensioning_force for span 8m, slab 0.25m, live load 5 kN/m2"
)
_PT_PARTIAL = "post_tensioning_force for span 8m and slab 0.25m"
_SEISMIC_ASK = (
    "seismic_base_shear ASCE W=10000 kN, SDS=1.0, R=8, Ie=1"
)


@pytest.mark.parametrize(
    "ask,result_key,result_value,abs_",
    (
        (_RC_ASK, "moment_capacity_kn_m", 288.50, 0.3),
        (_STEEL_ASK, "capacity_kn", 860.63, 0.05),
        (_PT_ASK, "tendon_force_kn", 1000.0, 0.1),
    ),
    ids=("rc_beam", "steel_tension", "post_tension"),
)
def test_action_only_construction_calc_injects_user_ask_and_binds(
    ask, result_key, result_value, abs_,
):
    """``{"action": "construction_calc"}`` + labeled ask → bind + compute."""
    env = _empty_call(ask)
    assert env.get("ok") is True, env
    assert env["result"]["status"] == "success", env
    inner = env["result"]["result"]
    assert inner[result_key] == pytest.approx(result_value, abs=abs_), env
    # Name still resolved from the ask (intent / registry), not invented.
    calc = env["result"].get("calculation")
    assert calc in ask or (calc or "").replace("_", " ") in ask.lower()


@pytest.mark.parametrize(
    "ask,missing_name,missing_unit",
    (
        (_RC_PARTIAL, "width_mm", "mm"),
        (_STEEL_PARTIAL, "gross_area_mm2", "mm2"),
        (_PT_PARTIAL, "live_load_kn_m2", "kN/m2"),
    ),
    ids=("rc_beam", "steel_tension", "post_tension"),
)
def test_action_only_partial_ask_names_missing_param_and_unit(
    ask, missing_name, missing_unit,
):
    """D7: numbers that are not in the ask stay missing (named + unit)."""
    env = _empty_call(ask)
    assert env.get("ok") is False, env
    result = env["result"]
    assert result["status"] == "error", result
    err = result["error"]
    assert "missing required" in err
    assert missing_name in err
    assert f"{missing_name} ({missing_unit})" in err
    assert missing_name in result["missing"]


def test_empty_ask_empty_args_does_not_invent_numbers():
    """Empty turn + empty tool args: honest missing-required, no defaults."""
    env = _empty_call("", {"action": "construction_calc"})
    assert env.get("ok") is False, env
    result = env["result"]
    assert result.get("status") == "error", result
    blob = json.dumps(result, default=str)
    assert "288.5" not in blob
    assert "860.63" not in blob
    assert "1000" not in blob or "missing required" in result.get("error", "")
    # Named calc with empty ask is the DIR7 empty-kwargs class.
    named = _empty_call("", {
        "action": "construction_calc",
        "calculation": "rc_beam_moment_capacity",
    })
    assert named.get("ok") is False, named
    nres = named["result"]
    assert nres["status"] == "error"
    assert "missing required" in nres["error"]
    assert "steel_area_mm2" in nres["error"]
    assert "steel_area_mm2 (mm2)" in nres["error"]
    assert "steel_area_mm2" in nres["missing"]
    assert nres.get("result") is None or "moment_capacity_kn_m" not in (
        nres.get("result") or {}
    )


def test_action_only_seismic_success_includes_unit_kn():
    """#628 / live tool-but-no-unit: seismic success must stamp kN."""
    env = _empty_call(_SEISMIC_ASK)
    assert env.get("ok") is True, env
    assert env["result"]["status"] == "success", env
    inner = env["result"]["result"]
    assert inner["base_shear_kn"] == pytest.approx(1250.0, abs=0.05)
    assert inner.get("unit") == "kN"
    assert inner.get("value") == pytest.approx(1250.0, abs=0.05)
    assert "kn" in str(inner.get("note") or "").lower()


@pytest.mark.parametrize("shape", ("empty_object", "action_only", "params_empty"))
def test_incomplete_shapes_all_inject_the_same_ask(shape):
    """{} / action-only / params={} are the same live incomplete class."""
    if shape == "empty_object":
        extra = {}
    elif shape == "action_only":
        extra = {"action": "construction_calc"}
    else:
        extra = {"action": "construction_calc", "params": {}}
    env = _empty_call(_RC_ASK, extra)
    assert env.get("ok") is True, (shape, env)
    inner = env["result"]["result"]
    assert inner["moment_capacity_kn_m"] == pytest.approx(288.50, abs=0.3)
