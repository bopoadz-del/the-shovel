"""Agent B standing-exit bind after SO2 tip 4ab55613 (#639+#652).

Live re-probe 1-42 scored beam_shear_simple as tool_error×2 because
unbound defaults of 0 returned max_shear 0.0 (same class as delay_damages).
dewatering_uplift_check stayed tool_error×2 after #652 — model aliases
(t_raft / storeys / extract-from-ask) never reached the signature.
excavation_volume was wrong_number on one of two asks (height / l,w,d /
L×W×D-in-text). The remaining 1-of-2 tool_error set is bind: common model
names and JSON-string ``params`` never reached the calculator.

Formulas stay the gated oracles. Bind / alias / extract-from-ask is what
this file pins. beam_moment_simple and fall_arrest_force are http_502
re-ask class — not touched.
"""
from __future__ import annotations

import json

import pytest

from app.lib.construction_formulas import (
    CALCULATORS,
    bind_calculation_params,
    coerce_calc_params,
    run_calculation,
)
from tests.test_calc_standing_exit_bind import (
    _assert_named_missing,
    _err,
    _ok,
    _tool,
)
from tests.test_construction_calc_tool import _agent, _call


# ── beam_shear_simple (unbound zeros were live tool_error / wrong_number) ──

# V = wL/2 + P/2. Hand: 20×6/2 = 60 kN. With P=40: 60+20 = 80 kN.
_SHEAR_REQUIRED = (
    ("udl_w_kn_m", "kN/m"),
    ("span_m", "m"),
)


def test_beam_shear_simple_empty_kwargs_is_missing_not_zero():
    """Unbound defaults of 0 used to succeed with max_shear_kn=0."""
    env = _err("beam_shear_simple", {})
    _assert_named_missing(env, _SHEAR_REQUIRED)
    inner = env.get("result") or {}
    if isinstance(inner, dict):
        assert inner.get("max_shear_kn") in (None, 0, 0.0)


def test_beam_shear_simple_partial_kwargs_is_missing_not_zero():
    env = _err("beam_shear_simple", {"udl_w_kn_m": 20})
    assert "span_m" in env["missing"]
    env = _err("beam_shear_simple", {"span_m": 6})
    assert "udl_w_kn_m" in env["missing"]


def test_beam_shear_simple_formula_matches_wl_over_two():
    expected = 20.0 * 6.0 / 2.0
    assert expected == pytest.approx(60.0)
    inner = _ok("beam_shear_simple", {"udl_w_kn_m": 20, "span_m": 6})
    assert inner["max_shear_kn"] == pytest.approx(expected, abs=0.01)
    both = _ok("beam_shear_simple", {
        "udl_w_kn_m": 20, "span_m": 6, "central_point_load_kn": 40,
    })
    assert both["max_shear_kn"] == pytest.approx(80.0, abs=0.01)


def test_beam_shear_simple_aliases_and_nests():
    for params in (
        {"w": 20, "span": 6},
        {"w": 20, "L": 6},
        {"udl": 20, "l": 6},
        {"input": {"w": 20, "span_m": 6}},
        {"params": {"udl_w_kn_m": 20, "span_m": 6}},
    ):
        inner = _ok("beam_shear_simple", params)
        assert inner["max_shear_kn"] == pytest.approx(60.0, abs=0.01), params


def test_beam_shear_simple_extract_from_ask_text():
    inner = _ok("beam_shear_simple", {
        "text": "Simply-supported beam shear: UDL w=20 kN/m, span L=6 m.",
    })
    assert inner["max_shear_kn"] == pytest.approx(60.0, abs=0.01)


def test_beam_shear_simple_construction_calc_empty_and_bound():
    empty = _tool("beam_shear_simple", {})
    assert empty.get("ok") is False, empty
    _assert_named_missing(empty["result"], _SHEAR_REQUIRED)
    bound = _call(_agent(["construction"]), "beam_shear_simple", {"w": 20, "L": 6})
    assert bound["ok"] is True, bound
    assert bound["result"]["result"]["max_shear_kn"] == pytest.approx(60.0, abs=0.01)


def test_beam_shear_simple_point_load_only_still_binds():
    """P-only is valid statics (V=P/2); span is still required."""
    inner = _ok("beam_shear_simple", {"central_point_load_kn": 40, "span_m": 6})
    assert inner["max_shear_kn"] == pytest.approx(20.0, abs=0.01)


# ── dewatering_uplift_check (still tool_error×2 after #652) ────────────────

_DEWATER_KW = dict(water_depth=23, raft_thickness=2.0, floor_count=5)
_DEWATER_REQUIRED = (
    ("water_depth", "m"),
    ("raft_thickness", "m"),
    ("floor_count", "floors"),
)


def test_dewatering_uplift_check_partial_kwargs_names_missing():
    env = _err("dewatering_uplift_check", {"water_depth": 23})
    assert "raft_thickness" in env["missing"]
    assert "floor_count" in env["missing"]
    _assert_named_missing(
        _err("dewatering_uplift_check", {}),
        _DEWATER_REQUIRED,
    )


def test_dewatering_uplift_check_live_model_aliases():
    """#652 bound water_depth_m / floors. Live still sent t_raft / storeys."""
    for params in (
        {"hw": 23, "t_raft": 2.0, "storeys": 5},
        {"groundwater": 23, "raft_t": 2.0, "stories": 5},
        {"gwl": 23, "raft_thk": 2.0, "levels": 5},
        {"water_head": 23, "raft_depth": 2.0, "n_storeys": 5},
        {"input": {"hw": 23, "t_raft": 2.0, "num_floors": 5}},
    ):
        inner = _ok("dewatering_uplift_check", params)
        assert inner["fos"] == pytest.approx(0.380, abs=0.001), params
        assert inner["can_stop"] is False


def test_dewatering_uplift_check_extract_from_ask_text():
    inner = _ok("dewatering_uplift_check", {
        "text": (
            "Run a dewatering uplift check for 23 m water depth, "
            "2 m raft, 5 floors."
        ),
    })
    assert inner["fos"] == pytest.approx(0.380, abs=0.001)
    assert inner["min_floors_for_stop"] == 32


def test_dewatering_uplift_check_construction_calc_alias_shapes():
    env = _tool("dewatering_uplift_check", {
        "hw": 23, "t_raft": 2.0, "storeys": 5,
    })
    assert env.get("ok") is True, env
    assert env["result"]["result"]["fos"] == pytest.approx(0.380, abs=0.001)


# ── excavation_volume (wrong_number 1-of-2) ────────────────────────────────

# Gated oracle: 20×10×3 m, 25% bulk → bank 600, loose 750.
# Leftover L6: 14.5×3.2×1.75 → bank 81.2.


def test_excavation_volume_harness_20x10x3_is_600_bank_750_loose():
    expected_bank = 20.0 * 10.0 * 3.0
    expected_loose = expected_bank * 1.25
    assert expected_bank == pytest.approx(600.0)
    assert expected_loose == pytest.approx(750.0)
    inner = _ok("excavation_volume", {
        "length_m": 20, "width_m": 10, "depth_m": 3,
    })
    assert inner["bank_volume_m3"] == pytest.approx(expected_bank, abs=0.01)
    assert inner["loose_volume_m3"] == pytest.approx(expected_loose, abs=0.01)


def test_excavation_volume_lwd_and_height_aliases():
    for params in (
        {"l": 20, "w": 10, "d": 3},
        {"length": 20, "width": 10, "height": 3},
        {"length_m": 20, "width_m": 10, "height_m": 3},
        {"L": 20, "B": 10, "depth": 3},
        {"input": {"l": 20, "w": 10, "d": 3}},
    ):
        inner = _ok("excavation_volume", params)
        assert inner["bank_volume_m3"] == pytest.approx(600.0, abs=0.01), params
        assert inner["loose_volume_m3"] == pytest.approx(750.0, abs=0.01), params


def test_excavation_volume_leftover_l6_is_81_2_bank():
    expected = 14.5 * 3.2 * 1.75
    assert expected == pytest.approx(81.2, abs=0.01)
    inner = _ok("excavation_volume", {
        "text": (
            "Bank volume of a rectangular trench 14.5 m long by "
            "3.2 m wide by 1.75 m deep."
        ),
    })
    assert inner["bank_volume_m3"] == pytest.approx(81.2, abs=0.05)
    # Default 25% bulking is the gated factor, not a second formula.
    assert inner["loose_volume_m3"] == pytest.approx(81.2 * 1.25, abs=0.05)


def test_excavation_volume_lwt_chain_in_text():
    inner = _ok("excavation_volume", {"text": "excavation 20x10x3 m"})
    assert inner["bank_volume_m3"] == pytest.approx(600.0, abs=0.01)


def test_excavation_volume_construction_calc_height_alias():
    env = _tool("excavation_volume", {
        "length_m": 14.5, "width_m": 3.2, "height_m": 1.75,
    })
    assert env.get("ok") is True, env
    assert env["result"]["result"]["bank_volume_m3"] == pytest.approx(81.2, abs=0.05)


# ── 1-of-2 tool_error set: common model aliases + JSON-string params ───────


def test_coerce_calc_params_accepts_json_object_string():
    assert coerce_calc_params('{"quantity_m3": 100}') == {"quantity_m3": 100}
    assert coerce_calc_params({"quantity_m3": 100}) == {"quantity_m3": 100}
    assert coerce_calc_params(None) == {}
    assert coerce_calc_params("") == {}
    assert coerce_calc_params("not-json") == {}
    assert coerce_calc_params("udl_w_kn_m=20, span_m=6") == {
        "udl_w_kn_m": 20, "span_m": 6,
    }
    assert coerce_calc_params("status: error") == {}


def test_live_probe_beam_shear_assignment_string_is_60_not_zero():
    """standing_exit_b_probe: params udl_w_kn_m=20, span_m=6 → Vmax=60."""
    inner = _ok("beam_shear_simple", "udl_w_kn_m=20, span_m=6")
    assert inner["max_shear_kn"] == pytest.approx(60.0, abs=0.01)
    env = _tool("beam_shear_simple", {"params": "udl_w_kn_m=20, span_m=6"})
    assert env.get("ok") is True, env
    assert env["result"]["result"]["max_shear_kn"] == pytest.approx(60.0, abs=0.01)


def test_live_probe_dewatering_assignment_string_binds():
    """standing_exit_b_probe: water_depth=23, raft_thickness=2.0, floor_count=5."""
    inner = _ok(
        "dewatering_uplift_check",
        "water_depth=23, raft_thickness=2.0, floor_count=5",
    )
    assert inner["fos"] == pytest.approx(0.380, abs=0.001)
    via_suffix = _ok("dewatering_uplift_check", {
        "params": "water_depth_m=23, raft_thickness_m=2.0, floors=5",
    })
    assert via_suffix["fos"] == pytest.approx(0.380, abs=0.001)
    nested = _ok("dewatering_uplift_check", {
        "input": {"water_depth": 23, "raft_thickness": 2.0, "storeys": 5},
    })
    assert nested["fos"] == pytest.approx(0.380, abs=0.001)
    via_input_str = _tool("dewatering_uplift_check", {
        "input": "water_depth=23, raft_thickness=2.0, floor_count=5",
    })
    assert via_input_str.get("ok") is True, via_input_str
    assert via_input_str["result"]["result"]["fos"] == pytest.approx(0.380, abs=0.001)


def test_live_probe_concrete_volume_includes_net_15_and_waste_15_75():
    """length=10,width=5,thickness=0.3 → net 15.0 and with-waste 15.75."""
    inner = _ok("concrete_volume", "length=10,width=5,thickness=0.3")
    assert inner["net_volume_m3"] == pytest.approx(15.0, abs=0.001)
    assert inner["volume_m3"] == pytest.approx(15.75, abs=0.001)
    assert inner["volume_with_waste_m3"] == pytest.approx(15.75, abs=0.001)
    assert inner["waste_factor"] == pytest.approx(0.05, abs=1e-9)
    env = run_calculation("concrete_volume", "length=10, width=5, thickness=0.3")
    assert env.get("status") == "success", env


def test_live_probe_excavation_keeps_bank_600_and_bulked_750():
    """length=20,width=10,depth=3. Ask-2 wrong_number was bulking 25 vs 0.25."""
    inner = _ok("excavation_volume", "length=20,width=10,depth=3")
    assert inner["bank_volume_m3"] == pytest.approx(600.0, abs=0.01)
    assert inner["loose_volume_m3"] == pytest.approx(750.0, abs=0.01)
    assert inner["bulked_volume_m3"] == pytest.approx(750.0, abs=0.01)
    percent = _ok("excavation_volume", {
        "length": 20, "width": 10, "depth": 3, "bulking": 25,
    })
    assert percent["bank_volume_m3"] == pytest.approx(600.0, abs=0.01)
    assert percent["loose_volume_m3"] == pytest.approx(750.0, abs=0.01)


def test_json_string_params_nest_binds():
    inner = _ok("cost_buildup_concrete", {
        "params": json.dumps({"quantity_m3": 100}),
    })
    assert inner["total_project_value_sar"] == pytest.approx(30883, abs=2)


def test_bolt_shear_capacity_model_aliases():
    # 0.75 * 372 * 490 * 1 / 1000 = 136.71 (existing bind oracle)
    inner = _ok("bolt_shear_capacity", {"area": 490, "fnv": 372})
    assert inner["capacity_kn"] == pytest.approx(136.71, abs=0.02)
    via_ab = _ok("bolt_shear_capacity", {"Ab": 490, "SHEAR_STRENGTH_MPA": 372})
    assert via_ab["capacity_kn"] == pytest.approx(136.71, abs=0.02)


def test_calculate_interim_payment_model_aliases():
    inner = _ok("calculate_interim_payment", {
        "gross_amount": 900_000, "retention": 10,
    })
    assert inner["net_payment"] == pytest.approx(810_000.0)
    via_amt = _ok("calculate_interim_payment", {"amount": 900_000})
    assert via_amt["net_payment"] == pytest.approx(810_000.0)


def test_carbon_footprint_concrete_model_aliases():
    inner = _ok("carbon_footprint_concrete", {"qty": 100, "grade": "C30"})
    assert inner["total_kgco2e"] == pytest.approx(32000.0)
    via_text = _ok("carbon_footprint_concrete", {
        "text": "embodied carbon of 100 m3 C30 concrete",
    })
    assert via_text["total_kgco2e"] == pytest.approx(32000.0)


def test_composite_column_design_model_aliases():
    canonical = _ok("composite_column_design", {
        "axial_load_kn": 2000, "column_diameter_mm": 400,
    })
    aliased = _ok("composite_column_design", {
        "load": 2000, "dia": 400,
    })
    assert aliased["recommended"] == canonical["recommended"]
    via_p = _ok("composite_column_design", {"P": 2000, "column_dia": 400})
    assert via_p["recommended"] == canonical["recommended"]


def test_concrete_maturity_strength_scalar_and_csv_lists():
    """One live ask sent lists; the other sent a scalar / CSV string."""
    lists = _ok("concrete_maturity_strength", {
        "temperature_history_c": [20, 22, 25],
        "time_intervals_hours": [6, 6, 6],
    })
    csv = _ok("concrete_maturity_strength", {
        "temps": "20, 22, 25",
        "hours": "6, 6, 6",
    })
    assert csv["maturity_index_c_hrs"] == pytest.approx(
        lists["maturity_index_c_hrs"], abs=0.5,
    )
    scalar = _ok("concrete_maturity_strength", {
        "temperature": 20, "hours": 168,
    })
    assert scalar["maturity_index_c_hrs"] == pytest.approx(5040.0, abs=0.5)
    assert scalar["percent_of_28d"] == pytest.approx(70.4, abs=0.2)


def test_concrete_mix_design_sg_wc_aliases():
    inner = _ok("concrete_mix_design_sg", {"wc": 0.48})
    assert inner["cement_kg_m3"] == pytest.approx(320.0, abs=0.5)
    via_slash = _ok("concrete_mix_design_sg", {"w/c": 0.48})
    assert via_slash["cement_kg_m3"] == pytest.approx(320.0, abs=0.5)
    via_text = _ok("concrete_mix_design_sg", {
        "text": "concrete mix design for W/C 0.48",
    })
    assert via_text["cement_kg_m3"] == pytest.approx(320.0, abs=0.5)


def test_concrete_volume_t_and_lwt_aliases():
    inner = _ok("concrete_volume", {"l": 10, "w": 5, "t": 0.3})
    assert inner["net_volume_m3"] == pytest.approx(15.0, abs=0.001)
    assert inner["volume_m3"] == pytest.approx(15.75, abs=0.001)
    cyl = _ok("concrete_volume", {
        "shape": "cylinder", "dia": 0.6, "h": 10,
    })
    assert cyl["net_volume_m3"] == pytest.approx(3.1415926535 * 0.09 * 10, abs=0.01)


def test_cost_buildup_concrete_qty_and_volume_aliases():
    inner = _ok("cost_buildup_concrete", {"qty": 100})
    assert inner["total_project_value_sar"] == pytest.approx(30883, abs=2)
    via_vol = _ok("cost_buildup_concrete", {"volume": 100})
    assert via_vol["total_project_value_sar"] == pytest.approx(30883, abs=2)


def test_cost_per_area_cost_and_area_m2_aliases():
    inner = _ok("cost_per_area", {"cost": 1_000_000, "area_m2": 5000})
    assert inner["cost_per_area"] == pytest.approx(200.0, abs=0.01)
    via_amt = _ok("cost_per_area", {"amount": 1_000_000, "area": 5000})
    assert via_amt["cost_per_area"] == pytest.approx(200.0, abs=0.01)


def test_diaphragm_wall_panel_volume_short_aliases():
    inner = _ok("diaphragm_wall_panel_volume", {
        "length": 6, "thickness": 0.8, "depth": 20, "panel_count": 3,
    })
    assert inner["volume_per_panel_m3"] == pytest.approx(96.0, abs=0.01)
    assert inner["volume_with_waste_m3"] == pytest.approx(316.8, abs=0.01)
    via_t = _ok("diaphragm_wall_panel_volume", {
        "l": 6, "t": 0.8, "d": 20,
    })
    assert via_t["volume_per_panel_m3"] == pytest.approx(96.0, abs=0.01)


def test_electrical_installation_sequence_area_aliases():
    inner = _ok("electrical_installation_sequence", {
        "area": 1000, "floors": 3,
    })
    assert inner["total_days"] == 360
    via_gfa = _ok("electrical_installation_sequence", {
        "gfa": 1000, "num_floors": 3,
    })
    assert via_gfa["total_days"] == 360


def test_unit_suffixed_numeric_strings_coerce():
    inner = _ok("beam_shear_simple", {"w": "20 kN/m", "span": "6m"})
    assert inner["max_shear_kn"] == pytest.approx(60.0, abs=0.01)
    dw = _ok("dewatering_uplift_check", {
        "water_depth": "23 m", "raft_thickness": "2.0m", "floor_count": "5",
    })
    assert dw["fos"] == pytest.approx(0.380, abs=0.001)


def test_construction_calc_json_string_params_tool_path():
    agent = _agent(["construction"])
    tc = {
        "id": "c1",
        "function": {
            "name": "construction_calc",
            "arguments": json.dumps({
                "calculation": "calculate_interim_payment",
                "params": json.dumps({
                    "gross_valuation": 900_000, "retention_percent": 10,
                }),
            }),
        },
    }
    from tests.test_construction_calc_tool import _run
    env = _run(agent._run_tool_call(tc))
    assert env.get("ok") is True, env
    assert env["result"]["result"]["net_payment"] == pytest.approx(810_000.0)


def test_bind_does_not_leak_envelope_on_fail_list():
    leaked = {
        "w": 20, "L": 6,
        "text": "compute shear",
        "formula": "not-a-kwarg",
        "project_id": "ws-b",
        "input": {"noise": 1},
    }
    bound = bind_calculation_params(CALCULATORS["beam_shear_simple"], leaked)
    for key in ("text", "formula", "input", "project_id", "noise"):
        assert key not in bound, bound
    assert bound["udl_w_kn_m"] == 20
    assert bound["span_m"] == 6
