"""Standing-exit B db20f40 re-probe (37/42) — remaining LIVE FAILs.

After Train G2, live still failed:
  concrete_maturity_strength      — tool_error×2 (temps/intervals list literals
                                    live only in ask text / predispatch)
  concrete_mix_design_sg          — tool_error (name collides with slip-form
                                    stem; sand suffix → dune_sand_pct; w/c
                                    assignment string)
  cost_buildup_concrete           — ask1 status error JSON (name collides with
                                    cost_buildup_rebar stem)
  cut_fill_balance                — ask1 cause/missing (cut= / fill= not bound)
  dewatering_well_point_spacing   — no_tool×2 (well point spacing does not
                                    unique-name the calc; wrong tool)

Formulas stay the gated oracles. Bind / alias / extract-from-ask / route
only. Never invent a figure that is not in the ask.
"""
from __future__ import annotations

import json

import pytest

from app.agents.runtime import (
    _forced_specific_tool,
    _formula_calculator_name_from_message,
    _message_is_formula_style_ask,
    _message_wants_drawing_qto,
    _message_wants_named_calculator,
)
from app.lib.construction_formulas import (
    CALCULATORS,
    bind_calculation_params,
    run_calculation,
)
from tests.test_calc_standing_exit_bind import (
    _assert_named_missing,
    _err,
    _ok,
    _tool,
)


_TOOLS = {
    "construction_calc",
    "drawing_qto",
    "formula_executor_v2",
    "formula_executor",
    "boq_processor",
    "generate_wbs",
    "primavera_parser",
}


def _forced(text: str) -> str | None:
    return _forced_specific_tool([{"role": "user", "content": text}], _TOOLS)


# ── concrete_maturity_strength (LIVE: tool_error×2) ────────────────────────

_MATURITY_TEXT_ASSIGN = "temps=[20,22,25] intervals=[6,6,6]"
_MATURITY_TEXT_BRACKETS = "temps [20, 22, 25] intervals [6, 6, 6]"


def _assert_maturity_582(inner):
    assert inner["maturity_index_c_hrs"] == pytest.approx(582.0, abs=0.5)
    assert inner["predicted_strength_n_mm2"] == pytest.approx(6.9, abs=0.1)


def test_maturity_list_literals_in_ask_text_not_tool_error():
    """Predispatch sends only {text: ask}. List literals must bind."""
    inner = _ok("concrete_maturity_strength", {"text": _MATURITY_TEXT_ASSIGN})
    _assert_maturity_582(inner)
    via_name = _ok("concrete_maturity_strength", {
        "text": f"concrete_maturity_strength {_MATURITY_TEXT_ASSIGN}",
    })
    _assert_maturity_582(via_name)
    env = _tool("concrete_maturity_strength", {"text": _MATURITY_TEXT_ASSIGN})
    assert env.get("ok") is True, env
    _assert_maturity_582(env["result"]["result"])


def test_maturity_bracketed_temps_intervals_without_equals():
    inner = _ok("concrete_maturity_strength", {"text": _MATURITY_TEXT_BRACKETS})
    _assert_maturity_582(inner)


def test_maturity_empty_is_named_missing_not_cause():
    env = _err("concrete_maturity_strength", {})
    _assert_named_missing(env, (
        ("temperature_history_c", "°C"),
        ("time_intervals_hours", "h"),
    ))
    assert env.get("cause") in (None, "")


# ── concrete_mix_design_sg (LIVE: tool_error) ──────────────────────────────

_MIX_ASK = "concrete mix design sg w/c=0.48 table cement 320"


def test_mix_design_sg_name_beats_slip_form_stem():
    """'concrete mix' is also the slip-form stem — full name must win."""
    assert _formula_calculator_name_from_message(_MIX_ASK) == (
        "concrete_mix_design_sg"
    )
    assert _formula_calculator_name_from_message(
        "concrete_mix_design_sg w/c=0.48",
    ) == "concrete_mix_design_sg"
    assert _forced(_MIX_ASK) == "construction_calc"


def test_mix_design_wc_assignment_string_is_320():
    """Live params string is w/c=0.48 (slash key), not wc=0.48."""
    inner = _ok("concrete_mix_design_sg", "w/c=0.48")
    assert inner["cement_kg_m3"] == pytest.approx(320.0, abs=0.5)
    via_text = _ok("concrete_mix_design_sg", {"text": _MIX_ASK})
    assert via_text["cement_kg_m3"] == pytest.approx(320.0, abs=0.5)
    env = _tool("concrete_mix_design_sg", {"text": _MIX_ASK})
    assert env.get("ok") is True, env
    assert env["result"]["result"]["cement_kg_m3"] == pytest.approx(320.0, abs=0.5)


def test_mix_design_sand_sg_is_not_dune_sand_pct_error():
    """Stem suffix sand → dune_sand_pct raised; table SGs must not."""
    inner = _ok("concrete_mix_design_sg", {"w/c": 0.48, "sand": 2.67})
    assert inner["cement_kg_m3"] == pytest.approx(320.0, abs=0.5)
    via_text = _ok("concrete_mix_design_sg", {
        "text": "w/c=0.48 dune sand 2.67",
    })
    assert via_text["cement_kg_m3"] == pytest.approx(320.0, abs=0.5)
    still = _err("concrete_mix_design_sg", {
        "w_c_ratio": 0.48, "dune_sand_pct": 0.3,
    })
    assert "dune_sand_pct" in still["error"]


def test_mix_design_empty_is_named_missing_wc():
    env = _err("concrete_mix_design_sg", {})
    assert "w_c_ratio" in env["missing"]
    assert "missing required" in env["error"]


# ── cost_buildup_concrete (LIVE: ask1 status error) ────────────────────────

_COST_ASK = "cost buildup concrete quantity_m3=100"


def test_cost_buildup_concrete_name_beats_rebar_stem():
    """'cost buildup' is also the rebar stem — full name must win."""
    assert _formula_calculator_name_from_message(_COST_ASK) == (
        "cost_buildup_concrete"
    )
    assert _forced(_COST_ASK) == "construction_calc"


def test_cost_buildup_concrete_quantity_m3_100_succeeds():
    inner = _ok("cost_buildup_concrete", {"quantity_m3": 100})
    assert inner["material_cost_sar_m3"] == pytest.approx(160.93, abs=0.02)
    assert inner["total_project_value_sar"] == pytest.approx(30883, abs=2)
    via_text = _ok("cost_buildup_concrete", {"text": _COST_ASK})
    assert via_text["material_cost_sar_m3"] == pytest.approx(160.93, abs=0.02)
    env = _tool("cost_buildup_concrete", {"text": _COST_ASK})
    assert env.get("ok") is True, env
    assert env["result"]["status"] == "success"
    assert env["result"]["result"]["selling_price_sar_m3"] == pytest.approx(
        308.83, abs=0.02,
    )


def test_cost_buildup_concrete_empty_is_named_missing_not_json_crash():
    env = _err("cost_buildup_concrete", {})
    _assert_named_missing(env, (("quantity_m3", "m3"),))
    assert env.get("cause") in (None, "")
    json.dumps(env)
    via_name_only = _err("cost_buildup_concrete", {
        "text": "cost buildup concrete",
    })
    assert "quantity_m3" in via_name_only["missing"]


# ── cut_fill_balance (LIVE: ask1 cause=error) ──────────────────────────────

def _assert_cut_fill(inner):
    assert inner["balance_bank_m3"] == pytest.approx(1500.0, abs=0.01)
    assert inner["haul_loose_m3"] == pytest.approx(1875.0, abs=0.01)


def test_cut_fill_cut_fill_aliases_are_1500_and_1875():
    inner = _ok("cut_fill_balance", {"cut": 5000, "fill": 3500})
    _assert_cut_fill(inner)
    via_lists = _ok("cut_fill_balance", {"cut": [5000], "fill": [3500]})
    _assert_cut_fill(via_lists)
    via_assign = _ok("cut_fill_balance", "cut=5000 fill=3500")
    _assert_cut_fill(via_assign)


def test_cut_fill_both_asks_from_text_stabilize():
    ask1 = _ok("cut_fill_balance", {"text": "cut=5000 fill=3500"})
    _assert_cut_fill(ask1)
    ask2 = _ok("cut_fill_balance", {
        "text": "cut_fill_balance cut=5000 fill=3500",
    })
    _assert_cut_fill(ask2)
    env = _tool("cut_fill_balance", {"text": "cut=5000 fill=3500"})
    assert env.get("ok") is True, env
    assert "cause" not in env.get("result", {})
    _assert_cut_fill(env["result"]["result"])


def test_cut_fill_empty_is_named_missing_not_cause():
    env = _err("cut_fill_balance", {})
    _assert_named_missing(env, (
        ("cut_volume_m3", "m3"),
        ("fill_volume_m3", "m3"),
    ))
    assert env.get("cause") in (None, "")


# ── dewatering_well_point_spacing (LIVE: no_tool×2) ────────────────────────

_WELL_ASK = "well point spacing permeability=0.002 drawdown=12"


def test_well_point_spacing_routes_construction_calc_not_other_tool():
    assert _formula_calculator_name_from_message(_WELL_ASK) == (
        "dewatering_well_point_spacing"
    )
    assert _formula_calculator_name_from_message(
        "well-point spacing k=0.002 drawdown=12",
    ) == "dewatering_well_point_spacing"
    assert _message_is_formula_style_ask(_WELL_ASK) is True
    assert _message_wants_named_calculator(_WELL_ASK) is True
    assert _message_wants_drawing_qto(_WELL_ASK) is False
    assert _forced(_WELL_ASK) == "construction_calc"
    assert _formula_calculator_name_from_message(_WELL_ASK) != (
        "dewatering_uplift_check"
    )


def test_well_point_permeability_drawdown_aliases_and_ask_text():
    inner = _ok("dewatering_well_point_spacing", {
        "permeability": 0.002, "drawdown": 12,
    })
    assert inner["well_point_spacing_m"] == pytest.approx(1.5, abs=1e-9)
    assert inner["stages_needed"] == 3
    via_k = _ok("dewatering_well_point_spacing", {"k": 0.002, "drawdown": 12})
    assert via_k["well_point_spacing_m"] == pytest.approx(1.5, abs=1e-9)
    via_text = _ok("dewatering_well_point_spacing", {"text": _WELL_ASK})
    assert via_text["well_point_spacing_m"] == pytest.approx(1.5, abs=1e-9)
    env = _tool("dewatering_well_point_spacing", {"text": _WELL_ASK})
    assert env.get("ok") is True, env
    assert env["result"]["result"]["stages_needed"] == 3


def test_well_point_empty_is_named_missing():
    env = _err("dewatering_well_point_spacing", {})
    assert "soil_permeability_m_s" in env["missing"]
    assert "required_drawdown_m" in env["missing"]
    assert "missing required" in env["error"]


# ── helpers stay unregistered ──────────────────────────────────────────────

def test_reprobe_helpers_are_not_registered_calculators():
    assert "calculator_name_from_text" not in CALCULATORS
    assert "coerce_calc_params" not in CALCULATORS
    bound = bind_calculation_params(
        CALCULATORS["cut_fill_balance"], {"cut": 5000, "fill": 3500},
    )
    assert bound["cut_volume_m3"] == 5000
    assert bound["fill_volume_m3"] == 3500
    env = run_calculation("concrete_mix_design_sg", "w/c=0.48")
    assert env.get("status") == "success", env
