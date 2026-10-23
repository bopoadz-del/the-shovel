"""Agent B calculator slice — three live defects, synthetic inputs only.

1. cost_buildup_concrete wastes plant, labour and the indirect line, so the
   selling price is not the build-up its own inputs describe.
2. dewatering_well_point_spacing mentions drawdown, which the cash-flow
   detector treats as an S-curve. Formula predispatch is then skipped and
   the assistant calls no calculator.
3. productivity_rate returns a bare rate. The output must carry its unit.
"""
from __future__ import annotations

import pytest

from app.agents.runtime import (
    _forced_specific_tool,
    _message_is_formula_style_ask,
    _message_wants_cash_flow,
    _message_wants_locked_deliverable,
)
from app.core.action_router import message_wants_cash_flow
from app.lib.construction_formulas import run_calculation
from app.lib.construction_formulas_commercial import productivity_rate


# Hand arithmetic from cost_buildup_concrete's own defaults.
# cement (400/1000)*190 = 76
# aggregate (1839/1000)*29 = 53.331
# water (160/1000)*10 = 1.6          # 160 L = 0.16 m3
# microsilica 12*1.5 = 18
# plasticizer 5*2.4 = 12
# material = 160.931
# waste is on material only: 160.931 * 1.03 = 165.75893
# direct = 165.75893 + 39.2 + 8 + 3 + 2.5 + 4 = 222.45893
# selling = 222.45893 * 1.18 / 0.85 = 308.82534 → 308.83 SAR/m3
# total = 308.83 * 100 = 30883 SAR
_SELLING_SAR_M3 = 308.83
_TOTAL_SAR_100 = 30883.0


def test_cost_buildup_concrete_waste_is_on_material_only():
    """Plant, labour, power and the indirect line are not wasted."""
    r = run_calculation("cost_buildup_concrete", {"quantity_m3": 100})
    assert r["status"] == "success", r
    inner = r["result"]
    assert inner["material_cost_sar_m3"] == pytest.approx(160.93, abs=0.01)
    assert inner["selling_price_sar_m3"] == pytest.approx(_SELLING_SAR_M3, abs=0.01)
    assert inner["total_project_value_sar"] == pytest.approx(_TOTAL_SAR_100, abs=0.01)
    # The displayed unit rate times the quantity is the project total.
    assert inner["total_project_value_sar"] == pytest.approx(
        inner["selling_price_sar_m3"] * 100, abs=0.01,
    )
    note = str(inner.get("note") or "")
    assert "308.83" in note
    assert "SAR/m3" in note
    assert "30883" in note.replace(",", "")


_WELL_ASK = "well point spacing permeability=0.002 drawdown=12"
_CASH_FLOW_ASK = "Give me the project cash flow forecast and cumulative drawdown"
_AGENT_TOOLS = {
    "construction_calc",
    "cash_flow_forecast",
    "drawing_qto",
    "formula_executor_v2",
    "search_project_documents",
    "generate_wbs",
}


def test_well_point_spacing_drawdown_is_not_a_cash_flow_lock():
    """Drawdown here is water-table lowering, not an S-curve.

    The live assistant has cash_flow_forecast. A bare 'drawdown' match
    locks the turn, skips construction_calc predispatch, and — because
    Kimi/Groq will not honour a forced tool_choice — calls no tool.
    """
    assert _message_wants_cash_flow(_WELL_ASK) is False
    assert message_wants_cash_flow(_WELL_ASK) is False
    assert _message_wants_locked_deliverable(_WELL_ASK) is False
    assert _message_is_formula_style_ask(_WELL_ASK) is True
    assert _forced_specific_tool(
        [{"role": "user", "content": _WELL_ASK}], _AGENT_TOOLS,
    ) == "construction_calc"
    # A real S-curve ask that says drawdown still selects cash flow.
    assert _message_wants_cash_flow(_CASH_FLOW_ASK) is True
    assert message_wants_cash_flow(_CASH_FLOW_ASK) is True
    assert _forced_specific_tool(
        [{"role": "user", "content": _CASH_FLOW_ASK}], _AGENT_TOOLS,
    ) == "cash_flow_forecast"


def test_productivity_rate_result_carries_per_hour_unit():
    """500 units in 40 man-hours is 12.5 per man-hour. The crew is not a second divisor."""
    direct = productivity_rate(output_quantity=500, labor_hours=40, crew_size=4)
    assert direct["rate_per_hour"] == pytest.approx(12.5, abs=0.001)
    assert direct["rate_per_worker_hour"] == pytest.approx(12.5, abs=0.001)
    assert direct["unit"] == "/hr"
    assert direct["value"] == pytest.approx(12.5, abs=0.001)
    assert direct["rate_per_worker_hour_unit"] == "/worker-hr"
    note = str(direct.get("note") or "")
    assert "/hr" in note
    assert "/worker-hr" in note

    env = run_calculation(
        "productivity_rate",
        {"output_quantity": 500, "labor_hours": 40, "crew_size": 4},
    )
    assert env["status"] == "success", env
    inner = env["result"]
    assert inner["unit"] == "/hr"
    assert inner["value"] == pytest.approx(12.5, abs=0.001)
    assert "/hr" in str(inner.get("note") or "")
