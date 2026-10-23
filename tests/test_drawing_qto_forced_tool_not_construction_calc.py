"""drawing_qto asks must force drawing_qto, not construction_calc.

Live exit 78bd9ca: a quantity-takeoff / drawing_qto ask was stolen by
construction_calc because ``_DELIVERABLE_PHRASES`` forced tool_choice=
required while ``_forced_specific_tool`` had no named drawing_qto row.
Same class as cash_flow_forecast → generate_wbs and procurement/RFI →
construction_calc.

Self-contained L×W×D volume asks (leftover L6) must stay on
construction_calc — they do not match the drawing_qto phrases.
"""
from __future__ import annotations

from app.agents.runtime import _forced_specific_tool, _looks_like_self_contained_calculation


DRAWING_QTO_ASKS = [
    "do a quantity takeoff from the infrastructure drawings - pipe lengths and manhole counts",
    "measure the floor area from the ground floor plan drawing",
    "Use drawing_qto. Synthetic DXF with 3 pipes and 2 manholes. Produce pipe lengths and manhole counts.",
    "drawing_qto on the ground floor plan",
    "extract quantities from this drawing",
    "quantity takeoff from drawing_qz_0101.pdf",
]


def test_drawing_qto_asks_force_drawing_qto_not_calc():
    avail = {"drawing_qto", "construction_calc", "generate_wbs", "search_project_documents"}
    for q in DRAWING_QTO_ASKS:
        forced = _forced_specific_tool([{"role": "user", "content": q}], avail)
        assert forced == "drawing_qto", (q, forced)


def test_drawing_qto_ask_is_not_forced_onto_construction_calc_when_missing():
    """If the toolkit omitted drawing_qto, do not steal onto the calculator."""
    avail = {"construction_calc", "generate_wbs"}
    forced = _forced_specific_tool(
        [{
            "role": "user",
            "content": (
                "do a quantity takeoff from the infrastructure drawings "
                "- pipe lengths and manhole counts"
            ),
        }],
        avail,
    )
    assert forced != "construction_calc"
    assert forced is None


def test_leftover_l6_volume_still_forces_construction_calc():
    """A self-contained trench volume must not be stolen onto drawing_qto."""
    q = (
        "Stay on smart-orchestrator. Bank volume of a rectangular trench "
        "14.5 m long by 3.2 m wide by 1.75 m deep. Start with the "
        "smart_orchestrator tool then a calculator."
    )
    assert _looks_like_self_contained_calculation(q)
    avail = {"drawing_qto", "construction_calc", "generate_wbs"}
    assert _forced_specific_tool(
        [{"role": "user", "content": q}], avail,
    ) == "construction_calc"
