"""drawing_qto QTO asks must not elect named_calculator → construction_calc.

Live tip 50c37f: probe asks with synthetic footing/slab L×W×D routed
reason=named_calculator and ran construction_calc ×3. #658 ranked
drawing_qto in intent_map / forced-tool, but `_message_wants_named_calculator`
still returned True via `_looks_like_self_contained_calculation` and
formula predispatch stole the turn.
"""
from __future__ import annotations

from app.agents.runtime import (
    _forced_specific_tool,
    _message_wants_drawing_qto,
    _message_wants_named_calculator,
)


LIVE_DRAWING_QTO_ASKS = [
    (
        "Use drawing_qto / quantity takeoff. SYNTHETIC footing and slab geometry in this message "
        "(no file): 6 pad footings each 2.0m x 2.0m x 0.6m deep; ground slab 18m x 12m x 0.2m thick. "
        "Compute concrete volumes (m3) for footings and slab, and total. Show numbers clearly."
    ),
    (
        "drawing_qto takeoff from synthetic dims: strip footing 40m long x 0.8m wide x 0.5m deep; "
        "raft slab 25m x 15m x 0.35m. Report footing volume, slab volume, and combined concrete m3."
    ),
]


def test_live_drawing_qto_asks_are_detected():
    for q in LIVE_DRAWING_QTO_ASKS:
        assert _message_wants_drawing_qto(q), q


def test_live_drawing_qto_asks_are_not_named_calculator():
    for q in LIVE_DRAWING_QTO_ASKS:
        assert not _message_wants_named_calculator(q), q


def test_live_drawing_qto_asks_force_drawing_qto():
    avail = {"drawing_qto", "construction_calc", "generate_wbs", "search_project_documents"}
    for q in LIVE_DRAWING_QTO_ASKS:
        assert _forced_specific_tool(
            [{"role": "user", "content": q}], avail,
        ) == "drawing_qto", q


def test_leftover_l6_volume_still_named_calculator():
    q = (
        "Stay on smart-orchestrator. Bank volume of a rectangular trench "
        "14.5 m long by 3.2 m wide by 1.75 m deep. Start with the "
        "smart_orchestrator tool then a calculator."
    )
    assert _message_wants_named_calculator(q)
    assert not _message_wants_drawing_qto(q)
