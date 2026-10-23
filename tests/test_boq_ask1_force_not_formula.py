"""Live tip 0a95d03: boq_process ask1 must not self-code / formula_executor_v2.

#660 accepted inline CSV and blocked named_calculator, but ask1 still routed
self-coding → formula_executor_v2 (HTTP 429). Force boq_processor, hard-exclude
formula path, and refuse self-coding handoff when lines are inline.
"""
from __future__ import annotations

from app.agents.runtime import (
    _forced_specific_tool,
    _inline_boq_hard_excludes,
    _should_handoff_unmatched_calc,
)
from app.core.site_vocab import message_has_inline_boq_lines

LIVE_ASK1 = (
    "Use boq_process on these SYNTHETIC CSV/BOQ lines in the message:\n"
    "Item,Desc,Qty,Unit,Rate,Amount\n"
    "1.1,Excavation,500,m3,45,22500\n"
    "1.2,Concrete footings,120,m3,480,57600\n"
    "1.3,Rebar,15,t,3200,48000\n"
    "1.4,Blockwork,600,m2,85,51000\n"
    "Process and give section/total values."
)


def test_live_ask1_detected_inline():
    assert message_has_inline_boq_lines(LIVE_ASK1)


def test_live_ask1_forces_boq_processor():
    avail = {"boq_processor", "formula_executor_v2", "construction_calc", "generate_wbs"}
    assert (
        _forced_specific_tool([{"role": "user", "content": LIVE_ASK1}], avail)
        == "boq_processor"
    )


def test_live_ask1_hard_excludes_formula_executor():
    ex = _inline_boq_hard_excludes(LIVE_ASK1)
    assert "formula_executor_v2" in ex


def test_live_ask1_no_self_coding_handoff():
    assert _should_handoff_unmatched_calc(LIVE_ASK1) is False
