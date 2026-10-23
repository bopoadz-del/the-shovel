"""Live tip d9d5971: boq_process ask1 must hit boq_*, not construction_calc.

Ask1 FAIL: route project-assistant / below_routing_gate; tools=[construction_calc]
only. The model refused boq_processor (“needs uploaded file_path”) and echoed
arithmetic. Ask2 PASSed with boq_processor against fixture_boq_d.csv.

Ask1 pastes SYNTHETIC CSV/BOQ lines inline and says Use boq_process…
Expected tools: boq_process OR boq_processor.

Root on current main: the CSV line “Excavation” trips _CALC_VERB_RE, so
_looks_like_self_contained_calculation / _message_is_formula_style_ask are
True and _predispatch_formula_calc steals to construction_calc. Hard-exclude
already lists construction_calc but does not stop predispatch.

Lock: inline BOQ is not a formula/named-calc steal; formula predispatch
must not run; remaining-deliverable predispatch must invoke boq_process.
"""
from __future__ import annotations

import pytest

from app.agents.runtime import (
    _forced_specific_tool,
    _inline_boq_hard_excludes,
    _looks_like_self_contained_calculation,
    _message_is_formula_style_ask,
    _message_wants_named_calculator,
    _predispatch_formula_calc,
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


def test_live_ask1_is_not_formula_or_named_calc():
    """'Excavation' in a pasted BOQ row must not elect construction_calc."""
    assert _message_wants_named_calculator(LIVE_ASK1) is False
    assert _looks_like_self_contained_calculation(LIVE_ASK1) is False
    assert _message_is_formula_style_ask(LIVE_ASK1) is False


def test_live_ask1_forces_boq_star_not_construction_calc():
    avail = {
        "boq_processor",
        "boq_process",
        "construction_calc",
        "formula_executor_v2",
        "search_project_documents",
    }
    forced = _forced_specific_tool([{"role": "user", "content": LIVE_ASK1}], avail)
    assert forced in {"boq_processor", "boq_process"}
    assert forced != "construction_calc"


def test_live_ask1_hard_excludes_construction_calc():
    ex = _inline_boq_hard_excludes(LIVE_ASK1)
    assert "construction_calc" in ex
    assert "formula_executor_v2" in ex


@pytest.mark.asyncio
async def test_live_ask1_formula_predispatch_does_not_steal():
    from app.agents.runtime import Agent

    agent = Agent(
        name="project-assistant",
        description="t",
        system_prompt="t",
        allowed_blocks=["construction"],
    )
    msgs = [{"role": "user", "content": LIVE_ASK1}]
    rec = await _predispatch_formula_calc(
        agent, msgs, "proj", operator_text=LIVE_ASK1,
    )
    assert rec is None
    assert not any(
        "PLATFORM PRE-DISPATCH: construction_calc" in str(m.get("content") or "")
        for m in msgs
    )


@pytest.mark.asyncio
async def test_live_ask1_predispatch_invokes_boq_process(monkeypatch):
    from app.agents.runtime import _predispatch_remaining_deliverables
    from app.blocks.boq_processor import BOQProcessorBlock
    from app.containers.construction import ConstructionContainer
    from app.dependencies import get_block_instance as _real_get_block

    # Virgin CI does not load the construction kit, so get_block_instance
    # ("boq_processor") 503s. Instantiate the block class directly — the
    # same path tests/test_boq_process_inline_lines.py already uses.
    boq_block = BOQProcessorBlock()

    def _get(name: str):
        if name == "construction":
            return ConstructionContainer()
        if name == "boq_processor":
            return boq_block
        return _real_get_block(name)

    monkeypatch.setattr("app.dependencies.get_block_instance", _get)

    class _A:
        allowed_blocks = ["construction"]
        name = "project-assistant"

    msgs = [{"role": "user", "content": LIVE_ASK1}]
    out = await _predispatch_remaining_deliverables(_A(), msgs, "proj")
    assert out is not None
    assert out["name"] in {"boq_process", "boq_processor"}
    assert out.get("ok") is True
    result = out.get("result") or {}
    assert result.get("status") in {"success", "ok", "completed"}, result
    blob = str(result).lower()
    assert "excavation" in blob or "1.1" in blob
    assert "179100" in str(result).replace(",", "").replace(" ", "") or result.get("total_cost") == pytest.approx(179100.0)
    injected = msgs[-1]["content"]
    assert "PLATFORM PRE-DISPATCH" in injected
    assert "boq_process" in injected or "boq_processor" in injected
    assert "construction_calc" not in injected.split("PLATFORM PRE-DISPATCH")[-1][:80]
