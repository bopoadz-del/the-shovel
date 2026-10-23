"""Live Phase 2: procurement_list_generator must be in the chat toolkit.

Second ask for a procurement list failed on project-assistant: the model
said ``procurement_list_generator`` is not in its toolkit and the turn
was forced onto ``construction_calc``. First ask sometimes worked via
the generic ``construction`` / ``boq_processor`` path.

The action is already registered on ConstructionContainer. This file
locks the missing half: expose it as a top-level tool (same pattern as
``generate_wbs`` / ``cash_flow_forecast``), force it for tool-shaped
procurement asks, and never treat ``construction_calc`` as the primary
path when both are available.

Works in both boot modes — advertise + force do not import the
construction kit; dispatch is mocked.
"""
from __future__ import annotations

import asyncio
import json

from app.agents.runtime import Agent, _forced_specific_tool, get_agent, load_agents


# Feature-matrix / live Phase 2 asks. Prompt 2 is the one that stayed on
# project-assistant (below_routing_gate) and then missed the tool.
_PROCUREMENT_ASKS = (
    "generate a procurement list for the MEP package",
    "what materials do we need to buy for the substructure works?",
    "use procurement_list_generator for the MEP buy list",
    "call procurement_list_generator",
    "produce a material list for the steel package",
)


def _agent(blocks=None):
    return Agent(
        name="t",
        description="t",
        system_prompt="t",
        allowed_blocks=blocks or [],
    )


def _tool_names(agent, project_id=None):
    return {
        (t.get("function") or {}).get("name") or t.get("name")
        for t in agent.tool_definitions(project_id=project_id)
    }


def _run(coro):
    return asyncio.run(coro)


def _call(agent, name, args):
    return _run(agent._run_tool_call({
        "id": "p1",
        "function": {"name": name, "arguments": json.dumps(args)},
    }))


# ── Exposure ────────────────────────────────────────────────────────────────


def test_procurement_list_generator_offered_when_construction_allowed():
    assert "procurement_list_generator" in _tool_names(_agent(["construction"]))
    assert "procurement_list_generator" not in _tool_names(_agent([]))
    assert "procurement_list_generator" not in _tool_names(
        _agent(["boq_processor"])
    )


def test_project_assistant_advertises_procurement_list_generator():
    """The live UI agent is project-assistant. If the name is missing
    here the model will say it is not in the toolkit."""
    load_agents()
    pa = get_agent("project-assistant")
    names = _tool_names(pa, project_id="fence-check")
    assert "procurement_list_generator" in names
    assert "construction_calc" in names  # still present; not the primary path


def test_construction_capable_agents_advertise_the_action():
    load_agents()
    for agent_id in ("project-assistant", "heavy-reasoning", "quantity-surveyor"):
        names = _tool_names(get_agent(agent_id), project_id="fence-check")
        assert "procurement_list_generator" in names, agent_id


# ── Forced-tool routing ─────────────────────────────────────────────────────


def test_procurement_asks_force_procurement_list_generator_not_calc():
    avail = {"procurement_list_generator", "construction_calc", "generate_wbs"}
    for q in _PROCUREMENT_ASKS:
        msgs = [{"role": "user", "content": q}]
        forced = _forced_specific_tool(msgs, avail)
        assert forced == "procurement_list_generator", (q, forced)


def test_second_ask_still_forces_procurement_not_calc():
    """Conversation continuity: first ask may have gone construction /
    boq_processor; the second user turn must still force the action."""
    avail = {"procurement_list_generator", "construction_calc", "construction"}
    msgs = [
        {"role": "user", "content": "generate a procurement list for the MEP package"},
        {"role": "assistant", "content": "Here is the MEP procurement list."},
        {
            "role": "user",
            "content": "what materials do we need to buy for the substructure works?",
        },
    ]
    assert _forced_specific_tool(msgs, avail) == "procurement_list_generator"


def test_procurement_ask_is_not_forced_onto_construction_calc_when_missing():
    """If the toolkit omitted the action, do not steal the turn onto the
    calculator — that is the live miss (model then invents a calc)."""
    avail = {"construction_calc", "generate_wbs"}
    forced = _forced_specific_tool(
        [{"role": "user", "content": "generate a procurement list for the MEP package"}],
        avail,
    )
    assert forced != "construction_calc"
    assert forced is None


def test_schedule_procurement_question_is_not_forced():
    """'how long is procurement on the schedule?' is RAG / WBS Q&A."""
    avail = {"procurement_list_generator", "construction_calc", "generate_wbs"}
    assert _forced_specific_tool(
        [{"role": "user", "content": "how long is procurement on the schedule?"}],
        avail,
    ) is None


def test_named_calculator_still_forces_construction_calc():
    avail = {"procurement_list_generator", "construction_calc"}
    assert _forced_specific_tool(
        [{"role": "user", "content": "run a dewatering uplift check for 23 m water depth"}],
        avail,
    ) == "construction_calc"
    assert _forced_specific_tool(
        [{"role": "user", "content": "calculate material consumption for 100 m3 of concrete"}],
        avail,
    ) == "construction_calc"


# ── Dispatch ────────────────────────────────────────────────────────────────


def test_synthetic_tool_hits_the_container(monkeypatch):
    calls = {}

    class FakeBlock:
        async def procurement_list_generator(self, input_data, params):
            calls["input"] = input_data
            calls["params"] = params
            return {
                "status": "success",
                "action": "procurement_list",
                "total_items": 1,
                "procurement_list": [{"item": "Rebar", "quantity": 3.2}],
            }

    import app.dependencies as deps
    monkeypatch.setattr(deps, "get_block_instance", lambda name: FakeBlock())
    r = _call(_agent(["construction"]), "procurement_list_generator", {
        "quantities": {"Rebar": {"quantity": 3.2, "unit": "t"}},
        "budget": "200000",
    })
    assert r["ok"], r
    assert r["name"] == "procurement_list_generator"
    assert calls["params"]["quantities"]["Rebar"]["quantity"] == 3.2
    assert float(calls["params"]["budget"]) == 200000


def test_procurement_list_maps_boq_qty_rate_onto_quantity_unit_price(monkeypatch):
    captured = {}

    class Stub:
        async def procurement_list_generator(self, input_data, params):
            captured["input"] = input_data
            captured["params"] = params
            return {"status": "success", "action": "procurement_list", "procurement_list": []}

    import app.dependencies as deps
    monkeypatch.setattr(deps, "get_block_instance", lambda name: Stub())

    from app.agents.core.tool_registry import ToolCall
    from app.agents.hats.procurement.tools import handle_procurement_list_generator

    call = ToolCall(
        agent=_agent(["construction"]),
        name="procurement_list_generator",
        args={"boq": [{
            "description": "High-yield reinforcement",
            "unit": "t",
            "qty": 520,
            "rate": 3150,
            "amount": 1638000,
        }]},
    )
    result = _run(handle_procurement_list_generator(call))
    assert result["ok"] is True, result
    row = captured["params"]["boq"][0]
    assert row["quantity"] == 520
    assert row["unit_price"] == 3150


def test_procurement_list_boq_field_table_does_not_clobber(monkeypatch):
    """One alias table for every BOQ row: canonical wins, strings coerce.

    ``unit_rate`` and ``amount`` are the same table as ``rate`` and ``qty``.
    Rows that arrive on ``input`` or nested ``params`` are mapped too.
    """
    captured = {}

    class Stub:
        async def procurement_list_generator(self, input_data, params):
            captured["input"] = input_data
            captured["params"] = params
            return {"status": "success", "action": "procurement_list", "procurement_list": []}

    import app.dependencies as deps
    monkeypatch.setattr(deps, "get_block_instance", lambda name: Stub())

    from app.agents.core.tool_registry import ToolCall
    from app.agents.hats.procurement.tools import handle_procurement_list_generator

    call = ToolCall(
        agent=_agent(["construction"]),
        name="procurement_list_generator",
        args={
            "params": {"boq": [{
                "description": "Kept",
                "qty": 1,
                "quantity": 9,
                "rate": 2,
                "unit_price": "8",
                "amount": 3,
                "total": 7,
            }]},
            "input": {"boq": [{
                "description": "Alias",
                "unit": "m",
                "unit_rate": "12.5",
                "amount": "1,000",
            }], "line_items": [{
                "description": "Lines",
                "qty": "4",
                "rate": 6,
            }]},
        },
    )
    result = _run(handle_procurement_list_generator(call))
    assert result["ok"] is True, result
    kept = captured["params"]["boq"][0]
    assert kept["quantity"] == 9
    assert kept["unit_price"] == 8
    assert kept["total"] == 7
    alias = captured["input"]["boq"][0]
    assert alias["unit_price"] == 12.5
    assert alias["total"] == 1000
    assert "quantity" not in alias
    lines = captured["input"]["line_items"][0]
    assert lines["quantity"] == 4
    assert lines["unit_price"] == 6


def test_procurement_list_parameters_are_fully_declared():
    from app.agents.core.tool_registry import get
    from app.lib.formula_registry import TOOL_KINDS, schema_gaps

    spec = get("procurement_list_generator")
    assert spec is not None and spec.params
    for name, param in spec.params.items():
        assert (param.label or "").strip(), name
        assert param.kind in TOOL_KINDS, name
        assert param.required is not None, name
        if param.kind in ("number", "integer"):
            assert param.unit and param.unit != "-", name
            assert param.lo is not None and param.hi is not None, name
            assert param.lo < param.hi, name
    assert not [
        gap for gap in schema_gaps()
        if gap.startswith("tool:procurement_list_generator")
    ]


def test_synthetic_tool_refuses_without_construction_block():
    r = _call(_agent([]), "procurement_list_generator", {})
    assert r["ok"] is False
    assert "allowed_blocks" in (r["result"].get("error") or "").lower()


# ── Tender evaluation keys ──────────────────────────────────────────────────


def test_evaluate_tender_reads_bare_criterion_keys():
    from app.agents.hats.procurement.formulas.construction_knowledge import (
        evaluate_tender,
    )
    r = evaluate_tender([
        {"name": "A", "technical": 80, "commercial": 70, "hse": 90},
        {"name": "B", "technical": 70, "commercial": 85, "hse": 80},
    ])
    assert r["recommended"]["name"] == "B"
    assert [t["weighted_total"] for t in r["ranked_tenderers"]] == [75.35, 73.8]


def test_evaluate_tender_refuses_when_no_criterion_scored():
    from app.agents.hats.procurement.formulas.construction_knowledge import (
        evaluate_tender,
    )
    r = evaluate_tender([{"name": "A"}, {"name": "B"}])
    assert "error" in r
    assert "ranked_tenderers" not in r


def test_evaluate_tender_one_rule_for_every_weight():
    """Caller-supplied criteria use the same ``<c>_score`` or bare ``<c>`` rule.

    A suffixed key wins when both forms are present. A criterion the weights
    do not name is ignored. An explicit zero is a score, so the panel is ranked.
    """
    from app.agents.hats.procurement.formulas.construction_knowledge import (
        evaluate_tender,
    )
    r = evaluate_tender(
        [
            {"name": "A", "quality_score": 40, "quality": 99, "price": 80},
            {"name": "B", "quality": 90, "price_score": 70, "ignored": 100},
        ],
        weights={"quality": 0.5, "price": 0.5},
    )
    assert r["recommended"]["name"] == "B"
    assert [t["weighted_total"] for t in r["ranked_tenderers"]] == [80.0, 60.0]

    zeros = evaluate_tender([
        {"name": "A", "technical": 0, "commercial": 0, "hse": 0},
    ])
    assert "ranked_tenderers" in zeros
    assert zeros["ranked_tenderers"][0]["weighted_total"] == 0.0

    only_local = evaluate_tender([{"name": "A", "local_content": 100}])
    assert only_local["ranked_tenderers"][0]["weighted_total"] == 3.0
