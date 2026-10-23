"""Live Phase 2: rfi_generator must be in the chat toolkit.

Both RFI draft asks failed on project-assistant: no tool, the wrong
tool (``construction_calc``), or RFI prose without calling
``rfi_generator``. The action is already registered on
ConstructionContainer. This file locks the missing half: expose it as
a top-level tool (same pattern as ``procurement_list_generator`` /
``generate_wbs`` / ``cash_flow_forecast``), force it for tool-shaped
RFI draft asks, and never treat ``construction_calc`` as the primary
path when both are available.

Works in both boot modes — advertise + force do not import the
construction kit; dispatch is mocked.
"""
from __future__ import annotations

import asyncio
import json

from app.agents.runtime import Agent, _forced_specific_tool, get_agent, load_agents


# Feature-matrix / live Phase 2 asks. Both failed: toolkit miss, then
# construction_calc or un-tooled RFI prose.
_RFI_DRAFT_ASKS = (
    "draft an RFI asking the engineer to clarify the rebar detail at the transfer beam",
    "create a request for information document",
    "use rfi_generator for the missing rebar lap detail",
    "call rfi_generator",
    "raise an RFI on the transfer-beam rebar clarification",
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
        "id": "r1",
        "function": {"name": name, "arguments": json.dumps(args)},
    }))


# ── Exposure ────────────────────────────────────────────────────────────────


def test_rfi_generator_offered_when_construction_allowed():
    assert "rfi_generator" in _tool_names(_agent(["construction"]))
    assert "rfi_generator" not in _tool_names(_agent([]))
    assert "rfi_generator" not in _tool_names(
        _agent(["boq_processor"])
    )


def test_project_assistant_advertises_rfi_generator():
    """The live UI agent is project-assistant. If the name is missing
    here the model will say it is not in the toolkit."""
    load_agents()
    pa = get_agent("project-assistant")
    names = _tool_names(pa, project_id="fence-check")
    assert "rfi_generator" in names
    assert "construction_calc" in names  # still present; not the primary path


def test_construction_capable_agents_advertise_the_action():
    load_agents()
    for agent_id in ("project-assistant", "heavy-reasoning", "contracts-manager"):
        names = _tool_names(get_agent(agent_id), project_id="fence-check")
        assert "rfi_generator" in names, agent_id


# ── Forced-tool routing ─────────────────────────────────────────────────────


def test_rfi_draft_asks_force_rfi_generator_not_calc():
    avail = {"rfi_generator", "construction_calc", "generate_wbs"}
    for q in _RFI_DRAFT_ASKS:
        msgs = [{"role": "user", "content": q}]
        forced = _forced_specific_tool(msgs, avail)
        assert forced == "rfi_generator", (q, forced)


def test_second_ask_still_forces_rfi_not_calc():
    """Conversation continuity: first ask may have gone construction /
    construction_calc; the second user turn must still force the action."""
    avail = {"rfi_generator", "construction_calc", "construction"}
    msgs = [
        {
            "role": "user",
            "content": (
                "draft an RFI asking the engineer to clarify the "
                "rebar detail at the transfer beam"
            ),
        },
        {"role": "assistant", "content": "Here is a draft RFI."},
        {
            "role": "user",
            "content": "create a request for information document",
        },
    ]
    assert _forced_specific_tool(msgs, avail) == "rfi_generator"


def test_rfi_ask_is_not_forced_onto_construction_calc_when_missing():
    """If the toolkit omitted the action, do not steal the turn onto the
    calculator — that is the live miss (model then invents a calc)."""
    avail = {"construction_calc", "generate_wbs"}
    forced = _forced_specific_tool(
        [{
            "role": "user",
            "content": (
                "draft an RFI asking the engineer to clarify the "
                "rebar detail at the transfer beam"
            ),
        }],
        avail,
    )
    assert forced != "construction_calc"
    assert forced is None


def test_rfi_status_question_is_not_forced():
    """'how many RFIs are open?' is rfi_management / RAG, not a draft."""
    avail = {"rfi_generator", "construction_calc", "generate_wbs"}
    assert _forced_specific_tool(
        [{"role": "user", "content": "how many RFIs are open and which ones are overdue?"}],
        avail,
    ) is None
    assert _forced_specific_tool(
        [{"role": "user", "content": "what is an RFI?"}],
        avail,
    ) is None


def test_named_calculator_still_forces_construction_calc():
    avail = {"rfi_generator", "construction_calc"}
    assert _forced_specific_tool(
        [{"role": "user", "content": "run a dewatering uplift check for 23 m water depth"}],
        avail,
    ) == "construction_calc"
    assert _forced_specific_tool(
        [{"role": "user", "content": "calculate rebar weight for 12 m of 16 mm bar"}],
        avail,
    ) == "construction_calc"


def test_empty_fixture_rfi_draft_is_not_an_unindexed_refusal():
    """Empty FIXTURE projects must not early-return the unindexed refusal."""
    from app.agents.runtime import _project_has_non_rag_context

    ask = (
        "draft an RFI asking the engineer to clarify the "
        "rebar detail at the transfer beam"
    )
    assert _project_has_non_rag_context("empty-fixture", ask) is True
    assert _project_has_non_rag_context("FIXTURE-Programme", ask) is True


def test_rfi_ask_does_not_predispatch_construction_calc(monkeypatch):
    """'rebar lap' is a named calculator; an RFI draft must not steal it."""
    from app.agents import runtime

    called = []

    async def boom(*_a, **_k):
        called.append(1)
        raise AssertionError("construction_calc must not predispatch on an RFI draft")

    agent = _agent(["construction"])
    monkeypatch.setattr(agent, "_run_tool_call", boom)
    rec = _run(runtime._predispatch_formula_calc(
        agent,
        [{
            "role": "user",
            "content": "use rfi_generator for the missing rebar lap detail",
        }],
        "empty-fixture",
    ))
    assert rec is None
    assert called == []


# ── Dispatch ────────────────────────────────────────────────────────────────


def test_synthetic_tool_hits_the_container(monkeypatch):
    calls = {}

    class FakeBlock:
        async def rfi_generator(self, input_data, params):
            calls["input"] = input_data
            calls["params"] = params
            return {
                "status": "success",
                "action": "rfi_generator",
                "total_rfis": 1,
                "rfis": [{
                    "rfi_number": "RFI-0001",
                    "subject": "Rebar detail — transfer beam",
                    "question": "Please clarify the rebar detail.",
                }],
            }

    import app.dependencies as deps
    monkeypatch.setattr(deps, "get_block_instance", lambda name: FakeBlock())
    r = _call(_agent(["construction"]), "rfi_generator", {
        "message": "draft an RFI asking the engineer to clarify the rebar detail",
        "drawing_ref": "S-201",
    })
    assert r["ok"], r
    assert r["name"] == "rfi_generator"
    assert "rebar" in (calls["input"].get("message") or "").lower()
    assert calls["params"].get("drawing_ref") == "S-201"


def test_synthetic_tool_refuses_without_construction_block():
    r = _call(_agent([]), "rfi_generator", {})
    assert r["ok"] is False
    assert "allowed_blocks" in (r["result"].get("error") or "").lower()
