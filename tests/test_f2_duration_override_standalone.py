"""OLD-pack F2 on a fresh thread must apply the 45-day tree-removal override.

Live Master Corpus, verbatim (prefix is the pack's grounding line)::

    Answer only from the client project documents. Use 45 days for the
    tree-removal activity and re-run.

Expected: ``duration_override_rerun`` + generate_wbs with 45 days on
tree-removal / site-clearance rows.

Observed FAIL: the ask is not generate/create/build, so `_apply_rag_context`
classified it as a lookup. The model obeyed "answer ONLY from the reference
context", said excerpts lacked activity durations, and offered to fetch a
WBS. Fresh thread — no prior F1 WBS in history. tool_json_leak false.

This is the same class as the calculation grounding defect: the model was
obeying. #500 / D4 already pins that 'tree removal' matches 'Remove trees'
and 'Site clearance'. Do not re-open that matcher here.
"""
from __future__ import annotations

import json

import pytest

from app.agents import runtime as runtime_module
from app.agents.runtime import (
    _apply_rag_context,
    _forced_specific_tool,
    _messages_user_and_history,
    _unwrap_rag_folded_operator_text,
)
from app.containers.construction import ConstructionContainer
from app.core.contract_lookup_intent import message_is_contract_data_lookup
from app.lib.wbs_duration_overrides import (
    message_wants_wbs_duration_rerun,
    parse_duration_overrides,
)

LIVE_PREFIX = "Answer only from the client project documents. "
F2_ASK = "Use 45 days for the tree-removal activity and re-run."
LIVE_F2 = LIVE_PREFIX + F2_ASK
# Typical Master Corpus lookup prose that used to steal the folded turn.
FOLDED_CORPUS = (
    "REFERENCE CONTEXT\n"
    "Time for Completion is 852 days. Schedule 10 of the contract is "
    "Not Used. Delay Damages are 0.1% of the Contract Price per day.\n"
)

REFUSAL = "using ONLY the reference context"


def _apply(question: str) -> str:
    msgs = [{"role": "user", "content": question}]
    assert _apply_rag_context(msgs, {"content": FOLDED_CORPUS}) is True
    return msgs[-1]["content"]


@pytest.mark.parametrize("ask", [F2_ASK, LIVE_F2])
def test_live_f2_parses_to_45_day_tree_removal_rerun(ask):
    parsed = parse_duration_overrides(ask)
    assert parsed, ask
    assert parsed[0]["days"] == 45
    assert parsed[0]["match"] == "tree removal"
    assert message_wants_wbs_duration_rerun(ask) is True


def test_live_prefixed_f2_is_not_told_to_answer_only_from_excerpts():
    """The load-bearing assertion — this instruction caused the refusal."""
    out = _apply(LIVE_F2)
    assert REFUSAL not in out, out[-600:]
    assert "DURATION OVERRIDE REQUEST:" in out
    assert "Do NOT refuse" in out
    assert LIVE_F2 in out


def test_bare_f2_also_takes_the_override_directive():
    out = _apply(F2_ASK)
    assert REFUSAL not in out
    assert "DURATION OVERRIDE REQUEST:" in out


def test_lookup_and_generate_directives_are_unchanged():
    lookup = _apply("What is the Time for Completion for the whole of the Works?")
    generate = _apply("Generate a method statement for the raft pour")
    override = _apply(LIVE_F2)

    assert REFUSAL in lookup
    assert "REQUEST TO GENERATE" in generate
    assert "DURATION OVERRIDE REQUEST:" in override
    assert override[len(FOLDED_CORPUS):] != lookup[len(FOLDED_CORPUS):]
    assert override[len(FOLDED_CORPUS):] != generate[len(FOLDED_CORPUS):]


def test_unwrap_recovers_operator_text_from_a_folded_f2_turn():
    folded = _apply(LIVE_F2)
    assert _unwrap_rag_folded_operator_text(folded) == LIVE_F2
    user_msg, history = _messages_user_and_history(
        [{"role": "user", "content": folded}]
    )
    assert user_msg == LIVE_F2
    assert history == []


@pytest.mark.asyncio
async def test_select_agent_marks_standalone_prefixed_f2_as_override_rerun(
    monkeypatch,
):
    class _Block:
        async def process(self, _data):
            return {"matched_actions": []}

    monkeypatch.setattr(runtime_module, "_get_smart_orchestrator_block", lambda: _Block())
    monkeypatch.setattr(runtime_module, "_routing_disabled", lambda: False)

    for name in ("construction-pm", "project-assistant"):
        agent = runtime_module.Agent(
            name=name,
            description="t",
            system_prompt="(test)",
            allowed_blocks=["construction"],
        )
        runtime_module.AGENT_REGISTRY[name] = agent
        runtime_module.AGENT_REGISTRY["heavy-reasoning"] = runtime_module.Agent(
            name="heavy-reasoning",
            description="h",
            system_prompt="(test)",
            allowed_blocks=["construction"],
        )
        final, info = await runtime_module.select_agent_for_message(LIVE_F2, agent)
        assert info["reason"] == "duration_override_rerun", (name, info)
        assert info["action"] == "generate_wbs"
        assert final.name in {name, "heavy-reasoning"}


def test_forced_tool_is_generate_wbs_on_construction_pm_and_after_rag_fold():
    available = {"generate_wbs", "construction_calc", "search_project_documents"}
    bare = [{"role": "user", "content": LIVE_F2}]
    assert _forced_specific_tool(bare, available) == "generate_wbs"

    folded = [{"role": "user", "content": _apply(LIVE_F2)}]
    assert _forced_specific_tool(folded, available) == "generate_wbs"


def test_named_calculator_does_not_steal_live_f2():
    assert runtime_module._message_wants_named_calculator(LIVE_F2) is False
    assert message_is_contract_data_lookup(LIVE_F2) is False


def test_orchestrator_scores_live_prefixed_f2_as_generate_wbs():
    import asyncio
    from app.blocks.smart_orchestrator import SmartOrchestratorBlock

    result = asyncio.run(
        SmartOrchestratorBlock().process({"user_message": LIVE_F2})
    )
    matched = result.get("matched_actions") or []
    actions = [m["action"] for m in matched]
    assert "generate_wbs" in actions
    gw = next(m for m in matched if m["action"] == "generate_wbs")
    assert gw["confidence"] >= 0.85


@pytest.mark.asyncio
async def test_predispatch_standalone_f2_applies_45_day_tree_removal(monkeypatch):
    """Fresh thread: no F1 WBS in history. Predispatch must still fire."""
    captured = {}

    async def fake_wbs(_data, params):
        captured["params"] = params
        return {
            "status": "success",
            "summary": {"total_duration_days": 90},
            "activities": [
                {"id": "1", "name": "Site clearance — [Zone 1]", "duration_days": 45}
            ],
            "duration_overrides_applied": [
                {"match": "tree removal", "days": 45, "activities_updated": 1}
            ],
        }

    class _C:
        generate_wbs = staticmethod(fake_wbs)

    monkeypatch.setattr(
        "app.dependencies.get_block_instance", lambda _n: _C()
    )
    agent = runtime_module.Agent(
        name="construction-pm",
        description="t",
        system_prompt="(test)",
        allowed_blocks=["construction"],
    )
    msgs = [{"role": "user", "content": LIVE_F2}]
    rec = await runtime_module._predispatch_wbs_duration_override(agent, msgs)
    assert rec is not None
    assert rec["name"] == "generate_wbs"
    assert rec["predispatched"] is True
    ovr = captured["params"]["duration_overrides"][0]
    assert ovr["days"] == 45
    assert ovr["match"] == "tree removal"
    assert "45" in json.dumps(rec["result"])
    assert "PLATFORM PRE-DISPATCH" in msgs[-1]["content"]


@pytest.mark.asyncio
async def test_predispatch_still_fires_when_the_user_bubble_was_rag_folded(
    monkeypatch,
):
    captured = {}

    async def fake_wbs(_data, params):
        captured["params"] = params
        return {
            "status": "success",
            "summary": {"total_duration_days": 90},
            "activities": [],
            "duration_overrides_applied": [
                {"match": "tree removal", "days": 45, "activities_updated": 1}
            ],
        }

    class _C:
        generate_wbs = staticmethod(fake_wbs)

    monkeypatch.setattr(
        "app.dependencies.get_block_instance", lambda _n: _C()
    )
    agent = runtime_module.Agent(
        name="construction-pm",
        description="t",
        system_prompt="(test)",
        allowed_blocks=["construction"],
    )
    folded = _apply(LIVE_F2)
    msgs = [{"role": "user", "content": folded}]
    rec = await runtime_module._predispatch_wbs_duration_override(agent, msgs)
    assert rec is not None
    ovr = captured["params"]["duration_overrides"][0]
    assert ovr["days"] == 45
    assert ovr["match"] == "tree removal"
    # Brief must be the operator ask, not the Contract Data excerpt.
    assert "852 days" not in (captured["params"].get("brief") or "")
    assert "tree-removal" in (captured["params"].get("user_message") or "").lower()


@pytest.mark.asyncio
async def test_generate_wbs_standalone_f2_applies_45_days_to_site_clearance():
    """Building template already has 'Site clearance'; D4 alias maps
    tree-removal onto it. Do not invent a new activity row."""
    container = ConstructionContainer()
    result = await container.generate_wbs(
        {},
        {
            "target_count": 40,
            "user_message": LIVE_F2,
            "brief": LIVE_F2,
        },
    )
    assert result.get("status") == "success"
    applied = result.get("duration_overrides_applied") or []
    assert applied, result.get("duration_overrides_unmatched")
    assert applied[0]["days"] == 45
    assert applied[0]["match"] == "tree removal"
    assert applied[0]["activities_updated"] >= 1
    cleared = [
        a for a in result["activities"]
        if "site clearance" in a["name"].lower()
    ]
    assert cleared
    assert all(a["duration_days"] == 45 for a in cleared)
