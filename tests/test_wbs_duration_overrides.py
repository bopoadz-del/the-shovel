"""User duration overrides: parse 'use N days and re-run', apply, recompute CPM."""

from __future__ import annotations

import json

import pytest

from app.lib.wbs_duration_overrides import (
    activity_matches_override,
    apply_duration_overrides,
    collect_overrides,
    message_wants_wbs_duration_rerun,
    parse_duration_overrides,
)
from app.containers.construction import ConstructionContainer
from app.agents import runtime as runtime_module


@pytest.mark.parametrize(
    "text,match,days",
    [
        ("use 6 days per slab and re-run", "slab", 6),
        ("consider 6 working days for each slab pour", "slab pour", 6),
        ("6 days per slab", "slab", 6),
        ("change slab duration to 6 days", "slab", 6),
        ("Regenerate the WBS using 6 days for each slab.", "slab", 6),
        (
            "Use 45 days for the tree-removal activity and re-run.",
            "tree removal",
            45,
        ),
    ],
)
def test_parse_explicit_duration_override(text, match, days):
    parsed = parse_duration_overrides(text)
    assert parsed, text
    assert parsed[0]["match"] == match
    assert parsed[0]["days"] == days


def test_bare_consider_n_uses_history_activity():
    parsed = parse_duration_overrides(
        "consider 6 and re-run",
        history=[{"role": "user", "content": "build a schedule with 7 days per slab"}],
    )
    assert parsed == [{
        "match": "slab",
        "days": 6,
        "source": "user_message+history",
    }]


def test_use_trucks_is_not_a_duration_override():
    assert parse_duration_overrides("use 6 trucks and re-run") == []
    assert message_wants_wbs_duration_rerun("use 6 trucks and re-run") is False


def test_spec_curing_days_without_override_verb_is_ignored():
    assert parse_duration_overrides(
        "the spec requires 7 days curing. re-run the BOQ"
    ) == []


def test_message_wants_rerun_for_use_n_days():
    assert message_wants_wbs_duration_rerun(
        "use 6 days per slab and re-run"
    ) is True
    assert message_wants_wbs_duration_rerun(
        "build a 200 activity schedule using 6 days per slab"
    ) is True


def test_apply_overrides_only_matching_names():
    acts = [
        {"id": "1", "name": "Floor slab concrete pour", "duration_days": 10},
        {"id": "2", "name": "Steel column erection", "duration_days": 21},
    ]
    out, applied, unmatched = apply_duration_overrides(
        acts, [{"match": "slab", "days": 6}]
    )
    assert out[0]["duration_days"] == 6
    assert out[1]["duration_days"] == 21
    assert applied[0]["activities_updated"] == 1
    assert unmatched == []


def test_tree_removal_override_hits_remove_trees_and_site_clearance():
    """UI-PHYS F2 / D4: override parsed as 'tree removal' must not 0-update.

    Live WBS rows were named 'Site clearance — Hall A/B'. A BOQ-shaped
    name 'Remove trees from sidewalks' also failed because 'removal' is
    not a substring of 'remove'. Neither is state loss nor a fuzzy floor.
    """
    assert parse_duration_overrides(
        "Use 45 days for the tree-removal activity and re-run."
    )[0]["match"] == "tree removal"

    assert activity_matches_override(
        "Remove trees from sidewalks", "tree removal"
    )
    assert activity_matches_override(
        "D290.1 Tree removal", "tree removal"
    )
    assert activity_matches_override(
        "Site clearance — Hall A", "tree removal"
    )
    assert activity_matches_override(
        "Site clearance — Hall B", "tree removal"
    )
    assert not activity_matches_override(
        "Steel column erection", "tree removal"
    )

    acts = [
        {"id": "A", "name": "Site clearance — Hall A", "duration_days": 10},
        {"id": "B", "name": "Site clearance — Hall B", "duration_days": 10},
        {"id": "C", "name": "Steel column erection", "duration_days": 21},
    ]
    out, applied, unmatched = apply_duration_overrides(
        acts, [{"match": "tree removal", "days": 45}]
    )
    assert unmatched == []
    assert applied[0]["activities_updated"] == 2
    assert out[0]["duration_days"] == 45
    assert out[1]["duration_days"] == 45
    assert out[2]["duration_days"] == 21


@pytest.mark.asyncio
async def test_generate_wbs_override_changes_slab_days_and_recomputes_cpm():
    container = ConstructionContainer()
    baseline = await container.generate_wbs(
        {},
        {"target_count": 40, "project_type": "building"},
    )
    overridden = await container.generate_wbs(
        {},
        {
            "target_count": 40,
            "project_type": "building",
            "user_message": "use 6 days per slab and re-run",
        },
    )
    base_slabs = [
        a for a in baseline["activities"] if "slab" in a["name"].lower()
    ]
    new_slabs = [
        a for a in overridden["activities"] if "slab" in a["name"].lower()
    ]
    assert base_slabs and new_slabs
    assert any(a["duration_days"] != 6 for a in base_slabs)
    assert all(a["duration_days"] == 6 for a in new_slabs)
    assert overridden.get("duration_overrides_applied")
    assert overridden["duration_overrides_applied"][0]["days"] == 6
    base_total = (baseline.get("summary") or {}).get("total_duration_days")
    new_total = (overridden.get("summary") or {}).get("total_duration_days")
    assert base_total and new_total
    assert new_total != base_total


@pytest.mark.asyncio
async def test_generate_wbs_explicit_param_overrides_brief():
    container = ConstructionContainer()
    result = await container.generate_wbs(
        {},
        {
            "target_count": 30,
            "project_type": "building",
            "brief": "use 9 days per slab",
            "duration_overrides": {"slab": 6},
        },
    )
    slabs = [a for a in result["activities"] if "slab" in a["name"].lower()]
    assert slabs
    assert all(a["duration_days"] == 6 for a in slabs)


def test_orchestrator_scores_duration_rerun_as_generate_wbs():
    import asyncio
    from app.blocks.smart_orchestrator import SmartOrchestratorBlock

    result = asyncio.run(
        SmartOrchestratorBlock().process({
            "user_message": "use 6 days per slab and re-run the schedule",
        })
    )
    matched = result.get("matched_actions") or []
    actions = [m["action"] for m in matched]
    assert "generate_wbs" in actions
    gw = next(m for m in matched if m["action"] == "generate_wbs")
    assert gw["confidence"] >= 0.85


def test_forced_tool_is_generate_wbs_not_calculator():
    msgs = [{
        "role": "user",
        "content": "use 6 days per slab and re-run",
    }]
    assert runtime_module._forced_specific_tool(
        msgs, {"generate_wbs", "construction_calc"}
    ) == "generate_wbs"


@pytest.mark.asyncio
async def test_duration_rerun_is_not_named_calculator(monkeypatch):
    from app.agents.runtime import (
        Agent,
        _message_wants_named_calculator,
        select_agent_for_message,
    )

    text = "use 6 days per slab and re-run"
    assert _message_wants_named_calculator(text) is False

    class _Block:
        async def process(self, _data):
            return {"matched_actions": []}

    monkeypatch.setattr(runtime_module, "_get_smart_orchestrator_block", lambda: _Block())
    monkeypatch.setattr(runtime_module, "_routing_disabled", lambda: False)

    agent = Agent(
        name="project-assistant",
        description="test",
        system_prompt="(test)",
        allowed_blocks=["construction"],
    )
    runtime_module.AGENT_REGISTRY["project-assistant"] = agent
    runtime_module.AGENT_REGISTRY["heavy-reasoning"] = Agent(
        name="heavy-reasoning",
        description="h",
        system_prompt="(test)",
        allowed_blocks=["construction"],
    )
    final, info = await select_agent_for_message(text, agent)
    assert info["action"] == "generate_wbs"
    assert info["reason"] == "duration_override_rerun"
    assert final.name == "heavy-reasoning"


@pytest.mark.asyncio
async def test_predispatch_reruns_generate_wbs_with_override(monkeypatch):
    captured = {}

    async def fake_wbs(_data, params):
        captured["params"] = params
        return {
            "status": "success",
            "summary": {"total_duration_days": 40},
            "activities": [
                {"id": "1", "name": "Floor slab pour", "duration_days": 6}
            ],
            "duration_overrides_applied": [
                {"match": "slab", "days": 6, "activities_updated": 1}
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
    msgs = [
        {"role": "user", "content": "create a schedule with 7 days per slab"},
        {"role": "assistant", "content": "Schedule built with 7-day slabs."},
        {"role": "user", "content": "consider 6 and re-run"},
    ]
    rec = await runtime_module._predispatch_wbs_duration_override(agent, msgs)
    assert rec is not None
    assert rec["name"] == "generate_wbs"
    assert rec["predispatched"] is True
    assert captured["params"]["duration_overrides"][0]["days"] == 6
    assert captured["params"]["duration_overrides"][0]["match"] == "slab"
    assert "PLATFORM PRE-DISPATCH" in msgs[-1]["content"]
    assert "6" in json.dumps(rec["result"])


def test_collect_overrides_merges_explicit_over_parsed():
    merged = collect_overrides(
        "use 9 days per slab",
        explicit={"slab": 6},
    )
    assert merged[0]["days"] == 6
    assert merged[0]["match"] == "slab"
