"""smart_orchestrator post-route CM enrichment (non-competing)."""
from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_smart_orchestrator_attaches_cm_enrichment():
    from app.blocks.smart_orchestrator import SmartOrchestratorBlock

    block = SmartOrchestratorBlock()
    result = await block.process({
        "user_message": "set up a new data center project and generate WBS",
    })
    assert result["status"] == "success"
    assert "action_queue" in result
    assert "cm_follow_up_tools" in result or "cm_matched_template" in result
    if result.get("cm_matched_template"):
        assert result["cm_matched_template"] == "new_project_setup"
        assert result.get("cm_workflow_plan", {}).get("step_count", 0) > 0
        assert result.get("cm_matched_template_score", 0) >= 0.15
    # Runtime contract: queue-append field present when enrichment succeeds.
    assert "cm_queue_appended" in result
    assert isinstance(result["cm_queue_appended"], list)


@pytest.mark.asyncio
async def test_wbs_seed_may_append_follow_ups_to_action_queue():
    """High-confidence WBS + template match promotes follow-ups onto the queue."""
    from app.blocks.smart_orchestrator import ACTION_PATTERNS, SmartOrchestratorBlock

    block = SmartOrchestratorBlock()
    result = await block.process({
        "user_message": "set up a new project and generate WBS schedule",
    })
    assert result["status"] == "success"
    assert "generate_wbs" in result["action_queue"]
    known = {a for a, _ in ACTION_PATTERNS}
    for tool in result.get("cm_queue_appended") or []:
        assert tool in known
        assert tool in result["action_queue"]
    for tool in result.get("cm_follow_up_tools") or []:
        assert isinstance(tool, str) and tool


@pytest.mark.asyncio
async def test_boq_seed_does_not_append_to_action_queue():
    """Only WBS/procurement seeds may grow the queue; BOQ stays metadata-only."""
    from app.blocks.smart_orchestrator import SmartOrchestratorBlock

    block = SmartOrchestratorBlock()
    queue = ["boq_process"]
    meta = block._cross_domain_enrichment(
        "process this bill of quantities and cost sheet",
        queue,
    )
    assert meta.get("cm_follow_up_tools"), meta
    assert meta.get("cm_queue_appended") == []
    assert queue == ["boq_process"]


@pytest.mark.asyncio
async def test_weak_template_match_does_not_publish_workflow_plan():
    """One-word triggers below TemplateMatcher 0.15 must not emit cm_workflow_plan."""
    from app.blocks.smart_orchestrator import SmartOrchestratorBlock

    block = SmartOrchestratorBlock()
    result = await block.process({"user_message": "dashboard"})
    assert result["status"] == "success"
    assert "cm_workflow_plan" not in result
    assert "cm_matched_template" not in result


@pytest.mark.asyncio
async def test_boq_process_seed_normalizes_for_post_tool_followups():
    """ACTION_PATTERNS uses boq_process; reasoner triggers key boq_processor."""
    from app.blocks.smart_orchestrator import SmartOrchestratorBlock

    block = SmartOrchestratorBlock()
    meta = block._cross_domain_enrichment(
        "process this bill of quantities and cost sheet",
        ["boq_process"],
    )
    assert meta.get("cm_follow_up_tools"), meta
    assert any(
        t in meta["cm_follow_up_tools"]
        for t in ("generate_wbs", "parse_primavera_schedule", "progress_tracker",
                  "cash_flow_forecast", "cost_estimate", "boq_processor")
    )
