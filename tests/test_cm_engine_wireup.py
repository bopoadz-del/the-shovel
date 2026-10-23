"""E2E: workflow template → aliased actions → ConstructionContainer.route."""
from __future__ import annotations

import pytest

from app.core.cm_step_aliases import resolve_action
from app.core.cross_domain_reasoner import MultiDomainPlanBuilder


@pytest.mark.asyncio
async def test_template_steps_dispatch_via_construction_route():
    from app.containers.construction import ConstructionContainer

    plan = MultiDomainPlanBuilder().build_from_template("change_order_impact")
    assert plan is not None
    container = ConstructionContainer()

    construction_steps = [
        s for s in plan["steps"]
        if s.get("dispatch", True) and s["block"] == "construction"
    ]
    assert len(construction_steps) >= 3

    for step in construction_steps[:4]:
        action = step["params"]["action"]
        raw = step.get("step_type")
        if raw:
            assert resolve_action(raw) == action or action == resolve_action(raw)
        result = await container.route(action, {}, {"action": action})
        assert result.get("error", "").startswith("Unknown action") is False, (
            f"step {step.get('step_type')} → {action}: {result}"
        )
        assert "known_actions" not in result or result.get("status") != "error" or (
            "Unknown action" not in str(result.get("error", ""))
        )


@pytest.mark.asyncio
async def test_new_project_setup_template_end_to_end_with_aliases():
    """Full template: every dispatchable construction step routes; render is NO_AUTO."""
    from app.containers.construction import ConstructionContainer
    from app.core.plan_executor import PlanExecutor
    from app.schemas.project_session import ProjectSession

    plan = MultiDomainPlanBuilder().build_from_template(
        "new_project_setup", {"project_type": "data_center"}
    )
    assert plan is not None
    assert plan["template_id"] == "new_project_setup"

    dispatchable = [s for s in plan["steps"] if s.get("dispatch", True)]
    caller_render = [s for s in plan["steps"] if s.get("needs_caller_render")]
    assert dispatchable, "expected executable steps"
    assert caller_render, "expected render/deliver steps"

    container = ConstructionContainer()
    for step in dispatchable:
        if step["block"] != "construction":
            continue
        action = step["params"]["action"]
        result = await container.route(action, {}, {"action": action})
        assert "Unknown action" not in str(result.get("error", "")), (
            f"{step.get('step_type')} → {action}: {result}"
        )

    session = ProjectSession.new("cm-e2e-test", user_id="test")
    pe = PlanExecutor()
    out = await pe.run_cm_plan_steps(caller_render, session)
    statuses = {r["step_type"]: r["status"] for r in out["step_results"]}
    if "deliver" in statuses:
        assert statuses["deliver"] == "success"
    if "render_artifact" in statuses:
        assert statuses["render_artifact"] in ("error", "needs_caller_render", "success")
    # Aggregate mirrors PlanExecutor.run(): error steps → partial/error, not blanket success.
    step_statuses = [r["status"] for r in out["step_results"]]
    if any(s == "error" for s in step_statuses) and any(s == "success" for s in step_statuses):
        assert out["status"] == "partial"
    elif any(s == "error" for s in step_statuses):
        assert out["status"] == "error"
    else:
        assert out["status"] == "success"


@pytest.mark.asyncio
async def test_route_accepts_template_vocabulary_directly():
    from app.containers.construction import ConstructionContainer

    container = ConstructionContainer()
    result = await container.route(
        "payment_cert", {}, {"action": "payment_cert"}
    )
    assert "Unknown action" not in str(result.get("error", ""))


@pytest.mark.asyncio
async def test_project_dashboard_run_workflow():
    from app.blocks.project_dashboard import ProjectDashboardBlock

    block = ProjectDashboardBlock()
    result = await block.process({
        "action": "run_workflow",
        "template_id": "new_project_setup",
        "project_type": "data_center",
    })
    assert result["status"] == "success"
    assert result["template_id"] == "new_project_setup"
    assert len(result["steps"]) > 0
    construction_actions = [
        s["params"]["action"]
        for s in result["steps"]
        if s.get("dispatch", True) and s["block"] == "construction"
    ]
    assert "generate_wbs" in construction_actions
    assert "build_wbs" not in construction_actions
    assert "health_check" not in construction_actions
    render_steps = [s for s in result["steps"] if s.get("step_type") == "render_artifact"]
    assert render_steps
    assert all(s.get("dispatch") is False for s in render_steps)
    assert all(s.get("params", {}).get("action") != "health_check" for s in render_steps)


@pytest.mark.asyncio
async def test_run_cm_plan_steps_aggregate_status_on_errors():
    """Top-level status must reflect step errors (not always success)."""
    from app.core.plan_executor import PlanExecutor
    from app.schemas.project_session import ProjectSession

    session = ProjectSession.new("cm-agg-status", user_id="test")
    pe = PlanExecutor()
    # render_artifact without prior cost_load → error; deliver still succeeds
    out = await pe.run_cm_plan_steps(
        [
            {
                "step_type": "render_artifact",
                "needs_caller_render": True,
                "params": {"name": "Schedule.xlsx"},
            },
            {
                "step_type": "deliver",
                "needs_caller_render": True,
                "dispatch": False,
                "params": {},
            },
        ],
        session,
    )
    assert out["status"] == "partial"
    by_type = {r["step_type"]: r["status"] for r in out["step_results"]}
    assert by_type["render_artifact"] == "error"
    assert by_type["deliver"] == "success"

    all_err = await pe.run_cm_plan_steps(
        [
            {
                "step_type": "render_artifact",
                "needs_caller_render": True,
                "params": {},
            },
        ],
        session,
    )
    assert all_err["status"] == "error"
