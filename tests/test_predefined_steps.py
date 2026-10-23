"""Predefined-reasoning plan steps (the schedule workflow spine): extract ->
build_wbs -> cost_load -> render_artifact, run by PlanExecutor against a
ProjectSession. cost_load STAGES the workbook; render_artifact MATERIALIZES it
— so a plan without render_artifact answers inline (the intent gate)."""
from __future__ import annotations

import asyncio
import os

from app.schemas.execution_plan import ExecutionPlan, PlanStep
from app.schemas.project_session import ProjectSession
from app.core.plan_executor import PlanExecutor


def _run(plan, session):
    return asyncio.run(PlanExecutor().run(plan, session))


def test_cost_load_stages_without_materializing():
    s = ProjectSession.new("s1")
    res = _run(ExecutionPlan(steps=[
        PlanStep(type="build_wbs", args={"project_type": "data_center", "target_count": 60, "start_date": "2026-01-05"}),
        PlanStep(type="cost_load", args={"project_name": "DC"}),
    ]), s)
    assert res.status == "success", res
    assert s.data["wbs"]["actual_count"] > 0
    assert s.data["schedule_summary"]["duration_days"] > 0
    assert s.data.get("schedule_workbook_path")   # built + staged
    assert s.artifacts == []                        # but NOT exposed


def test_render_artifact_materializes_the_staged_workbook():
    s = ProjectSession.new("s2")
    res = _run(ExecutionPlan(steps=[
        PlanStep(type="build_wbs", args={"project_type": "data_center", "target_count": 60}),
        PlanStep(type="cost_load", args={}),
        PlanStep(type="render_artifact", args={"name": "Schedule.xlsx"}),
    ]), s)
    assert res.status == "success", res
    assert len(s.artifacts) == 1 and s.artifacts[0].type == "excel"
    assert os.path.exists(s.artifacts[0].path)
    os.remove(s.artifacts[0].path)


def test_extract_feeds_lead_times_into_wbs(monkeypatch):
    async def fake_feed(doc_ids):
        return ([{"equipment": "Chiller", "lead_time_days": 150}],
                [{"name": "RFS", "target_date": "2027-12-01"}])
    monkeypatch.setattr("app.lib.schedule_feed.extract_schedule_feed", fake_feed)
    s = ProjectSession.new("s3")
    res = _run(ExecutionPlan(steps=[
        PlanStep(type="extract_document", args={"document_ids": ["d1"]}),
        PlanStep(type="build_wbs", args={"project_type": "data_center", "target_count": 60}),
        PlanStep(type="cost_load", args={}),
    ]), s)
    assert res.status == "success", res
    assert s.data["procurement_lead_times"] == [{"equipment": "Chiller", "lead_time_days": 150}]
    assert s.data["wbs"]["procurement_injected"] >= 1
    assert s.data["schedule_summary"]["target_milestones"]
    if s.data.get("schedule_workbook_path"):
        os.remove(s.data["schedule_workbook_path"])


def test_cost_load_without_wbs_errors():
    s = ProjectSession.new("s4")
    res = _run(ExecutionPlan(steps=[PlanStep(type="cost_load", args={})]), s)
    assert res.status == "error"
    assert "build_wbs" in res.step_results[0].error
