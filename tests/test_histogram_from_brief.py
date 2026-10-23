"""resource_histogram from a brief (golden pilot_manpower_histogram repro).

"produce a manpower histogram for the structure works over 12 months" used
to hit the bare container action, which honestly refuses without a .xer —
correct for the schedule-file contract, but the router had already
intercepted the turn, so the user got an error instead of a histogram.

The action now has a registry plan: build a norms-based WBS from the brief,
time-phase its crews, and deliver with explicit provenance ("planning
estimates, not site resource returns"). The .xer path and its
no-fabrication contract are untouched.
"""

from __future__ import annotations

import pytest

from tests.conftest import requires_construction_kit


class TestHistogramStepBridging:
    """Kit-independent: the plan step must bridge WBS-shaped activities
    (generate_wbs output keys) into the CPM schema and time-phase them.
    Pure-library path — no container, no network."""

    @pytest.mark.asyncio
    async def test_wbs_shaped_activities_are_bridged(self):
        from app.core.plan_executor import PlanExecutor
        from app.schemas.execution_plan import ExecutionPlan, PlanStep
        from app.schemas.project_session import ProjectSession

        session = ProjectSession.new("t-bridge", user_id="t")
        session.data["activities"] = [
            {"id": "1.1", "name": "Excavate", "duration_days": 10,
             "trades": ["civil"], "predecessors": []},
            {"id": "1.2", "name": "Blind", "duration_days": 5,
             "trades": ["civil"], "predecessors": ["1.1"]},
        ]
        plan = ExecutionPlan(understanding="hist", steps=[
            PlanStep(type="resource_histogram", args={"period_unit": "week"},
                     description="time-phase"),
        ])
        run = await PlanExecutor().run(plan, session)
        assert run.status == "success"
        hist = session.data["manpower"]
        assert hist["periods"], "expected at least one period bucket"
        assert hist["peak_total"] > 0

    @pytest.mark.asyncio
    async def test_empty_session_still_errors_honestly(self):
        from app.core.plan_executor import PlanExecutor
        from app.schemas.execution_plan import ExecutionPlan, PlanStep
        from app.schemas.project_session import ProjectSession

        session = ProjectSession.new("t-empty", user_id="t")
        plan = ExecutionPlan(understanding="hist", steps=[
            PlanStep(type="resource_histogram", args={}, description="x"),
        ])
        run = await PlanExecutor().run(plan, session)
        assert run.status != "success"

    def test_deliver_histogram_renders_provenance(self):
        from app.core.predefined_reasoning import deliver_histogram
        from app.schemas.project_session import ProjectSession

        session = ProjectSession.new("t-del", user_id="t")
        session.data["manpower"] = {
            "period_unit": "month",
            "periods": [{"index": 0, "label": "M1", "total": 42.0, "by_trade": {}}],
            "peak_total": 42.0, "peak_period": "M1",
            "by_trade_totals": {}, "total_manhours": 6720.0,
        }
        text = deliver_histogram(session, deliverable=True)
        assert "norms-derived" in text and "M1: 42 workers" in text
        assert "Peak manpower 42" in text

    def test_deliver_histogram_empty_is_honest(self):
        from app.core.predefined_reasoning import deliver_histogram
        from app.schemas.project_session import ProjectSession

        session = ProjectSession.new("t-del2", user_id="t")
        assert "could not build" in deliver_histogram(session, deliverable=False)


@requires_construction_kit
class TestHistogramFromBrief:
    @pytest.mark.asyncio
    async def test_brief_yields_periodised_histogram(self):
        from app.core.predefined_reasoning import run_workflow
        from app.schemas.project_session import ProjectSession

        session = ProjectSession.new("t-hist", user_id="t")
        out = await run_workflow(
            "resource_histogram",
            {
                "message": "produce a manpower histogram for the structure works over 12 months",
                "params": {},
                "project_id": None,
                "document_ids": [],
                "deliverable": True,
            },
            session,
        )
        assert out["handled"] is True
        answer = out["answer"]
        # Provenance is non-negotiable: estimates must say so.
        assert "norms-derived" in answer
        assert "planning estimates" in answer
        # Periodised rows with headcounts.
        assert "workers" in answer
        assert "Peak manpower" in answer

    @pytest.mark.asyncio
    async def test_xer_contract_untouched(self):
        """The direct container action still honestly refuses without a file."""
        from app.containers.construction import ConstructionContainer

        result = await ConstructionContainer().resource_histogram({}, {})
        assert result["status"] == "error"
        assert "schedule_file" in result["error"]
