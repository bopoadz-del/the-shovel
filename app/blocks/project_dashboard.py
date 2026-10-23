"""Project Dashboard Block — Phase 5 of the Construction Management Engine.

Aggregates data from all construction management domains via the unified
activity model and dependency graph. Produces a unified project health view
with cross-domain impact detection and suggested actions.

Usage:
    POST /execute { "block": "project_dashboard", "input": {
        "action": "health_check",
        "project_id": "...",
        "domain_status": { "schedule": {"status": "delayed", ...}, ... }
    }}
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.core.dependency_graph import DependencyGraph, Domain, ProjectHealthAggregator
from app.core.universal_base import UniversalBlock
from app.core.workflow_templates import WorkflowTemplateLibrary
from app.schemas.construction_activity import ActivityGraph, ConstructionActivity, RiskLevel, Status


class ProjectDashboardBlock(UniversalBlock):
    """Unified project health dashboard aggregator.

    Consumes per-domain status summaries and construction activities,
    applies cross-domain dependency analysis, and produces a unified
    project health view with recommended actions.
    """

    auto_validate = False
    name = "project_dashboard"
    version = "1.0.0"
    description = (
        "Unified project health dashboard: aggregates cross-domain status, "
        "detects cross-domain impacts, and suggests actions."
    )
    layer = 3
    tags = ["domain", "construction", "dashboard", "reporting", "analytics"]
    requires = []

    default_config = {
        "variance_threshold_days": 7,
        "variance_threshold_percent": 10.0,
        "risk_score_threshold": 60,
    }

    ui_schema = {
        "input": {
            "type": "json",
            "placeholder": "Pass domain_status or activities...",
        },
        "output": {
            "type": "json",
            "fields": [
                {"name": "overall_status", "type": "text", "label": "Overall"},
                {"name": "overall_score", "type": "number", "label": "Score"},
                {"name": "domain_scores", "type": "json", "label": "By Domain"},
                {"name": "cross_domain_flags", "type": "list", "label": "Flags"},
                {"name": "suggested_actions", "type": "list", "label": "Actions"},
            ],
        },
        "quick_actions": [
            {"icon": "📊", "label": "Project Health", "prompt": "Generate project health dashboard"},
            {"icon": "⚠️", "label": "Critical Issues", "prompt": "Show critical cross-domain issues"},
            {"icon": "📋", "label": "Workflows", "prompt": "List available multi-domain workflows"},
        ],
    }

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._dependency_graph: Optional[DependencyGraph] = None
        self._template_library: Optional[WorkflowTemplateLibrary] = None

    def _get_dependency_graph(self) -> DependencyGraph:
        if self._dependency_graph is None:
            self._dependency_graph = DependencyGraph()
        return self._dependency_graph

    def _get_template_library(self) -> WorkflowTemplateLibrary:
        if self._template_library is None:
            self._template_library = WorkflowTemplateLibrary()
        return self._template_library

    # ── Public operations ──────────────────────────────────────────────────

    async def process(self, input_data: Any, params: Dict = None) -> Dict:
        params = params or {}
        data = input_data if isinstance(input_data, dict) else {}
        action = data.get("action") or params.get("action") or "health_check"

        handlers = {
            "health_check": self._health_check,
            "activity_graph": self._activity_graph_analysis,
            "list_workflows": self._list_workflows,
            "workflow_detail": self._workflow_detail,
            "run_workflow": self._run_workflow,
        }
        handler = handlers.get(action)
        if handler is None:
            return {
                "status": "error",
                "error": f"Unknown action: {action}",
                "known_actions": list(handlers.keys()),
            }
        return await handler(data, params)

    # ── Action handlers ────────────────────────────────────────────────────

    async def _health_check(self, data: Dict, params: Dict) -> Dict:
        """Aggregate domain status into unified health view."""
        domain_status = data.get("domain_status") or params.get("domain_status", {})
        if not domain_status:
            return {
                "status": "success",
                "overall_status": "unknown",
                "overall_score": 0,
                "domain_scores": {},
                "cross_domain_flags": [],
                "suggested_actions": [],
                "note": "No domain status provided — pass domain_status with per-domain entries",
            }

        # Normalize domain_status keys to Domain enum
        normalized: Dict[Domain, Dict[str, Any]] = {}
        for key, value in domain_status.items():
            try:
                domain = Domain(key)
                normalized[domain] = value
            except ValueError:
                continue

        aggregator = ProjectHealthAggregator(self._get_dependency_graph())
        health = aggregator.aggregate(normalized)

        return {"status": "success", **health}

    async def _activity_graph_analysis(self, data: Dict, params: Dict) -> Dict:
        """Analyze a full ActivityGraph for cross-domain health."""
        activities_raw = data.get("activities") or params.get("activities", [])
        project_id = data.get("project_id") or params.get("project_id", "")
        project_name = data.get("project_name") or params.get("project_name", "")

        graph = ActivityGraph(project_id=project_id, project_name=project_name)
        for raw in activities_raw:
            try:
                activity = ConstructionActivity.model_validate(raw)
                graph.add_activity(activity)
            except Exception:
                continue

        summary = graph.summary()
        blocked = graph.blocked_activities()
        overdue = graph.overdue_activities()
        high_risk = graph.high_risk_activities()
        critical = graph.critical_path_activities()

        # Derive domain status from activities.
        # Quality blocks → "failed" (R002 NCR/hold), not "critical" (R003 safety).
        domain_status: Dict[str, Dict[str, Any]] = {}
        for activity in graph.activities:
            domain_key = activity.type.value
            if domain_key not in domain_status:
                domain_status[domain_key] = {
                    "status": "healthy",
                    "count": 0,
                    "blocked": 0,
                    "overdue": 0,
                    "high_risk": 0,
                }
            domain_status[domain_key]["count"] += 1
            if activity.is_blocked():
                domain_status[domain_key]["blocked"] += 1
                if domain_key == Domain.QUALITY.value:
                    domain_status[domain_key]["status"] = "failed"
                elif domain_key == Domain.SAFETY.value:
                    domain_status[domain_key]["status"] = "critical"
                else:
                    # Non-QA/safety holds still surface as failed for R002-style holds
                    domain_status[domain_key]["status"] = "failed"
            if activity.status == Status.OVERDUE:
                domain_status[domain_key]["overdue"] += 1
                if domain_status[domain_key]["status"] == "healthy":
                    domain_status[domain_key]["status"] = "warning"
            if activity.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
                domain_status[domain_key]["high_risk"] += 1

        # Run health aggregation on derived domain status
        normalized = {}
        for key, value in domain_status.items():
            try:
                normalized[Domain(key)] = value
            except ValueError:
                continue

        aggregator = ProjectHealthAggregator(self._get_dependency_graph())
        health = aggregator.aggregate(normalized)

        return {
            "status": "success",
            "project_id": project_id,
            "project_name": project_name,
            "activity_summary": summary,
            "health": health,
            "blocked_activities": [
                {"id": a.id, "name": a.name, "type": a.type.value, "reason": "quality/safety gate"}
                for a in blocked[:20]
            ],
            "overdue_activities": [
                {"id": a.id, "name": a.name, "days_overdue": a.days_overdue()}
                for a in overdue[:20]
            ],
            "high_risk_activities": [
                {"id": a.id, "name": a.name, "risk_level": a.risk_level.value}
                for a in high_risk[:20]
            ],
            "critical_path": [
                {"id": a.id, "name": a.name, "duration": a.schedule.duration_days}
                for a in critical[:20]
            ],
        }

    async def _list_workflows(self, data: Dict, params: Dict) -> Dict:
        """List all available multi-domain workflow templates."""
        lib = self._get_template_library()
        return {
            "status": "success",
            "templates": lib.list_templates(),
            "total": len(lib.all_template_ids()),
        }

    async def _workflow_detail(self, data: Dict, params: Dict) -> Dict:
        """Get detail for one workflow template including sample plan."""
        template_id = data.get("template_id") or params.get("template_id", "")
        lib = self._get_template_library()
        template = lib.get(template_id)
        if template is None:
            return {
                "status": "error",
                "error": f"Unknown template: {template_id}",
                "available": lib.all_template_ids(),
            }
        plan = template.build_plan(params)
        from app.core.cross_domain_reasoner import MultiDomainPlanBuilder

        executable = MultiDomainPlanBuilder(lib).build_from_template(template_id, params)
        return {
            "status": "success",
            "template": {
                "id": template.template_id,
                "name": template.name,
                "description": template.description,
                "domains": template.domains,
            },
            "sample_plan": plan.model_dump(mode="json"),
            "executable_plan": self._label_executable_plan(executable),
        }

    @staticmethod
    def _label_executable_plan(plan: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Annotate a built plan — plan_only labels, not fake execution."""
        if not plan:
            return plan
        labeled = dict(plan)
        labeled["execution_mode"] = "plan_only"
        labeled["execution_note"] = (
            "Aliased steps for dispatch — this endpoint builds the plan only; "
            "it does not execute construction actions or fabricate deliverables."
        )
        steps_out = []
        for step in plan.get("steps") or []:
            s = dict(step)
            if step.get("dispatch") is False or step.get("needs_caller_render"):
                s["execution_label"] = "caller_render"
            elif step.get("block"):
                s["execution_label"] = "dispatchable"
            else:
                s["execution_label"] = "unmapped"
            steps_out.append(s)
        labeled["steps"] = steps_out
        return labeled

    async def _run_workflow(self, data: Dict, params: Dict) -> Dict:
        """Build an aliased executable plan for a template (dispatch-ready).

        Does not invent a parallel executor — returns steps shaped for
        /v1/execute {block, params.action}. Callers (or chain) run each step.
        """
        template_id = data.get("template_id") or params.get("template_id", "")
        if not template_id:
            return {"status": "error", "error": "template_id required"}
        from app.core.cross_domain_reasoner import MultiDomainPlanBuilder

        plan = MultiDomainPlanBuilder(self._get_template_library()).build_from_template(
            template_id, {**params, **{k: v for k, v in data.items() if k != "action"}}
        )
        if plan is None:
            return {
                "status": "error",
                "error": f"Unknown template: {template_id}",
                "available": self._get_template_library().all_template_ids(),
            }
        labeled = self._label_executable_plan(plan)
        return {"status": "success", **labeled}
