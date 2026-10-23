"""Tests for app/blocks/project_dashboard.py — Project Dashboard Block."""

from __future__ import annotations

from typing import Any, Dict

import pytest

from app.blocks.project_dashboard import ProjectDashboardBlock
from app.core.dependency_graph import Domain


class TestProjectDashboardBlockBasics:
    def test_block_metadata(self):
        block = ProjectDashboardBlock()
        assert block.name == "project_dashboard"
        assert block.version == "1.0.0"
        assert "dashboard" in block.tags
        assert "construction" in block.tags

    def test_block_config(self):
        block = ProjectDashboardBlock()
        assert "variance_threshold_days" in block.default_config
        assert "variance_threshold_percent" in block.default_config


class TestHealthCheckAction:
    async def test_healthy_domains(self):
        block = ProjectDashboardBlock()
        result = await block.process({
            "action": "health_check",
            "domain_status": {
                "schedule": {"status": "healthy", "progress": 90},
                "cost": {"status": "healthy", "variance": 3},
                "quality": {"status": "healthy", "pass_rate": 95},
            }
        })
        assert result["status"] == "success"
        assert result["overall_status"] == "healthy"
        assert result["overall_score"] >= 80
        assert len(result["cross_domain_flags"]) == 0

    async def test_critical_schedule(self):
        block = ProjectDashboardBlock()
        result = await block.process({
            "action": "health_check",
            "domain_status": {
                "schedule": {"status": "delayed"},
            }
        })
        assert result["status"] == "success"
        assert len(result["cross_domain_flags"]) >= 1
        flag = result["cross_domain_flags"][0]
        assert flag["source"] == "schedule"

    async def test_empty_domain_status(self):
        block = ProjectDashboardBlock()
        result = await block.process({"action": "health_check", "domain_status": {}})
        assert result["status"] == "success"
        assert result["overall_status"] == "unknown"

    async def test_no_domain_status(self):
        block = ProjectDashboardBlock()
        result = await block.process({"action": "health_check"})
        assert result["status"] == "success"
        assert "note" in result


class TestActivityGraphAction:
    async def test_basic_activity_graph(self):
        block = ProjectDashboardBlock()
        result = await block.process({
            "action": "activity_graph",
            "project_id": "P-001",
            "project_name": "Test",
            "activities": [
                {
                    "id": "ACT-001",
                    "name": "Foundation",
                    "type": "schedule",
                    "status": "in_progress",
                    "schedule": {"duration_days": 10, "is_critical": True},
                },
                {
                    "id": "ACT-002",
                    "name": "Steel Order",
                    "type": "procurement",
                    "status": "in_progress",
                    "procurement_links": [{"item_id": "P-001", "is_long_lead": True}],
                },
            ]
        })
        assert result["status"] == "success"
        assert result["project_id"] == "P-001"
        assert result["activity_summary"]["total_activities"] == 2
        assert result["activity_summary"]["by_type"]["schedule"] == 1
        assert result["activity_summary"]["by_type"]["procurement"] == 1

    async def test_blocked_activities(self):
        block = ProjectDashboardBlock()
        result = await block.process({
            "action": "activity_graph",
            "activities": [
                {
                    "id": "BLK-001",
                    "name": "Blocked",
                    "type": "schedule",
                    "quality_gates": [{"gate_type": "inspection", "result": "fail", "status": "blocked"}],
                }
            ]
        })
        assert result["status"] == "success"
        assert len(result["blocked_activities"]) == 1
        assert result["blocked_activities"][0]["id"] == "BLK-001"

    async def test_overdue_activities(self):
        from datetime import date, timedelta
        past = (date.today() - timedelta(days=10)).isoformat()
        block = ProjectDashboardBlock()
        result = await block.process({
            "action": "activity_graph",
            "activities": [
                {
                    "id": "OVD-001",
                    "name": "Overdue",
                    "type": "schedule",
                    "status": "overdue",
                    "schedule": {"finish_date": past},
                }
            ]
        })
        assert result["status"] == "success"
        assert len(result["overdue_activities"]) == 1

    async def test_high_risk_activities(self):
        block = ProjectDashboardBlock()
        result = await block.process({
            "action": "activity_graph",
            "activities": [
                {
                    "id": "HR-001",
                    "name": "High Risk",
                    "type": "risk",
                    "risk_level": "critical",
                }
            ]
        })
        assert result["status"] == "success"
        assert len(result["high_risk_activities"]) == 1

    async def test_critical_path(self):
        block = ProjectDashboardBlock()
        result = await block.process({
            "action": "activity_graph",
            "activities": [
                {
                    "id": "CP-001",
                    "type": "schedule",
                    "schedule": {"is_critical": True, "duration_days": 5},
                }
            ]
        })
        assert result["status"] == "success"
        assert len(result["critical_path"]) == 1
        assert result["critical_path"][0]["id"] == "CP-001"


class TestWorkflowActions:
    async def test_list_workflows(self):
        block = ProjectDashboardBlock()
        result = await block.process({"action": "list_workflows"})
        assert result["status"] == "success"
        assert result["total"] == 10
        assert len(result["templates"]) == 10

    async def test_workflow_detail(self):
        block = ProjectDashboardBlock()
        result = await block.process({
            "action": "workflow_detail",
            "template_id": "new_project_setup",
        })
        assert result["status"] == "success"
        assert result["template"]["id"] == "new_project_setup"
        assert "sample_plan" in result
        plan = result.get("executable_plan") or {}
        assert plan.get("execution_mode") == "plan_only"

    async def test_workflow_detail_unknown(self):
        block = ProjectDashboardBlock()
        result = await block.process({
            "action": "workflow_detail",
            "template_id": "nonexistent",
        })
        assert result["status"] == "error"
        assert "available" in result

class TestUnknownAction:
    async def test_unknown_action(self):
        block = ProjectDashboardBlock()
        result = await block.process({"action": "destroy_everything"})
        assert result["status"] == "error"
        assert "known_actions" in result
        assert "health_check" in result["known_actions"]
        assert "run_workflow" in result["known_actions"]
