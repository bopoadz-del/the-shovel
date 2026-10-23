"""Quality status → R002 (FAILED) not R003 (CRITICAL_INCIDENT)."""
from __future__ import annotations

import pytest

from app.core.dependency_graph import (
    Domain,
    ProjectHealthAggregator,
    SuggestedAction,
    TriggerCondition,
    _status_to_trigger,
)


class TestQualityStatusTrigger:
    def test_quality_failed_maps_to_failed_not_incident(self):
        assert _status_to_trigger("failed", domain=Domain.QUALITY) == TriggerCondition.FAILED
        assert _status_to_trigger("critical", domain=Domain.QUALITY) == TriggerCondition.FAILED
        assert _status_to_trigger("blocked", domain=Domain.QUALITY) == TriggerCondition.FAILED

    def test_safety_critical_still_incident(self):
        assert (
            _status_to_trigger("critical", domain=Domain.SAFETY)
            == TriggerCondition.CRITICAL_INCIDENT
        )

    def test_quality_failed_fires_r002_actions(self):
        from app.core.dependency_graph import DependencyGraph

        g = DependencyGraph()
        agg = ProjectHealthAggregator(g)
        health = agg.aggregate({Domain.QUALITY: {"status": "failed"}})
        flags = health["cross_domain_flags"]
        assert flags, "expected cross-domain flags for quality failed"
        assert flags[0]["trigger"] == TriggerCondition.FAILED.value
        assert flags[0]["source"] == "quality"
        actions = set(health["suggested_actions"])
        assert SuggestedAction.ISSUE_NCR.value in actions
        assert SuggestedAction.HOLD_WORK.value in actions
        assert SuggestedAction.STOP_WORK.value not in actions

    def test_quality_critical_status_also_r002(self):
        from app.core.dependency_graph import DependencyGraph

        agg = ProjectHealthAggregator(DependencyGraph())
        health = agg.aggregate({Domain.QUALITY: {"status": "critical"}})
        assert health["cross_domain_flags"][0]["trigger"] == "failed"
        assert SuggestedAction.ISSUE_NCR.value in health["suggested_actions"]


@pytest.mark.asyncio
async def test_dashboard_blocked_quality_uses_failed_status():
    from app.blocks.project_dashboard import ProjectDashboardBlock

    block = ProjectDashboardBlock()
    result = await block.process({
        "action": "activity_graph",
        "activities": [
            {
                "id": "QA-1",
                "name": "Failed pour inspection",
                "type": "quality",
                "quality_gates": [
                    {"gate_type": "inspection", "result": "fail", "status": "blocked"}
                ],
            }
        ],
    })
    assert result["status"] == "success"
    health = result["health"]
    flags = health["cross_domain_flags"]
    assert any(f["source"] == "quality" and f["trigger"] == "failed" for f in flags)
    assert SuggestedAction.ISSUE_NCR.value in health["suggested_actions"]
