"""Tests for app/schemas/construction_activity.py — Unified Activity Model."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from app.schemas.construction_activity import (
    ActivityGraph,
    ActivityType,
    ConstructionActivity,
    CostMeta,
    ProcurementLink,
    QualityGate,
    RiskLevel,
    SafetyGate,
    ScheduleMeta,
    Status,
)


class TestActivityTypeEnum:
    def test_all_types_present(self):
        types = {t.value for t in ActivityType}
        expected = {
            "schedule", "cost", "quality", "safety", "procurement", "contract",
            "risk", "resource", "commissioning", "handover",
            "document", "reporting", "bim",
        }
        assert types == expected

    def test_schedule_is_member(self):
        assert ActivityType.SCHEDULE.value == "schedule"


class TestStatusEnum:
    def test_all_statuses_present(self):
        statuses = {s.value for s in Status}
        expected = {"planned", "in_progress", "complete", "blocked", "overdue", "on_hold"}
        assert statuses == expected


class TestRiskLevelEnum:
    def test_all_levels_present(self):
        levels = {r.value for r in RiskLevel}
        expected = {"low", "medium", "high", "critical"}
        assert levels == expected


class TestConstructionActivity:
    def test_basic_creation(self):
        a = ConstructionActivity(id="ACT-001", name="Concrete Pour L1", type=ActivityType.SCHEDULE)
        assert a.id == "ACT-001"
        assert a.name == "Concrete Pour L1"
        assert a.type == ActivityType.SCHEDULE
        assert a.status == Status.PLANNED
        assert not a.is_blocked()

    def test_validation_from_dict(self):
        raw = {
            "id": "ACT-002",
            "name": "Steel Erection",
            "type": "schedule",
            "status": "in_progress",
            "schedule": {"duration_days": 10, "is_critical": True},
            "cost": {"budgeted": 50000, "currency": "USD"},
        }
        a = ConstructionActivity.model_validate(raw)
        assert a.schedule.duration_days == 10
        assert a.schedule.is_critical is True
        assert a.cost.budgeted == 50000

    def test_is_blocked_by_quality_gate_fail(self):
        a = ConstructionActivity(
            id="ACT-003",
            name="Activity with failed inspection",
            type=ActivityType.SCHEDULE,
            quality_gates=[QualityGate(gate_type="inspection", result="fail", status=Status.BLOCKED)],
        )
        assert a.is_blocked() is True

    def test_is_blocked_by_safety_gate(self):
        a = ConstructionActivity(
            id="ACT-004",
            name="Activity with safety hold",
            type=ActivityType.SCHEDULE,
            safety_gates=[SafetyGate(gate_type="permit", status=Status.BLOCKED)],
        )
        assert a.is_blocked() is True

    def test_is_not_blocked(self):
        a = ConstructionActivity(
            id="ACT-005",
            name="Normal activity",
            type=ActivityType.SCHEDULE,
            quality_gates=[QualityGate(gate_type="inspection", result="pass", status=Status.COMPLETE)],
        )
        assert a.is_blocked() is False

    def test_days_overdue(self):
        past = date.today() - timedelta(days=5)
        a = ConstructionActivity(
            id="ACT-006",
            name="Overdue activity",
            type=ActivityType.SCHEDULE,
            status=Status.OVERDUE,
            schedule=ScheduleMeta(finish_date=past),
        )
        assert a.days_overdue() == 5

    def test_days_overdue_not_overdue(self):
        future = date.today() + timedelta(days=5)
        a = ConstructionActivity(
            id="ACT-007",
            name="On track",
            type=ActivityType.SCHEDULE,
            status=Status.IN_PROGRESS,
            schedule=ScheduleMeta(finish_date=future),
        )
        assert a.days_overdue() == 0

    def test_procurement_links(self):
        a = ConstructionActivity(
            id="ACT-008",
            name="Procurement-linked",
            procurement_links=[
                ProcurementLink(item_id="P-001", item_name="Switchgear", is_long_lead=True, lead_time_days=120),
            ],
        )
        assert len(a.procurement_links) == 1
        assert a.procurement_links[0].is_long_lead is True

    def test_cross_domain_activity(self):
        a = ConstructionActivity(
            id="ACT-009",
            name="Safety audit",
            type=ActivityType.SAFETY,
            risk_level=RiskLevel.HIGH,
            responsible_party="Safety Officer",
            location="Zone A",
        )
        assert a.type == ActivityType.SAFETY
        assert a.risk_level == RiskLevel.HIGH


class TestActivityGraph:
    def test_empty_graph(self):
        g = ActivityGraph()
        assert len(g.activities) == 0
        assert g.summary()["total_activities"] == 0

    def test_add_and_get_activity(self):
        g = ActivityGraph(project_id="P-001", project_name="Test Project")
        a = ConstructionActivity(id="ACT-010", name="Test", type=ActivityType.SCHEDULE)
        g.add_activity(a)
        assert g.get_by_id("ACT-010") == a
        assert len(g.activities) == 1

    def test_no_duplicate_add(self):
        g = ActivityGraph()
        a = ConstructionActivity(id="ACT-011", name="Test")
        g.add_activity(a)
        g.add_activity(a)
        assert len(g.activities) == 1

    def test_link_activities(self):
        g = ActivityGraph()
        a1 = ConstructionActivity(id="ACT-A", name="Upstream")
        a2 = ConstructionActivity(id="ACT-B", name="Downstream")
        g.add_activity(a1)
        g.add_activity(a2)
        g.link_activities("ACT-A", "ACT-B")
        assert "ACT-B" in a1.dependents
        assert "ACT-A" in a2.dependencies

    def test_filter_by_type(self):
        g = ActivityGraph()
        g.add_activity(ConstructionActivity(id="S-001", type=ActivityType.SCHEDULE))
        g.add_activity(ConstructionActivity(id="Q-001", type=ActivityType.QUALITY))
        g.add_activity(ConstructionActivity(id="S-002", type=ActivityType.SCHEDULE))
        assert len(g.by_type(ActivityType.SCHEDULE)) == 2
        assert len(g.by_type(ActivityType.QUALITY)) == 1

    def test_filter_by_status(self):
        g = ActivityGraph()
        g.add_activity(ConstructionActivity(id="A-001", status=Status.COMPLETE))
        g.add_activity(ConstructionActivity(id="A-002", status=Status.OVERDUE))
        assert len(g.by_status(Status.COMPLETE)) == 1
        assert len(g.by_status(Status.OVERDUE)) == 1

    def test_blocked_activities(self):
        g = ActivityGraph()
        g.add_activity(ConstructionActivity(id="OK-001", name="Normal"))
        g.add_activity(
            ConstructionActivity(
                id="BLK-001", name="Blocked",
                quality_gates=[QualityGate(result="fail", status=Status.BLOCKED)],
            )
        )
        blocked = g.blocked_activities()
        assert len(blocked) == 1
        assert blocked[0].id == "BLK-001"

    def test_high_risk_activities(self):
        g = ActivityGraph()
        g.add_activity(ConstructionActivity(id="L-001", risk_level=RiskLevel.LOW))
        g.add_activity(ConstructionActivity(id="H-001", risk_level=RiskLevel.HIGH))
        g.add_activity(ConstructionActivity(id="C-001", risk_level=RiskLevel.CRITICAL))
        assert len(g.high_risk_activities()) == 2

    def test_summary(self):
        g = ActivityGraph(project_id="P-002", project_name="Summary Test")
        g.add_activity(ConstructionActivity(id="S-001", type=ActivityType.SCHEDULE, status=Status.IN_PROGRESS))
        g.add_activity(ConstructionActivity(id="Q-001", type=ActivityType.QUALITY, status=Status.COMPLETE))
        summary = g.summary()
        assert summary["project_id"] == "P-002"
        assert summary["total_activities"] == 2
        assert summary["by_type"]["schedule"] == 1
        assert summary["by_type"]["quality"] == 1

    def test_critical_path_activities(self):
        g = ActivityGraph()
        g.add_activity(
            ConstructionActivity(
                id="CP-001", type=ActivityType.SCHEDULE,
                schedule=ScheduleMeta(is_critical=True),
            )
        )
        g.add_activity(
            ConstructionActivity(
                id="NC-001", type=ActivityType.SCHEDULE,
                schedule=ScheduleMeta(is_critical=False),
            )
        )
        cp = g.critical_path_activities()
        assert len(cp) == 1
        assert cp[0].id == "CP-001"

    def test_cross_domain_linking(self):
        g = ActivityGraph()
        g.add_activity(ConstructionActivity(id="PROC-001", type=ActivityType.PROCUREMENT, name="Switchgear"))
        g.add_activity(ConstructionActivity(id="SCH-001", type=ActivityType.SCHEDULE, name="Install Switchgear"))
        g.link_activities("PROC-001", "SCH-001")
        proc = g.get_by_id("PROC-001")
        sch = g.get_by_id("SCH-001")
        assert "SCH-001" in proc.dependents
        assert "PROC-001" in sch.dependencies
