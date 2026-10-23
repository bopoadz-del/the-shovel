"""Unified Construction Activity Model — Phase 1 of the Construction Management Engine.

Every construction management activity — whether a schedule task, QA inspection,
safety audit, procurement item, contract obligation, risk entry, BIM element,
commissioning test, or handover milestone — is represented as a single
ConstructionActivity node. This enables cross-domain dependency tracking,
multi-domain workflow templates, and unified project health dashboards.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ActivityType(str, Enum):
    """Construction management domain categories.

    Aligned with dependency_graph.Domain where possible. COST is included
    so cost-domain activities are not silently dropped when aggregating.
    """
    SCHEDULE = "schedule"
    COST = "cost"
    QUALITY = "quality"
    SAFETY = "safety"
    PROCUREMENT = "procurement"
    CONTRACT = "contract"
    RISK = "risk"
    RESOURCE = "resource"
    COMMISSIONING = "commissioning"
    HANDOVER = "handover"
    DOCUMENT = "document"
    REPORTING = "reporting"
    BIM = "bim"


class Status(str, Enum):
    """Activity lifecycle states."""
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    BLOCKED = "blocked"
    OVERDUE = "overdue"
    ON_HOLD = "on_hold"


class RiskLevel(str, Enum):
    """Risk severity classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ScheduleMeta(BaseModel):
    """Schedule-domain attributes for an activity."""
    start_date: Optional[date] = None
    finish_date: Optional[date] = None
    duration_days: int = 0
    total_float_days: int = 0
    free_float_days: int = 0
    is_critical: bool = False
    percent_complete: float = 0.0
    wbs_code: str = ""
    predecessors: List[str] = Field(default_factory=list)
    successors: List[str] = Field(default_factory=list)


class CostMeta(BaseModel):
    """Cost-domain attributes for an activity."""
    budgeted: float = 0.0
    actual: float = 0.0
    forecast: float = 0.0
    variance: float = 0.0
    currency: str = "USD"
    cost_account: str = ""  # e.g. BOQ line item reference


class QualityGate(BaseModel):
    """A quality inspection or hold point gating an activity."""
    gate_type: str = ""  # inspection | test | hold_point | sign_off
    description: str = ""
    status: Status = Status.PLANNED
    responsible_party: str = ""  # trade or role
    target_date: Optional[date] = None
    actual_date: Optional[date] = None
    result: str = ""  # pass | fail | pending
    linked_activities: List[str] = Field(default_factory=list)


class SafetyGate(BaseModel):
    """A safety check or permit gating an activity."""
    gate_type: str = ""  # permit | inspection | toolbox_talk | jsa
    description: str = ""
    status: Status = Status.PLANNED
    hazard_type: str = ""
    risk_level: RiskLevel = RiskLevel.LOW
    mitigations: List[str] = Field(default_factory=list)


class ProcurementLink(BaseModel):
    """Link to a procurement item driving or driven by this activity."""
    item_id: str = ""
    item_name: str = ""
    procurement_status: str = ""  # requisition | po_placed | in_transit | delivered | inspected
    lead_time_days: int = 0
    delivery_date: Optional[date] = None
    is_long_lead: bool = False


class ConstructionActivity(BaseModel):
    """Unified model for every activity across all construction management domains.

    An activity may originate from any domain (schedule, quality, safety, etc.)
    but carries enough metadata to participate in cross-domain dependency
    tracking and multi-domain workflow templates.
    """
    id: str = Field(min_length=1)
    type: ActivityType = ActivityType.SCHEDULE
    name: str = ""
    description: str = ""
    status: Status = Status.PLANNED

    # Cross-domain linking
    dependencies: List[str] = Field(default_factory=list)
    dependents: List[str] = Field(default_factory=list)

    # Domain-specific metadata
    schedule: ScheduleMeta = Field(default_factory=ScheduleMeta)
    cost: CostMeta = Field(default_factory=CostMeta)
    risk_level: RiskLevel = RiskLevel.LOW

    # Quality and safety gates
    quality_gates: List[QualityGate] = Field(default_factory=list)
    safety_gates: List[SafetyGate] = Field(default_factory=list)

    # Procurement linkage
    procurement_links: List[ProcurementLink] = Field(default_factory=list)

    # Contract and risk references
    contract_clauses: List[str] = Field(default_factory=list)
    risk_register_ids: List[str] = Field(default_factory=list)

    # Document and BIM linkage
    documents: List[str] = Field(default_factory=list)
    bim_element_guids: List[str] = Field(default_factory=list)

    # Location and responsibility
    location: str = ""  # zone | building | level | area
    responsible_party: str = ""  # trade | role | company

    # Handoff and downstream
    handoff_to: List[str] = Field(default_factory=list)

    # Free-form extras for domain-specific data not covered above
    meta: Dict[str, Any] = Field(default_factory=dict)

    # Audit
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def is_blocked(self) -> bool:
        """True if any quality or safety gate is blocking."""
        for g in self.quality_gates:
            if g.status == Status.BLOCKED or g.result == "fail":
                return True
        for g in self.safety_gates:
            if g.status == Status.BLOCKED:
                return True
        return self.status == Status.BLOCKED

    def days_overdue(self, ref_date: Optional[date] = None) -> int:
        """Number of days past target finish date, or 0 if not overdue."""
        if self.status != Status.OVERDUE and self.status != Status.IN_PROGRESS:
            return 0
        target = self.schedule.finish_date
        if target is None:
            return 0
        ref = ref_date or date.today()
        delta = (ref - target).days
        return max(0, delta)


class ActivityGraph(BaseModel):
    """A collection of construction activities with cross-domain linkage."""
    project_id: str = ""
    project_name: str = ""
    activities: List[ConstructionActivity] = Field(default_factory=list)

    def by_type(self, activity_type: ActivityType) -> List[ConstructionActivity]:
        return [a for a in self.activities if a.type == activity_type]

    def by_status(self, status: Status) -> List[ConstructionActivity]:
        return [a for a in self.activities if a.status == status]

    def critical_path_activities(self) -> List[ConstructionActivity]:
        return [
            a for a in self.activities
            if a.type == ActivityType.SCHEDULE and a.schedule.is_critical
        ]

    def blocked_activities(self) -> List[ConstructionActivity]:
        return [a for a in self.activities if a.is_blocked()]

    def overdue_activities(self) -> List[ConstructionActivity]:
        return [a for a in self.activities if a.status == Status.OVERDUE]

    def high_risk_activities(self) -> List[ConstructionActivity]:
        return [
            a for a in self.activities
            if a.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)
        ]

    def get_by_id(self, activity_id: str) -> Optional[ConstructionActivity]:
        for a in self.activities:
            if a.id == activity_id:
                return a
        return None

    def add_activity(self, activity: ConstructionActivity) -> None:
        if not self.get_by_id(activity.id):
            self.activities.append(activity)

    def link_activities(self, upstream_id: str, downstream_id: str) -> None:
        """Create a dependency link between two activities (cross-domain)."""
        upstream = self.get_by_id(upstream_id)
        downstream = self.get_by_id(downstream_id)
        if upstream and downstream:
            if downstream_id not in upstream.dependents:
                upstream.dependents.append(downstream_id)
            if upstream_id not in downstream.dependencies:
                downstream.dependencies.append(upstream_id)

    def summary(self) -> Dict[str, Any]:
        """High-level project health summary."""
        type_counts = {}
        status_counts = {}
        for a in self.activities:
            type_counts[a.type.value] = type_counts.get(a.type.value, 0) + 1
            status_counts[a.status.value] = status_counts.get(a.status.value, 0) + 1
        return {
            "project_id": self.project_id,
            "project_name": self.project_name,
            "total_activities": len(self.activities),
            "by_type": type_counts,
            "by_status": status_counts,
            "critical_count": len(self.critical_path_activities()),
            "blocked_count": len(self.blocked_activities()),
            "overdue_count": len(self.overdue_activities()),
            "high_risk_count": len(self.high_risk_activities()),
        }
