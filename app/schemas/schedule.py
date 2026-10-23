"""Schedule schemas — shim layer that re-exports the CPM schema family.

The L2 Schedule Engine roadmap expected a dedicated ``app/schemas/schedule.py``.
The platform's canonical schedule models already live in ``app/schemas/cpm.py``;
this module exposes them under the roadmap-named module so callers can import
``app.schemas.schedule`` without changing the underlying data model.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.schemas.cpm import (
    Activity,
    CPMInput,
    CPMOutput,
    CPMResult,
    Dependency,
    DependencyType,
    GanttBar,
    HistogramPeriod,
    ResourceAssignment,
    ResourceHistogram,
    WorkCalendar,
)


class ScheduleScope(BaseModel):
    """Scope extracted from a source document / brief."""

    project_type: str = "data_center"
    summary: str = ""
    requirements: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    schedule_targets: List[Dict[str, Any]] = Field(default_factory=list)
    equipment_lead_times: List[Dict[str, Any]] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)


class ScheduleGenerationRequest(BaseModel):
    """Input shape for the L2 schedule generator."""

    brief: str = ""
    project_type: Optional[str] = None
    target_count: int = Field(default=200, ge=20, le=1000)
    start_date: Optional[str] = None
    procurement_lead_times: List[Dict[str, Any]] = Field(default_factory=list)
    target_milestones: List[Dict[str, Any]] = Field(default_factory=list)


class ScheduleGenerationResponse(BaseModel):
    """Output shape returned by the L2 schedule generator."""

    status: str = "success"
    wbs_id: str = ""
    project_type: str = ""
    brief: str = ""
    target_count: int = 0
    actual_count: int = 0
    start_date: Optional[str] = None
    activities: List[Dict[str, Any]] = Field(default_factory=list)
    wbs_tree: Dict[str, str] = Field(default_factory=dict)
    summary: Dict[str, Any] = Field(default_factory=dict)
    procurement_injected: int = 0
    target_milestones: List[Dict[str, Any]] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)


class FastTrackScenario(BaseModel):
    """One compression / fast-track option."""

    strategy: str
    description: str
    potential_savings_days: float
    cost_impact: str
    feasibility: str


class FastTrackAnalysis(BaseModel):
    """Fast-track / crash analysis result."""

    baseline_duration: int
    revised_duration: int
    days_saved: int
    compressed_activity_ids: List[str]
    scenarios: List[FastTrackScenario]


__all__ = [
    "Activity",
    "CPMInput",
    "CPMOutput",
    "CPMResult",
    "Dependency",
    "DependencyType",
    "FastTrackAnalysis",
    "FastTrackScenario",
    "GanttBar",
    "HistogramPeriod",
    "ResourceAssignment",
    "ResourceHistogram",
    "ScheduleGenerationRequest",
    "ScheduleGenerationResponse",
    "ScheduleScope",
    "WorkCalendar",
]
