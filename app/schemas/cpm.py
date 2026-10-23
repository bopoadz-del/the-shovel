"""Pydantic schemas for the CPM engine — Reasoning Engine Plan 1."""

from __future__ import annotations

from datetime import date, timedelta
from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class DependencyType(str, Enum):
    FS = "FS"  # finish-to-start (default)
    SS = "SS"  # start-to-start
    FF = "FF"  # finish-to-finish
    SF = "SF"  # start-to-finish


class Dependency(BaseModel):
    predecessor_id: str
    type: DependencyType = DependencyType.FS
    lag: int = 0  # working days; may be negative


class WorkCalendar(BaseModel):
    """A working-day calendar. Monday=0 .. Sunday=6."""

    work_weekdays: List[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])
    holidays: List[date] = Field(default_factory=list)

    def nth_working_day(self, start: date, n: int) -> date:
        """Calendar date of the n-th working day (0-indexed); offset 0 is the
        first working day on or after `start`."""
        if not self.work_weekdays:
            raise ValueError("WorkCalendar.work_weekdays is empty — no working days")
        if n < 0:
            raise ValueError("nth_working_day: n must be >= 0")
        work = set(self.work_weekdays)
        hol = set(self.holidays)

        def is_working(d: date) -> bool:
            return d.weekday() in work and d not in hol

        d = start
        while not is_working(d):
            d += timedelta(days=1)
        count = 0
        while count < n:
            d += timedelta(days=1)
            if is_working(d):
                count += 1
        return d


class ResourceAssignment(BaseModel):
    trade: str
    count: float = 1.0  # crew size / headcount on the activity


class Activity(BaseModel):
    id: str = Field(min_length=1)
    name: str = ""
    duration: int = Field(ge=0)  # working days
    predecessors: List[Dependency] = Field(default_factory=list)
    wbs_code: str = ""
    resources: List[ResourceAssignment] = Field(default_factory=list)


class CPMResult(BaseModel):
    id: str
    name: str
    duration: int
    early_start_day: int
    early_finish_day: int
    late_start_day: int
    late_finish_day: int
    total_float: int
    free_float: int
    is_critical: bool
    early_start: Optional[date] = None
    early_finish: Optional[date] = None
    late_start: Optional[date] = None
    late_finish: Optional[date] = None


class CPMInput(BaseModel):
    activities: List[Activity]
    project_start: Optional[date] = None
    calendar: WorkCalendar = Field(default_factory=WorkCalendar)


class CPMOutput(BaseModel):
    results: List[CPMResult]
    project_duration: int
    project_finish: Optional[date] = None
    critical_path: List[str]
    critical_percentage: float
    near_critical: List[str]


class HistogramPeriod(BaseModel):
    index: int
    label: str
    total: float
    by_trade: Dict[str, float]


class ResourceHistogram(BaseModel):
    period_unit: Literal["week", "month"]
    periods: List[HistogramPeriod]
    peak_total: float
    peak_period: str
    by_trade_totals: Dict[str, float]
    total_manhours: float


class GanttBar(BaseModel):
    id: str
    name: str
    start_day: int
    end_day: int
    is_critical: bool
