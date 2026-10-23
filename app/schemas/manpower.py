"""Manpower schemas — shim layer for the L2 Schedule Engine roadmap.

The roadmap expected a dedicated ``app/schemas/manpower.py``. The canonical
histogram models already live in ``app/schemas/cpm.py``; this module exposes
them under the roadmap-named module and adds thin planner wrappers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.schemas.cpm import (
    Activity,
    CPMInput,
    CPMResult,
    HistogramPeriod,
    ResourceAssignment,
    ResourceHistogram,
)


class TradeAssignment(BaseModel):
    """Crew size for a single trade on a single activity."""

    trade: str = Field(min_length=1)
    count: float = Field(default=1.0, gt=0)


class ManpowerPlanInput(BaseModel):
    """Input for trade-based manpower planning."""

    activities: List[Activity]
    results: Optional[List[CPMResult]] = None
    task_resources: Optional[List[Dict[str, Any]]] = None
    period_unit: str = Field(default="week", pattern="^(week|month)$")


class ManpowerPlanOutput(BaseModel):
    """Trade-based manpower histogram output."""

    period_unit: str
    periods: List[HistogramPeriod]
    peak_total: float
    peak_period: str
    by_trade_totals: Dict[str, float]
    total_manhours: float


__all__ = [
    "Activity",
    "CPMInput",
    "CPMResult",
    "HistogramPeriod",
    "ManpowerPlanInput",
    "ManpowerPlanOutput",
    "ResourceAssignment",
    "ResourceHistogram",
    "TradeAssignment",
]
