"""Schedule router — roadmap-named shim over existing schedule exports.

The L2 Schedule Engine roadmap expected ``app/routers/schedule.py``. The
platform's schedule actions already live under ``/v1/execute`` (construction
container) and ``/v1/projects/{id}/export/*`` (``app.routers.exports``). This
router exposes the same capabilities under the roadmap-named ``/v1/schedule``
prefix using the shim blocks so the audit finds the expected file without
rewriting the working implementation.
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.dependencies import require_user

router = APIRouter()


class _ScheduleGenerateRequest(BaseModel):
    # An empty brief used to yield a 204-activity generic "building"
    # programme with status "success" -- a fabrication wearing a 200. There
    # is nothing to schedule without an ask, so the ask is required.
    brief: str = Field(..., min_length=3, description="What to schedule")
    project_type: Optional[str] = None
    target_count: int = Field(default=200, ge=20, le=1000)
    start_date: Optional[str] = None
    procurement_lead_times: List[Dict[str, Any]] = Field(default_factory=list)
    target_milestones: List[Dict[str, Any]] = Field(default_factory=list)


class _CPMRequest(BaseModel):
    activities: List[Dict[str, Any]]
    project_start: Optional[str] = None


class _ManpowerRequest(BaseModel):
    activities: List[Dict[str, Any]]
    results: Optional[List[Dict[str, Any]]] = None
    task_resources: Optional[List[Dict[str, Any]]] = None
    period_unit: str = Field(default="week", pattern="^(week|month)$")


class _FastTrackRequest(BaseModel):
    activities: List[Dict[str, Any]]
    reductions: Dict[str, int]
    total_delay_days: int = 0


class _ExportRequest(BaseModel):
    activities: List[Dict[str, Any]]
    meta: Dict[str, Any] = Field(default_factory=dict)


@router.post("/v1/schedule/generate")
async def schedule_generate(
    req: _ScheduleGenerateRequest,
    auth: Dict[str, Any] = Depends(require_user),
):
    """Generate a CPM-ready L2 WBS from a brief."""
    from app.blocks.schedule_generator import ScheduleGeneratorBlock

    block = ScheduleGeneratorBlock()
    payload = req.model_dump()
    result = await block.process(payload, params={})
    if result.get("status") != "success":
        raise HTTPException(422, result.get("error", "Schedule generation failed"))
    return result


@router.post("/v1/schedule/cpm")
async def schedule_cpm(
    req: _CPMRequest,
    auth: Dict[str, Any] = Depends(require_user),
):
    """Run CPM over an activity network."""
    from app.blocks.cpm_engine import CPMEngineBlock

    block = CPMEngineBlock()
    result = await block.process(req.model_dump(), params={})
    if result.get("status") != "success":
        raise HTTPException(422, result.get("error", "CPM failed"))
    return result


@router.post("/v1/schedule/manpower")
async def schedule_manpower(
    req: _ManpowerRequest,
    auth: Dict[str, Any] = Depends(require_user),
):
    """Compute a trade-based manpower histogram."""
    from app.blocks.manpower_planner import ManpowerPlannerBlock

    block = ManpowerPlannerBlock()
    result = await block.process(req.model_dump(), params={})
    if result.get("status") != "success":
        raise HTTPException(422, result.get("error", "Manpower planning failed"))
    return result


@router.post("/v1/schedule/fasttrack")
async def schedule_fasttrack(
    req: _FastTrackRequest,
    auth: Dict[str, Any] = Depends(require_user),
):
    """Fast-track / crash compression analysis."""
    from app.blocks.fasttrack_analyzer import FastTrackAnalyzerBlock

    block = FastTrackAnalyzerBlock()
    result = await block.process(req.model_dump(), params={})
    if result.get("status") != "success":
        raise HTTPException(422, result.get("error", "Fast-track analysis failed"))
    return result


@router.post("/v1/schedule/export")
async def schedule_export(
    req: _ExportRequest,
    auth: Dict[str, Any] = Depends(require_user),
):
    """Write a cost-loaded L2 schedule workbook and return its path."""
    from app.blocks.schedule_excel_writer import ScheduleExcelWriterBlock

    block = ScheduleExcelWriterBlock()
    result = await block.process(req.model_dump(), params={})
    if result.get("status") != "success":
        raise HTTPException(422, result.get("error", "Excel export failed"))
    return result
