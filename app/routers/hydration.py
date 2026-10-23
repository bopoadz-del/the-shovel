"""Hydration HTTP surface.

- ``GET  /v1/hydration/latest?scope=global``
- ``GET  /v1/hydration/latest?scope=project&project_id=<id>``
- ``GET  /v1/hydration/history?scope=...&project_id=...&limit=20``
- ``POST /v1/hydration/run`` — admin-triggered manual pass. Body may include
  ``target_date`` (YYYY-MM-DD) and ``project_ids`` (list) to force-reprocess.

All endpoints require the standard API-key dependency.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.core import hydration_store
from app.core import projects as projects_store
from app.dependencies import require_api_key


def _require_admin(auth: dict) -> None:
    if auth.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")


def _require_project_access(project_id: str, auth: dict) -> None:
    if projects_store.get_project_accessible(project_id, auth.get("user_id")) is None:
        raise HTTPException(status_code=404, detail="Project not found")


router = APIRouter()


class HydrationRunRequest(BaseModel):
    target_date: Optional[str] = None  # YYYY-MM-DD; defaults to "yesterday UTC"
    project_ids: Optional[List[str]] = None  # force specific projects


@router.get("/v1/hydration/latest")
async def hydration_latest(
    scope: str = Query("global"),
    project_id: Optional[str] = Query(None),
    auth: dict = Depends(require_api_key),
):
    if scope not in ("global", "project"):
        raise HTTPException(status_code=400, detail="scope must be 'global' or 'project'")
    if scope == "project" and not project_id:
        raise HTTPException(status_code=400, detail="project scope requires project_id")
    if scope == "project":
        _require_project_access(project_id, auth)
    elif auth.get("role") != "admin":
        # Global latest is operator-wide; a plain user gets an empty read
        # rather than every tenant's hydration summary.
        return {"status": "empty", "scope": scope, "project_id": project_id}
    row = hydration_store.get_latest(scope, project_id)
    if row is None:
        return {"status": "empty", "scope": scope, "project_id": project_id}
    return {"status": "success", **row}


@router.get("/v1/hydration/history")
async def hydration_history(
    scope: Optional[str] = Query(None),
    project_id: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=200),
    auth: dict = Depends(require_api_key),
):
    if project_id:
        _require_project_access(project_id, auth)
    elif auth.get("role") != "admin":
        return {"status": "success", "count": 0, "runs": []}
    rows = hydration_store.list_history(
        scope=scope, project_id=project_id, limit=limit
    )
    return {"status": "success", "count": len(rows), "runs": rows}


@router.post("/v1/hydration/run")
async def hydration_run(
    request: HydrationRunRequest,
    auth: dict = Depends(require_api_key),
):
    """Manually trigger a hydration pass. Useful for one-off backfills and
    end-to-end smoke tests; the nightly scheduler runs the same path.

    The standalone hydration block was retired — hydration is now an
    operation on ``learning_engine``. This route stays at ``/v1/hydration/*``
    for operator familiarity but dispatches into the merged surface.
    """
    _require_admin(auth)
    from app.blocks import BLOCK_REGISTRY

    cls = BLOCK_REGISTRY.get("learning_engine")
    if cls is None:
        raise HTTPException(status_code=503, detail="learning_engine block not loaded")
    # shared_instance() for consistency with the singleton everywhere else.
    # Hydration POST is not a hot path, but using cls() here would re-load
    # state from disk on every operator-triggered run.
    block = cls.shared_instance()
    payload: Dict[str, Any] = {"operation": "hydrate"}
    if request.target_date:
        payload["target_date"] = request.target_date
    if request.project_ids:
        payload["project_ids"] = request.project_ids
    return await block.execute(payload, {})
