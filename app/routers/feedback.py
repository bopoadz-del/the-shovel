"""Routing-feedback surface — the "Wrong action, should be X" signal that
makes the chat router actually self-correcting.

Persists corrections as ``routing_decisions`` patterns on
:class:`app.blocks.learning_engine.LearningEngineBlock` with
``corrected: true``. The data-extraction layer in
``app/core/learning/router.py`` already filters these via the
``label_quality="corrected"`` lever; ``train_router`` accepts a
``prefer_corrected: bool`` switch (PR 1) that drops the noisy "auto"
rows once enough corrections accumulate.

In plain terms: PR 1 trained on what the keyword router *did*. This
route lets users tell us what the keyword router *should have done*.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.routers.auth import require_api_key

logger = logging.getLogger(__name__)

router = APIRouter()


class RoutingCorrectionRequest(BaseModel):
    """A user telling us the router picked the wrong action.

    ``original_action`` is what the router actually dispatched (recorded
    by smart_orchestrator's _record_routing_decision hook); the value is
    optional so callers can post a correction without knowing what was
    originally picked, but providing it lets us also log a negative
    example for that action later.
    """

    message: str = Field(..., min_length=1, description="The user message that got misrouted")
    correct_action: str = Field(..., min_length=1, description="What the router SHOULD have picked")
    project_id: str = Field("default", min_length=1, description="Scope to one project")
    original_action: Optional[str] = Field(None, description="What the router actually picked, if known")


class RoutingCorrectionResponse(BaseModel):
    status: str
    message: str
    pattern_count: Optional[int] = None


@router.post("/v1/feedback/route", response_model=RoutingCorrectionResponse)
async def submit_routing_correction(
    req: RoutingCorrectionRequest,
    auth: dict = Depends(require_api_key),
) -> RoutingCorrectionResponse:
    """Record one user correction of a routing decision.

    The next ``train_router`` call picks this up via
    ``app/core/learning/router._runtime_data_from_patterns`` — the
    pattern's ``corrected: true`` flag promotes it to
    ``label_quality="corrected"`` in the training set. With
    ``prefer_corrected=true`` AND enough corrections, the classifier
    retrains on user-validated data only.

    No model training happens inside this request — that would block
    the response on a multi-second sklearn fit. Storage only; trigger
    a separate ``train_router`` call (nightly cron, or operator
    dashboard button) to consume the new rows.
    """
    from app.core import projects as projects_store

    pid = req.project_id
    # JWT callers must not stamp another tenant's real project_id.
    # API-key callers are the operator identity (the development key / SYSTEM_USER):
    # existing tests and the e2e surface walk post a learning-scope id
    # that is not owned by that singleton. Skip the grant for them.
    if (
        pid
        and pid != "default"
        and auth.get("auth_method") == "jwt"
        and projects_store.get_project_accessible(pid, auth.get("user_id")) is None
    ):
        raise HTTPException(status_code=404, detail="Project not found")

    from app.blocks import BLOCK_REGISTRY

    cls = BLOCK_REGISTRY.get("learning_engine")
    if cls is None:
        raise HTTPException(status_code=503, detail="learning_engine block not loaded")
    # shared_instance() so concurrent /v1/feedback/route POSTs share one
    # learning_engine instance with the lock around _record_pattern's
    # read-modify-write — without it, simultaneous corrections raced and
    # silently lost rows. Reviewer fix from PRs #19-#23 retro.
    le = cls.shared_instance()
    try:
        result = le._record_pattern(
            {
                "project_id": req.project_id,
                "category": "routing_decisions",
                "observation": json.dumps({
                    "text": req.message[:500],
                    "action": req.correct_action,
                    "score": 1.0,  # corrections are ground truth — full weight
                    "source": "user_correction",
                    "corrected": True,
                    "original_action": req.original_action,
                }, ensure_ascii=False),
                "source": "feedback_route",
            },
            {},
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("feedback route failed to record correction")
        raise HTTPException(status_code=500, detail=f"record failed: {exc}")

    if result.get("status") != "success":
        raise HTTPException(
            status_code=500,
            detail=f"record_pattern returned: {result.get('error', 'unknown error')}",
        )

    return RoutingCorrectionResponse(
        status="recorded",
        message=(
            f"correction recorded for project {req.project_id!r}; "
            f"call train_router (prefer_corrected=true once volume permits) to consume"
        ),
        pattern_count=int(result.get("total_observations") or 0),
    )
