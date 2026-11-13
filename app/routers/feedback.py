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
from enum import Enum
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.dependencies import require_user
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


# ── answer feedback (ML-2) ──────────────────────────────────────────────

#: Same window export resolves ``message_index`` in, so one index names the
#: same assistant message on both surfaces.
_ANSWER_WINDOW = 200
#: Characters of the stored answer echoed back for the client's drift check.
_ECHO_CHARS = 80


class AnswerFeedbackKind(str, Enum):
    correct = "correct"
    wrong_figure = "wrong_figure"
    wrong_document = "wrong_document"
    missing = "missing"


#: Which optional fields each kind may carry. Anything else is a 422.
_KIND_FIELDS: dict[AnswerFeedbackKind, frozenset[str]] = {
    AnswerFeedbackKind.correct: frozenset(),
    AnswerFeedbackKind.wrong_figure: frozenset({"correct_value", "unit"}),
    AnswerFeedbackKind.wrong_document: frozenset({"note"}),
    AnswerFeedbackKind.missing: frozenset({"note"}),
}


class AnswerFeedbackRequest(BaseModel):
    """A user's verdict on one stored assistant answer.

    The client names the answer by its position among the conversation's
    assistant messages (``-1`` = newest), exactly as the export button does.
    Message id, cited ids and build sha are derived on the server.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    project_id: str = Field(..., min_length=1, max_length=200)
    conversation_id: str = Field(..., min_length=1, max_length=200)
    message_index: int = -1
    kind: AnswerFeedbackKind
    correct_value: Optional[float] = None
    unit: Optional[str] = Field(None, min_length=1, max_length=32)
    note: Optional[str] = Field(None, min_length=1, max_length=200)
    #: The start of the answer the client shows. When sent, the stored answer
    #: must start with it (whitespace collapsed) or nothing is stored (409):
    #: the index named a different message than the one the user judged.
    answer_head: Optional[str] = Field(None, max_length=400)

    @model_validator(mode="after")
    def _fields_match_kind(self) -> "AnswerFeedbackRequest":
        allowed = _KIND_FIELDS[self.kind]
        sent = {name for name in ("correct_value", "unit", "note") if getattr(self, name) is not None}
        extra = sorted(sent - allowed)
        if extra:
            raise ValueError(f"{', '.join(extra)} not accepted with kind={self.kind.value}")
        return self


class AnswerFeedbackResponse(BaseModel):
    status: str
    message_id: str
    #: First characters of the stored answer; the client compares them with
    #: the answer it shows before it tells the user the verdict was recorded.
    answer_head: str


def _collapsed(text: str) -> str:
    return " ".join(str(text or "").split())


def _cited_doc_ids(provenance: Any) -> list[str]:
    """Distinct ``doc_id`` values of a stored provenance record, in order."""
    ids: dict[str, None] = {}
    for entry in provenance if isinstance(provenance, list) else []:
        value = entry.get("doc_id") if isinstance(entry, dict) else None
        if value not in (None, ""):
            ids[str(value)] = None
    return list(ids)


def _require_feedback_owner(project_id: str, conversation_id: str, auth: dict) -> None:
    """Feedback is the caller's verdict on their own turn.

    For a non-admin, read access is checked first so every denial has the
    same 404 shape. The conversation must then exist and belong to
    ``project_id`` (for an admin too), and a non-admin must own it. A
    NULL-owner (shared) conversation is readable through its project grant,
    but is refused here for non-admins. An admin may record on any
    conversation, so the read check (which serves an owned row only to its
    writer) is not applied to an admin.
    """
    from app.core import agent_memory
    from app.routers.agents import _enforce_conversation_access
    from app.routers.exports import _require_conversation_in_project

    is_admin = auth.get("role") == "admin"
    if not is_admin:
        _enforce_conversation_access(conversation_id, auth)
    conv = agent_memory.get_conversation(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    _require_conversation_in_project(project_id, conversation_id, auth)
    if is_admin:
        return
    if not conv.get("owner_id") or conv.get("owner_id") != auth.get("user_id"):
        raise HTTPException(status_code=404, detail="Conversation not found")


@router.post("/v1/feedback/answer", response_model=AnswerFeedbackResponse)
async def submit_answer_feedback(
    req: AnswerFeedbackRequest,
    auth: dict = Depends(require_user),
) -> AnswerFeedbackResponse:
    """Record a user's verdict on one stored answer into ``ml_feedback``.

    Append-only: a repeat verdict on the same answer is stored as a new row,
    never an upsert. A reader that wants one verdict per answer takes the
    latest row per ``(message_id, user_id)``.

    The client may send ``answer_head`` (the start of the answer it shows);
    a stored answer that does not start with it is a 409 and stores nothing.
    The response echoes ``message_id`` and the stored answer's first 80
    characters so the client can check the same thing before it says
    "recorded".

    Only ``ml_feedback`` is written; messages and answers are read, never
    changed. A store that does not report success (not configured, disabled,
    write error) is a 503 so the client never shows "recorded" for a verdict
    that was not kept.
    """
    from app.blocks.ml_loop import record as ml_record
    from app.core import agent_memory

    _require_feedback_owner(req.project_id, req.conversation_id, auth)

    messages = agent_memory.get_messages(req.conversation_id, limit=_ANSWER_WINDOW)
    answers = [m for m in messages if m.get("role") == "assistant"]
    try:
        chosen = answers[req.message_index]
    except IndexError:
        raise HTTPException(status_code=404, detail=f"message_index {req.message_index} out of range")

    if req.answer_head is not None:
        expected = _collapsed(req.answer_head)[:_ECHO_CHARS]
        if not expected or not _collapsed(chosen.get("content") or "").startswith(expected):
            raise HTTPException(status_code=409, detail="answer does not match the stored message")

    message_id = str(chosen.get("id") or "")
    result = ml_record("record_feedback", {
        "project_id": req.project_id,
        "conversation_id": req.conversation_id,
        "message_id": message_id or None,
        "kind": req.kind.value,
        "correct_value": req.correct_value,
        "unit": req.unit,
        "note": req.note,
        "cited_ids": _cited_doc_ids(chosen.get("provenance")),
        "user_id": str(auth.get("user_id") or "") or None,
    })
    if result.get("status") != "success":
        logger.warning("answer feedback not stored: status=%s", result.get("status"))
        raise HTTPException(status_code=503, detail="feedback store unavailable")

    return AnswerFeedbackResponse(
        status="recorded",
        message_id=message_id,
        answer_head=str(chosen.get("content") or "")[:_ECHO_CHARS],
    )
