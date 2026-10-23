"""Project reasoning API — Reasoning Engine Plan 6.

POST /v1/project/ask — run the Project Reasoner over a persistent session.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

from app.dependencies import require_user
from app.core.session_store import SessionStore, get_session_store
from app.schemas.project_session import ProjectSession

router = APIRouter()

# Process-wide session store. app/main.py overwrites this at startup with the
# shared instance; tests monkeypatch it. Lazily created so importing the module
# never fails.
_store: SessionStore = get_session_store()


def get_project_store() -> SessionStore:
    """Accessor for the active session store (overridable in tests)."""
    return _store


def _reasoner_factory():
    """Build a ProjectReasonerBlock. Indirected so tests can swap in a mock."""
    from app.blocks.project_reasoner import ProjectReasonerBlock
    return ProjectReasonerBlock()


class ProjectAskRequest(BaseModel):
    session_id: str = Field(min_length=1)
    request: str
    # optional: load/replace the session's activity list on this turn
    activities: Optional[List[Dict[str, Any]]] = None
    # optional: the project whose indexed documents the reasoner should
    # consult for evidence. Defaults to session_id for back-compat with the
    # UI convention where the project id IS the session id.
    project_id: Optional[str] = None


@router.post("/v1/project/ask")
async def project_ask(
    body: ProjectAskRequest, auth: dict = Depends(require_user)
):
    """Answer a project question. Creates the session on first use, persists
    it after the turn so follow-up questions build on prior state."""
    if not body.request.strip():
        raise HTTPException(422, "request must not be empty")

    # Cap client-supplied activities to bound session-store memory growth.
    _MAX_ACTIVITIES = 5000
    if body.activities is not None and len(body.activities) > _MAX_ACTIVITIES:
        raise HTTPException(
            422,
            f"activities exceeds the maximum of {_MAX_ACTIVITIES}",
        )

    # Session ownership: create on first use tagged with caller's user_id;
    # reject access if the session exists but was created by a different user.
    caller_id = auth["user_id"]
    session = await _store.get(body.session_id)
    if session is None:
        session = ProjectSession.new(body.session_id, user_id=caller_id)
        await _store.save(session)
    elif session.user_id != caller_id:
        raise HTTPException(404, "Session not found")
    if body.activities is not None:
        session.data["activities"] = body.activities

    reasoner = _reasoner_factory()
    # Use the explicit project_id when given; otherwise fall back to the
    # session_id (UI convention: activeProjectId IS the session id).
    project_id = body.project_id or body.session_id

    # Tenant gate (same pattern as heavy-reasoning at app/routers/chat.py:97-121):
    # drop the project_id when the caller can't access it. The reasoner's
    # search_project_documents then runs without a project scope and returns
    # nothing, instead of leaking another tenant's docs into the answer.
    # Reasoning is read-only over documents, so admin-approved platform
    # projects (and the master-corpus alias backed by one) are readable by
    # non-owners — same policy as chat and document search.
    safe_project_id: Optional[str] = project_id
    if project_id:
        try:
            from app.core import projects as projects_store
            if projects_store.get_project(
                project_id, user_id=caller_id, include_admin_approved=True
            ) is None:
                logger.warning(
                    "project_ask: user=%s cannot access project=%s; dropping project_id",
                    caller_id, project_id,
                )
                safe_project_id = None
        except Exception:
            logger.exception("project_ask: ownership check failed; dropping project_id (fail-closed)")
            safe_project_id = None

    # Resolve the master-corpus alias to its backing source corpus for
    # retrieval (e.g. master_corpus -> projects_folder), matching
    # app.routers.agents and doc_search. Without this, project mode queries the
    # alias id — which has no indexed chunks — and always falls back to the
    # no-context message. Non-alias ids pass through unchanged.
    retrieval_project_id = safe_project_id
    if safe_project_id:
        try:
            from app.core import projects as projects_store
            retrieval_project_id = (
                projects_store._master_corpus_source(safe_project_id)
                or safe_project_id
            )
        except Exception:
            logger.exception("project_ask: alias resolution failed; using original id")
            retrieval_project_id = safe_project_id

    result = await reasoner.process({"request": body.request,
                                     "session": session,
                                     "project_id": retrieval_project_id})

    await _store.save(session)   # persist the turn — history, computed state, cache

    # Cost-grounding gate (§3.2 coverage): refuse an ungrounded cost/rate figure
    # in the reasoner's answer exactly as the agent path does. The reasoner
    # surfaces the retrieved excerpts (classified per-chunk for rate-semantics)
    # and this turn's executed step results (authoritative computed figures) as
    # internal grounding evidence; it is used only for gating and never returned
    # to the client. Non-cost answers pass through untouched; never raises.
    answer = result.get("answer", "")
    try:
        from app.agents.runtime import (
            gate_cost_answer,
            format_excerpts_as_rag_context,
        )
        grounding = result.get("_grounding") or {}
        results_text = grounding.get("results_text") or ""
        answer = gate_cost_answer(
            answer,
            rag_context=format_excerpts_as_rag_context(grounding.get("excerpts")),
            authoritative_texts=[results_text] if results_text.strip() else None,
            user_message=body.request,
            project_id=retrieval_project_id,
        )
    except Exception:
        logger.exception(
            "project_ask: cost-grounding gate failed; passing answer through"
        )

    return {
        "session_id": body.session_id,
        "status": result.get("status"),
        "answer": answer,
        "understanding": result.get("understanding", ""),
        "plan": result.get("plan"),
        "execution": result.get("execution"),
        "artifacts": [a.model_dump() for a in session.artifacts],
        # Surface RAG sources when the reasoner answered from project docs
        # (e.g. the graceful-degradation fallback when no plan could be built).
        "sources": result.get("sources", []),
        "error": result.get("error"),
    }
