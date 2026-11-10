"""HTTP routes for runtime agents (the AI assistants users chat with).

Routes:
    GET  /v1/agents                       — list all agents
    GET  /v1/agents/{name}                — describe one agent (system prompt + tools)
    POST /v1/agents/{name}/chat           — single-turn JSON response (with tool calls)
    POST /v1/agents/{name}/chat/stream    — SSE: start / tool_call / tool_result / token / end / error

Auth: the same Bearer key (the development key from CEREBRUM_DEV_KEY, or any registered key) as the rest of /v1.
"""

from app.core.offload import off_loop as _off_loop
import asyncio
import json
import logging

logger = logging.getLogger(__name__)

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.agents import AGENT_REGISTRY, get_agent
from app.agents.runtime import ROUTING_GENERALISTS, select_agent_for_message
from app.core import agent_memory
from app.core import projects as store
from app.core.privileges import caller_may_use_agent
from app.routers.chat_watchdog import sanitize_error_frame
from app.dependencies import require_user

router = APIRouter()

_WS_PREFIX = "ws-"


def _workspace_project_candidates(conversation_id: str) -> list[str]:
    """Project ids a ``ws-*`` conversation id could refer to.

    The workspace UI has two shapes (``ProjectWorkspace.tsx``):
      * ``ws-{projectId}`` — default / legacy thread
      * ``ws-{projectId}-{unix_ms}`` — "New chat" mints a numeric suffix

    Ownership is derived from the id, not the stored row. Try the full remainder
    first (legacy), then strip one trailing ``-{suffix}`` so New-chat ids still
    bind to the project. Attackers targeting ``ws-{victim}-{anything}`` still
    404 because they do not own ``victim``.
    """
    if not conversation_id.startswith(_WS_PREFIX):
        return []
    rest = conversation_id[len(_WS_PREFIX):]
    if not rest:
        return []
    out = [rest]
    if "-" in rest:
        pid, suffix = rest.rsplit("-", 1)
        if pid and suffix:
            out.append(pid)
    return out


# Predefined deliverable flows may take over a turn only when the caller
# entered through a GENERALIST agent -- addressing a specialist by name is a
# deliberate choice the router must not override (F24). Same set as the
# smart_orchestrator redirect allowlist in runtime.ROUTING_GENERALISTS.
_PREDEFINED_GENERALISTS = ROUTING_GENERALISTS


def _predefined_may_intercept(requested_agent_name: str) -> bool:
    """True when a predefined workflow is allowed to replace the agent turn."""
    return requested_agent_name in _PREDEFINED_GENERALISTS


def _enforce_conversation_access(conversation_id: str, auth: dict) -> None:
    """Authoritative access check for a conversation, on both read and write.

    Workspace conversations use the id ``ws-{projectId}`` — the project id is IN
    the id, so access is derived from the id itself, NOT from the (spoofable)
    stored ``project_id`` column on the conversation row.

    - ``ws-{pid}`` id  → caller must be able to read project ``pid`` under the
      same grant as ``GET /v1/projects/{pid}``: owner, or an approved
      platform-shared row (``admin_drive_approved``, boot-seeded
      ``system_seed``, or an id in ``RAG_GENERAL_KNOWLEDGE_PROJECTS``).
      Covers both "project doesn't exist" and "private project belongs to
      someone else". The physical master-corpus source id stays owner-only;
      only the alias is shared. A row that does not exist yet may be
      created. A row that exists is served only to its writer; a NULL
      owner (written before owner tracking) is served only to the project
      row owner.
    - non-``ws-`` id   → not a project-scoped workspace conversation. If a stored
      conversation row exists with a ``project_id``, that same read grant is
      checked; a stored row with NO ``project_id`` has no ownership binding → 404;
      a non-existent row is allowed (ad-hoc API conversation). When the row
      records an owner, only that user may read it. A NULL owner keeps the
      project grant so a shared platform conversation stays readable.

    Raises HTTPException(404) when access is denied.
    """
    user_id = auth["user_id"]
    if conversation_id.startswith(_WS_PREFIX):
        accessible: str | None = None
        for project_id in _workspace_project_candidates(conversation_id):
            # Same include_admin_approved grant as project GET / documents /
            # rag search (PR #586). Master-corpus-only was the leftover that
            # 404'd New chat on a shared general-knowledge project for every non-owner.
            # Row-only check: listing documents here walked the whole corpus.
            if store.can_access_project(
                project_id, user_id=auth["user_id"], include_admin_approved=True
            ):
                accessible = project_id
                break
        if accessible is None:
            raise HTTPException(404, "Conversation not found")
        conv = agent_memory.get_conversation(conversation_id)
        if conv is None:
            return
        owner_id = store.project_owner(store.storage_project_id(accessible))
        if not agent_memory.session_visible_to(conv, user_id, owner_id):
            raise HTTPException(404, "Conversation not found")
        return

    # Non-workspace conversation id — fall back to the stored row's binding.
    conv = agent_memory.get_conversation(conversation_id)
    if conv is None:
        return  # ad-hoc API conversation, not yet created
    stored_pid = conv.get("project_id")
    if stored_pid is None:
        # No ownership binding at all — do not serve it.
        raise HTTPException(404, "Conversation not found")
    if not store.can_access_project(
        stored_pid, user_id=user_id, include_admin_approved=True
    ):
        raise HTTPException(404, "Conversation not found")
    if conv.get("owner_id") and conv.get("owner_id") != user_id:
        raise HTTPException(404, "Conversation not found")


class AgentChatRequest(BaseModel):
    message: str
    history: list[dict[str, str]] | None = None
    model: str | None = None  # override agent default if needed
    project_id: str | None = None
    conversation_id: str | None = None


@router.get("/v1/agents/conversations/{conversation_id}/messages")
async def get_conversation_messages(
    conversation_id: str,
    auth: dict = Depends(require_user),
):
    """Return the stored messages for a conversation.

    Ownership is enforced FIRST via ``_enforce_conversation_access`` — for a
    ``ws-{pid}`` id this is derived from the id, not the stored row, so an
    attacker probing a victim's conversation gets 404 (never an empty 200 that
    would confirm the id format). A non-existent but owned conversation still
    returns 200 with an empty list.
    """
    (await _off_loop(_enforce_conversation_access, conversation_id, auth))

    conv = agent_memory.get_conversation(conversation_id)
    if conv is None:
        return {"conversation_id": conversation_id, "messages": []}

    msgs = (await _off_loop(_reopened_messages, conversation_id))
    return {"conversation_id": conversation_id, "messages": msgs}


def _reopened_messages(conversation_id: str) -> list[dict]:
    """Stored messages as a reopened conversation shows them: an assistant
    answer stored before the exit check existed passes it on the way out."""
    from app.agents import answer_exit

    msgs = agent_memory.get_messages(conversation_id)
    for m in msgs:
        if m.get("role") == "assistant" and isinstance(m.get("content"), str):
            m["content"] = answer_exit.check_text_or_fallback(m["content"])
    return msgs


@router.get("/v1/agents")
async def list_agents(auth: dict = Depends(require_user)):
    visible = [a for a in AGENT_REGISTRY.values() if caller_may_use_agent(a, auth.get("role"))]
    return {
        "count": len(visible),
        "agents": [
            {
                "name": a.name,
                "description": a.description,
                "icon": a.icon,
                "model": a.model,
                "tools": a.allowed_blocks,
                "tool_count": len(a.allowed_blocks),
                # F35: an agent whose functional blocks all failed to load is
                # surfaced as unavailable instead of listing as a ghost.
                "available": a.unavailable_reason() is None,
                "unavailable_reason": a.unavailable_reason(),
            }
            for a in visible
        ],
    }


@router.get("/v1/agents/{name}")
async def get_agent_info(name: str, auth: dict = Depends(require_user)):
    agent = get_agent(name)
    if not agent or not caller_may_use_agent(agent, auth.get("role")):
        raise HTTPException(404, f"Agent '{name}' not found")
    return {
        "name": agent.name,
        "description": agent.description,
        "icon": agent.icon,
        "model": agent.model,
        "temperature": agent.temperature,
        "max_tokens": agent.max_tokens,
        "tools": agent.allowed_blocks,
        "system_prompt": agent.system_prompt,
    }


@router.post("/v1/agents/{name}/chat")
async def agent_chat(name: str, req: AgentChatRequest, auth: dict = Depends(require_user)):
    agent = get_agent(name)
    if not agent or not caller_may_use_agent(agent, auth.get("role")):
        raise HTTPException(404, f"Agent '{name}' not found")
    _reason = agent.unavailable_reason()
    if _reason:
        raise HTTPException(503, f"Agent '{name}' is unavailable: {_reason}")

    # Authoritative access check: a ws-{pid} conversation_id binds to a
    # project; the caller must be able to read it (owner or approved
    # platform-shared). This runs BEFORE the agent so an attacker cannot
    # create/write a victim's private ws-{pid} conversation.
    if req.conversation_id is not None:
        (await _off_loop(_enforce_conversation_access, req.conversation_id, auth))

    # Pilot: resolve the master-corpus alias to its backing source corpus so
    # retrieval and storage use the existing full-drive index without
    # re-importing anything. The conversation id stays ``ws-{alias}``.
    resolved_project_id = req.project_id
    if req.project_id == store.MASTER_CORPUS_PROJECT_ID:
        resolved_project_id = store.MASTER_CORPUS_SOURCE_PROJECT_ID

    # Defense in depth: if a project_id is provided, the caller must own it.
    # For the master-corpus alias, conversation access already gates the turn
    # and the source corpus belongs to the system user.
    if req.project_id is not None and req.project_id != store.MASTER_CORPUS_PROJECT_ID:
        project = store.get_project(
            req.project_id, user_id=auth["user_id"], include_admin_approved=True,
        )
        if project is None:
            raise HTTPException(404, "Project not found")

    if req.model:
        agent = _agent_with_override(agent, model=req.model)

    # PR #78 — smart_orchestrator routing gate. Real user chat traffic must
    # pass through the keyword router. When the message classifies as a
    # generative intent at sufficient confidence, redirect to heavy-reasoning
    # so the tool-call loop runs the requested pipeline instead of producing
    # prose. Pass-through cases (small talk, low confidence, already on
    # heavy-reasoning) return the original agent unchanged with classification
    # metadata in `routing` for observability. Kill-switch:
    # SMART_ORCH_ROUTING_DISABLED=true. See app/agents/runtime.py for the gate.
    agent, routing = await select_agent_for_message(req.message, agent)

    result = await agent.chat(
        req.message,
        history=req.history,
        project_id=resolved_project_id,
        conversation_id=req.conversation_id,
        user_id=auth["user_id"],
    )

    # Echo conversation_id back so the client can resume the conversation.
    if req.conversation_id is not None:
        result["conversation_id"] = req.conversation_id

    # Surface the routing decision so callers can render a "routed to ..."
    # badge or log it. Always present, with reason set even on pass-through.
    result["routing"] = routing

    return result


# The client sends the full conversation thread as `history` every turn. On a
# long pilot session that balloons the prompt — slower time-to-first-token,
# which on a memory-tight box reads like the cold-start stalls we saw. Bound it.
# NOTE: defense-in-depth, NOT the fix for the streaming hang (that was the
# frontend stall-timeout resetting on heartbeats). Tests showed history content
# is not what stalls the model; this just keeps the prompt from growing without
# limit on multi-turn conversations.
_MAX_HISTORY_MESSAGES = 24  # ~12 recent turns


def _bound_history(history: object) -> list:
    """Keep only the most recent messages and drop empty-content turns.

    Empty turns come from an interrupted stream (an assistant bubble that never
    produced text); they carry no signal, so dropping them is free. Capping the
    count guards prompt bloat on long conversations.
    """
    if not isinstance(history, list):
        return []
    cleaned = [
        m for m in history
        if isinstance(m, dict) and isinstance(m.get("content"), str) and m["content"].strip()
    ]
    return cleaned[-_MAX_HISTORY_MESSAGES:]


@router.post("/v1/agents/{name}/chat/stream")
async def agent_chat_stream(name: str, request: Request, auth: dict = Depends(require_user)):
    from app.agents import driver
    from app.core import turn_timing
    turn_timing.mark_arrival()
    if driver.mode() == "request":
        from app.core import users as _users
        from app.core.offload import off_loop as _driver_off_loop

        driver.mark_request(request.headers, await _driver_off_loop(_users.driver_mode_for, auth.get("user_id")))
    else:
        driver.mark_request(request.headers)
    agent = get_agent(name)
    if not agent or not caller_may_use_agent(agent, auth.get("role")):
        raise HTTPException(404, f"Agent '{name}' not found")
    _reason = agent.unavailable_reason()
    if _reason:
        raise HTTPException(503, f"Agent '{name}' is unavailable: {_reason}")
    try:
        body = await request.json()
    except Exception:
        body = {}
    message = body.get("message", "")
    history = _bound_history(body.get("history"))
    model = body.get("model")
    project_id = body.get("project_id")
    conversation_id = body.get("conversation_id")
    # Opt-in A/B path: when set, runtime runs a second LLM call without the
    # RAG system message and surfaces both responses in the SSE end event.
    rag_debug = request.query_params.get("rag_debug", "").lower() in ("true", "1", "yes")

    # Authoritative access check: a ws-{pid} conversation_id binds to a
    # project; the caller must be able to read it (owner or approved
    # platform-shared). Raised here (before StreamingResponse) so an
    # attacker cannot create/write a victim's private ws-{pid} conversation.
    if conversation_id is not None:
        (await _off_loop(_enforce_conversation_access, conversation_id, auth))
    from app.core.privileges import raise_if_inaccessible_document_ids
    raise_if_inaccessible_document_ids(auth, {"document_ids": body.get("document_ids") or []})

    # Defense in depth: if a project_id is provided, the caller must own it
    # OR it must be an admin-approved platform project visible to them (PR D).
    # The master-corpus alias is gated by conversation access above.
    if project_id is not None and project_id != store.MASTER_CORPUS_PROJECT_ID:
        project = store.get_project(
            project_id, user_id=auth["user_id"], include_admin_approved=True
        )
        if project is None:
            raise HTTPException(404, "Project not found")

    if model:
        agent = _agent_with_override(agent, model=model)

    # Pilot: resolve the master-corpus alias to its backing source corpus.
    resolved_project_id = project_id
    if project_id == store.MASTER_CORPUS_PROJECT_ID:
        resolved_project_id = store.MASTER_CORPUS_SOURCE_PROJECT_ID

    # PR #78 — smart_orchestrator routing gate. See agent_chat above for the
    # full design rationale. Run BEFORE we enter the StreamingResponse so the
    # redirect happens synchronously and the `route` event is the very first
    # thing the client sees (right after start).
    requested_agent_name = agent.name
    agent, routing = await select_agent_for_message(message, agent)

    async def event_stream():
        # Emit the routing decision as the first event so the UI can render
        # a "routed to heavy-reasoning because <reason>" badge. Always
        # emitted, even on pass-through, so the front-end has uniform
        # observability. Type is "route" (new) — clients that don't know
        # the event ignore it harmlessly per SSE conventions.
        yield f"data: {json.dumps({'type': 'route', **routing})}\n\n"

        # Reasoner-driven predefined dispatch: when the orchestrator picked a
        # real construction-container action, run it generically via the
        # predefined path (container.execute → rendered deliverable) instead of
        # the agent tool loop, which only hand-wires a couple of actions and
        # otherwise falls to a short chat answer. Gated by ORCHESTRATOR_PREDEFINED.
        try:
            from app.core.predefined_reasoning import lookup_question_hijack
            from app.routers.chat import (
                _predefined_enabled,
                _predefined_workflows,
                _stream_from_predefined,
            )
            _pd_action = routing.get("action")
            _use_predefined = bool(
                _pd_action and _predefined_enabled()
                and _pd_action in _predefined_workflows()
                # A caller who explicitly addressed a SPECIALIST agent chose
                # it deliberately (the agent-picker contract); a keyword in
                # their message must not hand the whole turn to a predefined
                # deliverable flow. Live find 2026-08-15 (F24): a costing
                # request to quantity-surveyor containing the word "manpower"
                # was answered by the canned histogram plan in 3 seconds,
                # three sessions in a row. Predefined interception is for
                # the generalist entry points only.
                and _predefined_may_intercept(requested_agent_name)
                # Low-confidence keyword routes must not intercept lookup
                # questions — they belong to the RAG-grounded agent path
                # (see lookup_question_hijack docstring for the live repro).
                and not lookup_question_hijack(
                    message, float(routing.get("confidence") or 0.0)
                )
            )
        except Exception:  # noqa: BLE001 — never block the normal agent path
            _pd_action, _use_predefined = None, False

        if _use_predefined:
            try:
                _pd_params = dict(body.get("params") or {})
                if not _pd_params.get("duration_overrides"):
                    from app.lib.wbs_duration_overrides import collect_overrides
                    _pd_params["duration_overrides"] = collect_overrides(
                        message, history=history,
                    ) or None
                async for chunk in _stream_from_predefined(
                    action=_pd_action,
                    user_message=message,
                    project_id=resolved_project_id,
                    user_id=auth["user_id"],
                    session_id=conversation_id or f"ag-{auth['user_id']}",
                    document_ids=body.get("document_ids") or [],
                    params=_pd_params,
                    emit_start=True,
                    conversation_id=conversation_id,
                ):
                    yield chunk
                    await asyncio.sleep(0)
                return
            except Exception:
                logger.exception("predefined dispatch failed for %s; falling back", _pd_action)

        try:
            async for evt in agent.chat_stream(
                message,
                history=history,
                project_id=resolved_project_id,
                conversation_id=conversation_id,
                user_id=auth["user_id"],
                rag_debug=rag_debug,
            ):
                # Same net as /v1/chat/stream: an error naming the plumbing
                # (provider, HTTP status, upstream body) is replaced for the user.
                if evt.get("type") == "token":
                    turn_timing.first_token()
                yield sanitize_error_frame(f"data: {json.dumps(evt, default=str)}\n\n")
                await asyncio.sleep(0)  # yield to the event loop
        except Exception as e:
            # Log the real error server-side; stream only a generic message so
            # infra detail (e.g. the LLM tunnel URL in a ConnectError) never
            # reaches the browser.
            logger.warning("agent chat stream failed: %s", e)
            yield f"data: {json.dumps({'type': 'error', 'message': 'The assistant is temporarily unavailable. Please try again.'})}\n\n"

    # Local import: keeps this router's import graph unchanged for the
    # non-streaming endpoints.
    from app.routers.exit_frames import with_answer_exit
    from app.routers.hat_frames import with_hat_signals

    from app.core import turn_gate, turn_progress
    with turn_timing.stage("queue"):
        _release_turn = await turn_gate.acquire()
    return StreamingResponse(
        turn_gate.hold_until_done(turn_progress.opened(
        with_hat_signals(with_answer_exit(event_stream()), message)), _release_turn),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _agent_with_override(base_agent, **overrides):
    """Return a shallow copy of the agent with the given fields overridden."""
    from dataclasses import replace
    return replace(base_agent, **overrides)
