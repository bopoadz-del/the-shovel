"""What a turn's model call is given: the messages and named inputs.

Moved from the Agent class in app/agents/runtime.py by
scripts/move_agent_methods.py (F-DRIVER Phase A). Each function is still
an Agent method (the class binds it); runtime names are read at call
time, so this module does not import runtime when it is imported.
"""
from __future__ import annotations

from typing import Any


async def _fetch_named_missing_input(
    self,
    final_text: str,
    messages: list[dict[str, Any]],
    *,
    user_message: str,
    project_id: str | None,
    api_key: str | None,
    user_id: str | None,
) -> tuple[str, str | None]:
    """ONE retrieval for an input the answer named as missing (item 4).

    A delay-damages daily-amount ask answered "do not state the specific
    monetary rate per calendar day" while the next question in the same
    session and the same corpus retrieved the Accepted Contract Amount to
    complete its arithmetic. The first answer identified its missing input
    and did not go and get it.

    Bounded means bounded: one retrieval, one re-ask, and every failure
    path returns the ORIGINAL answer. A search always returns something,
    so the answer is only revisited when what came back actually carries
    the named terms -- otherwise a correct refusal would be talked out
    of itself by whatever the index happened to rank first.
    """
    from app.agents.runtime import (  # noqa: F401 -- read at call time
        _LOG, _MISSING_INPUT_NUDGE, _final_text_needs_forced_retry, _off_loop,
        _rag_inject_off_process, _sanitize_final_text,
    )
    from app.agents.missing_input import (
        enabled as _mi_enabled,
        fetch_query as _mi_query,
        fetched_context_supports as _mi_supports,
        names_missing_input as _mi_names,
    )

    from app.core.rag.inject import BUDGET_PATH_MISSING_INPUT

    if not _mi_enabled() or not project_id or not (final_text or "").strip():
        return final_text, None
    missing = _mi_names(final_text)
    if not missing:
        return final_text, None
    try:
        sys_msg, _ = await _rag_inject_off_process(
            user_message=_mi_query(missing, user_message),
            project_id=project_id,
            # No conversation_id: this is a targeted lookup, not a turn,
            # and it must not enter the follow-up context of the next one.
            conversation_id=None,
            user_id=user_id,
            agent_name=self.name,
            history=None,
            budget_path=BUDGET_PATH_MISSING_INPUT,
        )
    except Exception:  # noqa: BLE001 - a failed extra fetch is not a failed turn
        _LOG.warning("missing-input fetch failed for %r", missing, exc_info=True)
        return final_text, None

    body = (sys_msg or {}).get("content") or ""
    if not _mi_supports(body, missing):
        _LOG.info(
            "missing-input fetch for %r returned nothing carrying it; "
            "the answer stands as written", missing,
        )
        return final_text, None

    follow = list(messages) + [
        {"role": "assistant", "content": final_text},
        sys_msg,
        {"role": "user", "content": _MISSING_INPUT_NUDGE % missing},
    ]
    resp = await self._call_llm(
        follow, api_key, project_id=project_id, with_tools=False, user_id=user_id,
    )
    if resp.get("status") == "error":
        return final_text, None
    retry = _sanitize_final_text(
        (resp["choice"].get("message") or {}).get("content") or "",
        messages=follow, tool_results=[],
    )
    if not retry.strip() or (await _off_loop(_final_text_needs_forced_retry, retry, user_message=user_message
    )):
        return final_text, None
    _LOG.info("missing-input fetch answered %r; answer revised", missing)
    return retry, missing


#: Tool name of the project search. One constant so the guard below and the
#: tool roster cannot disagree on it.
SEARCH_TOOL_NAME = "search_project_documents"

#: Sent after the search the guard ran. Says what happened in the turn and
#: what may follow; it names no document, value or question.
SEARCH_FIRST_NUDGE = (
    "The excerpts handed with the question did not state what was asked, so "
    "the project documents were searched for it (result above). Answer the "
    "question from those rows if they state it, and cite the document they "
    "came from. If they do not state it either, say plainly that it was not "
    "found and name what was searched for. Do not ask the user for a "
    "document name to search with."
)


def search_before_refusal_due(
    final_text: str,
    *,
    has_project: bool,
    search_calls: int,
    search_cap: int,
    excluded_tools: set | frozenset | None,
    force_synthesis: bool,
    already_ran: bool,
    iterations_left: int,
) -> bool:
    """May this draft answer end the turn, or must the project be searched first?

    S-R8: a first-pass miss ended as a refusal because nothing in the turn
    ever ran the search the model had. The decision reads only turn state
    plus one pure text test, and is bounded to ONE search round per turn:

      * the draft declines for lack of the asked thing, offers the user a
        search, or promises one it did not make;
      * the turn belongs to a project, the search tool is still offered
        (not excluded, not spent: ``search_calls < search_cap``), and the
        turn is not in forced synthesis;
      * the round has not already run this turn, and there is an iteration
        left to answer from its rows.

    Whether the project has indexed documents is checked by the caller only
    when this returns True, because it costs a round trip.
    """
    from app.agents.missing_input import declines_for_lack
    from app.agents.runtime import _looks_like_search_preamble

    if already_ran or force_synthesis or not has_project:
        return False
    if iterations_left < 2:
        return False
    if search_calls >= search_cap or SEARCH_TOOL_NAME in (excluded_tools or ()):
        return False
    return bool(declines_for_lack(final_text) or _looks_like_search_preamble(final_text))


def search_before_refusal_call(final_text: str, user_message: str, ordinal: int) -> dict[str, Any]:
    """The search the guard runs: the thing the draft named as missing leads,
    the question's own terms follow (missing_input.fetch_query). When the
    draft named nothing searchable, the question itself is the query."""
    import json

    from app.agents.missing_input import fetch_query, names_missing_input

    missing = names_missing_input(final_text)
    query = fetch_query(missing, user_message) if missing else (user_message or "").strip()
    return {
        "id": f"call_search_first_{ordinal}",
        "type": "function",
        "function": {
            "name": SEARCH_TOOL_NAME,
            "arguments": json.dumps({"query": query[:500]}),
        },
    }


async def _search_before_refusal(
    self,
    raw_content: str,
    messages: list[dict[str, Any]],
    state: dict[str, Any],
    *,
    user_message: str,
    project_id: str | None,
    tool_results: list[Any],
    search_calls: int,
    search_cap: int,
    excluded_tools: set,
    force_synthesis: bool,
    iterations_left: int,
) -> dict[str, Any] | None:
    """Shared by the streaming and non-streaming loops so they cannot drift.

    Returns the search tool call to run instead of finalising the draft, or
    None when the draft may stand (or be handled by the existing retries).
    ``state`` is the turn's own dict: it records that the round ran (so it
    runs once) and that its nudge is owed after the tool result.
    """
    from app.agents.runtime import (  # noqa: F401 -- read at call time
        _LOG, _off_loop, _project_is_rag_ready_off_process, _sanitize_final_text,
    )

    draft = _sanitize_final_text(raw_content, messages=messages, tool_results=tool_results)
    if not search_before_refusal_due(
        draft,
        has_project=bool(project_id),
        search_calls=search_calls,
        search_cap=search_cap,
        excluded_tools=excluded_tools,
        force_synthesis=force_synthesis,
        already_ran=bool(state.get("ran")),
        iterations_left=iterations_left,
    ):
        return None
    try:
        if not await _project_is_rag_ready_off_process(project_id):
            return None
    except Exception:  # noqa: BLE001 - unknown readiness: let the draft stand
        _LOG.warning("search-before-refusal readiness check failed", exc_info=True)
        return None
    _LOG.info("search-before-refusal: draft declines with search unused; searching")
    state["ran"] = state["nudge"] = True
    return search_before_refusal_call(draft, user_message, search_calls + 1)


def _search_first_nudges(self, state: dict[str, Any]) -> list[dict[str, Any]]:
    """The instruction owed after the guard's search result, once."""
    if not state.get("nudge"):
        return []
    state["nudge"] = False
    # The platform wrote this bubble, not the operator: it carries the
    # platform marker so every operator-text reader (_latest_operator_ask,
    # _operator_user_text, the provenance tool-run parser) skips it.
    from app.agents.runtime import _PREDISPATCH_PREFIX

    return [{
        "role": "user",
        "content": f"{_PREDISPATCH_PREFIX} {SEARCH_TOOL_NAME} ran. {SEARCH_FIRST_NUDGE}",
    }]


def _intent_hard_excludes(self, messages: list[dict[str, Any]]) -> tuple[set, str]:
    """Tools hard-excluded on the operator's draft intent, and that ask.

    Moved verbatim from the two turn loops (streaming and non-streaming),
    which carried identical copies. Live tip 0a95d03: VO ask2 stole to
    construction/sympy when predispatch missed -- hard-exclude on draft
    intent alone. Each exclusion fails on its own, as before.
    """
    from app.agents.runtime import (  # noqa: F401 -- read at call time
        _LOG, _inline_boq_hard_excludes, _latest_operator_ask, _vo_draft_hard_excludes,
    )
    excluded: set = set()
    op = ""
    try:
        op = _latest_operator_ask(messages) or ""
        excluded |= _vo_draft_hard_excludes(op)
    except Exception:  # noqa: BLE001
        _LOG.debug("vo hard-exclude skipped", exc_info=True)
    try:
        op = _latest_operator_ask(messages) or ""
        excluded |= _inline_boq_hard_excludes(op)
    except Exception:  # noqa: BLE001
        _LOG.debug("inline-boq hard-exclude skipped", exc_info=True)
    return excluded, op


# ── Internals ─────────────────────────────────────────────────────────
def _build_messages(
    self,
    user_message: str,
    history: list[dict[str, str]],
    project_id: str | None = None,
) -> list[dict[str, Any]]:
    from app.agents.runtime import (  # noqa: F401 -- read at call time
        MAX_HISTORY_TURNS, _CM_INJECT_AGENTS, _cm_prompt_fragment_for_turn,
    )
    msgs: list[dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]

    # Project context — facts + document listing — as a second system message.
    if project_id:
        try:
            from app.core.project_memory import build_project_context
            ctx = build_project_context(project_id, user_message)
        except Exception:
            ctx = ""
        if ctx:
            msgs.append({"role": "system", "content": ctx})

    # Remembered agent facts — durable across conversations, scoped to
    # this project so one project's facts never leak into another.
    try:
        from app.core import agent_memory
        facts = agent_memory.list_agent_facts(self.name, project_id)
    except Exception:
        facts = []
    if facts:
        lines = ["Known facts (you remembered):"]
        for f in facts:
            lines.append(f"- {f['key']}: {f['value']}")
        msgs.append({"role": "system", "content": "\n".join(lines)})

    # CM cross-domain inject (compact metadata; preserves RAG grounding).
    if self.name in _CM_INJECT_AGENTS:
        cm_fragment = _cm_prompt_fragment_for_turn(user_message)
        if cm_fragment:
            msgs.append({"role": "system", "content": cm_fragment})

    for turn in (history or [])[-MAX_HISTORY_TURNS:]:
        role = (turn.get("role") or "user").lower()
        if role not in ("user", "assistant"):
            continue
        content = (turn.get("content") or "")[:8000]
        if not content:
            continue
        msgs.append({"role": role, "content": content})
    msgs.append({"role": "user", "content": user_message})
    return msgs
