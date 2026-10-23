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
