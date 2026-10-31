"""The agent's turn: the non-streaming loop and the streaming entry point.

Moved from the Agent class in app/agents/runtime.py by
scripts/move_agent_methods.py (F-DRIVER Phase A). Each function is still
an Agent method (the class binds it); runtime names are read at call
time, so this module does not import runtime when it is imported.
"""
from __future__ import annotations

import asyncio  # noqa: F401 -- annotations

from typing import Any, AsyncIterator, Awaitable, Callable


def _log_turn_timing(marks: dict, conversation_id: str | None, steps: dict | None = None) -> None:
    """One line per streamed turn: milliseconds from receipt to each mark, and
    the tool-step counts (tool_call / tool_result frames before the first
    token). Numbers only -- no payloads, no names (ORDERS_0410Z long-ask
    tail)."""
    import json as _json
    import logging as _logging
    import time as _time

    log = _logging.getLogger("app.agents.core.loop")
    try:
        start = marks.get("received")
        if start is None:
            return
        out = {k: round((v - start) * 1000) for k, v in marks.items() if k != "received"}
        out["closed"] = round((_time.monotonic() - start) * 1000)
        if steps is not None:
            # ms from receipt to the first token = the user's wait before any
            # answer text; paired with how many tool steps ran before it.
            first_token = marks.get("first_token")
            out["before_first_token_ms"] = round((first_token - start) * 1000) if first_token else None
            out["tool_calls"] = steps.get("tool_calls", 0)
            out["tool_results"] = steps.get("tool_results", 0)
        log.info("TURN_TIMING %s", _json.dumps({"conversation": conversation_id or "", **out}))
    except Exception:  # noqa: BLE001 -- timing is evidence, never a failure
        log.debug("turn timing not logged", exc_info=True)


# ── Public chat API ───────────────────────────────────────────────────
async def chat(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """One turn. Identical retrievals within it are answered once
    (vector_store turn memo); see ``_chat_impl``."""
    import importlib

    from app.core.offload import off_loop

    # A large module: its first import (once per process) froze the loop for
    # 125-188 ms at 15 users. Already-imported modules return at once.
    answer_exit = await off_loop(importlib.import_module, "app.agents.answer_exit")
    from app.core.rag.vector_store import end_turn_memo, start_turn_memo

    token = start_turn_memo()
    turn, turn_token = answer_exit.begin_turn()
    from app.lib import construction_formulas as _calc_words

    # The user's own words reach every calculator run this turn makes.
    ask_token = _calc_words.set_current_ask(_user_message_of(self, args, kwargs))
    try:
        result = await self._chat_impl(*args, **kwargs)
        return await off_loop(answer_exit.check_result, result, turn)
    finally:
        _calc_words._CURRENT_ASK.reset(ask_token)
        answer_exit.reset_turn(turn_token)
        end_turn_memo(token)


def _user_message_of(agent: Any, args: tuple, kwargs: dict) -> str:
    """The user turn a ``chat`` call carries, by name or by position."""
    import inspect

    try:
        bound = inspect.signature(agent._chat_impl).bind_partial(*args, **kwargs)
    except (TypeError, ValueError):
        return str(kwargs.get("user_message") or "")
    value = bound.arguments.get("user_message")
    return value if isinstance(value, str) else ""


async def _chat_impl(
    self,
    user_message: str,
    history: list[dict[str, str]] | None = None,
    api_key: str | None = None,
    project_id: str | None = None,
    conversation_id: str | None = None,
    on_event: Callable[[str, dict[str, Any]], None | Awaitable[None]] | None = None,
    user_id: str | None = None,
    _depth: int = 0,
    _call_stack: list[str] | None = None,
) -> dict[str, Any]:
    """Single round-trip: returns {answer, tool_calls, history}.

    Optional new params (all default to today's behavior when omitted):
    - ``project_id`` — inject project facts/docs and expose document search.
    - ``conversation_id`` — load + persist conversation memory.
    - ``on_event`` — async/sync callback fired during the tool-call loop.
      Receives ``(event_name, payload)`` where event_name is one of:
        * ``"iteration"`` — ``{"n": int}`` at the top of each loop turn.
        * ``"tool_call"`` — ``{"name": str, "args": dict, "id": str}``
          fired immediately BEFORE the tool runs.
        * ``"tool_result"`` — ``{"name": str, "id": str, "ok": bool,
          "duration_ms": int, "error": str?}`` fired AFTER the tool runs.
        * ``"final"`` — ``{"answer": str}`` fired once when the agent
          produces a non-tool-call assistant message.
      The chat router uses this to emit SSE events to the browser so
      the user sees a live reasoning trace instead of a 10-second
      spinner. Callback errors are swallowed; the loop never breaks
      because of an event handler.
    - ``_depth`` / ``_call_stack`` — internal, for inter-agent delegation.
    """
    from app.agents.runtime import (  # noqa: F401 -- read at call time
        MAX_TOOL_ITERATIONS, _EMPTY_RESPONSE_FALLBACK, _EMPTY_ROUTER_NUDGE, _LOG,
        _SEARCH_PREAMBLE_RETRY_NUDGE, _TOOL_ERROR_NUDGE_CAP, _TOOL_FORMAT_FALLBACK,
        _TOOL_FORMAT_RETRY_NUDGE, _UNINDEXED_PROJECT_MESSAGE, _apply_hat_activation,
        _apply_rag_context, _build_capability_answer, _build_exports_from_audit,
        _build_missing_reference_answer, _build_sources_from_audit,
        _calc_input_question, _compose_boq_scope_wbs_answer, _compose_excerpt_boq_instead_of_retry,
        _conflicting_tools_after_predispatch, _empty_router_verdict, _file_tool_hint,
        _final_text_needs_forced_retry, _forced_retry_nudge, _fulfill_answer_report, _has_unread_windows,
        _inline_boq_hard_excludes, _is_capability_request, _latest_operator_ask, _llm_config,
        _looks_like_search_preamble, _message_wants_locked_deliverable,
        _normalize_tool_call_ids, _nudge_for_failed_tool, _off_loop, _parse_dsml_tool_calls,
        _postprocess_answer, _predispatch_file_tool, _predispatch_formula_calc,
        _predispatch_look_ahead, _predispatch_remaining_deliverables,
        _predispatch_resource_histogram, _predispatch_wbs_duration_override,
        _predispatch_wir_form, _project_has_non_rag_context, _project_is_rag_ready_off_process,
        _rag_inject_off_process, _recover_answer_from_tool_messages,
        _recover_tool_calls_from_content, _recovered_calls_are_search_only,
        _requested_char_offset, _sanitize_citation_labels, _sanitize_final_text,
        _sanitize_inline_paths, _scrub_history, _should_force_synthesis,
        _should_short_circuit_delay_damages_daily, _should_short_circuit_part_summary,
        _should_short_circuit_priced_boq, _should_short_circuit_rag_miss,
        _text_needs_tool_recovery, _tool_result_content, _tool_result_errored,
        _vo_draft_hard_excludes, _vo_draft_ready_for_synthesis, answer_report_export_enabled,
        fulfill_wbs_export, inspect, json, message_wants_answer_report,
        message_wants_wbs_export, os, time,
    )
    async def _emit(name: str, payload: dict[str, Any]) -> None:
        if on_event is None:
            return
        try:
            res = on_event(name, payload)
            if inspect.isawaitable(res):
                await res
        except Exception:
            # Event handler must never break the agent loop.
            _LOG.warning(
                "swallowed %s in _emit() — continuing",
                "Exception", exc_info=True,
            )

    # H1: conversation-answer docx. Before the API-key check — no LLM.
    if message_wants_answer_report(user_message) and answer_report_export_enabled():
        answer, exports = _fulfill_answer_report(
            user_message, project_id, conversation_id, history, self.name,
            owner_id=user_id,
        )
        await _emit("final", {"answer": answer})
        return {
            "status": "success",
            "answer": answer,
            "tool_calls": [],
            "iterations": 0,
            "messages": [{"role": "assistant", "content": answer}],
            "sources": [],
            "exports": exports,
        }

    # Export the conversation's staged WBS, not a new scaffold.
    if message_wants_wbs_export(user_message):
        answer, exports = fulfill_wbs_export(
            user_message, project_id, conversation_id, self.name,
            owner_id=user_id,
        )
        await _emit("final", {"answer": answer})
        return {
            "status": "success",
            "answer": answer,
            "tool_calls": [],
            "iterations": 0,
            "messages": [{"role": "assistant", "content": answer}],
            "sources": [],
            "exports": exports,
        }

    cfg = _llm_config()
    # A provider with no env_key (none today) would need no auth — the
    # empty-bearer guard downstream omits the header. DeepSeek/OpenRouter
    # both declare an env_key, so this check requires their API key.
    if cfg["env_key"]:
        api_key = api_key or os.getenv(cfg["env_key"])
        if not api_key:
            return {
                "status": "error",
                "error": f"No {cfg['env_key']} configured. Set it in .env or pass via env.",
            }
    else:
        api_key = api_key or ""

    _call_stack = _call_stack or [self.name]

    # Zero-chunk project guardrail: refuse before spending any LLM budget
    # unless the project has other (non-RAG) context such as facts.
    if (
        project_id
        and not (await _project_is_rag_ready_off_process(project_id))
        and not (await _off_loop(_project_has_non_rag_context, project_id, user_message))
    ):
        if conversation_id:
            from app.core import agent_memory
            (await _off_loop(agent_memory.get_or_create_conversation, conversation_id, self.name, project_id, owner_id=user_id))
            (await _off_loop(agent_memory.append_message, conversation_id, "user", user_message))
            (await _off_loop(agent_memory.append_message, conversation_id, "assistant", _UNINDEXED_PROJECT_MESSAGE))
        return {
            "status": "success",
            "answer": _UNINDEXED_PROJECT_MESSAGE,
            "tool_calls": [],
            "iterations": 0,
            "messages": [],
            "sources": [],
        }

    effective_history = list(history or [])
    if conversation_id:
        from app.core import agent_memory
        (await _off_loop(agent_memory.get_or_create_conversation, conversation_id, self.name, project_id, owner_id=user_id))
        prior = (await _off_loop(agent_memory.get_messages, conversation_id))
        prior_turns = [
            {"role": m["role"], "content": m["content"]}
            for m in prior
            if m.get("role") in ("user", "assistant")
        ]
        effective_history = prior_turns + effective_history
        # Persist the user turn up front so it survives even if the LLM
        # call errors mid-loop — otherwise the conversation history loses
        # the question and ends up inconsistent.
        (await _off_loop(agent_memory.append_message, conversation_id, "user", user_message))

    # Strip prior hallucinated WBS/BOQ tables from history before
    # sending. Prevents the model from pattern-matching to a prior
    # (often fabricated) table when it should be calling the tool.
    effective_history = _scrub_history(effective_history)

    messages = (await _off_loop(self._build_messages, user_message, effective_history, project_id=project_id))
    # Pre-iter-0 RAG injection. Runs for any project-scoped turn so that
    # routing to heavy-reasoning (or another agent) does not strip project
    # grounding. Adds a system message AFTER the prompt + project context
    # but BEFORE the latest user turn.
    # In a worker thread: rag_inject is synchronous (embedding + SQL +
    # rerank) and froze the single worker's event loop for its whole
    # duration -- live, one turn stalled /livez for 4.2 s. to_thread
    # copies contextvars, so the caller-role gate still applies.
    _rag_sys_msg, _rag_audit = await _rag_inject_off_process(
        user_message=user_message,
        project_id=project_id,
        conversation_id=conversation_id,
        user_id=user_id,
        agent_name=self.name,
        history=effective_history,
    )
    if _rag_sys_msg and _rag_sys_msg.get("content"):
        _apply_rag_context(
            messages, _rag_sys_msg,
            user_data_authoritative=self.user_data_authoritative,
        )

    _apply_hat_activation(messages, user_message, self.name)

    # Capability / self-introspection short-circuit: a question ABOUT this
    # agent (its tools, abilities, accessible documents) is answered from the
    # REAL tool roster + REAL document list, not from matched document text.
    if _is_capability_request(user_message):
        answer = _build_capability_answer(self, project_id)
        if conversation_id:
            from app.core import agent_memory
            (await _off_loop(agent_memory.append_message, conversation_id, "assistant", answer))
        await _emit("final", {"answer": answer})
        return {
            "status": "success",
            "answer": answer,
            "tool_calls": [],
            "iterations": 0,
            "messages": messages + [{"role": "assistant", "content": answer}],
            "sources": [],
        }

    # Deterministic file pre-dispatch BEFORE the RAG-miss short-circuit.
    # A timestamped ``<name>_spec_….docx`` the user names looks
    # like an identifier, retrieval misses (file not indexed), and the
    # canned "could not confirm this reference" used to fire in ~1s
    # without ever fetching bytes from disk.
    tool_calls_made: list[dict[str, Any]] = []
    _locked = (await _off_loop(_message_wants_locked_deliverable, user_message))
    _pre = None
    if not _locked:
        _pre = await _predispatch_file_tool(self, messages, project_id)
    if _pre:
        tool_calls_made.append(_pre)
    _wbs_pre = await _predispatch_wbs_duration_override(
        self, messages, project_id,
    )
    if _wbs_pre:
        tool_calls_made.append(_wbs_pre)
    _hist_pre = await _predispatch_resource_histogram(
        self, messages, project_id,
    )
    if _hist_pre:
        tool_calls_made.append(_hist_pre)
    _la_pre = await _predispatch_look_ahead(
        self, messages, project_id,
    )
    if _la_pre:
        tool_calls_made.append(_la_pre)
    _wir_pre = None
    if not _locked:
        _wir_pre = await _predispatch_wir_form(
            self, messages, project_id, operator_text=user_message,
        )
    if _wir_pre:
        tool_calls_made.append(_wir_pre)
    _more_pre = None
    if not _hist_pre:
        _more_pre = await _predispatch_remaining_deliverables(
            self, messages, project_id, operator_text=user_message,
            conversation_id=conversation_id,
        )
    if _more_pre:
        tool_calls_made.append(_more_pre)
    _calc_pre = None
    if not _locked:
        _calc_pre = await _predispatch_formula_calc(
            self, messages, project_id, operator_text=user_message,
        )
    if _calc_pre:
        tool_calls_made.append(_calc_pre)

    # Fast path: exact reference miss with no RAG context. Skip when a
    # named project file was already fetched/extracted from disk.
    if not _pre and not _wbs_pre and not _hist_pre and not _wir_pre and not _more_pre and not _calc_pre and (await _off_loop(_should_short_circuit_rag_miss, _rag_audit, _rag_sys_msg, user_message
    )):
        answer = _build_missing_reference_answer(project_id, user_id)
        if conversation_id:
            from app.core import agent_memory
            (await _off_loop(agent_memory.append_message, conversation_id, "assistant", answer))
        await _emit("final", {"answer": answer})
        return {
            "status": "success",
            "answer": answer,
            "tool_calls": [],
            "iterations": 0,
            "messages": messages + [{"role": "assistant", "content": answer}],
            "sources": [],
        }
    _has_pre = bool(
        _pre or _wbs_pre or _hist_pre or _wir_pre or _more_pre or _calc_pre
    )
    # Leftover F1: a BOQ-derived generate_wbs draft is the answer.
    # Skip the provider hop so AB-2022 CoC excerpts cannot refuse
    # the turn. Other deliverables still keep the LLM. A calculator that
    # refused a supplied value answers with its question for that input.
    _boq_wbs_fast = (_compose_boq_scope_wbs_answer(_more_pre, user_message)
                     or _calc_input_question(_calc_pre))
    if _boq_wbs_fast:
        answer = (await _off_loop(_postprocess_answer, _boq_wbs_fast, _rag_sys_msg, messages,
            fallback_used=bool(_rag_audit.get("fallback_used")),
            agent_name=self.name,
            project_id=project_id,
            audit_rec=_rag_audit,
        ))
        if conversation_id:
            from app.core import agent_memory
            (await _off_loop(agent_memory.append_message, conversation_id, "assistant", answer))
        await _emit("final", {"answer": answer})
        return {
            "status": "success",
            "answer": answer,
            "tool_calls": tool_calls_made,
            "iterations": 0,
            "messages": messages + [{"role": "assistant", "content": answer}],
            "sources": [],
        }
    # Delay-damages daily amount before the priced-BOQ path: rate × ACA
    # is already in the excerpts / loaded CD volume. Skip the provider hop so a
    # priced-BOQ refuse cannot close the turn. Predispatch
    # deliverables keep the LLM.
    _daily_damages_fast = (await _off_loop(_should_short_circuit_delay_damages_daily, _rag_sys_msg, messages,
        has_predispatch=_has_pre,
        project_id=project_id,
        audit_rec=_rag_audit,
    ))
    if _daily_damages_fast:
        answer = (await _off_loop(_postprocess_answer, _daily_damages_fast, _rag_sys_msg, messages,
            fallback_used=bool(_rag_audit.get("fallback_used")),
            agent_name=self.name,
            project_id=project_id,
            audit_rec=_rag_audit,
        ))
        if conversation_id:
            from app.core import agent_memory
            (await _off_loop(agent_memory.append_message, conversation_id, "assistant", answer))
        await _emit("final", {"answer": answer})
        return {
            "status": "success",
            "answer": answer,
            "tool_calls": [],
            "iterations": 0,
            "messages": messages + [{"role": "assistant", "content": answer}],
            "sources": (await _off_loop(_build_sources_from_audit, _rag_audit, answer, tool_calls_made)),
            "provenance": (_rag_audit or {}).get("provenance") or [],
        }
    # WAVE 2 B4: priced D599.5 is already in the excerpts. Skip the
    # provider hop so a transient OpenRouter / unavailable banner
    # cannot empty the turn. Predispatch deliverables keep the LLM.
    _priced_fast = (await _off_loop(_should_short_circuit_priced_boq, _rag_sys_msg, messages,
        has_predispatch=_has_pre,
    ))
    if _priced_fast:
        answer = (await _off_loop(_postprocess_answer, _priced_fast, _rag_sys_msg, messages,
            fallback_used=bool(_rag_audit.get("fallback_used")),
            agent_name=self.name,
            project_id=project_id,
            audit_rec=_rag_audit,
        ))
        if conversation_id:
            from app.core import agent_memory
            (await _off_loop(agent_memory.append_message, conversation_id, "assistant", answer))
        await _emit("final", {"answer": answer})
        return {
            "status": "success",
            "answer": answer,
            "tool_calls": [],
            "iterations": 0,
            "messages": messages + [{"role": "assistant", "content": answer}],
            "sources": (await _off_loop(_build_sources_from_audit, _rag_audit, answer, tool_calls_made)),
            "provenance": (_rag_audit or {}).get("provenance") or [],
        }
    # BOQ page Part Summary total already in the excerpts.
    _part_fast = (await _off_loop(_should_short_circuit_part_summary, _rag_sys_msg, messages,
        has_predispatch=_has_pre,
    ))
    if _part_fast:
        answer = (await _off_loop(_postprocess_answer, _part_fast, _rag_sys_msg, messages,
            fallback_used=bool(_rag_audit.get("fallback_used")),
            agent_name=self.name,
            project_id=project_id,
            audit_rec=_rag_audit,
        ))
        if conversation_id:
            from app.core import agent_memory
            (await _off_loop(agent_memory.append_message, conversation_id, "assistant", answer))
        await _emit("final", {"answer": answer})
        return {
            "status": "success",
            "answer": answer,
            "tool_calls": [],
            "iterations": 0,
            "messages": messages + [{"role": "assistant", "content": answer}],
            "sources": (await _off_loop(_build_sources_from_audit, _rag_audit, answer, tool_calls_made)),
            "provenance": (_rag_audit or {}).get("provenance") or [],
        }
    # Root fix for the tool-loop (mirrors chat_stream): cap explicit
    # search_project_documents calls, then stop offering the tool so the
    # model answers from injected context instead of grinding to the cap.
    SEARCH_TOOL_CAP = int(os.getenv("AGENT_SEARCH_TOOL_CAP", "2"))
    search_calls = 0
    excluded_tools: set = set()
    # Live tip 0a95d03: VO ask2 stole to construction/sympy when
    # predispatch missed — hard-exclude on draft intent alone.
    try:
        _op = _latest_operator_ask(messages) or ""
        excluded_tools |= _vo_draft_hard_excludes(_op)
    except Exception:  # noqa: BLE001
        _LOG.debug("vo hard-exclude skipped", exc_info=True)
    try:
        _op = _latest_operator_ask(messages) or ""
        excluded_tools |= _inline_boq_hard_excludes(_op)
    except Exception:  # noqa: BLE001
        _LOG.debug("inline-boq hard-exclude skipped", exc_info=True)
    # Once a deliverable (non-search) tool returns, force a tool-free
    # synthesis call — see the streaming loop for the full rationale (stops
    # the tool-loop and the Groq large-context tool_use_failed/429 hang).
    # Live M1/M2/M6/M9/M13/M14: remaining already drafted the artifact,
    # but force_synthesis stayed False so payment_certificate /
    # drawing_qto / wir_form / commissioning_checklist overwrote it.
    force_synthesis = bool(_wbs_pre or _hist_pre or _wir_pre or _more_pre)
    for _hit in (_wbs_pre, _hist_pre, _wir_pre, _more_pre):
        if _hit:
            excluded_tools |= _conflicting_tools_after_predispatch(
                str(_hit.get("name") or "")
            )
    # WAVE 2 B4: priced D599.5 is already in RAG. Offering
    # boq_processor here starts a 28 MB scan and never writes
    # quantity + amount (live 2ceef76 empty hang).
    if _compose_excerpt_boq_instead_of_retry("", _rag_sys_msg, messages):
        excluded_tools.add("boq_processor")
    error_nudges = 0
    _force_synth_enabled = os.getenv("AGENT_FORCE_SYNTHESIS", "1") != "0"

    for iteration in range(MAX_TOOL_ITERATIONS):
        await _emit("iteration", {"n": iteration + 1})
        resp = await self._call_llm(
            messages, api_key, project_id=project_id, user_id=user_id,
            exclude_tools=excluded_tools or None,
            with_tools=not force_synthesis,
        )
        if resp.get("status") == "error":
            err = resp.get("error") or _EMPTY_RESPONSE_FALLBACK
            recovered = _recover_answer_from_tool_messages(err, messages)
            if recovered != err and not (await _off_loop(_text_needs_tool_recovery, recovered)):
                final_text = _sanitize_inline_paths(
                    _sanitize_citation_labels(recovered)
                )
                final_text = (await _off_loop(_postprocess_answer, final_text, _rag_sys_msg, messages,
                    fallback_used=bool(_rag_audit.get("fallback_used")),
                    agent_name=self.name,
                    project_id=project_id,
                    audit_rec=_rag_audit,
                ))
                messages.append({"role": "assistant", "content": final_text})
                if conversation_id:
                    from app.core import agent_memory
                    (await _off_loop(agent_memory.append_message, conversation_id, "assistant", final_text,
                    ))
                await _emit("final", {"answer": final_text})
                return {
                    "status": "success",
                    "answer": final_text,
                    "tool_calls": tool_calls_made,
                    "iterations": iteration + 1,
                    "messages": messages,
                    "recovered_from_llm_error": True,
                    "sources": (await _off_loop(_build_sources_from_audit, _rag_audit, final_text, tool_calls_made)),
                    "provenance": (_rag_audit or {}).get("provenance") or [],
                    "exports": (await _off_loop(_build_exports_from_audit, _rag_audit, final_text, tool_calls_made,
                        conversation_id=conversation_id,
                    )),
                }
            return resp
        choice = resp["choice"]
        assistant_msg = choice.get("message") or {}

        tool_calls = assistant_msg.get("tool_calls") or []
        raw_content = assistant_msg.get("content") or ""

        # DeepSeek sometimes emits the tool call as inline DSML markup in
        # `content` with an empty structured `tool_calls` field. Groq
        # Llama-4-Scout sometimes leaks raw search args as JSON content.
        # Recover either shape before treating the turn as a final answer.
        if not tool_calls:
            cleaned_content, dsml_tool_calls = _parse_dsml_tool_calls(raw_content)
            recovered_tool_calls = (
                _recover_tool_calls_from_content(raw_content)
                if not dsml_tool_calls else []
            )
            # Leftover L4: after a deliverable, XML/DSML leaks must not
            # re-arm tools — graft the pipeline verdict instead.
            if force_synthesis and (dsml_tool_calls or recovered_tool_calls):
                dsml_tool_calls = []
                recovered_tool_calls = []
            # Once search is capped, do not re-execute the same
            # leaked search envelope for 12 iterations — synthesize instead.
            if (
                recovered_tool_calls
                and _recovered_calls_are_search_only(recovered_tool_calls)
                and (
                    "search_project_documents" in excluded_tools
                    or search_calls >= SEARCH_TOOL_CAP
                )
            ):
                recovered_tool_calls = []
            if dsml_tool_calls:
                # Treat this turn as a tool-calling turn.
                tool_calls = dsml_tool_calls
                assistant_msg = {
                    "role": "assistant",
                    "content": cleaned_content,
                    "tool_calls": dsml_tool_calls,
                }
            elif recovered_tool_calls:
                tool_calls = recovered_tool_calls
                assistant_msg = {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": recovered_tool_calls,
                }
            else:
                # Genuine final answer — sanitize DSML markup, raw tool JSON,
                # and empty content before it reaches the user.
                final_text = _sanitize_final_text(
                    raw_content, messages=messages, tool_results=tool_calls_made,
                )
                # If sanitization left nothing usable (empty or raw tool JSON
                # fallback), force one no-tools call so the model must produce
                # a plain-text answer instead of an empty bubble or leak.
                if (await _off_loop(_final_text_needs_forced_retry, final_text, user_message=user_message)):
                    priced = _compose_excerpt_boq_instead_of_retry(
                        final_text, _rag_sys_msg, messages,
                    )
                    if priced:
                        final_text = priced
                    else:
                        _nudge = await _off_loop(_forced_retry_nudge, final_text)
                        if _nudge:
                            messages.append({"role": "user", "content": _nudge})
                        forced_resp = await self._call_llm(messages, api_key, project_id=project_id, with_tools=False, user_id=user_id)
                        if forced_resp.get("status") == "error":
                            final_text = _EMPTY_RESPONSE_FALLBACK
                        else:
                            forced_msg = forced_resp["choice"].get("message") or {}
                            final_text = _sanitize_final_text(
                                forced_msg.get("content") or "",
                                messages=messages, tool_results=tool_calls_made,
                            )
                        if (await _off_loop(_final_text_needs_forced_retry, final_text, user_message=user_message)):
                            final_text = _EMPTY_RESPONSE_FALLBACK
                final_text = _sanitize_inline_paths(_sanitize_citation_labels(final_text))
                # An answer that names its own missing input gets ONE
                # bounded retrieval for it before it is final.
                final_text, _fetched_for = await self._fetch_named_missing_input(
                    final_text, messages, user_message=user_message,
                    project_id=project_id, api_key=api_key, user_id=user_id,
                )
                final_text = (await _off_loop(_postprocess_answer, final_text, _rag_sys_msg, messages, fallback_used=bool(_rag_audit.get("fallback_used")), agent_name=self.name, project_id=project_id, audit_rec=_rag_audit))
                messages.append({"role": "assistant", "content": final_text})
                if conversation_id:
                    from app.core import agent_memory
                    # User turn was already persisted up front.
                    (await _off_loop(agent_memory.append_message, conversation_id, "assistant", final_text))
                await _emit("final", {"answer": final_text})
                return {
                    "status": "success",
                    "answer": final_text,
                    "tool_calls": tool_calls_made,
                    "iterations": iteration + 1,
                    "messages": messages,
                    "sources": (await _off_loop(_build_sources_from_audit, _rag_audit, final_text, tool_calls_made)),
                    "provenance": (_rag_audit or {}).get("provenance") or [],
                    "exports": (await _off_loop(_build_exports_from_audit, _rag_audit, final_text, tool_calls_made, conversation_id=conversation_id)),
                }

        # Persist the assistant turn that contained the tool calls
        tool_calls = _normalize_tool_call_ids(list(tool_calls))
        assistant_msg = {**assistant_msg, "tool_calls": tool_calls}
        messages.append(assistant_msg)
        pending_nudges: list[dict[str, Any]] = []
        for tc in tool_calls:
            # Surface the tool call to the event stream BEFORE running it
            # so the UI can show "️ tool_name — running…" live.
            fn = tc.get("function") or {}
            tc_name = fn.get("name") or tc.get("name") or "unknown"
            if tc_name == "search_project_documents":
                search_calls += 1
                if search_calls >= SEARCH_TOOL_CAP:
                    excluded_tools.add("search_project_documents")
            tc_args_raw = fn.get("arguments") or tc.get("arguments") or "{}"
            try:
                tc_args = json.loads(tc_args_raw) if isinstance(tc_args_raw, str) else dict(tc_args_raw)
            except Exception:
                tc_args = {"_raw": str(tc_args_raw)[:200]}
            await _emit("tool_call", {
                "name": tc_name,
                "args": tc_args,
                "id": tc.get("id") or "",
            })
            _t0 = time.monotonic()
            tool_result = await self._run_tool_call(
                tc,
                api_key=api_key,
                project_id=project_id,
                conversation_id=conversation_id,
                _depth=_depth,
                _call_stack=_call_stack,
                user_message=user_message,
                history=effective_history,
            )
            duration_ms = int((time.monotonic() - _t0) * 1000)
            tool_calls_made.append(tool_result)
            # Determine ok/error by introspecting the tool's result payload.
            _inner = tool_result.get("result") if isinstance(tool_result, dict) else None
            ok = True
            err = None
            if isinstance(_inner, dict) and _inner.get("status") == "error":
                ok = False
                err = str(_inner.get("error") or "")[:200]
            _tool_content = _tool_result_content(
                {**(tool_result["result"] if isinstance(tool_result.get("result"), dict) else {"result": tool_result.get("result")}),
                 **({"validation": tool_result["validation"]} if "validation" in tool_result else {})},
                offset=_requested_char_offset(tc),
            )
            # Half a document is not a deliverable: locking synthesis here
            # leaves the model holding an instruction to read the next
            # window with tools already disarmed.
            if (
                _force_synth_enabled and ok
                and (await _off_loop(_should_force_synthesis, tool_result))
                and _vo_draft_ready_for_synthesis(_op, tool_result.get("name"))
                and not _has_unread_windows(_tool_content)
            ):
                force_synthesis = True
            await _emit("tool_result", {
                "name": tool_result.get("name", tc_name),
                "id": tc.get("id") or "",
                "ok": ok,
                "duration_ms": duration_ms,
                **({"error": err} if err else {}),
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id"),
                "name": tool_result["name"],
                "content": _tool_content,
            })
            if _empty_router_verdict(tool_result):
                pending_nudges.append({"role": "user", "content": _EMPTY_ROUTER_NUDGE})
            elif (_tool_result_errored(tool_result)
                  and error_nudges < _TOOL_ERROR_NUDGE_CAP):
                error_nudges += 1
                pending_nudges.append({"role": "user", "content": _nudge_for_failed_tool(
                    tool_result, self, project_id=project_id, messages=messages) + (await _off_loop(_file_tool_hint, messages, project_id, self.allowed_blocks))})
        messages.extend(pending_nudges)

    # Hit the cap without a final answer — force one more call with tools disabled
    # so the model is required to emit a plain-text summary.
    daily_damages_cap = (await _off_loop(_should_short_circuit_delay_damages_daily, _rag_sys_msg, messages, has_predispatch=False,
        project_id=project_id, audit_rec=_rag_audit,
    ))
    priced_cap = _compose_excerpt_boq_instead_of_retry("", _rag_sys_msg, messages)
    if daily_damages_cap:
        final_text = daily_damages_cap
    elif priced_cap:
        final_text = priced_cap
    else:
        forced_resp = await self._call_llm(messages, api_key, project_id=project_id, with_tools=False, user_id=user_id)
        if forced_resp.get("status") == "error":
            # Even the forced call failed; fall back to the original error shape.
            return {
                "status": "error",
                "error": f"Agent exceeded {MAX_TOOL_ITERATIONS} tool iterations without a final answer.",
                "tool_calls": tool_calls_made,
                "messages": messages,
            }
        forced_msg = forced_resp["choice"].get("message") or {}
        final_text = _sanitize_final_text(
            forced_msg.get("content") or "",
            messages=messages, tool_results=tool_calls_made,
        )
        if (await _off_loop(_final_text_needs_forced_retry, final_text, user_message=user_message)):
            final_text = _EMPTY_RESPONSE_FALLBACK
    final_text = _sanitize_inline_paths(_sanitize_citation_labels(final_text))
    final_text, _fetched_for = await self._fetch_named_missing_input(
        final_text, messages, user_message=user_message,
        project_id=project_id, api_key=api_key, user_id=user_id,
    )
    final_text = (await _off_loop(_postprocess_answer, final_text, _rag_sys_msg, messages, fallback_used=bool(_rag_audit.get("fallback_used")), agent_name=self.name, project_id=project_id, audit_rec=_rag_audit))
    messages.append({"role": "assistant", "content": final_text})
    if conversation_id:
        from app.core import agent_memory
        # User turn was already persisted up front.
        (await _off_loop(agent_memory.append_message, conversation_id, "assistant", final_text))
    return {
        "status": "success",
        "answer": final_text,
        "tool_calls": tool_calls_made,
        "iterations": MAX_TOOL_ITERATIONS,
        "messages": messages,
        "forced_final": True,
        "sources": (await _off_loop(_build_sources_from_audit, _rag_audit, final_text, tool_calls_made)),
        "provenance": (_rag_audit or {}).get("provenance") or [],
        "exports": (await _off_loop(_build_exports_from_audit, _rag_audit, final_text, tool_calls_made, conversation_id=conversation_id)),
    }


async def chat_stream(
    self,
    user_message: str,
    history: list[dict[str, str]] | None = None,
    api_key: str | None = None,
    user_id: str | None = None,
    project_id: str | None = None,
    conversation_id: str | None = None,
    rag_debug: bool = False,
    attached_documents: list[dict[str, Any]] | None = None,
    _depth: int = 0,
    _call_stack: list[str] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Generator: yields {type, ...} events. Types: start, hat_signals, tool_call, tool_result, token, end, error, heartbeat.

    Tool-calling is non-streamed (we collect the whole assistant turn before deciding),
    but the FINAL assistant answer streams token-by-token.

    **Emit guarantee (FOLLOW-UP #90):** every exit from this generator MUST
    emit either at least one ``token`` event OR a structured ``error`` event
    before a ``end`` event. Silent exits are bugs. The trailing safety net
    below converts any escaping exception or unhandled empty-content state
    into a synthetic ``error`` + ``end`` pair.

    **Wall-clock timeout + heartbeat (FOLLOW-UP #92):** a producer task
    runs ``_chat_stream_impl`` and pushes events into a queue; a heartbeat
    task injects ``{"type": "heartbeat"}`` after each ``CHAT_STREAM_HEARTBEAT_SECONDS``
    of silence; the consumer reads with an *absolute* wall-clock deadline
    of ``CHAT_STREAM_TIMEOUT_SECONDS`` (computed once, NOT reset by events),
    and emits a structured timeout error when the deadline expires before
    the producer finishes.
    """
    from app.agents.runtime import (  # noqa: F401 -- read at call time
        _EMPTY_RESPONSE_FALLBACK, _EmitLeakGuard, _LOG, _with_hat_signals, asyncio,
        hat_scores_sse, os, time,
    )
    agent_name = self.name
    token_emitted = False
    terminal_emitted = False  # True once we yield an `end` or `error` event
    # Floor scorer reads hat_signals on the stream (event + end metadata).
    # Compute once so early-return impl paths still record activation.
    #
    # Guarded, because this line runs BEFORE the first yield and outside
    # the last-line safety net below. Hat scoring walks the whole catalog
    # and is telemetry: if it raises here the generator dies with no
    # `error` event and the user gets a dead stream instead of an answer.
    # Losing the scores for one turn is the correct price of a bug in
    # scoring; losing the chat is not.
    try:
        _hat_evt = hat_scores_sse(user_message)
    except Exception:  # noqa: BLE001 - telemetry must never take down chat
        _LOG.exception("chat_stream: hat scoring failed; continuing without hat_signals")
        _hat_evt = None
    _hat_emitted = False
    # Tool names seen on the way out, so the synthetic `end` below can
    # still report them when the inner generator dies before its own end.
    tools_seen: list[str] = []

    # Read knobs at call-time so tests can monkeypatch env. Bad values
    # fall back to safe defaults rather than crashing the stream.
    try:
        timeout_s = float(os.getenv("CHAT_STREAM_TIMEOUT_SECONDS") or "240")
    except ValueError:
        timeout_s = 240.0
    try:
        heartbeat_s = float(os.getenv("CHAT_STREAM_HEARTBEAT_SECONDS") or "15")
    except ValueError:
        heartbeat_s = 15.0

    _SENTINEL = object()

    async def _inner():
        nonlocal token_emitted, terminal_emitted
        _LOG.info(
            "chat_stream: start agent=%s conv=%s project=%s timeout=%.1fs heartbeat=%.1fs",
            agent_name, conversation_id, project_id, timeout_s, heartbeat_s,
        )

        queue: asyncio.Queue = asyncio.Queue()

        # Absolute deadline — computed ONCE and NOT reset by events.
        # Heartbeats keep the queue active even when the upstream LLM is
        # hung, so a per-get() timeout would never trigger. The fixed
        # deadline is the only correct semantic for a wall-clock cap.
        # Computed here, before the producer, because the producer now
        # passes it down so LLM calls cannot outlive the turn.
        deadline = time.monotonic() + timeout_s

        # Shared, mutable: the producer writes what it is waiting on, the
        # consumer reads it when the wall clock runs out.
        phase: dict[str, Any] = {"name": "starting", "since": time.monotonic()}

        async def producer() -> None:
            # This task's own context: the turn's retrieval memo lives
            # and dies with it (vector_store._TURN_MEMO).
            from app.core.rag.vector_store import start_turn_memo
            start_turn_memo()
            from app.agents import driver
            try:
                if driver.enabled() and not _depth:
                    # Driver mode (F-DRIVER Phase B, DRIVER_MODE): the model
                    # drives the turn; same deadline, heartbeat and guarantee.
                    from app.agents.driver import loop as driver_loop
                    events = driver_loop.stream(
                        self, user_message=user_message, history=history,
                        project_id=project_id, conversation_id=conversation_id,
                        user_id=user_id, api_key=api_key,
                    )
                else:
                    events = self._chat_stream_impl(
                        user_message=user_message,
                        history=history,
                        api_key=api_key,
                        user_id=user_id,
                        project_id=project_id,
                        attached_documents=attached_documents,
                        conversation_id=conversation_id,
                        rag_debug=rag_debug,
                        _depth=_depth,
                        _call_stack=_call_stack,
                        _deadline=deadline,
                        _phase=phase,
                    )
                async for event in events:
                    await queue.put(event)
            finally:
                await queue.put(_SENTINEL)

        async def heartbeat() -> None:
            # Infinite loop — cancelled by the consumer's finally block.
            while True:
                await asyncio.sleep(heartbeat_s)
                await queue.put({"type": "heartbeat"})

        producer_task = asyncio.create_task(producer())
        heartbeat_task = asyncio.create_task(heartbeat())

        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    # Wall-clock cap exceeded. Emit a structured timeout
                    # error so the frontend's friendlyErrorMessage maps
                    # it (substring "timeout") to a clean banner.
                    _stalled = str(phase.get("name") or "unknown")
                    _stalled_for = time.monotonic() - float(
                        phase.get("since") or deadline
                    )
                    _LOG.warning(
                        "chat_stream: wall-clock deadline exceeded after "
                        "%.1fs — stalled in phase=%r for %.1fs",
                        timeout_s, _stalled, _stalled_for,
                    )
                    if not token_emitted:
                        yield {"type": "token", "content": _EMPTY_RESPONSE_FALLBACK}
                        token_emitted = True
                    yield {
                        "type": "error",
                        "message": (
                            f"Response timeout — stream exceeded "
                            f"the wall-clock timeout ({timeout_s:.0f}s) "
                            f"while waiting on {_stalled} "
                            f"({_stalled_for:.0f}s)."
                        ),
                    }
                    terminal_emitted = True
                    return

                try:
                    item = await asyncio.wait_for(queue.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    # Loop top will see remaining <= 0 and emit the error.
                    continue

                if item is _SENTINEL:
                    # Producer finished — re-raise any exception it caught
                    # so the outer safety net can convert it to error+end.
                    # (Necessary to keep test_inner_generator_exception_...
                    # passing under the producer-task indirection.)
                    await producer_task
                    return

                event = item
                if event.get("type") == "token":
                    token_emitted = True
                if event.get("type") == "tool_call":
                    _tn = event.get("tool") or event.get("name")
                    if _tn and _tn not in tools_seen:
                        tools_seen.append(_tn)
                if event.get("type") in ("end", "error"):
                    terminal_emitted = True
                yield event
        finally:
            # Cancel-then-gather so both tasks fully unwind even when the
            # wall-clock branch fires mid-stream. The bare cancel() alone
            # left coroutine warnings; gather with return_exceptions=True
            # swallows the CancelledError and any producer leftovers.
            producer_task.cancel()
            heartbeat_task.cancel()
            await asyncio.gather(
                producer_task, heartbeat_task, return_exceptions=True,
            )

    import importlib

    from app.core.offload import off_loop

    # A large module: its first import (once per process) froze the loop for
    # 125-188 ms at 15 users. Already-imported modules return at once.
    answer_exit = await off_loop(importlib.import_module, "app.agents.answer_exit")

    # Opened before the producer task exists, so the task (and the threads it
    # offloads to) record this turn's evidence into the same record.
    _exit_turn = answer_exit.open_turn()
    # The user's own words reach every calculator run this turn makes.
    from app.lib import construction_formulas as _calc_words

    _calc_words.set_current_ask(user_message)
    _streamed: list[str] = []
    _leak_guard = _EmitLeakGuard()
    import time as _time

    # Where a streamed turn's time goes (order 0520Z): one TURN_TIMING line
    # per turn with the offsets of its first event, first token, end event
    # and close -- no content, the conversation id only.
    _marks: dict[str, float] = {"received": _time.monotonic()}
    _steps: dict[str, int] = {"tool_calls": 0, "tool_results": 0, "tool_calls_before_first_token": 0}
    try:
        async for event in _inner():
            _marks.setdefault("first_event", _time.monotonic())
            event = _leak_guard.check(event)
            if event is None:
                continue
            _etype = event.get("type") if isinstance(event, dict) else None
            if _etype == "tool_call":
                _steps["tool_calls"] += 1
                if "first_token" not in _marks:
                    _steps["tool_calls_before_first_token"] = _steps["tool_calls"]
            elif _etype == "tool_result":
                _steps["tool_results"] += 1
            if isinstance(event, dict) and event.get("type") == "token" and isinstance(event.get("content"), str):
                _marks.setdefault("first_token", _time.monotonic())
                _streamed.append(event["content"])
            if isinstance(event, dict) and event.get("type") == "end":
                _marks["end_event"] = _time.monotonic()
                event = await off_loop(answer_exit.check_end_event, event, "".join(_streamed), _exit_turn)
            if (
                isinstance(event, dict)
                and event.get("type") == "start"
                and _hat_evt
                and not _hat_emitted
            ):
                yield event
                yield _hat_evt
                _hat_emitted = True
                continue
            if isinstance(event, dict) and event.get("type") == "end":
                event = _with_hat_signals(event, _hat_evt)
            yield event
    except Exception as exc:  # noqa: BLE001 - last-line safety net
        _LOG.exception("chat_stream: generator escaped with exception")
        if not token_emitted:
            # Make sure the UI does NOT render an empty bubble. A token
            # event populates the bubble with the friendly fallback; the
            # error event then triggers the styled error banner.
            yield {"type": "token", "content": _EMPTY_RESPONSE_FALLBACK}
            token_emitted = True
        yield {"type": "error", "message": f"chat_stream crashed: {exc}"}
        terminal_emitted = True
        return
    finally:
        _log_turn_timing(_marks, conversation_id, _steps)

    # Inner generator completed without exception but emitted no terminal
    # event — synthesise one so the SSE consumer sees a clean close.
    if not terminal_emitted:
        _LOG.warning(
            "chat_stream: inner generator returned with no terminal event "
            "(token_emitted=%s) — emitting synthetic end",
            token_emitted,
        )
        if not token_emitted:
            yield {"type": "token", "content": _EMPTY_RESPONSE_FALLBACK}
            _streamed.append(_EMPTY_RESPONSE_FALLBACK)
        yield _with_hat_signals(
            await off_loop(
                answer_exit.check_end_event,
                {"type": "end", "iterations": 0, "sources": [], "tools": list(tools_seen)},
                "".join(_streamed), _exit_turn,
            ),
            _hat_evt,
        )


def unavailable_reason(self) -> str | None:
    """F35: an agent whose every functional block failed to load is a GHOST
    -- it lists in /v1/agents, accepts a chat, and then truthfully tells
    the user it has no tools (observed live: external-mcp on a virgin
    deployment, where mcp_consumer only loads with CEREBRUM_VIRGIN=false).
    Blocks absent from BLOCK_REGISTRY are silently skipped when building
    tool definitions, so this is the one place the gap is visible.
    Infra-only blocks (cache_manager) don't count as functional; agents
    that declare no blocks at all are prose agents and stay available."""
    from app.agents.runtime import (  # noqa: F401 -- read at call time
        BLOCK_REGISTRY, _INFRA_ONLY_BLOCKS,
    )
    functional = [b for b in self.allowed_blocks if b not in _INFRA_ONLY_BLOCKS]
    if functional and not any(b in BLOCK_REGISTRY for b in functional):
        return (
            "none of this agent's functional blocks ("
            + ", ".join(functional)
            + ") are loaded on this deployment (extended platform blocks "
            "require CEREBRUM_VIRGIN=false)"
        )
    return None
