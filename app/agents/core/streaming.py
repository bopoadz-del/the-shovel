"""The streaming turn: retrieval, model calls, tools and the end event.

Moved from the Agent class in app/agents/runtime.py by
scripts/move_agent_methods.py (F-DRIVER Phase A). Each function is still
an Agent method (the class binds it); runtime names are read at call
time, so this module does not import runtime when it is imported.
"""
from __future__ import annotations

from typing import Any, AsyncIterator

# One shared label for the pre-first-token tool phase, so the bubble is
# never blank while the predispatch probes and retrieval run. Defined here
# rather than inline so the wording lives in one place.
_TOOL_PHASE_STATUS = "Working through your request…"

# Tool-phase budget (long-ask cap, ORDERS_2220Z step 2). Generic and
# project-independent: a count cap plus a wall-time backstop, with ONE fixed
# priority order -- no per-question or per-document rules. DEFAULT IS INERT:
# the count default covers all probes and the time budget is disabled, so this
# changes nothing until a budget is set with TESTER data. A skipped probe is
# logged. The count cap ranks probes by this fixed order and is therefore
# deterministic (independent of timing); the wall-time arm is the only
# non-deterministic cap, a backstop for a pathologically slow/hung probe, and
# defaults off so normal turns stay byte-stable.
_TOOL_PHASE_PRIORITY = (
    "file", "wbs_duration", "resource_histogram", "look_ahead",
    "wir_form", "remaining_deliverables", "formula_calc",
)


class _ToolPhaseBudget:
    """Decides, deterministically, whether the next predispatch probe may run.

    ``allow(key)`` returns False once the probe's rank in the fixed priority
    order reaches the count cap, or (backstop) once the wall-clock deadline is
    past; each refusal is recorded in ``skipped``. Rank-based counting means
    the same probes are skipped for the same ``max_probes`` regardless of how
    long earlier probes took, so outputs stay deterministic under the count cap.
    """

    def __init__(self, max_probes: int, time_budget_ms: float, clock):
        self._max = max_probes
        self._clock = clock
        self._deadline = (
            clock() + time_budget_ms / 1000.0 if time_budget_ms > 0 else None
        )
        self._rank = {k: i for i, k in enumerate(_TOOL_PHASE_PRIORITY)}
        self.skipped: list[str] = []

    def allow(self, key: str) -> bool:
        if self._rank.get(key, len(self._rank)) >= self._max:
            self.skipped.append(key)
            return False
        if self._deadline is not None and self._clock() > self._deadline:
            self.skipped.append(key)
            return False
        return True


def _make_tool_phase_budget(clock) -> "_ToolPhaseBudget":
    """Read the (env-overridable) budget at call time. Bad values fall back to
    the inert defaults rather than crashing the turn."""
    import os
    try:
        max_probes = int(
            os.getenv("TOOL_PHASE_MAX_PROBES") or len(_TOOL_PHASE_PRIORITY)
        )
    except ValueError:
        max_probes = len(_TOOL_PHASE_PRIORITY)
    try:
        time_ms = float(os.getenv("TOOL_PHASE_TIME_BUDGET_MS") or 0)
    except ValueError:
        time_ms = 0.0
    return _ToolPhaseBudget(max_probes, time_ms, clock)


def _safe_stream_margin(excerpts: str) -> int:
    """Characters to hold back behind the streaming edge so no withheld party
    name can straddle an emit boundary.

    It is the longest withheld variant/acronym derivable from THESE excerpts
    (+4 for the trailing word boundary), floored at 64 (which also covers a
    citation-label span). A name longer than the margin could leak a partial,
    so the margin must grow with the excerpts, never be a fixed constant. On
    any error, hold back MORE (128), never less.
    """
    safe = 64
    try:
        from app.core import party_names as _pn
        for _nm, _ in _pn.extract_party_names(excerpts or ""):
            for _v in _pn._variants(_nm):
                safe = max(safe, len(_v) + 4)
            # An acronym is always far shorter than the 64 floor, so it could
            # never raise the margin -- the term was dead and is removed; the
            # floor already covers it. (#49 post-hoc: dead-terms rule.)
    except Exception:  # noqa: BLE001 — on any doubt hold back MORE, never less
        return 128
    return safe


async def _chat_stream_impl(
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
    _deadline: float | None = None,
    _phase: dict[str, Any] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Internal implementation. ``chat_stream`` wraps this with an emit
    guarantee so silent / crashing exits become structured error events."""
    from app.agents.answer_exit import end_on_a_sentence
    from app.agents.runtime import (  # noqa: F401 -- read at call time
        MAX_TOOL_ITERATIONS, _CONTEXT_LEAK_RETRY_NUDGE, _EMPTY_RESPONSE_FALLBACK,
        _EMPTY_ROUTER_NUDGE, _LOG, _SEARCH_PREAMBLE_RETRY_NUDGE,
        _LENGTH_CUT_NOTICE, _STREAM_DELIVERY_MARGIN_SECONDS, _SYNTH_CUTOFF_NOTICE, _SynthLengthCut,
        _SynthStreamError,
        _TOOL_ERROR_NUDGE_CAP, _TOOL_FORMAT_FALLBACK, _TOOL_FORMAT_RETRY_NUDGE,
        _UNINDEXED_PROJECT_MESSAGE, _apply_hat_activation, _apply_rag_context,
        _build_attached_documents_note, _build_capability_answer, _build_exports_from_audit,
        _build_missing_reference_answer, _build_sources_from_audit, _chunks,
        _calc_input_question, _compose_boq_scope_wbs_answer, _compose_excerpt_boq_instead_of_retry,
        _conflicting_tools_after_predispatch, _cost_grounding_enabled, _empty_router_verdict,
        _file_tool_hint, _final_text_needs_forced_retry, _forced_retry_min_seconds, _forced_retry_nudge,
        _fulfill_answer_report, _has_unread_windows, _inline_boq_hard_excludes,
        _is_capability_request, _is_cost_shaped_query, _latest_operator_ask, _llm_config,
        _looks_like_internal_context_leak, _looks_like_internal_tool_json,
        _looks_like_search_preamble, _looks_like_tool_markup_leak,
        _message_wants_locked_deliverable, _normalize_tool_call_ids, _nudge_for_failed_tool,
        _off_loop, _parse_dsml_tool_calls, _persist_failed_turn, _postprocess_answer,
        _predispatch_file_tool, _predispatch_formula_calc, _predispatch_look_ahead,
        _predispatch_remaining_deliverables, _predispatch_resource_histogram,
        _predispatch_wbs_duration_override, _predispatch_wir_form,
        _project_has_non_rag_context, _project_is_rag_ready_off_process,
        _rag_inject_off_process, _recover_answer_from_tool_messages,
        _recover_tool_calls_from_content, _recovered_calls_are_search_only,
        _requested_char_offset, _sanitize_citation_labels, _sanitize_final_text,
        _sanitize_inline_paths, _scrub_history, _should_force_synthesis,
        _should_short_circuit_delay_damages_daily, _should_short_circuit_part_summary,
        _should_short_circuit_priced_boq, _should_short_circuit_rag_miss,
        _strip_answer_routing_preamble, _summarize_result, _text_needs_tool_recovery,
        _timing_log, _tool_result_content, _tool_result_errored, _turn_progress,
        _vo_draft_hard_excludes, _vo_draft_ready_for_synthesis,
        _withhold_names_in_streamed_segment, answer_report_export_enabled, fulfill_wbs_export,
        json, message_wants_answer_report, message_wants_wbs_export, os, time,
    )
    cfg = _llm_config()
    # Names of the tools this turn actually invoked, in call order.
    #
    # Every `end` event carried iterations/model/sources/exports but never
    # the tools, so a turn that ran formula_executor_v2 arrived at the UI
    # with the field ABSENT and rendered as tools=[] -- while the answer
    # prose said "Tool: code". The two disagreed because only the prose had
    # the information. The TIMING log knew the names all along; the SSE
    # contract just never carried them.
    tools_invoked: list[str] = []

    def _note_tool(name: str | None) -> None:
        if name and name not in tools_invoked:
            tools_invoked.append(name)

    # Answer export: compile the numbered answers to a downloadable docx.
    # Before the API-key check and before RAG (the numbered answer codes
    # used to retrieve RFP attachments). No LLM required.
    if message_wants_answer_report(user_message) and answer_report_export_enabled():
        yield {"type": "start", "agent": self.name}
        answer, exports = _fulfill_answer_report(
            user_message, project_id, conversation_id, history, self.name,
            owner_id=user_id,
        )
        for chunk in _chunks(answer, 80):
            yield {"type": "token", "content": chunk}
        yield {
            "type": "end",
            "iterations": 0,
            "sources": [],
            "tools": list(tools_invoked),
            "exports": exports,
        }
        return

    if message_wants_wbs_export(user_message):
        yield {"type": "start", "agent": self.name}
        answer, exports = fulfill_wbs_export(
            user_message, project_id, conversation_id, self.name,
            owner_id=user_id,
        )
        for chunk in _chunks(answer, 80):
            yield {"type": "token", "content": chunk}
        yield {
            "type": "end",
            "iterations": 0,
            "sources": [],
            "tools": ["export_wbs"],
            "exports": exports,
        }
        return

    # A provider with no env_key (none today) would need no auth. Both
    # DeepSeek and OpenRouter declare an env_key, so require their API key.
    if cfg["env_key"]:
        api_key = api_key or os.getenv(cfg["env_key"])
        if not api_key:
            _LOG.warning("chat_stream: missing %s — yielding error", cfg["env_key"])
            yield {"type": "error", "message": f"No {cfg['env_key']} configured."}
            return
    else:
        api_key = api_key or ""

    _call_stack = _call_stack or [self.name]

    yield {"type": "start", "agent": self.name}

    # hat_signals is injected by chat_stream after this start so every
    # early-return path (answer report, WBS export, capability) still
    # records activation on the SSE the floor scorer reads.

    # Zero-chunk project guardrail: refuse before spending LLM budget
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
        for chunk in _chunks(_UNINDEXED_PROJECT_MESSAGE, 80):
            yield {"type": "token", "content": chunk}
        yield {"type": "end", "iterations": 0, "sources": [],
               "tools": list(tools_invoked)}
        return

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
        # Persist the user turn up front so it survives a mid-loop error.
        (await _off_loop(agent_memory.append_message, conversation_id, "user", user_message))

    # Strip prior hallucinated WBS/BOQ tables from history before
    # sending. Prevents the model from pattern-matching to a prior
    # (often fabricated) table when it should be calling the tool.
    effective_history = _scrub_history(effective_history)

    messages = (await _off_loop(self._build_messages, user_message, effective_history, project_id=project_id))
    # Attachment grounding (S4). The chat router resolves the composer's
    # `[attached: X]` marker to concrete documents and passes them here.
    # A system note pins the reference so "the attached file" is never a
    # guessing game: the model gets the exact document_id to feed into
    # fetch_document or a file-consuming block.
    if attached_documents:
        _att_note = _build_attached_documents_note(attached_documents)
        if _att_note:
            messages.append({"role": "system", "content": _att_note})
    # Pre-iter-0 RAG injection. Runs for any project-scoped turn so that
    # routing to heavy-reasoning (or another agent) does not strip project
    # grounding. Adds a system message AFTER the prompt + project context
    # but BEFORE the latest user turn.
    # In a worker thread: rag_inject is synchronous (embedding + SQL +
    # rerank) and froze the single worker's event loop for its whole
    # duration -- live, one turn stalled /livez for 4.2 s. to_thread
    # copies contextvars, so the caller-role gate still applies.
    if project_id:
        yield _turn_progress.event("searching")
    _rag_sys_msg, _rag_audit = await _rag_inject_off_process(
        user_message=user_message,
        project_id=project_id,
        conversation_id=conversation_id,
        user_id=user_id,
        agent_name=self.name,
        history=effective_history,
    )
    # The message preparation below scans the question and the excerpts with
    # many patterns: in a worker thread, like retrieval (CI: 125-203 ms loop
    # stalls here at 15 users). Only this turn's coroutine touches `messages`.
    if _rag_sys_msg and _rag_sys_msg.get("content"):
        await _off_loop(
            _apply_rag_context, messages, _rag_sys_msg,
            user_data_authoritative=self.user_data_authoritative,
        )

    await _off_loop(_apply_hat_activation, messages, user_message, self.name)

    # Capability / self-introspection short-circuit: a question ABOUT this
    # agent (its tools, abilities, accessible documents) is answered from the
    # REAL tool roster + REAL document list, bypassing the grounded-RAG clamp
    # that would otherwise answer it from matched document text (e.g. "what
    # tools do you have" -> construction power saws). See _is_capability_request.
    if await _off_loop(_is_capability_request, user_message):
        answer = _build_capability_answer(self, project_id)
        if conversation_id:
            from app.core import agent_memory
            (await _off_loop(agent_memory.append_message, conversation_id, "assistant", answer))
        for chunk in _chunks(answer, 80):
            yield {"type": "token", "content": chunk}
        yield {"type": "end", "iterations": 0, "sources": [],
               "tools": list(tools_invoked)}
        return

    # Tool results accumulated this turn — feeds the schedule download offer
    # (a generate_wbs call becomes a 'Schedule (Excel)' export descriptor).
    stream_tool_results: list[dict[str, Any]] = []
    # Fill the bubble the instant the tool phase begins. The predispatch
    # probes below run before the first token, and in the common case they
    # return None -- emitting no tool_call -- so without this the bubble sits
    # blank for the whole pre-first-token phase (TESTER's long-ask stall).
    # The frontend shows a `status` label only until content arrives and
    # clears it on the first token, so a brief label before a fast answer is
    # harmless. One emit, deterministic, changes no computed result.
    yield {"type": "status", "label": _TOOL_PHASE_STATUS}
    # Generic tool-phase budget (default inert). Each probe below asks the
    # budget first; when a cap is set, lower-priority probes are skipped
    # deterministically and recorded in `_budget.skipped`.
    _budget = _make_tool_phase_budget(time.monotonic)
    # Deterministic file pre-dispatch BEFORE the RAG-miss short-circuit
    # (leftover L1 timestamped .docx looks like an identifier).
    _locked = (await _off_loop(_message_wants_locked_deliverable, user_message))
    _pre = None
    if not _locked and _budget.allow("file"):
        _pre = await _predispatch_file_tool(self, messages, project_id)
    if _pre:
        # The file read is a source for this turn. The other predispatches
        # are already on this list; the named-file read has to be too, or
        # Sources never sees the document the answer quoted.
        stream_tool_results.append(_pre)
        _note_tool(_pre["name"])
        # SSE contract: the browser reads "tool" + "args_preview" on a
        # tool_call and "summary" on a tool_result -- see the main tool
        # loop below, which emits "name" only as an alias. This
        # pre-dispatch pair emitted the REVERSE (name, no tool), so the
        # deterministic file fast path -- the IFC and named-document
        # route -- rendered as a blank, unnamed entry in the UI. The turn
        # then narrated work the user could not see it do, which reads as
        # the assistant wandering rather than as a tool that ran.
        _pre_file = ""
        if isinstance(_pre.get("result"), dict):
            _pre_file = str(_pre["result"].get("filename") or "")
        _pre_summary = _summarize_result(_pre["result"])
        yield {"type": "tool_call",
               "tool": _pre["name"],
               "name": _pre["name"],  # alias, see the main loop
               "args_preview": (
                   json.dumps({"filename": _pre_file})[:200]
                   if _pre_file else ""
               ),
               "predispatched": True}
        yield {"type": "tool_result",
               "tool": _pre["name"],
               "name": _pre["name"],  # alias, see the main loop
               "ok": True,
               "summary": _pre_summary[:400],
               "result": _pre_summary,
               "predispatched": True}

    _wbs_pre = None
    if _budget.allow("wbs_duration"):
        _wbs_pre = await _predispatch_wbs_duration_override(
            self, messages, project_id,
        )
    if _wbs_pre:
        _note_tool(_wbs_pre["name"])
        stream_tool_results.append(_wbs_pre)
        _wbs_summary = _summarize_result(_wbs_pre.get("result"))
        yield {"type": "tool_call",
               "tool": _wbs_pre["name"],
               "name": _wbs_pre["name"],
               "args_preview": json.dumps({
                   "duration_overrides": (
                       (_wbs_pre.get("result") or {}).get(
                           "duration_overrides_applied"
                       )
                       if isinstance(_wbs_pre.get("result"), dict)
                       else None
                   ),
               }, default=str)[:200],
               "predispatched": True}
        yield {"type": "tool_result",
               "tool": _wbs_pre["name"],
               "name": _wbs_pre["name"],
               "ok": True,
               "summary": _wbs_summary[:400],
               "result": _wbs_summary,
               "predispatched": True}

    _hist_pre = None
    if _budget.allow("resource_histogram"):
        _hist_pre = await _predispatch_resource_histogram(
            self, messages, project_id,
        )
    if _hist_pre:
        _note_tool(_hist_pre["name"])
        stream_tool_results.append(_hist_pre)
        _hist_summary = _summarize_result(_hist_pre.get("result"))
        _hist_file = ""
        if isinstance(_hist_pre.get("result"), dict):
            _hist_file = str(
                (_hist_pre.get("result") or {}).get("source") or ""
            )
        yield {"type": "tool_call",
               "tool": _hist_pre["name"],
               "name": _hist_pre["name"],
               "args_preview": json.dumps({
                   "source": _hist_file,
                   "histogram_kind": (
                       (_hist_pre.get("result") or {}).get("histogram_kind")
                       if isinstance(_hist_pre.get("result"), dict)
                       else None
                   ),
               }, default=str)[:200],
               "predispatched": True}
        yield {"type": "tool_result",
               "tool": _hist_pre["name"],
               "name": _hist_pre["name"],
               "ok": True,
               "summary": _hist_summary[:400],
               "result": _hist_summary,
               "predispatched": True}

    _la_pre = None
    if _budget.allow("look_ahead"):
        _la_pre = await _predispatch_look_ahead(
            self, messages, project_id,
        )
    if _la_pre:
        _note_tool(_la_pre["name"])
        stream_tool_results.append(_la_pre)
        _la_summary = _summarize_result(_la_pre.get("result"))
        _la_file = ""
        if isinstance(_la_pre.get("result"), dict):
            _la_file = str(
                (_la_pre.get("result") or {}).get("schedule_file") or ""
            )
        yield {"type": "tool_call",
               "tool": _la_pre["name"],
               "name": _la_pre["name"],
               "args_preview": json.dumps({
                   "schedule_file": _la_file,
                   "activity_count": (
                       (_la_pre.get("result") or {}).get("activity_count")
                       if isinstance(_la_pre.get("result"), dict)
                       else None
                   ),
               }, default=str)[:200],
               "predispatched": True}
        yield {"type": "tool_result",
               "tool": _la_pre["name"],
               "name": _la_pre["name"],
               "ok": True,
               "summary": _la_summary[:400],
               "result": _la_summary,
               "predispatched": True}

    _wir_pre = None
    if not _locked and _budget.allow("wir_form"):
        _wir_pre = await _predispatch_wir_form(
            self, messages, project_id, operator_text=user_message,
        )
    if _wir_pre:
        _note_tool(_wir_pre["name"])
        stream_tool_results.append(_wir_pre)
        _wir_summary = _summarize_result(_wir_pre.get("result"))
        yield {"type": "tool_call",
               "tool": _wir_pre["name"],
               "name": _wir_pre["name"],
               "args_preview": json.dumps({
                   "action": "wir_form",
                   "wir_number": (
                       (_wir_pre.get("result") or {}).get("wir_number")
                       if isinstance(_wir_pre.get("result"), dict)
                       else None
                   ),
               }, default=str)[:200],
               "predispatched": True}
        yield {"type": "tool_result",
               "tool": _wir_pre["name"],
               "name": _wir_pre["name"],
               "ok": True,
               "summary": _wir_summary[:400],
               "result": _wir_summary,
               "predispatched": True}

    _more_pre = None
    if not _hist_pre and _budget.allow("remaining_deliverables"):
        _more_pre = await _predispatch_remaining_deliverables(
            self, messages, project_id, operator_text=user_message,
            conversation_id=conversation_id,
        )
    if _more_pre:
        _note_tool(_more_pre["name"])
        stream_tool_results.append(_more_pre)
        _more_summary = _summarize_result(_more_pre.get("result"))
        yield {"type": "tool_call",
               "tool": _more_pre["name"],
               "name": _more_pre["name"],
               "args_preview": json.dumps({
                   "action": _more_pre["name"],
               }, default=str)[:200],
               "predispatched": True}
        yield {"type": "tool_result",
               "tool": _more_pre["name"],
               "name": _more_pre["name"],
               "ok": True,
               "summary": _more_summary[:400],
               "result": _more_summary,
               "predispatched": True}

    _calc_pre = None
    if not _locked and _budget.allow("formula_calc"):
        _calc_pre = await _predispatch_formula_calc(
            self, messages, project_id, operator_text=user_message,
        )
    if _budget.skipped:
        # Visible in the turn trace: which probes the cap dropped and why.
        _LOG.info(
            "tool-phase cap: skipped %d probe(s) %s (max=%s)",
            len(_budget.skipped), _budget.skipped, _budget._max,
        )
    if _calc_pre:
        _note_tool(_calc_pre["name"])
        stream_tool_results.append(_calc_pre)
        _calc_summary = _summarize_result(_calc_pre.get("result"))
        yield {"type": "tool_call",
               "tool": _calc_pre["name"],
               "name": _calc_pre["name"],
               "args_preview": json.dumps({
                   "action": "construction_calc",
               }, default=str)[:200],
               "predispatched": True}
        yield {"type": "tool_result",
               "tool": _calc_pre["name"],
               "name": _calc_pre["name"],
               "ok": bool(_calc_pre.get("ok")),
               "summary": _calc_summary[:400],
               "result": _calc_summary,
               "predispatched": True}

    # Fast path: exact reference miss with no RAG context. Skip when a
    # named project file was already fetched/extracted from disk.
    if not _pre and not _wbs_pre and not _hist_pre and not _wir_pre and not _more_pre and not _calc_pre and (await _off_loop(_should_short_circuit_rag_miss, _rag_audit, _rag_sys_msg, user_message
    )):
        answer = _build_missing_reference_answer(project_id, user_id)
        if conversation_id:
            from app.core import agent_memory
            (await _off_loop(agent_memory.append_message, conversation_id, "assistant", answer))
        for chunk in _chunks(answer, 80):
            yield {"type": "token", "content": chunk}
        yield {"type": "end", "iterations": 0, "sources": [],
               "tools": list(tools_invoked)}
        return
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
        for chunk in _chunks(answer, 80):
            yield {"type": "token", "content": chunk}
        yield {
            "type": "end",
            "content": answer,
            "iterations": 0,
            "sources": [],
            "tools": list(tools_invoked),
        }
        return
    # Delay-damages daily amount before the priced-BOQ path: compose rate × ACA so a priced-BOQ
    # refuse cannot close the turn. Predispatch keeps the LLM.
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
        for chunk in _chunks(answer, 80):
            yield {"type": "token", "content": chunk}
        yield {
            "type": "end",
            "content": answer,
            "iterations": 0,
            "sources": (await _off_loop(_build_sources_from_audit, _rag_audit, answer, stream_tool_results)),
            "provenance": (_rag_audit or {}).get("provenance") or [],
            "tools": list(tools_invoked),
        }
        return
    # WAVE 2 B4: priced row already in excerpts — skip the provider
    # hop so a transient unavailable banner cannot empty the turn.
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
        for chunk in _chunks(answer, 80):
            yield {"type": "token", "content": chunk}
        yield {
            "type": "end",
            "content": answer,
            "iterations": 0,
            "sources": (await _off_loop(_build_sources_from_audit, _rag_audit, answer, stream_tool_results)),
            "provenance": (_rag_audit or {}).get("provenance") or [],
            "tools": list(tools_invoked),
        }
        return
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
        for chunk in _chunks(answer, 80):
            yield {"type": "token", "content": chunk}
        yield {
            "type": "end",
            "content": answer,
            "iterations": 0,
            "sources": (await _off_loop(_build_sources_from_audit, _rag_audit, answer, stream_tool_results)),
            "provenance": (_rag_audit or {}).get("provenance") or [],
            "tools": list(tools_invoked),
        }
        return
    # Root fix for the tool-loop: RAG context is injected pre-loop, yet the
    # model keeps re-calling search_project_documents when an exact value
    # isn't found — grinding to the iteration cap. Allow a few explicit
    # searches, then STOP offering the tool so the model must answer from what
    # it has (or honestly say it doesn't) instead of looping. generate_wbs and
    # other tools stay available.
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
    # Once a deliverable (non-search) tool has returned its result, the model
    # has what it needs — the next call is synthesis. Offering tools on that
    # synthesis call is pure downside: it lets the model loop, and on Groq a
    # large context makes Llama's prose synthesis fail the tool-use validator
    # (HTTP 400 tool_use_failed) or blow the free-tier TPM (429) — each then
    # falls back to slow gpt-oss and the turn appears to hang for minutes.
    # Force a tool-free synthesis instead. Env-disable: AGENT_FORCE_SYNTHESIS=0.
    # Live M1/M2/M6/M9/M13/M14: remaining already drafted the artifact,
    # but force_synthesis stayed False so a later steal tool overwrote it.
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
    # True token streaming for the FINAL synthesis call only. Gated to:
    #   * SYNTHESIS_STREAMING != "0"  (default ON; "0" is the instant prod
    #     kill-switch) -- ORDERS_0305Z: the non-streaming path computed the
    #     whole answer then re-chunked it, so first_token ~= total end to end;
    #     this built, guarded streaming path was parked off and is now on.
    #   * provider in (openrouter, deepseek)  (the verified streaming paths)
    # When enabled, the force_synthesis iteration streams provider deltas as
    # token events instead of computing the whole answer then re-chunking it.
    # Any pre-first-token failure falls back to the untouched non-streaming
    # path below; every per-segment sanitise / name-withhold / tool-leak guard
    # is unchanged.
    _synth_stream_enabled = (
        os.getenv("SYNTHESIS_STREAMING", "1") != "0"
        # DeepSeek and OpenRouter both stream the OpenAI SSE shape. Keep
        # this list in step with the allowlist in _stream_synthesis.
        and cfg["provider"] in ("openrouter", "deepseek")
        # rag_debug needs the whole final text to run its with/without-RAG
        # A/B in the non-streaming branch; don't stream those turns.
        and not rag_debug
        # Token streaming yields the answer BEFORE _postprocess_answer can
        # act, so a fabricated price would already be on the client with
        # only the persisted copy corrected (Codex #223/#224). For
        # COST-SHAPED queries only, the gate wins: streaming disables itself
        # so the whole answer is price-gated before emission. Non-cost turns
        # stream normally. (The standards advisory is non-blocking, so under
        # streaming it remains best-effort on the persisted copy — an
        # accepted minor gap; streaming is off in prod regardless.)
        and not (_cost_grounding_enabled() and _is_cost_shaped_query(user_message))
    )
    # WARNING-level timing instrumentation: Render drops INFO app logs on this
    # service, so the deliverable-hang call-count/latency was invisible.
    # Gate behind AGENT_TIMING_LOG=1 so it's off by default.
    _timing = os.getenv("AGENT_TIMING_LOG") == "1"
    # The clock counts from the request's arrival (app.core.turn_timing),
    # so cum= covers retrieval and pre-dispatch too.
    from app.core import turn_timing as _turn_timing
    _turn_timing.mark("predispatch")
    _turn_t0 = _turn_timing.arrived_at() or time.monotonic()
    yield _turn_progress.event("writing")

    # `_deadline` is the caller's wall-clock cap for the WHOLE turn (a
    # time.monotonic() instant), or None when nothing is capping us.
    # Every LLM call below gets it minus a delivery margin, so a call
    # that runs out of budget fails inside the turn and we still stream
    # something instead of being cancelled mid-await.
    def _remaining_budget() -> float | None:
        return None if _deadline is None else _deadline - time.monotonic()

    def _llm_deadline() -> float | None:
        if _deadline is None:
            return None
        # Reserve a slice of what is LEFT rather than a flat 10s, so a
        # deliberately short CHAT_STREAM_TIMEOUT_SECONDS (the test suite
        # runs 1-3s) does not hand every LLM call a deadline already in
        # the past. At the 240s production default this is the full 10s.
        left = max(0.0, _deadline - time.monotonic())
        return _deadline - min(_STREAM_DELIVERY_MARGIN_SECONDS, left * 0.1)

    # What the turn is currently waiting on, so the caller's wall-clock
    # branch can NAME the stalled component instead of reporting a bare
    # elapsed time. Diagnosing 43e40b3a-e8f needed the ABSENCE of a
    # "forced-retry call=" line to infer where it hung.
    def _set_phase(name: str) -> None:
        if _phase is not None:
            _phase["name"] = name
            _phase["since"] = time.monotonic()
    # Served model string of the LAST successful LLM call this turn, surfaced
    # in the `end` event so observability (fork_cli/smoke) can tell the
    # configured provider apart from a silent fallback (e.g. DeepSeek vs an
    # OpenRouter fallback) per run. The model string alone is a sufficient
    # discriminator; None until the first successful call.
    served_model: str | None = None
    for iteration in range(MAX_TOOL_ITERATIONS):
        _LOG.info("chat_stream: iter=%d agent=%s force_synthesis=%s", iteration, self.name, force_synthesis)
        # ── True token streaming for the forced synthesis call ────────────
        # force_synthesis is set only AFTER a deliverable tool returns, so
        # this call is with_tools=False — no tool_calls can appear, honouring
        # "tool iterations stay non-streaming". Stream its provider deltas as
        # token events. On any pre-first-token failure we fall through to the
        # unchanged non-streaming path below (streaming is never a one-way
        # door). A mid-stream drop finishes with whatever streamed.
        if force_synthesis and _synth_stream_enabled:
            streamed_any = False
            fell_back = False
            cut_off = False
            length_cut = False
            tool_leak = False
            acc: list[str] = []
            pending = ""
            raw_emitted = 0        # chars of the raw cumulative already turned into output
            emitted_len = 0        # chars of SCRUBBED text already sent to the client
            # Hold back >= the longest withheld party-name string for THIS
            # turn's excerpts, so no name can straddle an emit boundary -- the
            # invariant runtime._withhold_names_in_streamed_segment documents
            # ("a name does not straddle two of them"). The old code kept it by
            # flushing ONLY at newlines, which left a no-newline paragraph
            # unstreamed; we keep it with a margin instead so a paragraph
            # streams. Derivation + its mutants live in _safe_stream_margin.
            _safe_margin = _safe_stream_margin(
                (_rag_sys_msg or {}).get("content", "") if _rag_sys_msg else ""
            )
            _MIN_FLUSH_CHARS = 24
            try:
                async for _delta in self._stream_synthesis(
                    messages, api_key, project_id=project_id, user_id=user_id,
                ):
                    streamed_any = True
                    acc.append(_delta)
                    pending += _delta
                    raw_so_far = "".join(acc)
                    # Tool-call leak: SYNTHESIS_STREAMING must not
                    # flush XML tool markup or raw tool-call JSON to the client
                    # before sanitization (frontend keeps accumulated tokens).
                    if (
                        (await _off_loop(_looks_like_tool_markup_leak, raw_so_far))
                        or (await _off_loop(_looks_like_internal_tool_json, raw_so_far))
                        or (await _off_loop(_looks_like_internal_context_leak, raw_so_far))
                    ):
                        tool_leak = True
                        pending = ""
                        continue
                    # Keep the un-emitted tail in sync every delta so a
                    # held (non-yielded) iteration still leaves `pending`
                    # correct for the post-loop flush.
                    pending = raw_so_far[raw_emitted:]
                    # Stream the STABLE prefix: everything that sits at least
                    # `_safe_margin` behind the live edge (or up to a trailing
                    # newline, which always completes a line). `_safe_margin`
                    # is the longest withheld party-name string this turn, so
                    # no name can straddle the boundary. This streams a
                    # no-newline paragraph that the old newline-only cadence
                    # left unshown until the end (ORDERS_30OCT_1535Z item 6a).
                    _stable_end = len(raw_so_far) - _safe_margin
                    _nl_at = raw_so_far.rfind("\n")
                    if _nl_at + 1 > _stable_end:
                        _stable_end = _nl_at + 1
                    if _stable_end <= raw_emitted:
                        continue
                    if (_stable_end - raw_emitted) < _MIN_FLUSH_CHARS and _nl_at < raw_emitted:
                        continue
                    # A streamed answer that is (so far) a search-promise must
                    # never reach the client -- #454. Gate on the WHOLE
                    # accumulated answer, not the stable prefix: the promise
                    # verb ("search"/"check") sits in the held-back tail, so a
                    # prefix-only check would miss it and flush the preamble
                    # ("I don't have ... Let me "). The old newline cadence held
                    # the promise by never flushing prose; this holds it until
                    # the post-loop promise-hold forces the no-tools retry.
                    _scrub_all = _strip_answer_routing_preamble(
                        _withhold_names_in_streamed_segment(
                            _sanitize_inline_paths(_sanitize_citation_labels(raw_so_far)),
                            _rag_sys_msg,
                        )
                    )
                    if await _off_loop(_final_text_needs_forced_retry, _scrub_all, user_message=user_message):
                        continue
                    # Scrub the WHOLE stable prefix (never a slice) so a
                    # withheld name is seen entire; `seg` is the cumulative
                    # scrubbed text and we emit only its new suffix.
                    seg = _sanitize_inline_paths(_sanitize_citation_labels(raw_so_far[:_stable_end]))
                    seg = _withhold_names_in_streamed_segment(seg, _rag_sys_msg)
                    seg = _strip_answer_routing_preamble(seg)
                    _new_out = seg[emitted_len:]
                    if _new_out and not (await _off_loop(_looks_like_internal_tool_json, _new_out)):
                        raw_emitted = _stable_end
                        emitted_len = len(seg)
                        pending = raw_so_far[raw_emitted:]
                        yield {"type": "token", "content": _new_out}
            except _SynthStreamError as _se:
                if streamed_any:
                    # Tokens already left the provider. Do not restart
                    # `_call_llm` (anti-duplicate). Finish with what
                    # streamed and mark the cut-off before persist/`end`.
                    cut_off = True
                    length_cut = isinstance(_se, _SynthLengthCut)
                    _LOG.warning("chat_stream: synthesis stream dropped mid-way (%s)", _se)
                else:
                    _LOG.info("chat_stream: synthesis stream unavailable (%s) — non-streaming fallback", _se)
                    fell_back = True
            if not fell_back:
                raw = "".join(acc)
                if cut_off:
                    # Whatever cut the stream, it fell mid-sentence: the
                    # answer ends on its last whole sentence (the client
                    # shows the end event's content in place of the tokens).
                    raw = end_on_a_sentence(raw)
                # No context-leak check here on purpose. The in-loop
                # guard above runs on `raw_so_far` after EVERY delta,
                # including the last, so a repeat at this point cannot
                # fail on any input the loop would have passed -- no test
                # can kill it, and a guard no test can kill is not a
                # guard. The chokepoint in _sanitize_final_text, which
                # `final_text` goes through two lines down, is the
                # backstop.
                if (
                    (await _off_loop(_looks_like_tool_markup_leak, raw))
                    or (await _off_loop(_looks_like_internal_tool_json, raw))
                ):
                    tool_leak = True
                # Fully-sanitised accumulated text: what we persist + feed
                # sources/exports (must match the non-streaming path, not a
                # concatenation of per-line flushes). Computed BEFORE the
                # tail flush so a dangling promise can be held back rather
                # than shown and then contradicted.
                final_text = _sanitize_inline_paths(
                    _sanitize_citation_labels(_sanitize_final_text(
                        raw, messages=messages, tool_results=stream_tool_results,
                    ))
                )
                # #454 taught the NON-streamed branch that a first-person
                # promise to search is not an answer. This branch never
                # learned it: it retried only on an EMPTY stream, and a
                # promise is not empty. SYNTHESIS_STREAMING is set in
                # production, so this is the branch a live turn takes.
                #
                # Live on 1594b32, with #454 and #455 both deployed, a
                # fresh thread asking for the Engineer's Representative
                # ended on:
                #
                #   "I don't have the Engineer's Representative's name in
                #    the retrieved excerpts. Let me search the Contract
                #    Data and Schedules volume more specifically for that
                #    appointment."
                #
                # _SEARCH_PROMISE_TAIL_RE matches that string. The
                # non-streamed branch would have retried. This one shipped
                # it as the answer.
                #
                # Held rather than flushed: the loop above releases a
                # segment only once 60 chars AND a newline have
                # accumulated, so a short end-of-turn promise is still
                # entirely in `pending` here and the user has seen nothing.
                # On a long answer that ends in a promise some prefix is
                # already out and the retry appends to it -- uglier, and
                # still better than a promise as the final word.
                promise_hold = bool(final_text.strip()) and (
                    (await _off_loop(_final_text_needs_forced_retry, final_text, user_message=user_message
                    ))
                )
                if promise_hold:
                    _LOG.info(
                        "chat_stream: streamed synthesis ended on a search "
                        "promise -- holding it and forcing a no-tools retry"
                    )
                # A context leak is not the same failure as an XML/JSON
                # tool-call leak, and must not share its outcome. There,
                # the model wanted a tool it no longer had, and the
                # controlled fallback ("retry or narrow the question") is
                # all that is left. Here the model was HANDED the answer
                # and copied the packaging instead of reading it -- the
                # Delay Damages rate the live leak buried was 0.1% of the
                # Contract Price per calendar day, sitting in the very
                # excerpts it echoed. Serving the dead-end fallback for
                # that would be a second failure stacked on the first, so
                # this takes the retry path #456 built instead.
                leak_hold = bool(raw.strip()) and (await _off_loop(_looks_like_internal_context_leak, raw))
                if leak_hold:
                    _LOG.warning(
                        "chat_stream: streamed synthesis returned the "
                        "platform's own context -- suppressing it and "
                        "forcing a no-tools retry"
                    )
                # Do not flush a held tool-leak tail — tool-call leak
                # streaming used to emit tool JSON/XML here before sanitize.
                if pending and not tool_leak and not promise_hold:
                    seg = _sanitize_inline_paths(_sanitize_citation_labels(pending))
                    seg = _withhold_names_in_streamed_segment(seg, _rag_sys_msg)
                    seg = _strip_answer_routing_preamble(seg)
                    if seg and not (await _off_loop(_looks_like_internal_tool_json, seg)):
                        yield {"type": "token", "content": seg}
                # Recover-from-tools lives in _postprocess_answer. Do not
                # run it before the empty-stream check — a successful
                # commissioning/WIR/IPC tool (or predispatch draft) would
                # fill the blank and skip the forced non-streaming retry
                # that test_empty_stream_triggers_forced_retry pins.
                if tool_leak and not leak_hold and final_text.strip():
                    final_text = (await _off_loop(_postprocess_answer, final_text, _rag_sys_msg, messages,
                        fallback_used=bool(_rag_audit.get("fallback_used")),
                        agent_name=self.name,
                        project_id=project_id,
                        audit_rec=_rag_audit,
                    ))
                    for chunk in _chunks(final_text, 80):
                        yield {"type": "token", "content": chunk}
                elif not final_text.strip() or promise_hold or leak_hold:
                    # Nothing usable streamed -- empty, or a promise to
                    # search that #454 already ruled is not an answer. One
                    # path for both, so the two branches cannot disagree
                    # about what counts as an answer again.
                    # leak_hold first: it is the more specific diagnosis,
                    # and it is reached with promise_hold ALSO set, because
                    # _sanitize_final_text has by then replaced the leak
                    # with _TOOL_FORMAT_FALLBACK -- whose own "retry or
                    # narrow the question" wording trips the search-promise
                    # detector. Ordering the other way would send the model
                    # "stop promising to search" for a turn where it never
                    # promised anything.
                    priced = _compose_excerpt_boq_instead_of_retry(
                        final_text, _rag_sys_msg, messages,
                    )
                    if priced:
                        final_text = priced
                        final_text = _sanitize_inline_paths(
                            _sanitize_citation_labels(final_text)
                        )
                        final_text = (await _off_loop(_postprocess_answer, final_text, _rag_sys_msg, messages,
                            fallback_used=bool(_rag_audit.get("fallback_used")),
                            agent_name=self.name,
                            project_id=project_id,
                            audit_rec=_rag_audit,
                        ))
                        for chunk in _chunks(final_text, 80):
                            yield {"type": "token", "content": chunk}
                        if conversation_id:
                            from app.core import agent_memory
                            (await _off_loop(agent_memory.append_message, conversation_id, "assistant", final_text,
                            ))
                        yield {
                            "type": "end",
                            "content": final_text,
                            "iterations": iteration + 1,
                            "model": served_model,
                            "tools": list(tools_invoked),
                            "sources": (await _off_loop(_build_sources_from_audit, _rag_audit, final_text, stream_tool_results,
                            )),
                            "exports": (await _off_loop(_build_exports_from_audit, _rag_audit, final_text, stream_tool_results,
                                conversation_id=conversation_id,
                            )),
                        }
                        return
                    if leak_hold:
                        messages.append(
                            {"role": "user", "content": _CONTEXT_LEAK_RETRY_NUDGE}
                        )
                    elif promise_hold:
                        messages.append(
                            {"role": "user", "content": (await _off_loop(_forced_retry_nudge, final_text))
                             or _SEARCH_PREAMBLE_RETRY_NUDGE}
                        )
                    _set_phase("forced-retry (streamed-synth)")
                    forced_resp = await self._call_llm(
                        messages, api_key, project_id=project_id,
                        with_tools=False, user_id=user_id,
                        deadline=_llm_deadline(),
                    )
                    if forced_resp.get("status") == "error":
                        final_text = _EMPTY_RESPONSE_FALLBACK
                    else:
                        served_model = (forced_resp.get("raw") or {}).get("model") or served_model
                        _fm = forced_resp["choice"].get("message") or {}
                        final_text = _sanitize_final_text(
                            _fm.get("content") or "",
                            messages=messages, tool_results=stream_tool_results,
                        )
                        # No leak check on final_text here: the
                        # _sanitize_final_text call two lines up has
                        # already turned any leak into
                        # _TOOL_FORMAT_FALLBACK, which this condition
                        # catches. Repeating it reads like a second guard
                        # and is one no test can kill. Pinned instead by
                        # test_a_retry_that_leaks_again_is_refused_too,
                        # which asserts the OUTCOME rather than the line.
                        if not final_text.strip() or (await _off_loop(_final_text_needs_forced_retry, final_text, user_message=user_message
                        )):
                            final_text = _EMPTY_RESPONSE_FALLBACK
                    final_text = _sanitize_inline_paths(_sanitize_citation_labels(final_text))
                    final_text = (await _off_loop(_postprocess_answer, final_text, _rag_sys_msg, messages, fallback_used=bool(_rag_audit.get("fallback_used")), agent_name=self.name, project_id=project_id, audit_rec=_rag_audit))
                    for chunk in _chunks(final_text, 80):
                        yield {"type": "token", "content": chunk}
                else:
                    final_text = (await _off_loop(_postprocess_answer, final_text, _rag_sys_msg, messages,
                        fallback_used=bool(_rag_audit.get("fallback_used")),
                        agent_name=self.name,
                        project_id=project_id,
                        audit_rec=_rag_audit,
                    ))
                    cut_notice = _LENGTH_CUT_NOTICE if length_cut else _SYNTH_CUTOFF_NOTICE
                    if cut_off and cut_notice not in final_text:
                        suffix = (
                            ("\n\n" if final_text.strip() else "")
                            + cut_notice
                        )
                        final_text = (
                            final_text.rstrip() + suffix
                            if final_text.strip()
                            else cut_notice
                        )
                        yield {"type": "token", "content": suffix}
                if _timing:
                    _timing_log("TIMING chat_stream STREAMED-SYNTH iter=%d chars=%d cum=%.1fs",
                                 iteration, len(final_text), time.monotonic() - _turn_t0)
                if conversation_id:
                    from app.core import agent_memory
                    (await _off_loop(agent_memory.append_message, conversation_id, "assistant", final_text))
                yield {
                    "type": "end",
                    "content": final_text,
                    "iterations": iteration + 1,
                    "model": served_model,
                    "tools": list(tools_invoked),
                    "sources": (await _off_loop(_build_sources_from_audit, _rag_audit, final_text, stream_tool_results)),
                    "provenance": (_rag_audit or {}).get("provenance") or [],
                    "exports": (await _off_loop(_build_exports_from_audit, _rag_audit, final_text, stream_tool_results, conversation_id=conversation_id)),
                }
                return
        _call_t0 = time.monotonic()
        _set_phase(f"llm-call iter={iteration}")
        resp = await self._call_llm(
            messages, api_key, project_id=project_id, user_id=user_id,
            exclude_tools=excluded_tools or None,
            with_tools=not force_synthesis,
            deadline=_llm_deadline(),
        )
        if _timing:
            _tcs = [((tc.get("function") or {}).get("name")) for tc in (resp.get("choice", {}).get("message", {}).get("tool_calls") or [])]
            _timing_log("TIMING chat_stream iter=%d call=%.1fs status=%s tools=%s cum=%.1fs",
                         iteration, time.monotonic() - _call_t0, resp.get("status"),
                         _tcs or "final", time.monotonic() - _turn_t0)
        if resp.get("status") == "error":
            err = resp.get("error", "LLM call failed")
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
                _LOG.warning(
                    "chat_stream: iter=%d recovered deliverable after LLM error %s",
                    iteration, err[:80],
                )
                for chunk in _chunks(final_text, 80):
                    yield {"type": "token", "content": chunk}
                if conversation_id:
                    from app.core import agent_memory
                    (await _off_loop(agent_memory.append_message, conversation_id, "assistant", final_text,
                    ))
                yield {
                    "type": "end",
                    "content": final_text,
                    "iterations": iteration + 1,
                    "model": served_model,
                    "tools": list(tools_invoked),
                    "recovered_from_llm_error": True,
                    "sources": (await _off_loop(_build_sources_from_audit, _rag_audit, final_text, stream_tool_results)),
                    "provenance": (_rag_audit or {}).get("provenance") or [],
                    "exports": (await _off_loop(_build_exports_from_audit, _rag_audit, final_text, stream_tool_results,
                        conversation_id=conversation_id,
                    )),
                }
                return
            _LOG.warning("chat_stream: iter=%d LLM error %s", iteration, err)
            _persist_failed_turn(conversation_id)
            yield {"type": "error", "message": err}
            return
        served_model = (resp.get("raw") or {}).get("model") or served_model
        assistant_msg = resp["choice"].get("message") or {}
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
            # Leftover L4: XML/DSML leaked during force_synthesis is the
            # answer to sanitize/graft, not another tool round.
            if force_synthesis and (dsml_tool_calls or recovered_tool_calls):
                dsml_tool_calls = []
                recovered_tool_calls = []
            # Once search is capped, do not re-execute the same
            # leaked search envelope — fall through to sanitize + synthesis.
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
                # Treat as a tool-calling turn — do NOT stream the markup.
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
                # Final answer — sanitize DSML markup, raw tool JSON, and
                # empty content before streaming it to the user.
                final_text = _sanitize_final_text(
                    raw_content, messages=messages, tool_results=stream_tool_results,
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
                        _left = _remaining_budget()
                        _need = _forced_retry_min_seconds()
                        if _left is not None and _left < _need:
                            # Not enough turn left to finish another LLM call.
                            # Starting one anyway is exactly how 43e40b3a-e8f
                            # spent its last 113s and still ended on a timeout
                            # banner. Deliver the fallback NOW instead: same
                            # text, ~2 minutes sooner, no error banner. The
                            # unusable text itself is never streamed -- that is
                            # the defect #454 closed and it stays closed.
                            _LOG.warning(
                                "chat_stream: skipping forced retry — %.1fs of "
                                "the turn left, need %.1fs",
                                _left, _need,
                            )
                            if _timing:
                                _timing_log(
                                    "TIMING chat_stream FORCED-RETRY-SKIPPED "
                                    "raw=%dc left=%.1fs cum=%.1fs",
                                    len(raw_content), _left,
                                    time.monotonic() - _turn_t0,
                                )
                            final_text = _EMPTY_RESPONSE_FALLBACK
                        else:
                            _LOG.info("chat_stream: unusable final_text, forcing no-tools retry")
                            if _timing:
                                _timing_log("TIMING chat_stream EMPTY-FINAL raw=%dc -> forced retry, cum=%.1fs",
                                             len(raw_content), time.monotonic() - _turn_t0)
                            _nudge = await _off_loop(_forced_retry_nudge, final_text)
                            if _nudge:
                                messages.append({"role": "user", "content": _nudge})
                            _fr_t0 = time.monotonic()
                            _set_phase("forced-retry")
                            forced_resp = await self._call_llm(
                                messages, api_key, project_id=project_id,
                                with_tools=False, user_id=user_id,
                                deadline=_llm_deadline(),
                            )
                            if _timing:
                                _timing_log("TIMING chat_stream forced-retry call=%.1fs status=%s cum=%.1fs",
                                             time.monotonic() - _fr_t0, forced_resp.get("status"), time.monotonic() - _turn_t0)
                            if forced_resp.get("status") == "error":
                                final_text = _EMPTY_RESPONSE_FALLBACK
                            else:
                                served_model = (forced_resp.get("raw") or {}).get("model") or served_model
                                forced_msg = forced_resp["choice"].get("message") or {}
                                final_text = _sanitize_final_text(
                                    forced_msg.get("content") or "",
                                    messages=messages, tool_results=stream_tool_results,
                                )
                                if (await _off_loop(_final_text_needs_forced_retry, final_text, user_message=user_message)):
                                    final_text = _EMPTY_RESPONSE_FALLBACK
                final_text = _sanitize_inline_paths(_sanitize_citation_labels(final_text))
                final_text = (await _off_loop(_postprocess_answer, final_text, _rag_sys_msg, messages, fallback_used=bool(_rag_audit.get("fallback_used")), agent_name=self.name, project_id=project_id, audit_rec=_rag_audit))
                if _timing:
                    _timing_log("TIMING chat_stream STREAMING-FINAL iter=%d chars=%d cum=%.1fs",
                                 iteration, len(final_text), time.monotonic() - _turn_t0)
                _set_phase(f"streaming-final iter={iteration}")
                _LOG.info("chat_stream: final_text iter=%d chars=%d", iteration, len(final_text))
                for chunk in _chunks(final_text, 80):
                    yield {"type": "token", "content": chunk}
                if conversation_id:
                    from app.core import agent_memory
                    # User turn was already persisted up front.
                    (await _off_loop(agent_memory.append_message, conversation_id, "assistant", final_text))
                # rag_debug opt-in: run a second LLM call with the RAG
                # system message stripped so the caller can compare
                # on/off responses for the same turn. Audit record is
                # passed through unmodified for downstream inspection.
                if rag_debug and _rag_sys_msg is not None:
                    no_rag_messages = [m for m in messages if m is not _rag_sys_msg]
                    try:
                        no_rag_resp = await self._call_llm(
                            no_rag_messages, api_key,
                            project_id=project_id, user_id=user_id,
                            deadline=_llm_deadline(),
                        )
                        off_response = (no_rag_resp.get("choice", {})
                                        .get("message", {})
                                        .get("content", "") or "")
                    except Exception as _e:
                        off_response = f"[rag_debug off-run failed: {_e}]"
                    yield {
                        "type": "end",
                        "content": final_text,
                        "iterations": iteration + 1,
                        "tools": list(tools_invoked),
                        "rag_debug": {
                            "on_response": final_text,
                            "off_response": off_response,
                            "audit": _rag_audit,
                        },
                    }
                    return
                yield {
                    "type": "end",
                    "content": final_text,
                    "iterations": iteration + 1,
                    "model": served_model,
                    "tools": list(tools_invoked),
                    "sources": (await _off_loop(_build_sources_from_audit, _rag_audit, final_text, stream_tool_results)),
                    "provenance": (_rag_audit or {}).get("provenance") or [],
                    "exports": (await _off_loop(_build_exports_from_audit, _rag_audit, final_text, stream_tool_results, conversation_id=conversation_id)),
                }
                return

        tool_calls = _normalize_tool_call_ids(list(tool_calls))
        assistant_msg = {**assistant_msg, "tool_calls": tool_calls}
        messages.append(assistant_msg)
        pending_nudges: list[dict[str, Any]] = []
        for tc in tool_calls:
            fn = (tc.get("function") or {})
            if fn.get("name") == "search_project_documents":
                search_calls += 1
                if search_calls >= SEARCH_TOOL_CAP:
                    # Stop offering the search tool for the rest of this turn —
                    # the model has searched enough; force it to answer.
                    excluded_tools.add("search_project_documents")
            # SSE contract (distinct from the ``on_event`` callback above,
            # which uses "name"/"args"/"id"): the browser reads "tool" and
            # "args_preview" — see frontend ProjectWorkspace.tsx. "name" is
            # emitted as an ALIAS so a consumer written against the callback
            # shape reads the tool name instead of silently getting None.
            _note_tool(fn.get("name"))
            yield {
                "type": "tool_call",
                "tool": fn.get("name"),
                "name": fn.get("name"),
                "args_preview": (fn.get("arguments") or "")[:200],
            }
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
            stream_tool_results.append(tool_result)
            _tool_content = _tool_result_content(
                {**(tool_result["result"] if isinstance(tool_result.get("result"), dict) else {"result": tool_result.get("result")}),
                 **({"validation": tool_result["validation"]} if "validation" in tool_result else {})},
                offset=_requested_char_offset(tc),
            )
            # A successful deliverable tool (anything but search) means the
            # model now has authoritative data to synthesize from — stop
            # offering tools so the next call is a clean, tool-free answer.
            # Unless it does NOT have the data yet: a result with windows
            # left to read is half a document, and disarming here leaves
            # the model holding an instruction to read the rest with no
            # tool to read it with.
            if (
                _force_synth_enabled
                and tool_result.get("ok", True)
                and (await _off_loop(_should_force_synthesis, tool_result))
                and _vo_draft_ready_for_synthesis(
                    locals().get("_op") or user_message or "",
                    tool_result.get("name"),
                )
                and not _has_unread_windows(_tool_content)
            ):
                force_synthesis = True
            yield {
                "type": "tool_result",
                "tool": tool_result["name"],
                "name": tool_result["name"],  # alias, see tool_call above
                "ok": tool_result.get("ok", True),
                "summary": _summarize_result(tool_result["result"])[:400],
            }
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id"),
                "name": tool_result["name"],
                "content": _tool_content,
            })
            if _empty_router_verdict(tool_result):
                pending_nudges.append({"role": "user", "content": _EMPTY_ROUTER_NUDGE})
            else:
                if (_tool_result_errored(tool_result)
                        and error_nudges < _TOOL_ERROR_NUDGE_CAP):
                    error_nudges += 1
                    pending_nudges.append({"role": "user",
                                     "content": _nudge_for_failed_tool(
                                         tool_result, self, project_id=project_id, messages=messages)
                                     + (await _off_loop(_file_tool_hint, messages, project_id,
                                                       self.allowed_blocks))})
        messages.extend(pending_nudges)

    # Hit the cap without a final answer — force one more call with tools disabled.
    daily_damages_cap = (await _off_loop(_should_short_circuit_delay_damages_daily, _rag_sys_msg, messages, has_predispatch=False,
        project_id=project_id, audit_rec=_rag_audit,
    ))
    priced_cap = _compose_excerpt_boq_instead_of_retry("", _rag_sys_msg, messages)
    if daily_damages_cap:
        final_text = daily_damages_cap
    elif priced_cap:
        final_text = priced_cap
    else:
        _LOG.warning("chat_stream: hit MAX_TOOL_ITERATIONS=%d, forcing no-tools retry",
                     MAX_TOOL_ITERATIONS)
        _set_phase("forced-retry (iteration cap)")
        forced_resp = await self._call_llm(
            messages, api_key, project_id=project_id, with_tools=False,
            user_id=user_id, deadline=_llm_deadline(),
        )
        if forced_resp.get("status") == "error":
            _persist_failed_turn(conversation_id)
            yield {"type": "error", "message": f"Hit {MAX_TOOL_ITERATIONS}-iteration cap."}
            return
        served_model = (forced_resp.get("raw") or {}).get("model") or served_model
        forced_msg = forced_resp["choice"].get("message") or {}
        final_text = _sanitize_final_text(
            forced_msg.get("content") or "",
            messages=messages, tool_results=stream_tool_results,
        )
        if (await _off_loop(_final_text_needs_forced_retry, final_text, user_message=user_message)):
            # Forced retry returned empty or raw tool JSON — substitute the
            # user-safe fallback so the UI never renders an empty bubble.
            _LOG.warning("chat_stream: forced final unusable, using fallback")
            final_text = _EMPTY_RESPONSE_FALLBACK
    final_text = _sanitize_inline_paths(_sanitize_citation_labels(final_text))
    final_text = (await _off_loop(_postprocess_answer, final_text, _rag_sys_msg, messages, fallback_used=bool(_rag_audit.get("fallback_used")), agent_name=self.name, project_id=project_id, audit_rec=_rag_audit))
    for chunk in _chunks(final_text, 80):
        yield {"type": "token", "content": chunk}
    if conversation_id:
        from app.core import agent_memory
        # User turn was already persisted up front.
        (await _off_loop(agent_memory.append_message, conversation_id, "assistant", final_text))
    _LOG.info("chat_stream: end (forced_final) chars=%d", len(final_text))
    yield {
        "type": "end",
        "content": final_text,
        "iterations": MAX_TOOL_ITERATIONS,
        "forced_final": True,
        "model": served_model,
        "tools": list(tools_invoked),
        "sources": (await _off_loop(_build_sources_from_audit, _rag_audit, final_text, stream_tool_results)),
        "provenance": (_rag_audit or {}).get("provenance") or [],
        "exports": (await _off_loop(_build_exports_from_audit, _rag_audit, final_text, stream_tool_results, conversation_id=conversation_id)),
    }
