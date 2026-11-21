"""The deterministic pre-first-token tool phase of the streaming turn.

Split out of app/agents/core/streaming.py so that file stays under the
core-module size limit without compressing code. Each probe is still an
Agent method: the predispatch functions are read from app.agents.runtime at
call time, so this module imports no runtime names at import time (the
core modules must not, or the runtime could not bind them).
"""
from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator

from app.core import turn_progress as _turn_progress

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


async def _run_tool_phase(
    self,
    messages: list[dict[str, Any]],
    project_id: str | None,
    user_message: str,
    conversation_id: str | None,
    *,
    stream_tool_results: list[dict[str, Any]],
    note_tool,
    state: dict[str, Any],
) -> AsyncIterator[dict[str, Any]]:
    """Run the deterministic predispatch probes and yield their SSE events.

    The probes answer from project data with no model call, before the first
    token: a named file read, WBS duration overrides, a resource histogram,
    look-ahead, a WIR form, remaining deliverables, and formula calculation.
    Each probe asks the (default-inert) tool-phase budget first.

    Results are appended to ``stream_tool_results`` (the same list the turn
    later reads for sources/exports) and their names are recorded through
    ``note_tool``. The probes' outcomes are written to
    ``state["probes"]`` as ``(file, wbs, histogram, look_ahead, wir, more,
    calc)`` so the caller can keep its per-probe conditions unchanged.
    """
    from app.agents.runtime import (  # noqa: F401 -- read at call time
        _LOG, _message_is_formula_style_ask, _message_wants_locked_deliverable,
        _off_loop, _predispatch_file_tool, _predispatch_formula_calc,
        _predispatch_look_ahead, _predispatch_remaining_deliverables,
        _predispatch_resource_histogram, _predispatch_wbs_duration_override,
        _predispatch_wir_form, _summarize_result,
    )

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
        note_tool(_pre["name"])
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
        yield _turn_progress.tool_call({"type": "tool_call",
               "tool": _pre["name"],
               "name": _pre["name"],  # alias, see the main loop
               "args_preview": (
                   json.dumps({"filename": _pre_file})[:200]
                   if _pre_file else ""
               ),
               "predispatched": True})
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
        note_tool(_wbs_pre["name"])
        stream_tool_results.append(_wbs_pre)
        _wbs_summary = _summarize_result(_wbs_pre.get("result"))
        yield _turn_progress.tool_call({"type": "tool_call",
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
               "predispatched": True})
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
        note_tool(_hist_pre["name"])
        stream_tool_results.append(_hist_pre)
        _hist_summary = _summarize_result(_hist_pre.get("result"))
        _hist_file = ""
        if isinstance(_hist_pre.get("result"), dict):
            _hist_file = str(
                (_hist_pre.get("result") or {}).get("source") or ""
            )
        yield _turn_progress.tool_call({"type": "tool_call",
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
               "predispatched": True})
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
        note_tool(_la_pre["name"])
        stream_tool_results.append(_la_pre)
        _la_summary = _summarize_result(_la_pre.get("result"))
        _la_file = ""
        if isinstance(_la_pre.get("result"), dict):
            _la_file = str(
                (_la_pre.get("result") or {}).get("schedule_file") or ""
            )
        yield _turn_progress.tool_call({"type": "tool_call",
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
               "predispatched": True})
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
        note_tool(_wir_pre["name"])
        stream_tool_results.append(_wir_pre)
        _wir_summary = _summarize_result(_wir_pre.get("result"))
        yield _turn_progress.tool_call({"type": "tool_call",
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
               "predispatched": True})
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
        note_tool(_more_pre["name"])
        stream_tool_results.append(_more_pre)
        _more_summary = _summarize_result(_more_pre.get("result"))
        yield _turn_progress.tool_call({"type": "tool_call",
               "tool": _more_pre["name"],
               "name": _more_pre["name"],
               "args_preview": json.dumps({
                   "action": _more_pre["name"],
               }, default=str)[:200],
               "predispatched": True})
        yield {"type": "tool_result",
               "tool": _more_pre["name"],
               "name": _more_pre["name"],
               "ok": True,
               "summary": _more_summary[:400],
               "result": _more_summary,
               "predispatched": True}

    _calc_pre = None
    if _budget.allow("formula_calc") and (not _locked or _message_is_formula_style_ask(user_message)):
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
        note_tool(_calc_pre["name"])
        stream_tool_results.append(_calc_pre)
        _calc_summary = _summarize_result(_calc_pre.get("result"))
        yield _turn_progress.tool_call({"type": "tool_call",
               "tool": _calc_pre["name"],
               "name": _calc_pre["name"],
               "args_preview": json.dumps({
                   "action": "construction_calc",
               }, default=str)[:200],
               "predispatched": True})
        yield {"type": "tool_result",
               "tool": _calc_pre["name"],
               "name": _calc_pre["name"],
               "ok": bool(_calc_pre.get("ok")),
               "summary": _calc_summary[:400],
               "result": _calc_summary,
               "predispatched": True}

    state["probes"] = (
        _pre, _wbs_pre, _hist_pre, _la_pre, _wir_pre, _more_pre, _calc_pre,
    )
