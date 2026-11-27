"""A driver turn: the model calls tools until it answers, within the
configured number of steps. Events have the shapes the chat stream already
uses (status, tool_call, tool_result, token, end)."""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from app.agents import driver
from app.agents.core import computed_answer
from app.agents.core.computed_answer import has_words
from app.agents.core.tool_registry import is_notes
from app.agents.driver import catalogue, context, llm, render, routes, sheet, sources, tools, trace_record, trace_store
from app.core import turn_progress
from app.core.llm_memory import nested_deeper_than
from app.core.offload import off_loop

_LOG = logging.getLogger(__name__)

#: Where a streamed answer may be cut into checkable pieces: after a
#: sentence's closing mark, or at a line break (list items, table rows).
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n")
_HEADING = re.compile(r"^\s*#{1,6}\s+\S")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")

#: How much of one tool result the model is given.
_TOOL_RESULT_CHARS = 20000


def _shown_to_model(text: str) -> str:
    """A tool result as the model gets it: whole when it fits; otherwise a
    valid envelope that keeps every figure, unit and selection whole, with
    how many rows each row list holds, and a preview of the rest -- never a
    raw slice that no longer parses."""
    if len(text) <= _TOOL_RESULT_CHARS:
        return text
    from app.agents.runtime import _compacted_tool_payload

    note = f"Tool result exceeded {_TOOL_RESULT_CHARS} characters and was cut."
    # The preview is re-escaped inside the envelope and the envelope adds its
    # own fields, so keeping _TOOL_RESULT_CHARS of preview overshoots the cap.
    # The PREVIEW gives way first, measured on the real envelope; the outline
    # (figure, unit, selection, row counts) keeps a fixed budget. max_chars
    # makes the fit a guarantee: an outline over the cap by itself (a
    # statement is never cut) is replaced by a minimal one.
    return _compacted_tool_payload(text, _TOOL_RESULT_CHARS, note=note,
                                   outline_budget=_TOOL_RESULT_CHARS // 4, max_chars=_TOOL_RESULT_CHARS)

_STOP = ("I could not finish this answer within the steps a turn may take. "
         "Please narrow the question or split it into parts.")

#: When the steps run out: one last reply, with no tools, from what the tools
#: already returned -- instead of discarding the work and showing _STOP.
_FINISH = ("No more tool calls are available for this question. Answer the user now from the results "
           "above. Where something could not be worked out, say plainly what is missing and what the "
           "user would need to provide.")


def exit_check(answer: str, msgs: List[Dict[str, Any]], shared: bool = False,
               platform: bool = False) -> tuple:
    """The one check before a driver answer leaves: an attribution no
    evidence record backs is removed (citation_provenance.gate), and every
    figure is credited to the user's words, a tool run or a retrieved
    passage -- one that matches none is removed (figure_provenance). The
    evidence is this turn's own messages: the user's words and the tool
    runs. Returns (answer, provenance trail, report): the report names the
    figures removed and counts the passages read, for the end event."""
    from app.agents import citation_provenance as cp

    # The text's length after each stage, for the turn's trace: a stage
    # that cuts an answer to a stub is named by the numbers (lengths only).
    stages: List[tuple] = [("model", len(answer.strip()))]
    # No registry identifier leaves: each becomes the name a user reads.
    answer, identifiers = render.readable(answer)
    stages.append(("readable", len(answer.strip())))
    if shared:
        # The model's words also pass the platform's shared exit
        # (answer_exit) here, so the stream shows what the final answer
        # shows: no chunk numbers, ids or markers that point at nothing.
        from app.agents import answer_exit

        answer = answer_exit.check_text(answer)
        stages.append(("answer_exit", len(answer.strip())))
    checked = cp.gate(answer, None, msgs, tool_passages=True, annotate=False)
    stages.append((computed_answer.CITATION_STAGE, len(checked.strip())))
    _, every = cp.figure_provenance(checked, None, msgs, enforce=False, tool_passages=True)
    if not platform:
        out, trail = cp.figure_provenance(checked, None, msgs, enforce=True, tool_passages=True)
    else:
        # The platform's own words (a calculator's question, a workflow's
        # request, a rendered result) come from the tool's own result: their
        # figures are the tool's, so none is removed. Lane 5: the range in
        # "accepts (0% to 100%). What bulking should I use?" was cut, and the
        # question with it.
        out, trail = checked, []
        every = []
    stages.append(("figure_provenance", len(out.strip())))
    out = _tidy(out)
    stages.append(("tidy", len(out.strip())))
    if shared:
        # A library file the turn read is named by its title, never by its
        # stored file name -- after the gate, which matches citations
        # against the stored names the evidence carries.
        for stored, title in sorted(sources.library_names(msgs).items(), key=lambda kv: -len(kv[0])):
            out = out.replace(stored, title)
    report = {
        "identifiers_replaced": identifiers,
        "figures_removed": [e["figure"] for e in every if not e.get("source")],
        "stages": stages,
        "passages_read": sum(1 for r in cp.build_evidence(None, msgs, tool_passages=True).records
                             if r.kind == "retrieval"),
    }
    return out, trail, report


def _without_missing_pages(value: Any) -> Any:
    """A tool result as the model reads it, minus pages that are not pages
    (-1 / 0 / None mean "no page"): left in, they came back as "(source: -1)"."""
    if isinstance(value, dict):
        return {k: _without_missing_pages(v) for k, v in value.items()
                if not (k in ("page", "page_number") and (v is None or (isinstance(v, (int, float))
                                                                      and not isinstance(v, bool) and v <= 0)))}
    if isinstance(value, list):
        return [_without_missing_pages(v) for v in value]
    return value


#: Brackets left empty, or holding only separators, once the exit check has
#: removed an unbacked figure or attribution from them.
_EMPTY_BRACKETS = re.compile(r"\s*[(\[]\s*[,;:\-–—]*\s*[)\]]")


def _tidy(text: str) -> str:
    return _EMPTY_BRACKETS.sub("", text)


def _asks_for_input(result: Dict[str, Any]) -> bool:
    """True when a tool's result is a request for an input (a calculator's
    question, a workflow naming what it needs), not a failure."""
    if not isinstance(result, dict):
        return False
    env = result.get("result") if isinstance(result.get("result"), dict) else {}
    return bool(result.get("needs_input") or result.get("missing") or env.get("needs_input")
                or str(env.get("question") or "").strip())


def _calc_verdict(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """A calculator run as the trace records it: its status, whether it asked,
    and which inputs it was missing or used a default for (names only)."""
    env = result.get("result") if isinstance(result, dict) else None
    if not isinstance(env, dict):
        return None
    inner = env.get("result") if isinstance(env.get("result"), dict) else {}
    return {"status": env.get("status"), "asked": bool(env.get("needs_input") or env.get("question")),
            "missing": list(env.get("missing") or [])[:8],
            "defaults": sorted(str(k) for k in (inner.get("defaults_used") or env.get("defaults_used") or {}))[:8]}


def _given_inputs(args: Dict[str, Any]) -> Dict[str, Any]:
    """The inputs the model passed a calculator, as the trace records them:
    numbers as passed; a text value only as "text" (the trace carries no user
    words); never the ask itself; at most ten."""
    params = args.get("params") if isinstance(args.get("params"), dict) else {}
    out: Dict[str, Any] = {}
    for key, value in params.items():
        if key in ("text", "message", "user_message") or len(out) >= 10:
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[str(key)] = value
        elif isinstance(value, str):
            out[str(key)] = "text"
    return out


def _offered_workflows(offered: List[Dict[str, Any]]) -> int:
    """How many workflows the run_workflow tool offered this step (0: none)."""
    for t in offered:
        fn = t.get("function") or {}
        if fn.get("name") == routes.RUN_WORKFLOW:
            return len(((fn.get("parameters") or {}).get("properties") or {}).get("workflow", {}).get("enum") or [])
    return 0


async def _inputs_stated_in_ask(agent: Any, user_message: str, history: Any,
                                refusal: Dict[str, Any]) -> Dict[str, Any]:
    """The refused workflow's inputs that the user's own words state, as
    params -- read by one tool-free model call, never invented: a key the
    words do not give is left out. Empty when nothing is stated."""
    inputs = refusal.get("inputs") or {}
    if not inputs:
        return {}
    said = [str(user_message or "")]
    for turn in history or []:
        if isinstance(turn, dict) and turn.get("role") == "user" and turn.get("content"):
            said.append(str(turn["content"]))
    listing = "\n".join(f"- {key}: {label}" for key, label in inputs.items())
    prompt = ("A platform workflow could not run: " + str(refusal.get("missing") or "").strip()
              + "\nIts inputs (key: what it is):\n" + listing
              + "\nFrom the user's own words below, give the inputs they state, as one JSON "
              "object keyed exactly as listed. A list input is a JSON list of the items the user "
              "names. Leave out every input the words do not state; never guess, never use outside "
              "knowledge. Reply with the JSON object only ({} when nothing is stated).")
    messages = [{"role": "system", "content": prompt},
                {"role": "user", "content": "\n\n".join(reversed(said))}]
    try:
        import copy

        reader = copy.copy(agent)
        reader.temperature = llm.DECISION_TEMPERATURE  # reading, not writing: unsampled
        reply = await llm.call(reader, messages, [], model=driver.model())
        text = str(((reply or {}).get("message") or {}).get("content") or "").strip()
        start, end = text.find("{"), text.rfind("}")
        found = json.loads(text[start:end + 1]) if start != -1 and end > start else {}
    except Exception:  # noqa: BLE001 -- reading the ask is help; the turn asks instead
        _LOG.warning("driver: inputs from the ask could not be read", exc_info=True)
        return {}
    if not isinstance(found, dict):
        return {}
    return {k: v for k, v in found.items() if k in inputs and v not in (None, "", [], {})}


def _named_calculator_note(user_message: str) -> str:
    """When the ask names one platform calculator by its declared name, say
    so before the model chooses: a figure that calculator computes comes
    from running it, never from the model's own knowledge. The model still
    chooses; this only tells it what the ask named."""
    try:
        from app.lib import formula_registry
        from app.lib.construction_formulas import calculator_name_from_text

        name = calculator_name_from_text(user_message or "")
        spec = formula_registry.get(name) if name else None
    except Exception:  # noqa: BLE001 -- a note is help, never a failure
        _LOG.debug("named-calculator lookup failed", exc_info=True)
        return ""
    if spec is None:
        return ""
    where = "" if spec.owner == "base" else f" It is in the {spec.owner} hat."
    return (f"This ask names the platform calculator {spec.display_name} (construction_calc, "
            f"calculation \"{spec.name}\").{where} A figure, level or result it computes comes from "
            "running it with the user's inputs, not from your own knowledge; if it cannot run, "
            "say what it needs.")


#: What the model is told with its sheet each round. The same words for
#: every turn and project: how the sheet works, never what to answer.
_SHEET_RULES = (
    "Working sheet for this turn (JSON, below). Write the givens (each figure with its unit and "
    "where it came from), a plan of formula steps whose inputs name givens or earlier steps, and "
    "any input still missing, with sheet_update. Run each step with run_step: code runs the formula "
    "and writes the result here. Every arithmetic step, even one division, is a step "
    "(formula:arithmetic_step); never work a figure in your own text. Ask the user for a missing input "
    "only when no document or tool gives it.")


def _sheet_note(work: Dict[str, Any]) -> str:
    return _SHEET_RULES + "\n" + sheet.render(work)


def _build_sha() -> str:
    from app.routers.health import _deployed_build_sha

    return _deployed_build_sha() or ""


#: Arguments nested deeper than this are not run: past it the JSON writer can
#: run out of stack on them inside a turn (the trace, the tool's own result).
_ARGS_MAX_DEPTH = 200
_ARGS_TOO_DEEP = ("The arguments of this call are nested too deeply to read. "
                  "Send them again with less nesting.")


def _args(tool_call: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    """``(arguments, None)``, or ``({}, why)`` for arguments nested too deeply
    to be read or written out again: that call is answered with ``why`` and
    its tool is not run."""
    raw = (tool_call.get("function") or {}).get("arguments") or "{}"
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except RecursionError:
        return {}, _ARGS_TOO_DEEP
    except ValueError:
        _LOG.warning("driver: tool %r arguments are not JSON; calling it with none",
                     (tool_call.get("function") or {}).get("name"))
        parsed = {}
    if nested_deeper_than(parsed, _ARGS_MAX_DEPTH):
        return {}, _ARGS_TOO_DEEP
    return (parsed if isinstance(parsed, dict) else {}), None



def _given_check(messages: List[Dict[str, Any]]):
    """The check the sheet marks a given with (citation_provenance.given_check)."""
    from app.agents import citation_provenance as cp

    return cp.given_check(messages)


async def stream(agent: Any, user_message: str, history: Optional[list] = None,
                 project_id: Optional[str] = None, conversation_id: Optional[str] = None,
                 user_id: Optional[str] = None, api_key: Optional[str] = None,
                 sheet_mode: bool = False, **_ignored: Any) -> AsyncIterator[Dict[str, Any]]:
    """One driver turn. ``sheet_mode`` (the project's ``answer_path`` is
    ``catalogue``): the turn carries a working sheet, handed back to the model
    every round; the tools stay offered every round and no code ends the turn
    with a tool's question -- a needs-input result goes back to the model as
    the tool result -- short of the round / seconds ceiling; one trace record
    is written per turn."""
    t0 = time.monotonic()
    # 6b: anchor llm_first_token to the generation-turn start (the model-call
    # wait), not to turn_timing's `last` stage marker, which intervening
    # retrieval / pre-dispatch / hat stages reset to ~the first token (the live
    # ~1 ms artifact). Shares the turn's contextvar dict with the request.
    from app.core import turn_timing as _turn_timing
    _turn_timing.mark_llm_dispatch(t0)
    disciplines = [h["discipline"] for h in context.hats()]
    hat: Optional[str] = None
    # The profile lookup reads the database: in a thread, never on the event
    # loop (live: 0.4-1.1 s stalls at 5 driver users).
    msgs: List[Dict[str, Any]] = await off_loop(context.messages, user_message, history,
                                                 project_id, user_id, hat)
    note = await off_loop(_named_calculator_note, user_message)
    if note:
        msgs.insert(len(msgs) - 1, {"role": "system", "content": note})
    work: Optional[Dict[str, Any]] = None  # the working sheet (sheet_mode only)
    sheet_msg: Dict[str, Any] = {"role": "system", "content": ""}
    question_set = trace_record.UNSET
    if sheet_mode:
        # Carried only for this user, project and stored conversation, through
        # the conversation's own access check.
        previous = await off_loop(trace_store.carried_sheet, conversation_id, user_id, project_id)
        if conversation_id:
            # Stored as the default path stores it at a turn's start (after the
            # door's access check), so the next turn's carry finds the row.
            # Read above first: a row this turn creates carries nothing now.
            from app.core import agent_memory

            await off_loop(agent_memory.get_or_create_conversation, conversation_id,
                           getattr(agent, "name", "") or "driver", project_id, owner_id=user_id)
        work = sheet.new_sheet(f"{conversation_id or ''}@{int(time.time() * 1000)}", user_message, previous)
        sheet_msg["content"] = _sheet_note(work)
        msgs.insert(len(msgs) - 1, sheet_msg)
        # The request's own question set (a per-turn sheet request) wins;
        # else the project's profile names one.
        chosen_set = (driver.requested_question_set()
                      or await off_loop(driver.project_setting, project_id, user_id, "question_set"))
        if chosen_set in trace_record.QUESTION_SETS:
            question_set = chosen_set
    used: List[str] = []
    ran: List[tuple] = []  # (tool, arguments) of every tool run, for the credit lines
    computed: List[tuple] = []  # (envelope, inputs) of every calculation that succeeded
    tabled: List[list] = []  # every list of per-period / per-item rows another tool returned
    model_answered = False
    failed: List[tuple] = []  # (tool, its own error) of every run that failed
    succeeded = False  # any tool run this turn came back ok
    user_numbers = set(_NUMBER.findall(user_message or ""))
    evidence: List[Dict[str, Any]] = []  # tool results, whole, for the exit check
    exports: List[Dict[str, Any]] = []
    yield {"type": "start", "mode": "driver", "agent": agent.name}
    yield turn_progress.event("writing")
    answer_done = False
    provenance: List[Dict[str, Any]] = []
    check: Dict[str, Any] = {"figures_removed": [], "passages_read": 0}

    def _trail() -> List[Dict[str, Any]]:
        whole = iter(evidence)  # same order as the tool messages in msgs
        return [next(whole) if m.get("role") == "tool" else m for m in msgs]

    sent: List[str] = []
    # The opening of the model's current reply, held until it is known to be
    # the answer: SHORT_REPLY_CHARS written with no more than CUT_SHARE cut
    # by the stages, or the reply ended (computed_answer.not_the_answer).
    # Withheld only on that positive evidence -- never on its words, length
    # or punctuation alone (GRADER 1250Z: "Yes", "Grade B" are answers).
    opening: List[str] = []
    written = [0, 0]  # characters of the current reply: the model wrote / the stages kept
    streamed = [0]  # characters of the current reply the provider has sent so far
    withheld: Dict[str, Any] = {}  # why, and how many characters (never the text)
    #: Per stage, the characters that went in and came out over the turn.
    stage_chars: Dict[str, int] = {}

    def _flush() -> Optional[Dict[str, Any]]:
        text = "".join(opening)
        opening.clear()
        if text:  # what is shown is what the turn records as said
            sent.append(text)
        return {"type": "token", "content": text} if text.strip() else None

    def _settle(finish: str) -> Optional[Dict[str, Any]]:
        """The model's reply ended: its held opening is shown, or withheld
        when it is positive evidence of not being the answer."""
        why = computed_answer.not_the_answer(written[0], written[1], finish) if opening else ""
        if why:
            withheld.update(reason=why, finish=finish, written=written[0], kept=written[1])
            opening.clear()
            return None
        return _flush()

    def _release(piece: str, model_text: bool = True) -> Optional[Dict[str, Any]]:
        """Exit-check one finished piece of the answer and hand it out.
        ``model_text`` False: the platform's own words (a calculator's
        question, a computed result), which no failed run makes unbacked."""
        if not piece.strip():  # layout only (a blank line): nothing to check
            if opening:  # it belongs after the held opening, not before it
                opening.append(piece)
                return None
            sent.append(piece)
            return {"type": "token", "content": piece}
        if model_text and failed and not succeeded and set(_NUMBER.findall(piece)) - user_numbers:
            # Every tool this turn failed: nothing backs a number the user
            # did not write, bare counts included ("0 items were found").
            if opening or not has_words("".join(sent)):
                written[0] += len(piece.strip())  # removed whole: counted as cut
            return None
        checked, trail, report = exit_check(piece, _trail(), shared=model_text, platform=not model_text)
        # The checks tidy whitespace; the piece's own ending (a line break
        # between list items or paragraphs) belongs to the answer's layout.
        ending = piece[len(piece.rstrip()):]
        if checked.strip():
            checked = checked.rstrip() + ending
        for name, length in report.get("stages") or ():
            stage_chars[name] = stage_chars.get(name, 0) + length
        provenance.extend(trail)
        check["figures_removed"].extend(report["figures_removed"])
        check["identifiers_replaced"] = check.get("identifiers_replaced", 0) + report["identifiers_replaced"]
        check["passages_read"] = report["passages_read"]
        if model_text and (opening or not has_words("".join(sent))):
            written[0] += len(piece.strip())
            # Kept: what the stages left, citations removed not counted as cut.
            written[1] += computed_answer.kept_after_stages(report.get("stages") or ())
            opening.append(checked)
            # The reply is at least as long as what has streamed: a sentence
            # held back for the next one is not held for its length as well.
            if max(written[0], streamed[0]) < computed_answer.SHORT_REPLY_CHARS or computed_answer.not_the_answer(
                    written[0], written[1], computed_answer.NORMAL_FINISH):
                return None
            return _flush()
        sent.append(checked)
        return {"type": "token", "content": checked} if checked.strip() else None

    # One trace line per turn: what the model was offered and chose, and how
    # each step ended. Tool and workflow names only -- never the user's words.
    trace: List[Dict[str, Any]] = []
    # Workflows that refused for a missing input and have not run since. While
    # one is open, the model's next reply is held until it is complete: a
    # reply that only answers would be the workflow's deliverable written by
    # hand, so the workflow's own request for its inputs is given instead.
    refused: Dict[str, Dict[str, Any]] = {}
    filled_from_ask: set = set()  # workflows whose inputs were read from the ask once
    reminded = False  # the model was told once to call a refused workflow again
    step = 0
    ceiling = False  # the sheet turn ran out of rounds or seconds
    for step in range(driver.sheet_max_rounds() if sheet_mode else driver.max_steps()):
        if sheet_mode and time.monotonic() - t0 > driver.sheet_max_seconds():
            ceiling = True
            break
        offered = tools.offered(agent, project_id, hat, disciplines)
        if work is not None:
            # The sheet as it stands, handed back every round; its tools are
            # offered every round -- never withdrawn while a step is open.
            sheet_msg["content"] = _sheet_note(work)
            offered = offered + tools.sheet_schemas()
        trace.append({"step": step + 1, "hat": hat, "offered": len(offered),
                      "workflows": _offered_workflows(offered), "calls": []})
        # Streamed: each finished sentence is checked and sent while the model
        # is still writing. One sentence is held back, so a short preamble
        # ("Let me look that up.") is dropped when the reply turns out to be
        # tool calls.
        buffer, held, result = "", None, None
        opening.clear()
        written[:], streamed[0] = [0, 0], 0
        hold_reply = bool(refused)
        async for kind, value in llm.stream(agent, msgs, offered, model=driver.model()):
            if kind == "done":
                result = value
                break
            buffer += value
            streamed[0] += len(value)
            while not hold_reply:
                cut = _SENTENCE_END.search(buffer)
                if cut is None:
                    break
                sentence, buffer = buffer[:cut.end()], buffer[cut.end():]
                if held is not None:
                    event = await off_loop(_release, held)
                    if event:
                        yield event
                held = sentence
        if result is None or result["status"] != "success":
            yield {"type": "error", "message": (result or {}).get("error") or "The language model did not answer."}
            return
        message = result["message"]
        calls = message.get("tool_calls") or []
        finish = str(result.get("finish_reason") or computed_answer.NORMAL_FINISH)
        # The completion as the provider returned it: its length and why it
        # ended (a length and a reason only, never the text).
        trace[-1].update(finish=finish, reply_chars=len(message.get("content") or ""))
        if calls or refused:
            opening.clear()  # a preamble beside tool calls, or a held reply: never shown
        if not calls and refused and not reminded:
            # The model answered while a refused workflow is still open. Its
            # reply is not shown; it is told once to call the workflow again
            # with the inputs the user's words and the documents give.
            reminded = True
            msgs.append({"role": "system", "content": routes.call_again_note(refused)})
            continue
        if not calls and refused:
            # Reminded once and still answering: the deliverable comes from
            # the workflow, so the turn asks for what the workflow needs
            # instead of showing a hand-written one.
            for request in dict.fromkeys(routes.input_request(w, r) for w, r in refused.items()):
                event = await off_loop(_release, request + "\n", model_text=False)
                if event:
                    yield event
            answer_done = True
            break
        if not calls:
            # A reply with no words (empty, or markdown alone) is not shown;
            # the turn is then answered from its tool results below.
            for piece in ((held, buffer) if has_words((held or "") + buffer) else ()):
                if piece:
                    event = await off_loop(_release, piece)
                    if event:
                        yield event
            event = _settle(finish)
            if event:
                yield event
            answer_done = model_answered = True
            break
        msgs.append({"role": "assistant", "content": message.get("content") or "", "tool_calls": calls})
        questions: List[str] = []  # calculators that need an input this round
        for call in calls:
            name = (call.get("function") or {}).get("name") or ""
            args, unreadable = _args(call)
            yield turn_progress.tool_call({"type": "tool_call", "tool": name, "name": name,
                                            "args_preview": json.dumps(args)[:200]})
            if unreadable:
                result_of_call: Dict[str, Any] = {"ok": False, "error": unreadable}
            elif name == tools.SELECT_HAT:
                chosen = str(args.get("hat") or "")
                if chosen in disciplines:
                    hat = chosen
                    msgs[0] = await off_loop(context.system_message, project_id, user_id, hat)
                    result_of_call = {"ok": True, "hat": hat}
                else:
                    result_of_call = {"ok": False,
                                      "error": f"No hat named {chosen!r}; choose one of {sorted(disciplines)}."}
            elif work is not None and name == tools.SHEET_UPDATE:
                work, problems = sheet.apply_update(work, args)
                # A given is checked when it is written: against the user's
                # messages, the passages and the calculator results so far.
                work = sheet.mark_given(work, _given_check(_trail()))
                result_of_call = {"ok": not problems, "problems": problems, "sheet": work}
                if problems:
                    result_of_call["error"] = "; ".join(problems)
            elif work is not None and name == tools.RUN_STEP:
                used.append(name)
                step_id = str(args.get("step") or "")
                # Checked again before a step runs: a passage fetched after
                # the given was written counts too.
                work = sheet.mark_given(work, _given_check(_trail()))
                work = await off_loop(sheet.run_step, work, step_id, call.get("id") or "")
                entry = next((s for s in work["steps"] if s.get("id") == step_id), {})
                result_of_call = {"ok": entry.get("status") == "done", "step": entry, "sheet": work}
                if entry.get("status") == "waiting_input":
                    result_of_call["needs_input"] = True
                elif entry.get("status") != "done":
                    result_of_call["error"] = entry.get("error") or "the step did not run"
            elif name == catalogue.LOOKUP:
                # The full entries for the ids the model names from the index.
                result_of_call = await off_loop(catalogue.lookup, args.get("ids"))
            elif name == routes.RUN_WORKFLOW:
                used.append(f"{name}:{args.get('workflow')}")
                result_of_call = await routes.run(args, user_message, project_id, user_id, conversation_id)
                workflow = str(args.get("workflow") or "")
                if result_of_call.get("missing") and workflow not in filled_from_ask:
                    # The workflow refused for a missing input. Before any
                    # question, its inputs that the user's own words already
                    # state are read out and it runs once more with them.
                    filled_from_ask.add(workflow)
                    stated = await _inputs_stated_in_ask(agent, user_message, history, result_of_call)
                    if stated:
                        args = {**args, "params": {**(args.get("params") or {}), **stated}}
                        result_of_call = await routes.run(args, user_message, project_id, user_id,
                                                          conversation_id)
                exports.extend(result_of_call.get("exports") or [])
                if result_of_call.get("ok"):
                    refused.pop(workflow, None)
                elif result_of_call.get("missing"):
                    refused[workflow] = result_of_call
            else:
                used.append(name)
                result_of_call = await agent._run_tool_call(call, api_key, project_id, conversation_id,
                                                            user_message=user_message, history=history)
                if result_of_call.get("ok", True):
                    ran.append((name, args))  # a failed run is credited with nothing
            ok = bool(result_of_call.get("ok", True))
            # The platform's shared exit check replaces a tool's internal error
            # text copied into the answer, for the errors this turn recorded.
            from app.agents import answer_exit
            answer_exit.note_tool_result(name, result_of_call)
            call_record = {
                "tool": name, "workflow": args.get("workflow") if name == routes.RUN_WORKFLOW else None,
                "calculation": args.get("calculation") if name == "construction_calc" else None,
                "ok": ok, "missing": bool(result_of_call.get("missing") or result_of_call.get("needs_input")),
                # A calculator's own verdict: success, or needs_input and which inputs.
                "result": _calc_verdict(result_of_call) if name == "construction_calc" else None,
                "given": _given_inputs(args) if name == "construction_calc" else None,
                **({"step": str(args.get("step") or "")} if name == tools.RUN_STEP else {}),
            }
            # What the tool itself states for the trace record (e.g. a memory
            # note's operation, name, size and turn), as given. A tool that
            # states nothing leaves the record's shape as it was.
            if isinstance(result_of_call.get("trace"), dict):
                call_record["trace"] = result_of_call["trace"]
            trace[-1]["calls"].append(call_record)
            if name == catalogue.LOOKUP:
                # The entry ids the model was given, on the lookup's own call
                # only, with counts of the ids it was not given: read from the
                # lookup's result, with no second pass over the catalogue.
                call_record.update(catalogue.read_ids(result_of_call))
            if name not in (tools.SELECT_HAT, catalogue.LOOKUP):
                if ok:
                    # The model's own notes back no figure: a note that was
                    # read or written does not stand in for a tool that ran.
                    if not is_notes(name):
                        succeeded = True
                elif not _asks_for_input(result_of_call):
                    # A tool asking for an input has not failed: its question is
                    # the answer. A real failure is told in its own words; one
                    # with no words of its own adds no line.
                    from app.lib.source_labels import tool_label

                    shown_name = (str(args.get("workflow") or "").replace("_", " ").capitalize()
                                  if name == routes.RUN_WORKFLOW else tool_label(name))
                    error = str(result_of_call.get("error") or result_of_call.get("missing") or "").strip()
                    if error:
                        failed.append((shown_name, error))
            yield {"type": "tool_result", "tool": name, "name": name, "ok": ok}
            full = json.dumps(result_of_call, default=str)
            tool_msg = {"role": "tool", "name": name, "tool_call_id": call.get("id") or name}
            # The model gets at most _TOOL_RESULT_CHARS of a result; the exit
            # check reads it whole -- cut, the JSON no longer parses and every
            # figure it backs looks unsourced (live: 23 figures removed).
            # A calculation reaches the model as prose in the registry's own
            # words (render.render_calculation), never as raw keys it could
            # copy into the answer; the exit check still reads the full result.
            envelope = result_of_call.get("result") if isinstance(result_of_call, dict) else None
            if isinstance(envelope, dict) and envelope.get("calculation"):
                inputs = args.get("params") if isinstance(args.get("params"), dict) else args
                shown = render.render_calculation(envelope, inputs, on_sheet=work is not None)
                if envelope.get("status") == "success":
                    computed.append((envelope, inputs))
                if envelope.get("status") != "success" and str(envelope.get("question") or "").strip():
                    questions.append(str(envelope["question"]).strip())
            else:
                shown = _shown_to_model(json.dumps(_without_missing_pages(result_of_call), default=str))
                # The model's own notes are never rows for the user: a note
                # kept as a list of records is not added to the answer as a table.
                if isinstance(envelope, dict) and envelope.get("status") == "success" and not is_notes(name):
                    tabled.extend(render.record_lists(envelope))
            msgs.append({**tool_msg, "content": shown})
            evidence.append({**tool_msg, "content": full})
            answer_exit.note_tool_message(call, evidence[-1])
        if work is not None:
            trace[-1]["sheet"] = json.loads(sheet.render(work))
        # Off the sheet, a calculator that rejected or lacks an input ends the
        # turn with its own question: the model is never given the chance to
        # compute a figure the calculator refused. On the sheet no code ends a
        # turn with a tool's question: the needs-input result went back to the
        # model as the tool result, and the model places it -- computes an
        # earlier step, reads a document, or records it under missing and asks.
        if questions and work is None:
            for question in dict.fromkeys(questions):
                event = await off_loop(_release, question + "\n", model_text=False)
                if event:
                    yield event
            answer_done = True
            break
    if not answer_done and refused and work is None:
        # Out of steps with a workflow still refusing: its deliverable comes
        # from the workflow, so the turn asks for what it needs. On the sheet
        # the closing call below answers, naming what is still missing.
        for request in dict.fromkeys(routes.input_request(w, r) for w, r in refused.items()):
            event = await off_loop(_release, request + "\n", model_text=False)
            if event:
                yield event
        answer_done = True
    if not answer_done:
        # Out of steps: answer from the evidence already gathered, no tools.
        ceiling = True  # the round (or seconds) ceiling ended the tool rounds
        msgs.append({"role": "system", "content": _FINISH})
        finished = False
        buffer = ""
        opening.clear()
        written[:], streamed[0] = [0, 0], 0
        finish = computed_answer.NORMAL_FINISH
        async for kind, value in llm.stream(agent, msgs, [], model=driver.model()):
            if kind == "done":
                finished = bool(value and value.get("status") == "success")
                finish = str((value or {}).get("finish_reason") or finish)
                break
            buffer += value
            streamed[0] += len(value)
            while True:
                cut = _SENTENCE_END.search(buffer)
                if cut is None:
                    break
                sentence, buffer = buffer[:cut.end()], buffer[cut.end():]
                event = await off_loop(_release, sentence)
                if event:
                    finished = True
                    yield event
        if buffer:
            event = await off_loop(_release, buffer)
            if event:
                finished = True
                yield event
        event = _settle(finish)
        if event:
            finished = True
            yield event
        trace.append({"step": "finish", "finish": finish, "withheld": withheld.get("reason")})
        model_answered = finished
        if not finished or not "".join(sent).strip():
            event = await off_loop(_release, _STOP, model_text=False)
            if event:
                yield event
    if model_answered and failed and not succeeded:
        # The failure itself, in the tool's own words, when the answer does
        # not already say it: the user learns what could not be done and why.
        written = "".join(sent)
        for shown_name, error in dict.fromkeys(failed):
            if error.lower() in written.lower():
                continue
            from app.agents import answer_exit

            # The tool's own words, unless they carry internal tokens: the
            # shared exit check gives those as one plain sentence.
            line = answer_exit.check_text(f"{shown_name} could not complete: {error}")
            if not line.strip() or line.strip().lower() in written.lower():
                continue
            sent.append(("\n\n" if written.strip() else "") + line + "\n")
            yield {"type": "token", "content": sent[-1]}
            written = "".join(sent)
    if model_answered:
        # A result the platform computed reaches the user even when the
        # model's words stop short of it (a table header with no rows, a
        # heading with nothing under it): what the answer does not carry is
        # added in the renderer's words, under a heading left empty.
        for envelope, inputs in computed:
            written = "".join(sent)
            if render.carried(envelope, inputs, written):
                continue
            block = "\n".join(render.result_lines(envelope, inputs))
            if not block:
                continue
            last = written.rstrip("\n").rsplit("\n", 1)[-1]
            if _HEADING.match(last):
                piece = ("" if written.endswith("\n") else "\n") + block + "\n"
            else:
                piece = "\n\n" + block + "\n"
            event = await off_loop(_release, piece, model_text=False)
            if event:
                yield event
    if model_answered:
        # Rows a tool returned per period or per item reach the user as one
        # row each (Lane 1: a cash flow turned sideways, or Month 1 as
        # bullets): when the answer shows a list but not row by row, the
        # list is added as a table in that orientation.
        for rows in tabled:
            written = "".join(sent)
            if not render.rows_presented(rows, written) or render.rows_carried(rows, written):
                continue
            block = "\n".join(render.record_table(rows))
            if block:
                event = _release("\n\n" + block + "\n", model_text=False)
                if event:
                    yield event
    if model_answered and evidence and not has_words("".join(sent)):
        # The model's reply carried no words and nothing above stood in for
        # it: the same recovery every other path uses answers from this
        # turn's tool results (a computed figure, or what was asked and why
        # there is none) -- never a blank answer. It reads the turn's own
        # messages, so each tool result keeps the call (and the arguments)
        # that asked for it.
        from app.agents.runtime import _recover_answer_from_tool_messages

        recovered = await off_loop(_recover_answer_from_tool_messages, "", _trail())
        if has_words(recovered):
            event = await off_loop(_release, recovered.rstrip() + "\n", model_text=False)
            if event:
                yield event
    if model_answered and not has_words("".join(sent)):
        # Nothing above stood in for a reply that was withheld or carried no
        # words: say the reply was incomplete, in the question's language --
        # never "not found", which only a search that found nothing may say.
        lab = computed_answer.labels_for(computed_answer.question_language(user_message))
        event = await off_loop(_release, lab["incomplete"] + "\n", model_text=False)
        if event:
            yield event
    from app.agents import citation_provenance as cp

    # Once, at the end: the calculator credit the answer does not carry, and
    # the note if any piece lost an unbacked attribution.
    notes = cp.closing_notes("".join(sent), _trail(), tool_passages=True,
                             removed=cp._REMOVED_IN_PIECE.get())
    credit = sources.tool_credit_lines(ran)
    if credit and credit not in "".join(sent):
        notes = "\n\n" + credit + notes
    if notes.strip():
        yield {"type": "token", "content": notes}
    record_id = None
    if sheet_mode:
        ended = "ceiling" if ceiling else "answer" if model_answered else "asked_input"
        if work is not None and ended == "answer" and sheet.unasked(work):
            # The model ended the turn with inputs still missing on its sheet.
            # Every sheet write closes an input whose step has its result
            # (sheet.close_answered), so these wait on steps that never ran:
            # its answer is where it asked for them. Mark them asked, so the
            # next turn does not ask again.
            work = sheet.mark_asked(work)
            ended = "asked_input"
        record = trace_record.build(
            turn=(work or {}).get("turn", ""), question=user_message, rounds=trace,
            final_answer="".join(sent), figures=provenance, ended=ended,
            question_set=question_set, seconds=time.monotonic() - t0,
            provider=str(getattr(agent, "provider", "") or ""), model=driver.model(),
            temperature=llm.DECISION_TEMPERATURE, seed=llm.DECISION_SEED,
            build_sha=_build_sha(), project_setting=driver.CATALOGUE,
            sheet=work)
        record_id = await off_loop(trace_store.save, record, conversation_id,
                                   user_id=user_id, project_id=project_id)
    _LOG.info("DRIVER_TURN %s", json.dumps({
        "conversation": str(conversation_id or ""),  # an id, never the user's words
        "steps": trace, "ended": ("answer" if model_answered else "question" if answer_done else "step_limit"),
        "figures_removed": check["figures_removed"], "identifiers_replaced": check.get("identifiers_replaced", 0),
        # Lengths only: the answer's characters after each stage, what was
        # shown, and why a reply was withheld with its lengths (never text).
        "stage_chars": stage_chars, "shown_chars": len("".join(sent)),
        "withheld": withheld, "trace_record": record_id,
        "elapsed_s": round(time.monotonic() - t0, 2)}, default=str))
    yield {"type": "end", "complete": True, "mode": "driver", "hat": hat, "tools": used, "exports": exports,
           "provenance": provenance, "exit_check": check,
           "sources": sources.panel(_trail(), provenance, "".join(sent),
                                    calculated=any(n == "construction_calc" for n, _ in ran)),
           "steps": step + 1, "elapsed_s": round(time.monotonic() - t0, 2),
           **({"trace_record": record_id} if sheet_mode else {})}
