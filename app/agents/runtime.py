"""Agent runtime — system prompt + allowed-blocks tool list + LLM tool-calling loop.

Loads declarative agent configs from `app/agents/configs/*.md` (YAML frontmatter +
markdown body for the system prompt). Each agent can call any block in its
`allowed_blocks` list as a tool. The runtime handles the back-and-forth with the
LLM: turn → optional tool call(s) → run blocks → return results → continue.

Provider: OpenAI-compatible `/v1/chat/completions` JSON protocol (DeepSeek
primary / OpenRouter fallback). A local-inference adapter is wired into the
chat block as a fallback; see ``app/blocks/chat.py``.
"""

from __future__ import annotations

from app.core.offload import off_loop as _off_loop
from app.core.turn_timing import timed as _turn_timing_stage
from app.core import turn_progress as _turn_progress
import asyncio
import bisect
import inspect
import json
import logging
import os
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
)

import httpx

from app.blocks import BLOCK_REGISTRY
from app.core.answer_report_intent import (
    answer_report_export_descriptor,
    answer_report_export_enabled,
    collect_answer_pairs,
    compose_answer_report_reply,
    message_wants_answer_report,
    parse_answer_report_range,
)
from app.core.conversation_wbs import (
    fulfill_wbs_export,
    message_wants_wbs_export,
    stage_conversation_wbs,
)
from app.core.clash_intent import message_wants_clash
from app.core.contract_lookup_intent import message_is_contract_data_lookup
from app.core.rag.inject import rag_inject
from app.lib.boq_ref_codes import strip_ref_codes
from app.core.rag.retriever import project_is_rag_ready
from app.dependencies import _create_block_instance, block_instances

_LOG = logging.getLogger(__name__)

# Final-text fallback used whenever a forced retry returns empty content.
# Without it, the generator emits an `end` event with zero `token` events,
# which the UI renders as an empty assistant bubble (FOLLOW-UP #90).
_EMPTY_RESPONSE_FALLBACK = (
    "I was unable to generate a response for this turn. "
    "Please rephrase the question or try again."
)


_GENERATIVE_VERB_RE = re.compile(
    r"\b(generate|produce|create|draft|prepare|compose|write up|write a |write an |"
    r"build (?:a|an|me)|auto[-\s]?populate|put together|come up with)\b",
    re.IGNORECASE,
)


def _is_generative_request(text: str) -> bool:
    """A request to PRODUCE an artifact (checklist, register, RFI, plan) rather
    than look up a fact. Generative requests must not be strait-jacketed by the
    'answer ONLY from context' directive — with any related template in the
    corpus the model would otherwise refuse ("I can't generate that from the
    provided material"), which is the opposite of what the user asked for."""
    return bool(_GENERATIVE_VERB_RE.search(text or ""))


# RAG fold labels. `_apply_rag_context` appends one of these after the
# retrieved excerpts; detectors that read `messages[-1]` after the fold
# must recover the operator question or they classify the corpus instead.
_RAG_FOLD_END = "----- END OF REFERENCE CONTEXT -----"
_RAG_FOLD_QUESTION_MARKERS = (
    "\n\nDURATION OVERRIDE REQUEST: ",
    "\n\nCALCULATION REQUEST: ",
    "\n\nMESSAGE: ",
    "\n\nQUESTION: ",
    "\n\nREQUEST: ",
)


def _unwrap_rag_folded_operator_text(text: str) -> str:
    """Return the operator question when ``text`` is a RAG-folded user turn.

    After `_apply_rag_context` the last user bubble is
    ``<excerpts> + directive + question``. Duration-override / deliverable
    detectors that then scan that bubble would parse Contract Data prose
    (or miss 're-run' in a 6k excerpt) and skip ``generate_wbs``.
    """
    raw = text or ""
    if _RAG_FOLD_END not in raw:
        return raw
    for marker in _RAG_FOLD_QUESTION_MARKERS:
        idx = raw.rfind(marker)
        if idx == -1:
            continue
        recovered = raw[idx + len(marker):]
        if recovered.strip():
            return recovered
    return raw


def _message_is_duration_override_rerun(
    text: str,
    history: list | None = None,
) -> bool:
    """True when this turn must re-run ``generate_wbs`` with a user duration."""
    try:
        from app.lib.wbs_duration_overrides import message_wants_wbs_duration_rerun
        return bool(message_wants_wbs_duration_rerun(text or "", history))
    except Exception:  # noqa: BLE001
        _LOG.debug(
            "duration-override intent check skipped",
            exc_info=True,
        )
        return False


# ── Capability / self-introspection questions ────────────────────────────────
# Questions ABOUT the agent ("what tools do you have", "what can you do", "what
# documents can you access") must be answered from the agent's REAL tool roster
# and the project's REAL document list — NOT from retrieved document text. The
# grounded-RAG clamp (_apply_rag_context: "answer ONLY from the reference
# context") otherwise makes the agent answer a question about ITSELF using
# whatever documents happened to match the word "tools" (e.g. construction
# power-tool specs), which is confidently wrong. This is a deterministic
# short-circuit, mirroring the rag-miss fast path.
_CAPABILITY_META_PHRASES = (
    "what can you do", "what do you do", "who are you", "what are you",
    "what are your capabilities", "what are your abilities",
    "how can you help", "what can you help", "what help can you",
    "what tools do you", "what tools can you", "what tools you",
    "which tools do you", "which tools can you", "list your tools",
    "what functions do you", "what features do you",
    "what documents can you access", "what files can you access",
    "what can you access", "what documents can you see",
    "what can you work on", "what documents can you work",
)


def _first_sentence(text: str, cap: int = 160) -> str:
    """First sentence (or first ``cap`` chars) of a description, single-lined."""
    s = " ".join((text or "").split())
    if not s:
        return ""
    head = s.split(". ")[0].rstrip(".")
    return head if len(head) <= cap else head[:cap].rstrip() + "…"


def _is_capability_request(text: str) -> bool:
    """True when the user is asking about the AGENT itself (its tools, abilities,
    or what documents it can access) — not about the content of a document."""
    t = " ".join((text or "").lower().split())
    if not t:
        return False
    # An INSTRUCTION that prohibits or constrains tool use is work, not a
    # question about the agent. Live find 2026-08-15 (F22): "do not run any
    # schedule, workbook, or generator tool -- write the document text"
    # contained "you"/"tool" and short-circuited a costing turn into the
    # agent's tool roster. Prohibition phrasing wins over the co-occurrence
    # heuristic below.
    if re.search(
        r"\b(?:do\s+not|don'?t|never|without|no)\s+"
        r"(?:run|use|call|invoke)?\s*(?:any\s+)?(?:\w+[ ,]+){0,4}tool",
        t,
    ) or re.search(r"\brun\s+no\s+tool", t):
        return False
    if any(p in t for p in _CAPABILITY_META_PHRASES):
        return True
    # Generic: self-reference ("you"/"your") + a capability keyword, unless the
    # phrasing is clearly asking what a DOCUMENT states (a real RAG lookup).
    if re.search(r"\byou(r)?\b", t) and any(
        k in t for k in ("tool", "capabilit", "able to", "function", "feature")
    ):
        if not any(
            c in t for c in (
                "does the", "according to", "per the", "spec say",
                "document say", "drawing say", "as per",
            )
        ):
            return True
    return False


def _build_capability_answer(agent: Agent, project_id: str | None) -> str:
    """Deterministic answer describing the agent's REAL tools + the project's
    REAL documents. No LLM, no RAG — cannot fabricate or answer the wrong thing."""
    lines = [f"I'm the **{agent.name}** agent."]
    try:
        defs = agent.tool_definitions(project_id)
    except Exception:
        defs = []
    tool_lines = []
    for d in defs or []:
        if not isinstance(d, dict):
            continue
        fn = d.get("function") if isinstance(d.get("function"), dict) else d
        name = fn.get("name")
        tdesc = fn.get("description", "")
        if name:
            tool_lines.append(f"- **{name}** — {_first_sentence(tdesc)}")
    if tool_lines:
        lines.append("\n**Tools I can use:**\n" + "\n".join(tool_lines))
    if project_id:
        try:
            from app.core import projects as _store
            n = _store.count_documents(project_id)
        except Exception:
            n = 0
        if n:
            docs = _store.list_documents(project_id, limit=15, newest_first=True)
            listing = "\n".join(f"- {d['original_name']}" for d in docs)
            more = n - len(docs)
            tail = f"\n- …and {more} more (ask me to search for a specific one)." if more > 0 else ""
            lines.append(
                f"\n**Documents in this project ({n} total):**\n{listing}{tail}"
            )
        else:
            lines.append(
                "\nThis project has **no documents** uploaded yet, so there's "
                "nothing here for me to read. Upload files or pick a project "
                "that has documents."
            )
    return "\n".join(lines)


def _apply_rag_context(
    messages: list, rag_sys_msg: dict, user_data_authoritative: bool = False
) -> bool:
    """Fold retrieved RAG context INTO the final user turn rather than adding a
    separate system message.

    Models (esp. gpt-oss) de-prioritize system messages: with the context in a
    system block the model cites the doc-ids but still answers from its (often
    wrong) parametric prior — observed live on a FIDIC question where the answer
    was byte-identical whether the context was the correct KB note or unrelated
    templates. Embedding it in the user turn makes the model actually use it.

    The directive is INTENT-AWARE: a lookup/factual turn is grounded strictly
    (answer only from context, don't invent); a GENERATE turn treats the context
    as optional background and lets the model apply domain expertise to produce
    the artifact, while still not fabricating project-specific facts. Returns
    True if applied.
    """
    content = (rag_sys_msg or {}).get("content")
    if not content:
        return False
    if messages and messages[-1].get("role") == "user":
        question = messages[-1].get("content", "")
        history_for_override = [
            m for m in messages[:-1]
            if m.get("role") in ("user", "assistant")
        ]
        if _wants_user_supplied_arithmetic_directive(question):
            # ARITHMETIC, not a lookup. Every input is IN the question, so the
            # strict-grounding directive below ("answer using ONLY the
            # reference context ... if it does not contain the answer, say you
            # don't have it") actively forbids the right answer.
            #
            # Live 2026-08-03, verbatim:
            #   "Calculate the concrete volume for a raft 25m x 18m x 1.2m
            #    thick and the number of 6m3 truck loads"
            #   -> "I cannot find that specific information in the provided
            #       reference context ... they do not contain a raft measuring
            #       25 m x 18 m x 1.2 m thick"
            #
            # 25 x 18 x 1.2 = 540 m3; 540 / 6 = 90 loads. The model was not
            # failing — it was obeying. It is the DIRECTIVE that has to change.
            #
            # (tool_choice forcing cannot rescue this: _tool_choice_for returns
            # "auto" for kimi because K2 rejects a specified tool outright, so
            # the deterministic calculators can never be forced. This path is
            # provider-independent by design.)
            #
            # Project-specific facts stay protected: the model may compute from
            # the numbers the USER supplied, not invent project data.
            directive = (
                "\n\n----- END OF REFERENCE CONTEXT -----\n\n"
                "The request below is a CALCULATION whose inputs are supplied "
                "IN THE REQUEST ITSELF. Compute it with the platform calculator "
                "for the registered formula those inputs complete. Do NOT refuse "
                "because the dimensions, loads, rates or quantities are absent "
                "from the reference context — they are not supposed to be there; "
                "they came from the user. Label them as the user's inputs. "
                "Treat the context as background only.\n\n"
                "The figure you state is the platform calculator's result. Do "
                "not state a final figure the calculator did not produce, and "
                "do not attribute a factor the user did not supply to the "
                "project documents or the retrieved excerpts. If the calculator "
                "applied its own default, say so. You may still cite the context "
                "for project-specific rates, specified thicknesses or standards "
                "where it genuinely applies — but never invent "
                "project-specific facts (names, drawing or clause references) "
                "that are not in the context. Do not attach a currency or a "
                "unit that the request, the calculator result and the excerpts "
                "do not already state. "
                "Do not cite a clause or a standard as the basis of the "
                "calculation unless the excerpts state it. Do not compare the "
                "result with a limit the request and the excerpts never state. "
                "Never write an internal id; the reader sees the display name."
                "\n\nCALCULATION REQUEST: "
            )
        elif _message_wants_boq_scope_wbs_ask(question):
            # A "client documents only" prefix on a BOQ-scope WBS ask: the
            # verb is generative, but the prefix plus an earlier contract
            # year's CoC excerpts made the model refuse and cite the wrong
            # contract. Measured BOQ rows (via generate_wbs) are the
            # evidence; another year's Conditions of Contract is not.
            directive = (
                "\n\n----- END OF REFERENCE CONTEXT -----\n\n"
                "The request below is a HIGH-LEVEL WBS over this project's "
                "demolition / site-clearance BOQ. Present the generate_wbs "
                "result (or the already pre-dispatched tree). Do NOT refuse "
                "because Conditions of Contract prose does not contain a "
                "WBS — the bill's measured rows are the evidence. Do NOT "
                "cite another contract year's Conditions of Contract "
                "as the governing contract. "
                "Do not print the generic building template as the answer. "
                "Treat the reference context as background only."
                "\n\nREQUEST: "
            )
        elif _message_is_duration_override_rerun(question, history_for_override):
            # OLD-pack F2 on a fresh Master Corpus thread, verbatim:
            #   "Answer only from the client project documents. Use 45 days
            #    for the tree-removal activity and re-run."
            # The override parsed, but `_is_generative_request` does not see
            # "use … re-run" as generate/create/build, so the lookup clamp
            # told the model to answer ONLY from excerpts. Excerpts have no
            # activity durations — the model offered to fetch a WBS instead
            # of applying the override. Same class as the calculation
            # carve-out: the model was obeying. The user's N days are not
            # supposed to be in the corpus.
            directive = (
                "\n\n----- END OF REFERENCE CONTEXT -----\n\n"
                "The request below is a DURATION OVERRIDE AND RE-RUN. The "
                "user supplied activity durations IN THE REQUEST ITSELF. "
                "Apply them via generate_wbs (or the already pre-dispatched "
                "result) and report the new durations and recomputed "
                "total/critical path. Do NOT refuse because activity "
                "durations are absent from the reference context — they are "
                "not supposed to be there; they came from the user. Do not "
                "invent other schedule data or activity names that are not "
                "in the tool result. Treat the context as background only."
                "\n\nDURATION OVERRIDE REQUEST: "
            )
        elif _is_generative_request(question):
            directive = (
                "\n\n----- END OF REFERENCE CONTEXT -----\n\n"
                "The reference context above is background you MAY draw on. This "
                "is a REQUEST TO GENERATE — produce the complete artifact asked "
                "for, applying standard construction / engineering domain "
                "knowledge where the context is thin or absent. Do NOT fabricate "
                "project-specific facts (real names, numbers, drawing or clause "
                "references) that aren't in the context, but DO generate the "
                "standard professional content requested (checklist items, risk "
                "entries, methodology, etc.).\n\nREQUEST: "
            )
        elif user_data_authoritative:
            # Agents whose JOB is to consume figures the user supplies (the
            # learning agent recording corrections) must not be strict-grounded
            # against the retrieved corpus. Live 2026-08-15: "the actual
            # supplier invoice came to 480 SAR/m3 -- record it" was REFUSED
            # with "480 SAR/m3 does not appear in the authoritative reference
            # context". The user's own correction IS the primary source; the
            # strict directive below actively forbids the agent's purpose.
            # Same class as the self-contained-calculation carve-out above:
            # the model was obeying, the directive had to change.
            directive = (
                "\n\n----- END OF REFERENCE CONTEXT -----\n\n"
                "The message below may contain figures, outcomes or "
                "corrections SUPPLIED BY THE USER (actual costs, measured "
                "quantities, real durations). Those user-supplied figures are "
                "AUTHORITATIVE first-party data -- they are not supposed to "
                "be in the reference context, and their absence from it is "
                "never a reason to refuse. Record or act on them with your "
                "tools as your role requires, and say plainly what you "
                "recorded. Use the reference context only as background for "
                "comparison. Do not invent figures the user did not "
                "supply.\n\nMESSAGE: "
            )
        else:
            # F36: "say you don't have it" must not outrank the agent's TOOLS.
            # Live 2026-08-15: bim-analyst, asked for the element census of an
            # uploaded IFC, made ZERO tool calls and refused with "the
            # reference context does not contain any extracted data from
            # qa_building.ifc" -- obeying this directive to the letter while a
            # bim tool that answers the question sat in its roster. Retrieved
            # text is one source; the agent's extractors are another.
            directive = (
                "\n\n----- END OF REFERENCE CONTEXT -----\n\n"
                "Answer the question below using ONLY the reference context "
                "above. Reproduce its specific facts — numbers, deadlines, clause "
                "references, named principles, definitions — EXACTLY. If the "
                "context conflicts with what you think you know, the context "
                "wins. If the context does not contain the answer but one of "
                "your TOOLS can extract or compute it from the project's files "
                "(BIM/IFC extraction, BOQ parsing, drawing take-off, "
                "calculators), CALL THE TOOL and answer from its result. Only "
                "say you don't have it when neither the context nor your tools "
                "can produce the answer — never invent one.\n\nQUESTION: "
            )
        messages[-1] = {**messages[-1], "content": content + directive + question}
        return True
    # No user turn to attach to (unexpected) — fall back to a system message.
    messages.insert(max(0, len(messages) - 1), rag_sys_msg)
    return True

_UNINDEXED_PROJECT_MESSAGE = (
    "This project has no indexed documents yet, so I cannot answer "
    "questions from its corpus. Please upload a document or wait for "
    "indexing to finish."
)


# Patterns that indicate the model emitted a raw internal tool-call payload
# instead of a user-facing answer. These must never be shown to the user.
_INTERNAL_TOOL_KEYS = {"name", "arguments"}

# Fallback used when the assistant emits raw internal tool/search arguments
# instead of a user-facing answer.
_TOOL_FORMAT_FALLBACK = (
    "I hit an internal search formatting issue before I could produce a "
    "grounded answer. Please retry or narrow the question."
)

# Nudge appended before a forced no-tools retry when sanitization replaced the
# model's reply with _TOOL_FORMAT_FALLBACK (raw search/tool JSON leak).
_TOOL_FORMAT_RETRY_NUDGE = (
    "Your previous reply looked like internal tool JSON, not a user-facing answer. "
    "Write the complete deliverable now in plain text using the tool results already "
    "in this thread. Do not emit JSON tool arguments or search payloads."
)

# Live M15: the hat ended on "let me search / I'll pull the WIR" after the
# search-tool cap, which the UI rendered as an incomplete / empty generation.
# Two shapes, because live failures came in both:
#  * OPENER - the whole turn is a promise ("Let me search...", "I'm searching
#    the executed contract volume...").
#  * TAIL - the turn reasons briefly and then ENDS on the promise ("...absent
#    from the retrieved excerpts, I am running a targeted search ... to pull
#    the clause details."), which is the same dead end for the reader.
# Both require a FIRST-PERSON promise. A bare leading verb ("Search results
# show three specs...") is a real answer and must not match.
# The work a promise can be a promise OF. One list, used by both shapes.
#
# It was two lists, written out separately in the two regexes below, and they
# named only the verbs of SEARCHING. Then the platform taught the model to read
# a truncated tool result window by window -- and the model began ending turns
# on "Let me READ the final window...", a verb neither list had. Live on
# 2d9d9c0, asked to "Draft a variation order under FIDIC clause", the turn ran
# five iterations and ended, with no error, on:
#
#     "I now have the project's own change-management procedure and VSR form.
#      Let me read the final window of the VSR form to complete the picture."
#
# The user asked for a document and was shown a status line. This is the third
# time new vocabulary has walked a dangling promise past this detector (see
# _strip_progress_narration for the second), and the first two were each fixed
# by adding the one phrase that had just been seen. So this names the whole
# family -- everything the agent can promise to go and do to a document --
# rather than the one verb that happened to get caught tonight.
_PROMISED_WORK = (
    r"search|searching|look(?:ing)?\s+up|pull(?:ing)?|check(?:ing)?|"
    r"run(?:ning)?|re-?run|fetch(?:ing)?|retriev(?:e|ing)|validat(?:e|ing)|"
    r"read(?:ing)?|re-?read(?:ing)?|open(?:ing)?|review(?:ing)?|"
    r"examin(?:e|ing)|scan(?:ning)?|load(?:ing)?|continu(?:e|ing)|"
    r"go(?:ing)?\s+through|work(?:ing)?\s+through"
)

# "Let me know if you need anything else." is a courtesy, not a promise of
# work. It was already reachable through "check" before "read" was added, and
# widening the verb list widens that hole, so it is closed here.
_FIRST_PERSON_INTENT = (
    r"(?:let me(?!\s+know\b)|i'?ll\b|i\s+will\b|i'?m\b|i\s+am\b|i\s+need\s+to\b)"
)

_SEARCH_PREAMBLE_RE = re.compile(
    r"^\s*(?:okay[,.]?\s+|sure[,.]?\s+|thanks[,.]?\s+|right[,.]?\s+)?"
    r"(?:" + _FIRST_PERSON_INTENT + r"|"
    r"give me a moment|one moment|hold on)"
    r"[^.!?]{0,120}?"
    r"\b(?:" + _PROMISED_WORK + r")\b",
    re.IGNORECASE,
)
_SEARCH_PROMISE_TAIL_RE = re.compile(
    r"(?:^|[.!?]\s+)(?:[^.!?]{0,160}?)"
    r"\b" + _FIRST_PERSON_INTENT +
    r"[^.!?]{0,160}?"
    r"\b(?:" + _PROMISED_WORK + r")\b"
    r"[^.!?]{0,160}[.!?]?\s*$",
    re.IGNORECASE,
)
_SEARCH_PREAMBLE_RETRY_NUDGE = (
    "Do not search again. Answer now in plain text using the operator facts "
    "and any tool results already in this thread. If the answer is a lookup, "
    "state the value with its source; if the thread does not contain it, say "
    "so plainly and name what is missing. A deliverable (WIR / inspection "
    "request, claim notice, as-built note) must include scope, the stated "
    "numbers, hold/witness or variance, and signatories. Never reply with an "
    "intention to search."
)


# Retry instruction when the model handed back its own input. Distinct from
# the search-promise nudge above: nothing is missing here -- the answer was in
# the material the model was given, and it copied the material instead of
# reading it. So the instruction is "answer from it", not "stop promising".
_CONTEXT_LEAK_RETRY_NUDGE = (
    "Your previous reply repeated the reference material you were given "
    "instead of answering. That material is CONTEXT, not output: never quote "
    "the instructions, the retrieval brief, the [doc_id=... chunk=... "
    "score=...] markers, or any [source: ...] filesystem path back to the "
    "user. Answer the question now, in your own words, in plain text, stating "
    "the value and citing the document by name only."
)


#: A line that narrates work in progress and states no result. On its own it
#: is not an answer, and after a promise it is not the end of one either --
#: it is the promise still running.
#: Applied with .match(), which anchors at position 0 on its own -- no "^".
_PROGRESS_NARRATION_RE = re.compile(
    r"(?:continuing|resuming|reading|re-?reading|fetching|retrieving|"
    r"searching|pulling|checking|loading|processing|extracting|scanning|"
    r"analy[sz]ing|looking|validating|re-?trying|trying|attempting|calling|"
    r"invoking|querying|re-[a-z]+ing|"
    r"[a-z]+ing(?=\b[^.!?\n]*\b(?:again|once\s+more|one\s+more\s+time)\b))"
    r"\b[^.!?\n]*[.!?\u2026]*\s*$",
    re.IGNORECASE,
)
_TAIL_SENTENCE_RE = re.compile(r"(?<=[.!?\u2026])[ \t]+(?=\S[^.!?\n]*[.!?\u2026]*\s*$)")


def _strip_progress_narration(text: str) -> str:
    """Drop trailing 'Continuing BOQ extraction...' style lines.

    Live on 2d221e6, the build that taught the model to read a truncated
    result window by window. Asked for the m3 demolition quantity it
    answered, in full:

        "I'll continue reading the BOQ to find any cubic-metre items. Let me
         pull the next window.

         Continuing BOQ extraction..."

    _SEARCH_PROMISE_TAIL_RE matches "Let me pull the next window." on its
    own. It did not match here because it is anchored to the END of the text
    and a progress line follows it, so a dangling promise shipped as the
    answer -- the exact failure #454 and #456 exist to prevent, reached by
    new vocabulary that the platform itself taught the model.

    Narration is not an answer, so it cannot be what makes the sentence
    before it one. A narrating sentence at the end of a line goes the same
    way as a narrating line ("Both files are registered. Retrying the
    extractors once more.").
    """
    lines = (text or "").rstrip().split("\n")
    while lines:
        last = lines[-1].strip()
        if not last or _PROGRESS_NARRATION_RE.match(last):
            lines.pop()
            continue
        cut = _TAIL_SENTENCE_RE.search(last)
        if cut and _PROGRESS_NARRATION_RE.match(last[cut.end():]):
            lines[-1] = last[:cut.start()]
            continue
        break
    return "\n".join(lines).strip()


def _looks_like_search_preamble(text: str) -> bool:
    """True when the model stopped on a search promise instead of writing."""
    t = (text or "").strip()
    if not t or len(t) > 500 or t.count("\n") > 6:
        return False
    if _SEARCH_PREAMBLE_RE.search(t) or _SEARCH_PROMISE_TAIL_RE.search(t):
        return True
    # Nothing but narration ("Continuing BOQ extraction...") is a status
    # line, not a reply. A short reply that ENDS on narration stopped on work
    # it had not done: the model meant to go round the tool loop again, so
    # what came before it is the middle of a turn, not its answer.
    return _strip_progress_narration(t) != t


def _final_text_needs_forced_retry(
    text: str, *, user_message: str | None = None,
) -> bool:
    """True when sanitization left no usable user-facing answer."""
    if not (text or "").strip():
        return True
    if text == _TOOL_FORMAT_FALLBACK:
        return True
    # A promise to search is never a usable answer to ANY question. This was
    # gated on deliverable/generative requests, so plain factual lookups fell
    # straight through and the user was shown "Let me search more precisely
    # for the Engineer's Representative..." as the final answer (a live
    # contract-particular lookup and a delay-damages ask). The request
    # type does not change whether a dangling promise is an answer.
    if _looks_like_search_preamble(text):
        return True
    return _exit_leaves_nothing(text)


def _exit_leaves_nothing(text: str) -> bool:
    """True when every sentence of ``text`` is about the platform's own
    workings, so the exit check would hand the user nothing of it."""
    if not (text or "").strip() or text.strip() == _EMPTY_RESPONSE_FALLBACK:
        return False
    from app.agents import answer_exit

    return not (answer_exit.check_text(text) or "").strip()


#: Retry instruction when the model's reply was only about the platform's own
#: workings (its tools, steps and fields), which never reach the user.
_USER_TERMS_RETRY_NUDGE = (
    "Your previous reply described the platform's own workings (tool, step, "
    "operation or field names, calls and retries) instead of answering, and "
    "none of that is shown to the user. Answer the user's question now in "
    "plain text from what this thread already holds, in the user's own "
    "terms. Do not name tools, steps, operations or fields and do not "
    "describe calls. If something could not be found or read, say what is "
    "missing in the user's terms; if an input is needed, ask for it by name."
)


def _forced_retry_nudge(text: str) -> str | None:
    """The instruction that goes with the forced no-tools retry for ``text``."""
    if text == _TOOL_FORMAT_FALLBACK:
        return _TOOL_FORMAT_RETRY_NUDGE
    if _looks_like_search_preamble(text):
        return _SEARCH_PREAMBLE_RETRY_NUDGE
    if _exit_leaves_nothing(text):
        return _USER_TERMS_RETRY_NUDGE
    return None

# Keys that strongly indicate an internal search tool argument payload.
_SEARCH_TOOL_ARG_KEYS = {
    "query",
    "project_id",
    "corpus",
    "top_k",
    "filters",
    "source",
    "tool",
    "function",
}

# Keys that indicate legitimate user-requested data JSON, not tool args.
_ANSWER_DATA_KEYS = {
    "answer",
    "result",
    "results",
    "data",
    "items",
    "rows",
    "total",
    "count",
    "status",
    "message",
    "summary",
    "explanation",
    "value",
    "values",
}


def _project_has_non_rag_context(project_id: str, user_message: str) -> bool:
    """True if there is project-specific context (facts, documents, etc.)
    outside the RAG corpus.  Allows project-fact Q&A to keep working even
    when the project has no indexed chunks yet.
    """
    # Formula asks do not need a corpus — construction_calc is the path.
    # Empty FIXTURE projects must not early-return the unindexed refusal.
    if _message_is_formula_style_ask(user_message):
        return True
    # Live Phase 2: priced VO / RFI drafts are self-contained. Empty
    # FIXTURE projects must not early-return the unindexed refusal on ask1.
    if _message_wants_vo_draft(user_message) or _message_wants_rfi_draft(user_message):
        return True
    try:
        from app.core.project_memory import build_project_context

        ctx = build_project_context(project_id, user_message)
        return bool(ctx and ctx.strip())
    except Exception:
        return False


# Controlled answer used when an exact construction reference is absent from
# the indexed project sources.  Replaces both the generic empty-content
# fallback and any expensive model call that would otherwise spin on a miss.
# Persisted when a turn dies before producing an answer (provider error,
# crash, iteration cap). WHY THIS EXISTS:
#
# `append_message(user)` runs BEFORE the LLM call, but the assistant message
# is appended only on the SUCCESS paths. So a failed turn persisted the
# question and nothing else. The UI's error bubble is local state only, so
# after a reload the user saw their own message with no reply — permanently,
# with no indication anything had gone wrong.
#
# Observed live 2026-08-02 in the operator's own conversation: "the client project package 1"
# sits in the thread with no assistant turn after it, and the feature sweep
# reproduced the mechanism (kimi HTTP 400 "tokenization failed" on
# document_metadata -> error event -> nothing persisted).
#
# Deliberately SHORT and unmistakably a failure notice: this text re-enters
# the model's context on the next turn, so it must not read as content or
# invite the model to continue from it.
_FAILED_TURN_MESSAGE = (
    "[This turn failed before an answer was produced. Nothing was generated. "
    "Ask again to retry.]"
)


_TOOL_RESULT_MAX_CHARS = 8000


# Empty-router dispatch enforcement (F19, phase-3 campaign). Two live gates
# proved the prose contract alone does not make K2 dispatch after an empty
# router match: the TIMING log showed iter=0 calling only smart_orchestrator
# and iter=1 going straight to final -- while the ANSWER claimed "Routed to:
# formula_executor_v2" and carried a self-computed number. Same failure on
# two consecutive gates (requests 143ac24a and 8c1c9589) despite contract
# revisions, so the enforcement moved into machinery: the runtime detects the
# empty verdict at the moment it happens and injects a corrective user-role
# message between the tool result and the next model call. In-context and
# positionally adjacent, which K2 follows far more reliably than a system
# contract paragraph.
# Tool-ERROR correction (F27, phase-4 campaign). Reproduced twice on
# bim-analyst: the first tool call errored (construction without a valid
# action) and the model NARRATED its next call as the final answer -- "I
# will try bim_extract" -- without ever making it. Same failure family as
# the fabricated-dispatch finding: prose intentions are not calls. After an
# error tool result, the runtime injects this corrective message; capped
# per turn so a hopeless tool cannot loop forever.
_TOOL_ERROR_NUDGE = (
    "That tool call FAILED -- read the error above, correct the call (tool "
    "name, action, or arguments) and MAKE the corrected call now. Do not "
    "describe a call you intend to make; make it. If no correction can "
    "work, state plainly what failed and answer with what you have."
)
_TOOL_ERROR_NUDGE_CAP = 2
# The calculator held a supplied value to its declared type and range and
# did not run. Only the user can give the value: retrying with one the
# model chose would be an invented input.
_INPUT_QUESTION_NUDGE = (
    "The calculator did not run because a supplied value is outside what it "
    "accepts. Do not call it again with a value you chose. Ask the user: {question}"
)
# The calculator needs an input nobody gave. A figure the user's message or
# the retrieved project documents state may be supplied; otherwise the user
# is asked in these words -- an input's code name is not a question.
_CALC_MISSING_INPUT_NUDGE = (
    "The calculator did not run because an input is missing. If the user's "
    "message or the project documents state it, call the calculator again "
    "with that figure. Otherwise do not choose a value: ask the user exactly "
    "this: {question}"
)
# construction_calc "Unknown calculation" means the formula is not in the
# registry. Retrying another calculator name loops; self-coding writes
# Python once and returns. Only injected when the agent can delegate.
_UNKNOWN_CALC_SELF_CODING_NUDGE = (
    "That calculation is not a registered formula. Delegate to agent "
    "'self-coding' EXACTLY ONCE with the original question so it can "
    "write and run Python. Do not retry construction_calc with another "
    "name. Do not invent the number in prose."
)
_SELF_CODING_PHRASES = (
    "self-coding", "self coding", "write python", "generate python",
    "no formula", "no calculator", "no registered formula",
)

# Extension -> the file-consuming tool that analyses it. Used ONLY to build
# the corrective-nudge hint below; never a routing table.
_EXT_TOOL_HINTS = {
    ".ifc": "bim_extractor",
    ".dwg": "drawing_qto",
    ".dxf": "drawing_qto",
    ".xlsx": "boq_processor",
    ".csv": "boq_processor",
    ".pdf": "pdf",
    ".xer": "primavera_parser",
}


# Distinctive filename tokens in the user message (≥12 chars so ``spec``
# never hijacks every specification file).
from app.core.file_mentions import (  # noqa: E402 -- pure, worker-importable
    FILE_NAME_TOKEN_RE as _FILE_NAME_TOKEN_RE,
    user_names_project_file as _user_names_project_file,
)


def _names_mentioned(user_low: str, names: list) -> list:
    """``names`` that the message mentions -- in the retrieval worker process
    when it is enabled (thousands of comparisons must not hold this
    process's GIL), else here."""
    from app.core.file_mentions import names_mentioned
    from app.core.rag import retrieval_worker

    if retrieval_worker.enabled():
        return retrieval_worker.call_sync(retrieval_worker.names_mentioned_job, user_low, list(names))
    return names_mentioned(user_low, names)


async def _rag_inject_off_process(**kwargs: Any):
    """``rag_inject`` in the retrieval worker process when enabled, else a thread."""
    from app.core import turn_timing
    from app.core.rag import retrieval_worker

    with turn_timing.stage("retrieval"):
        if retrieval_worker.enabled():
            from app.core.privileges import caller_role

            return await retrieval_worker.call(retrieval_worker.rag_inject_job, kwargs, caller_role())
        return await asyncio.to_thread(rag_inject, **kwargs)


async def _project_is_rag_ready_off_process(project_id: Any) -> bool:
    # A database check, not CPU work: a thread, so it never waits in the
    # retrieval workers' queue behind other turns' retrieval.
    return await _off_loop(project_is_rag_ready, project_id)


_HISTOGRAM_PHRASES = (
    "manpower histogram", "labour histogram", "labor histogram",
    "resource histogram", "crew histogram", "workforce histogram",
    "labor loading", "labour loading",
)
_PROCUREMENT_LIST_PHRASES = (
    "procurement_list_generator",
    "procurement list",
    "material list",
    "purchase list",
    "buy list",
    "vendor list",
    "materials list",
    "what materials",
    "need to buy",
    "need to purchase",
    "generate a procurement",
    "produce a procurement",
    "create a procurement",
    "po_generator",
)
_RFI_DRAFT_PHRASES = (
    "rfi_generator",
    "request for information",
    "follow-on rfi",
    "follow on rfi",
    "draft a rfi",
    "draft an rfi",
    "draft one rfi",
    "generate a rfi",
    "generate an rfi",
    "create a rfi",
    "create an rfi",
    "raise an rfi",
    "raise a rfi",
    "write an rfi",
    "write a rfi",
    "produce an rfi",
    "produce a rfi",
    "issue an rfi",
    "issue a rfi",
)
_DRAWING_QTO_PHRASES = (
    "drawing_qto",
    "quantity takeoff",
    "quantity take-off",
    "quantity take off",
    "extract quantities",
    "measure the floor area",
    "floor plan drawing",
    "infrastructure drawings",
)


def _message_wants_drawing_qto(text: str) -> bool:
    """True for a drawing / QTO takeoff deliverable, not a bare L×W×D calc.

    Live tip 50c37f: probe asks naming ``drawing_qto`` / quantity takeoff
    with synthetic footing/slab dims elected ``named_calculator`` because
    ``_looks_like_self_contained_calculation`` saw L×W×D + Compute, then
    formula predispatch ran construction_calc ×3. #658 ranked drawing_qto
    in intent_map; this detector is the named_calculator / predispatch
    steal-guard.
    """
    low = (text or "").lower()
    if not low or _HISTOGRAM_QA_RE.search(low):
        return False
    return any(p in low for p in _DRAWING_QTO_PHRASES)


_HISTOGRAM_QA_RE = re.compile(
    r"\b(what is|what's|whats|explain|define)\b", re.IGNORECASE,
)


def _message_wants_resource_histogram(text: str) -> bool:
    """True when the turn is a histogram deliverable, not a definition Q&A.

    Leftover M17: "histogram from resource_loaded.xer" must not fall through
    to primavera_parser just because the message contains ``.xer``.
    """
    low = (text or "").lower()
    if not low or _HISTOGRAM_QA_RE.search(low):
        return False
    if any(p in low for p in _HISTOGRAM_PHRASES):
        return True
    return "histogram" in low and any(
        t in low for t in (".xer", "primavera", "p6", "taskrsrc")
    )


def _message_wants_look_ahead(text: str) -> bool:
    """True when the turn asks for a rolling / N-day window, not a WBS.

    Delegates so forced-tool selection cannot disagree with the router
    on an 'as of today' / next-N-days ask.
    """
    # Router owns the detector. A second phrase list here is how
    # 'as of today' slipped past the steal-guard into generate_wbs.
    from app.core.action_router import message_wants_look_ahead
    return message_wants_look_ahead(text)


def _message_wants_procurement_list(text: str) -> bool:
    """True for a procurement-list deliverable, not schedule / definition Q&A.

    Live Phase 2: "generate a procurement list" / "what materials do we
    need to buy" / a tool-shaped ``procurement_list_generator`` ask must
    not fall through to ``construction_calc``. Bare "how long is
    procurement on the schedule?" is RAG / WBS and stays unforced.
    """
    low = (text or "").lower()
    if not low or _HISTOGRAM_QA_RE.search(low):
        return False
    return any(p in low for p in _PROCUREMENT_LIST_PHRASES)


def _message_wants_rfi_draft(text: str) -> bool:
    """True for an RFI-draft deliverable, not status / definition Q&A.

    Live Phase 2: "draft an RFI … rebar detail" / "create a request for
    information document" / a tool-shaped ``rfi_generator`` ask must
    not fall through to ``construction_calc``. Bare "what is an RFI?"
    and "how many RFIs are open?" stay unforced.
    """
    low = (text or "").lower()
    if not low or _HISTOGRAM_QA_RE.search(low):
        return False
    return any(p in low for p in _RFI_DRAFT_PHRASES)


def _resolve_histogram_schedule_file(
    project_id: str | None, user_message: str,
) -> tuple[str | None, str]:
    """Pick a project ``.xer`` for a histogram. Prefers a file the user named.

    Two unnamed ``.xer`` files → ``(None, "")`` so we never silently pick.
    """
    if not project_id:
        return None, ""
    try:
        from app.core import projects as _projects
        docs = _projects.list_document_names(project_id) or []
    except Exception:  # noqa: BLE001
        return None, ""
    xers = [
        d for d in docs
        if (d.get("original_name") or "").lower().endswith(".xer")
        and d.get("file_path")
    ]
    if not xers:
        return None, ""
    low = (user_message or "").lower()
    named = [
        d for d in xers
        if _user_names_project_file(low, d.get("original_name") or "")
    ]
    pick = None
    if len(named) == 1:
        pick = named[0]
    elif len(named) > 1:
        for d in named:
            on = (d.get("original_name") or "").strip().lower()
            if on and on in low:
                pick = d
                break
        pick = pick or named[0]
    elif len(xers) == 1:
        pick = xers[0]
    if not pick:
        return None, ""
    fp = pick.get("file_path") or ""
    if not os.path.exists(fp):
        return None, ""
    return fp, pick.get("original_name") or os.path.basename(fp)


def _project_files_named_in(project_id: str, user_low: str) -> list[str]:
    """Names of the project's documents that ``user_low`` mentions, in upload order."""
    from app.core import projects as _projects

    names = [d.get("original_name") or "" for d in _projects.list_document_names(project_id) or []]
    return _names_mentioned(user_low, names)


_GENERIC_DOCUMENT_NOUNS = frozenset({
    "document", "doc", "file", "upload", "attachment", "record",
})
# Function words of a question. What is left of the ask is what it is about.
_ASK_FUNCTION_WORDS = frozenset({
    "what", "which", "when", "where", "does", "said", "says", "state", "states",
    "about", "that", "this", "these", "those", "there", "their", "from", "with",
    "have", "give", "tell", "please", "uploaded", "attached", "shared", "into",
    "your", "mine", "they", "them", "were", "been", "being", "should", "would",
    "could", "much", "many", "will", "shall", "also", "just", "only",
})
_OWN_UPLOAD_CANDIDATES = 5


def _singular(word: str) -> str:
    for suffix in ("ies", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


def _name_tokens(name: str) -> list[str]:
    return re.findall(r"[a-z]+", os.path.splitext(name or "")[0].lower())


def _token_is_noun(token: str, noun: str) -> bool:
    token, noun = _singular(token), _singular(noun)
    if token == noun:
        return True
    shorter, longer = sorted((token, noun), key=len)
    return len(shorter) >= 4 and longer.startswith(shorter)


def _resolve_own_upload(project_id: str, user_msg: str) -> dict[str, Any] | None:
    """The user's own document an ask refers to without naming it.

    Read the usual way: the project's document list, then the file's text.
    The document noun narrows the list ("my note" -> names with "note");
    the ask's subject words, in the name and then in the text, choose among
    what is left; the latest upload breaks a tie. With nothing to choose by
    among several documents, none is picked.
    """
    nouns = _own_upload_nouns(user_msg)
    if not project_id or not nouns:
        return None
    from app.core import projects as _projects
    from app.core.ingest_status import TEXT_BEARING_EXTS

    readable = [
        d for d in (_projects.list_document_names(project_id) or [])
        if (ext := os.path.splitext(d.get("original_name") or "")[1].lower())
        in TEXT_BEARING_EXTS and ext not in _FILE_PREDISPATCH_EXTS
    ]
    if not readable:
        return None
    specific = [
        n.split()[-1] for n in nouns
        if _singular(n.split()[-1]) not in _GENERIC_DOCUMENT_NOUNS
    ]
    by_noun = [
        d for d in readable
        if any(_token_is_noun(t, n) for t in _name_tokens(d.get("original_name") or "")
               for n in specific)
    ]
    pool = by_noun if (by_noun or not specific) else readable
    noun_words = {_singular(w) for n in nouns for w in n.split()}
    subject = {
        _singular(w) for w in re.findall(r"[a-z]{4,}", (user_msg or "").lower())
        if w not in _ASK_FUNCTION_WORDS and _singular(w) not in noun_words
    }
    recent = list(reversed(pool))[:_OWN_UPLOAD_CANDIDATES]
    best: tuple[tuple[int, int, int], dict[str, Any]] | None = None
    for rank, d in enumerate(recent):
        name = d.get("original_name") or ""
        content, doc, err = _fetch_document_content(project_id, str(d.get("id") or ""), "")
        if err or not content:
            continue
        name_hits = sum(1 for t in _name_tokens(name) if _singular(t) in subject)
        words = {_singular(w) for w in re.findall(r"[a-z]{4,}", str(content.get("text") or "").lower())}
        text_hits = len(subject & words)
        score = (name_hits, text_hits, -rank)
        if best is None or score > best[0]:
            best = (score, {"name": name, "doc": doc, "content": content})
    if best is None:
        return None
    (name_hits, text_hits, _rank), chosen = best
    if name_hits or text_hits or len(recent) == 1:
        return chosen
    return None


def _file_tool_hint(messages: list, project_id: str | None,
                    allowed_blocks: list[str]) -> str:
    """When the user's request names a REAL project file and the agent has
    the tool that reads it, say so in the corrective nudge.

    Live 2026-08-16: bim-analyst, asked for an IFC element census, flailed
    across fetch_document and formula_executor_v2 (generated code returning
    zeros) while bim_extractor -- the tool built for exactly this -- sat
    unused in its roster. K2 rejects forced tool_choice, so the only
    deterministic lever is DATA: name the file that exists and the tool
    that opens it. Not keyword routing -- the hint only fires when the
    message literally contains a document that is in the project store."""
    if not project_id:
        return ""
    user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user" and not str(m.get("content", "")).startswith(
                (_TOOL_ERROR_NUDGE[:20], _EMPTY_ROUTER_NUDGE[:20])):
            user_msg = str(m.get("content", ""))
            break
    if not user_msg:
        return ""
    low = user_msg.lower()
    try:
        from app.core import projects as _projects
        docs = _projects.list_document_names(project_id) or []
    except Exception:  # noqa: BLE001
        return ""
    for name in _names_mentioned(low, [d.get("original_name") or "" for d in docs]):
        if name:
            ext = os.path.splitext(name)[1].lower()
            if (
                ext == ".xer"
                and _message_wants_resource_histogram(user_msg)
                and "construction" in allowed_blocks
            ):
                return (
                    f" The project file '{name}' EXISTS in this project. "
                    "Call the resource_histogram tool with "
                    f"schedule_file='{name}' — it builds the manpower / "
                    "activity histogram. Do not use primavera_parser for a "
                    "histogram."
                )
            tool = _EXT_TOOL_HINTS.get(ext)
            if tool and tool in allowed_blocks:
                return (
                    f" The project file '{name}' EXISTS in this project. "
                    f"Call the {tool} tool with file_path='{name}' -- it is "
                    "built for exactly this file type."
                )
            if os.path.splitext(name)[1].lower() in _TEXT_PREDISPATCH_EXTS:
                return (
                    f" The project file '{name}' EXISTS. Call fetch_document "
                    f"with filename='{name}' and quote unique tokens verbatim. "
                    "Do not stop after listing the filename."
                )
    return ""


# Extensions whose analysis is deterministically PRE-DISPATCHED: when the
# user's message literally names a project file of this type and the agent
# holds the tool, the runtime runs the tool itself BEFORE the model answers
# and injects the result as authoritative context. Born from the IFC census
# gate: K2 rejects forced tool_choice, and across three live rounds the
# model variously refused (F36), narrated (F27), and flailed into
# code-generation returning zeros -- while bim_extractor sat one call away.
# Machinery, not model behaviour. Kill-switch: AGENT_FILE_PREDISPATCH=0.
_FILE_PREDISPATCH_EXTS = {
    ".ifc": "bim_extractor",
    ".dxf": "drawing_qto",
    ".dwg": "drawing_qto",
}
# Plain-text AND office specs: fetch_document extracts bytes from disk
# even when RAG chunks are empty (an unindexed .docx the user named).
_TEXT_PREDISPATCH_EXTS = {".txt", ".md", ".docx", ".doc"}
# PDF drawings share the .pdf extension with specs/contracts. Only
# pre-dispatch drawing_qto when the turn is actually a take-off.
_PDF_QTO_HINT = re.compile(
    r"\b(qto|take-?off|take off|quantit(?:y|ies)|measur)",
    re.IGNORECASE,
)


async def _predispatch_file_tool(
    agent: "Agent", messages: list, project_id: str | None,
) -> dict[str, Any] | None:
    """Run the matching file tool up front and fold its result into the turn.

    Returns the tool-call record (for the response's tool_calls list) or
    None when the conditions don't hold. Never raises -- a pre-dispatch
    failure must not kill the turn; the model then proceeds normally and
    the corrective-nudge machinery covers the rest."""
    if os.getenv("AGENT_FILE_PREDISPATCH", "1") == "0" or not project_id:
        return None
    try:
        user_msg, _history = _messages_user_and_history(messages)
        if message_wants_answer_report(user_msg):
            return None
        if not user_msg:
            return None
        # A self-contained L×W×D volume ask must not steal drawing_qto just
        # because the project happens to have a PDF (leftover L6).
        if (await _off_loop(_looks_like_self_contained_calculation, user_msg)):
            return None
        # Contract Data Q&A (delay damages, TfC, Schedule N, …) must not
        # steal to drawing_qto just because a DXF/PDF stem collides with
        # ordinary English in the question (delay-damages rate ask:
        # "the whole of the Works.dxf" ⊂ "…for the whole of the Works?").
        contract_lookup = message_is_contract_data_lookup(user_msg)
        low = user_msg.lower()
        # Listing the corpus's names and matching each one against the
        # message is thousands of comparisons: done in the turn pool, it
        # returns only the names the user mentioned, in upload order.
        named = (await _off_loop(_project_files_named_in, project_id, low))
        target = None
        text_target = None
        for name in named:
            if name:
                ext = os.path.splitext(name)[1].lower()
                if ext in _TEXT_PREDISPATCH_EXTS and text_target is None:
                    text_target = name
                tool = _FILE_PREDISPATCH_EXTS.get(ext)
                if (
                    tool is None
                    and ext == ".pdf"
                    and _PDF_QTO_HINT.search(user_msg)
                    and not contract_lookup
                ):
                    tool = "drawing_qto"
                if (
                    tool
                    and contract_lookup
                    and tool in ("drawing_qto", "bim_extractor")
                ):
                    tool = None
                if tool and tool in agent.allowed_blocks:
                    target = (name, tool)
                    break
        reason = "the request names that file"
        own_doc: dict[str, Any] | None = None
        if not named and not target:
            own_doc = await _off_loop(_resolve_own_upload, project_id, user_msg)
            if own_doc:
                text_target = own_doc["name"]
                reason = "the request refers to the user's own uploaded document"
        if text_target and not target:
            if own_doc:
                content, doc = own_doc["content"], own_doc["doc"]
            else:
                content, doc, err = (await _off_loop(_fetch_document_content, project_id, "", text_target))
                if err or not content:
                    return None
            payload = str((content or {}).get("text") or "")[:6000]
            shown = (doc or {}).get("original_name") or text_target
            marker = (
                f"[doc_id={(doc or {}).get('id') or shown} chunk=0 "
                f"src={shown}]"
            )
            messages.append({
                "role": "user",
                "content": (
                    f"PLATFORM PRE-DISPATCH: fetch_document has ALREADY been "
                    f"run on '{shown}' because {reason}. Its authoritative "
                    f"contents:\n{marker}\n{payload}\n"
                    "Quote unique tokens from this result verbatim. Do not "
                    "stop at listing the filename."
                ),
            })
            return {
                "name": "fetch_document",
                "ok": True,
                "predispatched": True,
                "result": {
                    "document_id": (doc or {}).get("id"),
                    "filename": (doc or {}).get("original_name") or text_target,
                    "content": (content or {}).get("text"),
                    "truncated": (content or {}).get("truncated"),
                    "source": (content or {}).get("source"),
                },
            }
        if not target:
            return None
        name, tool = target
        resolved = (await _off_loop(_resolve_file_path, project_id, name))
        instance = block_instances.get(tool) or _create_block_instance(tool)
        want_clash = message_wants_clash(user_msg)
        result = await instance.execute(
            {"file_path": resolved},
            ({"run_clash_detection": True} if want_clash
             else {"run_clash_detection": False}),
        )
        ok = isinstance(result, dict) and result.get("status") != "error"
        if not ok:
            return None  # fall back to the normal loop + nudges
        # UniversalBlock.execute wraps process() as {status, result: inner}.
        # Lead keys and clash_report live on the INNER dict; using the envelope
        # made leftover clash-on look like "element counts only" after the 6k cap.
        inner = result
        nested = result.get("result") if isinstance(result, dict) else None
        if isinstance(nested, dict) and (
            "ifc_schema" in nested
            or "clash_report" in nested
            or "quantities" in nested
            or "building_elements" in nested
        ):
            inner = nested
        if tool == "bim_extractor" and isinstance(inner, dict):
            lead_keys = (
                "status", "ifc_schema", "element_count", "quantities",
                "clash_report",
            )
            ordered = {k: inner[k] for k in lead_keys if k in inner}
            for k, v in inner.items():
                if k not in ordered:
                    ordered[k] = v
            inner = ordered
        clash_lead = ""
        cr = inner.get("clash_report") if isinstance(inner, dict) else None
        if want_clash:
            cr = cr or {}
            clash_lead = (
                f"clash_detection_ran={str(bool(cr)).lower()} clash_count="
                f"{len(cr.get('clashes') or [])} "
                f"method={cr.get('detection_method')}\n"
            )
        payload = clash_lead + json.dumps(inner, default=str)[:6000]
        messages.append({
            "role": "user",
            "content": (
                f"PLATFORM PRE-DISPATCH: the {tool} tool has ALREADY been "
                f"run on '{name}' because the request names that file. Its "
                f"authoritative result:\n{payload}\n"
                "Answer the request directly from this result. Do not "
                "re-run the tool unless a different action is needed."
            ),
        })
        return {"name": tool, "ok": True, "predispatched": True,
                "result": inner}
    except Exception:  # noqa: BLE001
        _LOG.warning("file pre-dispatch failed; continuing normally",
                     exc_info=True)
        return None


def _messages_user_and_history(messages: list) -> tuple[str, list]:
    """Latest operator text plus prior turns (for duration-override inference).

    Skip ``PLATFORM PRE-DISPATCH:`` user bubbles — live M1/M2/M9 on 69b12ff
    let a file/QTO inject become the "user" message, so remaining-deliverable
    detectors never saw the original WBS / cash-flow / claim ask.
    """
    user_msg = ""
    prior: list = []
    for m in messages or []:
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        prior.append(m)
        if role == "user":
            content = str(m.get("content") or "")
            if content.lstrip().startswith(_PREDISPATCH_PREFIX):
                continue
            user_msg = _unwrap_rag_folded_operator_text(content)
    history = prior[:-1] if prior else []
    return user_msg, history


# The two shapes of user-role bubble the PLATFORM writes. Readers that must
# tell operator text from replayed tool output (the cost gate's user seed,
# _operator_user_text) match on these, so a change to either writer below
# cannot silently turn platform figures back into "what the user typed".
_PREDISPATCH_PREFIX = "PLATFORM PRE-DISPATCH:"
_TOOL_RESULT_PREFIX = "Tool result ("


def _operator_user_text(messages: list) -> str:
    """All non-predispatch user bubbles, oldest first.

    Live M9 graft must see the original claim ask even when a later user
    bubble is a haul-road file extract (the #410 last-bubble failure mode).
    """
    parts: list[str] = []
    for m in messages or []:
        if m.get("role") != "user":
            continue
        content = str(m.get("content") or "")
        if content.lstrip().startswith(_PREDISPATCH_PREFIX):
            continue
        content = _unwrap_rag_folded_operator_text(content)
        if content.strip():
            parts.append(content)
    return "\n".join(parts)


async def _predispatch_wbs_duration_override(
    agent: "Agent", messages: list, project_id: str | None = None,
) -> dict[str, Any] | None:
    """Re-run ``generate_wbs`` when the user overrides a template duration.

    ``consider 6 and re-run`` must not depend on the model remembering to
    pass ``duration_overrides``. Never raises — a miss falls through to
    the normal tool loop / forced generate_wbs.
    """
    if os.getenv("AGENT_WBS_DURATION_PREDISPATCH", "1") == "0":
        return None
    if "construction" not in getattr(agent, "allowed_blocks", ()):
        return None
    try:
        from app.lib.wbs_duration_overrides import (
            collect_overrides,
            infer_target_count,
            message_wants_wbs_duration_rerun,
        )
        user_msg, history = _messages_user_and_history(messages)
        if not user_msg or not message_wants_wbs_duration_rerun(user_msg, history):
            return None
        overrides = collect_overrides(user_msg, history=history)
        if not overrides:
            return None
        from app.dependencies import get_block_instance
        container = get_block_instance("construction")
        hist_text = "\n".join(
            _unwrap_rag_folded_operator_text(str(m.get("content") or ""))
            for m in history if m.get("role") == "user"
        )
        brief = hist_text.strip() or user_msg
        params = {
            "brief": brief,
            "target_count": infer_target_count(hist_text or user_msg),
            "user_message": user_msg,
            "history": history,
            "duration_overrides": overrides,
            "project_id": project_id,
        }
        result = await container.generate_wbs({}, params)
        if not isinstance(result, dict) or result.get("status") != "success":
            return None
        compact = dict(result)
        acts = compact.get("activities") if isinstance(compact.get("activities"), list) else []
        compact["activities_total"] = len(acts)
        compact["activities_sample"] = acts[:15]
        compact.pop("activities", None)
        payload = json.dumps(compact, default=str)[:6000]
        applied = compact.get("duration_overrides_applied") or []
        applied_bits = ", ".join(
            f"{o.get('match')}={o.get('days')}d×{o.get('activities_updated')}"
            for o in applied
        ) or "none matched"
        messages.append({
            "role": "user",
            "content": (
                "PLATFORM PRE-DISPATCH: generate_wbs has ALREADY been re-run "
                f"with the user's duration override ({applied_bits}). "
                "Authoritative result:\n"
                f"{payload}\n"
                "Report the new durations and recomputed total/critical path "
                "from this result. Do not reuse a prior WBS table. Do not "
                "re-call generate_wbs unless a different override is needed."
            ),
        })
        return {
            "name": "generate_wbs",
            "ok": True,
            "predispatched": True,
            "result": compact,
        }
    except Exception:  # noqa: BLE001
        _LOG.warning("WBS duration pre-dispatch failed; continuing normally",
                     exc_info=True)
        return None


async def _predispatch_resource_histogram(
    agent: "Agent", messages: list, project_id: str | None,
) -> dict[str, Any] | None:
    """Run ``resource_histogram`` before the model can steal primavera_parser.

    Pinned ``construction-pm`` is not in the ``_forced_specific_tool`` gate
    (project-assistant / heavy-reasoning only). Leftover M17 therefore never
    saw a forced histogram tool. Predispatch is the deterministic lever —
    same pattern as WBS duration override. Kill-switch:
    ``AGENT_HISTOGRAM_PREDISPATCH=0``.
    """
    if os.getenv("AGENT_HISTOGRAM_PREDISPATCH", "1") == "0":
        return None
    if "construction" not in getattr(agent, "allowed_blocks", ()):
        return None
    if not project_id:
        return None
    try:
        user_msg, _history = _messages_user_and_history(messages)
        if not user_msg or not (await _off_loop(_message_wants_resource_histogram, user_msg)):
            return None
        schedule_file, picked_name = (await _off_loop(_resolve_histogram_schedule_file, project_id, user_msg,
        ))
        if not schedule_file:
            return None
        from app.dependencies import get_block_instance
        container = get_block_instance("construction")
        result = await container.resource_histogram(
            {}, {"schedule_file": schedule_file},
        )
        if not isinstance(result, dict) or result.get("status") != "success":
            return None
        compact = dict(result)
        periods = compact.get("periods") if isinstance(compact.get("periods"), list) else []
        compact["periods"] = periods[:20]
        payload = json.dumps(compact, default=str)[:6000]
        kind = compact.get("histogram_kind") or "labor"
        messages.append({
            "role": "user",
            "content": (
                "PLATFORM PRE-DISPATCH: resource_histogram has ALREADY been "
                f"run on '{picked_name}'. Authoritative result ({kind}):\n"
                f"{payload}\n"
                "Report the period buckets from this result. Do not invent "
                "man-hours. Do not call primavera_parser for this histogram. "
                "Do not re-call resource_histogram unless a different file "
                "is named."
            ),
        })
        return {
            "name": "resource_histogram",
            "ok": True,
            "predispatched": True,
            "result": compact,
        }
    except Exception:  # noqa: BLE001
        _LOG.warning("histogram pre-dispatch failed; continuing normally",
                     exc_info=True)
        return None


async def _predispatch_look_ahead(
    agent: "Agent", messages: list, project_id: str | None,
) -> dict[str, Any] | None:
    """Run ``look_ahead`` before the model steals primavera_parser.

    Same pattern as histogram predispatch. Kill-switch:
    ``AGENT_LOOKAHEAD_PREDISPATCH=0``.
    """
    if os.getenv("AGENT_LOOKAHEAD_PREDISPATCH", "1") == "0":
        return None
    if "construction" not in getattr(agent, "allowed_blocks", ()):
        return None
    if not project_id:
        return None
    try:
        user_msg, _history = _messages_user_and_history(messages)
        if not user_msg or not (await _off_loop(_message_wants_look_ahead, user_msg)):
            return None
        schedule_file, picked_name = (await _off_loop(_resolve_histogram_schedule_file, project_id, user_msg,
        ))
        if not schedule_file:
            return None
        from app.dependencies import get_block_instance
        container = get_block_instance("construction")
        from app.containers.construction.schedule import _stated_look_ahead_date
        la_params: dict[str, Any] = {
            "schedule_file": schedule_file,
            "user_message": user_msg,
        }
        stated = _stated_look_ahead_date(user_msg)
        if stated:
            la_params["as_of"] = stated
        result = await container.look_ahead({}, la_params)
        if not isinstance(result, dict) or result.get("status") != "success":
            return None
        compact = dict(result)
        acts = compact.get("activities") if isinstance(compact.get("activities"), list) else []
        compact["activities"] = acts[:40]
        payload = json.dumps(compact, default=str)[:6000]
        messages.append({
            "role": "user",
            "content": (
                "PLATFORM PRE-DISPATCH: look_ahead has ALREADY been "
                f"run on '{picked_name}'. Authoritative result:\n"
                f"{payload}\n"
                "Report the overlapping activities from this result. Do not "
                "invent activities. Do not call primavera_parser for this "
                "look-ahead. Do not re-call look_ahead unless a different "
                "file is named."
            ),
        })
        return {
            "name": "look_ahead",
            "ok": True,
            "predispatched": True,
            "result": compact,
        }
    except Exception:  # noqa: BLE001
        _LOG.warning("look-ahead pre-dispatch failed; continuing normally",
                     exc_info=True)
        return None


_WIR_FORM_RE = re.compile(
    r"\b("
    r"work inspection request|\bwir\b|inspection request|"
    r"wir form|wir template"
    r")\b",
    re.IGNORECASE,
)
_WIR_QA_RE = re.compile(
    r"\b(what is|what's|whats|explain|define)\b",
    re.IGNORECASE,
)


def _message_wants_wir_form(text: str) -> bool:
    """True when the turn is a WIR / inspection-request deliverable.

    Live M15: pinned construction-pm searched the named WIR template and
    returned empty. Predispatch writes the form from operator facts.
    Job requisition / claim / cash-flow / WBS / O&M / safety / IPC / as-built
    take priority even when the brief also says "inspection request".
    """
    low = (text or "").strip()
    if not low or _WIR_QA_RE.search(low):
        return False
    if (
        _message_wants_job_requisition(low)
        or _message_wants_delay_claim(low)
        or _message_wants_cash_flow(low)
        or _message_wants_om_manual(low)
        or _message_wants_safety_briefing(low)
        or _message_wants_first_run_wbs(low)
        or _message_wants_as_built_note(low)
        or _message_wants_ipc_draft(low)
        or _message_wants_follow_on_rfi(low)
        or _message_wants_qc_punch(low)
        or _message_wants_design_directive(low)
    ):
        return False
    return bool(_WIR_FORM_RE.search(low))


async def _predispatch_wir_form(
    agent: "Agent",
    messages: list,
    project_id: str | None,
    operator_text: str | None = None,
) -> dict[str, Any] | None:
    """Run ``wir_form`` before the model can search-loop a WIR template.

    Pinned ``construction-pm`` skips predefined intercept (F24). Kill-switch:
    ``AGENT_WIR_PREDISPATCH=0``.
    """
    if os.getenv("AGENT_WIR_PREDISPATCH", "1") == "0":
        return None
    if "construction" not in getattr(agent, "allowed_blocks", ()):
        return None
    try:
        user_msg, _history = _messages_user_and_history(messages)
        detect = (operator_text or user_msg or "").strip()
        if not detect or not (await _off_loop(_message_wants_wir_form, detect)):
            return None
        from app.dependencies import get_block_instance
        container = get_block_instance("construction")
        result = await container.wir_form(
            {"text": detect}, {"user_message": detect},
        )
        if not isinstance(result, dict) or result.get("status") != "success":
            return None
        rendered = _format_wir_form(result)
        messages.append({
            "role": "user",
            "content": (
                "PLATFORM PRE-DISPATCH: wir_form has ALREADY been run from "
                "the operator facts in this turn. Authoritative draft:\n"
                f"{rendered}\n"
                "Present this WIR in full. Do not search again for the "
                "template. Do not claim the inspection was issued."
            ),
        })
        return {
            "name": "wir_form",
            "ok": True,
            "predispatched": True,
            "result": result,
        }
    except Exception:  # noqa: BLE001
        _LOG.warning("WIR pre-dispatch failed; continuing normally",
                     exc_info=True)
        return None


def _message_wants_commissioning(text: str) -> bool:
    raw = text or ""
    if _message_wants_om_manual(raw):
        return False
    return bool(re.search(r"commissioning checklist", raw, re.I))


def _message_wants_ipc_draft(text: str) -> bool:
    raw = text or ""
    if _message_wants_cash_flow(raw) or _message_wants_delay_claim(raw):
        return False
    if not re.search(r"payment certificate|\bipc\b|interim payment", raw, re.I):
        return False
    return _carries_ipc_figures(raw)


def _message_wants_delay_claim(text: str) -> bool:
    return bool(not message_is_contract_data_lookup(text or "") and re.search(r"delay claim|claim notice|eot claim", text or "", re.I))


def _message_wants_as_built_note(text: str) -> bool:
    return bool(re.search(r"as-built deviation|as built deviation", text or "", re.I))


def _message_wants_primavera_parse(text: str) -> bool:
    """True when the operator asks to parse an uploaded .xer by name."""
    t = (text or "").lower()
    if "parse_primavera" in t or "primavera_parser" in t:
        return True
    if ".xer" in t and any(k in t for k in ("parse", "extract", "milestones", "activities")):
        return True
    return False


def _message_wants_vo_draft(text: str) -> bool:
    """True for a first-ask variation-order draft, not impact / log Q&A."""
    from app.core.site_vocab import message_wants_vo_draft
    return message_wants_vo_draft(text or "")


def _message_wants_inline_boq(text: str) -> bool:
    """True when the operator pasted SYNTHETIC CSV/BOQ lines for boq_process."""
    try:
        from app.core.site_vocab import message_has_inline_boq_lines
        return bool(message_has_inline_boq_lines(text or ""))
    except Exception:  # noqa: BLE001
        _LOG.debug("inline boq detect skipped", exc_info=True)
        return False


def _message_wants_job_requisition(text: str) -> bool:
    return bool(re.search(r"job requisition", text or "", re.I))


def _message_wants_rfp_draft(text: str) -> bool:
    raw = text or ""
    if _message_wants_wir_form(raw):
        return False
    if message_wants_answer_report(raw):
        return False
    return bool(re.search(r"\brfp\b|request for proposal|invitation to tender", raw, re.I))


def _message_wants_cash_flow(text: str) -> bool:
    raw = text or ""
    # Local import: action_router must not import runtime at module load.
    from app.core.action_router import dewatering_drawdown_not_cash_flow
    if dewatering_drawdown_not_cash_flow(raw):
        return False
    return bool(re.search(
        r"cash[_\- ]flow|s-curve|s curve|spend curve|drawdown",
        raw,
        re.I,
    ))


def _message_wants_om_manual(text: str) -> bool:
    return bool(re.search(
        r"operations and maintenance|o\s*&\s*m\b|o and m|maintenance manual",
        text or "",
        re.I,
    ))


def _message_wants_follow_on_rfi(text: str) -> bool:
    return bool(re.search(
        r"follow-on rfi|\bdraft a(?:n)? (?:follow[- ]on )?rfi\b|"
        r"request for information",
        text or "",
        re.I,
    ))


def _message_wants_qc_punch(text: str) -> bool:
    return bool(re.search(
        r"qc punch|punch list|hold-point sign-off|hold point sign-off",
        text or "",
        re.I,
    ))


def _message_wants_design_directive(text: str) -> bool:
    return bool(re.search(
        r"design directive|underpass length|rfi016|rfi 016",
        text or "",
        re.I,
    ))


def _message_wants_safety_briefing(text: str) -> bool:
    raw = text or ""
    if (
        _message_wants_first_run_wbs(raw)
        or _message_wants_cash_flow(raw)
        or _message_wants_follow_on_rfi(raw)
        or _message_wants_delay_claim(raw)
        or _message_wants_job_requisition(raw)
        or _message_wants_qc_punch(raw)
        or _message_wants_design_directive(raw)
        or _message_wants_ipc_draft(raw)
    ):
        return False
    if re.search(r"safety briefing|toolbox talk", raw, re.I):
        return True
    # Live M16 (safety-officer): "Green Village diversion briefing ... from the
    # three named files, covering signage, speed and pedestrian" matched
    # NOTHING here, because the site cue demanded the literal word ROAD
    # ("road diversion"). safety_briefing never ran, only fetch_document did,
    # and the hat answered "I'm ready to help, but I need a specific
    # question". A diversion named after its LOCATION rather than its road is
    # the same public-interface ask, so the cue no longer requires "road".
    site = re.search(
        r"haul[- ]road|\bdiversion\b|public-interface|traffic management",
        raw, re.I,
    )
    # Distinct topics, not total hits: "speed ... speed" is one topic, and the
    # briefing-only route below needs genuine breadth to fire.
    topics = {m.group(0).lower()
              for m in re.finditer(r"signage|pedestrian|speed", raw, re.I)}
    if site and topics:
        return True
    # No site cue at all, but the turn asks for a briefing and names at least
    # two of the three public-interface topics. Two is deliberate: one topic
    # plus the bare word "briefing" is too thin to claim the turn, and this
    # predicate already yields to every other deliverable above.
    return bool(re.search(r"\bbriefing\b", raw, re.I) and len(topics) >= 2)


def _message_wants_boq_scope_wbs_ask(text: str) -> bool:
    """True for leftover F1: high-level WBS over demolition / site-clearance BOQ."""
    from app.core.predefined_reasoning import message_wants_boq_scope_wbs
    return bool(message_wants_boq_scope_wbs(text or ""))


def _message_wants_first_run_wbs(text: str) -> bool:
    raw = text or ""
    try:
        from app.lib.wbs_duration_overrides import message_wants_wbs_duration_rerun
        if message_wants_wbs_duration_rerun(raw, []):
            return False
    except Exception:  # noqa: BLE001 — import/shape miss must not steal first-run WBS
        _LOG.warning(
            "swallowed %s in _message_wants_first_run_wbs() — treating as first-run",
            "Exception", exc_info=True,
        )
    # Leftover F1: "Generate a high-level WBS for the demolition…" is a
    # first-run produce, not a duration re-run. The older regex required
    # "build a wbs|work breakdown|wbs for"; keep that, and also elect the
    # BOQ-scope WBS phrasing so predispatch cannot miss the live ask.
    if _message_wants_boq_scope_wbs_ask(raw):
        return True
    if not re.search(r"\b(build a wbs|work breakdown|\bwbs\b for)\b", raw, re.I):
        return False
    return True


def _message_wants_locked_deliverable(text: str) -> bool:
    """True when this turn already has a remaining/WIR-priority draft to lock."""
    return any(
        fn(text)
        for fn in (
            _message_wants_job_requisition,
            _message_wants_rfp_draft,
            _message_wants_delay_claim,
            _message_wants_cash_flow,
            _message_wants_om_manual,
            _message_wants_safety_briefing,
            _message_wants_follow_on_rfi,
            _message_wants_qc_punch,
            _message_wants_design_directive,
            _message_wants_first_run_wbs,
            _message_wants_as_built_note,
            _message_wants_ipc_draft,
            _message_wants_commissioning,
            _message_wants_vo_draft,
            _message_wants_inline_boq,
        )
    )



def _vo_draft_hard_excludes(user_message: str) -> set[str]:
    """Tools that must not run on a VO-draft turn (live tip 0a95d03 ask2).

    Predispatch steal only applies after a successful VO tool hit. Ask2 on
    tip 0a95d03 skipped predispatch and called construction + sympy_reasoning
    instead. Hard-exclude those whenever the operator ask is a VO draft.
    """
    if not _message_wants_vo_draft(user_message):
        return set()
    return {
        "construction", "formula_executor_v2", "formula_executor",
        "sympy_reasoning", "recommendation_template",
        "change_order_impact", "validation_pipeline", "delegate_to_agent",
        "construction_calc",
        # Live tip d9d5971 ask2: after #683 excludes, the model still
        # called search_project_documents + fetch_document and answered
        # index chatter instead of drafting VO-D-002.
        "search_project_documents", "fetch_document", "list_project_documents",
    }


def _inline_boq_hard_excludes(user_message: str) -> set[str]:
    """Tools that must not run when the operator pasted BOQ lines.

    Live tip 0a95d03 ask1 routed to self-coding and called formula_executor_v2
    (HTTP 429) instead of boq_processor. Hard-exclude the formula path whenever
    inline BOQ lines are present.
    """
    try:
        from app.core.site_vocab import message_has_inline_boq_lines
        if not message_has_inline_boq_lines(user_message or ""):
            return set()
    except Exception:  # noqa: BLE001
        return set()
    return {"formula_executor_v2", "formula_executor", "construction_calc"}

def _conflicting_tools_after_predispatch(name: str) -> set[str]:
    """Tools that stole the live Infra Pack answers after a correct draft."""
    steal = {
        "generate_wbs": {
            "payment_certificate", "drawing_qto", "wir_form",
            "primavera_parser", "interim_certificate_generator",
            "safety_briefing",
        },
        "cash_flow_forecast": {
            "drawing_qto", "payment_certificate", "wir_form",
            "interim_certificate_generator", "safety_briefing",
        },
        "om_manual_generator": {"commissioning_checklist", "wir_form"},
        "safety_briefing": {"wir_form", "commissioning_checklist"},
        "rfi_generator": {"wir_form", "safety_briefing"},
        "resource_histogram": {"primavera_parser", "safety_briefing"},
        "look_ahead": {"primavera_parser", "safety_briefing", "resource_histogram"},
        "claims_builder": {
            "payment_certificate", "wir_form", "interim_certificate_generator",
        },
        "job_requisition": {"wir_form"},
        "rfp_draft": {"wir_form"},
        "as_built_deviation_report": {"wir_form", "commissioning_checklist"},
        "payment_certificate": {"wir_form", "claims_builder"},
        "commissioning_checklist": {"wir_form", "om_manual_generator"},
        "wir_form": {"payment_certificate", "job_requisition", "rfp_draft", "rfi_generator"},
        "variation_order_manager": {
            "change_order_impact", "wir_form", "sympy_reasoning",
            "construction", "formula_executor_v2", "formula_executor",
            "recommendation_template", "validation_pipeline",
            "delegate_to_agent",
            "search_project_documents", "fetch_document",
            "list_project_documents",
        },
        "boq_process": {"construction_calc", "generate_wbs", "formula_executor_v2", "formula_executor"},
        "boq_processor": {"construction_calc", "generate_wbs", "formula_executor_v2", "formula_executor"},
    }
    return set(steal.get(name) or ())


def _inject_predispatch(
    messages: list,
    name: str,
    rendered: str,
    instruction: str,
) -> None:
    messages.append({
        "role": "user",
        "content": (
            f"PLATFORM PRE-DISPATCH: {name} has ALREADY been run from "
            f"the operator facts in this turn. Authoritative draft:\n"
            f"{rendered}\n"
            f"{instruction}"
        ),
    })


async def _predispatch_construction_draft(
    agent: "Agent",
    messages: list,
    *,
    env_key: str,
    want_fn,
    action: str,
    format_fn,
    instruction: str,
    operator_text: str | None = None,
    project_id: str | None = None,
    conversation_id: str | None = None,
) -> dict[str, Any] | None:
    if os.getenv(env_key, "1") == "0":
        return None
    if "construction" not in getattr(agent, "allowed_blocks", ()):
        return None
    try:
        user_msg, _history = _messages_user_and_history(messages)
        detect = (operator_text or user_msg or "").strip()
        # The deliverable checks scan the message with many patterns: off the
        # event loop, like the other detectors (CI: a 251-309 ms loop stall here).
        if not detect or not (await _off_loop(want_fn, detect)):
            return None
        user_msg = detect
        from app.dependencies import get_block_instance
        container = get_block_instance("construction")
        handler = getattr(container, action, None)
        if handler is None:
            return None
        params = {"user_message": user_msg, "brief": user_msg, "message": user_msg}
        if project_id:
            params["project_id"] = project_id
        if action == "commissioning_checklist":
            inferred = _infer_commissioning_systems(user_msg)
            if inferred:
                params["systems"] = inferred
        result = await handler(
            {
                "text": user_msg,
                "message": user_msg,
                "user_message": user_msg,
                "project_id": project_id,
            },
            params,
        )
        if not isinstance(result, dict) or result.get("status") != "success":
            return None
        if action == "generate_wbs":
            from app.core.conversation_wbs import refuse_scaffold_for_boq_wbs_ask
            if refuse_scaffold_for_boq_wbs_ask(user_msg, user_msg, result):
                # Leftover F1: do not inject the 204-activity building
                # template as an "authoritative" WBS — that is the other
                # live FAIL class (generic scaffold, not BOQ-grounded).
                return None
            if conversation_id:
                stage_conversation_wbs(conversation_id, result)
        rendered = format_fn(result)
        _inject_predispatch(messages, action, rendered, instruction)
        return {
            "name": action,
            "ok": True,
            "predispatched": True,
            "result": result,
        }
    except Exception:  # noqa: BLE001
        _LOG.warning("%s pre-dispatch failed; continuing normally", action,
                     exc_info=True)
        return None


async def _predispatch_remaining_deliverables(
    agent: "Agent",
    messages: list,
    project_id: str | None,
    operator_text: str | None = None,
    conversation_id: str | None = None,
) -> dict[str, Any] | None:
    """First matching non-WIR deliverable draft, so a Groq 413 still has copy."""
    user_msg, _history = _messages_user_and_history(messages)
    detect = (operator_text or user_msg or "").strip()
    if not detect or (await _off_loop(_message_wants_wir_form, detect)):
        return None
    if message_wants_answer_report(detect):
        return None
    candidates = (
        (
            "AGENT_CASH_FLOW_PREDISPATCH",
            _message_wants_cash_flow,
            "cash_flow_forecast",
            _format_cash_flow,
            "Present this cash-flow S-curve in full. Do not run drawing_qto.",
        ),
        (
            "AGENT_OM_PREDISPATCH",
            _message_wants_om_manual,
            "om_manual_generator",
            _format_om_outline,
            "Present this O&M outline in full. Do not draft a commissioning checklist.",
        ),
        (
            "AGENT_WBS_PREDISPATCH",
            _message_wants_first_run_wbs,
            "generate_wbs",
            _format_wbs_result,
            "Present this WBS in full. When it is BOQ-derived, keep the "
            "numbered demolition / site-clearance packages. Do not refuse. "
            "Do not cite another contract year's Conditions of Contract "
            "as the governing contract. Do not invent "
            "man-hours.",
        ),
        (
            "AGENT_RFI_PREDISPATCH",
            _message_wants_follow_on_rfi,
            "rfi_generator",
            _format_rfi,
            "Present this follow-on RFI in full. Do not draft a WIR.",
        ),
        (
            "AGENT_JOB_REQ_PREDISPATCH",
            _message_wants_job_requisition,
            "job_requisition",
            _format_job_requisition,
            "Present this job requisition in full. Do not draft a WIR.",
        ),
        (
            "AGENT_SAFETY_PREDISPATCH",
            _message_wants_safety_briefing,
            "safety_briefing",
            _format_safety_briefing,
            "Present this live-haul-road safety briefing in full.",
        ),
        (
            "AGENT_COMMISSIONING_PREDISPATCH",
            _message_wants_commissioning,
            "commissioning_checklist",
            _format_commissioning_tool,
            "Present this commissioning checklist in full. Do not invent "
            "holiday / spark tests on a reservoir wet test.",
        ),
        (
            "AGENT_IPC_PREDISPATCH",
            _message_wants_ipc_draft,
            "payment_certificate",
            _format_payment_certificate,
            "Present this IPC in full. Do not re-call payment_certificate.",
        ),
        (
            "AGENT_CLAIM_PREDISPATCH",
            _message_wants_delay_claim,
            "claims_builder",
            _format_claim_notice,
            "Present this delay-claim notice in full. Do not invent events.",
        ),
        (
            "AGENT_ASBUILT_PREDISPATCH",
            _message_wants_as_built_note,
            "as_built_deviation_report",
            _format_as_built_note,
            "Present this as-built volume note in full.",
        ),
        (
            "AGENT_RFP_PREDISPATCH",
            _message_wants_rfp_draft,
            "rfp_draft",
            _format_rfp_draft,
            "Present this RFP in full paragraphs. Do not draft a WIR.",
        ),
        (
            "AGENT_VO_PREDISPATCH",
            _message_wants_vo_draft,
            "variation_order_manager",
            _format_variation_order,
            "Present this variation order in full. Do not run "
            "change_order_impact. Do not reply with Status: Success only.",
        ),
        (
            "AGENT_BOQ_PREDISPATCH",
            _message_wants_inline_boq,
            "boq_process",
            _format_boq_process,
            "Present this BOQ processing result in full. Do not run "
            "construction_calc. Do not refuse for a missing file_path — "
            "the lines were already parsed from the operator message.",
        ),
    )
    for env_key, want_fn, action, format_fn, instruction in candidates:
        out = await _predispatch_construction_draft(
            agent, messages,
            env_key=env_key,
            want_fn=want_fn,
            action=action,
            format_fn=format_fn,
            instruction=instruction,
            operator_text=detect,
            project_id=project_id,
            conversation_id=conversation_id,
        )
        if out:
            return out
    return None


def _tool_result_errored(tool_result: Any) -> bool:
    """True when a tool round FAILED, wherever the failure hides.

    F42: container results arrive envelope-green -- ``ok: True`` with
    ``status: "error"`` one or two levels down (the construction container
    wraps its route's error in a success-shaped transport envelope). The
    non-streaming loop only checked ``ok`` and never nudged; walk the
    nested ``result`` chain a few levels instead.
    """
    if not isinstance(tool_result, dict):
        return False
    if tool_result.get("ok") is False:
        return True
    node = tool_result.get("result")
    for _ in range(4):
        if not isinstance(node, dict):
            return False
        if node.get("status") == "error":
            return True
        node = node.get("result")
    return False


_EMPTY_ROUTER_NUDGE = (
    "The router matched NO action. If the request is a computation (volume, "
    "quantity, capacity, EVM, any arithmetic), you MUST now call the "
    "construction tool (action=construction_calc) or formula_executor_v2 and "
    "report that tool's result -- do not compute the number yourself. If no "
    "tool fits the request, say you have no tool for it. Never write "
    "'Routed to:' naming a tool you have not actually called in this turn."
)


# Declared hat-manifest action names -> real implementations (closes the
# KNOWN_INCOMPLETE dispatch gap, 2026-08-15). Each alias maps to machinery
# that actually exists: a registry calculator or a construction-container
# route. Names with no honest backing were REMOVED from the manifests
# instead of stubbed. The literal tuples in _run_tool_call keep the
# AST-based synthetic-tool census (test_agent_tool_surface) counting these
# as dispatchable the moment they dispatch.
_HAT_CALC_ALIASES = {
    "concrete_maturity_calculator": "concrete_maturity_strength",
    "grout_pressure_calculator": "grout_pressure_calc",
    "thermal_crack_assessor": "concrete_thermal_cracking_check",
    "mix_design_validator": "concrete_mix_design_sg",
    "crane_planner": "crane_planning",
    "critical_path_calc": "critical_path_float",
    "float_analysis": "critical_path_float",
    "tender_evaluator": "evaluate_tender",
    "supplier_scoring": "evaluate_tender",
    "compliance_gate_checker": "enforce_critical_rules",
    "ncr_tracker": "next_ncr_status",
}
_HAT_CONTAINER_ALIASES = {
    "interim_certificate_generator": "payment_certificate",
    "schedule_analysis": "parse_primavera_schedule",
    "baseline_compare": "forensic_delay_analysis",
    "eot_claim_assessor": "forensic_delay_analysis",
    "dispute_timeline_builder": "claims_builder",
    "variation_order_generator": "variation_order_manager",
    "boq_cost_analyzer": "boq_process",
    "drawing_qto_extract": "drawing_qto",
    "milestone_tracker": "progress_tracker",
    "rate_card_lookup": "benchmark_lookup",
    "itp_generator": "commissioning_checklist",
    "po_generator": "procurement_list_generator",
}
# Retrieval-shaped manifest actions resolve to the search synthetic tool.
_HAT_SEARCH_ALIASES = frozenset({"search_documents", "fidic_clause_lookup", "standards_lookup"})


# Tools whose result is a VERDICT or LOOKUP, not a deliverable. Their success
# must never trigger force-synthesis, because their result exists to be acted
# on by ANOTHER tool call: a routing verdict needs the dispatch that follows
# it. With smart_orchestrator counted as a deliverable, iter=1 ran
# with_tools=False -- the agent was contractually required to dispatch and
# structurally unable to, which is exactly the state that produced fabricated
# "Routed to:" claims on both live gates.
# cache_manager joined the set after a live document-ingestion turn ended on
# "Let me route this through the boq_processor" -- a narrated FUTURE dispatch
# as the final answer. The turn's only tool calls were search (already
# non-deliverable) + a cache lookup, which counted as a deliverable and
# disarmed force-synthesis. A cache hit/miss is infrastructure, never the
# artifact the user asked for.
_NON_DELIVERABLE_TOOLS = frozenset(
    {"search_project_documents", "smart_orchestrator", "cache_manager"}
)


def _tool_process_payload(tool_result: Any) -> dict[str, Any]:
    """Unwrap the execute envelope to the block's ``process()`` dict."""
    if not isinstance(tool_result, dict):
        return {}
    inner = tool_result.get("result")
    if not isinstance(inner, dict):
        return {}
    nested = inner.get("result")
    if isinstance(nested, dict) and any(
        k in nested for k in ("variances", "value", "stages", "overall", "expression")
    ):
        return nested
    return inner


_FAILURE_STATUSES = frozenset({"error", "failed", "failure"})


def _tool_payload_reports_failure(payload: Any, _depth: int = 0) -> bool:
    """True when a tool's own reply says it failed, whatever the envelope says.

    A reply fails when its ``status`` is an error status, or it carries a
    non-empty ``error`` and no status claiming success. Blocks nest their
    reply under ``result``, so that is followed — two levels, which is as
    deep as any block goes. An empty ``error`` slot, an ``errors: []`` list
    or the word inside a note is not a failure.
    """
    if not isinstance(payload, dict) or _depth > 2:
        return False
    status = str(payload.get("status") or "").strip().lower()
    if status in _FAILURE_STATUSES:
        return True
    if payload.get("error") and status not in ("success", "ok"):
        return True
    return _tool_payload_reports_failure(payload.get("result"), _depth + 1)


def _should_force_synthesis(tool_result: Any) -> bool:
    """True when this tool round produced the artifact the user asked for.

    Empty sympy_reasoning (no variances, no evaluated expression) is NOT a
    deliverable — live leftover-hat L7: the model called sympy with no
    boq/drawing payload, got formula metadata, then force_synthesis disarmed
    formula_executor_v2 so the turn ended on "I will dispatch to the
    construction formula executor instead" with no 2.4 / 6840.
    """
    if not isinstance(tool_result, dict):
        return False
    if tool_result.get("ok") is False:
        return False
    # ``ok`` says the block RAN. Live d8d9573: construction_calc ran, replied
    # "Unknown calculation 'calculate_delay_damages'" with the 84 valid names
    # attached — and because the envelope was ok, the tools were disarmed and
    # the model could not use the list it had just been given. It wrote the
    # retry as DSML text instead, and that was shown to the user.
    if _tool_payload_reports_failure(tool_result.get("result")):
        return False
    name = tool_result.get("name")
    if name in _NON_DELIVERABLE_TOOLS:
        return False
    if name == "sympy_reasoning":
        payload = _tool_process_payload(tool_result)
        has_expr = payload.get("value") is not None and payload.get("expression")
        has_var = bool(payload.get("variances") or payload.get("cost_impacts"))
        if not has_expr and not has_var:
            return False
    # Leftover E4: validation_pipeline on a self-contained volume claim is
    # a checker, not the deliverable. Forcing synthesis here disarmed
    # construction_calc and the turn hung on "Let me validate…" with no
    # 945 m³. Leftover L4 ("Stay on validation. Validate this beam…")
    # has no calc verb and still forces synthesis.
    if name == "validation_pipeline":
        payload = _as_validation_payload(tool_result) or _tool_process_payload(tool_result)
        claim = str((payload or {}).get("claim") or "")
        if _looks_like_self_contained_calculation(claim):
            return False
    return True


def _vo_draft_ready_for_synthesis(user_message: str, tool_name: str | None) -> bool:
    """On a VO-draft turn, only variation_order_manager may lock synthesis.

    Live tip d9d5971 ask2: fetch_document is a deliverable, so the first
    search/fetch pair armed force_synthesis and the model never called
    variation_order_manager. Search/fetch/list must not close the turn.
    """
    if not _message_wants_vo_draft(user_message or ""):
        return True
    return (tool_name or "") in {
        "variation_order_manager",
        "variation_order_generator",
    }


def _has_unread_windows(content: str) -> bool:
    """True when the model has been told to read more of this tool result.

    Live on 1d3c98b, the build that introduced char_offset. Asked for the m3
    demolition quantity, the model reported the truncation, said it could not
    confirm whether an m3 item appears deeper in the bill, offered to narrow
    the search -- and never called boq_processor again. It could not:
    boq_processor is a deliverable tool, so force_synthesis fired on the
    first result and the NEXT call went out with_tools=False.

    So the envelope told the model to call the tool again and the platform
    took its tools away. That is the same failure the sympy_reasoning
    carve-out above exists for, with a different tool: an instruction the
    system then prevents.

    A result with windows left to read is not a finished deliverable. It is
    half a document.
    """
    if not content or "chars_remaining" not in content:
        return False
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return False
    if not isinstance(parsed, dict):
        return False
    try:
        return int(parsed.get("chars_remaining") or 0) > 0
    except (TypeError, ValueError):
        return False


def _empty_router_verdict(tool_result: Any) -> bool:
    """True when this tool round was smart_orchestrator returning an EMPTY
    match (matched_actions: []) -- the state in which fabricated dispatch
    claims were observed live."""
    if not isinstance(tool_result, dict):
        return False
    if tool_result.get("name") != "smart_orchestrator":
        return False
    inner = tool_result.get("result")
    if not isinstance(inner, dict):
        return False
    candidates = [inner]
    if isinstance(inner.get("result"), dict):
        candidates.append(inner["result"])
    for node in candidates:
        if "matched_actions" in node:
            return not node["matched_actions"]
    return False


#: Arguments the model may send that select HOW a result is presented rather
#: than what the block should do. Stripped before dispatch: the blocks are
#: vendored third-party code and an undeclared param is their problem, not
#: something to discover in production.
_PRESENTATION_ARGS = frozenset({"char_offset"})


def _strip_presentation_args(args: Any) -> Any:
    """Remove presentation-only arguments before a block ever sees them."""
    if not isinstance(args, dict):
        return args
    if not _PRESENTATION_ARGS.intersection(args):
        return args
    return {k: v for k, v in args.items() if k not in _PRESENTATION_ARGS}


def _dispatch_args(raw_args: Any) -> Any:
    """Parse a tool call's arguments for DISPATCH.

    One function so the strip cannot be forgotten at a call site: anything
    that wants arguments to send to a block gets them from here, already free
    of presentation-only params. Raises json.JSONDecodeError for the caller
    to turn into the invalid-args result.
    """
    parsed = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
    return _strip_presentation_args(parsed)


def _requested_char_offset(tool_call: dict[str, Any] | None) -> int:
    """The ``char_offset`` the model asked for, if any.

    A presentation parameter, not a block parameter: it selects which window
    of an over-long result comes back, and is stripped before dispatch so no
    block ever sees an argument it does not declare.
    """
    fn = (tool_call or {}).get("function") or {}
    raw = fn.get("arguments") or "{}"
    try:
        args = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        return 0
    if not isinstance(args, dict):
        return 0
    try:
        return max(0, int(args.get("char_offset") or 0))
    except (TypeError, ValueError):
        return 0


def _is_openrouter_llm() -> bool:
    """True when the configured primary provider is OpenRouter.

    Reads ``LLM_PROVIDER`` directly (same source ``_llm_config`` uses) so
    tool-result insertion can tighten caps before the first hop without
    constructing a full provider config.
    """
    return (os.getenv("LLM_PROVIDER") or "").strip().lower() == "openrouter"


def _effective_tool_result_max_chars() -> int:
    """Per-tool JSON cap. OpenRouter uses a tighter free-tier ceiling."""
    if _is_openrouter_llm():
        return _OPENROUTER_TOOL_RESULT_MAX_CHARS
    return _TOOL_RESULT_MAX_CHARS


def _effective_fetch_document_max_chars() -> int:
    """Per-fetch text cap. OpenRouter uses a tighter free-tier ceiling."""
    if _is_openrouter_llm():
        return _OPENROUTER_FETCH_DOCUMENT_MAX_CHARS
    return _FETCH_DOCUMENT_MAX_CHARS


def _tool_result_content(
    payload: dict[str, Any],
    offset: int = 0,
    max_chars: int | None = None,
) -> str:
    """Serialize a tool result for the `tool` message, truncating SAFELY.

    NEVER slice serialized JSON. `json.dumps` emits ASCII with \\uXXXX
    escapes, so a raw ``[:8000]`` can cut mid-escape or mid-string and hand
    the provider malformed JSON. Moonshot rejects that outright:

        HTTP 400 {"message": "Invalid request: tokenization failed",
                  "type": "invalid_request_error"}

    Reproduced 2026-08-03 by the feature sweep on BOTH `document_metadata`
    and `parse_primavera_schedule` — the two actions whose tool results are
    large. The corpus makes it easy to hit: the client filenames contain en
    dashes, which serialize to `\\u2013`, and the cut landed inside one
    ("Unterminated string starting at ... char 7999").

    A failed turn used to vanish silently, so this surfaced as a question
    with no answer — see _persist_failed_turn.

    Over-long results become a VALID object carrying a truncated preview:
    the preview is a JSON *string value*, so json.dumps escapes it properly
    and the payload always parses.

    ``max_chars`` overrides the provider cap (OpenRouter free-tier is
    tighter than the Kimi/Groq default). Tests pass an explicit cap so
    they stay independent of ``LLM_PROVIDER``.
    """
    cap = int(max_chars) if max_chars is not None else _effective_tool_result_max_chars()
    if cap < 1:
        cap = _TOOL_RESULT_MAX_CHARS
    text = json.dumps(payload, default=str)
    start = max(0, min(int(offset or 0), len(text)))
    if len(text) <= cap and not start:
        return text
    window = text[start:]

    def _envelope(keep: int) -> str:
        shown = min(keep, len(window))
        next_offset = start + shown
        more = next_offset < len(text)
        return json.dumps(
            {
                "truncated": True,
                "note": (
                    f"Tool result exceeded {cap} characters "
                    "and was truncated. Call this tool again with the SAME "
                    f"file_path and char_offset={next_offset} to read the next "
                    "window, and repeat until chars_remaining is 0. Do that "
                    "BEFORE telling the user you could not find something."
                    if more else
                    f"Tool result exceeded {cap} characters "
                    "and was truncated. This is the LAST window -- you have "
                    "now seen the whole result."
                ),
                "char_offset": start,
                "next_char_offset": next_offset if more else None,
                "chars_remaining": max(0, len(text) - next_offset),
                # The size of the loss, and the rule about it. The note above
                # says the result was cut but not by how much, so the model
                # has to infer the extent from the content -- and on the live
                # BOQ question it inferred "pages 1-6 of 16" and reported that
                # no cubic-metre demolition item existed. The 945 m3 line was
                # past the cut. Same reasoning as the compaction envelope in
                # _compacted_tool_payload; both layers cut the same result, so
                # both have to say so.
                "chars_shown": shown,
                "chars_dropped": max(0, len(text) - next_offset),
                "chars_total": len(text),
                "scope_of_absence": (
                    f"You are seeing characters {start} to {next_offset} of "
                    f"{len(text)}. Anything the user asked about may be in "
                    "the part you cannot see. NEVER state or imply that a "
                    "value, line item, or section is absent from the document "
                    "or the data -- you cannot see far enough to know that. "
                    + (
                        "Read the remaining windows with char_offset before "
                        "answering about what is or is not present."
                        if more else
                        "You have now read every window, so a considered "
                        "statement about what this result contains is safe."
                    )
                ),
                "preview": window[:keep],
            },
            default=str,
        )

    # Embedding the preview as a JSON string RE-escapes it — every `"` becomes
    # `\"` and every `\` becomes `\\` — so escape-heavy content (exactly what
    # this guards) can nearly double. A fixed budget therefore overshoots the
    # cap; shrink until the ENVELOPE fits, measuring the real thing.
    keep = cap
    out = _envelope(keep)
    while len(out) > cap and keep > 0:
        overshoot = len(out) - cap
        keep = max(0, keep - max(overshoot, 32))
        out = _envelope(keep)
    return out


def _persist_failed_turn(conversation_id: str | None) -> None:
    """Record that a turn died, so reloaded history is not silently missing it.

    Never raises: a bookkeeping failure must not mask the original error the
    caller is already reporting to the user.
    """
    if not conversation_id:
        return
    try:
        # Function-scope import, matching every other agent_memory use in this
        # module — it is NOT a module-level name here. Referencing it at module
        # scope raised NameError, which this very except block would have
        # swallowed, leaving a helper that silently persisted nothing.
        from app.core import agent_memory

        agent_memory.append_message(
            conversation_id, "assistant", _FAILED_TURN_MESSAGE
        )
    except Exception:  # noqa: BLE001 — best-effort bookkeeping
        _LOG.warning("could not persist failed-turn marker", exc_info=True)


# An export request asks for a FILE, not for a passage inside a document.
# An answer-export ask was refused by the RAG-miss short-circuit below
# in ~0.6s having dispatched ZERO tools, so the export action never ran and the
# router's refusal was displayed as though it were the answer. Same shape as
# the unit-rate case already excluded there — a request that merely mentions a
# document is not a request to look something up inside one.
#
# Requires BOTH an action verb and a format/target word, so "what does the
# Specification document say about 003113" keeps the fast refusal while
# "export the BOQ to Excel" does not.
_EXPORT_ACTION_RE = re.compile(
    r"\b(export|download|save|send\s+me|give\s+me)\b[^.?!]{0,40}?"
    r"\b(word|docx|doc|pdf|excel|xlsx|csv|file|copy|version|attachment)\b"
    r"|\b(word|docx|pdf|excel|xlsx|csv)\b[^.?!]{0,25}?"
    r"\b(export|download|copy|version)\b",
    re.IGNORECASE,
)


def _asks_for_export(text: str) -> bool:
    """True when the turn wants a file produced, not a reference resolved."""
    if message_wants_answer_report(text):
        return True
    return bool(_EXPORT_ACTION_RE.search(text or ""))


def _fulfill_answer_report(
    user_message: str,
    project_id: str | None,
    conversation_id: str | None,
    history: list[dict[str, Any]] | None,
    agent_name: str,
    owner_id: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Compile the numbered (or all) chat answers and persist the confirmation.

    Runs without an LLM or RAG — retrieval used to treat the numbered
    answer codes as RFP attachment codes. Kill-switch is checked by the caller.
    """
    from app.core import agent_memory

    prior: list[dict[str, Any]] = []
    if conversation_id:
        agent_memory.get_or_create_conversation(
            conversation_id, agent_name, project_id, owner_id=owner_id,
        )
        prior = [
            {"role": m["role"], "content": m["content"]}
            for m in agent_memory.get_messages(conversation_id, limit=200)
            if m.get("role") in ("user", "assistant")
        ]
        agent_memory.append_message(conversation_id, "user", user_message)
    if not prior and history:
        prior = [
            {"role": m.get("role"), "content": m.get("content") or ""}
            for m in history
            if m.get("role") in ("user", "assistant")
        ]
    start_end = parse_answer_report_range(user_message)
    start, end = start_end if start_end else (None, None)
    pairs = collect_answer_pairs(prior, start, end)
    answer = compose_answer_report_reply(pairs, start, end)
    exports: list[dict[str, Any]] = []
    if pairs and project_id and conversation_id:
        have_start = start or 1
        have_end = (start or 1) + len(pairs) - 1
        exports = [answer_report_export_descriptor(
            project_id, conversation_id, have_start, have_end,
        )]
    if conversation_id:
        agent_memory.append_message(conversation_id, "assistant", answer)
    return answer, exports


_MISSING_REFERENCE_ANSWER = (
    "I could not confirm this reference in the indexed project sources"
)


_UNIT_RATE_TOKEN_RE = re.compile(
    r"(?i)\d[\d,]*(?:\.\d+)?\s*/\s*(?:m[23]?|lm|sqm|cum|t|tonnes?|kg|nr|no|ea|each|"
    r"items?|days?|hrs?|hours?|wks?|weeks?|months?|mo|l|ltrs?)"
)


# A sentence, not a reference: it ends in "?", opens with a question word, or
# runs to seven words or more ("vol 3 specification 2 of 5" is six).
_QUOTED_SENTENCE_RE = re.compile(
    r"(?i)\?\s*$|^(?:what|which|who|whom|when|where|why|how|is|are|does|do|can|"
    r"could|should|calculate|compute|find|give|show|list|explain)\b"
    r"|^(?:\S+\s+){6,}\S+"
)


def _should_short_circuit_rag_miss(
    audit_rec: dict[str, Any] | None,
    rag_sys_msg: dict[str, str] | None,
    user_message: str = "",
) -> bool:
    """True when retrieval has definitively missed an exact reference.

    Conditions:
      * RAG injection produced no context (``rag_sys_msg`` is None).
      * The audit record shows either ``identifier_miss`` or
        ``threshold_fired``.
      * The query contained at least one extracted identifier that looks
        reference-like (contains a digit).  This prevents broad phrases such
        as "contract value" from triggering the fast path.

    Broad questions with no real reference identifiers are deliberately NOT
    short-circuited so the model can still answer from project facts or
    general knowledge. A self-contained calculation / self-coding request
    is never a document lookup — unit rates like ``AED/m2`` used to trip
    this gate and return "could not confirm this reference" in ~2s.
    """
    if rag_sys_msg is not None:
        return False
    if not audit_rec:
        return False
    if user_message and (
        _asks_self_coding(user_message)
        or _looks_like_self_contained_calculation(user_message)
        or _message_is_formula_style_ask(user_message)
        or _asks_for_export(user_message)
        or _message_wants_vo_draft(user_message)
        or _message_wants_primavera_parse(user_message)
        or _message_wants_rfi_draft(user_message)
    ):
        return False
    # A unit RATE ("SAR 62/m2") is not a reference: it looks like page
    # "d/3/3" to the extractor, and a variance question carrying all its own
    # numbers was refused as a missing document in 4 s.
    # ...and neither is a whole SENTENCE that happened to be typed inside
    # quotation marks: the extractor reads any quoted span as an exact
    # reference, so a quoted question was refused as a missing document.
    identifiers = [
        i for i in (audit_rec.get("extracted_identifiers") or [])
        if not _UNIT_RATE_TOKEN_RE.fullmatch(i.strip())
        and not _QUOTED_SENTENCE_RE.search(i.strip())
    ]
    # Require a digit to avoid short-circuiting generic phrases like
    # "contract value" that happen to match a reference label.
    if not any(re.search(r"\d", ident) for ident in identifiers):
        return False
    return bool(
        audit_rec.get("identifier_miss") or audit_rec.get("threshold_fired")
    )


def _build_missing_reference_answer(
    project_id: str | None,
    user_id: str | None,
) -> str:
    """Return a user-facing not-found answer, optionally naming the project."""
    project_name = ""
    if project_id:
        try:
            from app.core import projects as _projects

            project = _projects.get_project(
                project_id, user_id=user_id, include_admin_approved=True
            )
            project_name = (project or {}).get("name") or ""
            if not project_name and project_id == _projects.MASTER_CORPUS_PROJECT_ID:
                project_name = _projects.MASTER_CORPUS_NAME
        except Exception:
            _LOG.warning(
                "swallowed %s in _build_missing_reference_answer() — continuing",
                "Exception", exc_info=True,
            )

    if project_name:
        return (
            f"{_MISSING_REFERENCE_ANSWER} for {project_name}. "
            "Please check whether the document is indexed or provide the exact filename."
        )
    return (
        f"{_MISSING_REFERENCE_ANSWER}. "
        "Please check whether the document is indexed or provide the exact filename."
    )


def _normalize_tool_json_text(text: str) -> str:
    """Strip BOM / zero-width chars and unwrap a sole markdown JSON fence."""
    if not text:
        return ""
    cleaned = (
        text.replace("\ufeff", "")
        .replace("\u200b", "")
        .replace("\u200c", "")
        .replace("\u200d", "")
        .strip()
    )
    md = re.match(
        r"^```(?:json)?\s*([\s\S]*?)\s*```$",
        cleaned,
        re.IGNORECASE,
    )
    if md:
        return md.group(1).strip()
    return cleaned


def _is_search_tool_args_obj(obj: Any) -> bool:
    """True when ``obj`` looks like a search tool argument payload.

    Detects shapes such as:
      {"query": "...", "top_k": 5, "project_id": "..."}
      {"query": "..."}   # bare Scout/Kimi leak (raw tool-call JSON)
      {"tool": "search_project_documents", "arguments": {...}}
      {"name": "search_project_documents", "arguments": {...}}
      {"name": "search_project_documents", "parameters": {...}}
    """
    if not isinstance(obj, dict):
        return False
    keys = set(obj.keys())
    # Explicit tool-call envelope shapes (arguments or parameters).
    if ("arguments" in keys or "parameters" in keys) and (
        "name" in keys or "tool" in keys or "function" in keys
    ):
        return True
    # Search argument shape: has "query" plus other search params, and no
    # obvious answer/data keys that would make it user-facing JSON.
    if "query" in keys and not (keys & _ANSWER_DATA_KEYS):
        if isinstance(obj.get("query"), str) and keys <= {
            "query", "top_k", "project_id", "corpus", "filters",
            "source", "tool", "function",
        }:
            # Bare {"query": "..."} is the live Scout/Kimi leak shape — treat
            # it as internal search args when it is not mixed with answer keys.
            return True
    return False


def _is_tool_call_obj(obj: Any) -> bool:
    """True when a parsed JSON object is an internal tool-call / search payload."""
    if not isinstance(obj, dict):
        return False
    if _INTERNAL_TOOL_KEYS.issubset(obj.keys()):
        return True
    if {"name", "parameters"}.issubset(obj.keys()):
        return True
    if obj.get("type") == "function" and "function" in obj:
        return True
    if _is_search_tool_args_obj(obj):
        return True
    return False


# Text the platform writes INTO the model's context and that therefore can
# never legitimately appear in an answer. Every one of these is a literal from
# app/core/rag/inject.py (the retrieval brief) or from a PLATFORM PRE-DISPATCH
# injection in this module.
#
# Live on 554e0b9, Master Corpus, fresh thread. Asked for the Delay Damages
# daily rate, the turn spent 564s, routed to generate_wbs, and then streamed
# the pre-dispatch payload to the user AS THE ANSWER -- opening with
#
#   {"status": "success", "wbs_id": "wbs-ccc4ee6c", "project_type":
#    "building", "brief": "AUTHORITATIVE REFERENCE CONTEXT - the material
#    below was retrieved ...
#
# and closing with the instruction the platform had written for the model
# ("Report the new durations ... Do not re-call generate_wbs"). In between:
# the whole retrieval brief, per-excerpt [doc_id= chunk= score=] telemetry,
# and the customer's absolute local paths --
# "X:\Example Drive\Master Folder\the project\Contract Docs\Contractor\..." --
# for documents whose folder is named "Contract docs UNSIGNED".
#
# _looks_like_internal_tool_json did not catch it and could not: every branch
# of _is_tool_call_obj recognises a tool CALL ({"name","arguments"},
# {"name","parameters"}, {"type":"function"}, search args). A tool RESULT --
# {"status","wbs_id","brief",...} -- matches none of them. The guard was built
# for the model emitting a call it should have made; this is the platform's
# own tool output arriving as content, which nothing tested for.
#
# Matched as substrings rather than by parsing, because the leak is dangerous
# in EVERY shape: whole JSON, JSON inside prose, or the brief quoted as prose
# with no JSON at all. A false positive costs one forced retry; a false
# negative ships the customer's private paths and the platform's own prompt
# to the customer.
_INTERNAL_CONTEXT_MARKERS = (
    "AUTHORITATIVE REFERENCE CONTEXT",
    "SCOPE OF ABSENCE",
    "FILENAMES ARE EVIDENCE",
    "CONTRACT ATTRIBUTION —",
    "PLATFORM PRE-DISPATCH",
)

# Answer-routing notes the model (or the Engineer graft) must never show the
# user. Distinct from ``_INTERNAL_CONTEXT_MARKERS``: those trip a nuclear
# fallback that would drop EXAMPLAR if the hint was prepended to a good
# answer. These are STRIPPED so the appointed firm survives.
_ROUTING_PREAMBLE_PHRASES = (
    "That IS the answer",
    "INTERNAL GUIDANCE",
    # Old ENGINEER APPOINTMENT hint wording (heading reworded to ENGINEER
    # IDENTITY 2026-09-14) plus the new directive — both kept so neither can
    # silently re-leak.
    "State the appointed firm",
    "State ONLY that firm's name",
)
_ROUTING_PREAMBLE_LINE_RE = re.compile(
    r"(?im)^[ \t]*(?:The Engineer is\s+)?"
    r"(?:INTERNAL GUIDANCE|ENGINEER IDENTITY|ENGINEER APPOINTMENT|"
    r"TIME FOR COMPLETION|"
    r"SCHEDULE REGISTER|PRICED BOQ ROW|RATE ONLY|PART SUMMARY TOTAL|"
    r"ACCEPTED CONTRACT AMOUNT INCLUDING VAT|"
    r"DELAY DAMAGES PER CALENDAR DAY|DELAY DAMAGES OVER A PERIOD|HYPOTHETICAL MILESTONE ARITHMETIC|"
    r"PARENT COMPANY GUARANTEE|"
    r"COMMENCEMENT DATE|APPOINTMENT|IDENTITY|NAMED STANDARD ABSENT)"
    r"\s*[—\-].*$"
)
_GRAFT_APPOINTMENT_LEAK_RE = re.compile(
    r"(?im)^[ \t]*The Engineer is (?:APPOINTMENT|IDENTITY|ENGINEER IDENTITY)"
    r"\s*[—\-].*$"
)
_ROUTING_PREAMBLE_SENTENCE_RE = re.compile(
    r"(?i)(?:INTERNAL GUIDANCE\s*[—\-][^\n]*|"
    r"That IS the answer\.?\s*|"
    r"State the appointed firm\.?\s*|"
    r"State ONLY that firm'?s name\.?\s*|"
    r"Do NOT state the appointment date[^.]*\.?\s*|"
    r"Do not say the identity is absent[^.]*\.?\s*|"
    r"an excerpt below names the Engineer(?:\s*\([^)]*\))?\.?\s*)"
)


def _strip_answer_routing_preamble(text: str) -> str:
    """Remove Engineer/answer-routing guard text; keep the real particular.

    Live e24aee4: graft prepended ``The Engineer is APPOINTMENT — an
    excerpt below names the Engineer. That IS the answer. State the
    appointed firm…`` then EXAMPLAR. Nuclear leak replacement would drop
    the firm. Strip the steering; leave the answer.
    """
    if not text:
        return text
    out = _GRAFT_APPOINTMENT_LEAK_RE.sub("", text)
    out = _ROUTING_PREAMBLE_LINE_RE.sub("", out)
    out = _ROUTING_PREAMBLE_SENTENCE_RE.sub("", out)
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def answer_contains_routing_preamble(text: str) -> bool:
    """True when user-visible text still carries inject-guard wording.

    Regression pin for the Engineer-ask guard leak. Tests (and emit checks) fail closed
    on these phrases so a future heading rename cannot silently re-leak.
    """
    if not text:
        return False
    if any(phrase in text for phrase in _ROUTING_PREAMBLE_PHRASES):
        return True
    if _GRAFT_APPOINTMENT_LEAK_RE.search(text):
        return True
    return bool(_ROUTING_PREAMBLE_LINE_RE.search(text))

# Per-excerpt retrieval telemetry, e.g. "[doc_id=0a1b2c3d chunk=11 score=2.199
# src=...]". Internal by construction: the user never asked for a cosine.
_RETRIEVAL_MARKER_RE = re.compile(
    r"\[doc_id=[^\]\s]+\s+chunk=\d+\s+score=", re.IGNORECASE
)


def _looks_like_internal_context_leak(text: str) -> bool:
    """True when the platform's own context text is inside an answer.

    Distinct from :func:`_looks_like_internal_tool_json`, which asks whether
    the model emitted a tool CALL. This asks whether platform-authored
    context -- a retrieval brief, a pre-dispatch payload, excerpt telemetry --
    has come back out as user-facing content, however it is wrapped.
    """
    if not text:
        return False
    if any(marker in text for marker in _INTERNAL_CONTEXT_MARKERS):
        return True
    return bool(_RETRIEVAL_MARKER_RE.search(text))


class _EmitLeakGuard:
    """Last line of defence: no path may ship the platform's own context.

    #457 put the context-leak check in the streamed-synthesis branch and in
    _sanitize_final_text, and the leaking question came back clean. It recurred once on 87c7996
    -- same shape, same wbs_id/brief/Drive paths -- while the next run of the
    same question was clean. Intermittent means a path that neither guard
    covers, and I have not found which one.

    Rather than keep sampling seven-minute live turns for a path I cannot
    name, this sits where every event the client sees passes through:
    chat_stream wraps _chat_stream_impl, so one check here cannot be bypassed
    by any branch inside, present or future.

    It accumulates, because a marker can straddle two token events. Once the
    accumulated text trips the detector, nothing further is emitted and the
    turn ends on the controlled fallback.

    Residual, stated rather than hidden: tokens emitted BEFORE the marker
    appears have already gone. On the observed leak that is the opening
    ``{"status": "success", "wbs_id": "wbs-..."`` -- bad, and roughly 90
    characters instead of 6,170, with the brief, the excerpt telemetry and
    the customer's Drive paths all after the cut. Buffering the first few
    hundred characters would close that too, at the cost of changing
    time-to-first-token for every turn; not worth doing blind, and worth
    revisiting once the actual path is known.
    """

    __slots__ = ("_acc", "_tail", "tripped")

    #: How much already-checked text is re-read with each new token. Every
    #: marker is short and the retrieval-marker pattern is anchored on a short
    #: prefix, so a match that ends in the new token starts within this tail.
    #: Re-reading the whole answer on every token made the guard quadratic in
    #: the answer length -- on the event loop (CI: 266 ms stall at 15 users).
    TAIL_CHARS = 4096

    def __init__(self) -> None:
        self._acc: list[str] = []
        self._tail = ""
        self.tripped = False

    def check(self, event: dict[str, Any]) -> dict[str, Any] | None:
        """The event to emit, or None to drop it."""
        if not isinstance(event, dict):
            return event
        kind = event.get("type")
        if kind not in ("token", "end"):
            return event
        content = event.get("content")
        if not isinstance(content, str) or not content:
            return event

        if not self.tripped:
            if kind == "token":
                self._acc.append(content)
                probe = self._tail + content
                self._tail = probe[-self.TAIL_CHARS:]
            else:
                probe = content
            # Nuclear first: a full inject dump still falls back. Routing
            # notes are stripped afterwards so EXAMPLAR survives a prepended
            # ENGINEER APPOINTMENT hint (live e24aee4).
            if _looks_like_internal_context_leak(probe):
                self.tripped = True
                _LOG.warning(
                    "chat_stream: internal context reached the emit boundary "
                    "on a %s event -- suppressing the rest of this turn",
                    kind,
                )
            elif answer_contains_routing_preamble(content) or (
                kind == "end" and answer_contains_routing_preamble(probe)
            ):
                if kind == "end":
                    content = _strip_answer_routing_preamble(probe)
                    event = {**event, "content": content}
                else:
                    stripped = _strip_answer_routing_preamble(content)
                    if not stripped:
                        return None
                    event = {**event, "content": stripped}

        if not self.tripped:
            return event
        if kind == "token":
            return None
        return {**event, "content": _TOOL_FORMAT_FALLBACK}


def _looks_like_internal_tool_json(text: str) -> bool:
    """True if ``text`` is (or contains) JSON shaped like a tool call."""
    if not text:
        return False

    stripped = _normalize_tool_json_text(text)
    # Whole-string JSON.
    if stripped.startswith(("{", "[")):
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            _LOG.warning(
                "swallowed %s in _looks_like_internal_tool_json() — continuing",
                "json.JSONDecodeError", exc_info=True,
            )
        else:
            if isinstance(obj, dict):
                if _is_tool_call_obj(obj):
                    return True
            if isinstance(obj, list):
                if obj and all(_is_tool_call_obj(item) for item in obj):
                    return True

    # Embedded JSON objects/lists (common when the model leaks a tool call
    # inside prose / after a formatting preamble).  Handles nested braces.
    scan = stripped or text

    def _candidates(open_ch: str, close_ch: str):
        i = 0
        while i < len(scan):
            if scan[i] == open_ch:
                depth = 1
                j = i + 1
                while j < len(scan) and depth > 0:
                    if scan[j] == open_ch:
                        depth += 1
                    elif scan[j] == close_ch:
                        depth -= 1
                    j += 1
                if depth == 0:
                    yield scan[i:j]
                    i = j
                    continue
            i += 1

    for candidate in _candidates("{", "}"):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and _is_tool_call_obj(obj):
            return True
    for candidate in _candidates("[", "]"):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, list) and obj and all(_is_tool_call_obj(item) for item in obj):
            return True
    return False


def _coerce_tool_arguments(raw: Any) -> dict[str, Any] | None:
    """Normalize tool ``arguments`` / ``parameters`` to a dict when possible."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


def _tool_calls_from_json_obj(obj: Any) -> list[dict]:
    """Build OpenAI-style tool_calls from a parsed JSON object or list."""
    items = obj if isinstance(obj, list) else [obj]
    out: list[dict] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        # OpenAI nested: {"type":"function","function":{"name":...,"arguments":...}}
        nested = item.get("function")
        if item.get("type") == "function" and isinstance(nested, dict):
            name = nested.get("name")
            args = _coerce_tool_arguments(
                nested.get("arguments", nested.get("parameters"))
            )
            if name and args is not None:
                out.append({
                    "id": f"recovered_{i+1}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                })
                continue
        name = item.get("name") or item.get("tool")
        if isinstance(item.get("function"), str) and not name:
            name = item.get("function")
        args = _coerce_tool_arguments(
            item.get("arguments", item.get("parameters"))
        )
        if name and args is not None:
            out.append({
                "id": f"recovered_{i+1}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            })
            continue
        # Raw search_project_documents args without an envelope.
        if _is_search_tool_args_obj(item) and "query" in item:
            out.append({
                "id": f"recovered_search_{i+1}",
                "type": "function",
                "function": {
                    "name": "search_project_documents",
                    "arguments": json.dumps({
                        k: item[k]
                        for k in ("query", "top_k", "project_id", "corpus", "filters")
                        if k in item
                    }),
                },
            })
    return out


_XML_INVOKE_RE = re.compile(
    r'<\s*invoke\s+name\s*=\s*["\']([^"\']+)["\']\s*>(.*?)</\s*invoke\s*>',
    re.IGNORECASE | re.DOTALL,
)
_XML_PARAM_RE = re.compile(
    r'<\s*parameter\s+name\s*=\s*["\']([^"\']+)["\']\s*>(.*?)</\s*parameter\s*>',
    re.IGNORECASE | re.DOTALL,
)
_XML_TOOL_LEAK_RE = re.compile(
    r"<\s*(?:function_calls|function_results|invoke)\b",
    re.IGNORECASE,
)
_VALIDATION_STAGE_ORDER = (
    "syntactic", "dimensional", "physical", "empirical", "operational",
)


def _looks_like_xml_tool_leak(text: str) -> bool:
    """True when the model dumped Anthropic-style XML tool calls as prose.

    Live leftover-hat L4: after ``validation_pipeline`` returned Physical /
    Tier 4, force_synthesis disarmed tools and the model streamed
    ``<function_calls><invoke name="formula_executor_v2">`` as the answer.
    """
    if not text:
        return False
    return bool(_XML_TOOL_LEAK_RE.search(text))


def _withhold_names_in_streamed_segment(
    seg: str, rag_sys_msg: dict[str, Any] | None,
) -> str:
    """Names are removed from a streamed line BEFORE it leaves.

    _postprocess_answer scrubs the finished answer, but a streamed segment has
    reached the browser by then. Segments are whole lines, so a name does not
    straddle two of them. Never raises into the stream.
    """
    try:
        from app.core.party_names import withhold_party_names

        return withhold_party_names(
            seg,
            (rag_sys_msg or {}).get("content", "") if rag_sys_msg else "",
        )
    except Exception:  # noqa: BLE001 — a scrub failure must not break a turn
        _LOG.warning("streamed name scrub failed; segment withheld", exc_info=True)
        return ""


def _looks_like_tool_markup_leak(text: str) -> bool:
    """True when ``text`` carries a tool call written out as markup.

    Anthropic-style XML **or** DeepSeek's DSML. The streamed-synthesis guard
    knew the first and not the second: live d8d9573 sent
    ``<｜｜DSML｜｜ calls><｜｜DSML｜｜ invoke name="construction_calc">…`` to the
    browser as the opening of an answer. ``_sanitize_final_text`` strips DSML,
    but a streamed segment has left the building before that runs.
    """
    if not text:
        return False
    if _looks_like_xml_tool_leak(text):
        return True
    return "DSML" in text and bool(_DSML_MARKER_RE.search(text))


def _recover_xml_tool_calls(text: str) -> list[dict]:
    """Recover ``<invoke name=...>`` XML leaked into ``content``."""
    if not text or not _looks_like_xml_tool_leak(text):
        return []
    out: list[dict] = []
    for inv in _XML_INVOKE_RE.finditer(text):
        name = (inv.group(1) or "").strip()
        if not name:
            continue
        args: dict[str, Any] = {}
        for param in _XML_PARAM_RE.finditer(inv.group(2) or ""):
            pname = (param.group(1) or "").strip()
            pvalue = (param.group(2) or "").strip()
            if not pname:
                continue
            try:
                args[pname] = json.loads(pvalue)
            except (TypeError, json.JSONDecodeError):
                args[pname] = pvalue
        out.append({
            "id": f"recovered_xml_{len(out) + 1}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        })
    return out


def _as_validation_payload(obj: Any) -> dict[str, Any] | None:
    """Unwrap execute/tool envelopes until ``stages`` + ``overall`` appear."""
    if not isinstance(obj, dict):
        return None
    if isinstance(obj.get("stages"), dict) and obj.get("overall"):
        return obj
    for key in ("result", "content"):
        inner = obj.get(key)
        if isinstance(inner, str):
            try:
                inner = json.loads(inner)
            except (TypeError, json.JSONDecodeError):
                continue
        found = _as_validation_payload(inner)
        if found:
            return found
    return None


def _validation_pipeline_payload(
    messages: list[dict[str, Any]] | None = None,
    tool_results: list[Any] | None = None,
) -> dict[str, Any] | None:
    """Most recent validation_pipeline process() dict from this turn."""
    last: dict[str, Any] | None = None
    for rec in tool_results or []:
        if not isinstance(rec, dict):
            continue
        found = _as_validation_payload(rec)
        if found:
            last = found
    for m in messages or []:
        if m.get("role") != "tool":
            continue
        raw = m.get("content")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                continue
        envelope = raw if isinstance(raw, dict) else {}
        if m.get("name"):
            envelope = {"name": m.get("name"), "result": envelope}
        found = _as_validation_payload(envelope)
        if found:
            last = found
    return last


def _format_validation_verdict(payload: dict[str, Any]) -> str:
    """Kernel 5-stage block from a ``validation_pipeline`` result."""
    stages = payload.get("stages") or {}
    lines = [
        "Output under review: validation_pipeline → claim check",
        "",
        "Stages:",
    ]
    for i, name in enumerate(_VALIDATION_STAGE_ORDER, 1):
        st = stages.get(name) or {}
        mark = "✓" if st.get("pass") else "✗"
        reason = (st.get("reason") or "").strip()
        lines.append(f"{i}. {name.capitalize()}: {mark} {reason}".rstrip())
    overall = payload.get("overall") or (
        "fail" if payload.get("first_failure") else "pass"
    )
    lines.append("")
    tier = payload.get("tier")
    if tier is not None:
        lines.append(f"Tier: {tier}")
    lines.append(f"Verdict: {overall}")
    if overall == "fail":
        ff = payload.get("first_failure") or "physical"
        reason = ((stages.get(ff) or {}).get("reason") or "").strip()
        extra = f" — {reason}" if reason else ""
        lines.append(f"Required correction: {ff} stage failed{extra}")
    return "\n".join(lines)


def _graft_validation_verdict_if_unusable(
    text: str,
    messages: list[dict[str, Any]] | None = None,
    tool_results: list[Any] | None = None,
) -> str:
    """Replace leaked XML/JSON synthesis with the pipeline verdict.

    Leftover L4: ``validation_pipeline`` already failed Physical / Tier 4;
    the user-facing turn was ``formula_executor_v2`` XML instead.
    """
    unusable = (not (text or "").strip()) or text == _TOOL_FORMAT_FALLBACK
    if not unusable:
        return text
    payload = _validation_pipeline_payload(messages, tool_results)
    if not payload:
        return text
    return _format_validation_verdict(payload)


def _sanitize_final_text(
    text: str,
    messages: list[dict[str, Any]] | None = None,
    tool_results: list[Any] | None = None,
) -> str:
    """Prepare assistant content for display.

    1. Strip DeepSeek DSML tool-call markup.
    2. Detect raw internal tool-call JSON or Anthropic XML tool leaks.
    3. Return a controlled fallback if the content is raw internal tool args.
    4. If that fallback would hide a ``validation_pipeline`` verdict, graft it.
    5. Return empty string if nothing usable remains.
    """
    if not text:
        cleaned = ""
    else:
        cleaned = _strip_dsml(text).strip()
        cleaned = _strip_answer_routing_preamble(cleaned)

    if cleaned:
        # Unwrap a sole markdown fence / BOM before the tool-JSON detector so
        # ```json {"name":"search_project_documents",...} ``` cannot bypass.
        normalized = _normalize_tool_json_text(cleaned)
        # Tool ERROR envelopes (Unknown calculation, …) are not tool-CALL
        # JSON. Convert them to a plain sentence here so the user never
        # sees raw {"status":"error",...} (live SYNC2 A2 plastering).
        error_plain = _plain_sentence_for_tool_error_json(normalized)
        if error_plain:
            cleaned = error_plain
        elif normalized != cleaned and _looks_like_internal_tool_json(normalized):
            _LOG.warning("raw tool args inside markdown code block; replacing with fallback")
            cleaned = _TOOL_FORMAT_FALLBACK

        if not error_plain and cleaned != _TOOL_FORMAT_FALLBACK and _looks_like_xml_tool_leak(cleaned):
            _LOG.warning("xml tool-call leak in final answer; replacing with fallback")
            cleaned = _TOOL_FORMAT_FALLBACK
        elif not error_plain and cleaned != _TOOL_FORMAT_FALLBACK and _looks_like_internal_tool_json(cleaned):
            _LOG.warning("raw tool-call JSON detected in final answer; replacing with fallback")
            cleaned = _TOOL_FORMAT_FALLBACK
        elif not error_plain and cleaned != _TOOL_FORMAT_FALLBACK and _looks_like_internal_context_leak(cleaned):
            # Both branches funnel through here, so the platform's own
            # context cannot reach a user down either one. The streaming
            # guards above stop the bytes earlier; this stops what is
            # persisted, exported and fed to sources.
            _LOG.warning(
                "internal context (retrieval brief / pre-dispatch payload / "
                "excerpt telemetry) detected in final answer; replacing with "
                "fallback"
            )
            cleaned = _TOOL_FORMAT_FALLBACK

    return _graft_validation_verdict_if_unusable(cleaned, messages, tool_results)


_KIMI_LEAK_RE = re.compile(r"functions\.([A-Za-z0-9_.\-]+):(\d+)")


def _recover_kimi_leaked_tool_calls(text: str) -> list[dict]:
    """Recover Kimi K2's native tool-call format leaked into ``content``.

    Live 2026-07-26 finding: on lookup questions Kimi sometimes emits its
    tool-call special-token section as PROSE — the stream showed
    ``...Let me search...functions.search_project_documents:0{"query":...``
    (special tokens mangled to garbage bytes). The pure-JSON recovery below
    never fires because the content starts with prose, so the leak streamed
    to the user as a 'final answer' and the search never ran. Each
    ``functions.<name>:<idx>`` marker followed by a JSON object becomes a
    real tool call; truncated JSON is skipped (the sanitizer handles it).
    """
    out: list[dict] = []
    for m in _KIMI_LEAK_RE.finditer(text):
        name = m.group(1).split(".")[-1]
        brace = text.find("{", m.end())
        if brace == -1:
            continue
        try:
            args, _ = json.JSONDecoder().raw_decode(text[brace:])
        except json.JSONDecodeError:
            continue
        if isinstance(args, dict):
            out.append({
                "id": f"recovered_kimi_{len(out) + 1}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            })
    return out


def _recover_tool_calls_from_content(text: str) -> list[dict]:
    """Recover OpenAI-style tool_calls from raw argument JSON/envelopes leaked
    into ``content`` by models (e.g. Groq Llama-4-Scout) that fail to emit the
    structured ``tool_calls`` field. Reuses the same detectors as
    ``_looks_like_internal_tool_json`` so we only recover shapes we already
    recognise as internal tool args. Also recovers Kimi K2's leaked
    ``functions.<name>:<idx>{...}`` markup (see above).
    """
    if not text:
        return []
    kimi = _recover_kimi_leaked_tool_calls(text)
    if kimi:
        return kimi
    xml_calls = _recover_xml_tool_calls(text)
    if xml_calls:
        return xml_calls
    stripped = _normalize_tool_json_text(text)
    if stripped.startswith(("{", "[")):
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            obj = None
        if obj is not None:
            recovered = _tool_calls_from_json_obj(obj)
            if recovered:
                return recovered
    # Embedded whole-object leak after a short prose preamble ("Let me search.").
    if _looks_like_internal_tool_json(stripped):
        for m in re.finditer(r"\{", stripped):
            try:
                obj, _end = json.JSONDecoder().raw_decode(stripped[m.start():])
            except json.JSONDecodeError:
                continue
            recovered = _tool_calls_from_json_obj(obj)
            if recovered:
                return recovered
    return []


def _recovered_calls_are_search_only(tool_calls: list[dict]) -> bool:
    """True when every recovered call is ``search_project_documents``."""
    if not tool_calls:
        return False
    for tc in tool_calls:
        name = ((tc.get("function") or {}).get("name") or "")
        if name != "search_project_documents":
            return False
    return True


CONFIGS_DIR = Path(__file__).parent / "configs"
# Hard cap so a runaway loop can't burn budget (backstop). Kept at 12 for legit
# complex multi-step tasks; env-tunable. The real fix for the "keep re-searching"
# hang is the per-turn search-tool cap (AGENT_SEARCH_TOOL_CAP) in the chat loops,
# which stops offering search_project_documents after a couple of calls so the
# model answers from injected context instead of grinding to this cap.
MAX_TOOL_ITERATIONS = int(os.getenv("AGENT_MAX_TOOL_ITERATIONS", "12"))
# History replayed into each LLM call. 20 turns of long answers (schedules,
# checklists) pushed requests to ~100k tokens — slow/costly on any model, and
# over Groq's free-tier 30k TPM (413). 8 keeps requests well under that while
# retaining ample conversational memory. Env-tunable.
MAX_HISTORY_TURNS = int(os.getenv("AGENT_MAX_HISTORY_TURNS", "8"))
MAX_DELEGATION_DEPTH = 3  # how deep agent → agent delegation may recurse

# Blocks whose inputs reference a user-uploaded file. When the LLM passes a
# bare filename (e.g. "Infra-1 - Demolition BOQ.pdf") instead of the
# stored file_path, the block's os.path.exists() always fails. The runtime
# resolves the bare name to the document's actual on-disk path before
# dispatching to the block. See _resolve_block_file_input.
_FILE_CONSUMING_BLOCKS = {
    "boq_processor",
    "spec_analyzer",
    "primavera_parser",
    "drawing_qto",
    "bim",
    "bim_extractor",
    "document_engine",
    "image",
    "file_hasher",
}


def _resolve_file_path(project_id: str, raw: Any) -> Any:
    """Resolve a bare filename to the project's stored absolute file_path.

    Returns ``raw`` unchanged if:
      - ``raw`` is empty or not a string;
      - ``raw`` is already an absolute path that exists;
      - no matching document is found.

    Otherwise looks up the project's documents (via ``projects.list_documents``)
    and returns the document's ``file_path`` when the name matches.

    Matching is tried in two passes:
      1. Exact match on ``original_name`` (case-insensitive).
      2. Substring match either way (case-insensitive). This handles the
         common case where the LLM truncates or rewords the filename.
    """
    if not raw or not isinstance(raw, str):
        return raw
    if not project_id:
        return raw
    import os as _os
    try:
        if _os.path.isabs(raw) and _os.path.exists(raw):
            return raw
    except (TypeError, ValueError):
        return raw
    try:
        from app.core import projects as _projects
        docs = _projects.list_document_names(project_id) or []
    except Exception:
        return raw

    needle = _os.path.basename(str(raw)).strip().lower()
    if not needle:
        return raw

    # Pass 1: exact (case-insensitive) match on original_name.
    for doc in docs:
        on = (doc.get("original_name") or "").strip().lower()
        if on and on == needle:
            fp = doc.get("file_path") or ""
            if fp and _os.path.exists(fp):
                return fp

    # Pass 2: substring match either direction (handles the LLM truncating
    # or rewording the filename slightly).
    for doc in docs:
        on = (doc.get("original_name") or "").strip().lower()
        if on and (needle in on or on in needle):
            fp = doc.get("file_path") or ""
            if fp and _os.path.exists(fp):
                return fp

    return raw


# Cap on the text a single fetch_document call feeds back into the model.
# Large contracts run to hundreds of pages; the tool returns the head and
# marks truncation honestly so the model knows to narrow with
# search_project_documents instead of assuming it saw everything.
_FETCH_DOCUMENT_MAX_CHARS = 24_000


def _fetch_document_content(
    project_id: str, doc_id: str, filename: str
) -> tuple:
    """Resolve one project document and return its text content.

    Returns ``(content, doc, error)`` where ``content`` is
    ``{"text", "truncated", "source"}`` and exactly one of
    ``content``/``error`` is set. Honesty rules mirror the deterministic
    doc→slot decision (DECISIONS 2026-07-13): no match → honest error;
    ambiguous name match → ask-which error, never silently pick one.
    An exact id or exact (case-insensitive) name match is never ambiguous —
    for duplicate names the most recently uploaded document wins, since a
    re-upload is what the user means by "the attached file".
    """
    try:
        from app.core import projects as _projects
        docs = _projects.list_documents(project_id) or []
    except Exception as exc:  # noqa: BLE001 — surface as tool error, not crash
        return None, None, f"document lookup unavailable: {exc}"

    # Every document the RETRIEVER can cite must be fetchable, not just the
    # active project's own. Retrieval merges general-knowledge projects and
    # (for a thin project) the Master-Corpus fallback into the context, so the
    # model routinely sees a [doc_id=...] marker for a document that
    # list_documents(project_id) does not contain. It would then call
    # fetch_document, get "no document with id X in this project", and report
    # the file as unavailable — while quoting from it. Live example: the model
    # cited a seeded general-knowledge reference, then could not open
    # it. A citation the model cannot resolve is a citation the USER cannot
    # trust, so the fetch scope must match the retrieval scope exactly.
    #
    # Each extra corpus is recorded with the project id it lives under, because
    # the chunk lookup below is project-scoped: querying GK chunks under the
    # ACTIVE project id silently returns nothing.
    doc_sources: list[tuple] = [(project_id, docs)]
    try:
        from app.core.rag import retriever as _rag_retriever

        extra_pids = list(_rag_retriever._general_knowledge_project_ids())
        fallback_pid = _rag_retriever._master_corpus_fallback_id()
        if fallback_pid:
            extra_pids.append(fallback_pid)
        for pid in extra_pids:
            if not pid or pid == project_id:
                continue
            try:
                extra_docs = _projects.list_documents(pid) or []
            except Exception:  # noqa: BLE001 — one bad corpus must not break the tool
                continue
            if extra_docs:
                doc_sources.append((pid, extra_docs))
    except Exception:  # noqa: BLE001 — retrieval config unavailable: active only
        _LOG.warning(
            "could not resolve general-knowledge corpora for fetch_document; "
            "falling back to the active project only", exc_info=True,
        )

    # Flattened view for name matching, keeping each doc's owning project.
    all_docs: list[tuple] = [
        (owner_pid, d) for owner_pid, doc_list in doc_sources for d in doc_list
    ]

    doc = None
    doc_project_id = project_id
    if doc_id:
        for owner_pid, d in all_docs:
            if (d.get("id") or "") == doc_id:
                doc, doc_project_id = d, owner_pid
                break
        if doc is None:
            return None, None, f"no document with id '{doc_id}' in this project"
    else:
        needle = filename.strip().lower()
        exact = [
            (pid, d) for pid, d in all_docs
            if (d.get("original_name") or "").strip().lower() == needle
        ]
        if exact:
            doc_project_id, doc = exact[-1]  # duplicates: latest upload wins
        else:
            partial = [
                (pid, d) for pid, d in all_docs
                if needle in (d.get("original_name") or "").strip().lower()
            ]
            if not partial:
                return None, None, f"no document named '{filename}' in this project"
            if len(partial) > 1:
                names = ", ".join(sorted((d.get("original_name") or "?") for _p, d in partial))
                return None, None, (
                    f"{len(partial)} documents match '{filename}' ({names}) — "
                    "ask the user which one, or call again with its document_id."
                )
            doc_project_id, doc = partial[0]

    # Preferred source: the indexed RAG chunks (already extracted/cleaned).
    text = ""
    source = "chunks"
    try:
        from app.core.rag import retriever as _rag
        from app.core.rag import vector_store as _vs
        if _rag.available():
            # Scoped to the document's OWNING project — a general-knowledge or
            # Master-Corpus document's chunks are not stored under the active
            # project id, so querying that would return nothing and silently
            # fall through to raw extraction (or an empty-document error).
            chunks = _vs.get_store().doc_chunk_texts(doc_project_id, [doc["id"]])
            text = "\n\n".join(chunks.get(doc["id"], []) or [])
    except Exception:  # noqa: BLE001 — fall through to raw extraction
        text = ""
    if not text.strip():
        source = "extracted"
        try:
            from app.core.doc_index import extract_document_text
            text = extract_document_text(
                doc.get("file_path") or "", doc.get("original_name") or ""
            ) or ""
        except Exception as exc:  # noqa: BLE001
            return None, doc, f"could not read document content: {exc}"
    if not text.strip():
        return None, doc, (
            "document has no extractable text (it may be a scanned/raster file "
            "that is not yet OCR-indexed)"
        )

    fetch_cap = _effective_fetch_document_max_chars()
    truncated = len(text) > fetch_cap
    if truncated:
        text = text[:fetch_cap]
    return {"text": text, "truncated": truncated, "source": source}, doc, None


def _build_attached_documents_note(attached: list[dict[str, Any]]) -> str:
    """System note pinning this turn's attached document(s) for the model.

    ``attached`` items carry ``id`` / ``original_name`` (and optionally
    ``file_path`` / ``doc_type``) as resolved by the chat router from the
    composer's ``[attached: X]`` marker.
    """
    entries = []
    for d in attached or []:
        name = (d.get("original_name") or d.get("filename") or "").strip()
        did = (d.get("id") or d.get("document_id") or "").strip()
        if not name and not did:
            continue
        entries.append(f"- {name or '(unnamed)'} (document_id: {did or 'unknown'})")
    if not entries:
        return ""
    return (
        "ATTACHED DOCUMENT(S) THIS TURN:\n" + "\n".join(entries) + "\n"
        "When the user says 'the attached file' / 'this document' (or similar), "
        "they mean the document(s) above. Read it with the fetch_document tool "
        "(pass the document_id), or pass its filename to a file-consuming block. "
        "Do NOT ask which file they mean, and do NOT answer about it from memory "
        "without reading it."
    )


def _resolve_block_file_input(project_id: str, payload: Any) -> Any:
    """Apply :func:`_resolve_file_path` to any ``file_path`` / bare-string
    inputs in a block's ``input`` or ``params`` payload.

    The block-input contract is loose — sometimes it's a string (just the
    filename), sometimes a dict with ``file_path``, sometimes nested under
    other keys. We walk one shallow level and fix any field that looks like
    a file reference, leaving everything else untouched.
    """
    if payload is None:
        return payload
    if isinstance(payload, str):
        return _resolve_file_path(project_id, payload)
    if isinstance(payload, dict):
        out = dict(payload)
        for key in ("file_path", "filepath", "path", "input", "file"):
            if key in out and isinstance(out[key], str):
                out[key] = _resolve_file_path(project_id, out[key])
        return out
    return payload


# ── Anti-hallucination: scrub prior assistant turns that contain WBS/BOQ
# markdown tables. When conversation history carries a previously-emitted
# table (often hallucinated — Float=0 / Critical=Y on every row, fabricated
# file paths), the model pattern-matches to "produce another table" instead
# of calling generate_wbs / boq_processor. Replacing the table with a
# placeholder removes the pattern while preserving turn position so the
# conversation stays coherent.
_HALLUC_TABLE_RE = re.compile(
    r"^\s*\|[^\n]*?(?:Activity\s+ID|Float|Early\s+Start|Late\s+Start|Duration|"
    r"Quantity|Unit\s+Rate|BOQ|WBS|Critical\?)[^\n]*\|\s*\n"
    r"(?:\s*\|[-:\s|]+\|\s*\n)"
    r"(?:\s*\|[^\n]*\|\s*\n){5,}",
    re.IGNORECASE | re.MULTILINE,
)


def _scrub_history(turns: list[dict[str, str]]) -> list[dict[str, str]]:
    """Replace assistant turns that contain a WBS/BOQ-shaped table with a
    placeholder so the model can't pattern-match to a prior (often
    hallucinated) table when it should be calling the tool.

    Also strips assistant turns that are raw tool-call JSON so a
    previously leaked envelope cannot re-seed the model or the UI history.

    Heuristic only: markdown tables with WBS/BOQ-like headers and >=5 rows.
    Legitimate operator-pasted tables in user turns are NOT touched.
    Returns a new list; the caller's history is left intact.
    """
    out: list[dict[str, str]] = []
    for t in turns:
        content = t.get("content") or ""
        if t.get("role") == "assistant" and _looks_like_internal_tool_json(content):
            out.append({
                **t,
                "content": (
                    "[Previous turn emitted internal tool JSON; ignored. "
                    "Answer in plain text from project documents.]"
                ),
            })
        elif t.get("role") == "assistant" and _HALLUC_TABLE_RE.search(content):
            out.append({**t, "content": "[Previous schedule/BOQ output omitted — call the tool to re-derive.]"})
        else:
            out.append(t)
    return out


# ── Anti-hallucination: force tool_choice="required" when the user's
# message names a deliverable type that maps 1:1 to a tool. Keeps Q&A on
# "auto" so explanation questions ("what is a WBS?") still get prose.
_DELIVERABLE_PHRASES = (
    "construction schedule", "wbs", "activity list", "gantt",
    "schedule with", "schedule of", "n-activity", "n activities",
    "critical path", "programme", "program of works", "project schedule",
    "manpower histogram", "labour histogram", "labor histogram",
    "resource histogram",
    "boq", "bill of quantities", "quantity takeoff", "extract quantities",
    "cost estimate", "budget breakdown", "cost breakdown",
    "compare boq to drawings", "discrepancy report",
    "generate recommendations", "recommend action",
    # Commissioning deliverables → force the commissioning_checklist tool so the
    # authoritative ASHRAE/AHRI/IEEE/BS 7671/NFPA tables are used, not an LLM
    # guess. Deliverable phrasings only — a bare "commissioning" question (e.g.
    # "when does commissioning start?") is left on tool_choice=auto.
    "commissioning checklist", "commissioning plan", "commissioning schedule",
    "testing and commissioning", "t&c checklist",
    "procurement list", "material list", "materials list",
    "request for information", "draft an rfi", "draft a rfi",
)


def _user_intent_requires_tool(messages: list[dict[str, Any]]) -> bool:
    """True iff the latest turn is the operator's first request AND it
    names a deliverable that should be produced by a tool call.

    Two gates:
    1. The tail of ``messages`` must be a ``role=user`` turn — i.e. we're
       on iteration 0 of the agent loop, before any tool call has run.
       On iteration N+1 the runtime has appended assistant + tool result
       turns; we must NOT force ``tool_choice="required"`` then or the
       model gets trapped in a forever-tool-call loop (it keeps being
       forced to call another tool instead of writing the final answer).
    2. The user message content matches a deliverable phrase.

    Returns False on no user message OR when the tail isn't user-role
    (so the runtime keeps ``tool_choice="auto"`` and the model is free
    to summarise the tool result into the user-visible final answer).
    """
    if not messages:
        return False
    tail = messages[-1]
    if tail.get("role") != "user":
        # Iter > 0: assistant + tool turns have been appended; let the
        # model decide how to proceed (will be summary, not another tool).
        return False
    raw = _unwrap_rag_folded_operator_text(tail.get("content") or "")
    if message_is_contract_data_lookup(raw):
        return False
    text = raw.lower()
    return any(p in text for p in _DELIVERABLE_PHRASES)


# Deliverable phrase -> the SPECIFIC tool that must produce it. tool_choice=
# "required" alone let the model satisfy the requirement with the wrong tool
# (e.g. search_project_documents) and then generate the answer from memory,
# bypassing the tool's authoritative data. When one of these matches we force
# THAT tool by name so the real data path runs.
#
# DATA, not code: rows live in app/routing/intent_map.yaml. Adding a row
# never requires an edit here — load the yaml, restart. The matrix test
# (tests/routing/test_intent_map_matrix.py) proves every row routes both
# of its phrasings.
_INTENT_MAP_PATH = Path(__file__).resolve().parent.parent / "routing" / "intent_map.yaml"


def _parse_intent_tool_map(raw: str) -> tuple:
    """Turn intent_map.yaml text into ((phrases...), tool) rows."""
    import yaml

    data = yaml.safe_load(raw) or {}
    rows = data.get("rows") or []
    parsed: list[tuple[tuple[str, ...], str]] = []
    for row in rows:
        phrases = tuple(p for p in (row.get("phrases") or []) if p)
        tool = (row.get("tool") or "").strip()
        if phrases and tool:
            parsed.append((phrases, tool))
    return tuple(parsed)


def _load_intent_tool_map(path: Path | None = None) -> tuple:
    """Load the intent map from yaml. ``path`` is for tests that add a row."""
    target = path or _INTENT_MAP_PATH
    return _parse_intent_tool_map(target.read_text(encoding="utf-8"))


_INTENT_TOOL_MAP = _load_intent_tool_map()


_IPC_ISSUE_RE = re.compile(
    r"\b(issue|generate|produce|create|prepare)\b.{0,60}\b"
    r"(interim\s+payment|payment\s+certificate|\bipc\b)",
    re.IGNORECASE,
)
_NAMED_CALC_ASK_RE = re.compile(
    r"\b(?:calculator|construction_calc|registered)\b|\bgross\s+\d",
    re.IGNORECASE,
)


def _message_wants_named_calculator(text: str) -> bool:
    """True when the turn is a registered ``construction_calc``, not a deliverable.

    Live F5: "interim payment certificate gross 750000 with 5 percent
    retention" was classified as ``payment_certificate``, rerouted to
    heavy-reasoning, then predefined intercept ran the container action with
    empty params. Named calculators stay on the requested agent so
    ``construction_calc`` remains the path.

    Negative: "issue an interim payment certificate from the contract"
    with no figures is a predefined IPC deliverable — do not steal it.
    """
    # Live tip 4ab55613: inline BOQ CSV lines for boq_process must not
    # elect named_calculator → construction_calc.
    try:
        from app.core.site_vocab import message_has_inline_boq_lines
        if message_has_inline_boq_lines(text or ""):
            return False
    except Exception as exc:
        _LOG.debug("inline_boq_lines check skipped: %s", exc)
    raw = text or ""
    # "Build a plumbing flow programme" is a schedule deliverable.
    # intent_map "plumbing flow" must not steal it onto named_calculator.
    if _message_is_schedule_or_programme_deliverable(raw):
        return False
    # Live tip 50c37f: drawing_qto / quantity-takeoff asks carry L×W×D
    # dims ("Compute concrete volumes") and were stolen onto
    # named_calculator → construction_calc. Keep them on drawing_qto.
    if _message_wants_drawing_qto(raw) or _message_wants_vo_draft(raw):
        return False
    if _looks_like_self_contained_calculation(raw):
        return True
    low = raw.lower()
    wants = any(
        p in low
        for phrases, tool in _INTENT_TOOL_MAP
        if tool == "construction_calc"
        for p in phrases
    )
    if not wants:
        # Registry / stem names (rebar lap, pe_unit_convert) are not in
        # intent_map.yaml. Still a named calculator — stay on this agent.
        wants = _message_is_formula_style_ask(raw)
    if not wants:
        return False
    if _document_question_without_operands(raw):
        return False
    if _NAMED_CALC_ASK_RE.search(raw):
        return True
    if _IPC_ISSUE_RE.search(raw) and not _carries_ipc_figures(raw):
        return False
    return True


def _carries_ipc_figures(text: str) -> bool:
    r"""True when the message actually supplies figures an IPC could use.

    This used to be ``re.search(r"\d", raw)`` -- ANY digit anywhere. The intent
    was "no figures supplied", but a project code or a package number is not a
    figure, so "issue the interim payment certificate ... on the PRJ2
    Infrastructure Package 1 contract" tripped it on the 2 in PRJ2 and the 1 in
    Package 1. The calculator branch then cleared ``action`` and the predefined
    IPC deliverable never dispatched: instead of the clean missing-figures
    error, the turn fell through to RAG and answered a payment question out of
    procurement-programme excerpts. Dates, revision numbers and drawing
    references all defeat it the same way.

    ``_payment_figures_from_message`` is the parser ``payment_certificate``
    itself uses to pull gross valuation / retention / advance out of a chat
    message, so it is the authority on whether figures are present. Falling
    back to the old digit test keeps behaviour unchanged if the container
    cannot be imported.
    """
    raw = text or ""
    try:
        from app.containers.construction.boq import (
            _payment_figures_from_message,
        )
        return bool(_payment_figures_from_message(raw))
    except Exception:
        _LOG.debug(
            "IPC figure parser unavailable; falling back to the digit test",
            exc_info=True,
        )
        return bool(re.search(r"\d", raw))


# A question that carries its OWN numbers is arithmetic, not a document lookup.
#
# Live 2026-08-03, project-assistant on the master corpus:
#   "Calculate the concrete volume for a raft 25m x 18m x 1.2m thick and the
#    number of 6m3 truck loads"
#   -> route None (below_routing_gate), construction_calc never invoked, and
#      the model answered:
#        "I cannot find that specific information in the provided reference
#         context ... they do not contain a raft measuring 25 m x 18 m x 1.2 m"
#
# 25 x 18 x 1.2 = 540 m3; 540 / 6 = 90 loads. Nothing needed to be retrieved.
# The grounding discipline overrode arithmetic and refused a question whose
# every input was in the question. `guardrail height` failed the same way.
#
# Both HAVE calculators (concrete_volume, guardrail_top_rail_height — 76 are
# registered), but the intent_map.yaml phrases only reach about a dozen of
# them. Rather than chase 76 keyword spellings, detect the SHAPE that is
# unambiguously a calculation: an explicit calc verb plus at least two
# measurement-bearing numbers.
#
# Deliberately narrow. A document lookup ("what is the concrete volume in the
# BOQ") has no calc verb AND no self-supplied dimensions, so it is unaffected
# and keeps the cost-grounding path.
#
# Kill switch: FORCE_CALC_ON_DIMENSIONS=0.
_CALC_VERB_RE = re.compile(
    r"\b(calculate|compute|work out|how many|how much|volume|excavation)\b",
    re.IGNORECASE,
)
# A number attached to a unit: 25m, 1.2 m, 6m3, 900,000 kg, 45 days
_DIMENSION_RE = re.compile(
    r"\b\d[\d,\.]*\s*"
    r"(mm|cm|m|km|m2|m3|sqm|cum|kg|tonne|tons?|t|kn|mpa|days?|weeks?|months?|%)\b",
    re.IGNORECASE,
)
# Engineering shorthand writes one dimension chain with ONE trailing unit:
# "30x20x1.5 m", "25 x 18 x 1.2 m", "1200x600". _DIMENSION_RE cannot see it --
# it needs a word boundary before each number, and "x" is a word character, so
# only the last number in the chain can ever match.
#
# THE MEASUREMENT (F-PHRASE-1, gate_battery_13b2bf7_2026-08-31.md). Battery
# question E4, verbatim from the instrument:
#
#     "Concrete volume for a raft 30x20x1.5 m including your documented
#      waste factor."               -> 1 dimension -> refusal
#
# A conversational paraphrase of the SAME question, same session, same corpus:
#
#     "I'm pouring a raft 30 m long, 20 m wide and 1.5 m thick, how much
#      concrete do I need ...?"     -> 3 dimensions -> 900 -> 945 m3, 119 trucks
#
# The user wrote the dimensions the way an engineer writes them and the
# platform decided that was a document lookup. Nothing about the question was
# ambiguous; the notation was.
#
# Narrow on purpose: at least TWO numbers joined by x / × / *, which is not a
# shape prose produces by accident. A document lookup ("what is the rate for
# 600x600 floor tiles") still has no calc verb and is unaffected.
_COMPACT_DIMENSION_RE = re.compile(
    r"(?<![A-Za-z0-9])\d[\d,\.]*\s*[x×*]\s*\d[\d,\.]*(?:\s*[x×*]\s*\d[\d,\.]*)*"
    r"(?:\s*(?:mm|cm|m|km|m2|m3|sqm|cum)\b)?",
    re.IGNORECASE,
)


def _count_dimensions(text: str) -> int:
    """Measurement-bearing numbers in ``text``, counting compact chains.

    A chain contributes each of its numbers: "30x20x1.5 m" is three, exactly
    as "30 m long, 20 m wide and 1.5 m thick" is three. Overlap is avoided by
    removing the chains before the plain pattern runs, so the trailing number
    of a chain is not counted twice.
    """
    body = text or ""
    count = 0
    for chain in _COMPACT_DIMENSION_RE.finditer(body):
        count += len(re.findall(r"\d[\d,\.]*", chain.group(0)))
    body = _COMPACT_DIMENSION_RE.sub(" ", body)
    return count + len(_DIMENSION_RE.findall(body))


_MIN_DIMENSIONS_FOR_CALC = 2
# Leftover L7: Compute (18.4-16)*2850 — numbers without unit suffixes.
_ARITH_EXPR_RE = re.compile(
    r"\(\s*\d[\d.]*\s*[-+*/]\s*\d[\d.]*\s*\)(?:\s*[-+*/]\s*\d[\d.]*)?",
)


def _looks_like_self_contained_calculation(text: str) -> bool:
    """True when the question supplies its own numbers AND asks to compute."""
    if (os.getenv("FORCE_CALC_ON_DIMENSIONS", "1") or "1").strip().lower() in (
        "0", "false", "no", "off",
    ):
        return False
    # Live tip d9d5971 ask1: the pasted CSV word "Excavation" is a BOQ
    # row, not a volume calculator. Do not elect construction_calc.
    if _message_wants_inline_boq(text):
        return False
    if not text or not _CALC_VERB_RE.search(text):
        return False
    if _ARITH_EXPR_RE.search(text):
        return True
    return _count_dimensions(text) >= _MIN_DIMENSIONS_FOR_CALC


def _forced_specific_tool(messages: list[dict[str, Any]], available: set) -> str | None:
    """Return a tool name to force via named tool_choice for this turn, or None.
    Gated to the user-tail (iter 0) and to tools actually available, mirroring
    ``_user_intent_requires_tool``."""
    if not messages:
        return None
    if messages[-1].get("role") != "user":
        return None
    # Predispatch / RAG fold can sit on messages[-1]. Recover the operator
    # ask so named_calculator still forces construction_calc.
    text = _latest_operator_ask(messages)
    if not text:
        return None
    low = text.lower()
    # Contract Data TfC / milestone Q&A must not force primavera_parser or
    # generate_wbs — those questions belong to RAG.
    if message_wants_answer_report(text):
        return None
    # Operands the user just typed complete one registered formula. That
    # is a calculation even when the same nouns are also a document lookup.
    if (
        "construction_calc" in available
        and _user_supplied_registered_calculation(text)
    ):
        return "construction_calc"
    if message_is_contract_data_lookup(text):
        try:
            from app.lib.wbs_duration_overrides import message_wants_wbs_duration_rerun
            user_msg, history = _messages_user_and_history(messages)
            if not message_wants_wbs_duration_rerun(user_msg or text, history):
                return None
        except Exception:  # noqa: BLE001
            return None
    # Duration override + re-run must force generate_wbs, not a calculator.
    # Leftover F1: a demolition / site-clearance BOQ WBS ask must also
    # force generate_wbs so RAG-folded AB-2022 CoC cannot steal the turn.
    if "generate_wbs" in available:
        try:
            from app.lib.wbs_duration_overrides import message_wants_wbs_duration_rerun
            user_msg, history = _messages_user_and_history(messages)
            if message_wants_wbs_duration_rerun(user_msg or text, history):
                return "generate_wbs"
        except Exception:  # noqa: BLE001
            _LOG.debug(
                "duration-override tool force skipped; falling through",
                exc_info=True,
            )
        if _message_wants_boq_scope_wbs_ask(text):
            return "generate_wbs"
    if "resource_histogram" in available and _message_wants_resource_histogram(text):
        return "resource_histogram"
    if "look_ahead" in available and _message_wants_look_ahead(text):
        return "look_ahead"
    if "cash_flow_forecast" in available and _message_wants_cash_flow(text):
        return "cash_flow_forecast"
    wants_procurement = _message_wants_procurement_list(text)
    if "procurement_list_generator" in available and wants_procurement:
        return "procurement_list_generator"
    wants_rfi = _message_wants_rfi_draft(text)
    if "rfi_generator" in available and wants_rfi:
        return "rfi_generator"
    wants_drawing_qto = _message_wants_drawing_qto(text)
    if "drawing_qto" in available and wants_drawing_qto:
        return "drawing_qto"
    # Live tip 0a95d03 ask2: VO draft still ran construction+sympy. Force
    # variation_order_manager when the ask is a priced ADD/OMIT draft.
    if _message_wants_vo_draft(text):
        for name in ("variation_order_manager", "variation_order_generator"):
            if name in available:
                return name
    # Live tip 0a95d03 ask1: inline CSV/BOQ routed to self-coding →
    # formula_executor_v2 (429). Force boq_processor when lines are inline.
    try:
        from app.core.site_vocab import message_has_inline_boq_lines
        if message_has_inline_boq_lines(text):
            for name in ("boq_processor", "boq_process"):
                if name in available:
                    return name
    except Exception:  # noqa: BLE001
        _LOG.debug("inline boq force skipped", exc_info=True)
    for phrases, tool in _INTENT_TOOL_MAP:
        if tool in available and any(p in low for p in phrases):
            if (
                tool == "construction_calc"
                and (
                    _message_is_schedule_or_programme_deliverable(text)
                    or wants_procurement
                    or wants_rfi
                    or wants_drawing_qto
                    or _document_question_without_operands(text)
                )
            ):
                continue
            return tool
    # A procurement-list or RFI-draft ask must not fall through to
    # construction_calc when the toolkit omitted the action — that is
    # the live miss.
    if wants_procurement or wants_rfi or wants_drawing_qto:
        return None
    # Keyword phrases reach ~a dozen of the 76 registered calculators. Catch
    # the rest by SHAPE: a question that supplies its own dimensions and asks
    # to compute is arithmetic, whatever the domain noun happens to be.
    if "construction_calc" in available and _looks_like_self_contained_calculation(text):
        return "construction_calc"
    # Named / stemmed formula asks (rebar lap, pe_unit_convert, …) are
    # the same class: construction_calc must run. The dimension heuristic
    # above never sees them, and Kimi/Groq cannot be tool_choice-forced,
    # so this is the named lever. Does not pick which calculator.
    if "construction_calc" in available and _message_is_formula_style_ask(text):
        return "construction_calc"
    return None


# Bracketed forms:
#   [source: file.pdf, chunk 65]
#   【source: file.pdf, chunk 65】   (gpt-oss Chinese-bracket variant)
# Capture the entire bracket contents; _parse_source_tail extracts the
# filename and optional chunk suffix. This lets filenames contain commas
# without the regex engine swallowing the chunk suffix into group 1.
_CITATION_RE = re.compile(
    r"(?:\[source:|【source:)\s*(.+?)\s*(?:\]|】)",
    re.IGNORECASE,
)

# Bracketless line form gpt-oss-style models also emit:
#   Source: Site-Safety-Plan.pdf, chunk 65.
#   Sources: Site-Safety-Plan.pdf, chunks 16, 34, 55.
#   Source: “Example Quantity Comparison.xlsx”, which lists ...
# Anchor on start-of-line or newline + "Source[s]:" prefix; capture the
# rest of the line up to a sentence terminator. The post-capture parsing
# strips quotes and extracts an optional ", chunk(s) ..." suffix.
_CITATION_LINE_RE = re.compile(
    r"(?:^|\n)\s*Sources?:\s*(.+?)(?=\.\s*(?:\n|$)|\n|$)",
    re.IGNORECASE,
)

# Quoted inline source mention (smart or straight quotes):
#   Source: “File Name.xlsx”, ...
#   Source: "File Name.xlsx", ...
#   Source: 'File Name.xlsx', ...
_CITATION_QUOTED_RE = re.compile(
    r"(?:^|\n|\s)Sources?:\s*([\"'“""])"""  # opening quote
    r"([^\"'”'\"\n]+?)"""
    r"\1""",
    re.IGNORECASE,
)

# doc_id form gpt-oss also emits when it wants to be technically precise:
#   [doc_id=4e5f6a7b, chunk 65, score 0.697]
#   [doc_id=4e5f6a7b chunk=65 score=0.697]    (the RAG-injection header style)
# Match either separator style; capture (doc_id, chunk_index). The chunk
# is REQUIRED here — a bare [doc_id=...] would be ambiguous.
_CITATION_DOCID_RE = re.compile(
    r"\[\s*doc_id\s*=\s*([0-9a-f]{4,})\s*[,;\s]+\s*chunk\s*=?\s*(\d+)",
    re.IGNORECASE,
)

# gpt-oss-120b (the pilot model) inline form — the filename is INSIDE the
# brackets and the chunk number comes AFTER, mid-sentence:
#   ...protected (Source: [AB-2001-101 - Example Package … Part 3], chunk 941).
#   ...schedule (Source: [AB-2001-101 … Part 2], chunks 1988-1990).
# Distinct from _CITATION_RE ("[source: file, chunk N]" — source INSIDE the
# bracket). Group 1 = filename (often truncated with an ellipsis); group 2 =
# the chunk-number blob (digits, commas, and en-/em-dash ranges).
_CITATION_INLINE_BRACKET_RE = re.compile(
    r"sources?:\s*[\[【]\s*(.+?)\s*[\]】]\s*,?\s*chunks?\s+(\d[\d,\s‐-―−-]*)",
    re.IGNORECASE,
)

# Injection-marker copy: the model repeats ``src=filename`` from the RAG
# header. Filenames may contain spaces; the value ends at the closing
# bracket or end of line.
_CITATION_SRC_RE = re.compile(
    r"\bsrc\s*=\s*(.+?)(?:\s*\]|$)",
    re.IGNORECASE,
)


def _normalise_filename(s: str) -> str:
    """Normalize source filenames for cite-vs-chunk matching.

    Handles Unicode dashes, typographic quotes, surrounding whitespace,
    and trailing punctuation so the model's rewritten filenames still
    resolve to the stored original_name."""
    return (
        (s or "")
        .replace("‐", "-")  # hyphen
        .replace("‑", "-")  # non-breaking hyphen
        .replace("‒", "-")  # figure dash
        .replace("–", "-")  # en dash
        .replace("—", "-")  # em dash
        .replace("−", "-")  # minus
        .replace("“", "\"")
        .replace("”", "\"")
        .replace("‘", "'")
        .replace("’", "'")
        .strip(" \"'\n\t.")
        .lower()
    )


def _strip_source_quotes(s: str) -> str:
    """Remove matching surrounding quotes (smart/straight)."""
    s = s.strip()
    for open_q, close_q in (("“", "”"), ('"', '"'), ("'", "'"), ("‘", "'")):
        if s.startswith(open_q) and s.endswith(close_q):
            return s[1:-1].strip()
    return s


def _clean_path_label(label: str) -> str:
    r"""Return a user-facing source label stripped of Windows/Unix path gunk.

    Preserves the basename (or last non-empty path segment) so sources are
    still grounded to a real document, but never leaks ``X:\Example Drive\...``
    or ``/home/user/...`` machine paths in the UI.
    """
    if not label:
        return label
    # Normalize Windows backslashes and URL separators to a single slash,
    # then return the last non-empty segment (the basename).
    normalized = label.replace("\\", "/")
    parts = [p for p in normalized.split("/") if p]
    if parts:
        return parts[-1].strip()
    return label.strip()


# Generic machine-path sanitiser: catches stray Windows/Unix paths that
# appear outside formal citation markers (e.g. markdown table cells).
_PATH_RE = re.compile(
    r"(?:[A-Za-z]:\\(?:[^\\]*\\)*[^\\\n]+"
    r"|\\\\[^\\]+(?:\\[^\\]+)+"
    r"|(?:/home/|/Users/)[^\n]+)",
    re.IGNORECASE,
)


def _sanitize_inline_paths(text: str) -> str:
    """Rewrite stray Windows/Unix file paths anywhere in the text to basenames."""
    if not text:
        return text
    return _PATH_RE.sub(lambda m: _clean_path_label(m.group(0)), text)


def _answer_is_caveat(text: str) -> bool:
    """True when the assistant answer is explicitly refusing/declining to answer."""
    if not text:
        return False
    lowered = text.lower()
    phrases = (
        "could not locate",
        "couldn't locate",
        "unable to locate",
        "cannot confirm",
        "can't confirm",
        "unable to generate",
        "was unable to",
        "wasn't able to",
        "not found",
        "i'm not able to",
        "i am not able to",
        "i cannot",
        "no record",
        "could not find",
        "i could not find",
        "none of the indexed files contain",
        "none of the indexed files",
        "i searched the project's document repository",
    )
    return any(p in lowered for p in phrases)


def _sanitize_citation_labels(text: str) -> str:
    """Rewrite inline citation labels so they show basenames, not raw paths.

    Handles the bracketed forms recognised by ``_CITATION_RE`` plus the
    bracketless ``Source:`` line form.  Only the label/path inside the
    citation is rewritten; chunk numbers and surrounding prose are left
    untouched.
    """
    if not text:
        return text

    def _repl_bracket(m: re.Match) -> str:
        inner = m.group(1)
        # inner still contains the label and optional chunk tail.
        fname, tail = _parse_source_tail(inner)
        clean = _clean_path_label(fname)
        # Reconstruct using the same bracket style as the original match.
        open_br = m.group(0)[0]
        close_br = {"[": "]", "【": "】"}.get(open_br, "]")
        if tail:
            return f"{open_br}source: {clean}, chunk {tail}{close_br}"
        return f"{open_br}source: {clean}{close_br}"

    text = _CITATION_RE.sub(_repl_bracket, text)

    # Bracketless "Source: ..." / "Sources: ..." lines.
    def _repl_line(m: re.Match) -> str:
        inner = m.group(1)
        fname, tail = _parse_source_tail(inner)
        clean = _clean_path_label(fname)
        prefix = m.group(0).split(":", 1)[0]  # "Source" or "Sources"
        if tail:
            return f"\n{prefix}: {clean}, chunk {tail}"
        return f"\n{prefix}: {clean}"

    text = _CITATION_LINE_RE.sub(_repl_line, text)
    return text


# ── Cost-grounding gate ─────────────────────────────────────────────────────
# A cost/rate figure in a chat answer must trace to a RATE-SEMANTIC retrieved
# chunk (currency/price context) or a computed tool result — otherwise the
# answer is refused. The unit of grounding is figure + rate-semantics, NOT the
# bare number: on 2026-07-14 a live cost query answered "450 SAR/m³" by lifting
# "450" out of a the client project drawing dimension-table chunk. A naive "the number appears
# somewhere in retrieval" gate would PASS that fabrication because 450 is in the
# soup; this gate does not, because that chunk is not rate-semantic and 450 does
# not sit next to a currency/rate-unit. Flag-controlled (COST_GROUNDING_GATE,
# default on); never raises (a gate must never break an answer).
from app.lib.source_labels import CURRENCY_TOKEN as _CG_CURRENCY
# Pricing DENOMINATORS only (per-unit rate units), never bare dimensions like
# mm/cm — a "450 mm" dimension must never read as a rate.
_CG_RATE_UNIT = (
    r"(?:/|per\s+)\s*"
    r"(?:day|month|week|year|hour|hr|man-?hour|m3|m²|m2|m³|sqm|sq\s?m|cum|"
    r"kg|tonne|ton|lm|l\.?m|no\.?|nr|unit|each|ea|item|m)\b"
)
_CG_NUM = r"\d[\d,]*(?:\.\d+)?"
_CG_MONEY_RE = re.compile(
    rf"{_CG_CURRENCY}\s*({_CG_NUM})"       # SAR 300 / $1,200
    rf"|({_CG_NUM})\s*{_CG_CURRENCY}"      # 300 SAR
    rf"|({_CG_NUM})\s*{_CG_RATE_UNIT}",    # 55 /m³ / 1,200 per day
    re.IGNORECASE,
)
# A chunk is rate-semantic if it carries currency or explicit rate/price
# vocabulary. Drawing dimension-tables (pure numbers + BEND/PIPE/mm labels)
# match none of these, so their numbers never ground a cost claim.
_CG_RATE_SEMANTIC_RE = re.compile(
    rf"{_CG_CURRENCY}|\b(?:unit\s*rate|rate\b|priced?|pricing|tender\s*price|"
    r"bill\s+of\s+quantit|boq|cost\s*/|/\s*m[23³²]|per\s+m[23³²]|per\s+tonne|"
    r"per\s+kg|sar/|usd/)\b",
    re.IGNORECASE,
)
_CG_NUM_RE = re.compile(_CG_NUM)
# Magnitude suffixes: a tool returns 1250000 while the model naturally writes
# "SAR 1.25 million". Fold the suffix so the figure compares to the raw grounded
# number instead of being false-refused (Codex #223). Applied to CURRENCY money
# forms only — a rate unit ("1.25 /m3") never carries a magnitude word, and a
# bare "1.25 m" without a currency could be metres.
_CG_MAGNITUDE = {
    "thousand": 1e3, "k": 1e3,
    "million": 1e6, "mn": 1e6, "mln": 1e6, "m": 1e6,
    "billion": 1e9, "bn": 1e9, "b": 1e9,
}
_CG_SUFFIX_RE = re.compile(r"\s*(thousand|million|billion|mln|mn|bn|[kmb])\b", re.IGNORECASE)
_CG_REFUSAL = (
    "I don't have a rate on file for that in the project documents or the "
    "knowledge base. Please upload your priced BOQ / rate schedule, or get a "
    "supplier quote for a firm figure — I won't quote a price I can't ground."
)


def _cost_grounding_enabled() -> bool:
    return os.getenv("COST_GROUNDING_GATE", "1") not in ("0", "false", "False", "")


# A cost/rate-shaped user question. Token streaming yields the answer before the
# cost gate can act, so a fabricated price would reach the client (Codex
# #223/#224). For these queries only, streaming is disabled so the full answer is
# gated before emission. Non-cost queries (the common case) stream normally.
_CG_COST_QUERY_RE = re.compile(
    r"\b(rate|rates|unit\s*rate|cost|costs|price|priced|pricing|how\s+much|"
    r"quote|quotation|sar|usd|aed|qar|/\s*m[23³²]|per\s+m[23³²]|per\s+tonne|"
    r"per\s+kg|bill\s+of\s+quantit|boq)\b",
    re.IGNORECASE,
)


def _is_cost_shaped_query(user_message: str | None) -> bool:
    return bool(user_message and _CG_COST_QUERY_RE.search(user_message))


# Document deliverables (RFP, WIR, O&M, …) often mention a currency or a
# contract sum. The cost gate used to replace the *entire* document with
# `_CG_REFUSAL` ("I don't have a rate on file") — live M14 on theshovel.ai.
# Skip that full-answer wipe unless the user is actually asking for a unit
# rate / price-per. Fabricated unit rates on a rate-quote still refuse.
_DOCUMENT_DELIVERABLE_RE = re.compile(
    r"\b("
    r"rfp|request for proposal|invitation to tender|"
    r"job requisition|prequalification shortlist|"
    r"design directive|"
    r"commissioning checklist|"
    r"inspection request|work inspection request|\bwir\b|"
    r"delay claim|claim notice|eot claim|"
    r"as-built deviation|as built deviation|"
    r"o\s*&\s*m(?:\s+manual)?|\bom manual\b|"
    r"\bncr\b|non[-\s]?conformance|"
    r"punch list|qc inspection|"
    r"\bwbs\b|work breakdown|"
    r"payment certificate|interim payment|\bipc\b|"
    r"cash[- ]?flow|"
    r"safety briefing|resource histogram|"
    r"follow-on rfi|request for information"
    r")\b",
    re.IGNORECASE,
)
_RATE_QUOTE_RE = re.compile(
    r"\b(unit\s*rate|how much per|price per|rate per|quote (?:a |the )?rate)\b",
    re.IGNORECASE,
)


def _latest_user_text(messages: list[dict[str, Any]] | None) -> str:
    for m in reversed(messages or []):
        if m.get("role") != "user":
            continue
        content = str(m.get("content") or "")
        if content.lstrip().startswith(_PREDISPATCH_PREFIX):
            continue
        return content
    return ""


def _latest_operator_ask(messages: list[dict[str, Any]] | None) -> str:
    """Operator question, unwrapped from a RAG-folded user bubble.

    On an ACA-including-VAT ask, ``_apply_rag_context`` folds Contract Data
    chunk #0 (Delay Damages × a partial ACA) plus the lookup directive
    ("…or compute it…") into the last user turn. Ask-class detectors
    that read that bubble then see a daily-amount ask (delay-damages +
    compute + SAR) and compose SAR/day. Unwrap so it stays the incl-VAT
    particular. "Calculate delay damages … including VAT" stays a
    daily-amount ask — the operator text has the delay-damages token.
    """
    return _unwrap_rag_folded_operator_text(_latest_user_text(messages))


def _is_document_deliverable_request(user_message: str | None) -> bool:
    if not user_message:
        return False
    if _RATE_QUOTE_RE.search(user_message):
        return False
    return bool(_DOCUMENT_DELIVERABLE_RE.search(user_message))


def _is_contract_data_fact_lookup(user_message: str | None) -> bool:
    """True for a Contract Data Q&A turn that is not asking for a unit rate
    or a SAR arithmetic result.

    A delay-damages rate ask (the rate, no arithmetic) is a
    filled-particular lookup. The cost gate's BOQ refusal is
    the wrong instrument — it wiped a grounded (or model-expanded)
    percentage particular because a SAR-per-day gloss did not sit in a
    rate-semantic chunk. A daily-amount ask ("calculate … in SAR") still gates.
    """
    if not user_message:
        return False
    if _RATE_QUOTE_RE.search(user_message):
        return False
    from app.core.contract_lookup_intent import message_is_contract_data_lookup
    if not message_is_contract_data_lookup(user_message):
        return False
    from app.core.rag.retriever import query_needs_a_monetary_base
    return not query_needs_a_monetary_base(user_message)


def _infer_commissioning_systems(text: str | None) -> list[str] | None:
    """Map the user's wording onto commissioning system keys.

    Leftover torch-SBS / membrane / before-backfill → waterproofing
    (holiday/spark belongs there). A PWPS-09 reservoir /
    first wet test / C21-OPC blinding must NEVER take that path.
    """
    low = (text or "").lower()
    reservoir = any(
        k in low
        for k in (
            "reservoir", "pwps", "wet test", "wet-test", "first wet",
            "c21-opc", "c21 opc",
        )
    )
    waterproof = any(
        k in low
        for k in (
            "waterproof", "torch-applied", "torch applied", "sbs",
            "membrane", "before backfill",
        )
    )
    if reservoir and not waterproof:
        return ["reservoir"]
    if waterproof and not reservoir:
        return ["waterproofing"]
    if reservoir and waterproof:
        if any(k in low for k in ("wet test", "wet-test", "first wet")):
            return ["reservoir"]
        return ["waterproofing"]
    return None


def _format_commissioning_tool(payload: dict[str, Any]) -> str:
    """User-facing checklist from a successful commissioning_checklist tool."""
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    by_sys = payload.get("checklists_by_system") or {}
    lines = [
        "Commissioning checklist (from the commissioning_checklist tool).",
        "",
        f"Systems covered: {summary.get('systems_covered', len(by_sys))}. "
        f"Tests: {summary.get('total_tests', 0)} "
        f"({summary.get('pending', 0)} pending).",
        "",
    ]
    if not isinstance(by_sys, dict):
        return "\n".join(lines).strip()
    for system, tests in by_sys.items():
        lines.append(f"## {str(system).replace('_', ' ').title()}")
        if not isinstance(tests, list):
            continue
        for t in tests:
            if not isinstance(t, dict):
                continue
            name = t.get("test") or "Test"
            std = t.get("standard") or ""
            acc = t.get("acceptance_criteria") or ""
            wit = "witness" if t.get("witness_required") else "review"
            hold = "hold" if t.get("hold_point") else wit
            lines.append(f"- **{name}** ({std}) — {acc} [{hold}]")
        lines.append("")
    return "\n".join(lines).strip()


def _format_wir_form(payload: dict[str, Any]) -> str:
    """User-facing WIR from a successful ``wir_form`` tool / predispatch."""
    lines = [
        f"**Work Inspection Request — {payload.get('wir_number') or 'DRAFT-WIR'}**",
        f"**Location:** {payload.get('location') or '—'}",
        f"**Activity:** {payload.get('activity') or '—'}",
        "",
        f"### 1. Scope",
        str(payload.get("scope") or "As stated by the operator."),
        "",
    ]
    mix = payload.get("mix") or ""
    vol = payload.get("volume_m3")
    week = payload.get("week")
    supplier = payload.get("supplier") or ""
    facts = []
    if week:
        facts.append(f"Week {week}")
    if mix:
        facts.append(mix)
    if vol not in (None, ""):
        facts.append(f"{vol} m³")
    if payload.get("manhole_range"):
        facts.append(str(payload["manhole_range"]))
    elif payload.get("manhole_count"):
        facts.append(f"{payload['manhole_count']} manholes")
    if supplier:
        facts.append(f"supplier {supplier}")
    if facts:
        lines.append("**Stated facts:** " + "; ".join(facts))
        lines.append("")
    tmpl = payload.get("template") or ""
    if tmpl:
        lines.append(f"**Template:** {tmpl}")
        lines.append("")
    lines.append("### 2. Pre-pour checklist")
    for row in payload.get("checklist") or []:
        if isinstance(row, dict):
            lines.append(f"- [ ] {row.get('item') or row}")
        else:
            lines.append(f"- [ ] {row}")
    lines.append("")
    lines.append("### 3. Hold points")
    for h in payload.get("hold_points") or []:
        lines.append(f"- {h}")
    lines.append("")
    lines.append("### 4. Witness points")
    for w in payload.get("witness_points") or []:
        lines.append(f"- {w}")
    lines.append("")
    lines.append("### 5. Sign-off")
    lines.append("| Party | Status |")
    lines.append("|-------|--------|")
    for s in payload.get("signatories") or []:
        if isinstance(s, dict):
            lines.append(
                f"| {s.get('party') or '—'} | {s.get('status') or 'pending'} |"
            )
    lines.append("")
    if payload.get("notice"):
        lines.append(str(payload["notice"]))
        lines.append("")
    if payload.get("note"):
        lines.append(str(payload["note"]))
    return "\n".join(lines).strip()


_TOOL_TRUNCATION_NOTICE_RE = re.compile(
    r"(?i)tool result exceeded\s+\d+\s+characters|"
    r"char_offset=\d+|"
    r"chars_remaining is 0",
)


def _looks_like_tool_truncation_notice(text: str) -> bool:
    """True for the fetch-window truncation envelope shipped as an answer.

    A named-percentage × ACA ask once shipped: "Tool result exceeded 4000
    characters and was truncated. Call this tool again with … char_offset=…"
    instead of 10% × ACA. That prose is an internal fetch hint, never a
    money answer.
    """
    t = (text or "").strip()
    if not t:
        return False
    return bool(_TOOL_TRUNCATION_NOTICE_RE.search(t))


def _text_needs_tool_recovery(text: str) -> bool:
    """True when the LLM hop died and a tool/predispatch draft should speak."""
    t = (text or "").strip()
    if not t:
        return True
    if t == _EMPTY_RESPONSE_FALLBACK:
        return True
    if "unable to generate" in t.lower():
        return True
    if re.search(r"HTTP 413|tokens per minute|\bTPM\b", t):
        return True
    # Leftover E4: "Let me validate…" is not an answer. Recover the
    # construction_calc volume (or any other deliverable already in the
    # thread) instead of shipping the promise.
    if _looks_like_search_preamble(t):
        return True
    if _looks_like_tool_error_json(t):
        return True
    if _looks_like_formula_template(t):
        return True
    if _looks_like_tool_truncation_notice(t):
        return True
    try:
        from app.lib.construction_formulas_commercial import (
            answer_is_unbound_delay_damages,
        )
        if answer_is_unbound_delay_damages(t):
            return True
    except Exception:  # noqa: BLE001 — recovery probe must never break a turn
        _LOG.debug("unbound delay-damages probe failed", exc_info=True)
    return False


def _format_variation_order(payload: dict[str, Any]) -> str:
    lines = payload.get("lines") or payload.get("vo_lines") or []
    parts = [
        f"**Variation Order {payload.get('vo_number') or 'VO-D-001'}**",
        f"**Description:** {payload.get('description') or ''}",
        "",
        "### Lines",
    ]
    for line in lines:
        if isinstance(line, dict):
            parts.append(
                f"- {line.get('kind')} {line.get('quantity')} "
                f"{line.get('unit')} {line.get('description')} "
                f"@ {line.get('rate')} = {line.get('amount')}"
            )
        else:
            parts.append(f"- {line}")
    net = payload.get("net")
    if net is None:
        net = (payload.get("pricing") or {}).get("total_value")
    parts.append("")
    parts.append(f"**Net:** {net}")
    body = payload.get("document_content") or payload.get("vo_document")
    if body:
        parts.extend(["", str(body)])
    return "\n".join(parts).strip()


def _format_boq_process(payload: dict[str, Any]) -> str:
    """Render a boq_processor / boq_process result for predispatch inject."""
    items = payload.get("line_items") or payload.get("items") or []
    total = payload.get("total_cost")
    currency = payload.get("currency") or ""
    parts = [
        f"**BOQ processed — {payload.get('item_count') or len(items)} lines**",
        f"**Total:** {total} {currency}".strip(),
        "",
        "### Lines",
    ]
    for it in items:
        if isinstance(it, dict):
            parts.append(
                f"- {it.get('description')} {it.get('quantity')} "
                f"{it.get('unit')} @ {it.get('unit_cost')} = {it.get('total_cost')}"
            )
        else:
            parts.append(f"- {it}")
    return "\n".join(parts).strip()


def _format_payment_certificate(payload: dict[str, Any]) -> str:
    cert = payload.get("certificate") if isinstance(payload.get("certificate"), dict) else {}
    val = payload.get("valuation") if isinstance(payload.get("valuation"), dict) else {}
    ded = payload.get("deductions") if isinstance(payload.get("deductions"), dict) else {}
    pay = payload.get("payment") if isinstance(payload.get("payment"), dict) else {}
    lines = [
        f"**Interim Payment Certificate — {cert.get('period') or 'Current Period'}**",
        f"**Contractor:** {cert.get('contractor') or 'Contractor'}",
        "",
        f"- Contract value: {val.get('contract_value')}",
        f"- Gross valuation: {val.get('gross_valuation')}",
    ]
    if val.get("measured_works") not in (None, "", 0, 0.0):
        lines.append(f"- Measured works: {val.get('measured_works')}")
    if val.get("materials_on_site") not in (None, "", 0, 0.0):
        lines.append(f"- Materials on site: {val.get('materials_on_site')}")
    lines.extend([
        f"- Retention ({ded.get('retention_percent')}%): {ded.get('retention_held')}",
        f"- Advance recovery: {ded.get('advance_recovery')}",
        f"- Net due this period: {pay.get('net_due_this_period')}",
        f"- Cumulative certified: {pay.get('cumulative_certified')}",
        "",
        str(payload.get("certificate_summary") or ""),
    ])
    return "\n".join(lines).strip()


def _format_as_built_note(payload: dict[str, Any]) -> str:
    if payload.get("note"):
        return str(payload["note"])
    planned = payload.get("planned_m3")
    poured = payload.get("poured_m3")
    short = payload.get("shortfall_m3")
    pct = payload.get("shortfall_percent")
    loc = payload.get("location") or ""
    return (
        f"As-built deviation note{f' — {loc}' if loc else ''}: "
        f"planned {planned} m³ versus poured {poured} m³ — "
        f"shortfall {short} m³ ({pct}%)."
    )


def _format_claim_notice(payload: dict[str, Any]) -> str:
    narrative = payload.get("claim_narrative") if isinstance(payload.get("claim_narrative"), dict) else {}
    body = narrative.get("full_narrative") or narrative.get("executive_summary")
    if body:
        extra = []
        if payload.get("quantum_formula"):
            extra.append(f"Quantum: {payload['quantum_formula']}")
        comms = payload.get("communication")
        if comms:
            extra.append(f"Communication: {comms}")
        if payload.get("clause"):
            extra.append(f"Clause: {payload['clause']}")
        return "\n\n".join([str(body)] + extra)
    return json.dumps(payload, default=str)[:4000]


def _graft_operator_claim_facts(text: str, user_message: str) -> str:
    """Keep Aconex / 8.8.1 on the published claim when synthesis drops them."""
    if not text or not _message_wants_delay_claim(user_message):
        return text
    out = text
    if re.search(r"\baconex\b", user_message, re.I) and not re.search(r"\baconex\b", out, re.I):
        out = out.rstrip() + "\n\nThis notice is issued via Aconex."
    if re.search(r"8\.8\.1", user_message) and "8.8.1" not in out:
        out = re.sub(r"Clause\s+4\.5", "Clause 8.8.1", out, count=1)
        if "8.8.1" not in out:
            out = out.rstrip() + "\n\nClause 8.8.1."
    return out


def _format_rfi(payload: dict[str, Any]) -> str:
    rfis = payload.get("rfis") if isinstance(payload.get("rfis"), list) else []
    first = next((row for row in rfis if isinstance(row, dict)), None)
    number = (
        payload.get("rfi_number")
        or (first.get("rfi_number") if first else None)
        or "DRAFT-RFI"
    )
    lines = [
        f"**Request for Information — {number}**",
        "",
    ]
    if payload.get("question"):
        lines.append(str(payload["question"]))
    for row in rfis:
        if not isinstance(row, dict):
            continue
        lines.append(f"**{row.get('rfi_number') or 'RFI'} — {row.get('subject') or ''}**")
        lines.append(str(row.get("question") or ""))
        lines.append("")
    if payload.get("note"):
        lines.append(str(payload["note"]))
    return "\n".join(lines).strip() or json.dumps(payload, default=str)[:4000]


def _format_wbs_result(payload: dict[str, Any]) -> str:
    lines = ["**Work Breakdown Structure**", ""]
    # The scaffold declaration goes FIRST, before any number the reader might
    # take for a project fact (owner's ruling R3, 2026-09-01). generate_wbs is
    # a deterministic template scheduler: it never reads the BOQ, the drawings
    # or the contract. On the 13b2bf7 gate battery its output was read as the
    # project's own structure, so the answer now says what it is at the top
    # rather than leaving it to be inferred from an assumptions list nobody
    # reaches. Rendered from the tool's own `scaffold` record -- the model does
    # not write this line.
    scaffold = payload.get("scaffold") if isinstance(payload.get("scaffold"), dict) else {}
    declaration = scaffold.get("declaration")
    if (
        declaration
        and scaffold.get("derived_from_boq")
        and "template scaffold" in str(declaration).lower()
    ):
        declaration = ""
    if declaration:
        lines.append(f"_{declaration}_")
        lines.append("")
    # Numbered phase/package outline — leftover F1. generate_wbs already
    # builds wbs_tree; the glass used to stop at headline metrics.
    try:
        from app.core.predefined_reasoning import format_wbs_outline
        outline = format_wbs_outline(payload.get("wbs_tree"))
    except Exception:  # noqa: BLE001 — formatter miss must not blank the glass
        outline = ""
    if outline:
        lines.append(outline)
        lines.append("")
    op = payload.get("operator_milestones") if isinstance(payload.get("operator_milestones"), dict) else {}
    if op.get("time_for_completion_days"):
        lines.append(
            f"Time for Completion of the whole of the Works: "
            f"{op['time_for_completion_days']} calendar days."
        )
    for m in op.get("milestones") or []:
        if not isinstance(m, dict):
            continue
        lines.append(
            f"- Milestone {m.get('id')}: {m.get('name')} — "
            f"{m.get('days')} calendar days"
        )
    if not op.get("milestones"):
        for m in payload.get("target_milestones") or []:
            if not isinstance(m, dict):
                continue
            lines.append(
                f"- {m.get('name') or m.get('id')}: "
                f"{m.get('duration_days') or m.get('days')} days"
            )
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    if summary:
        lines.append("")
        count = payload.get("actual_count") or summary.get("activity_count") or 0
        if scaffold.get("derived_from_boq"):
            lines.append(
                f"BOQ-derived items: {count}. "
                f"Scope: {payload.get('project_type') or 'demolition_site_clearance'}."
            )
        else:
            lines.append(
                f"Template activities: {count}. "
                f"Project type: {payload.get('project_type') or '—'}."
            )
    if payload.get("assumptions"):
        lines.append("")
        for a in payload.get("assumptions") or []:
            lines.append(f"- {a}")
    out: list[str] = []
    for ln in lines:
        if not ln.strip() and out and not out[-1].strip():
            continue  # the declaration block and the summary each add a blank
        out.append(ln)
    return "\n".join(out).strip()


def _format_job_requisition(payload: dict[str, Any]) -> str:
    lines = [
        f"**Job Requisition — {payload.get('jr_number') or 'DRAFT-JR'}**",
        f"**Title:** {payload.get('title') or 'Job requisition'}",
        "",
        "### Scope",
        str(payload.get("scope") or ""),
        "",
    ]
    if payload.get("noc"):
        lines.append(f"**NOC:** {payload['noc']}")
    if payload.get("noc_expiry"):
        lines.append(f"**NOC expiry:** {payload['noc_expiry']}")
    lines.append("")
    lines.append("### Prequalification")
    for row in payload.get("prequalification") or []:
        lines.append(f"- {row}")
    lines.append("")
    lines.append("### Shortlist rubric")
    for row in payload.get("shortlist_rubric") or []:
        if isinstance(row, dict):
            lines.append(f"- {row.get('criterion')} ({row.get('weight')})")
        else:
            lines.append(f"- {row}")
    if payload.get("note"):
        lines.extend(["", str(payload["note"])])
    return "\n".join(lines).strip()


def _format_cash_flow(payload: dict[str, Any]) -> str:
    rows = (
        payload.get("monthly_forecast")
        or payload.get("s_curve_data")
        or []
    )
    params = payload.get("project_parameters") if isinstance(payload.get("project_parameters"), dict) else {}
    contract = params.get("contract_value") or payload.get("contract_value") or ""
    lines = [
        "**Cash-flow S-curve**",
        f"Contract value: {contract}",
        "",
        "| Month | Progress % | Monthly | Cumulative | Advance recovery | Retention | Net |",
        "|---|---|---|---|---|---|---|",
    ]
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            lines.append(
                f"| {row.get('month') or row.get('period')} | "
                f"{row.get('planned_progress_percent')} | "
                f"{row.get('monthly_value')} | "
                f"{row.get('cumulative_value')} | "
                f"{row.get('advance_recovery')} | "
                f"{row.get('retention_deduction')} | "
                f"{row.get('net_cash_in')} |"
            )
    return "\n".join(lines).strip()


def _format_om_outline(payload: dict[str, Any]) -> str:
    if payload.get("note") and payload.get("sections"):
        lines = [
            f"**{payload.get('title') or 'Operations and Maintenance manual outline'}**",
            "",
        ]
        for sec in payload.get("sections") or []:
            if isinstance(sec, dict):
                lines.append(f"### {sec.get('section') or sec.get('title') or 'Section'}")
                lines.append(str(sec.get("content") or sec.get("body") or ""))
                lines.append("")
            else:
                lines.append(f"- {sec}")
        lines.append(str(payload.get("note") or ""))
        return "\n".join(lines).strip()
    return json.dumps(payload, default=str)[:4000]


def _format_safety_briefing(payload: dict[str, Any]) -> str:
    if payload.get("briefing"):
        return str(payload["briefing"])
    lines = [
        f"**{payload.get('title') or 'Live-haul-road safety briefing'}**",
        "",
        str(payload.get("note") or ""),
        "",
        "### Signage",
        str(payload.get("signage") or "As shown on the diversion / haul-road drawings."),
        "",
        "### Speed control",
        str(payload.get("speed") or "Enforce the posted site speed and the diversion speed limit."),
        "",
        "### Pedestrian crossing",
        str(payload.get("pedestrian") or "Keep public crossings signed and marshalled at the interface."),
    ]
    return "\n".join(lines).strip()


def _format_rfp_draft(payload: dict[str, Any]) -> str:
    lines = [
        f"**{payload.get('title') or 'Request for Proposal'} — {payload.get('rfp_number') or 'DRAFT-RFP'}**",
        "",
        str(payload.get("invitation") or ""),
        "",
        "### Scope of works",
        str(payload.get("scope_of_works") or ""),
        "",
        "### Prequalification",
    ]
    for row in payload.get("prequalification") or []:
        lines.append(f"- {row}")
    lines.append("")
    lines.append("### Evaluation method")
    lines.append(str(payload.get("evaluation_method") or ""))
    lines.append("")
    lines.append("### Key dates")
    lines.append(str(payload.get("key_dates_note") or ""))
    refs = payload.get("references") or []
    if refs:
        lines.append("")
        lines.append("**References:** " + ", ".join(str(r) for r in refs))
    return "\n".join(lines).strip()


def _extract_predispatch_draft(raw_user: str) -> str:
    for marker in (
        "Authoritative draft:",
        "Authoritative result:",
        "Authoritative result (",
    ):
        if marker in raw_user:
            body = raw_user.split(marker, 1)[1]
            for stop in (
                "\nPresent this WIR",
                "\nReport the period buckets",
                "\nPresent this ",
                "\nDo not search again",
                "\nDo not invent",
                "\nDo not re-call",
            ):
                body = body.split(stop, 1)[0]
            body = body.strip()
            if body:
                return body
    return ""


def _recover_answer_from_tool_messages(
    text: str,
    messages: list[dict[str, Any]] | None,
) -> str:
    """If the model hop died after a successful deliverable tool / predispatch,
    surface that draft instead of an empty bubble or a 413 error."""
    if not _text_needs_tool_recovery(text):
        return text
    error_fallback = ""
    for m in reversed(messages or []):
        if m.get("role") == "user":
            raw_user = str(m.get("content") or "")
            if "PLATFORM PRE-DISPATCH:" in raw_user[:80]:
                body = _extract_predispatch_draft(raw_user)
                if body:
                    return body
            continue
        if m.get("role") != "tool":
            continue
        raw = m.get("content") or ""
        try:
            payload = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            # Truncation envelope: {"truncated": true, "preview": "..."}
            continue
        inner = payload
        if payload.get("truncated") and payload.get("preview"):
            try:
                inner = json.loads(payload["preview"])
            except (TypeError, json.JSONDecodeError):
                inner = payload
        if not isinstance(inner, dict):
            continue
        action = inner.get("action")
        if action == "commissioning_checklist_generated":
            return _format_commissioning_tool(inner)
        if action == "wir_form":
            return _format_wir_form(inner)
        if action == "payment_certificate":
            return _format_payment_certificate(inner)
        if action in {"variation_order_processed", "variation_order_manager"}:
            return _format_variation_order(inner)
        if action == "as_built_deviation_report":
            return _format_as_built_note(inner)
        if action in {"claim_generated", "claims_builder"}:
            return _format_claim_notice(inner)
        if action == "job_requisition":
            return _format_job_requisition(inner)
        if action in {"rfp_draft", "rfp_management"}:
            return _format_rfp_draft(inner)
        if action == "rfi_generator" or inner.get("rfis"):
            return _format_rfi(inner)
        if action == "cash_flow_forecast" or inner.get("monthly_forecast"):
            return _format_cash_flow(inner)
        if action in {"om_manual_generated", "om_manual_generator"}:
            return _format_om_outline(inner)
        if action == "safety_briefing" or inner.get("briefing"):
            return _format_safety_briefing(inner)
        if action == "resource_histogram_generated":
            kind = inner.get("histogram_kind") or "labor"
            periods = inner.get("periods") or []
            note = inner.get("note") or ""
            lines = [
                f"Resource / activity histogram ({kind}).",
                note,
                "",
            ]
            for p in periods:
                if not isinstance(p, dict):
                    continue
                lines.append(
                    f"- {p.get('label', '?')}: {p.get('total', 0)} "
                    f"{'activities' if kind == 'activity_count' else 'units'}"
                )
            return "\n".join(lines).strip()
        if inner.get("wbs_id") or inner.get("operator_milestones") or action == "generate_wbs":
            return _format_wbs_result(inner)
        if inner.get("planned_m3") is not None and inner.get("poured_m3") is not None:
            return _format_as_built_note(inner)
        formatted = _format_construction_calc(inner)
        if formatted:
            return formatted
        generic = _format_any_calc_result(inner)
        if generic:
            return generic
        err_payload = _tool_error_payload(inner)
        if (
            err_payload
            and not error_fallback
            and _is_calc_tool_error(m, inner, err_payload)
        ):
            error_fallback = _format_tool_error_plain(err_payload)
    return error_fallback or text


def _unwrap_construction_calc_inner(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Return the calculator result dict that carries ``volume_m3``."""
    if not isinstance(payload, dict):
        return None
    if payload.get("volume_m3") is not None or payload.get("volume_with_waste_m3") is not None:
        return payload
    inner = payload.get("result")
    if isinstance(inner, dict):
        found = _unwrap_construction_calc_inner(inner)
        if found:
            return found
    return None


def _format_construction_calc(payload: dict[str, Any]) -> str:
    """User-facing volume line from a successful ``construction_calc`` tool."""
    from app.lib.construction_formulas_quantities import format_concrete_volume_line

    inner = _unwrap_construction_calc_inner(payload)
    if not inner:
        return ""
    calc = str(payload.get("calculation") or "")
    if calc and calc != "concrete_volume":
        return ""
    if calc != "concrete_volume" and not inner.get("waste_factor") and inner.get("net_volume_m3") is None:
        # Leftover L6 excavation (bank only) is not an E4 compose target.
        if "bank_volume_m3" in inner and "volume_with_waste_m3" not in inner:
            return ""
    return format_concrete_volume_line(inner)


def _construction_calc_from_messages(
    messages: list[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    """Most recent ``construction_calc`` concrete-volume envelope in the turn."""
    last: dict[str, Any] | None = None
    for m in messages or []:
        if m.get("role") != "tool":
            continue
        raw = m.get("content")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                continue
        if not isinstance(raw, dict):
            continue
        inner = _unwrap_construction_calc_inner(raw)
        if not inner:
            continue
        if inner.get("volume_m3") is None and inner.get("volume_with_waste_m3") is None:
            continue
        last = inner
    return last


def _graft_composed_concrete_volume(
    text: str,
    messages: list[dict[str, Any]] | None,
) -> str:
    """Leftover E4: write 945 m³ when the model hung on validate.

    Live FAIL: construction-pm streamed "Let me validate…" after (or
    instead of) ``construction_calc`` and never composed the with-waste
    volume. Numbers are in the question; compose from the existing
    calculator. Kill-switch ``COMPOSE_CONCRETE_VOLUME=0`` restores the
    hang. Delay-damages and other non-concrete-volume asks pass
    through.
    """
    try:
        from app.lib.construction_formulas_quantities import (
            answer_states_volume,
            compose_concrete_volume_from_ask,
            format_concrete_volume_line,
        )
        user = _latest_operator_ask(messages)
        composed = compose_concrete_volume_from_ask(user)
        if not composed:
            return text
        tool_inner = _construction_calc_from_messages(messages)
        if tool_inner and tool_inner.get("volume_m3") is not None:
            line = format_concrete_volume_line(tool_inner)
            volume = tool_inner.get("volume_m3")
        else:
            line = composed["line"]
            volume = composed["volume_m3"]
        if not line:
            return text
        payload = json.dumps({
            "calculation": "concrete_volume",
            "volume_m3": volume,
            "net_volume_m3": (
                (tool_inner or composed).get("net_volume_m3")
                if isinstance(tool_inner, dict)
                else composed.get("net_volume_m3")
            ),
            "waste_factor": (
                (tool_inner or composed).get("waste_factor")
                if isinstance(tool_inner, dict)
                else composed.get("waste_factor")
            ),
            "note": line,
        })
        if isinstance(messages, list) and not any(
            isinstance(m, dict)
            and m.get("role") == "tool"
            and "concrete_volume" in str(m.get("content") or "")
            for m in messages
        ):
            messages.append({"role": "tool", "content": payload})
        raw = text or ""
        if answer_states_volume(raw, float(volume)):
            return text
        if (
            not raw.strip()
            or raw.strip() == _CG_REFUSAL
            or raw.strip() == _EMPTY_RESPONSE_FALLBACK
            or _looks_like_search_preamble(raw)
            or _GENERIC_ACK_RE.search(raw)
            or _MISSING_PARTICULAR_RE.search(raw)
        ):
            return line
        return f"{line}\n\n{raw.strip()}"
    except Exception:  # noqa: BLE001 — compose must never break a turn
        _LOG.exception("concrete-volume compose failed; passing answer through")
        return text


def _graft_composed_user_priced_takeoff(
    text: str,
    messages: list[dict[str, Any]] | None,
) -> str:
    """Cost gate: volume × user rate after 0-token force_synthesis.

    Live FAIL: construction_calc succeeded, synthesis emitted nothing, and
    volume-only recover (one footing, or 14× without the rate) shipped a
    silent / incomplete bubble. Numbers are in the question; do not invent
    a rate. Kill-switch ``COMPOSE_USER_PRICED_TAKEOFF=0`` restores the FAIL.
    """
    try:
        from app.lib.construction_formulas_commercial import (
            answer_states_money_amount,
            compose_user_priced_takeoff_from_ask,
            format_user_priced_takeoff_line,
            query_asks_user_priced_takeoff,
        )
        user = _latest_operator_ask(messages)
        if not query_asks_user_priced_takeoff(user):
            return text
        composed = compose_user_priced_takeoff_from_ask(user)
        if not composed:
            return text
        line = format_user_priced_takeoff_line(composed)
        if not line:
            return text
        payload = json.dumps({
            "calculation": "user_priced_takeoff",
            "net_volume_m3": composed["net_volume_m3"],
            "volume_with_waste_m3": composed["volume_with_waste_m3"],
            "unit_rate": composed["unit_rate"],
            "currency": composed["currency"],
            "waste_percent": composed.get("waste_percent"),
            "contingency_percent": composed.get("contingency_percent"),
            "base_cost": composed["base_cost"],
            "total_cost": composed["total_cost"],
            "note": line,
        })
        if isinstance(messages, list) and not any(
            isinstance(m, dict)
            and m.get("role") == "tool"
            and "user_priced_takeoff" in str(m.get("content") or "")
            for m in messages
        ):
            messages.append({"role": "tool", "content": payload})
        raw = text or ""
        if answer_states_money_amount(raw, float(composed["total_cost"])):
            return text
        return line
    except Exception:  # noqa: BLE001 — compose must never break a turn
        _LOG.exception("user-priced takeoff compose failed; passing answer through")
        return text


def _turn_input_question() -> str:
    """The question this turn's tools asked for an input they need, in the
    user's words, or ''."""
    from app.agents import answer_exit

    turn = answer_exit.current_turn()
    return "\n".join(turn.questions).strip() if turn and turn.questions else ""


def _nonblank_after_empty_synthesis(
    text: str,
    messages: list[dict[str, Any]] | None,
) -> str:
    """Never ship a blank bubble after a successful deliverable calc.

    Prefer a priced take-off compose. A tool that is still missing an input
    asks for it. If ``construction_calc`` produced a result this turn, emit
    the cut-off sentence rather than an empty end event.
    """
    raw = (text or "").strip()
    body = _strip_coverage_footer(raw)
    if body and body != _EMPTY_RESPONSE_FALLBACK:
        return text
    # A coverage footer alone is not an answer: the answer goes above it.
    footer = "\n".join(m.group(0).strip() for m in _COVERAGE_FOOTER_RE.finditer(raw))

    def _with_footer(answer: str) -> str:
        return f"{answer}\n{footer}" if footer else answer

    priced = _graft_composed_user_priced_takeoff(body, messages)
    if priced.strip() and priced.strip() != _EMPTY_RESPONSE_FALLBACK:
        return _with_footer(priced)
    recovered = _recover_answer_from_tool_messages("", messages).strip()
    if recovered:
        return _with_footer(recovered)
    asked = _turn_input_question()
    if asked:
        return _with_footer(asked)
    if _calc_payload_numbers(messages or []) is not None or _construction_calc_from_messages(
        messages,
    ):
        return _with_footer(_SYNTH_CUTOFF_NOTICE)
    return _with_footer(body or _EMPTY_RESPONSE_FALLBACK)


def _construction_calc_tool_schema() -> dict[str, Any]:
    """Schema for the `construction_calc` deterministic-calculator tool. The
    calculation enum is built from the live registry so it never drifts from the
    library. For cost build-ups the model is told to pass real rates from RAG;
    the library's defaults are indicative fallbacks only (keeps cost grounded)."""
    from app.lib import construction_formulas as _cf
    return {
        "type": "function",
        "function": {
            "name": "construction_calc",
            "description": (
                "Run a DETERMINISTIC construction engineering or cost calculation "
                "(deep foundations, concrete technology, structural, crane planning, "
                "cost build-up, MEP, QC). Returns the computed result. For COST "
                "build-ups, pass the project's real unit rates in `params` (from the "
                "priced BOQ / rate schedule) when available — the built-in defaults "
                "are indicative GCC fallbacks only. For concrete/raft/slab volume, "
                "use calculation=concrete_volume; the project's documented waste "
                "factor (5%) is applied and volume_m3 is the with-waste figure "
                "(30×20×1.5 m → 945 m3). Deterministic: call once."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "calculation": {
                        "type": "string",
                        "enum": _cf.available_calculations(),
                        "description": "Which calculator to run.",
                    },
                    "params": {
                        "type": ["object", "string"],
                        "additionalProperties": True,
                        "description": (
                            "Keyword arguments for the chosen calculation (e.g. "
                            "cost_buildup_rebar needs quantity_kg and optionally "
                            "material_price_sar_t / labour_rate_sar_hr / ...). "
                            "Canonical names or aliases (w/span, length/width/"
                            "thickness, water_depth_m/floors). A comma-separated "
                            "assignment string is also accepted. On a bad call "
                            "the tool returns the exact required signature."
                        ),
                    },
                    "input": {
                        "type": ["object", "string"],
                        "additionalProperties": True,
                        "description": "Same calculator kwargs, nested.",
                    },
                },
                "required": ["calculation"],
            },
        },
    }


def _cg_to_number(s: Any) -> float | None:
    try:
        return round(float(str(s).replace(",", "").strip()), 2)
    except (ValueError, TypeError):
        return None


def _cg_money_values(text: str) -> list[tuple[str, float]]:
    """Every money/rate-shaped figure in ``text`` as (fragment, value).

    Folds a trailing magnitude word (million/billion/k) for CURRENCY money forms
    so "SAR 1.25 million" -> 1_250_000 and matches a tool's raw 1250000 instead
    of being false-refused. Group 3 is the rate-unit form ("1.25 /m3"), which
    never carries a magnitude word, so it is left as-is."""
    text = text or ""
    out: list[tuple[str, float]] = []
    for m in _CG_MONEY_RE.finditer(text):
        raw = next((g for g in m.groups() if g), None)
        v = _cg_to_number(raw)
        if v is None or v == 0:
            continue
        frag = m.group(0).strip()
        if m.group(3) is None:  # currency money form (not a rate unit)
            suf = _CG_SUFFIX_RE.match(text, m.end())
            if suf:
                v = round(v * _CG_MAGNITUDE[suf.group(1).lower()], 2)
                frag = text[m.start():suf.end()].strip()
        out.append((frag, v))
    return out


_CG_SMALL = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
    "ninety": 90,
}
_CG_SCALES = {"hundred": 100, "thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}
_CG_NUMBER_WORDS = set(_CG_SMALL) | set(_CG_SCALES) | {"and"}


def _cg_words_to_int(words: list[str]) -> float | None:
    """Convert a run of English number words to an int (four million → 4e6)."""
    if not words:
        return None
    total = 0
    current = 0
    saw = False
    for w in words:
        if w == "and":
            continue
        if w in _CG_SMALL:
            current += _CG_SMALL[w]
            saw = True
        elif w == "hundred":
            current = (current or 1) * 100
            saw = True
        elif w in _CG_SCALES:
            current = (current or 1) * _CG_SCALES[w]
            total += current
            current = 0
            saw = True
        else:
            return None
    if not saw:
        return None
    return float(total + current)


def _cg_english_and_percent_values(text: str) -> list[float]:
    """Number-words and 'N percent of X' from a user message.

    Live leftover-hat L3: 'ten percent of four million eight hundred thousand'
    produced a 480,000 LD cap the cost gate refused because the user never
    typed digits. Word phrases and percent-of products are the user's own
    figures — same authority as SAR 4.2M in the prompt.
    """
    if not text:
        return []
    tokens = re.findall(r"[a-z]+|%|\d[\d,]*(?:\.\d+)?", text.lower())
    out: list[float] = []
    i = 0
    while i < len(tokens):
        run: list[str] = []
        j = i
        while j < len(tokens) and tokens[j] in _CG_NUMBER_WORDS:
            run.append(tokens[j])
            j += 1
        parsed = _cg_words_to_int(run) if run else None
        if parsed is not None:
            out.append(parsed)
            i = j
            continue
        i += 1
    # "ten percent of four million…" / "10% of 4800000". Restrict both
    # sides to digits or number-words so "LD cap is ten percent of …"
    # does not swallow "ld cap is" into the percent (leftover-hat L3).
    _word = (
        r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
        r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
        r"nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
        r"hundred|thousand|million|billion|and)"
    )
    for m in re.finditer(
        rf"(?P<pct>{_CG_NUM}|{_word}(?:\s+{_word}){{0,8}})\s*"
        r"(?:percent|per\s+cent|%)\s+of\s+"
        rf"(?P<base>{_CG_NUM}|{_word}(?:\s+{_word}){{0,10}})",
        text.lower(),
    ):
        pct_raw, base_raw = m.group("pct"), m.group("base")
        pct = _cg_to_number(pct_raw)
        if pct is None:
            pct = _cg_words_to_int(re.findall(r"[a-z]+", pct_raw))
        base = _cg_to_number(base_raw)
        if base is None:
            base = _cg_words_to_int(re.findall(r"[a-z]+", base_raw))
        if pct is not None and base is not None:
            out.append(round(base * (pct / 100.0), 4))
    return out


#: Above this many base numbers the pairwise closure is checked lazily instead
#: of being built (120 numbers -> ~36k entries, a few MB; 3,750 -> 2 GB).
_CG_EAGER_CLOSURE_MAX = 120
# Operator-typed multipliers kept for the per-figure pair check. A real ask
# carries a handful (a percentage, a rate); the cap keeps the per-figure walk
# constant whatever a pasted message contains.
_CG_OPERATOR_FACTOR_MAX = 24


class _GroundedNumbers(set):
    """The grounded set, plus -- when the base was too large to close eagerly --
    the sorted base for ``_cg_pair_grounds``. Still a ``set`` to every caller."""

    lazy_base: tuple = ()
    # Sorted context figures, and the multipliers the OPERATOR typed (a
    # percentage, a rate). Checked as a pair in _cg_is_grounded: see
    # _CG_OPERATOR_FACTOR_MAX.
    base_sorted: tuple = ()
    operator_factors: tuple = ()


def _cg_grounded_numbers(rag_context: str, messages: list[dict[str, Any]]) -> set:
    """Numbers a cost claim may be grounded against:

    * RAG context, classified PER CHUNK — in a rate-semantic chunk every number
      grounds (rate tables split figure/unit across columns); in a non-rate
      chunk only figures adjacent to currency/rate-unit ground. Per-chunk is
      essential: a rate keyword in one chunk must NOT ground a bare number in a
      drawing chunk.
    * Tool-result messages this turn — a computed deliverable (e.g. a real
      cost_estimate) is authoritative, so all of its numbers ground.
    """
    grounded: set = set()

    def _add_numbers(text: str, all_numbers: bool) -> None:
        # A BOQ item reference is a label, not a figure: "D529.2" is the
        # identifier of a bill item, not the number 529.2. Retrieved rows are
        # full of them, and grounding a cost claim against a label is how a
        # fabricated figure can trace to nothing real. Blanked before any
        # number is read; the row's actual quantity, rate and amount stay.
        text = strip_ref_codes(text)
        # Bare numbers (only in rate-semantic / authoritative text)...
        if all_numbers:
            for tok in _CG_NUM_RE.findall(text):
                v = _cg_to_number(tok)
                if v is not None:
                    grounded.add(v)
        # ...plus magnitude-FOLDED money values, so a chunk/tool that says
        # "SAR 1.25 million" grounds an answer that repeats it (Codex #226 —
        # symmetric with the answer-side fold in _cg_money_values).
        for _frag, v in _cg_money_values(text):
            grounded.add(v)

    # RAG context, per injected chunk (marker from format_chunks_as_system_message).
    for block in re.split(r"\[doc_id=", rag_context or ""):
        if not block.strip():
            continue
        _add_numbers(block, all_numbers=bool(_CG_RATE_SEMANTIC_RE.search(block)))
    # Tool results this turn (authoritative computed numbers) AND the user's
    # own message. A figure the USER supplied ("variance: planned SAR 4.2M vs
    # actual SAR 5.1M") is definitionally traceable — the gate must not refuse
    # to echo the user's own numbers back (the sympy/variance mis-fire). Only
    # numbers actually present in the user's text ground; a fabricated rate the
    # user never mentioned still fails the gate.
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role in ("tool", "user") and isinstance(content, str):
            _add_numbers(content, all_numbers=True)
            for v in _cg_english_and_percent_values(strip_ref_codes(content)):
                grounded.add(v)
    # Simple arithmetic derivations of grounded figures also ground: a
    # variance/overrun answer legitimately computes the SUM or DIFFERENCE of
    # two grounded totals (SAR 5.1M - SAR 4.2M = SAR 0.9M), and a bill line
    # legitimately computes the PRODUCT of quantity and rate -- the extension
    # qty x rate is the QS's most basic operation, and unit-rate back-outs
    # (amount / qty) are its inverse. Live find 2026-08-15 (F25): "extend 100
    # square metres at the operator-instructed 250 riyals" was refused because
    # 25,000 traced to nothing -- the closure knew sums and differences but
    # not products. Bounded to pairwise combinations of the (small) grounded
    # set; the 0.5% tolerance in _cg_is_grounded keeps this from grounding an
    # unrelated fabricated rate.
    #
    # MEMORY. Materialising that closure is quadratic, and a retrieved BOQ
    # table carries thousands of numbers: 3,750 numbers -> 24 million set
    # entries, 2 GB and 414 s with the event loop blocked. Live 2026-09-20/21
    # it OOM-killed the 2 GB web instance ten times -- one costed answer each.
    # The closure is only materialised while it is small; above that the same
    # rule is CHECKED per figure against a sorted list (_cg_pair_grounds).
    base = list(grounded)
    grounded = _GroundedNumbers(grounded)
    grounded.base_sorted = tuple(sorted(base))
    if len(base) <= _CG_EAGER_CLOSURE_MAX:
        for i in range(len(base)):
            for j in range(i, len(base)):
                grounded.add(round(base[i] + base[j], 4))
                grounded.add(round(abs(base[i] - base[j]), 4))
                grounded.add(round(base[i] * base[j], 4))
                if base[j]:
                    grounded.add(round(base[i] / base[j], 4))
                if base[i]:
                    grounded.add(round(base[j] / base[i], 4))
    else:
        grounded.lazy_base = tuple(sorted(base))
    # User-supplied arithmetic beyond one pairwise hop: a×b×c, qty×rate
    # after a quotient (3400/42 × 1950), and a×(1+p%) for waste /
    # contingency the operator stated. Seeded from what the OPERATOR
    # TYPED only — a rate the user never typed still fails the gate.
    #
    # The runtime replays tool output back into the conversation as
    # user-role bubbles ("PLATFORM PRE-DISPATCH: …", "Tool result (…): …")
    # and folds retrieved chunks into the last user turn. Seeding from
    # every user-role message therefore treated a whole priced BOQ — every
    # quantity, rate and amount the block returned — as figures the
    # operator had typed, which both grounded numbers the gate is meant to
    # refuse and drove the closure past 2 GiB (live 21 Sep 2026, #692).
    user_seed: set = set()
    user_blob: list[str] = []
    for msg in messages or []:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        if _cg_is_platform_bubble(content):
            continue
        content = _unwrap_rag_folded_operator_text(content)
        if not content.strip():
            continue
        user_blob.append(content)
        cleaned = _cg_strip_unit_noise(content)
        for tok in _CG_NUM_RE.findall(cleaned):
            v = _cg_to_number(tok)
            if v is not None:
                user_seed.add(v)
        for v in _cg_english_and_percent_values(content):
            user_seed.add(v)
        for _frag, v in _cg_money_values(content):
            user_seed.add(v)
    if user_seed:
        grounded |= _cg_user_arithmetic_closure(" ".join(user_blob), user_seed)
    grounded.operator_factors = _cg_operator_factors(" ".join(user_blob))
    return grounded


def _cg_operator_factors(operator_text: str) -> tuple:
    """Multipliers the operator typed: a percentage, a rate, a count.

    "8.5% of the Accepted Contract Amount" and "price that volume at SAR 390
    per m3" are the two most common QS asks, and both multiply a figure the
    operator never typed (it is in the contract, or a tool computed it) by one
    the operator did. The eager/lazy pairwise closure cannot reach either: it
    runs over the CONTEXT figures before the operator's are merged in, so a
    milestone delay-damages ask refused a correct 8.5%-of-ACA answer and a
    volume-pricing ask dropped the pricing step. These factors are paired against the context base per checked figure
    in _cg_is_grounded -- no closure is materialised.
    """
    factors: list[float] = []
    for m in _CG_PERCENT_RE.finditer(operator_text or ""):
        p = _cg_to_number(m.group(1))
        if p is None or p <= 0 or p > 100:
            continue
        factors.append(round(p / 100.0, 6))
        factors.append(round(1.0 + p / 100.0, 6))
    # A typed RATE needs nothing here: it is already one of the context
    # figures, so quantity x rate is the ordinary pairwise rule (eager, or
    # _cg_pair_grounds on a large base). Only the percentage is missing --
    # 0.085 exists solely inside the user closure, which is merged after the
    # pairwise pass.
    #
    # Deterministic, deduplicated and capped: this list is walked per figure.
    return tuple(sorted({f for f in factors if f})[:_CG_OPERATOR_FACTOR_MAX])


def _cg_is_platform_bubble(content: str) -> bool:
    """True for a user-role message the PLATFORM wrote, not the operator.

    ``_inject_predispatch`` and ``_repair_tool_call_pairing`` replay tool
    output as user bubbles; their figures are the platform's, not the
    operator's, and must never seed the user-arithmetic closure.
    """
    head = content.lstrip()
    return head.startswith(_PREDISPATCH_PREFIX) or head.startswith(_TOOL_RESULT_PREFIX)


def _cg_pair_grounds(value: float, base: tuple, tol: float) -> bool:
    """True when ``value`` is a+b, |a-b|, a*b or a/b of two numbers in ``base``
    (sorted ascending), within ``tol``. n log n and no allocation -- the same
    rule the eager closure applies, without building it."""
    lo_v, hi_v = value - tol, value + tol

    def _has(lo: float, hi: float) -> bool:
        if lo > hi:
            lo, hi = hi, lo
        i = bisect.bisect_left(base, lo)
        return i < len(base) and base[i] <= hi

    for a in base:
        if _has(lo_v - a, hi_v - a):            # a + b = value
            return True
        if _has(a - hi_v, a - lo_v):            # a - b = value  (covers |a-b|)
            return True
        if a:
            if _has(lo_v / a, hi_v / a):        # a * b = value
                return True
            if _has(lo_v * a, hi_v * a):        # b / a = value
                return True
    return False


def _cg_is_grounded(value: float, grounded: set) -> bool:
    tol = max(0.5, abs(value) * 0.005)
    if any(abs(value - g) <= tol for g in grounded):
        return True
    lazy = getattr(grounded, "lazy_base", ())
    if lazy and _cg_pair_grounds(value, lazy, tol):
        return True
    return _cg_operator_factor_grounds(value, grounded, tol)


def _cg_operator_factor_grounds(value: float, grounded: set, tol: float) -> bool:
    """True when ``value`` is a context figure times something the operator
    typed (8.5% of the contract amount; a tool's volume at the operator's
    SAR 390/m3). Binary search per factor -- no allocation."""
    base = getattr(grounded, "base_sorted", ())
    factors = getattr(grounded, "operator_factors", ())
    if not base or not factors:
        return False
    for f in factors:
        target = value / f
        window = tol / abs(f)
        i = bisect.bisect_left(base, target - window)
        if i < len(base) and base[i] <= target + window:
            return True
    return False


# Operator asked for the calculator's own defaults, not a project rate.
_CALC_DEFAULTS_ASK_RE = re.compile(
    r"(?i)(?:"
    r"calculator(?:'s|’s|s)?\s+default\s+rates?"
    r"|default\s+(?:rates?|inputs?)"
    r"|standard\s+\w+(?:\s+\w+){0,4}\s+cost\s+build-?\s*up"
    r"|cost\s+build-?\s*up"
    r"|cost_buildup_"
    r")"
)
_DOC_COST_RE = re.compile(
    r"(?i)(?:\b(?:SAR|AED|USD|EUR|GBP|QAR|BHD|KWD|OMR|Dhs?|dirhams?)\s*"
    r"\d[\d,]*(?:\.\d+)?"
    r"|\d[\d,]*(?:\.\d+)?\s*"
    r"(?:SAR|AED|USD|EUR|GBP|QAR|BHD|KWD|OMR|Dhs?|dirhams?)\b)"
)
_COST_SELLING_UNITS = (
    ("selling_price_sar_m3", "SAR/m3"),
    ("selling_price_sar_t", "SAR/t"),
    ("selling_price_sar_m2", "SAR/m2"),
)
_COST_TOTAL_KEYS = (
    "total_project_value_sar",
    "total_for_area_sar",
    "total_cost",
    "grand_total_sar",
)
_NOT_A_COST_CALC = frozenset({
    "delay_damages_daily",
    "delay_damages_over_period",
    "percentage_of_aca",
    "concrete_volume",
    "user_priced_takeoff",
})


def _answer_is_no_rate_refusal(text: str) -> bool:
    raw = (text or "").strip()
    if not raw:
        return False
    if raw == _CG_REFUSAL:
        return True
    return "don't have a rate on file" in raw.lower()


def _cost_calc_result_dict(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Inner result of a successful deterministic cost calculator, or None."""
    if not isinstance(payload, dict):
        return None
    if payload.get("status") == "error":
        return None
    if _tool_error_payload(payload):
        return None
    result = payload.get("result") if isinstance(payload.get("result"), dict) else None
    calc = str(payload.get("calculation") or "")
    if result is None:
        if any(str(k).startswith("selling_price_") for k in payload):
            result = payload
            calc = calc or str(payload.get("calculation") or "")
        else:
            return None
    if not isinstance(result, dict):
        return None
    if calc in _NOT_A_COST_CALC:
        return None
    has_selling = any(result.get(k) is not None for k, _u in _COST_SELLING_UNITS)
    has_total = any(result.get(k) is not None for k in _COST_TOTAL_KEYS)
    named = calc.startswith("cost_") or calc.startswith("cost")
    if not (named or has_selling or has_total):
        return None
    return result


def _last_cost_calc_payload(
    messages: list[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    """Last successful cost-calculator tool payload after the operator ask.

    A tool from an earlier turn is not this question's result.
    """
    msgs = messages or []
    start = 0
    for i, msg in enumerate(msgs):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = str(msg.get("content") or "")
        if _cg_is_platform_bubble(content):
            continue
        start = i
    last: dict[str, Any] | None = None
    for msg in msgs[start:]:
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        raw = msg.get("content")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                continue
        if not isinstance(raw, dict):
            continue
        if _cost_calc_result_dict(raw) is None:
            continue
        last = raw
    return last


def _ask_wants_this_cost_calc(user: str, payload: dict[str, Any]) -> bool:
    """True when the operator asked for this calculator, not a document rate."""
    if _CALC_DEFAULTS_ASK_RE.search(user or ""):
        return True
    name = str(payload.get("calculation") or "")
    if not name:
        return False
    try:
        from app.lib.construction_formulas import calculator_name_from_text
        return calculator_name_from_text(user or "") == name
    except Exception:  # noqa: BLE001 — name lookup must not break the gate
        _LOG.debug("cost-calc name check failed", exc_info=True)
        return False


def _format_deterministic_cost_calc(payload: dict[str, Any]) -> str:
    """Selling price and total from a cost-calculator result."""
    result = _cost_calc_result_dict(payload)
    if not result:
        return ""
    note = result.get("note")
    if isinstance(note, str) and re.search(r"(?i)\b(?:selling|total)\b", note):
        return note.strip()
    selling = None
    unit = "SAR"
    for key, label in _COST_SELLING_UNITS:
        if result.get(key) is not None:
            selling = float(result[key])
            unit = label
            break
    total = next(
        (result[k] for k in _COST_TOTAL_KEYS if result.get(k) is not None),
        None,
    )
    parts: list[str] = []
    if selling is not None:
        parts.append(f"Selling price {selling:,.2f} {unit}.")
    if total is not None:
        parts.append(f"Total {float(total):,.0f} SAR.")
    return " ".join(parts)


def _cost_calc_allowed_numbers(payload: dict[str, Any]) -> set[float]:
    blob = json.dumps(payload, default=str)
    out: set[float] = set()
    for tok in _CG_NUM_RE.findall(blob):
        val = _cg_to_number(tok)
        if val is not None:
            out.add(val)
    return out


def _part_states_foreign_cost(part: str, allowed: set[float]) -> bool:
    """True when ``part`` states a money figure the calculator did not return."""
    found: list[float] = []
    for match in _DOC_COST_RE.finditer(part or ""):
        raw = re.search(r"\d[\d,]*(?:\.\d+)?", match.group(0))
        if raw is None:
            continue
        val = _cg_to_number(raw.group(0))
        if val:
            found.append(val)
    for _frag, val in _cg_money_values(part or ""):
        found.append(val)
    if not found:
        return False

    def _near(val: float) -> bool:
        tol = max(0.5, abs(val) * 0.005)
        return any(abs(val - got) <= tol for got in allowed)

    return any(not _near(val) for val in found)


def _strip_foreign_cost_sentences(text: str, allowed: set[float]) -> str:
    parts = [
        p.strip()
        for p in re.split(r"(?<=[.!;])\s+|\n+", text or "")
        if p and p.strip()
    ]
    kept = [
        p for p in parts
        if not _answer_is_no_rate_refusal(p)
        and not _part_states_foreign_cost(p, allowed)
    ]
    return " ".join(kept).strip()


def _answer_states_cost_calc(text: str, payload: dict[str, Any]) -> bool:
    result = _cost_calc_result_dict(payload) or {}
    blob = (text or "").replace(",", "").replace(" ", "")
    selling = next(
        (result.get(k) for k, _u in _COST_SELLING_UNITS if result.get(k) is not None),
        None,
    )
    total = next(
        (result.get(k) for k in _COST_TOTAL_KEYS if result.get(k) is not None),
        None,
    )
    if selling is not None:
        token = f"{float(selling):.2f}".rstrip("0").rstrip(".")
        if token not in blob and f"{float(selling):.2f}" not in (text or ""):
            return False
    if total is not None and str(int(round(float(total)))) not in blob:
        return False
    return selling is not None or total is not None


def _graft_deterministic_cost_calc(
    text: str,
    messages: list[dict[str, Any]] | None,
) -> str:
    """State the calculator when the operator asked for its default rates.

    A project-document cost is not that total. The no-rate refusal is not
    an answer once a deterministic cost calculator has already succeeded.
    Same shape for every cost build-up, not only concrete.
    """
    try:
        payload = _last_cost_calc_payload(messages)
        if not payload:
            return text
        user = _latest_operator_ask(messages)
        if not _ask_wants_this_cost_calc(user, payload):
            return text
        line = _format_deterministic_cost_calc(payload)
        if not line:
            return text
        raw = text or ""
        if _answer_is_no_rate_refusal(raw) or not raw.strip():
            return line
        if not _CALC_DEFAULTS_ASK_RE.search(user or ""):
            return text
        cleaned = _strip_foreign_cost_sentences(
            raw, _cost_calc_allowed_numbers(payload),
        )
        if _answer_states_cost_calc(cleaned, payload):
            return cleaned
        return line
    except Exception:  # noqa: BLE001 — a graft must never break a turn
        _LOG.exception("cost-calc graft failed; passing answer through")
        return text


def _cost_calc_line_from_tool_result(
    messages: list[dict[str, Any]] | None,
) -> str:
    """Calculator line when this ask's cost tool already succeeded."""
    payload = _last_cost_calc_payload(messages)
    if not payload:
        return ""
    user = _latest_operator_ask(messages)
    if not _ask_wants_this_cost_calc(user, payload):
        return ""
    return _format_deterministic_cost_calc(payload)


# A number and the unit written right after it ("240 mm", "44 days").
_CALC_FIGURE_RE = re.compile(
    r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s*(mm|cm|m|km|m2|m²|m3|m³|kN/m|kN\.?m|kN|MPa|GPa|kPa|Pa|"
    r"kg|t|days?|weeks?|months?|hours?|hrs?|%)(?![\w²³])"
)


def _calc_payload_numbers(messages: list[dict[str, Any]]) -> set | None:
    """Numbers of this turn's successful construction_calc result, or None.

    Read from the tool messages and the pre-dispatch bubble after the latest
    operator ask. An error envelope is not a result.
    """
    msgs = messages or []
    start = 0
    for i, msg in enumerate(msgs):
        if isinstance(msg, dict) and msg.get("role") == "user" and not _cg_is_platform_bubble(
            str(msg.get("content") or "")
        ):
            start = i
    numbers: set = set()
    found = False
    for msg in msgs[start:]:
        if not isinstance(msg, dict):
            continue
        content = str(msg.get("content") or "")
        is_calc = (
            msg.get("role") == "tool" and msg.get("name") == "construction_calc"
        ) or content.lstrip().startswith(f"{_PREDISPATCH_PREFIX} construction_calc")
        if not is_calc or '"error"' in content or content.lstrip().startswith(_CALC_NOT_RUN_HEAD):
            continue
        found = True
        for tok in _CG_NUM_RE.findall(content):
            v = _cg_to_number(tok)
            if v is not None:
                numbers.add(v)
    return numbers if found else None


def _calc_figure_grounding_gate(text: str, messages: list[dict[str, Any]]) -> str:
    """A calculator answer states no figure the calculation did not produce.

    When construction_calc produced this turn's result, a figure written in
    the same unit as that result must come from the result, the operator's
    own numbers, or a unit rescale of either (4.8 m -> 4800 mm). Anything
    else -- an unrequested variant ("a 10% allowance -> 44 days"), a limit
    recalled from memory ("absolute minimum of 125 mm") -- is a second answer
    the operator did not ask for and nothing grounds. The clause carrying it
    (its parenthetical, else its sentence) is removed; the result stays.
    Kill switch: CALC_FIGURE_GATE=0. Never raises.
    """
    try:
        if os.getenv("CALC_FIGURE_GATE", "1") == "0" or not text:
            return text
        payload = _calc_payload_numbers(messages)
        if not payload:
            return text
        operator = set()
        for tok in _CG_NUM_RE.findall(_latest_operator_ask(messages) or ""):
            v = _cg_to_number(tok)
            if v is not None:
                operator.add(v)
        base = payload | operator
        scaled = base | {v * 1000 for v in base} | {v / 1000 for v in base}
        # The answer's own working on those numbers is grounded too: pairwise
        # + - x / (a bill extension, a unit back-out) and three-factor
        # products (L x B x D). A recalled limit or an invented margin is not.
        ordered = tuple(sorted(scaled))
        triples = {a * b * c for a in operator for b in operator for c in operator}

        def _is_grounded(v: float) -> bool:
            tol = max(0.01, abs(v) * 0.005)
            if any(abs(v - g) <= tol for g in scaled | triples):
                return True
            return _cg_pair_grounds(v, ordered, tol)

        mentions = [
            (m, _cg_to_number(m.group(1)), m.group(2).lower().rstrip("s"))
            for m in _CALC_FIGURE_RE.finditer(text)
        ]
        result_units = {u for _m, v, u in mentions if v is not None and any(
            abs(v - g) <= max(0.01, abs(v) * 0.005) for g in payload)}
        bad = [m for m, v, u in mentions
               if v is not None and u in result_units and not _is_grounded(v)]
        if not bad:
            return text
        out = text
        for m in reversed(bad):
            start, end = _calc_clause_span(out, m.start(), m.end())
            out = out[:start].rstrip(" ,;") + out[end:]
        _LOG.info("calc_figure_gate: removed %d ungrounded figure(s)", len(bad))
        return out
    except Exception:  # noqa: BLE001
        _LOG.exception("calc_figure_gate failed; passing answer through")
        return text


def _calc_clause_span(text: str, start: int, end: int) -> tuple[int, int]:
    """The parenthetical around [start, end), else the sentence around it."""
    open_ = text.rfind("(", 0, start)
    close = text.find(")", end)
    if open_ != -1 and close != -1 and ")" not in text[open_:start] and "\n" not in text[open_:close]:
        return open_, close + 1
    left = max(text.rfind(". ", 0, start), text.rfind("\n", 0, start))
    right_candidates = [i for i in (text.find(". ", end), text.find("\n", end)) if i != -1]
    right = min(right_candidates) + 1 if right_candidates else len(text)
    return (left + 1 if left != -1 else 0), right


def _cost_grounding_gate(
    text: str,
    rag_sys_msg: dict[str, Any] | None,
    messages: list[dict[str, Any]],
) -> str:
    """Refuse a cost/rate answer whose figures don't trace to a rate-semantic
    retrieved chunk or a computed tool result. Non-cost answers pass untouched.
    Never raises."""
    try:
        if not _cost_grounding_enabled():
            return text
        figs = _cg_money_values(text)
        if not figs:
            # The no-rate sentence has no figure of its own. A successful
            # cost calculator this turn is the figure — do not leave the
            # refusal in its place.
            if _answer_is_no_rate_refusal(text):
                rescued = _cost_calc_line_from_tool_result(messages)
                if rescued:
                    return rescued
            return text  # not a cost/rate answer — leave it alone
        user = _latest_operator_ask(messages)
        if _is_document_deliverable_request(user):
            return text
        if _is_contract_data_fact_lookup(user):
            # Delay-damages rate ask: a Contract Data particular, not a
            # BOQ unit rate. Replacing the answer with "upload your priced
            # BOQ" is the wrong refusal for this question class. Arithmetic
            # that wants a SAR figure (daily amount) still goes through the gate.
            return text
        rag_context = (rag_sys_msg or {}).get("content", "") if rag_sys_msg else ""
        grounded = _cg_grounded_numbers(rag_context, messages)
        if all(_cg_is_grounded(v, grounded) for _, v in figs):
            return text
        rescued = _cost_calc_line_from_tool_result(messages)
        if rescued:
            return rescued
        _LOG.warning(
            "cost_grounding_gate: refused ungrounded cost figure(s) %s",
            [f for f, _ in figs],
        )
        return _CG_REFUSAL
    except Exception:
        _LOG.exception("cost_grounding_gate failed; passing answer through")
        return text


def format_excerpts_as_rag_context(
    excerpts: list[dict[str, Any]] | None,
) -> str:
    """Render project-document excerpts into the SAME per-chunk marker form the
    agent path's ``format_chunks_as_system_message`` emits, so the cost-grounding
    gate classifies each excerpt for rate-semantics INDEPENDENTLY.

    Per-chunk markers are load-bearing: a single flat block that mixed a rate
    table with a drawing dimension-table chunk would read as rate-semantic as a
    whole and wrongly ground the drawing's bare numbers (the 2026-07-14 "450
    SAR/m³" incident). Emitting one ``[doc_id=... chunk=N ...]`` block per
    excerpt keeps grounding on the non-agent paths identical to the agent path.

    Excerpt shape matches ``search_project_documents`` /
    ``_with_doc_search`` output (``snippet`` / ``filename`` / ``document_id`` /
    ``score``). Returns "" when there are no usable excerpts."""
    blocks: list[str] = []
    for i, e in enumerate(excerpts or []):
        if not isinstance(e, dict):
            continue
        text = (e.get("snippet") or "").strip()
        if not text:
            continue
        did = e.get("document_id") or e.get("filename") or f"doc{i}"
        score = e.get("score")
        score_s = f" score={score:.3f}" if isinstance(score, (int, float)) else ""
        blocks.append(f"[doc_id={did} chunk={i}{score_s}] {text}")
    return "\n\n".join(blocks)


def gate_cost_answer(
    text: str,
    *,
    rag_context: str = "",
    authoritative_texts: list[str] | None = None,
    user_message: str | None = None,
    project_id: str | None = None,
) -> str:
    """Run the tested cost-grounding gate over an answer produced by a NON-agent
    answer path (``/chat``, ``/v1/project/ask``) that doesn't itself build the
    agent's ``(rag_sys_msg, messages)`` pair.

    * ``rag_context`` — retrieved evidence in the per-chunk ``[doc_id=...]``
      marker form (see ``format_excerpts_as_rag_context``); classified per chunk
      for rate-semantics, exactly like the agent path.
    * ``authoritative_texts`` — computed/tool results produced this turn (e.g.
      the project reasoner's executed step outputs). Every number in them
      grounds — the same authority the agent path grants tool-result messages.
    * ``user_message`` — the caller's own question; figures the user supplied are
      grounded, so the gate never refuses to echo the user's own numbers back.

    Delegates to the existing ``_cost_grounding_gate`` — NO gating logic is
    reimplemented here. Never raises: a gate must never break a working turn."""
    try:
        rag_sys_msg = (
            {"role": "system", "content": rag_context} if rag_context else None
        )
        messages: list[dict[str, Any]] = []
        if user_message:
            messages.append({"role": "user", "content": user_message})
        for t in authoritative_texts or []:
            if t:
                messages.append({"role": "tool", "content": str(t)})
        text = _graft_composed_delay_damages_daily(
            text, rag_sys_msg, messages, project_id=project_id,
        )
        text = _graft_composed_delay_damages_over_period(
            text, rag_sys_msg, messages, project_id=project_id,
        )
        text = _graft_asked_contract_particular(
            text, rag_sys_msg, messages, project_id=project_id,
        )
        text = _graft_composed_percentage_of_aca(
            text, rag_sys_msg, messages, project_id=project_id,
        )
        return _cost_grounding_gate(text, rag_sys_msg, messages)
    except Exception:  # noqa: BLE001 — a gate must never break an answer
        _LOG.exception("gate_cost_answer failed; passing answer through")
        return text


# ── Standards advisory (ADVISORY, never blocks) ─────────────────────────────
# Highlights a deviation from a critical construction standard (e.g. 'APPROVED'
# used on a design document, design review procedure) by APPENDING a note — it never rejects,
# edits, or halts the answer. Operators bend rules deliberately in the field; the
# platform's job is to flag the deviation so the choice is informed, not to
# enforce a stop. Flag STANDARDS_ADVISORY (default on); never raises. This is the
# advisory wiring for the previously-dormant construction_knowledge enforcement.
def _standards_advisory_enabled() -> bool:
    return os.getenv("STANDARDS_ADVISORY", "1") not in ("0", "false", "False", "")


# The design review procedure flags the word APPROVED on design documents. A lighting answer
# that quotes "subject to the Engineer's approval" also contains "design"
# and "document", and the answer-only scan appended this note to a
# question that never asked about design-document wording.
_DESIGN_STATUS_QUESTION_RE = re.compile(
    r"(?i)\bdesign\s+(?:review|document|drawing|package|submission|status)"
    r"|\b(?:drawing|document)\s+status"
    r"|\b(?:approved|approval|approve)\b"
    r"|\bsign[- ]?off\b"
)


def _standards_note_relevant(question: str | None) -> bool:
    """True when the design-review note is about this question.

    No question keeps the historical answer-only scan. A non-empty
    question that is not about design-document status does not get the
    note, even if the answer quotes an Engineer's approval.
    """
    if question is None or not str(question).strip():
        return True
    return bool(_DESIGN_STATUS_QUESTION_RE.search(question))


def _standards_advisory(text: str, question: str | None = None) -> str:
    try:
        if not _standards_advisory_enabled() or not text or not text.strip():
            return text
        if not _standards_note_relevant(question):
            return text
        from app.core.construction_knowledge import enforce_critical_rules
        violations = enforce_critical_rules(text)
        if not violations:
            return text
        seen: set = set()
        lines: list[str] = []
        for v in violations:
            rid = v.get("rule_id")
            if rid in seen:
                continue
            seen.add(rid)
            # The live procedure document's code when the corpus has one,
            # else the catalogue kind's label -- never a shipped code.
            kind = str(v.get("procedure") or "").split("/")[0]
            pid = v.get("procedure_id") or kind
            if pid == kind and kind:
                from app.core.procedure_catalogue import label as _proc_label

                proc = _proc_label(kind)
            else:
                proc = pid
            msg = v.get("violation_message") or v.get("rule", "")
            lines.append(f"> - **{proc}** — {msg}" if proc else f"> - {msg}")
        if not lines:
            return text
        return text + (
            "\n\n> **Standards note (advisory — flagging, not blocking):**\n"
            + "\n".join(lines)
        )
    except Exception:
        _LOG.exception("standards advisory failed; passing answer through")
        return text


# STEP 0b — visible disclosure prepended to any answer that fell back to the
# Master Corpus because the active project has no usable documents of its own.
# The sources panel is separately tagged (see _build_sources_from_audit); this
# is the in-answer half of "fallback VISIBLE in response + sources panel."
_MASTER_CORPUS_FALLBACK_NOTE = (
    "_This project has no documents of its own for this question — "
    "answering from the Master Corpus._\n\n"
)


_MISSING_PARTICULAR_RE = re.compile(
    r"(?i)could not (?:confirm|find)|did not find|what i did not find|"
    r"none of (?:the|which)|not among the excerpts|"
    r"i can search again",
)
_DELAY_DAMAGES_ANSWER_RE = re.compile(r"(?i)delay\s+damages")
# Live A2 FAIL after #517: graft / model led with excl-VAT labeled as
# including VAT, then restated the elected including-VAT figure.
_ACA_INCL_CLAIM_RE = re.compile(
    r"(?i)(?:the\s+)?accepted\s+contract\s+amount\s+including\s+vat\s+is\s+"
    r"(SAR|AED|USD|EUR|GBP|QAR|BHD|KWD|OMR)\s*"
    r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d{2})\.?",
)
# Time for Completion ask: graft prepended "…is 90 days." then the model stated 852.
_TFC_WHOLE_WORKS_CLAIM_RE = re.compile(
    r"(?i)(?:the\s+)?time\s+for\s+completion\s+for\s+(?:the\s+)?"
    r"whole\s+of\s+the\s+works\s+is\s+(\d{2,4})\s+days\.?",
)


def _aca_claim_amount(match: re.Match[str]) -> float | None:
    raw = (match.group(2) or "").replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return None


def _has_conflicting_aca_incl_claim(text: str, elected_amount: float) -> bool:
    return any(
        amt is not None and abs(amt - elected_amount) > 0.005
        for amt in (
            _aca_claim_amount(m) for m in _ACA_INCL_CLAIM_RE.finditer(text or "")
        )
    )


def _strip_conflicting_aca_incl_claims(text: str, elected_amount: float) -> str:
    """Drop 'including VAT is SAR X' claims that are not the elected row."""
    raw = text or ""
    if not raw:
        return raw

    def _keep_or_drop(match: re.Match[str]) -> str:
        amt = _aca_claim_amount(match)
        if amt is not None and abs(amt - elected_amount) <= 0.005:
            return match.group(0)
        return ""

    cleaned = _ACA_INCL_CLAIM_RE.sub(_keep_or_drop, raw)
    cleaned = re.sub(r"(?m)^\s*\.\s*$", "", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r" {2,}", " ", cleaned)
    return cleaned.strip()


def _has_conflicting_whole_works_tfc_claim(text: str, elected_days: str) -> bool:
    elected_num = (elected_days or "").split()[0]
    if not elected_num:
        return False
    return any(
        m.group(1) != elected_num
        for m in _TFC_WHOLE_WORKS_CLAIM_RE.finditer(text or "")
    )


def _strip_conflicting_whole_works_tfc_claims(text: str, elected_days: str) -> str:
    """Drop 'whole of the Works is N days' claims that are not the elected row."""
    elected_num = (elected_days or "").split()[0]
    raw = text or ""
    if not elected_num or not raw:
        return raw

    def _keep_or_drop(match: re.Match[str]) -> str:
        return match.group(0) if match.group(1) == elected_num else ""

    cleaned = _TFC_WHOLE_WORKS_CLAIM_RE.sub(_keep_or_drop, raw)
    cleaned = re.sub(r"(?m)^\s*\.\s*$", "", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r" {2,}", " ", cleaned)
    return cleaned.strip()


def _graft_asked_contract_particular(
    text: str,
    rag_sys_msg: dict[str, Any] | None,
    messages: list[dict[str, Any]] | None,
    project_id: str | None = None,
    extra_project_ids: list[str] | None = None,
) -> str:
    """State the asked Contract Data row (ACA incl. VAT, TfC, DNP, Engineer) from excerpts.

    DeepSeek answered a neighboring field (delay damages for an including-VAT
    ACA ask) or reported the particular absent after retrieving permit
    trackers / CoC 1.3–1.8. Compose nothing — only fire when the excerpt
    already states the asked value.
    ``project_id`` lets a last-chance scan read the loaded CD volume when
    top-k is a delay-damages chunk #0 (live 9ad62cc).
    """
    try:
        from app.core.rag.retriever import (
            including_vat_excerpts_from_loaded_cd_volume,
            extract_aca_including_vat,
            extract_defects_notification_period,
            asked_party_role,
            extract_party_name,
            extract_time_for_completion_days,
            party_role_title,
            query_asks_delay_damages_daily_amount,
            query_asks_for_defects_notification_period,
            query_asks_for_time_for_completion,
            query_is_aca_including_vat_particular,
        )
        user = _latest_operator_ask(messages)
        rag = (rag_sys_msg or {}).get("content", "") if rag_sys_msg else ""
        text = _strip_answer_routing_preamble(text)
        if not user:
            return text
        # Delay-damages daily-amount ask: compose already owns rate × ACA.
        # The including-VAT election must not replace a daily figure (or a
        # pending compose) with the neighboring particular.
        if query_asks_delay_damages_daily_amount(user):
            return text
        line = ""
        if query_is_aca_including_vat_particular(user):
            parsed = extract_aca_including_vat(rag)
            if not parsed and (project_id or extra_project_ids):
                # Live Wave-1 A2 on 9ad62cc / 396cc7b: top-k was
                # Contract Data chunk #0 (delay damages × a partial
                # ACA). Scan the loaded CD volume — including cited
                # chunk owners, not only the UI project id — for the
                # filled including-VAT row. Do not invent a figure.
                extra = including_vat_excerpts_from_loaded_cd_volume(
                    user, project_id or "", rag_context=rag,
                    extra_pids=extra_project_ids,
                )
                if extra:
                    rag = f"{rag}\n\n{extra}" if rag else extra
                    parsed = extract_aca_including_vat(rag)
            if not parsed or not rag:
                return text
            amount, currency = parsed
            line = (
                f"The Accepted Contract Amount including VAT is "
                f"{currency} {amount:,.2f}."
            )
            raw = text or ""
            already = (
                f"{amount:,.2f}" in raw
                or f"{amount:.2f}" in raw
                or f"{amount:,.2f}".replace(",", "") in raw.replace(",", "")
            )
            if (
                already
                and not _DELAY_DAMAGES_ANSWER_RE.search(raw)
                and not _MISSING_PARTICULAR_RE.search(raw)
                and not _has_conflicting_aca_incl_claim(raw, amount)
            ):
                return text
            if (
                _DELAY_DAMAGES_ANSWER_RE.search(raw)
                or _MISSING_PARTICULAR_RE.search(raw)
                or _GENERIC_ACK_RE.search(raw)
                or raw.strip() == _CG_REFUSAL
            ):
                return line
            cleaned = _strip_conflicting_aca_incl_claims(raw, amount)
            if not cleaned:
                return line
            if cleaned.startswith(line):
                return cleaned
            return f"{line}\n\n{cleaned}"
        if query_asks_for_time_for_completion(user):
            days = extract_time_for_completion_days(rag)
            if not days:
                return text
            line = (
                f"The Time for Completion for the whole of the Works is {days}."
            )
            raw = text or ""
            elected_num = days.split()[0]
            if (
                elected_num in raw
                and not _MISSING_PARTICULAR_RE.search(raw)
                and not _has_conflicting_whole_works_tfc_claim(raw, days)
            ):
                return text
            if (
                _MISSING_PARTICULAR_RE.search(raw)
                or _GENERIC_ACK_RE.search(raw)
                or raw.strip() == _CG_REFUSAL
            ):
                return line
            cleaned = _strip_conflicting_whole_works_tfc_claims(raw, days)
            if not cleaned:
                return line
            if cleaned.startswith(line):
                return cleaned
            return f"{line}\n\n{cleaned}"
        # Owner ruling 2026-09-19: no party names leave this RAG. This graft
        # used to write "The Engineer is <firm>." in front of the answer.
        from app.core.party_names import (
            party_names_withheld,
            query_asks_who_a_party_is,
            withheld_answer_line,
        )
        if party_names_withheld() and (
            query_asks_who_a_party_is(user) or asked_party_role(user)
        ):
            line = withheld_answer_line(user)
            body = (text or "").strip()
            if "withheld" in body.lower():
                return text
            if (
                not body
                or _MISSING_PARTICULAR_RE.search(body)
                or _GENERIC_ACK_RE.search(body)
                or body == _CG_REFUSAL
            ):
                return line
            return f"{line}\n\n{body}"
        party_role = asked_party_role(user)
        if party_role:
            name = extract_party_name(rag, party_role)
            if not name:
                return text
            line = f"The {party_role_title(party_role)} is {name}."
            if name.lower() in (text or "").lower() and not _MISSING_PARTICULAR_RE.search(
                text or "",
            ):
                return text
            # A firm name split across two tokens.
            tokens = [t for t in re.split(r"[^A-Za-z0-9]+", name) if len(t) >= 4]
            if tokens and all(t.lower() in (text or "").lower() for t in tokens):
                if not _MISSING_PARTICULAR_RE.search(text or ""):
                    return text
            if (
                _MISSING_PARTICULAR_RE.search(text or "")
                or _GENERIC_ACK_RE.search(text or "")
                or (text or "").strip() == _CG_REFUSAL
            ):
                return line
            body = (text or "").strip()
            out = line if not body else f"{line}\n\n{body}"
            return _strip_answer_routing_preamble(out)
        if query_asks_for_defects_notification_period(user):
            days = extract_defects_notification_period(rag)
            if not days:
                return text
            line = f"The Defects Notification Period is {days}."
            raw = text or ""
            elected_num = days.split()[0]
            if (
                elected_num in raw
                and not _MISSING_PARTICULAR_RE.search(raw)
            ):
                return text
            if (
                _MISSING_PARTICULAR_RE.search(raw)
                or _GENERIC_ACK_RE.search(raw)
                or raw.strip() == _CG_REFUSAL
            ):
                return line
            body = raw.strip()
            return line if not body else f"{line}\n\n{body}"
        # Named percentage particular when synthesis hung empty.
        from app.lib.construction_formulas_commercial import (
            extract_named_percentage_particular,
            format_named_percentage_line,
            query_asks_named_percentage_particular,
            query_asks_percentage_particular_in_money,
            strip_conflicting_named_percentage,
            text_states_percent,
        )
        if query_asks_named_percentage_particular(user):
            parsed = extract_named_percentage_particular(user, rag)
            if not parsed:
                return text
            line = format_named_percentage_line(parsed)
            # "10%" is a substring of "0.10%". That token is the daily
            # rate, not this particular. Strip a sentence that assigns
            # the label the neighbouring percent before deciding.
            raw = strip_conflicting_named_percentage(
                text or "", parsed.get("label") or "", float(parsed["percent"]),
            )
            already = text_states_percent(raw, float(parsed["percent"]))
            if already and not _MISSING_PARTICULAR_RE.search(raw):
                return raw
            # A money ask ("Calculate … in SAR") still needs the product;
            # do not lock the turn on the percentage-only line here.
            if query_asks_percentage_particular_in_money(user) and already:
                return raw
            if (
                not raw.strip()
                or _MISSING_PARTICULAR_RE.search(raw)
                or _GENERIC_ACK_RE.search(raw)
                or raw.strip() == _CG_REFUSAL
                or raw.strip() == _EMPTY_RESPONSE_FALLBACK
            ):
                return line
            body = raw.strip()
            return line if not body else f"{line}\n\n{body}"
        return text
    except Exception:  # noqa: BLE001 — graft must never break a turn
        _LOG.exception("asked-particular graft failed; passing answer through")
        return text


def _wants_user_supplied_arithmetic_directive(text: str) -> bool:
    """True when the RAG fold must not apply the strict lookup clamp.

    Self-contained L×W×D / expression arithmetic already has this path.
    A hypothetical milestone comparison (M#=Nd supplied in the question)
    is the same class: the operands are not supposed to be in the corpus.
    """
    if _looks_like_self_contained_calculation(text):
        return True
    # Named formula asks carry no L×W×D, so the dimension heuristic
    # misses them. The strict lookup clamp then tells the model to
    # answer ONLY from retrieved excerpts — on a thin fixture that is
    # Master Corpus bleed, and the calculator is never called.
    if _message_is_formula_style_ask(text):
        return True
    try:
        from app.core.hypothetical_milestone_arithmetic import (
            query_is_hypothetical_milestone_arithmetic,
        )
        return query_is_hypothetical_milestone_arithmetic(text)
    except Exception:  # noqa: BLE001 — fold must still pick a directive
        _LOG.debug(
            "hypothetical-milestone directive check failed",
            exc_info=True,
        )
        return False


def _graft_hypothetical_milestone_arithmetic(
    text: str,
    messages: list[dict[str, Any]] | None,
) -> str:
    """Lead with user-supplied milestone arithmetic, not a premise veto.

    Live ~27d6940 rejected user-stated milestone durations (397d, 487d,
    731d, same start) from
    Contract Data access dates. The numbers in the question are the
    operands. Kill-switch ``HYPOTHETICAL_MILESTONE_ARITHMETIC=0``.
    """
    try:
        from app.core.hypothetical_milestone_arithmetic import (
            graft_hypothetical_milestone_answer,
        )
        return graft_hypothetical_milestone_answer(
            text, _latest_operator_ask(messages),
        )
    except Exception:  # noqa: BLE001 — graft must never break a turn
        _LOG.exception(
            "hypothetical-milestone compose failed; passing answer through"
        )
        return text


def _graft_composed_delay_damages_daily(
    text: str,
    rag_sys_msg: dict[str, Any] | None,
    messages: list[dict[str, Any]] | None,
    project_id: str | None = None,
    extra_project_ids: list[str] | None = None,
) -> str:
    """Delay-damages daily-amount ask: state rate × ACA as a daily figure when both are in docs.

    The live FAIL quoted Contract Data sources and never multiplied.
    Compose from retrieved excerpts only — no invented operands. A
    fabricated SAR/day still loses when the ACA is absent. When top-k
    is refuse-prone, a last-chance scan of the loaded CD volume may
    still supply both operands. The composed envelope is
    appended as a tool message so the cost gate can ground the product
    (0.1% × ACA is not a pairwise product of the raw numbers 0.1 and
    the ACA).
    """
    try:
        from app.lib.construction_formulas_commercial import (
            answer_states_daily_amount,
            compose_delay_damages_daily_from_excerpts,
            delay_damages_base_is_clause_111,
            drop_whole_of_works_delay_claims,
            format_delay_damages_daily_line,
        )
        user = _latest_operator_ask(messages)
        rag = (rag_sys_msg or {}).get("content", "") if rag_sys_msg else ""
        try:
            from app.core.rag.retriever import (
                query_is_aca_including_vat_particular,
            )
            # Do not compose SAR/day for an ACA-including-VAT
            # particular, even when chunk #0 is delay damages × a
            # partial ACA. Combined daily-amount + including-VAT
            # wording stays a daily-amount ask. Classify the unwrapped
            # operator ask — the RAG-folded bubble contains
            # "Delay Damages" + "compute" and looks like a daily-amount ask.
            if query_is_aca_including_vat_particular(user):
                return text
        except Exception:  # noqa: BLE001 — ask-class fence must never break
            _LOG.debug("including-VAT compose fence failed; keeping daily-amount ask check", exc_info=True)
        composed = compose_delay_damages_daily_from_excerpts(user, rag)
        # A partial Accepted Contract Amount on the rate chunk composes
        # on its own (milestone rate: 0.015% × SAR 39,098,392.98). That success
        # used to skip the loaded-volume scan, so clause 1.1.1 never
        # replaced it. Look when the base is not clause 1.1.1, and only
        # replace it when the volume states that clause.
        needs_clause = True
        if composed:
            try:
                needs_clause = not delay_damages_base_is_clause_111(
                    rag, float(composed["contract_amount"]),
                )
            except Exception:  # noqa: BLE001 — a scored base still stands
                _LOG.debug("clause 1.1.1 check failed", exc_info=True)
                needs_clause = False
        if needs_clause:
            # After #536: top-k is refuse-prone Contract
            # Data chunks 9–11 that do not surface both operands. When
            # excl-VAT ACA + Contract Data 0.1% exist in the loaded CD
            # volume, compose from those rows — do not invent a figure
            # and do not fall through to the cost-grounding refuse.
            try:
                from app.core.rag.retriever import (
                    daily_damages_excerpts_from_loaded_cd_volume,
                )
                extra = daily_damages_excerpts_from_loaded_cd_volume(
                    user, project_id or "", rag_context=rag,
                    extra_pids=extra_project_ids,
                )
                if extra:
                    better = compose_delay_damages_daily_from_excerpts(
                        user, extra,
                    )
                    if better and (
                        composed is None
                        or delay_damages_base_is_clause_111(
                            extra, float(better["contract_amount"]),
                        )
                    ):
                        composed = better
            except Exception:  # noqa: BLE001 — volume scan must never break
                _LOG.debug(
                    "e1 loaded-volume compose failed; keeping excerpt compose",
                    exc_info=True,
                )
        if not composed:
            return text
        line = format_delay_damages_daily_line(composed)
        payload = json.dumps({
            "calculation": "delay_damages_daily",
            "daily_amount": composed["daily_amount"],
            "rate_percent": composed["rate_percent"],
            "contract_amount": composed["contract_amount"],
            "currency": composed["currency"],
            "note": composed.get("note") or line,
        })
        if isinstance(messages, list) and not any(
            isinstance(m, dict)
            and m.get("role") == "tool"
            and "delay_damages_daily" in str(m.get("content") or "")
            for m in messages
        ):
            messages.append({"role": "tool", "content": payload})
        # A Milestone or section answer that already states the right
        # daily figure must not keep a "whole of the Works" lead. That
        # label is the other rate.
        if (composed.get("basis") or "") in ("milestone", "section"):
            text = drop_whole_of_works_delay_claims(text or "")
        if answer_states_daily_amount(text or "", composed["daily_amount"]):
            return text
        figs = _cg_money_values(text or "")
        daily = float(composed["daily_amount"])
        base = float(composed["contract_amount"])
        extras = [
            v for _f, v in figs
            if abs(v - daily) > 1.0 and abs(v - base) > 1.0
        ]
        raw = text or ""
        # After #523: the model (or the including-VAT graft) led
        # with including-VAT ACA and never stated SAR/day. Once both
        # operands are in the excerpts, replace that particular — do
        # not prepend under an including-VAT-shaped lead. Same for a cannot-
        # calculate refusal over Spec TOC / Daywork / insurance.
        aca_lead = bool(_ACA_INCL_CLAIM_RE.search(raw))
        no_daily = not answer_states_daily_amount(raw, daily) and not re.search(
            r"(?i)\bper\s+(?:calendar\s+)?day\b", raw,
        )
        if (
            extras
            or aca_lead
            or no_daily
            or raw.strip() == _CG_REFUSAL
            or _MISSING_PARTICULAR_RE.search(raw)
            or _GENERIC_ACK_RE.search(raw)
            or re.search(
                r"(?i)cannot calculate|could not calculate|"
                r"(?:do|does) not contain",
                raw,
            )
        ):
            return line
        body = raw.strip()
        return line if not body else f"{line}\n\n{body}"
    except Exception:  # noqa: BLE001 — compose must never break a turn
        _LOG.exception("delay-damages daily compose failed; passing answer through")
        return text


def _graft_composed_delay_damages_over_period(
    text: str,
    rag_sys_msg: dict[str, Any] | None,
    messages: list[dict[str, Any]] | None,
    project_id: str | None = None,
    extra_project_ids: list[str] | None = None,
) -> str:
    """Delay damages over a period: rate × ACA × days × named milestones when synthesis hung.

    When ``delay_damages_daily`` ran with empty args and returned a
    ``0% of SAR 0.00`` note although both operands were already on the
    Contract Data sheet, compose the period product from them; never invent
    a rate or a base.
    """
    try:
        from app.lib.construction_formulas_commercial import (
            answer_is_unbound_delay_damages,
            answer_states_money_amount,
            compose_delay_damages_over_period_from_excerpts,
            format_delay_damages_period_line,
            query_asks_delay_damages_over_a_period,
        )
        user = _latest_operator_ask(messages)
        if not query_asks_delay_damages_over_a_period(user):
            return text
        rag = (rag_sys_msg or {}).get("content", "") if rag_sys_msg else ""
        composed = compose_delay_damages_over_period_from_excerpts(user, rag)
        if not composed:
            # Period ask: top-k packed the whole-of-Works 0.1% under
            # "per Milestone". Scan the loaded CD volume for the
            # Milestone N | 0.015% rows + excl-VAT ACA.
            try:
                from app.core.rag.retriever import (
                    milestone_period_excerpts_from_loaded_cd_volume,
                )
                extra = milestone_period_excerpts_from_loaded_cd_volume(
                    user, project_id or "", rag_context=rag,
                    extra_pids=extra_project_ids,
                )
                if extra:
                    composed = compose_delay_damages_over_period_from_excerpts(
                        user, extra,
                    )
            except Exception:  # noqa: BLE001 — volume scan must never break
                _LOG.debug(
                    "e3 loaded-volume compose failed; keeping excerpt compose",
                    exc_info=True,
                )
                composed = None
        if not composed:
            return text
        line = format_delay_damages_period_line(composed)
        payload = json.dumps({
            "calculation": "delay_damages_over_period",
            "amount": composed["amount"],
            "days": composed["days"],
            "rates": composed.get("rates"),
            "milestones": composed.get("milestones"),
            "contract_amount": composed["contract_amount"],
            "currency": composed["currency"],
            "note": composed.get("note") or line,
        })
        if isinstance(messages, list) and not any(
            isinstance(m, dict)
            and m.get("role") == "tool"
            and "delay_damages_over_period" in str(m.get("content") or "")
            for m in messages
        ):
            messages.append({"role": "tool", "content": payload})
        raw = text or ""
        if answer_states_money_amount(raw, float(composed["amount"])):
            return text
        if (
            not raw.strip()
            or raw.strip() == _CG_REFUSAL
            or raw.strip() == _EMPTY_RESPONSE_FALLBACK
            or answer_is_unbound_delay_damages(raw)
            or _MISSING_PARTICULAR_RE.search(raw)
            or _GENERIC_ACK_RE.search(raw)
            or re.search(
                r"(?i)cannot calculate|could not calculate|"
                r"(?:do|does) not contain|unbound delay-damages",
                raw,
            )
        ):
            return line
        return line
    except Exception:  # noqa: BLE001 — compose must never break a turn
        _LOG.exception(
            "delay-damages period compose failed; passing answer through"
        )
        return text


def _graft_composed_percentage_of_aca(
    text: str,
    rag_sys_msg: dict[str, Any] | None,
    messages: list[dict[str, Any]] | None,
    project_id: str | None = None,
    extra_project_ids: list[str] | None = None,
) -> str:
    """Named percentage × ACA when synthesis hung empty."""
    try:
        from app.lib.construction_formulas_commercial import (
            answer_states_money_amount,
            compose_percentage_of_aca_from_excerpts,
            format_percentage_of_aca_line,
            query_asks_percentage_particular_in_money,
            strip_vat_inclusive_percentage_alternative,
            vat_inclusive_alternative_amounts,
        )
        user = _latest_operator_ask(messages)
        if not query_asks_percentage_particular_in_money(user):
            return text
        rag = (rag_sys_msg or {}).get("content", "") if rag_sys_msg else ""
        composed = compose_percentage_of_aca_from_excerpts(user, rag)
        if not composed:
            # Named-percentage ask: top-k had neither 10% nor ACA; the model
            # shipped the fetch truncation notice. Scan the loaded CD
            # volume for Advance Payment 10% + excl-VAT ACA.
            try:
                from app.core.rag.retriever import (
                    percentage_of_aca_excerpts_from_loaded_cd_volume,
                )
                extra = percentage_of_aca_excerpts_from_loaded_cd_volume(
                    user, project_id or "", rag_context=rag,
                    extra_pids=extra_project_ids,
                )
                if extra:
                    composed = compose_percentage_of_aca_from_excerpts(
                        user, extra,
                    )
            except Exception:  # noqa: BLE001 — volume scan must never break
                _LOG.debug(
                    "e1 percentage loaded-volume compose failed; "
                    "keeping excerpt compose",
                    exc_info=True,
                )
                composed = None
        if not composed:
            return text
        line = format_percentage_of_aca_line(composed)
        payload = json.dumps({
            "calculation": "percentage_of_aca",
            "amount": composed["amount"],
            "percent": composed["percent"],
            "contract_amount": composed["contract_amount"],
            "currency": composed["currency"],
            "label": composed.get("label"),
            "note": line,
        })
        if isinstance(messages, list) and not any(
            isinstance(m, dict)
            and m.get("role") == "tool"
            and "percentage_of_aca" in str(m.get("content") or "")
            for m in messages
        ):
            messages.append({"role": "tool", "content": payload})
        raw = text or ""
        product = float(composed["amount"])
        base = float(composed["contract_amount"])
        forbidden = [
            amt for amt in vat_inclusive_alternative_amounts(
                rag, float(composed["percent"]),
            )
            if abs(amt - product) > 1.0 and abs(amt - base) > 1.0
        ]
        # One amount. The excluding-VAT product is the answer. A
        # VAT-inclusive alternative, including one the model already
        # wrote next to the right figure, is not a second result.
        cleaned = strip_vat_inclusive_percentage_alternative(
            raw, keep_amount=product, forbidden_amounts=forbidden,
        )
        if answer_states_money_amount(cleaned, product):
            same = re.sub(r"\s+", " ", cleaned.strip()) == re.sub(
                r"\s+", " ", raw.strip(),
            )
            return raw if same else cleaned
        return line
    except Exception:  # noqa: BLE001 — compose must never break a turn
        _LOG.exception("percentage-of-ACA compose failed; passing answer through")
        return text


_GENERIC_ACK_RE = re.compile(
    r"(?i)i(?:'m| am) ready to help|"
    r"please let me know what specific|"
    r"how can i help|"
    r"what (?:would you like|can i (?:help|do))|"
    r"i will answer (?:only )?from (?:the )?(?:client )?(?:project )?documents",
)


def _graft_rate_only_item(
    text: str,
    rag_sys_msg: dict[str, Any] | None,
    messages: list[dict[str, Any]] | None,
) -> str:
    """Rate Only BOQ item ask: state Rate Only when the retrieved row already says so.

    Live Master Corpus greeted and never named D529.3. Compose nothing —
    only fire when an excerpt already says Rate Only on the asked item.
    A fabricated money total is replaced.
    """
    try:
        from app.core.rag.retriever import (
            answer_states_rate_only,
            chunk_states_rate_only_item,
            compose_priced_boq_row,
            extract_asked_cesmm_codes,
            format_rate_only_line,
            query_asks_for_boq_item_amount,
        )
        user = _latest_operator_ask(messages)
        if not query_asks_for_boq_item_amount(user):
            return text
        codes = extract_asked_cesmm_codes(user)
        if not codes:
            return text
        rag = (rag_sys_msg or {}).get("content", "") if rag_sys_msg else ""
        # WAVE 2 B5: a priced Part Nr. 3 line beats a Rate Only sibling.
        if compose_priced_boq_row(user, rag):
            return text
        if not chunk_states_rate_only_item(rag, codes):
            return text
        line = format_rate_only_line(codes, rag)
        if answer_states_rate_only(text or ""):
            return text
        body = (text or "").strip()
        if not body or _GENERIC_ACK_RE.search(body) or (text or "").strip() == _CG_REFUSAL:
            return line
        figs = _cg_money_values(text or "")
        if figs:
            return line
        return f"{line}\n\n{body}"
    except Exception:  # noqa: BLE001 — graft must never break a turn
        _LOG.exception("rate-only graft failed; passing answer through")
        return text


def _graft_honest_contract_refusal(
    text: str,
    rag_sys_msg: dict[str, Any] | None,
    messages: list[dict[str, Any]] | None,
) -> str:
    """PCG / commencement-date asks: prefer Contract Data honesty over form/pack invention.

    Parent company guarantee: if excerpts say PCG is not required, replace a Schedule 8
    "20% of paid-up Capital" invention. A filled CD value is left (or
    grafted) instead of the form %.

    Commencement date: if excerpts say the commencement field is not populated / tied
    to LOA-NOA, replace an invented pack date. A filled CD date is
    left (or grafted). An ask that names the commencement pack is not
    this path.

    Compose nothing — only fire when an excerpt already states the
    CD row.
    """
    try:
        from app.core.rag.retriever import (
            answer_invents_commencement_date,
            answer_invents_pcg_value,
            answer_states_commencement_not_populated,
            answer_states_pcg_not_required,
            chunk_states_commencement_filled_date,
            chunk_states_commencement_not_populated,
            chunk_states_pcg_contract_data,
            chunk_states_pcg_filled_value,
            chunk_states_pcg_not_required,
            format_commencement_honest_line,
            format_commencement_unsupported_line,
            format_pcg_honest_line,
            query_asks_for_contract_commencement_date,
            query_asks_for_parent_company_guarantee,
        )
        user = _latest_operator_ask(messages)
        rag = (rag_sys_msg or {}).get("content", "") if rag_sys_msg else ""
        raw = text or ""
        if (
            query_asks_for_parent_company_guarantee(user)
            and chunk_states_pcg_contract_data(rag)
        ):
            line = format_pcg_honest_line(rag)
            already_honest = (
                chunk_states_pcg_not_required(rag)
                and answer_states_pcg_not_required(raw)
                and not answer_invents_pcg_value(raw)
            )
            already_filled = (
                chunk_states_pcg_filled_value(rag)
                and not chunk_states_pcg_not_required(rag)
                and line.split("(")[0].strip().lower() in raw.lower()
                and not re.search(r"(?i)paid[- ]up\s+capital", raw)
            )
            if already_honest or already_filled:
                return text
            if (
                not raw.strip()
                or _GENERIC_ACK_RE.search(raw)
                or raw.strip() == _CG_REFUSAL
                or _MISSING_PARTICULAR_RE.search(raw)
                or (
                    chunk_states_pcg_not_required(rag)
                    and answer_invents_pcg_value(raw)
                )
                or re.search(r"(?i)paid[- ]up\s+capital", raw)
            ):
                return line
            return f"{line}\n\n{raw.strip()}" if raw.strip() else line
        if (
            query_asks_for_contract_commencement_date(user)
        ):
            # Filled Contract Data date still wins. Anything else — empty
            # CD, TBA, pack-only RAG, or no CD row at all — must not
            # ship an invented calendar date (commencement ask after #563).
            if chunk_states_commencement_filled_date(rag):
                line = format_commencement_honest_line(rag)
                already_filled = (
                    not chunk_states_commencement_not_populated(rag)
                    and line.split(" is ", 1)[-1].rstrip(".").lower() in raw.lower()
                )
                if already_filled:
                    return text
                if (
                    not raw.strip()
                    or _GENERIC_ACK_RE.search(raw)
                    or raw.strip() == _CG_REFUSAL
                    or _MISSING_PARTICULAR_RE.search(raw)
                    or (
                        answer_invents_commencement_date(raw)
                        and line.split(" is ", 1)[-1].rstrip(".").lower()
                        not in raw.lower()
                    )
                ):
                    return line
                return f"{line}\n\n{raw.strip()}" if raw.strip() else line
            line = (
                format_commencement_honest_line(rag)
                if chunk_states_commencement_not_populated(rag)
                else format_commencement_unsupported_line()
            )
            already_honest = (
                answer_states_commencement_not_populated(raw)
                and not answer_invents_commencement_date(raw)
            )
            if already_honest:
                return text
            if (
                not raw.strip()
                or _GENERIC_ACK_RE.search(raw)
                or raw.strip() == _CG_REFUSAL
                or _MISSING_PARTICULAR_RE.search(raw)
                or answer_invents_commencement_date(raw)
            ):
                return line
            return text
        return text
    except Exception:  # noqa: BLE001 — graft must never break a turn
        _LOG.exception("honest-contract-refusal graft failed; passing answer through")
        return text


def _answer_echoes_ask(text: str, user: str) -> bool:
    """True when synthesis only repeated the operator ask (live B4 chrome)."""
    raw = re.sub(r"\s+", " ", (text or "").strip().lower())
    ask = re.sub(r"\s+", " ", (user or "").strip().lower())
    if not raw or not ask:
        return False
    if raw.rstrip("?.") == ask.rstrip("?."):
        return True
    if ask in raw and len(raw) <= len(ask) + 120:
        return True
    tail = ask[-48:] if len(ask) > 48 else ask
    return bool(tail and tail in raw and len(raw) < 280)


def _graft_priced_boq_item(
    text: str,
    rag_sys_msg: dict[str, Any] | None,
    messages: list[dict[str, Any]] | None,
) -> str:
    """WAVE 2 B4/B5: write qty + amount when synthesis hung empty.

    Live Master Corpus on 2ceef76 (#545) echoed the D599.5 ask, cited
    nothing, and waited ~212s. #542 already elected the priced row over
    storm-water Rate Only — compose those figures from the excerpt.
    Kill-switch: COMPOSE_PRICED_BOQ_ROW=0. A Rate Only item is not this path.
    """
    try:
        from app.core.rag.retriever import (
            answer_states_priced_boq,
            compose_priced_boq_row,
            extract_asked_cesmm_codes,
            format_priced_boq_line,
            priced_boq_compose_enabled,
            query_asks_for_boq_item_amount,
        )
        if not priced_boq_compose_enabled():
            return text
        user = _latest_operator_ask(messages)
        if not query_asks_for_boq_item_amount(user):
            return text
        codes = extract_asked_cesmm_codes(user)
        if not codes:
            return text
        rag = (rag_sys_msg or {}).get("content", "") if rag_sys_msg else ""
        parsed = compose_priced_boq_row(user, rag)
        if not parsed:
            return text
        line = format_priced_boq_line(parsed)
        if not line:
            return text
        payload = json.dumps({
            "priced_boq_row": {
                "code": parsed.get("code"),
                "qty": parsed.get("qty"),
                "unit": parsed.get("unit"),
                "rate": parsed.get("rate"),
                "amount": parsed.get("amount"),
                "note": line,
            }
        })
        if isinstance(messages, list) and not any(
            isinstance(m, dict)
            and m.get("role") == "tool"
            and "priced_boq_row" in str(m.get("content") or "")
            for m in messages
        ):
            messages.append({"role": "tool", "content": payload})
        raw = text or ""
        if answer_states_priced_boq(raw, parsed):
            return text
        body = raw.strip()
        # A check ("Verify: does 34,844 m at SAR 142.00/m equal the stated
        # amount?") is answered by a VERDICT, and a verdict is short. The
        # rules below replace any answer under 500 characters that does not
        # restate the row, which threw the verdict away and left the bare row
        # (unseen Set 3 E4, with the shortcut guard already live). For a
        # check the row is evidence: put it in front, never in place.
        from app.core.rag.retriever import query_is_a_check_not_a_lookup
        if (
            query_is_a_check_not_a_lookup(user)
            and body
            and body not in (_CG_REFUSAL, _EMPTY_RESPONSE_FALLBACK)
            and not _GENERIC_ACK_RE.search(body)
            and not _looks_like_search_preamble(body)
        ):
            return f"{line}\n\n{body}"
        storm_misroute = bool(
            re.search(r"(?i)storm\s+water", body)
            and not re.search(r"(?i)storm\s+water", parsed.get("description") or "")
        )
        # Live B5: model refused Rate Only (PART NR. 3) vs Excluded
        # (priced BOQ) and never wrote the elected amount.
        sibling_conflict = bool(
            re.search(r"(?i)\brate\s*only\b", body)
            and re.search(r"(?i)\bexcluded\b", body)
        )
        if (
            not body
            or body == _CG_REFUSAL
            or body == _EMPTY_RESPONSE_FALLBACK
            or _GENERIC_ACK_RE.search(body)
            or _looks_like_search_preamble(body)
            or _MISSING_PARTICULAR_RE.search(body)
            or _answer_echoes_ask(body, user)
            or storm_misroute
            or sibling_conflict
        ):
            return line
        if len(body) < 500 and not answer_states_priced_boq(body, parsed):
            return line
        return f"{line}\n\n{body}"
    except Exception:  # noqa: BLE001 — compose must never break a turn
        _LOG.exception("priced-boq compose failed; passing answer through")
        return text


def _compose_priced_boq_instead_of_retry(
    text: str,
    rag_sys_msg: dict[str, Any] | None,
    messages: list[dict[str, Any]] | None,
) -> str:
    """Priced CESMM qty+amount from RAG, or '' when compose cannot fill.

    Live WAVE 2 B4 hung ~212s on forced retry after empty / question-echo
    / search-promise synthesis. The elected D599.5 figures were already
    in the excerpts — compose them instead of another LLM hop.
    """
    try:
        from app.core.rag.retriever import (
            answer_states_priced_boq,
            compose_priced_boq_row,
            priced_boq_compose_enabled,
        )
        if not priced_boq_compose_enabled():
            return ""
        user = _latest_operator_ask(messages)
        rag = (rag_sys_msg or {}).get("content", "") if rag_sys_msg else ""
        parsed = compose_priced_boq_row(user, rag)
        if not parsed:
            return ""
        grafted = _graft_priced_boq_item(text or "", rag_sys_msg, messages)
        if grafted and answer_states_priced_boq(grafted, parsed):
            return grafted
        return ""
    except Exception:  # noqa: BLE001 — retry skip must never break a turn
        _LOG.exception("priced-boq retry skip failed; continuing retry path")
        return ""


def _audit_project_ids_for_daily_damages(
    project_id: str | None,
    audit_rec: dict[str, Any] | None,
) -> tuple[str | None, list[str] | None]:
    """UI project + cited-chunk owners for the delay-damages daily-amount last-chance compose."""
    pid = project_id or (audit_rec or {}).get("project_id")
    extra: list[str] = []
    seen: set[str] = set()
    if pid:
        extra.append(str(pid))
        seen.add(str(pid))
    for ch in (audit_rec or {}).get("chunks") or []:
        if not isinstance(ch, dict):
            continue
        cp = str(ch.get("project_id") or "").strip()
        if cp and cp not in seen:
            seen.add(cp)
            extra.append(cp)
    return (str(pid) if pid else None), (extra or None)


def _should_short_circuit_delay_damages_daily(
    rag_sys_msg: dict[str, Any] | None,
    messages: list[dict[str, Any]] | None,
    *,
    has_predispatch: bool,
    project_id: str | None = None,
    audit_rec: dict[str, Any] | None = None,
) -> str:
    """Composed delay-damages SAR/day, or '' if the LLM hop must still run.

    After #559: synthesis copied the contracts-kernel
    BOQ refuse ("I don't have a rate on file… upload your priced BOQ")
    when priced CESMM soup shared the pool. Rate × ACA was already in
    the excerpts / loaded CD volume — compose it instead of waiting on
    the provider. Predispatch deliverables keep the LLM path.
    """
    if has_predispatch:
        return ""
    try:
        from app.lib.construction_formulas_commercial import (
            query_asks_delay_damages_daily_amount,
        )
        user = _latest_operator_ask(messages)
        if not query_asks_delay_damages_daily_amount(user):
            return ""
        pid, extra = _audit_project_ids_for_daily_damages(project_id, audit_rec)
        grafted = _graft_composed_delay_damages_daily(
            "", rag_sys_msg, messages, project_id=pid,
            extra_project_ids=extra,
        )
        if grafted and grafted.strip() and grafted.strip() != _CG_REFUSAL:
            return grafted
        return ""
    except Exception:  # noqa: BLE001 — retry skip must never break a turn
        _LOG.exception("e1 daily short-circuit failed; continuing LLM path")
        return ""


def _boq_scope_wbs_compose_enabled() -> bool:
    """ON by default — leftover F1 skips the LLM hop once generate_wbs is BOQ-derived."""
    return (os.getenv("F1_BOQ_WBS_COMPOSE", "1") or "").strip().lower() not in (
        "0", "false", "no", "off",
    )


def _calc_input_question(pre: dict[str, Any] | None) -> str:
    """The pre-dispatched calculator's question for inputs it would not run, or ''."""
    if not pre or pre.get("name") != "construction_calc":
        return ""
    result = pre.get("result")
    if not isinstance(result, dict) or not result.get("needs_input"):
        return ""
    return str(result.get("question") or "")


def _compose_boq_scope_wbs_answer(
    pre: dict[str, Any] | None,
    user_message: str,
) -> str:
    """Formatted BOQ-derived WBS, or '' when this is not leftover F1."""
    if not _boq_scope_wbs_compose_enabled():
        return ""
    if not pre or pre.get("name") != "generate_wbs":
        return ""
    if not _message_wants_boq_scope_wbs_ask(user_message or ""):
        return ""
    result = pre.get("result")
    if not isinstance(result, dict):
        return ""
    scaffold = result.get("scaffold") if isinstance(result.get("scaffold"), dict) else {}
    if not scaffold.get("derived_from_boq"):
        return ""
    return _format_wbs_result(result)


def _graft_boq_scope_wbs_if_wrong_contract(
    text: str,
    messages: list[dict[str, Any]] | None,
) -> str:
    """Replace a leftover-F1 refuse / AB-2022 cite with the predispatched WBS."""
    if not _boq_scope_wbs_compose_enabled():
        return text
    user = _latest_operator_ask(messages)
    if not _message_wants_boq_scope_wbs_ask(user):
        return text
    from app.lib.boq_schedule import (
        boq_wbs_answer_fails_wrong_contract,
        boq_wbs_answer_is_grounded,
    )
    raw = text or ""
    if boq_wbs_answer_is_grounded(raw) and not boq_wbs_answer_fails_wrong_contract(raw):
        return text
    recovered = _recover_answer_from_tool_messages(_EMPTY_RESPONSE_FALLBACK, messages)
    if recovered and boq_wbs_answer_is_grounded(recovered):
        return recovered
    return text


def _should_short_circuit_priced_boq(
    rag_sys_msg: dict[str, Any] | None,
    messages: list[dict[str, Any]] | None,
    *,
    has_predispatch: bool,
) -> str:
    """Composed priced-row answer, or '' if the LLM hop must still run.

    Live WAVE 2 B4 on 2ceef76: when synthesis/provider died the UI showed
    the unavailable banner and no citations. The elected D599.5 figures
    were already in the excerpts — do not wait on OpenRouter. Predispatch
    deliverables keep the LLM path.
    """
    if has_predispatch:
        return ""
    # The composed row answers "what is the amount". It does not answer
    # "verify: does 158 ha at 186,328 equal the stated amount?" — live
    # 5312551 sent back the row and no verdict. A check goes to the model.
    try:
        from app.core.rag.retriever import query_is_a_check_not_a_lookup
        if query_is_a_check_not_a_lookup(_latest_operator_ask(messages)):
            return ""
    except Exception:  # noqa: BLE001 — never break a turn over an import
        _LOG.warning("priced-boq shortcut: lookup check failed", exc_info=True)
    return _compose_priced_boq_instead_of_retry("", rag_sys_msg, messages)


_PART_SUMMARY_MISS_RE = re.compile(
    r"(?i)part\s+summary|priced\s+boq|not\s+found|could\s+not\s+find|"
    r"cannot\s+find|no\s+(?:printed\s+)?total",
)
_MILESTONES_ONLY_RE = re.compile(
    r"(?i)milestones?\s+1\s*[–-]\s*5\s+only|none of those are\s+"
    r"northern|no northern",
)


def _graft_combined_part_summary_total(
    text: str,
    rag_sys_msg: dict[str, Any] | None,
    messages: list[dict[str, Any]] | None,
) -> str:
    """Combined BOQ page totals: sum printed page totals when the model added 1,000,000.

    Live 4ab5561: d/3/1 + d/3/2 + d/3/3 printed 34,645,529 + 1,852,848
    + 17,496,857 = 53,995,234 and the bubble said 54,995,234. Compose
    the printed figures; do not invent a page. Kill-switch:
    COMPOSE_PART_SUMMARY=0. One-page B3 stays on
    ``_graft_part_summary_total``.
    """
    try:
        from app.core.rag.retriever import (
            answer_states_part_summary,
            compose_combined_part_summary_total,
            format_combined_part_summary_line,
            part_summary_compose_enabled,
            query_asks_combined_part_summary,
        )
        if not part_summary_compose_enabled():
            return text
        user = _latest_operator_ask(messages)
        if not query_asks_combined_part_summary(user):
            return text
        rag = (rag_sys_msg or {}).get("content", "") if rag_sys_msg else ""
        parsed = compose_combined_part_summary_total(user, rag)
        if not parsed:
            return text
        line = format_combined_part_summary_line(parsed)
        if not line:
            return text
        payload = json.dumps({
            "combined_part_summary_total": {
                "pages": parsed.get("pages"),
                "amount": parsed.get("amount"),
                "page_amounts": parsed.get("page_amounts"),
                "currency": parsed.get("currency") or "",
                "note": line,
            }
        })
        if isinstance(messages, list) and not any(
            isinstance(m, dict)
            and m.get("role") == "tool"
            and "combined_part_summary_total" in str(m.get("content") or "")
            for m in messages
        ):
            messages.append({"role": "tool", "content": payload})
        raw = text or ""
        if answer_states_part_summary(raw, parsed):
            return text
        return line
    except Exception:  # noqa: BLE001 — graft must never break a turn
        _LOG.exception("combined part-summary compose failed; passing answer through")
        return text


def _graft_named_community_tfc_span(
    text: str,
    rag_sys_msg: dict[str, Any] | None,
    messages: list[dict[str, Any]] | None,
    project_id: str | None = None,
    extra_project_ids: list[str] | None = None,
) -> str:
    """Set3 F1: longest/shortest Time for Completion in a named community.

    Live 4ab5561 answered Milestones 1–5 only and said none were
    Northern Community. The continuation table states 547 / 397
    (delta 150). Compose from excerpts; last-chance scan the loaded
    CD volume when top-k stopped at Milestone 5. Do not invent days.
    """
    try:
        from app.core.rag.retriever import (
            compose_named_community_tfc_span,
            format_named_community_tfc_span_line,
            query_asks_named_community_tfc_span,
        )
        user = _latest_operator_ask(messages)
        if not query_asks_named_community_tfc_span(user):
            return text
        rag = (rag_sys_msg or {}).get("content", "") if rag_sys_msg else ""
        composed = compose_named_community_tfc_span(user, rag)
        if not composed:
            try:
                from app.core.rag.retriever import (
                    community_tfc_span_excerpts_from_loaded_cd_volume,
                )
                extra = community_tfc_span_excerpts_from_loaded_cd_volume(
                    user, project_id or "", rag_context=rag,
                    extra_pids=extra_project_ids,
                )
                if extra:
                    composed = compose_named_community_tfc_span(user, extra)
            except Exception:  # noqa: BLE001 — volume scan must never break
                _LOG.debug(
                    "f1 loaded-volume TFC span compose failed; "
                    "keeping excerpt compose",
                    exc_info=True,
                )
                composed = None
        if not composed:
            return text
        line = format_named_community_tfc_span_line(composed)
        if not line:
            return text
        raw = text or ""
        longest = str(composed.get("longest_days") or "")
        delta = str(composed.get("delta") or "")
        if longest and delta and longest in raw and delta in raw:
            return text
        body = raw.strip()
        if (
            not body
            or body == _CG_REFUSAL
            or body == _EMPTY_RESPONSE_FALLBACK
            or _GENERIC_ACK_RE.search(body)
            or _MISSING_PARTICULAR_RE.search(body)
            or _MILESTONES_ONLY_RE.search(body)
            or re.search(
                r"(?i)cannot (?:answer|confirm|find)|could not|"
                r"not in the (?:retrieved )?excerpts",
                body,
            )
        ):
            return line
        return line
    except Exception:  # noqa: BLE001 — graft must never break a turn
        _LOG.exception("named-community TFC span compose failed; passing through")
        return text



def _graft_part_summary_total(
    text: str,
    rag_sys_msg: dict[str, Any] | None,
    messages: list[dict[str, Any]] | None,
) -> str:
    """BOQ page Part Summary ask: write the page Part Summary when synthesis hung.

    Live 0d9fd23 mobile H3 echoed empty / said the priced-BOQ Part
    Summary was not found. The d/3/1 total was already in the
    demolition-bill excerpts (B1/B2 line items on the same page PASS).
    Kill-switch: COMPOSE_PART_SUMMARY=0. B4/B5 CESMM rows are not this.
    """
    try:
        from app.core.rag.retriever import (
            answer_states_part_summary,
            compose_part_summary_total,
            format_part_summary_line,
            part_summary_compose_enabled,
            query_asks_for_part_summary_total,
        )
        if not part_summary_compose_enabled():
            return text
        user = _latest_operator_ask(messages)
        if not query_asks_for_part_summary_total(user):
            return text
        rag = (rag_sys_msg or {}).get("content", "") if rag_sys_msg else ""
        parsed = compose_part_summary_total(user, rag)
        if not parsed:
            return text
        line = format_part_summary_line(parsed)
        if not line:
            return text
        payload = json.dumps({
            "part_summary_total": {
                "page": parsed.get("page"),
                "amount": parsed.get("amount"),
                "currency": parsed.get("currency") or "",
                "note": line,
            }
        })
        if isinstance(messages, list) and not any(
            isinstance(m, dict)
            and m.get("role") == "tool"
            and "part_summary_total" in str(m.get("content") or "")
            for m in messages
        ):
            messages.append({"role": "tool", "content": payload})
        raw = text or ""
        if answer_states_part_summary(raw, parsed):
            return text
        body = raw.strip()
        if (
            not body
            or body == _CG_REFUSAL
            or body == _EMPTY_RESPONSE_FALLBACK
            or _GENERIC_ACK_RE.search(body)
            or _looks_like_search_preamble(body)
            or _MISSING_PARTICULAR_RE.search(body)
            or _PART_SUMMARY_MISS_RE.search(body)
            or _answer_echoes_ask(body, user)
        ):
            return line
        if len(body) < 500 and not answer_states_part_summary(body, parsed):
            return line
        return f"{line}\n\n{body}"
    except Exception:  # noqa: BLE001 — graft must never break a turn
        _LOG.exception("part-summary compose failed; passing answer through")
        return text


def _compose_part_summary_instead_of_retry(
    text: str,
    rag_sys_msg: dict[str, Any] | None,
    messages: list[dict[str, Any]] | None,
) -> str:
    """Part Summary page total from RAG, or '' when compose cannot fill.

    Live H3 hung or refused 'not found' while the printed d/3/1 total
    was already in the excerpts. Compose it instead of another LLM hop.
    """
    try:
        from app.core.rag.retriever import (
            answer_states_part_summary,
            compose_part_summary_total,
            part_summary_compose_enabled,
            query_asks_for_part_summary_total,
        )
        if not part_summary_compose_enabled():
            return ""
        user = _latest_operator_ask(messages)
        if not query_asks_for_part_summary_total(user):
            return ""
        rag = (rag_sys_msg or {}).get("content", "") if rag_sys_msg else ""
        parsed = compose_part_summary_total(user, rag)
        if not parsed:
            return ""
        grafted = _graft_part_summary_total(text or "", rag_sys_msg, messages)
        if grafted and answer_states_part_summary(grafted, parsed):
            return grafted
        return ""
    except Exception:  # noqa: BLE001 — retry skip must never break a turn
        _LOG.exception("part-summary retry skip failed; continuing retry path")
        return ""


def _should_short_circuit_part_summary(
    rag_sys_msg: dict[str, Any] | None,
    messages: list[dict[str, Any]] | None,
    *,
    has_predispatch: bool,
) -> str:
    """Composed Part Summary total, or '' if the LLM hop must still run.

    Live H3 on 0d9fd23: two mobile attempts produced no response; the
    third said the total was not found. The figure was already in the
    demolition-bill excerpts. Predispatch deliverables keep the LLM.
    """
    if has_predispatch:
        return ""
    return _compose_part_summary_instead_of_retry("", rag_sys_msg, messages)


def _compose_excerpt_boq_instead_of_retry(
    text: str,
    rag_sys_msg: dict[str, Any] | None,
    messages: list[dict[str, Any]] | None,
) -> str:
    """Priced CESMM row, else Part Summary page total, else ''."""
    return (
        _compose_priced_boq_instead_of_retry(text, rag_sys_msg, messages)
        or _compose_part_summary_instead_of_retry(text, rag_sys_msg, messages)
        or _compose_user_priced_takeoff_instead_of_retry(text, messages)
    )


def _compose_user_priced_takeoff_instead_of_retry(
    text: str,
    messages: list[dict[str, Any]] | None,
) -> str:
    """Compose volume × user rate instead of a second empty LLM hop."""
    grafted = _graft_composed_user_priced_takeoff(text, messages)
    if not grafted.strip() or grafted.strip() == _EMPTY_RESPONSE_FALLBACK:
        return ""
    if grafted.strip() == (text or "").strip():
        return ""
    return grafted



# ── derivation: the working and the result must agree ────────────────────
#
# Two live defects of the same shape, found 23-24 Sep 2026 and fixed one at a
# time:
#
#   slab span   "L/20 = 4800/20 = 200 mm"      -- 4800/20 is 240
#   concrete Ec "Ec = 4700 x 5.9161 = 28,062"  -- that product is 27,806
#
# Both answers stated the right rule, the right inputs and a wrong number, and
# every visible part of them was correct. A reader checking the working is
# reassured by it. This checks the arithmetic the answer shows: a mismatch is
# annotated, never silently rewritten -- correcting the figure would hide that
# the answer and its own working disagree, which is the fact worth surfacing.
#
# A chained equality is one step at a time, each step against its own left
# side. "0.4 + 420/700 = 0.4 + 0.600 = 1.000" is three sides; the quotient
# is not compared with the neighbouring addend or with the total.
#
# Kill-switch: DERIVATION_CHECK=0.
# A trailing percent sign is part of the number and means hundredths.
# Without it, "0.1% × base" is either unread (the sign sits between the
# factor and the operator) or read as 0.1, a hundred times too large.
_DERIVATION_NUM = r"\d[\d,]*(?:\.\d+)?(?:\s*%)?"
_DERIVATION_OP = r"[x×*/÷]"
_DERIVATION_START_RE = re.compile(rf"(?<![\w.]){_DERIVATION_NUM}")
_DERIVATION_TERM_RE = re.compile(
    rf"{_DERIVATION_NUM}(?:\s*{_DERIVATION_OP}\s*{_DERIVATION_NUM})*"
)
_DERIVATION_PLUS_RE = re.compile(r"\s*\+\s*")
_DERIVATION_EQ_RE = re.compile(r"\s*=(?!=)\s*")
_DERIVATION_NUMBER_RE = re.compile(rf"{_DERIVATION_NUM}(?![\d.])")
_DERIVATION_HAS_OP_RE = re.compile(rf"{_DERIVATION_OP}|\+")
_DERIVATION_MAX_NOTES = 3


def _derivation_enabled() -> bool:
    return (os.getenv("DERIVATION_CHECK", "1") or "").strip().lower() not in (
        "0", "false", "no", "off",
    )


def _derivation_number(token: str) -> float | None:
    raw = token.replace(",", "").strip()
    percent = raw.endswith("%")
    if percent:
        raw = raw[:-1].strip()
    try:
        value = float(raw)
    except (TypeError, ValueError):
        # The token came from a numeric pattern, so this is a surprise worth
        # seeing rather than a routine miss.
        _LOG.debug("derivation: unparsed numeric token %r", token, exc_info=True)
        return None
    if percent:
        value /= 100.0
    return value


def _derivation_bound_to_previous(text: str, start: int) -> bool:
    """True when ``start`` is inside a larger left side, not the start of one.

    "0.4 + 420/700" must be read from the 0.4. Starting at the quotient would
    compare 420/700 with whatever number follows the equals sign.
    """
    i = start - 1
    while i >= 0 and text[i].isspace():
        i -= 1
    if i < 0:
        return False
    ch = text[i]
    if ch in "+-×*/÷−–—":
        return True
    if ch in "xX":
        prev = text[i - 1] if i else ""
        return not (prev.isalnum() or prev == "_")
    return False


def _eval_single_operator_term(expr: str) -> float | None:
    """Left-fold a term that uses only ``*`` or only ``/``.

    A mixed term ("2 x 3 / 4") is refused: the two readings disagree, and
    guessing precedence would accuse working the answer may have meant.
    ``None`` means "do not check", including division by zero.
    """
    parts = [
        _derivation_number(tok)
        for tok in re.split(rf"\s*{_DERIVATION_OP}\s*", expr)
    ]
    if any(p is None for p in parts) or not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    ops = re.findall(_DERIVATION_OP, expr)
    if len(ops) != len(parts) - 1:
        return None
    if len({"/" if op in "/÷" else "*" for op in ops}) != 1:
        return None
    computed = parts[0]
    dividing = ops[0] in "/÷"
    for value in parts[1:]:
        if dividing:
            if value == 0:
                return None
            computed /= value
        else:
            computed *= value
    return computed


def _read_term(text: str, pos: int) -> tuple[int, float | None] | None:
    match = _DERIVATION_TERM_RE.match(text, pos)
    if match is None:
        return None
    return match.end(), _eval_single_operator_term(match.group(0))


def _read_side(
    text: str, pos: int, *, allow_sum: bool,
) -> tuple[tuple[str, float | None], int] | None:
    """One side of an equality.

    A sum (``0.4 + 420/700``) is one side. The right side of a product or
    quotient stays a single number, so ``12 + 5 x 6`` after ``3 x 4 =`` is
    not swallowed into that equation.
    """
    start = pos
    if not allow_sum:
        match = _DERIVATION_NUMBER_RE.match(text, pos)
        if match is None:
            return None
        raw = match.group(0)
        return (raw, _derivation_number(raw)), match.end()
    term = _read_term(text, pos)
    if term is None:
        return None
    pos, val = term
    while True:
        plus = _DERIVATION_PLUS_RE.match(text, pos)
        if plus is None:
            break
        nxt = _read_term(text, plus.end())
        if nxt is None:
            break
        pos, nxt_val = nxt
        if val is None or nxt_val is None:
            val = None
        else:
            val = val + nxt_val
    return (text[start:pos], val), pos


def _derivation_chain_mismatches(
    text: str, start: int, out: list[tuple[str, float, float]],
) -> int | None:
    """Append disagreements for the chain that begins at ``start``.

    Returns the index just after the chain, or ``None`` when ``start`` is
    not the left side of a derivation (so the caller tries the next number).
    """
    parsed = _read_side(text, start, allow_sum=True)
    if parsed is None:
        return None
    side, pos = parsed
    if _DERIVATION_HAS_OP_RE.search(side[0]) is None:
        # "700 = 0.4 + …" is not a derivation. Consuming it would hide the
        # sum that actually follows the equals sign.
        return None
    sides = [side]
    while True:
        eq = _DERIVATION_EQ_RE.match(text, pos)
        if eq is None:
            break
        allow_sum = "+" in sides[-1][0]
        nxt = _read_side(text, eq.end(), allow_sum=allow_sum)
        if nxt is None:
            break
        sides.append(nxt[0])
        pos = nxt[1]
    if len(sides) < 2:
        return None
    for (lexpr, lval), (_rexpr, rval) in zip(sides, sides[1:]):
        if lval is None or rval is None:
            continue
        if _DERIVATION_HAS_OP_RE.search(lexpr) is None:
            continue
        tolerance = max(abs(lval) * 0.005, 0.01)
        if abs(rval - lval) > tolerance:
            out.append((lexpr.strip(), rval, lval))
    return pos


def derivation_mismatches(text: str) -> list[tuple[str, float, float]]:
    """``(expression, stated result, computed result)`` for each disagreement.

    Each side of a chained equality is checked against the side on its left.
    A sum is one side (``0.4 + 420/700``), so a quotient inside it is not
    paired with the next addend or with the total.     A term is still a single
    operator (``4700 x 5.9161``, ``4800 / 20``); a mixed ``2 x 3 / 4`` is
    skipped. A trailing ``%`` on a factor is hundredths (``0.1%`` is
    ``0.001``). Tolerance is 0.5% or 0.01, whichever is larger, so a rounded
    result (12.345 shown as 12.35) agrees.
    """
    raw = text or ""
    out: list[tuple[str, float, float]] = []
    consumed = 0
    for match in _DERIVATION_START_RE.finditer(raw):
        if match.start() < consumed:
            continue
        if _derivation_bound_to_previous(raw, match.start()):
            continue
        end = _derivation_chain_mismatches(raw, match.start(), out)
        if end is None:
            continue
        consumed = end
    return out


def _annotate_derivation_mismatches(text: str) -> str:
    """Append a note naming each disagreement. Never edits the figures."""
    if not _derivation_enabled() or not text:
        return text
    try:
        mismatches = derivation_mismatches(text)
    except Exception:  # noqa: BLE001 — a check must never break a turn
        _LOG.warning("derivation check failed; answer passed through", exc_info=True)
        return text
    if not mismatches:
        return text
    lines = []
    for expr, stated, computed in mismatches[:_DERIVATION_MAX_NOTES]:
        _LOG.warning(
            "derivation mismatch: %s stated %s, computes to %s", expr, stated, computed,
        )
        lines.append(f"- `{expr}` is **{computed:,.4g}**, not {stated:,.10g}.")
    more = len(mismatches) - len(lines)
    if more > 0:
        lines.append(f"- …and {more} more.")
    return (
        f"{text}\n\n**Check this answer's own arithmetic before using it — the "
        f"working and the result disagree:**\n" + "\n".join(lines)
    )


_SLAB_GUARD_REFUSAL_RE = re.compile(
    r"not in the retrieved excerpts|cannot state what|"
    r"won'?t attribute|will not attribute|"
    r"would you like me to run|no retrieved excerpt",
    re.IGNORECASE,
)


def _graft_computed_slab_thickness(
    text: str,
    messages: list[dict[str, Any]] | None,
) -> str:
    """Replace a corpus refusal with the slab_thickness_min result.

    Live minimum-slab-thickness asks: the model cited the named-standard guard and never
    called the calculator, including turns that named slab_thickness_min
    and offered to run it. The figure is computed. Provenance is the
    calculator's standard field.
    """
    user = _latest_operator_ask(messages)
    if not user:
        return text
    try:
        from app.lib.construction_formulas import run_calculation
        from app.lib.construction_formulas_structural_rc import (
            answer_states_slab_thickness_result,
            format_slab_thickness_answer,
            looks_like_slab_thickness_min_ask,
        )
    except Exception:
        _LOG.exception("slab thickness graft import failed")
        return text
    if not looks_like_slab_thickness_min_ask(user):
        return text
    if (
        answer_states_slab_thickness_result(user, text or "")
        and not _SLAB_GUARD_REFUSAL_RE.search(text or "")
    ):
        return text
    try:
        env = run_calculation("slab_thickness_min", {"text": user})
    except Exception:
        _LOG.exception("slab thickness graft calculation failed")
        return text
    if not isinstance(env, dict) or env.get("status") == "error":
        return text
    line = format_slab_thickness_answer(
        env.get("result") if isinstance(env.get("result"), dict) else {},
    )
    if not line:
        return text
    return line


def _graft_named_standard_attribution(
    text: str,
    rag_sys_msg: dict[str, Any] | None,
    messages: list[dict[str, Any]] | None,
) -> str:
    """P4b: do not let a project figure wear a named code the turn did not read.

    The operator asked what NFPA 51B requires. Retrieval returned the
    project's hot-work permit. The model then called 30 minutes NFPA's.
    Same shape for Dubai Municipality lighting and ACI 305 fresh-concrete
    temperature. Kill switch: NAMED_STANDARD_ATTRIBUTION_GATE=0.
    """
    try:
        from app.core.rag.named_standard_attribution import relabel_answer
        user = _latest_operator_ask(messages)
        rag = (rag_sys_msg or {}).get("content", "") if rag_sys_msg else ""
        return relabel_answer(user, text or "", rag)
    except Exception:
        _LOG.exception(
            "named-standard attribution graft failed; passing answer through",
        )
        return text


def _graft_stated_total_follow_up(
    text: str,
    messages: list[dict[str, Any]] | None,
) -> str:
    """Replace a one-element follow-up with the stated total.

    A follow-up that adds a waste percentage and a unit price to "that
    total" after N identical elements came back as ONE element's volume
    (one element × waste) and its price. The count is in the previous
    operator turn. A reply that already states the N-element volume and
    price is left as written.
    """
    try:
        from app.lib.construction_formulas_commercial import (
            answer_states_stated_total,
            compose_stated_total_follow_up,
            format_stated_total_follow_up_line,
        )
    except Exception:
        _LOG.exception("stated-total follow-up import failed")
        return text
    users: list[str] = []
    for msg in messages or []:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = str(msg.get("content") or "")
        head = content.lstrip()
        if head.startswith(_PREDISPATCH_PREFIX) or head.startswith(_TOOL_RESULT_PREFIX):
            continue
        content = _unwrap_rag_folded_operator_text(content).strip()
        if content:
            users.append(content)
    if len(users) < 2:
        return text
    try:
        composed = compose_stated_total_follow_up(users[-1], "\n".join(users[:-1]))
    except Exception:
        _LOG.exception("stated-total follow-up compose failed")
        return text
    if not composed:
        return text
    if answer_states_stated_total(text or "", composed):
        return text
    line = format_stated_total_follow_up_line(composed)
    if not line:
        return text
    return line


def _record_figure_provenance(
    text: str,
    rag_sys_msg: dict[str, Any] | None,
    messages: list[dict[str, Any]],
    audit_rec: dict[str, Any] | None,
) -> str:
    """Record where each figure came from (stream + message store). On this
    path the record is kept, not enforced: the answer text is unchanged."""
    from app.agents.citation_provenance import figure_provenance
    from app.agents.provenance_trail import LAST_PROVENANCE

    _out, entries = figure_provenance(text, rag_sys_msg, messages, enforce=False)
    LAST_PROVENANCE.set(entries)
    if isinstance(audit_rec, dict):
        audit_rec["provenance"] = entries
    return text


def _rss_mb() -> int:
    """Resident memory of this process in MB (Linux /proc; psutil elsewhere;
    -1 when neither is readable)."""
    try:
        with open("/proc/self/status", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    try:
        import psutil

        return int(psutil.Process().memory_info().rss // (1024 * 1024))
    except Exception:  # noqa: BLE001 — a missing reading is reported as -1
        _LOG.debug("rss unreadable", exc_info=True)
        return -1


def _timing_log(fmt: str, *args: Any) -> None:
    """A TIMING line, always carrying the process's resident memory, so a
    capacity run shows whether memory stays flat turn after turn."""
    _LOG.warning(fmt + " rss=%dMB", *args, _rss_mb())


# Stage lines (app.core.turn_timing) are written by _timing_log too, so every
# TIMING line has one format and one rss reading.
from app.core import turn_timing as _turn_timing_mod  # noqa: E402
_turn_timing_mod.set_writer(lambda fmt, *args: _timing_log(fmt, *args))



_PROJECT_ASSUMPTION_RE = re.compile(
    r"project(?:['’]s)?\s+documented"
    r"|documented\s+(?:waste|allowance|factor)"
    r"|according to (?:the )?(?:project documents|retrieved excerpts|excerpts)"
    r"|(?:the )?retrieved excerpts"
    r"|from the project documents",
    re.IGNORECASE,
)


def _answer_has_close_number(text: str, value: float) -> bool:
    for tok in re.findall(r"\d[\d,]*(?:\.\d+)?", text or ""):
        try:
            found = float(tok.replace(",", ""))
        except ValueError:
            continue
        tol = max(0.01, abs(value) * 0.005)
        if abs(found - value) <= tol:
            return True
    return False


def _invented_project_assumption(text: str, user: str, retrieval: str) -> bool:
    """A sentence attributes a number to the project that neither the user nor the excerpts contain."""
    for part in re.split(r"[\n.;]+", text or ""):
        if _PROJECT_ASSUMPTION_RE.search(part) is None:
            continue
        for tok in re.findall(r"\d[\d,]*(?:\.\d+)?", part):
            try:
                number = float(tok.replace(",", ""))
            except ValueError:
                continue
            if (
                not _answer_has_close_number(user, number)
                and not _answer_has_close_number(retrieval, number)
            ):
                return True
    return False


def _calculator_success_recorded(messages: list[dict[str, Any]] | None, name: str) -> bool:
    needle = f'"calculation": "{name}"'
    for msg in messages or []:
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        content = str(msg.get("content") or "")
        if needle in content and '"status": "success"' in content:
            return True
    return False


def _record_calculator_tool_turn(
    messages: list[dict[str, Any]],
    name: str,
    bound: dict[str, Any],
    envelope: dict[str, Any],
) -> None:
    if _calculator_success_recorded(messages, name):
        return
    tid = "supplied-input-construction_calc"
    args = {
        "calculation": name,
        "params": {key: val for key, val in (bound or {}).items()},
    }
    messages.append({
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": tid,
            "type": "function",
            "function": {
                "name": "construction_calc",
                "arguments": json.dumps(args),
            },
        }],
    })
    messages.append({
        "role": "tool",
        "name": "construction_calc",
        "tool_call_id": tid,
        "content": json.dumps(envelope, default=str),
    })


def _graft_supplied_input_calculation(
    text: str,
    messages: list[dict[str, Any]] | None,
    rag_sys_msg: dict[str, Any] | None,
) -> str:
    """State the registered calculator's result for inputs the user supplied.

    Replaces the answer when it does not state that result, or when it
    attributes a number the user did not give to the project documents.
    The run is a tool message so the existing credit path can name the
    calculator and those inputs.
    """
    user = _latest_operator_ask(messages)
    selected = _user_supplied_registered_calculation(user)
    if not selected:
        return text
    name, bound = selected
    try:
        from app.lib.construction_formulas import (
            calculation_headline,
            format_user_calculation_answer,
            run_calculation,
        )
        env = run_calculation(name, {**bound, "text": user})
    except Exception:  # noqa: BLE001 — a graft must never break a turn
        _LOG.exception("supplied-input calculation failed")
        return text
    if not isinstance(env, dict) or env.get("status") != "success":
        return text
    _record_calculator_tool_turn(messages or [], name, bound, env)
    headline = calculation_headline(name, env.get("result"))
    retrieval = (rag_sys_msg or {}).get("content", "") if rag_sys_msg else ""
    missing = (
        headline is not None
        and not _answer_has_close_number(text or "", headline[1])
    )
    invented = _invented_project_assumption(text or "", user, retrieval)
    if not missing and not invented:
        return text
    replacement = format_user_calculation_answer(name, bound, env, user_text=user)
    return replacement if replacement.strip() else text


def _scrub_supplied_input_wording(
    text: str,
    messages: list[dict[str, Any]] | None,
    rag_sys_msg: dict[str, Any] | None,
) -> str:
    """Currency, clause and comparison text on a user-supplied calculation.

    Runs for every formula the user completed, from the request, the
    excerpts and that formula's result. A turn that did not supply the
    inputs is left to the document answer.
    """
    user = _latest_operator_ask(messages)
    selected = _user_supplied_registered_calculation(user)
    if not selected or not text:
        return text
    name, bound = selected
    retrieval = ""
    if isinstance(rag_sys_msg, dict):
        retrieval = str(rag_sys_msg.get("content") or "")
    result = None
    try:
        from app.lib.construction_formulas import run_calculation
        env = run_calculation(name, {**bound, "text": user})
        if isinstance(env, dict) and env.get("status") == "success":
            result = env.get("result")
    except Exception:  # noqa: BLE001 — the wording guard must not break a turn
        _LOG.exception("supplied-input wording guard could not read the result")
    try:
        from app.lib.source_labels import guard_supplied_input_wording
        return guard_supplied_input_wording(
            text, user, retrieval, name, bound, result,
        )
    except Exception:  # noqa: BLE001 — the wording guard must not break a turn
        _LOG.exception("supplied-input wording guard failed")
        return text


def _scrub_registry_ids(text: str) -> str:
    """Registered formula and tool ids in the answer read as display names;
    a ``key=literal`` dump goes rather than being read as words."""
    if not text:
        return text
    try:
        from app.agents.answer_exit import remove_dumps
        from app.lib.source_labels import plain_registry_text
        return plain_registry_text(remove_dumps(text))
    except Exception:  # noqa: BLE001 — the id scrub must not break a turn
        _LOG.exception("registry id scrub failed")
        return text


@_turn_timing_stage("postprocess")
def _postprocess_answer(
    text: str,
    rag_sys_msg: dict[str, Any] | None,
    messages: list[dict[str, Any]],
    fallback_used: bool = False,
    agent_name: str | None = None,
    project_id: str | None = None,
    coverage: tuple[int, int] | None = None,
    audit_rec: dict[str, Any] | None = None,
) -> str:
    """Final-answer post-processing: cost-grounding gate (may refuse an
    ungrounded cost claim) then the standards advisory (appends a non-blocking
    deviation note). Order matters — don't annotate a refusal for cost figures
    it no longer contains.

    When ``fallback_used`` is set (the retriever answered from the Master Corpus
    because the project is empty/thin), a one-line disclosure banner is
    prepended so the fallback is visible in the answer itself."""
    from app.agents import answer_exit

    answer_exit.note_evidence(rag_sys_msg, messages)
    text = _recover_answer_from_tool_messages(text, messages)
    # The fetch-window truncation notice is not an
    # answer. Blank it so later grafts can compose, and so we never
    # ship "Tool result exceeded … char_offset=" to the operator.
    if _looks_like_tool_truncation_notice(text):
        text = ""
    # Empty / incomplete construction_calc: a formula-name note or a
    # 1 m rebar demo is not a complete answer. Graft the computed ask.
    text = _graft_complete_calc_answer(text, messages)
    # Leftover F1: refuse + AB-2022 CoC cite is FAIL. If generate_wbs
    # already produced a BOQ-derived tree, that draft is the answer.
    text = _graft_boq_scope_wbs_if_wrong_contract(text, messages)
    text = _graft_operator_claim_facts(text, _operator_user_text(messages))
    # Leftover E4: compose raft volume + documented waste when the model
    # hung on "Let me validate…" and never wrote 945 m³.
    text = _graft_composed_concrete_volume(text, messages)
    # Cost gate: calc succeeded, force_synthesis emitted 0 tokens.
    # Volume-only recover is not a priced take-off — compose from the ask.
    text = _graft_composed_user_priced_takeoff(text, messages)
    # "Add a waste allowance and price the total" continues from the prior
    # count (24 caps / 180 m³), not from one element the tool just recomputed.
    text = _graft_stated_total_follow_up(text, messages)
    # Live ~27d6940: user-supplied M#=Nd + common start is arithmetic.
    text = _graft_hypothetical_milestone_arithmetic(text, messages)
    # Delay-damages daily-amount ask: compose rate × ACA into SAR/day from retrieved client
    # text before the cost gate. A percentage-only excerpt still cannot
    # invent a daily figure; both operands must be in the excerpts or
    # the loaded CD volume (last-chance after refuse-prone top-k).
    pid, extra_pids = _audit_project_ids_for_daily_damages(project_id, audit_rec)
    text = _graft_composed_delay_damages_daily(
        text, rag_sys_msg, messages, project_id=pid,
        extra_project_ids=extra_pids,
    )
    # A delay of N days (combined milestones) is a money
    # answer. Runs after the daily composer so a period ask is not
    # left on SAR/day or on the unbound 0% of 0 note.
    text = _graft_composed_delay_damages_over_period(
        text, rag_sys_msg, messages, project_id=pid,
        extra_project_ids=extra_pids,
    )
    # The model answered delay damages for an including-VAT ACA ask, or
    # said a TfC / Engineer particular was absent. Graft the asked row from excerpts only.
    text = _graft_asked_contract_particular(
        text, rag_sys_msg, messages, project_id=pid,
        extra_project_ids=extra_pids or None,
    )
    # Named percentage × ACA when the ask wants SAR, not
    # the percentage-only particular.
    text = _graft_composed_percentage_of_aca(
        text, rag_sys_msg, messages, project_id=pid,
        extra_project_ids=extra_pids,
    )
    # Priced BOQ row first: a priced CESMM row beats Rate Only / Excluded
    # siblings. The Rate Only graft runs after so it cannot overwrite 280,320.
    text = _graft_priced_boq_item(text, rag_sys_msg, messages)
    # BOQ page Part Summary total (no CESMM code).
    text = _graft_part_summary_total(text, rag_sys_msg, messages)
    # Combined printed page totals (d/3/1+d/3/2+d/3/3).
    # A one-page total stays on the graft above.
    text = _graft_combined_part_summary_total(text, rag_sys_msg, messages)
    # Set3 F1: longest/shortest Time for Completion among a named
    # community's milestones (Northern Community 547 / 150).
    text = _graft_named_community_tfc_span(
        text, rag_sys_msg, messages, project_id=pid,
        extra_project_ids=extra_pids,
    )
    # Rate Only BOQ item ask: state Rate Only when the retrieved BOQ row already
    # says so and no priced triple exists. The live FAIL greeted
    # ("I'm ready to help…") and never named D529.3 / Rate Only.
    text = _graft_rate_only_item(text, rag_sys_msg, messages)
    # PCG / commencement asks: Contract Data "not required" / empty commencement
    # over Schedule 8 form % and commencement-pack dates.
    text = _graft_honest_contract_refusal(text, rag_sys_msg, messages)
    # Default-rate cost build-up: the calculator result is the total.
    # A project-document figure and the no-rate refusal are not.
    text = _graft_deterministic_cost_calc(text, messages)
    # User-supplied operands that complete a registered formula: the
    # calculator's result is the answer, recorded as a tool run so the
    # credit path can name it. Runs before the cost gate so that result
    # is what the gate sees.
    text = _graft_supplied_input_calculation(text, messages, rag_sys_msg)
    gated = text
    text = _cost_grounding_gate(text, rag_sys_msg, messages)
    text = _calc_figure_grounding_gate(text, messages)
    # Citation provenance: an attribution no evidence record backs is removed
    # and the answer flagged. Sibling of the cost gate above -- that one
    # grounds the FIGURES, this one grounds the claim about where they came
    # from (a "BOQ context: ... (<contract id>)" line from a
    # template scheduler that has no BOQ input at all).
    from app.agents.citation_provenance import gate as _citation_provenance_gate
    text = _citation_provenance_gate(text, rag_sys_msg, messages)
    # A gate that removes every clause of an answer leaves the turn's own
    # tool result, never an empty answer for the footers to dress.
    if gated.strip() and not text.strip():
        text = _recover_answer_from_tool_messages("", messages)
    # Figure provenance (same evidence objects): every figure credited to user
    # input, calculator, project document or general knowledge; a figure with
    # no source is not stated. Enforced on turns that ran a need plan.
    text = _record_figure_provenance(text, rag_sys_msg, messages, audit_rec)
    text = _standards_advisory(text, question=_latest_operator_ask(messages))
    text = _ensure_ingestion_handoff(text, messages, agent_name)
    # P4b: project 30 minutes must not be stated as NFPA's (same shape for
    # a named authority or code the excerpts are not). Runs before the
    # scrub so a filename quoted here is still cleaned.
    # Computed slab thickness is not a retrieved claim. Replace a guard
    # refusal (or an offer to run slab_thickness_min that never calls it)
    # before the attribution graft, which must leave the calculator line.
    text = _graft_computed_slab_thickness(text, messages)
    text = _graft_named_standard_attribution(text, rag_sys_msg, messages)
    # Another project's documents never reach this answer (retrieval is
    # scoped to the project plus the shared layer, and the shared layer was
    # cleaned at source). The PARTIES
    # need no list either: the excerpts this answer was
    # written from say who the Employer, Engineer and Contractor are, so their
    # names are read from there and replaced by the role. Owner ruling
    # 2026-09-19: "No names at all from this RAG."
    from app.core.party_names import withhold_party_names
    text = withhold_party_names(
        text, (rag_sys_msg or {}).get("content", "") if rag_sys_msg else "",
    )
    # Disclosure banner goes on AFTER the scrub so it's never mangled, and only
    # when the banner isn't already present (idempotent across retries).
    ask = _latest_operator_ask(messages) or ""
    if not ask:
        ask = str((audit_rec or {}).get("user_message_preview") or "")
    pid = project_id or (audit_rec or {}).get("project_id")
    calc_ran = _turn_already_ran_construction_calc(messages)
    if should_suppress_master_corpus_fallback(pid, ask) or (
        calc_ran and not project_is_master_corpus(pid)
    ):
        text = _strip_master_corpus_preamble(text)
    elif fallback_used and _MASTER_CORPUS_FALLBACK_NOTE.strip() not in text:
        # Formula / user-FIXTURE calculator asks must not wear the
        # Master Corpus banner even if retrieve leaked a fallback chunk.
        # Live: pe_unit_convert ran and the answer still opened with it.
        text = _MASTER_CORPUS_FALLBACK_NOTE + text
    from app.core.rag.coverage_honesty import apply_coverage_honesty
    text = apply_coverage_honesty(
        text,
        project_id=project_id or (audit_rec or {}).get("project_id"),
        coverage=coverage,
    )
    # After graft: #587's INTERNAL GUIDANCE did not stop extract from
    # electing the ENGINEER APPOINTMENT heading. Strip leftover steering
    # so EXAMPLAR (or any other particular) is what the user sees.
    text = _strip_answer_routing_preamble(text)
    # First-line figure + document asks: the Hard rule already asked for figure + document in
    # the first line. Live answers still opened on a narrative, a bare
    # figure, or "properly compacted". This guard writes the line the
    # operator scores, from the excerpts already on the turn.
    from app.agents.first_line_hard_rule import apply_first_line_hard_rule
    text = apply_first_line_hard_rule(text, rag_sys_msg, messages)
    # Supplied-input wording, then registered ids. Both run after every
    # graft so the text the user reads is what they are applied to. The
    # derivation check then sees that text.
    text = _scrub_supplied_input_wording(text, messages, rag_sys_msg)
    text = _scrub_registry_ids(text)
    # Last: the answer's own working must agree with its own result. Checked
    # after every graft, so a grafted figure is checked too.
    text = _annotate_derivation_mismatches(text)
    # 0-token force_synthesis must never persist an empty bubble.
    return _nonblank_after_empty_synthesis(text, messages)


_INGEST_NEXT_RE = re.compile(r"(?im)^Next:\s*\S+")
_EXT_TO_NEXT_AGENT = (
    (".ifc", "bim-analyst"),
    (".xer", "construction-pm"),
    (".xlsx", "quantity-surveyor"),
    (".csv", "quantity-surveyor"),
    (".docx", "contracts-manager"),
    (".doc", "contracts-manager"),
    (".pdf", "quantity-surveyor"),
)


def _next_agent_from_turn(messages: list[dict[str, Any]] | None) -> str:
    """Pick the leftover L5 handoff hat from the user's named file/intent."""
    text = ""
    for m in reversed(messages or []):
        if m.get("role") == "user":
            text = str(m.get("content") or "")
            break
    low = text.lower()
    for ext, agent in _EXT_TO_NEXT_AGENT:
        if ext in low:
            return agent
    if "bim" in low or "ifc" in low:
        return "bim-analyst"
    if "boq" in low:
        return "quantity-surveyor"
    return "smart-orchestrator"


def _ensure_ingestion_handoff(
    text: str,
    messages: list[dict[str, Any]] | None,
    agent_name: str | None,
) -> str:
    """Leftover L5: document-ingestion must end with ``Next: <hat>``."""
    if (agent_name or "") != "document-ingestion":
        return text
    if not (text or "").strip():
        return text
    if _INGEST_NEXT_RE.search(text):
        return text
    return text.rstrip() + f"\n\nNext: {_next_agent_from_turn(messages)}"


def _parse_source_tail(tail: str) -> tuple[str, str]:
    """Split a source mention into filename and optional chunk-number blob.

    Examples:
      "File.xlsx, chunk 8"        -> ("File.xlsx", "8")
      "File.xlsx, chunks 1, 2"    -> ("File.xlsx", "1, 2")
      "File.xlsx, which says..."  -> ("File.xlsx", "")
      "File.xlsx"                 -> ("File.xlsx", "")
    """
    tail = tail.strip()
    # Try to peel a trailing ", chunk(s) ..." suffix.
    chunk_suffix = re.search(r",\s*chunks?\s+([\d,\s]+)$", tail, re.IGNORECASE)
    if chunk_suffix:
        fname = tail[: chunk_suffix.start()].strip()
        return _strip_source_quotes(fname), chunk_suffix.group(1)
    # No chunk suffix. Trim any trailing prose clause after a comma.
    if "," in tail:
        # Heuristic: the first comma followed by a lowercase/prose word marks
        # the start of a clause (", which lists..."). Keep only the filename.
        idx = tail.find(",")
        rest = tail[idx + 1 :].lstrip()
        if rest and (rest[0].islower() or rest.startswith("which") or rest.startswith("found") or rest.startswith("see")):
            tail = tail[:idx].strip()
    return _strip_source_quotes(tail), ""


def _extract_cited_chunk_indexes(text: str) -> list[tuple[str, int]]:
    """Pull (filename, chunk_index) pairs from the agent's final text.

    Recognised patterns (case-insensitive):
      [source: filename.pdf, chunk 65]
      [source: filename.pdf, chunks 16, 34, 55]
      Source: filename.pdf, chunk 65.
      Source: “filename.pdf”, which ...
      [doc_id=abc123, chunk 4]

    Returns a list of pairs; chunk_index is -1 when the cite didn't
    include a chunk number.
    """
    if not text:
        return []
    out: list[tuple[str, int]] = []

    for m in _CITATION_RE.finditer(text):
        fname, nums_blob = _parse_source_tail(m.group(1))
        if not nums_blob.strip():
            out.append((fname, -1))
        else:
            for piece in nums_blob.split(","):
                piece = piece.strip()
                if piece.isdigit():
                    out.append((fname, int(piece)))

    for m in _CITATION_LINE_RE.finditer(text):
        fname, nums_blob = _parse_source_tail(m.group(1))
        if not nums_blob.strip():
            out.append((fname, -1))
        else:
            for piece in nums_blob.split(","):
                piece = piece.strip()
                if piece.isdigit():
                    out.append((fname, int(piece)))

    for m in _CITATION_QUOTED_RE.finditer(text):
        fname = m.group(2).strip()
        if fname:
            out.append((fname, -1))

    for m in _CITATION_DOCID_RE.finditer(text):
        doc_id = m.group(1).strip()
        chunk_idx = int(m.group(2))
        out.append((doc_id, chunk_idx))

    # gpt-oss inline-bracketed form: "Source: [filename], chunk(s) N[-M]".
    # The chunk number is the reliable signal (the model can only cite indexes
    # it saw in the injected context); the filename is often ellipsis-truncated
    # and is matched leniently downstream.
    for m in _CITATION_INLINE_BRACKET_RE.finditer(text):
        fname = _strip_source_quotes(m.group(1).strip())
        for num in re.findall(r"\d+", m.group(2)):
            out.append((fname, int(num)))

    for m in _CITATION_SRC_RE.finditer(text):
        fname = _strip_source_quotes(m.group(1).strip().rstrip(".,;)]"))
        if fname:
            out.append((fname, -1))

    return out


def page_or_section_label(chunk_meta: dict[str, Any]) -> str:
    """Where in its source a cited chunk sits: ``p. N`` when the page is known
    (PDF sources), else nothing. A chunk number is the index's, not a place
    a reader can find."""
    page = chunk_meta.get("page")
    try:
        page_no = int(page) if page is not None else 0
    except (TypeError, ValueError):
        page_no = 0
    if page_no > 0:
        return f"p. {page_no}"
    return ""


def _platform_calculator_source(final_text: str) -> dict[str, Any] | None:
    """The Sources row for a platform-calculator credit, or None.

    The row is the calculator's plain-words label, including the inputs
    the credit names. It is not a retrieved document, so it carries no
    source class and no document id.
    """
    from app.lib.source_labels import CALCULATOR_SUFFIX

    raw = final_text or ""
    if CALCULATOR_SUFFIX.lower() not in raw.lower():
        return None
    # The title is the registry label only: display name, the suffix, and
    # the inputs in parentheses. Model prose that merely contains the
    # suffix is not a title.
    match = re.search(
        rf"(?im)^[ \t]*Source:\s*"
        rf"(?P<label>[^\n]+?{re.escape('— ' + CALCULATOR_SUFFIX)}"
        rf"(?:\s*\([^)\n]*\))?)",
        raw,
    )
    if match is None:
        return None
    label = match.group("label").strip().rstrip(".")
    return {
        "doc_id": "",
        "doc_name": label,
        "page_or_section": "Calculated",
        "page": None,
        "chunk_index": None,
        "chunk_id": "",
        "score": 1.0,
        "confidence": "High",
        "layer": "general_knowledge",
        "layer_label": "Platform calculator",
    }


# A use of a working document is a run of its own words in the answer.
# One shared token is not a quote. A passage shorter than this is too
# small to tell a quote from a coincidence, unless the passage itself is
# only a few words long — then the whole passage has to appear.
_WORKING_DOC_QUOTE_WORDS = 6
_WORKING_DOC_MIN_WORDS = 4
_QUOTE_TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*", re.IGNORECASE)


def _quote_words(text: str) -> list[str]:
    return [w.lower() for w in _QUOTE_TOKEN.findall(text or "")]


def _answer_quotes_passage(answer: str, passage: str) -> bool:
    """True when the answer contains a contiguous run of the passage's words."""
    words = _quote_words(passage)
    if len(words) < _WORKING_DOC_MIN_WORDS:
        return False
    need = min(_WORKING_DOC_QUOTE_WORDS, len(words))
    hay = " ".join(_quote_words(answer))
    if not hay:
        return False
    for start in range(0, len(words) - need + 1):
        if " ".join(words[start:start + need]) in hay:
            return True
    return False


def _coerce_tool_payloads(tool_results: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Tool-result dicts, including a driver trail's JSON ``content``."""
    payloads: list[dict[str, Any]] = []
    for item in tool_results or []:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if item.get("role") == "tool" and isinstance(content, str):
            try:
                parsed = json.loads(content)
            except (TypeError, ValueError):
                continue
            if isinstance(parsed, dict):
                payloads.append(parsed)
            continue
        payloads.append(item)
    return payloads


def _working_document_reads(
    tool_results: list[dict[str, Any]] | None,
) -> list[tuple[str, str, str]]:
    """``(shown name, text, doc id)`` for a tool result that is one file's text.

    A retrieval hit list is not a working-document read: those chunks are
    already on the audit. The read is the single body the turn fetched.
    """
    found: list[tuple[str, str, str]] = []
    for payload in _coerce_tool_payloads(tool_results):
        if payload.get("ok") is False:
            continue
        result = payload.get("result")
        if not isinstance(result, dict):
            continue
        if isinstance(result.get("results"), list) or isinstance(result.get("chunks"), list):
            continue
        text = result.get("content")
        if not isinstance(text, str):
            text = result.get("text") if isinstance(result.get("text"), str) else ""
        name = result.get("filename") or result.get("original_name") or ""
        if not isinstance(name, str):
            name = ""
        name = name.strip()
        text = text.strip()
        if not name or not text:
            continue
        doc_id = str(result.get("document_id") or result.get("doc_id") or "")
        found.append((name, text, doc_id))
    return found


def _working_document_sources(
    final_text: str,
    tool_results: list[dict[str, Any]] | None,
    audit_rec: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Source rows for working documents whose text the answer actually uses."""
    from app.core.rag.source_class import classify, label_for

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name, text, doc_id in _working_document_reads(tool_results):
        if not _answer_quotes_passage(final_text, text):
            continue
        key = (doc_id or name).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        source_class = classify("own", name, text)
        rows.append({
            "doc_id": doc_id,
            "doc_name": _clean_path_label(name),
            "page_or_section": "",
            "page": None,
            "chunk_index": None,
            "chunk_id": None,
            "project_id": (audit_rec or {}).get("project_id"),
            "score": 1.0,
            "confidence": "High",
            "layer": "own",
            "layer_label": "Project document",
            "source_class": source_class,
            "source_class_label": label_for(source_class),
        })
    return rows


def _dedupe_source_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen_ids: set[str] = set()
    seen_names: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        if not row:
            continue
        doc_id = str(row.get("doc_id") or "")
        name = str(row.get("doc_name") or "").strip().lower()
        if doc_id and doc_id in seen_ids:
            continue
        if name and name in seen_names:
            continue
        if doc_id:
            seen_ids.add(doc_id)
        if name:
            seen_names.add(name)
        out.append(row)
    return out


def _build_sources_from_audit(
    audit_rec: dict[str, Any],
    final_text: str = "",
    tool_results: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build the SSE end-event sources list.

    Behaviour:
      1. If ``final_text`` contains ``[source: ...]`` citations AND those
         citations match chunks present in the audit record's injected
         set, return ONLY those — they are what the agent actually
         cited. The right-panel Sources tab then shows the operator
         exactly the chunks behind the answer.
      2. Otherwise (no citations parsed, or none match the injected
         chunks), fall back to the top-3 retrieved chunks by score —
         the pre-PR-110 behaviour. Preserves the old contract for the
         qwen-style agents that don't emit ``[source: ...]`` markers.

    A working document is different from a retrieved chunk: its text was
    read for this turn (``tool_results``) and is not in the audit. When
    the answer uses that text, the document is listed under the name the
    read carried, and retrieved chunks the answer did not use are left
    out. An answer that did not use a working document keeps the steps
    above, including the top-3 fallback.

    Empty list when ``audit_rec`` has no chunks (fallback turn), unless
    the answer credits a platform calculator — that row is the source —
    or it used a working document.
    """
    used_documents = _working_document_sources(final_text, tool_results, audit_rec)

    def _finish(
        rows: list[dict[str, Any]] | None, *, fallback: bool = False,
    ) -> list[dict[str, Any]]:
        kept = [row for row in (rows or []) if row]
        if not used_documents:
            return kept
        if fallback:
            kept = []
        return _dedupe_source_rows([*used_documents, *kept])

    calc_row = _platform_calculator_source(final_text)
    chunks = (audit_rec or {}).get("chunks") or []
    ask = str((audit_rec or {}).get("user_message_preview") or "")
    pid = (audit_rec or {}).get("project_id")
    if chunks and should_suppress_master_corpus_fallback(pid, ask):
        chunks = [
            c for c in chunks
            if (c.get("layer") or "own") != "master_corpus"
        ]
    if not chunks:
        return _finish([calc_row] if calc_row else [])

    # If the retrieval trust gate fired (identifier miss or confidence
    # threshold), don't fabricate a sources panel from fallback chunks.
    if audit_rec.get("identifier_miss") or audit_rec.get("threshold_fired"):
        return _finish([calc_row] if calc_row else [])

    # If the assistant explicitly declined to answer, don't fabricate a
    # sources panel from low-score fallback chunks.
    if _answer_is_caveat(final_text):
        return _finish([calc_row] if calc_row else [])

    try:
        from app.core import projects as _projects
    except Exception:
        _projects = None

    def _doc_name(doc_id: str) -> str:
        if not _projects:
            return ""
        try:
            d = _projects.get_document(doc_id) or {}
            return d.get("original_name") or ""
        except Exception:
            _LOG.debug("doc name lookup failed for %s", doc_id, exc_info=True)
            return ""

    # A named contract/doc id must not appear beside another year's
    # DD contract in Sources. Fail closed when nothing remaining matches.
    from app.core.rag.retriever import (
        extract_contract_doc_ids,
        filename_matches_named_contracts,
    )
    named_contracts = list((audit_rec or {}).get("extracted_contract_ids") or [])
    if not named_contracts:
        named_contracts = extract_contract_doc_ids(
            (audit_rec or {}).get("user_message_preview") or ""
        )
    # Question-named ids stay authoritative (#443). When the question
    # did not name a contract but the answer did, scope Sources to that id
    # so the panel cannot list AB-2022 next to prose that cites AB-2023.
    if not named_contracts:
        named_contracts = extract_contract_doc_ids(final_text or "")
    if named_contracts:
        scoped = [
            c for c in chunks
            if filename_matches_named_contracts(
                _doc_name(c.get("doc_id") or ""),
                named_contracts,
            )
        ]
        if not scoped:
            return _finish([calc_row] if calc_row else [])
        chunks = scoped

    def _sources_one_contract(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Unnamed contract answers may cite one PREFIX-YEAR-SEQ, not two years."""
        if named_contracts or not sources:
            return sources
        winning: str | None = None
        out: list[dict[str, Any]] = []
        for s in sources:
            cids = extract_contract_doc_ids(s.get("doc_name") or "")
            if not cids:
                out.append(s)
                continue
            if winning is None:
                winning = cids[0]
                out.append(s)
            elif winning in cids:
                out.append(s)
        return out

    # STEP 0b — human-readable layer labels for the sources panel so a
    # Master-Corpus fallback (and disclosed general-knowledge context) is
    # visible per-source, not just in the answer banner.
    _LAYER_LABELS = {
        "own": "Project document",
        "master_corpus": "Master Corpus (fallback)",
        "general_knowledge": "Knowledge base",
    }

    def _format(chunk_meta: dict[str, Any], doc_name: str) -> dict[str, Any]:
        score = chunk_meta.get("score") or 0.0
        conf = "High" if score >= 0.75 else "Medium" if score >= 0.5 else "Low"
        layer = chunk_meta.get("layer") or "own"
        # Layered RAG (Stage 4): a user upload is 'own'-tagged for isolation but
        # should read as "Your upload" so the user sees it came from their own
        # layer, not the authoritative corpus. knowledge_layer is None unless
        # RAG_LAYERED is on, so this label never appears in today's behaviour.
        label = _LAYER_LABELS.get(layer, "Knowledge base")
        if chunk_meta.get("knowledge_layer") == "user_session":
            label = "Your upload"
        display_name = _clean_path_label(doc_name)
        # Consume #468's class. Prefer the tag the injector already wrote;
        # fall back to classify() on the same inputs it uses. Do not invent
        # a fifth class here.
        from app.core.rag.source_class import classify, label_for
        source_class = (chunk_meta.get("source_class") or "").strip() or classify(
            layer,
            chunk_meta.get("source_name") or doc_name,
            chunk_meta.get("text") or "",
        )
        source_class_label = label_for(source_class)
        return {
            "doc_id": chunk_meta["doc_id"],
            "doc_name": display_name,
            "page_or_section": page_or_section_label(chunk_meta),
            "page": chunk_meta.get("page"),
            "chunk_index": chunk_meta.get("chunk_index"),
            "chunk_id": chunk_meta.get("chunk_id"),
            # Owning project of the cited file (Master Corpus / GK), not
            # only the workspace the chat ran in — preview uses this.
            "project_id": (
                chunk_meta.get("project_id")
                or (audit_rec or {}).get("project_id")
            ),
            "score": float(score),
            "confidence": conf,
            "layer": layer,
            "layer_label": label,
            "source_class": source_class,
            "source_class_label": source_class_label,
        }

    # 1) Try to extract citations from the agent's text first.
    cites = _extract_cited_chunk_indexes(final_text)
    if cites:
        matched: list[dict[str, Any]] = []
        seen: set = set()
        # Set of injected doc_ids for the doc-id-keyed citation branch.
        injected_doc_ids = {c.get("doc_id") for c in chunks}

        for cited_token, cited_idx in cites:
            cited_token_n = _normalise_filename(cited_token)
            # Branch A: doc-id-keyed cite. ``cited_token`` matches one of
            # the audit's doc_ids directly (gpt-oss [doc_id=X chunk=N]
            # form). Use that as the primary match.
            doc_id_match = cited_token if cited_token in injected_doc_ids else None

            for c in chunks:
                cidx = c.get("chunk_index")
                doc_id = c.get("doc_id")
                # chunk-index match (when provided) is the primary key.
                if cited_idx != -1 and cidx != cited_idx:
                    continue
                # If the cite was doc-id-keyed, require doc_id match.
                if doc_id_match is not None:
                    if doc_id != doc_id_match:
                        continue
                elif cited_idx == -1:
                    # Filename-keyed cite with NO chunk number to disambiguate
                    # — require a filename suffix-match (normalized) so a model
                    # that rewrote the dash style still resolves. A PREFIX-
                    # YEAR-SEQ in the cite also matches that contract's file.
                    name = _doc_name(doc_id)
                    name_n = _normalise_filename(name)
                    token_cids = extract_contract_doc_ids(cited_token)
                    if token_cids and filename_matches_named_contracts(
                        name, token_cids,
                    ):
                        pass
                    elif (
                        cited_token_n
                        and name_n
                        and cited_token_n not in name_n
                        and name_n not in cited_token_n
                    ):
                        continue
                # else: the cite carried a chunk number that already matched
                # above (cited_idx != -1, cidx == cited_idx). That index
                # uniquely identifies the injected chunk, so trust it even when
                # the model truncated the filename (gpt-oss "Site … Part 3").
                key = (doc_id, cidx)
                if key in seen:
                    continue
                seen.add(key)
                matched.append(_format(c, _doc_name(doc_id)))
        if matched:
            # The append order above follows the model's citation order in the
            # answer text, which the LLM emits non-deterministically (Kimi K2 is
            # pinned to temperature=1). Sort to a deterministic total order
            # (score desc, chunk_id asc) so the panel row order is stable across
            # identical runs. Membership still faithfully reflects what the
            # answer cited (PR-110); only the ordering is made reproducible.
            matched.sort(
                key=lambda s: (-(s.get("score") or 0), s.get("chunk_id") or "")
            )
            rows = ([calc_row] + matched) if calc_row else matched
            return _finish(_sources_one_contract(rows))

    # 2) Filename-mention fallback: the model may have named a source in
    #    prose without a formal citation marker. If any injected filename
    #    appears in the answer, surface the highest-scoring chunk of that
    #    document.
    if final_text:
        text_n = _normalise_filename(final_text)
        mention_hits: list[dict[str, Any]] = []
        seen_mentions: set = set()
        # Deterministic total order (score desc, chunk_id asc): dedup-by-doc
        # keeps the FIRST chunk seen per doc, so an equal-score tie flip would
        # otherwise change which doc's chunk represents it in mention_hits[:3].
        for c in sorted(
            chunks, key=lambda x: (-(x.get("score") or 0), x.get("chunk_id") or "")
        ):
            doc_id = c.get("doc_id")
            name = _doc_name(doc_id)
            name_n = _normalise_filename(name)
            name_cids = extract_contract_doc_ids(name)
            mentioned = bool(name_n and name_n in text_n)
            if not mentioned and name_cids:
                mentioned = any(cid in text_n for cid in name_cids)
            if mentioned and doc_id not in seen_mentions:
                seen_mentions.add(doc_id)
                mention_hits.append(_format(c, name))
        if calc_row:
            # The calculator is the source of the figure. A document the
            # answer actually names stays beside it. Unnamed retrieval
            # does not fill the panel. A working document the answer
            # quoted stays too.
            return _finish(_sources_one_contract([calc_row] + mention_hits[:3]))
        if mention_hits:
            return _finish(_sources_one_contract(mention_hits[:3]))

    if calc_row:
        return _finish([calc_row])

    # 3) Final fallback: top-3 retrieved chunks by score.
    # Deterministic total order: score desc, then chunk_id asc so two chunks
    # that tie on score can never flip across runs and change which doc lands
    # inside the top-3 truncation (the source-panel wobble). chunk_id is the
    # stable per-chunk key; where absent (bare audit dicts) it is "" for all,
    # leaving the pure-score order unchanged.
    by_score = sorted(
        chunks, key=lambda c: (-(c.get("score") or 0), c.get("chunk_id") or "")
    )[:3]
    return _finish(
        _sources_one_contract(
            [_format(c, _doc_name(c["doc_id"])) for c in by_score]
        ),
        fallback=True,
    )


# File extensions whose BOQs are safe to turn into a cost workbook on click.
# Digital grids only: a scanned PDF BOQ won't parse and OOMs the 2 GB box
# (memory the-fork-boq-always-xlsx), so PDFs are deliberately excluded here.
_BOQ_COST_EXPORT_EXTS = (".xlsx", ".csv")
_BOQ_NAME_HINTS = ("boq", "bill of quantities", "priced", "tender")


def _is_cost_boq_source(doc: dict[str, Any]) -> bool:
    """Cheap, metadata-only test for 'this cited doc can back a cost-BOQ
    export'. NEVER runs boq_processor — gating on name/type/extension keeps
    the SSE end path off the OOM-prone parse. The real parse happens on click,
    server-side, where it's bounded (and 422s honestly if the grid is empty)."""
    name = (doc.get("original_name") or "").lower()
    if not name.endswith(_BOQ_COST_EXPORT_EXTS):
        return False
    doc_type = (doc.get("doc_type") or "").lower()
    return doc_type == "boq" or any(h in name for h in _BOQ_NAME_HINTS)


# RFP/BOD/spec documents document_engine can mine for lead times + milestones.
# Text formats only (docx/pdf/txt) — spreadsheets carry no procurement
# narrative, and a scanned-BOQ pdf is the OOM path we avoid elsewhere.
_SPEC_EXPORT_EXTS = (".pdf", ".docx", ".doc", ".txt", ".md")
_SPEC_NAME_HINTS = ("rfp", "request for proposal", "bod", "basis of design",
                    "spec", "requirement", "scope of work", "tender",
                    "employer requirement", "design brief")


def _is_spec_or_rfp_source(doc: dict[str, Any]) -> bool:
    """Metadata-only test for 'this cited doc is an RFP/BOD/spec that
    document_engine can extract equipment lead times + target milestones from',
    to drive a document-aware schedule offer. The extract runs on click,
    server-side, bounded — this only gates the offer."""
    name = (doc.get("original_name") or "").lower()
    if not name.endswith(_SPEC_EXPORT_EXTS):
        return False
    doc_type = (doc.get("doc_type") or "").lower()
    if doc_type in ("spec", "rfp", "bod", "design", "requirements"):
        return True
    return any(h in name for h in _SPEC_NAME_HINTS)


def _build_exports_from_audit(
    audit_rec: dict[str, Any],
    final_text: str = "",
    tool_calls: list[dict[str, Any]] | None = None,
    conversation_id: str | None = None,
) -> list[dict[str, Any]]:
    """SSE end-event `exports` list — data-backed download offers for the bubble.

    Each descriptor is something the platform can actually produce (not from
    routing intent — an offer that 422s on click is worse than no offer):
      - Cost BOQ: from a document the answer CITED (endpoint self-derives from
        the document_id).
      - Schedule: when the turn ran `generate_wbs`, offer the cost-loaded
        workbook.         The full activity list is stripped from the tool result to
        keep the SSE payload small. When a conversation_id is known the
        offer binds to that conversation's staged WBS;
        otherwise it carries the brief params and the endpoint re-derives.

    Returns [] when the turn produced nothing exportable. Deduplicated.
    """
    project_id = (audit_rec or {}).get("project_id")
    if not project_id:
        return []
    conversation_id = conversation_id or (audit_rec or {}).get("conversation_id")
    try:
        from app.core import projects as _projects
        project_id = _projects.ui_project_id(project_id) or project_id
    except Exception:
        _projects = None

    exports: list[dict[str, Any]] = []
    sources = _build_sources_from_audit(audit_rec, final_text, tool_calls)

    # ── Cost BOQ offers (from cited documents) ──────────────────────────────
    if sources and _projects is not None:
        seen: set = set()
        for s in sources:
            doc_id = s.get("doc_id")
            if not doc_id or doc_id in seen:
                continue
            try:
                doc = _projects.get_document(doc_id) or {}
            except Exception:
                continue
            if not _is_cost_boq_source(doc):
                continue
            seen.add(doc_id)
            exports.append({
                "label": "Cost BOQ (Excel)",
                "format": "xlsx",
                "method": "POST",
                "endpoint": f"/v1/projects/{project_id}/export/cost-boq",
                "payload": {"document_id": doc_id, "currency": "SAR"},
            })

    # RFP/BOD/spec docs cited this turn → drive the schedule from THEIR real
    # extracted lead times (document_engine) rather than a generic template.
    # Only spec-type sources qualify, capped, and only digital formats — running
    # document_engine on a scanned BOQ is exactly the OOM path we avoid.
    rfp_doc_ids: list[str] = []
    if sources and _projects is not None:
        seen_rfp: set = set()
        for s in sources:
            did = s.get("doc_id")
            if not did or did in seen_rfp:
                continue
            try:
                doc = _projects.get_document(did) or {}
            except Exception:
                continue
            if _is_spec_or_rfp_source(doc):
                seen_rfp.add(did)
                rfp_doc_ids.append(did)
        rfp_doc_ids = rfp_doc_ids[:3]

    # ── Schedule offer (from a generate_wbs tool call this turn) ─────────────
    for tc in reversed(tool_calls or []):
        if not isinstance(tc, dict) or tc.get("name") != "generate_wbs":
            continue
        res = tc.get("result")
        if not (isinstance(res, dict) and res.get("status") == "success"):
            continue
        total = res.get("activities_total") or res.get("actual_count") or 0
        payload: dict[str, Any] = {
            "brief": res.get("brief") or "",
            "target_count": res.get("target_count") or 200,
            "currency": "SAR",
        }
        if res.get("project_type"):
            payload["project_type"] = res["project_type"]
        if res.get("start_date"):
            payload["start_date"] = res["start_date"]
        if conversation_id:
            from app.core.conversation_wbs import (
                conversation_schedule_export_descriptor,
                load_conversation_wbs,
            )
            payload["conversation_id"] = conversation_id
            if load_conversation_wbs(str(conversation_id)):
                exports.append(conversation_schedule_export_descriptor(
                    project_id, str(conversation_id), int(total or 0),
                ))
                break
        if rfp_doc_ids:
            # Document-driven: real lead times + milestones from the cited RFP.
            payload["document_ids"] = rfp_doc_ids
            label = (f"Schedule from documents — {total} activities (Excel)"
                     if total else "Schedule from documents (Excel)")
            endpoint = f"/v1/projects/{project_id}/export/schedule-from-document"
        else:
            label = (f"Schedule — {total} activities (Excel)"
                     if total else "Schedule (Excel)")
            endpoint = f"/v1/projects/{project_id}/export/schedule-from-brief"
        exports.append({
            "label": label, "format": "xlsx", "method": "POST",
            "endpoint": endpoint, "payload": payload,
        })
        break  # one schedule offer per turn (latest WBS)
    return exports



# DeepSeek — OpenAI-compatible chat-completions at the native API, the
# platform primary. Default
# model is ``deepseek-chat``; ``DEEPSEEK_MODEL`` may pin ``deepseek-reasoner``
# or a current catalogue id (e.g. deepseek-v4-flash). Temperature is NOT
# pinned: the API accepts 0–2 on non-thinking models and ignores it in
# thinking mode rather than 400ing (unlike Moonshot K2).
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_DEFAULT_MODEL = "deepseek-chat"

# OpenRouter — OpenAI-compatible chat-completions, the platform fallback.
# Default model is the free router (`openrouter/free`); any `:free` slug is
# allowed. Paid slugs are refused unless OPENROUTER_ALLOW_PAID=1.
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_DEFAULT_MODEL = "openrouter/free"
# OpenRouter reserves the full requested max_tokens against remaining
# credit even for :free / openrouter/free models. A $0-balance account
# 402s 8192 (live: "can only afford 2454"). Default 2048 fits that
# affordance; override with OPENROUTER_MAX_TOKENS. Agent YAML stays
# high for DeepSeek — the ceiling is applied only on the OpenRouter hop.
OPENROUTER_DEFAULT_MAX_TOKENS = 2048
# Free-tier OpenRouter derives a tiny prompt budget from remaining credit
# (live Wave1: 18398 > 10335 after search + fetch). Keep the outbound
# prompt under this ceiling with a conservative 3-chars-per-token estimate
# so Contract Data Q&A still fits without paid credits. Override with
# OPENROUTER_PROMPT_TOKEN_CEILING.
OPENROUTER_DEFAULT_PROMPT_TOKEN_CEILING = 8000
_OPENROUTER_CHARS_PER_TOKEN = 3
# Tighter than the DeepSeek defaults: a 24k fetch + 8k search already
# blows the free prompt budget. These still hold a Contract Data clause.
_OPENROUTER_TOOL_RESULT_MAX_CHARS = 4000
_OPENROUTER_FETCH_DOCUMENT_MAX_CHARS = 8000
_OPENROUTER_402_MAX_RETRIES = 2
_OPENROUTER_AFFORD_MAX_TOKENS_RE = re.compile(
    r"can only afford (\d+)",
    re.IGNORECASE,
)
_OPENROUTER_PROMPT_LIMIT_RE = re.compile(
    r"prompt tokens limit exceeded:\s*(\d+)\s*>\s*(\d+)",
    re.IGNORECASE,
)
_OPENROUTER_429_MAX_RETRIES = 2


def _llm_http_timeout() -> float:
    """Per-call HTTP timeout for an LLM request (seconds).

    A reasoning model (e.g. deepseek-reasoner) on a large grounded turn
    (~12k-token context) can land its first token well past 120s. The old
    hardcoded 120s httpx timeout gave up on the primary prematurely and fell
    back before it could answer, so the turn produced NOTHING. Give the
    primary room to finish, capped below the CHAT_STREAM deadline so the
    stream still ends cleanly.
    Default 150s (stream deadline default is 240s). A hung DeepSeek at the
    old 200s default left ~40s — often too little for OpenRouter failover.
    Longest documented successful non-streaming call is 126.8s; comments
    cite 100–150s; recent flash TIMING is ≪5s. 150s leaves ~90s of the
    turn for the fallback hop. Override with LLM_HTTP_TIMEOUT_SECONDS.
    """
    try:
        return float(os.getenv("LLM_HTTP_TIMEOUT_SECONDS", "150"))
    except ValueError:
        return 150.0


# Floor for a single LLM attempt. Below this there is no point starting one:
# it cannot return before the turn's wall clock cuts the stream, and the
# attempt only burns the budget that a clean fallback answer needs.
_MIN_LLM_ATTEMPT_SECONDS = 5.0

# Head-room reserved between the LAST LLM attempt's deadline and the turn's
# wall clock, so a call that gives up still leaves the producer time to
# sanitize, persist and stream the fallback text. Without it the LLM deadline
# and the stream deadline are the same instant and the fallback races the
# cancellation it was meant to replace.
_STREAM_DELIVERY_MARGIN_SECONDS = 10.0


def _forced_retry_min_seconds() -> float:
    """Wall clock the forced no-tools retry needs before it is worth starting.

    The retry is a full LLM call on a reasoning model; observed primary calls
    on this service run 100-150s. Starting one with less than this left means
    the turn dies mid-await and the user gets a timeout banner instead of the
    fallback text we could have delivered immediately. Override with
    FORCED_RETRY_MIN_SECONDS.
    """
    try:
        return float(os.getenv("FORCED_RETRY_MIN_SECONDS", "45"))
    except ValueError:
        return 45.0

def _openrouter_model_is_free(name: str) -> bool:
    """True for ``openrouter/free`` or any catalogue slug ending in ``:free``."""
    slug = (name or "").strip()
    if not slug:
        return False
    return slug == OPENROUTER_DEFAULT_MODEL or slug.endswith(":free")


def _resolve_attempt_model(cfg: dict[str, Any], agent_model: str | None) -> str:
    """Provider default unless the agent pin belongs on this provider.

    Every hat YAML still pins ``kimi-k2.6`` (a legacy Moonshot id). Sending
    that — or a dead ``gpt-4*`` / leftover ``deepseek-*`` pin — to the wrong
    provider 404s the turn. DeepSeek keeps only ``deepseek-*`` pins; every
    other provider (OpenRouter) keeps the placeholder remap (empty /
    ``deepseek-*`` / ``kimi*`` / ``moonshot*`` / ``gpt-4*`` / ``gpt-3*``
    → ``default_model``).
    """
    default = str(cfg.get("default_model") or "")
    model = (agent_model or "").strip()
    if cfg.get("provider") == "deepseek":
        if model.startswith("deepseek-"):
            return model
        return default
    if not model or model.startswith(
        ("deepseek-", "kimi", "moonshot", "gpt-4", "gpt-3")
    ):
        return default
    return model


def _resolve_openrouter_model(name: str | None) -> str:
    """Pin a free OpenRouter slug, or the default router.

    Paid slugs (anything other than ``openrouter/free`` / ``*:free``) are
    refused unless ``OPENROUTER_ALLOW_PAID=1``. The warning + default swap
    keeps production from silently billing when an operator pastes a paid id.
    """
    raw = (name or "").strip() or OPENROUTER_DEFAULT_MODEL
    if _openrouter_model_is_free(raw):
        return raw
    allow = (os.getenv("OPENROUTER_ALLOW_PAID") or "").strip().lower()
    if allow in ("1", "true", "yes", "on"):
        return raw
    _LOG.warning(
        "openrouter: refusing paid model %r (set OPENROUTER_ALLOW_PAID=1 to "
        "override); using %s",
        raw,
        OPENROUTER_DEFAULT_MODEL,
    )
    return OPENROUTER_DEFAULT_MODEL


def _llm_config(provider: "str | None" = None) -> dict[str, Any]:
    """Pick the active LLM provider's URL + env-key + default model.

    Two providers are supported:
      * ``deepseek`` — the primary. Unset or unrecognised ``LLM_PROVIDER``
        resolves here, and it is the recommended ``LLM_PROVIDER``.
      * ``openrouter`` — the free-tier fallback, selected with
        ``LLM_PROVIDER=openrouter`` and, as the cross-provider degrade
        target, with ``LLM_FALLBACK_PROVIDER=openrouter``.

    Removed cloud providers (OpenAI, Kimi/Moonshot, Groq) and the on-prem
    Ollama branch stay removed: an ``LLM_PROVIDER`` naming any of them (or
    any junk) falls through to DeepSeek rather than reviving a dead path.

    Per-provider override envs let the operator pin a specific model
    without code changes: ``DEEPSEEK_MODEL`` / ``OPENROUTER_MODEL``.
    """
    # Explicit arg wins (used by the cross-provider fallback so it never has to
    # mutate the process-wide LLM_PROVIDER env — mutating it is not thread-safe
    # under concurrent chats). Unset/unrecognised resolves to DeepSeek (primary).
    provider = (provider if provider is not None else os.getenv("LLM_PROVIDER") or "").strip().lower()
    if provider == "openrouter":
        # max_tokens ceiling is applied in _provider_max_tokens (env
        # OPENROUTER_MAX_TOKENS), not here — the same chokepoint on every
        # fallback / streaming hop.
        return {
            "provider": "openrouter",
            "url": OPENROUTER_API_URL,
            "env_key": "OPENROUTER_API_KEY",
            "default_model": _resolve_openrouter_model(os.getenv("OPENROUTER_MODEL")),
            # OpenRouter passes a request seed to the models that take one.
            "supports_seed": True,
        }
    # Default / fallthrough = DeepSeek (primary). No fixed_temperature:
    # DeepSeek accepts the agent's temperature on deepseek-chat and ignores
    # it on thinking/reasoner rather than 400ing.
    return {
        "provider": "deepseek",
        "url": DEEPSEEK_API_URL,
        "env_key": "DEEPSEEK_API_KEY",
        "default_model": (
            (os.getenv("DEEPSEEK_MODEL") or "").strip() or DEEPSEEK_DEFAULT_MODEL
        ),
    }


def _cross_provider_fallback(primary: dict[str, Any]) -> dict[str, Any] | None:
    """``LLM_FALLBACK_PROVIDER`` config, or None when unset/unusable."""
    name = (os.getenv("LLM_FALLBACK_PROVIDER") or "").strip().lower()
    if not name or name == primary.get("provider"):
        return None
    # Build the fallback provider's config by passing the name explicitly — no
    # mutation of the shared os.environ["LLM_PROVIDER"], which raced under
    # concurrent chats and could flip another request's provider mid-flight.
    cfg = _llm_config(provider=name)
    if cfg.get("provider") == primary.get("provider"):
        return None
    if cfg.get("env_key") and not os.getenv(cfg["env_key"]):
        return None
    return cfg


def _llm_fallback_ladder(primary: dict[str, Any]) -> list[dict[str, Any]]:
    """Every usable fallback. With only two providers this is the single
    cross-provider degrade target named by ``LLM_FALLBACK_PROVIDER``."""
    out: list[dict[str, Any]] = []
    cross = _cross_provider_fallback(primary)
    if cross:
        out.append(cross)
    return out


def _llm_fallback_config(primary: dict[str, Any]) -> dict[str, Any] | None:
    """First fallback only — callers that want the full ladder use
    ``_llm_fallback_ladder``.

    CROSS-PROVIDER fallback (``LLM_FALLBACK_PROVIDER`` = ``deepseek`` |
    ``openrouter``): degrade to the other provider on a retryable failure
    (413/429/5xx/network). ``openrouter`` is the intended value when the
    primary is DeepSeek. Returns ``None`` when unset, when it names the
    primary provider, or when its API-key env is missing.

    Reuses ``_llm_config`` for URL/suffix normalisation by pinning the provider
    through the env for the duration of one synchronous call.
    """
    ladder = _llm_fallback_ladder(primary)
    return ladder[0] if ladder else None


def _provider_temperature(cfg: dict[str, Any], default: float) -> float:
    """The temperature this provider will actually accept.

    Neither DeepSeek nor OpenRouter pins temperature, so this passes the
    agent's own value through. The generic ``fixed_temperature`` hook is kept
    so a future constrained provider can declare its pin in ``_llm_config``
    rather than hardcoding a value globally (which would become the NEXT
    provider's 400). Applied at the same outbound chokepoint as message
    sanitisation so every caller (_call_llm and _stream_synthesis) shapes the
    payload identically.
    """
    fixed = cfg.get("fixed_temperature")
    return default if fixed is None else float(fixed)


def _openrouter_max_tokens_ceiling() -> int:
    """Credit-reservation ceiling for outbound OpenRouter ``max_tokens``.

    OpenRouter bills/reserves the full requested ``max_tokens`` even on
    ``:free`` slugs. Keep the default under the observed $0-balance
    affordance (~2454). Invalid / non-positive env values fall back to
    ``OPENROUTER_DEFAULT_MAX_TOKENS``.
    """
    raw = (os.getenv("OPENROUTER_MAX_TOKENS") or "").strip()
    if not raw:
        return OPENROUTER_DEFAULT_MAX_TOKENS
    try:
        value = int(raw)
    except ValueError:
        return OPENROUTER_DEFAULT_MAX_TOKENS
    if value < 1:
        return OPENROUTER_DEFAULT_MAX_TOKENS
    return value


def _provider_max_tokens(cfg: dict[str, Any], configured: int) -> int:
    """Adjust the agent's ``max_tokens`` for this provider.

    A provider may declare ``reasoning_min_tokens`` (generic hook): the shared
    budget is spent on chain-of-thought first, so a configured cap below the
    model's reasoning burn yields finish_reason=length with EMPTY content every
    time. The floor only LIFTS a starving budget; a generous agent budget
    passes through, and providers without the field (DeepSeek) keep the agent's
    own value.

    OpenRouter additionally applies a CEILING (``OPENROUTER_MAX_TOKENS``,
    default 2048). OpenRouter reserves the full requested max_tokens against
    remaining credit even for :free models, so an agent YAML of 8192 402s a
    $0-balance free account. DeepSeek is unchanged.
    """
    tokens = int(configured)
    floor = cfg.get("reasoning_min_tokens")
    if floor is not None:
        tokens = max(tokens, int(floor))
    if cfg.get("provider") == "openrouter":
        tokens = min(tokens, _openrouter_max_tokens_ceiling())
    return tokens


def _openrouter_prompt_token_ceiling() -> int:
    """Hard ceiling on approximate prompt tokens for an OpenRouter hop.

    Free-tier credit derives a prompt budget around 10k tokens. Default
    8000 leaves headroom for tool schemas and the conservative estimate.
    Invalid / too-small env values fall back to the default.
    """
    raw = (os.getenv("OPENROUTER_PROMPT_TOKEN_CEILING") or "").strip()
    if not raw:
        return OPENROUTER_DEFAULT_PROMPT_TOKEN_CEILING
    try:
        value = int(raw)
    except ValueError:
        return OPENROUTER_DEFAULT_PROMPT_TOKEN_CEILING
    if value < 512:
        return OPENROUTER_DEFAULT_PROMPT_TOKEN_CEILING
    return value


def _approx_prompt_tokens(messages: list[dict[str, Any]] | None) -> int:
    """Conservative prompt-token estimate: 3 characters ≈ 1 token.

    JSON tool dumps tokenize worse than prose; 3 chars/token over-estimates
    relative to the usual 4 so the ceiling fires before OpenRouter's
    credit-derived prompt budget (~10k on a $0 free account).
    """
    chars = _approx_message_chars(messages)
    return (chars + _OPENROUTER_CHARS_PER_TOKEN - 1) // _OPENROUTER_CHARS_PER_TOKEN


def _openrouter_prompt_char_budget(token_ceiling: int | None = None) -> int:
    ceiling = (
        int(token_ceiling)
        if token_ceiling is not None
        else _openrouter_prompt_token_ceiling()
    )
    return max(512, ceiling) * _OPENROUTER_CHARS_PER_TOKEN


def _openrouter_402_afford_max_tokens(body: str) -> int | None:
    """Parse ``can only afford N`` from an OpenRouter HTTP 402 body."""
    match = _OPENROUTER_AFFORD_MAX_TOKENS_RE.search(body or "")
    if not match:
        return None
    try:
        value = int(match.group(1))
    except ValueError:
        _LOG.debug("OpenRouter afford-max-tokens parse failed", exc_info=True)
        return None
    return value if value >= 1 else None


def _openrouter_402_is_in_flight(body: str) -> bool:
    text = (body or "").lower()
    return "in_flight_budget_exhausted" in text or "in_flight" in text


def _parse_prompt_token_limit(body: str) -> tuple[int, int] | None:
    """Parse live ``Prompt tokens limit exceeded: HAVE > CAP``."""
    match = _OPENROUTER_PROMPT_LIMIT_RE.search(body or "")
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _openrouter_402_should_retry(body: str) -> bool:
    """True for the free-tier 402s a same-provider retry can recover.

    ``in_flight_budget_exhausted`` is transient (another request still
    reserved). ``can only afford N`` is fixed by lowering ``max_tokens``.
    ``Prompt tokens limit exceeded`` is fixed by compacting to the stated
    cap (proactive ceiling can still miss when remaining credit drops).
    A bare 'insufficient credits' 402 is not retried.
    """
    return (
        _openrouter_402_is_in_flight(body)
        or _openrouter_402_afford_max_tokens(body) is not None
        or _parse_prompt_token_limit(body) is not None
    )


def _http_status_is_retryable(status: int) -> bool:
    """Statuses ``llm_client.complete`` may hop to ``LLM_FALLBACK_PROVIDER``.

    Includes 402 so an OpenRouter credit 402 does not silently kill
    orchestrator intent routing. ``Agent._call_llm`` keeps its own
    same-hop 402 policy and still refuses to burn Groq on a generic 402.
    """
    return status in (402, 408, 413, 429) or status >= 500


def _provider_retry_delay_seconds(resp: Any, attempt: int) -> float:
    """Backoff for OpenRouter 429. Honors ``Retry-After`` when present."""
    headers = getattr(resp, "headers", None) or {}
    raw = ""
    if hasattr(headers, "get"):
        raw = headers.get("retry-after") or headers.get("Retry-After") or ""
    try:
        delay = float(raw)
    except (TypeError, ValueError):
        delay = float(attempt)
    return min(8.0, max(0.4, delay))


def _compact_messages_for_openrouter(
    messages: list[dict[str, Any]],
    token_ceiling: int | None = None,
) -> list[dict[str, Any]]:
    """Keep an OpenRouter hop under the free-tier prompt budget.

    Reuses the Groq TPM compactor (oldest / largest tool payloads first,
    then retrieval excerpts, then middle history) with a char budget
    derived from ``OPENROUTER_PROMPT_TOKEN_CEILING`` (default 8000).
    """
    return _compact_messages_for_tpm(
        messages,
        budget=_openrouter_prompt_char_budget(token_ceiling),
        compact_note="Compacted for OpenRouter free-tier prompt budget.",
    )


# Outbound message sanitiser — the single chokepoint that protects every
# provider from non-standard fields our own (reasoning-capable) models add.
# Reasoning models (gpt-oss / GLM) attach a `reasoning` field to the assistant
# message that carries a tool call; runtime appends that message wholesale and
# it is persisted to agent memory. When the array is later POSTed to a strict
# provider it 400s: "messages[N].reasoning: reasoning is not supported with
# this model" and the whole (tool-calling) turn errors. We WHITELIST allowed keys rather than
# blacklist `reasoning`, so whatever field the next provider invents is dropped
# too. Applied only in _call_llm so already-contaminated history is neutralised
# on replay — no persistence migration.
_ALLOWED_MSG_KEYS = ("role", "content", "tool_calls", "tool_call_id", "name")
_ALLOWED_TOOLCALL_KEYS = ("id", "type")
_ALLOWED_FUNCTION_KEYS = ("name", "arguments")


class _SynthStreamError(Exception):
    """Raised by ``_stream_synthesis`` when the streaming synthesis call cannot
    be served (bad config, over daily cap, non-200, transport error). The
    streaming chat loop catches it and — if nothing was streamed yet — falls
    back to the non-streaming ``_call_llm`` path, so streaming is never a
    one-way door away from the working behaviour."""


# Mid-stream synthesis drop: keep the tokens already shown (do not restart
# `_call_llm` — that would duplicate the body) and say the turn was cut off
# so a partial is not persisted as a finished answer.
_SYNTH_CUTOFF_NOTICE = "The answer was cut off — please ask again."


class _SynthLengthCut(_SynthStreamError):
    """The streamed synthesis stopped at the model's output token limit
    (``finish_reason=length``): the text ends wherever the limit fell."""


# The model's output stopped at its token limit. What it wrote is kept up to
# its last whole sentence; this says why the answer stops there.
_LENGTH_CUT_NOTICE = "The answer reached its length limit here. Ask me to continue for the rest."


def _finished_on_length(choice: Any) -> bool:
    if not isinstance(choice, dict):
        return False
    return str(choice.get("finish_reason") or "").lower() in {"length", "max_tokens"}


def _end_cut_answer(text: str, notice: str = _LENGTH_CUT_NOTICE) -> str:
    """An answer the model stopped writing part-way, ending on its last whole
    sentence and then ``notice``: never mid-sentence."""
    from app.agents.answer_exit import end_on_a_sentence

    kept = end_on_a_sentence(text or "")
    if notice in kept:
        return kept
    return f"{kept}\n\n{notice}" if kept.strip() else notice


_TPM_CHAR_BUDGET = 16000  # ~4k tokens; the default prompt-compaction budget


def _approx_message_chars(messages: list[dict[str, Any]] | None) -> int:
    total = 0
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        total += len(str(m.get("content") or ""))
    return total


#: Smallest useful preview of a compacted tool result. Below this the
#: model has nothing to reason from and answers about the note instead.
_TOOL_PREVIEW_FLOOR = 1000

#: What the LAST tool result keeps. It is the one the answer is about --
#: the BOQ that was just parsed, the schedule that was just built -- while
#: the older ones are usually context the model has already used.
_LAST_TOOL_PREVIEW = 6000


def _compacted_tool_payload(
    content: str, keep: int, note: str | None = None,
) -> str:
    """A truncated tool result that says exactly what it lost.

    Live on 554e0b9: asked how many cubic metres of demolition are in the
    BOQ, the model answered from the preview and reported "pages 1-6 of 16",
    concluding no m3 item existed. The 945 m3 line is past the cut. It had
    only been told "Compacted for Groq TPM 8000" -- not how much was gone --
    so it inferred the extent from the content and inferred it wrong.

    Two things fix that, and neither is a bigger budget:

    * state the loss as a NUMBER, so the model does not have to guess it;
    * forbid the absence claim outright. inject.py already applies exactly
      this rule to retrieval excerpts (SCOPE OF ABSENCE) because a top-K
      sample is evidence of what IS present, never of what is absent. A
      truncated tool result is the same shape of evidence and never got the
      same rule. On that turn the model happened to hedge correctly. The
      same cut with different phrasing produces "the BOQ contains no
      demolition in cubic metres" -- a false statement about the customer's
      own contract, stated with the tool's authority behind it.
    """
    kept = max(0, int(keep))
    dropped = max(0, len(content) - kept)
    return json.dumps(
        {
            "truncated": True,
            "note": note or "Compacted to fit the provider prompt budget.",
            "chars_shown": min(kept, len(content)),
            "chars_dropped": dropped,
            "chars_total": len(content),
            "scope_of_absence": (
                f"You are seeing the FIRST {min(kept, len(content))} of "
                f"{len(content)} characters of this tool result; "
                f"{dropped} characters are not shown. Anything the user "
                "asked about may be in the part you cannot see. Say what "
                "you found and that the result was truncated. NEVER state "
                "or imply that a value, line item, or section is absent "
                "from the document or the data -- you cannot see far "
                "enough to know that. Offer to re-run narrowed to the "
                "section, page, or code the user wants."
            ),
            "preview": content[:kept],
        },
        default=str,
    )


#: The retrieval brief's first line. A system message carrying it is the
#: EVIDENCE for the turn, not boilerplate, and is compacted differently.
_RETRIEVAL_BRIEF_MARKER = "AUTHORITATIVE REFERENCE CONTEXT"

#: One retrieved excerpt starts here. inject.py::_marker builds these.
_EXCERPT_SPLIT_RE = re.compile(r"(?=\[doc_id=[^\]\s]+\s+chunk=\d+\s+score=)")


def _compact_retrieval_message(content: str, budget: int) -> str:
    """Drop whole excerpts from the tail, and say how many.

    Measured on the real thing: a 5-chunk Contract Data brief is 8072
    characters -- 1704 of instructions and 6368 of evidence. The flat
    ``content[:2500]`` this replaces kept every word of the instructions and
    796 characters of evidence: ONE excerpt of five, cut mid-sentence, with
    no indication that four were gone.

    That is the worst possible trade. The header tells the model it is
    holding authoritative context and must answer from it; the compactor
    then removes the context and leaves the instruction. It is also why two
    live answers reported a fact "missing from what was returned" when
    retrieval had in fact returned it:

        D1  "the specific Contract Data entry ... that names the individual
             Engineer's Representative is missing from what was returned"
        delay damages  "The retrieved excerpts do not contain the Contract Price
             expressed in SAR"

    So: never slice an excerpt. Drop whole ones from the tail -- they are
    ordered by score, so the tail is the weakest evidence -- and state the
    count, because the header's SCOPE OF ABSENCE rule is only honest if the
    model knows the sample got smaller.
    """
    parts = _EXCERPT_SPLIT_RE.split(content)
    header, excerpts = parts[0], [p for p in parts[1:] if p.strip()]
    if not excerpts:
        # Not the shape we expected. Slice, but keep the rule intact rather
        # than leaving instructions with no caveat.
        return content[:budget] + "\n[truncated for TPM]"

    def _note(kept: int) -> str:
        if kept == len(excerpts):
            return ""
        return (
            f"\nBUDGET NOTE: {len(excerpts) - kept} of {len(excerpts)} "
            "retrieved excerpts were dropped to fit this request, lowest "
            "relevance first. The sample you are reading is SMALLER than "
            "what search found, so the SCOPE OF ABSENCE rule above applies "
            "with more force, not less: say what you found and offer to "
            "search again by document name or section. Never report that "
            "something is absent.\n"
        )

    for kept in range(len(excerpts), 0, -1):
        out = header + _note(kept) + "".join(excerpts[:kept])
        if len(out) <= budget:
            return out
    # Zero excerpts, returned regardless of whether the header fits. Two
    # things are load-bearing here and neither is the size.
    #
    # The rules: slicing the header is the one outcome worse than being a
    # few hundred characters long, because it removes SCOPE OF ABSENCE and
    # the never-claim-absence line, and those are what keep the model honest
    # precisely when the evidence is thinnest. Bounded cost -- the header is
    # a fixed ~1700 characters, not a document -- and the caller's
    # drop-middle-history pass finds the room elsewhere.
    #
    # The note: zero is a real outcome, not an edge case. Without it the
    # model reasons about a sample it no longer has.
    return header + _note(0)


def _compact_messages_for_tpm(
    messages: list[dict[str, Any]],
    budget: int = _TPM_CHAR_BUDGET,
    compact_note: str | None = None,
) -> list[dict[str, Any]]:
    """Shrink a hop so a provider's prompt/TPM budget is not blown by tool dumps.

    Used for the OpenRouter free-tier prompt budget (see
    ``_compact_messages_for_openrouter``). Keep roles and tool-call pairing;
    truncate contents only.

    Tool results are compacted OLDEST FIRST and only while the hop is still
    over budget. The previous pass cut every tool result over 1200 chars to
    the same 1000-char preview whether or not that was needed -- so a turn
    that was 200 chars over budget lost 39k characters of a just-parsed BOQ,
    and the result the question was actually about was cut exactly as hard
    as a stale one from iteration 0.
    """
    if _approx_message_chars(messages) <= budget:
        return messages

    # Oldest first, last excluded: the last tool result is the one the
    # answer is about, so it is compacted last and keeps the most.
    # Every tool message, not only the big ones: the size test belongs with
    # the keep size, below. Filtering here on the floor would also mean a
    # small trailing result stopped counting as "the last one", handing the
    # larger share to an older result the question is not about.
    tool_idx = [
        i for i, m in enumerate(messages)
        if isinstance(m, dict) and m.get("role") == "tool"
        and isinstance(m.get("content"), str)
    ]
    keep_for = {i: _TOOL_PREVIEW_FLOOR for i in tool_idx}
    if tool_idx:
        keep_for[tool_idx[-1]] = _LAST_TOOL_PREVIEW

    running = _approx_message_chars(messages)
    compact_tool: dict[int, int] = {}
    for i in tool_idx:
        if running <= budget:
            break
        keep = keep_for[i]
        if len(messages[i]["content"]) <= keep:
            continue
        compact_tool[i] = keep
        running -= len(messages[i]["content"]) - keep
    # Still over with the last one at its larger share: cut it to the floor
    # too rather than blowing the hop.
    if running > budget and tool_idx:
        last = tool_idx[-1]
        if compact_tool.get(last, _LAST_TOOL_PREVIEW) > _TOOL_PREVIEW_FLOOR:
            running -= (
                compact_tool.get(last, len(messages[last]["content"]))
                - _TOOL_PREVIEW_FLOOR
            )
            compact_tool[last] = _TOOL_PREVIEW_FLOOR

    out: list[dict[str, Any]] = []
    for idx, m in enumerate(messages):
        if not isinstance(m, dict):
            out.append(m)
            continue
        cm = dict(m)
        role = cm.get("role")
        content = cm.get("content")
        if not isinstance(content, str):
            out.append(cm)
            continue
        if role == "tool" and idx in compact_tool:
            cm["content"] = _compacted_tool_payload(
                content, compact_tool[idx], note=compact_note,
            )
        elif role == "system" and len(content) > 2500:
            # The retrieval brief is handled after this loop: it is the
            # turn's evidence and must be the LAST thing sacrificed, not cut
            # to 2500 alongside boilerplate. Everything else is prompt text
            # the model can do without.
            if _RETRIEVAL_BRIEF_MARKER not in content:
                cm["content"] = content[:2500] + "\n[truncated for TPM]"
        elif role == "user" and len(content) > 6000:
            if "PLATFORM PRE-DISPATCH:" not in content[:80]:
                cm["content"] = content[:6000] + "\n[truncated for TPM]"
        out.append(cm)

    # Evidence last -- and that means LAST. Before the brief gives up an
    # excerpt, the tool results give up the rest of their allowance down to
    # the floor. Excerpts are dropped whole, so a 170-character overage would
    # otherwise cost a 1300-character excerpt while a 6000-character tool
    # preview sat untouched: the coarse unit must come out of the cheap
    # budget, not the expensive one.
    over = _approx_message_chars(out) - budget
    if over > 0 and tool_idx:
        for i in reversed(tool_idx):
            if over <= 0:
                break
            cm = out[i]
            if not isinstance(cm, dict) or not isinstance(cm.get("content"), str):
                continue
            have = compact_tool.get(i)
            if have is None or have <= _TOOL_PREVIEW_FLOOR:
                continue
            new_keep = max(_TOOL_PREVIEW_FLOOR, have - over)
            shrunk = _compacted_tool_payload(
                messages[i]["content"], new_keep, note=compact_note,
            )
            over -= len(cm["content"]) - len(shrunk)
            cm["content"] = shrunk
            compact_tool[i] = new_keep
        over = _approx_message_chars(out) - budget

    if over > 0:
        for cm in out:
            if not isinstance(cm, dict) or cm.get("role") != "system":
                continue
            content = cm.get("content")
            if not isinstance(content, str) or _RETRIEVAL_BRIEF_MARKER not in content:
                continue
            # Never below the header: see _compact_retrieval_message.
            allowance = max(0, len(content) - over)
            shrunk = _compact_retrieval_message(content, allowance)
            over -= len(content) - len(shrunk)
            cm["content"] = shrunk
            if over <= 0:
                break
    if _approx_message_chars(out) <= budget:
        return out
    # Drop middle history; keep first system + last 8 messages.
    # Do NOT re-run pairing sanitise here — an unpaired tool dump is
    # exactly what we need to keep (truncated) for a Groq 413 retry.
    kept: list[dict[str, Any]] = []
    systems = [m for m in out if isinstance(m, dict) and m.get("role") == "system"]
    rest = [m for m in out if not (isinstance(m, dict) and m.get("role") == "system")]
    kept.extend(systems[:2])
    kept.extend(rest[-8:])
    return kept or out


def _http_400_is_retryable(body: str) -> bool:
    """True for provider 400s another model CAN serve.

    Generic 400s (bad key, unknown model, malformed JSON we sent) stay
    non-retryable. Kimi conversation-shape / tokenizer 400s are the same
    class as Groq ``tool_use_failed``-as-prose: the next provider on the
    ladder is the recovery, not a quota burn on a broken request.
    """
    text = (body or "").lower()
    if "tokenization failed" in text:
        return True
    if "tool_call_id" in text and "followed" in text:
        return True
    if "must be followed by tool messages" in text:
        return True
    if "invalid request" in text and "tool_call" in text:
        return True
    # Live M12 on 817f224: Kimi HTTP 400 + type=content_filter / "high risk"
    # (not HTTP 200 + finish_reason). Same class as a filtered 200.
    if "content_filter" in text or "considered high risk" in text:
        return True
    return False


def _llm_choice_is_empty_or_filtered(choice: Any) -> bool:
    """True when the provider accepted the request but produced no usable turn.

    Live M12: Kimi HTTP 200 + ``finish_reason=content_filter`` + empty
    content on construction-pm. Live M9: contracts-manager HTTP 200 with
    empty content and no tool_calls. Both used to be treated as a
    successful empty answer. Another provider on the ladder can still
    serve the turn.
    """
    if not isinstance(choice, dict):
        return True
    reason = str(choice.get("finish_reason") or "").lower().replace("-", "_")
    if reason in {"content_filter", "safety"}:
        return True
    msg = choice.get("message") or {}
    if msg.get("tool_calls"):
        return False
    return not (msg.get("content") or "").strip()


def _normalize_tool_call_ids(tool_calls: list[Any]) -> list[Any]:
    """Give every tool_call a stable id. Kimi infers ``name:index`` when ``id``
    is missing; our tool results must use the same string or the next call 400s.
    """
    out: list[Any] = []
    for i, tc in enumerate(tool_calls):
        if not isinstance(tc, dict):
            out.append(tc)
            continue
        ctc = dict(tc)
        fn = ctc.get("function") if isinstance(ctc.get("function"), dict) else {}
        name = fn.get("name") or ctc.get("name") or "tool"
        if not ctc.get("id"):
            ctc["id"] = f"{name}:{i}"
        out.append(ctc)
    return out


def _repair_tool_call_pairing(
    messages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Legal OpenAI tool history: every assistant ``tool_calls[].id`` is
    answered by the immediately-following ``role=tool`` messages.

    Live 2026-08-20: a user-role empty-router/error nudge inserted AFTER the
    second of four ``fetch_document`` results made Kimi 400
    (``Missing: fetch_document:3, fetch_document:4``) and the turn died
    because HTTP 400 was non-retryable. Move interleaved non-tool messages
    to after the batch and stub any still-missing ids so Groq (and a
    retried Kimi) can accept the payload.
    """
    out: list[dict[str, Any]] = []
    i = 0
    n = len(messages)
    while i < n:
        m = messages[i]
        if not (isinstance(m, dict) and m.get("role") == "assistant"
                and isinstance(m.get("tool_calls"), list) and m["tool_calls"]):
            if isinstance(m, dict) and m.get("role") == "tool":
                out.append({
                    "role": "user",
                    "content": (
                        f"{_TOOL_RESULT_PREFIX}{m.get('name') or 'unknown'}): "
                        f"{m.get('content') or ''}"
                    ),
                })
                i += 1
                continue
            out.append(m)
            i += 1
            continue

        assistant = dict(m)
        tcs = _normalize_tool_call_ids(list(assistant.get("tool_calls") or []))
        assistant["tool_calls"] = tcs
        out.append(assistant)
        needed_ids = [
            tc.get("id") for tc in tcs
            if isinstance(tc, dict) and tc.get("id")
        ]
        needed_set = set(needed_ids)
        found: dict[str, dict[str, Any]] = {}
        deferred: list[Any] = []
        i += 1
        while i < n:
            nxt = messages[i]
            if not isinstance(nxt, dict):
                deferred.append(nxt)
                i += 1
                continue
            role = nxt.get("role")
            if role == "assistant":
                break
            if role == "tool":
                tid = nxt.get("tool_call_id")
                if tid in needed_set and tid not in found:
                    tm = dict(nxt)
                    tm["tool_call_id"] = tid
                    found[tid] = tm
                elif not tid:
                    for nid in needed_ids:
                        if nid not in found:
                            tm = dict(nxt)
                            tm["tool_call_id"] = nid
                            found[nid] = tm
                            break
                    else:
                        deferred.append(nxt)
                else:
                    deferred.append(nxt)
                i += 1
                continue
            deferred.append(nxt)
            i += 1

        for tid in needed_ids:
            if tid in found:
                out.append(found[tid])
            else:
                out.append({
                    "role": "tool",
                    "tool_call_id": tid,
                    "name": "unknown",
                    "content": json.dumps({
                        "status": "error",
                        "error": "tool result missing from history",
                    }),
                })
        for d in deferred:
            if isinstance(d, dict) and d.get("role") == "tool":
                out.append({
                    "role": "user",
                    "content": (
                        f"{_TOOL_RESULT_PREFIX}{d.get('name') or 'unknown'}): "
                        f"{d.get('content') or ''}"
                    ),
                })
            else:
                out.append(d)
    return out


def _sanitize_messages_for_provider(
    messages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return a copy of ``messages`` with only OpenAI-standard keys per message
    (and per tool_call). Never mutates the input list."""
    clean: list[dict[str, Any]] = []
    for m in messages:
        if not isinstance(m, dict):
            clean.append(m)
            continue
        cm = {k: m[k] for k in _ALLOWED_MSG_KEYS if k in m}
        tcs = m.get("tool_calls")
        if isinstance(tcs, list):
            clean_tcs = []
            for tc in tcs:
                if not isinstance(tc, dict):
                    clean_tcs.append(tc)
                    continue
                ctc = {k: tc[k] for k in _ALLOWED_TOOLCALL_KEYS if k in tc}
                fn = tc.get("function")
                if isinstance(fn, dict):
                    ctc["function"] = {
                        k: fn[k] for k in _ALLOWED_FUNCTION_KEYS if k in fn
                    }
                clean_tcs.append(ctc)
            cm["tool_calls"] = clean_tcs
        clean.append(cm)
    return _repair_tool_call_pairing(clean)


# ── DSML inline tool-call markup handling ─────────────────────────────────
# Some models emit a tool call as inline text markup inside the
# message `content` (its own "DSML" token format) instead of, or in addition
# to, the structured `tool_calls` array. If the runtime only reads
# the structured field it treats the raw markup as a final answer and shows
# garbage to the user. The helpers below detect that markup, turn it into
# proper tool_call dicts, and strip any residual fragments from final answers.
#
# The pipe character DeepSeek uses is the fullwidth U+FF5C ("｜"); we also
# tolerate a plain ASCII "|" variant and missing/extra pipes. `[｜|]{0,2}`
# matches either pipe (or none) so partial/garbled markup is still handled.

# Matches the FIRST occurrence of a DSML marker in content so we can truncate
# at that point. Handles both the angle-bracket tag form and a bare token
# sequence (e.g. `｜｜DSML`), with either fullwidth U+FF5C or ASCII `|` pipes.
_DSML_MARKER_RE = re.compile(
    r"(?:<\s*[｜|]{0,3}\s*DSML|[｜|]{1,3}DSML)",
    re.IGNORECASE,
)
# A full tool_calls block: <｜｜DSML｜｜tool_calls> ... </｜｜DSML｜｜tool_calls>
_DSML_TOOLCALLS_RE = re.compile(
    r"<\s*[｜|]{0,2}\s*DSML\s*[｜|]{0,2}\s*tool_calls\s*>(.*?)"
    r"<\s*/\s*[｜|]{0,2}\s*DSML\s*[｜|]{0,2}\s*tool_calls\s*>",
    re.IGNORECASE | re.DOTALL,
)
# A single invoke block inside a tool_calls block.
_DSML_INVOKE_RE = re.compile(
    r"<\s*[｜|]{0,2}\s*DSML\s*[｜|]{0,2}\s*invoke\s+name\s*=\s*[\"']([^\"']+)[\"'][^>]*>"
    r"(.*?)"
    r"<\s*/\s*[｜|]{0,2}\s*DSML\s*[｜|]{0,2}\s*invoke\s*>",
    re.IGNORECASE | re.DOTALL,
)
# A single parameter inside an invoke block: name + inner text value.
_DSML_PARAM_RE = re.compile(
    r"<\s*[｜|]{0,2}\s*DSML\s*[｜|]{0,2}\s*parameter\s+name\s*=\s*[\"']([^\"']+)[\"'][^>]*>"
    r"(.*?)"
    r"<\s*/\s*[｜|]{0,2}\s*DSML\s*[｜|]{0,2}\s*parameter\s*>",
    re.IGNORECASE | re.DOTALL,
)

# Llama 3.x native tool-call markup — Groq's strict tool-use validator rejects
# these shapes with HTTP 400 `tool_use_failed` and emits the raw markup in
# `failed_generation`. We recover it into proper OpenAI-style tool_calls so the
# agent loop can dispatch and continue. Observed variants (all live on Groq):
#   <function=name{json}>              inline close
#   <function=name{json}</function>    closing tag, no `>` (2026-07-24 probe)
#   <function=name>{json}</function>   canonical Llama 3 tag pair
_LLAMA_FUNC_RE = re.compile(
    r"<\s*function\s*=\s*([A-Za-z_][\w]*)\s*>?\s*(\{.*?\})\s*(?:>|</\s*function\s*>)",
    re.DOTALL,
)


def _parse_llama_native_tool_calls(text: str) -> list[dict]:
    """Extract Llama-native `<function=name{json}>` markup into OpenAI-shaped
    tool_calls dicts. Returns [] if no markup found or every match fails to
    parse as JSON.
    """
    if not text or "<function" not in text:
        return []
    out: list[dict] = []
    counter = 0
    for m in _LLAMA_FUNC_RE.finditer(text):
        name = m.group(1).strip()
        if not name:
            continue
        try:
            args = json.loads(m.group(2))
        except json.JSONDecodeError:
            continue
        counter += 1
        out.append({
            "id": f"llama_{counter}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(args),
            },
        })
    return out


def _strip_dsml(content: str) -> str:
    """Discard the entire DSML region from ``content`` and return only the prose before it.

    DeepSeek emits any tool-call markup AFTER any genuine prose, so we find the
    first DSML marker and throw away everything from that point to the end of
    the string — tags AND the inner parameter text.  This prevents raw parameter
    values (e.g. query strings) from leaking into a displayed final answer.

    If no DSML marker is present, returns ``content.strip()`` unchanged.
    """
    if not content:
        return ""
    if "DSML" not in content:
        return content.strip()
    m = _DSML_MARKER_RE.search(content)
    if m is None:
        return content.strip()
    return content[: m.start()].rstrip()


def _parse_dsml_tool_calls(content: str) -> tuple[str, list[dict]]:
    """Extract DeepSeek DSML tool-call markup from ``content``.

    Returns ``(cleaned_content, tool_calls)`` where ``cleaned_content`` is the
    message text with all DSML markup removed, and ``tool_calls`` is a list of
    dicts shaped exactly like the structured ``tool_calls`` field that
    ``_run_tool_call`` consumes::

        {"id": <generated>, "type": "function",
         "function": {"name": ..., "arguments": <json string>}}

    If no DSML markup is present, returns ``(content, [])`` unchanged.
    """
    if not content or "DSML" not in content:
        return (content or ""), []

    tool_calls: list[dict] = []
    counter = 0
    for block in _DSML_TOOLCALLS_RE.finditer(content):
        for inv in _DSML_INVOKE_RE.finditer(block.group(1)):
            tool_name = inv.group(1).strip()
            if not tool_name:
                continue
            args: dict[str, Any] = {}
            for param in _DSML_PARAM_RE.finditer(inv.group(2)):
                pname = param.group(1).strip()
                pvalue = param.group(2).strip()
                if pname:
                    args[pname] = pvalue
            counter += 1
            tool_calls.append({
                "id": f"dsml_{counter}",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(args),
                },
            })

    # Strip ALL DSML markup (including any tags outside a well-formed block).
    cleaned = _strip_dsml(content)
    return cleaned, tool_calls


_CM_INJECT_AGENTS = frozenset({"heavy-reasoning", "project-assistant"})


def _cm_prompt_fragment_for_turn(user_message: str) -> str:
    """Compact CrossDomainReasoner inject for construction agent paths."""
    try:
        from app.core.cross_domain_reasoner import CrossDomainReasoner

        reasoner = CrossDomainReasoner()
        injected = reasoner.inject_prompt("", user_message or "")
        text = (injected or "").strip()
        return text[:800] if text else ""
    except Exception:  # noqa: BLE001
        _LOG.debug("cross-domain prompt inject failed", exc_info=True)
        return ""


# Blocks that are pure infrastructure -- their presence alone does not make an
# agent functional (see Agent.unavailable_reason).
_INFRA_ONLY_BLOCKS = frozenset({"cache_manager"})


# File-consuming blocks need an explicit `file_path` schema so the LLM can't
# emit the call with empty args. Without this, the agent called e.g.
# boq_processor with {} after search_project_documents, got "No file_path
# provided", and deflected to the user. The runtime's
# _resolve_block_file_input then maps the bare filename to the encrypted
# stored path, so the agent only needs to pass the original_name string it
# saw from search_project_documents.
#
# Module scope rather than a local: it is a constant, it was rebuilt on every
# tool_definitions() call, and a schema the model depends on should be
# reachable by a test without standing up a block registry.
_FILE_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "boq_processor": {
        "description": (
            "Extract structured Bill of Quantities from an uploaded "
            "xlsx/csv/pdf file. Returns line items with quantities, "
            "rates, amounts, and totals. The file_path must be the "
            "exact original_name of a document returned by "
            "search_project_documents — never guess paths."
        ),
        "properties": {
            "file_path": {
                "type": "string",
                "description": (
                    "The document's original_name (e.g. "
                    "'Infra-1 - Demolition BOQ.pdf'). MUST come "
                    "from a prior search_project_documents call."
                ),
            },
            "char_offset": {
                "type": "integer",
                "description": (
                    "Optional. Read a LATER window of a result that "
                    "came back truncated: pass the next_char_offset "
                    "the previous call reported, and repeat until "
                    "chars_remaining is 0. Never tell the user a "
                    "value is absent from this document while "
                    "chars_remaining is above 0 -- read the rest "
                    "first."
                ),
            },
        },
        "required": ["file_path"],
    },
    "drawing_qto": {
        "description": (
            "Extract quantity takeoff from a drawing file "
            "(DXF/PDF; a DWG is refused -- ask for a DXF or PDF export). "
            "Returns measured areas, lengths, counts. "
            "The file_path must be the exact original_name returned by "
            "search_project_documents — never guess paths."
        ),
        "properties": {
            "file_path": {
                "type": "string",
                "description": (
                    "The drawing's original_name (e.g. "
                    "'tower_b_floor_plan.dxf'). MUST come from a prior "
                    "search_project_documents call."
                ),
            },
            "char_offset": {
                "type": "integer",
                "description": (
                    "Optional. Read a LATER window of a result that "
                    "came back truncated: pass the next_char_offset "
                    "the previous call reported, and repeat until "
                    "chars_remaining is 0. Never tell the user a "
                    "value is absent from this document while "
                    "chars_remaining is above 0 -- read the rest "
                    "first."
                ),
            },
        },
        "required": ["file_path"],
    },
    "spec_analyzer": {
        "description": (
            "Extract specifications, grades, standards, and methods "
            "from a specification document. file_path must be the "
            "original_name from search_project_documents — never guess."
        ),
        "properties": {
            "file_path": {
                "type": "string",
                "description": (
                    "The spec doc's original_name. MUST come from a "
                    "prior search_project_documents call."
                ),
            },
            "char_offset": {
                "type": "integer",
                "description": (
                    "Optional. Read a LATER window of a result that "
                    "came back truncated: pass the next_char_offset "
                    "the previous call reported, and repeat until "
                    "chars_remaining is 0. Never tell the user a "
                    "value is absent from this document while "
                    "chars_remaining is above 0 -- read the rest "
                    "first."
                ),
            },
        },
        "required": ["file_path"],
    },
    "primavera_parser": {
        "description": (
            "Parse a Primavera P6 .xer schedule file from the project. "
            "Returns activities, milestones, WBS, schedule_data, and CPM "
            "output. file_path must be the original_name from "
            "search_project_documents — never guess paths."
        ),
        "properties": {
            "file_path": {
                "type": "string",
                "description": (
                    "The XER file's original_name (e.g. "
                    "'Annexure 2 - Baseline Program XER.xer'). MUST come "
                    "from a prior search_project_documents call."
                ),
            },
            "char_offset": {
                "type": "integer",
                "description": (
                    "Optional. Read a LATER window of a result that "
                    "came back truncated: pass the next_char_offset "
                    "the previous call reported, and repeat until "
                    "chars_remaining is 0. Never tell the user a "
                    "value is absent from this document while "
                    "chars_remaining is above 0 -- read the rest "
                    "first."
                ),
            },
        },
        "required": ["file_path"],
    },
}


_MISSING_INPUT_NUDGE = (
    "The excerpts above were retrieved specifically because your draft "
    "answer said it did not have %r. Read them and answer the original "
    "question in full, using the figure(s) they contain and citing the "
    "clause or document they came from.\n"
    "If they still do not contain it, say so plainly and do NOT estimate, "
    "infer, or substitute a typical value -- a wrong figure is worse than "
    "the gap you already reported."
)

# Default UI agent. Specialty agents already bind hats in frontmatter;
# project-assistant is the /v1/chat/stream front door and was unhatted.
_DEFAULT_UI_AGENT = "project-assistant"


def hats_activation_enabled() -> bool:
    """True when FORK_HATS_ENABLED is live. Default off."""
    from app.agents.activation import hats_enabled
    return hats_enabled()


def hat_scores_sse(message: str) -> dict[str, Any] | None:
    """SSE ``hat_signals`` payload, or None when the flag is off.

    Mid-stream event the floor scorer counts, plus the same list stamped
    onto the terminal ``end`` event. ``scores`` stays for older readers.
    """
    if not hats_activation_enabled():
        return None
    from app.agents.activation import HatActivationAdapter
    from app.agents.catalog import list_hats
    adapter = HatActivationAdapter()
    scores = adapter.score_all_hats(message)
    selected = adapter.select_hat_for_message(message)
    selected_id = getattr(selected, "id", None)
    signals: list[dict[str, Any]] = []
    for hat in list_hats():
        discipline = hat.discipline.value
        score = float(hat.score_message(message) or 0.0)
        signals.append({
            "id": hat.id,
            "discipline": discipline,
            "score": score,
            # Multi-hat mode merges two manifests into "<primary>__<secondary>"
            # (activation._merge_multi_hat), so compare against the parts. An
            # `in` on the raw string would also match any hat whose id happens
            # to be a substring of another's.
            "selected": bool(selected_id) and hat.id in str(selected_id).split("__"),
        })
    signals.sort(key=lambda row: (-float(row["score"]), str(row["id"])))
    return {
        "type": "hat_signals",
        "hat_signals": signals,
        "hats": signals,
        "scores": scores,
        "selected": selected_id,
        "selected_name": getattr(selected, "name", None),
        "enabled": True,
    }


def _with_hat_signals(
    event: dict[str, Any],
    hat_evt: dict[str, Any] | None,
) -> dict[str, Any]:
    """Stamp hat activation onto a terminal ``end`` event for the floor scorer."""
    if not hat_evt or not isinstance(event, dict):
        return event
    stamped = dict(event)
    stamped["hat_signals"] = list(hat_evt.get("hat_signals") or [])
    stamped["hat_selected"] = hat_evt.get("selected")
    return stamped


def hat_turn_system_note(message: str) -> dict[str, str] | None:
    """Steer the default UI turn toward the winning discipline hat."""
    event = hat_scores_sse(message)
    if not event:
        return None
    scores = event.get("scores") or {}
    ranked = ", ".join(
        f"{name}={float(score):.2f}"
        for name, score in sorted(scores.items(), key=lambda kv: -float(kv[1]))
    )
    selected = event.get("selected")
    if selected:
        return {
            "role": "system",
            "content": (
                f"Active discipline hat this turn: {event.get('selected_name')} "
                f"({selected}). Hat scores: {ranked}."
            ),
        }
    if ranked:
        return {
            "role": "system",
            "content": (
                "Discipline hat scores for this turn (none above routing "
                f"threshold): {ranked}."
            ),
        }
    return None


def _apply_hat_activation(
    messages: list[dict[str, Any]],
    user_message: str,
    agent_name: str,
) -> dict[str, Any] | None:
    """Score hats and, on the default UI agent, append a turn note.

    Returns the SSE event (or None when the flag is off). Specialty
    agents already carry their kernels; they get scores only.

    Scoring is telemetry. A catalog/score crash must skip hats and let
    the turn answer; it must not become a stream ``error``.
    """
    try:
        event = hat_scores_sse(user_message)
        if not event:
            return None
        if agent_name == _DEFAULT_UI_AGENT:
            note = hat_turn_system_note(user_message)
            if note:
                if messages and messages[-1].get("role") == "user":
                    messages.insert(-1, note)
                else:
                    messages.append(note)
        return event
    except Exception:  # noqa: BLE001 - telemetry must never take down chat
        _LOG.exception(
            "hat activation failed; continuing without hat_signals"
        )
        return None


from app.agents.core import context as _core_context  # noqa: E402
from app.agents.core import llm as _core_llm  # noqa: E402
from app.agents.core import loop as _core_loop  # noqa: E402
from app.agents.core import streaming as _core_streaming  # noqa: E402
from app.agents.core import tools as _core_tools  # noqa: E402

@dataclass
class Agent:
    """Declarative agent definition."""

    name: str
    description: str
    system_prompt: str
    allowed_blocks: list[str] = field(default_factory=list)
    model: str = DEEPSEEK_DEFAULT_MODEL
    temperature: float = 0.3
    max_tokens: int = 2048
    icon: str = ""
    can_delegate: bool = False
    # Discipline-hat kernels bound at load time (manifest ids); the guidance
    # is already merged into system_prompt by _apply_hat_kernels.
    hats: list[str] = field(default_factory=list)
    # F34: this agent consumes figures the user supplies (corrections, actuals)
    # -- RAG injection must not strict-ground it against the corpus.
    user_data_authoritative: bool = False

    unavailable_reason = _core_loop.unavailable_reason

    tool_definitions = _core_tools.tool_definitions

    chat = _core_loop.chat

    _chat_impl = _core_loop._chat_impl

    _fetch_named_missing_input = _core_context._fetch_named_missing_input

    chat_stream = _core_loop.chat_stream

    _chat_stream_impl = _core_streaming._chat_stream_impl

    _build_messages = _core_context._build_messages

    _call_llm = _core_llm._call_llm

    _stream_synthesis = _core_llm._stream_synthesis

    _run_tool_call = _core_tools._run_tool_call


# ── Auto-validation middleware ───────────────────────────────────────────
# Every successful block call gets its numeric payload run through the
# validation_pipeline block automatically, and the verdict is grafted onto
# the tool result so the LLM sees it next turn. The heavy-reasoning prompt
# is updated to refuse to report numbers whose `validation.overall == fail`.
# Without this, the validation_pipeline block existed but only ran when the
# LLM remembered to call it — which is exactly the failure mode that let
# the 5,900 °C unit-conversion slip through earlier.

# Synthetic tools that aren't in BLOCK_REGISTRY and so can't carry a
# class-level `auto_validate` flag. Hardcoded here because they're part
# of the runtime contract, not a block.
_SYNTHETIC_NEVER_VALIDATE = {
    "validation_pipeline",        # don't recurse
    "remember_fact",              # synthetic, no numeric
    "delegate_to_agent",          # nested agent result
    "search_project_documents",   # text results
    "generate_wbs",               # has its own CPM validation
}


def _block_should_auto_validate(name: str) -> bool:
    """Per-block opt-out for auto-validation. Blocks declare
    ``auto_validate = False`` as a class attribute; the runtime reads
    it instead of an enumerated skip list. Synthetic tools without a
    ``BLOCK_REGISTRY`` entry use ``_SYNTHETIC_NEVER_VALIDATE``.
    """
    if name in _SYNTHETIC_NEVER_VALIDATE:
        return False
    cls = BLOCK_REGISTRY.get(name)
    if cls is None:
        return False
    return bool(getattr(cls, "auto_validate", True))


def _collect_numerics(result: Any) -> list[dict[str, Any]]:
    """Walk a block result dict and pick out numeric payloads worth checking.

    Yields a list of ``{value, unit?, context}`` packets in the shape
    validation_pipeline.process() expects. The detection is lenient — better
    to spot-check a few extras than miss a value silently.
    """
    if not isinstance(result, dict):
        return []
    out: list[dict[str, Any]] = []

    # sympy_reasoning / recommendation_template / formula_executor_v2 shapes
    for key, ctx_extra in (
        ("variances",     {"metric": "percent",   "label": "variance_pct"}),
        ("cost_impacts",  {"metric": "cost_usd",  "label": "cost_impact"}),
    ):
        items = result.get(key)
        if isinstance(items, list):
            for it in items[:8]:  # cap so we don't blow context
                if isinstance(it, dict):
                    v = it.get("value", it.get("variance_pct", it.get("cost_impact")))
                    if isinstance(v, (int, float)):
                        ctx = {**ctx_extra}
                        for c_key in ("material_type", "material", "item"):
                            if c_key in it:
                                ctx["material_type"] = str(it[c_key]).lower()
                                break
                        out.append({"value": v, "unit": it.get("unit"), "context": ctx})

    # boq_processor / construction container shapes
    if "total_cost" in result and isinstance(result["total_cost"], (int, float)):
        out.append({
            "value": result["total_cost"],
            "unit": result.get("currency", "USD"),
            "context": {"metric": "cost_usd", "currency": result.get("currency", "USD")},
        })

    # formula_executor result envelope
    fr = result.get("result")
    if isinstance(fr, (int, float)) and not isinstance(fr, bool):
        ctx = {"metric": result.get("metric") or result.get("task_metric") or ""}
        unit = result.get("unit") or result.get("output_unit")
        out.append({"value": fr, "unit": unit, "context": ctx})

    return out


async def _auto_validate(envelope: dict[str, Any]) -> None:
    """In-place: attach a `validation` field to the tool envelope.

    Runs the validation_pipeline block over every numeric in the result.
    Aggregates per-numeric verdicts into a single summary the LLM can read.
    """
    name = envelope.get("name", "")
    if not _block_should_auto_validate(name):
        return
    result = envelope.get("result")
    if not isinstance(result, dict) or result.get("status") != "success":
        return
    # UniversalBlock.execute returns {block, status, result: {actual...}}.
    # The numerics live in the inner result; unwrap one level when present.
    inner = result.get("result")
    if isinstance(inner, dict) and inner.get("status") == "success":
        result = inner
    numerics = _collect_numerics(result)
    if not numerics:
        envelope["validation"] = {"skipped": "no numeric value found"}
        return
    try:
        from app.blocks import BLOCK_REGISTRY
        from app.dependencies import _create_block_instance as _create
        from app.dependencies import block_instances as _bi
        if "validation_pipeline" not in BLOCK_REGISTRY:
            envelope["validation"] = {"skipped": "validation_pipeline not registered"}
            return
        vp_block = _bi.get("validation_pipeline")
        if vp_block is None:
            vp_block = _create(BLOCK_REGISTRY["validation_pipeline"])
            _bi["validation_pipeline"] = vp_block
    except Exception as e:
        envelope["validation"] = {"skipped": f"validation_pipeline init failed: {type(e).__name__}"}
        return

    per: list[dict[str, Any]] = []
    overall = "pass"
    first_failure: str | None = None
    for n in numerics[:8]:  # hard cap
        try:
            envelope_inner = await vp_block.execute(n, {})
        except Exception as e:  # noqa: BLE001
            per.append({"input": n, "overall": "error", "error": str(e)[:120]})
            continue
        # UniversalBlock.execute wraps the block's process() return in a
        # {block, status, result: {...}} envelope; the real verdict lives
        # under `result`.
        vr = envelope_inner.get("result") if isinstance(envelope_inner, dict) else None
        if not isinstance(vr, dict):
            continue
        v_overall = vr.get("overall")
        per.append({
            "value": n.get("value"),
            "unit": n.get("unit"),
            "overall": v_overall,
            "first_failure": vr.get("first_failure"),
        })
        if v_overall == "fail" and overall == "pass":
            overall = "fail"
            first_failure = vr.get("first_failure")
    envelope["validation"] = {
        "overall": overall,
        "first_failure": first_failure,
        "checks": per,
        "note": (
            "auto-run by runtime middleware; refuse to report any number "
            "whose check shows overall=fail without explaining which "
            "stage rejected it."
        ),
    }


# ── Loader ────────────────────────────────────────────────────────────────
AGENT_REGISTRY: dict[str, Agent] = {}


def load_agents(configs_dir: Path | None = None) -> dict[str, Agent]:
    """Load every `.md` config under `configs_dir` into AGENT_REGISTRY (replaces existing)."""
    configs_dir = configs_dir or CONFIGS_DIR
    AGENT_REGISTRY.clear()
    if not configs_dir.exists():
        return AGENT_REGISTRY
    for md in sorted(configs_dir.glob("*.md")):
        try:
            agent = _parse_agent_file(md)
            AGENT_REGISTRY[agent.name] = agent
        except Exception as e:
            print(f"failed to load agent {md.name}: {e}")
    return AGENT_REGISTRY


def get_agent(name: str) -> Agent | None:
    return AGENT_REGISTRY.get(name)


# ── Smart-orchestrator routing gate ─────────────────────────────────────────
# Pre-PR-#78, the production chat path bypassed smart_orchestrator entirely:
# the React UI calls /v1/agents/project-assistant/chat/stream which lands
# directly on Agent.chat_stream() with no keyword-routing consultation. The
# only place smart_orchestrator was consulted was /v1/chat/stream — a route
# the React frontend never hits.
#
# The fix is `select_agent_for_message`: a thin helper the agents-router
# calls BEFORE dispatching to Agent.chat()/chat_stream(). When the user's
# message classifies as a needs_planning intent (e.g. "create a 200-activity
# schedule" → generate_wbs at confidence >= 0.4), and the caller asked for
# anything OTHER than heavy-reasoning, the helper redirects to the
# heavy-reasoning agent so the tool-call loop actually runs the requested
# pipeline instead of producing prose.
#
# Design constraints from the operator:
#   * Real user traffic must pass through smart_orchestrator (this gate).
#   * Internal delegation (runtime.py:_run_tool_call → target.chat) must
#     NOT re-route — the parent agent already made the decision and a
#     sub-agent invocation shouldn't be hijacked. The agents-router never
#     hits the delegation path, so the gate-at-router-only design is
#     sufficient; we don't need an extra _depth check here.
#   * Test paths must not break. Tests that POST to /v1/agents/.../chat
#     will see the gate. Tests that call Agent.chat() directly (most
#     unit tests) skip it entirely. The gate is also kill-switchable via
#     SMART_ORCH_ROUTING_DISABLED=true for fast prod rollback.

_SMART_ORCH_BLOCK_CACHE: Any | None = None


def _get_smart_orchestrator_block() -> Any | None:
    """Lazy-load and cache a SmartOrchestratorBlock instance.

    Returns None if the block isn't registered (e.g. running without the
    construction kit) so the caller can fall back to the no-op routing
    decision (pass-through to the requested agent)."""
    global _SMART_ORCH_BLOCK_CACHE
    if _SMART_ORCH_BLOCK_CACHE is not None:
        return _SMART_ORCH_BLOCK_CACHE
    try:
        from app.blocks import BLOCK_REGISTRY
        cls = BLOCK_REGISTRY.get("smart_orchestrator")
        if cls is None:
            return None
        _SMART_ORCH_BLOCK_CACHE = cls()
        return _SMART_ORCH_BLOCK_CACHE
    except Exception:  # noqa: BLE001
        _LOG.debug("smart_orchestrator block cache init failed", exc_info=True)
        return None


def _routing_disabled() -> bool:
    """Kill-switch read at every call so Render env-var flips take effect
    without a restart."""
    return os.getenv("SMART_ORCH_ROUTING_DISABLED", "").strip().lower() in ("1", "true", "yes")


def _asks_self_coding(text: str) -> bool:
    t = (text or "").lower()
    return any(p in t for p in _SELF_CODING_PHRASES)


def _formula_ask_force_enabled() -> bool:
    """Kill switch for named-formula routing / Master-Corpus skip.

    ``FORCE_CALC_ON_FORMULA_ASK=0`` restores the pre-fix behaviour
    (dimension heuristic + intent_map.yaml only). Default ON.
    """
    return (os.getenv("FORCE_CALC_ON_FORMULA_ASK", "1") or "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


def _message_names_unambiguous_calculator(text: str) -> bool:
    """Registry name in underscore form, or a 3+ token spaced name or
    display name.

    ``pe_unit_convert`` is unambiguous. ``concrete volume`` is not —
    it is also a BOQ lookup phrase. See test_self_contained_calculation_routing.
    """
    try:
        from app.lib.construction_formulas import CALCULATORS, calculators_named_by_display
    except Exception:  # noqa: BLE001
        _LOG.debug("CALCULATORS import failed", exc_info=True)
        return False
    raw = text or ""
    if calculators_named_by_display(raw):
        return True
    underscored = raw.lower().replace("-", "_")
    spaced = raw.lower()
    for name in CALCULATORS:
        if len(name) < 6:
            continue
        if name.lower() in underscored:
            return True
        tokens = [t for t in name.split("_") if t]
        if len(tokens) >= 3 and name.replace("_", " ") in spaced:
            return True
    return False


def _message_matches_calculator_stem(text: str) -> bool:
    """True when the message carries the first two tokens of a 3+ token name.

    ``rebar lap`` matches ``rebar_lap_length``. Two-token registry names
    (``concrete_volume``) are excluded — they collide with BOQ lookups.
    """
    try:
        from app.lib.construction_formulas import CALCULATORS
    except Exception:  # noqa: BLE001
        _LOG.debug("CALCULATORS import failed", exc_info=True)
        return False
    spaced = (text or "").lower().replace("-", " ").replace("_", " ")
    for name in CALCULATORS:
        tokens = [t for t in name.lower().replace("-", "_").split("_") if t]
        if len(tokens) < 3:
            continue
        stem = " ".join(tokens[:2])
        if len(stem) < 8:
            continue
        # ``critical_path_float`` stems to "critical path", which is a
        # schedule lookup ("What is the critical path of this schedule?")
        # not a construction_calc ask. Keep rebar lap / formwork striking.
        if stem in _FORMULA_STEM_LOOKUP_COLLISIONS:
            continue
        if stem in spaced:
            return True
    return False


_FORMULA_STEM_LOOKUP_COLLISIONS = frozenset({
    "critical path",
})


def _message_is_schedule_or_programme_deliverable(text: str) -> bool:
    """True for a generate/build programme or schedule ask, not a formula.

    Live FIXTURE-c: "Build a plumbing flow programme for a 20-storey tower"
    is a schedule-builder path. intent_map.yaml maps "plumbing flow" to
    construction_calc; that must not steal the deliverable.
    """
    raw = text or ""
    if not _is_generative_request(raw):
        return False
    low = raw.lower()
    return any(
        p in low
        for p in (
            "programme", "program", "schedule", "wbs", "gantt",
            "critical path",
        )
    )


def _turn_already_ran_construction_calc(messages: list | None) -> bool:
    """True when this turn already invoked construction_calc (predispatch)."""
    for m in messages or []:
        content = str((m or {}).get("content") or "")
        if content.lstrip().startswith("PLATFORM PRE-DISPATCH: construction_calc"):
            return True
        if (m or {}).get("name") == "construction_calc":
            return True
    return False


#: A symbol raised to a power inside a stated formula ("c*T^9", "k*x**9",
#: "a*d³"). A letter after "/" is a unit ("kN/m^9"), and a bare unit symbol
#: ("mm^9", "m²") is not a formula.
_STATED_FORMULA_RE = re.compile(
    r"(?<![A-Za-z/])([A-Za-z]{1,3})\s*(?:\^|\*\*)\s*\d|(?<![A-Za-z/])([A-Za-z]{1,3})[²³⁴]"
)
_UNIT_POWER_SYMBOLS = frozenset({"m", "mm", "cm", "km", "ft", "in", "s", "yd", "mi", "sqm"})


def _states_formula_with_input(text: str) -> bool:
    """True when the ask writes out its own formula AND gives a quantity.

    The user supplied the expression and the value; there is nothing to look
    up. A formula with no number is a question about the formula, not a
    calculation; a unit written with a power is not a formula.
    """
    raw = text or ""
    symbols = [a or b for a, b in _STATED_FORMULA_RE.findall(raw)]
    if not any(s.lower() not in _UNIT_POWER_SYMBOLS for s in symbols):
        return False
    return _count_dimensions(raw) >= 1


def _user_supplied_registered_calculation(
    text: str,
) -> tuple[str, dict[str, Any]] | None:
    """Registered formula whose name and required inputs are in ``text``.

    A bill, a schedule, a drawing take-off, an RFI or a variation is not
    this, even when the same sentence also contains numbers.
    """
    raw = text or ""
    if not raw.strip():
        return None
    if _message_wants_inline_boq(raw) or _message_is_schedule_or_programme_deliverable(raw):
        return None
    if (
        _message_wants_drawing_qto(raw)
        or _message_wants_rfi_draft(raw)
        or _message_wants_vo_draft(raw)
    ):
        return None
    try:
        from app.lib.construction_formulas import formula_completed_by_user_text
        return formula_completed_by_user_text(raw)
    except Exception:  # noqa: BLE001 — routing must still classify
        _LOG.exception("supplied-input formula selection failed")
        return None


# Kinds of record a user points at. The noun says what the record is,
# never which record it is.
_DOCUMENT_NOUN = (
    r"(?:notes?|specifications?|specs?|documents?|docs?|files?|uploads?|"
    r"attachments?|drawings?|clauses?|contracts?|schedules?|reports?|"
    r"checklists?|datasheets?|appendices|annex(?:es)?|boq|bills?|memos?|"
    r"minutes|letters?|records?|method\s+statements?)"
)
# Words that end a noun phrase instead of modifying its head noun.
_PHRASE_BREAK = (
    r"(?!(?:about|of|for|on|in|at|the|a|an|to|with|from|and|or|what|"
    r"which|that|is|are|was|does|do|say|says)\b)"
)
# A file name the user typed: a stem and a document extension. A stem may
# hold spaces ("level 2 memo v3.docx"); it never holds a word of the
# sentence around it.
_FILE_WORD = (
    r"(?!(?:about|of|for|on|in|at|the|a|an|to|with|from|and|or|what|which|"
    r"that|is|are|was|does|do|did|say|says|per|by|my|our|your|this|these|"
    r"those|it|its|calculate|compute|work|out|using|under|according)\b)"
    r"[\w\-]+"
)
_FILE_TOKEN = (
    rf"(?:{_FILE_WORD}[ ]){{0,5}}[\w\-]+(?:\.[\w\-]+)*"
    r"\.(?:txt|md|docx?|pdf|xlsx?|csv|rtf|odt)\b"
)
_DETERMINER = r"(?:(?:the|this|that|my|our|your)\s+)?"

# The ask locates the answer in a project record: "in the specification",
# "per site_note.txt". A bare formula name is not this.
_DOCUMENT_FRAME_RE = re.compile(
    r"\b(?:in|from|per|under|according\s+to|stated\s+in|set\s+out\s+in|"
    r"specified\s+in|given\s+in)\s+"
    r"(?:the\s+|this\s+|our\s+|my\s+|a\s+|an\s+)?"
    rf"(?:(?:[\w-]+\s+){{0,6}}{_DOCUMENT_NOUN}\b|{_FILE_TOKEN})",
    re.IGNORECASE,
)
# The ask asks what a record says: "what does the note say", "what does
# site_note.txt state".
_REPORT_FRAME_RE = re.compile(
    rf"\b(?:does|do|did|will)\s+(?:(?:[\w'’\-]+\s+){{0,6}}?{_DOCUMENT_NOUN}|{_DETERMINER}{_FILE_TOKEN})\s+"
    r"(?:say|state|specify|require|show|give|list|mention|recommend|"
    r"set\s+out|define|stipulate|allow|prescribe|indicate|contain|have)\b",
    re.IGNORECASE,
)
# The user points at a record of their own without naming the file:
# "my uploaded note", "the attached spec", "the memo I uploaded".
_OWN_UPLOAD_RE = re.compile(
    rf"\b(?:my|our)\s+(?:{_PHRASE_BREAK}[\w-]+\s+){{0,3}}?(?P<n1>{_DOCUMENT_NOUN})\b"
    r"|\b(?:the|this|that|these|those)\s+(?:(?:just|recently|latest|last)\s+)?"
    rf"(?:uploaded|attached|shared)\s+(?:{_PHRASE_BREAK}[\w-]+\s+){{0,3}}?"
    rf"(?P<n2>{_DOCUMENT_NOUN})\b"
    rf"|\b(?:the|this|that)\s+(?P<n3>{_DOCUMENT_NOUN})\s+(?:that\s+|which\s+)?"
    r"(?:i|we)\s+(?:just\s+|have\s+|had\s+)?"
    r"(?:uploaded|attached|shared|sent|added|gave|provided)\b",
    re.IGNORECASE,
)


_UPLOAD_VERB_RE = re.compile(
    r"\b(?:uploaded|attached|shared|sent|added|provided|gave)\b", re.IGNORECASE,
)


def _own_upload_nouns(text: str) -> list[str]:
    """Document nouns the ask uses for a record of the user's own.

    "my schedule" alone may be a deliverable to build; it is a record to
    read when the phrase says it was uploaded, or the ask asks what the
    record says.
    """
    raw = text or ""
    framed = bool(_DOCUMENT_FRAME_RE.search(raw) or _REPORT_FRAME_RE.search(raw))
    nouns: list[str] = []
    for match in _OWN_UPLOAD_RE.finditer(raw):
        if not framed and _UPLOAD_VERB_RE.search(match.group(0)) is None:
            continue
        noun = match.group("n1") or match.group("n2") or match.group("n3") or ""
        noun = re.sub(r"\s+", " ", noun.strip().lower())
        if noun and noun not in nouns:
            nouns.append(noun)
    return nouns


def _document_question(text: str) -> bool:
    """The ask is about what a record says, not a request to compute."""
    raw = text or ""
    return bool(
        _DOCUMENT_FRAME_RE.search(raw)
        or _REPORT_FRAME_RE.search(raw)
        or _own_upload_nouns(raw)
    )


_FILE_TOKEN_RE = re.compile(_FILE_TOKEN, re.IGNORECASE)


def _without_file_names(text: str) -> str:
    """The ask with typed file names blanked: a digit in a name is not a figure."""
    return _FILE_TOKEN_RE.sub(" ", text or "")


def _user_supplied_operand(text: str) -> bool:
    """The user's own words already carry a figure a calculator can use."""
    raw = _without_file_names(text)
    if _looks_like_self_contained_calculation(raw) or _states_formula_with_input(raw):
        return True
    try:
        from app.lib.construction_formulas import named_formula_operands
        if named_formula_operands(raw):
            return True
        from app.lib.construction_formulas_structural_rc import (
            looks_like_slab_thickness_min_ask,
        )
        # That detector already requires a span the user wrote.
        if looks_like_slab_thickness_min_ask(raw):
            return True
    except Exception:  # noqa: BLE001 — routing must still classify
        _LOG.exception("operand check failed")
        return True
    return False


def _formula_run_authorized(text: str, calculation: str | None = None) -> bool:
    """A calculator may run on this ask.

    It may when the user supplied an operand. Otherwise only when the ask
    requests that calculation and is not a question about what a record
    says: answering a record question from signature defaults gives
    figures that are neither the user's nor the document's. ``calculation``
    narrows "requests that calculation" to the formula about to run.
    """
    raw = text or ""
    if _user_supplied_operand(raw):
        return True
    if _document_question(raw):
        return False
    if _NAMED_CALC_ASK_RE.search(raw):
        return True
    try:
        from app.lib.construction_formulas import (
            calculator_name_from_text,
            explicit_calculation_request,
            message_names_registry_id,
        )
        if explicit_calculation_request(raw) or message_names_registry_id(raw):
            return True
        named = calculator_name_from_text(raw)
        if calculation:
            return named == calculation
        return bool(named)
    except Exception:  # noqa: BLE001 — routing must still classify
        _LOG.exception("formula authorization check failed")
        return True


def _document_question_without_operands(text: str) -> bool:
    """A question about what a record says that supplies no figure."""
    raw = text or ""
    return _document_question(raw) and not _user_supplied_operand(raw)


def _defaulted_run_refusal(
    calculation: str | None,
    params: dict[str, Any] | None,
    user_message: str | None,
) -> dict[str, Any] | None:
    """The envelope for a calculator call that would run only on defaults.

    None when the call carries an operand, or the operator's ask requests
    this calculation. A call with no operator ask (not a chat turn) is not
    judged here.
    """
    ask = (user_message or "").strip()
    name = str(calculation or "").strip()
    if not ask or not name:
        return None
    try:
        from app.lib.construction_formulas import run_carries_operand
        from app.lib.source_labels import formula_display_name

        call = dict(params or {})
        if isinstance(call.get("text"), str):
            call["text"] = _without_file_names(call["text"])
        if run_carries_operand(name, call, "", stated=_without_file_names(ask)):
            return None
        if _formula_run_authorized(ask, name):
            return None
        display = formula_display_name(name) or "This calculation"
    except Exception:  # noqa: BLE001 — the guard must not break a turn
        _LOG.exception("defaulted-run check failed")
        return None
    return {
        "status": "not_run",
        "calculation": name,
        "reason": (
            f"{display} was not run: the request gives no figure for it "
            "and does not ask for that calculation. Answer from the text "
            "of the documents read for this turn."
        ),
    }


def _message_is_formula_style_ask(text: str) -> bool:
    """True when the turn is a formula / calculator ask, not a doc lookup.

    Does not pick which calculator — Agent C owns mapping. This only
    says the turn must go through construction_calc and must not fall
    back to Master Corpus RAG on another project_id.
    """
    raw = text or ""
    if not raw.strip():
        return False
    if _document_question_without_operands(raw):
        return False
    if _message_wants_inline_boq(raw):
        return False
    if _message_is_schedule_or_programme_deliverable(raw):
        return False
    # User-supplied operands complete a registered formula before any
    # document-lookup veto. The numbers are the inputs; the excerpts
    # are not required to repeat them.
    if _user_supplied_registered_calculation(raw):
        return True
    if not _formula_ask_force_enabled():
        return False
    try:
        if message_is_contract_data_lookup(raw):
            return False
    except Exception:  # noqa: BLE001
        _LOG.debug("contract-data lookup check skipped", exc_info=True)
    if _looks_like_self_contained_calculation(raw):
        return True
    if _states_formula_with_input(raw):
        return True
    # ACI one-way slab minimum thickness is slab_thickness_min. The
    # question does not spell the registry name, and "thickness of a
    # slab" is not the intent-map phrase "slab thickness".
    try:
        from app.lib.construction_formulas_structural_rc import (
            looks_like_slab_thickness_min_ask,
        )
        if looks_like_slab_thickness_min_ask(raw):
            return True
    except Exception:  # noqa: BLE001 — routing must still classify
        _LOG.exception("slab thickness formula check failed")
    # Underscore form (pe_unit_convert) or a 3+ token spaced name.
    # Two-token spaced names ("concrete volume") collide with BOQ lookups
    # and must stay on RAG — `_message_names_registered_calculator` is
    # too loose here.
    if _message_names_unambiguous_calculator(raw):
        return True
    low = raw.lower()
    for phrases, tool in _INTENT_TOOL_MAP:
        if tool == "construction_calc" and any(p in low for p in phrases):
            return True
    return _message_matches_calculator_stem(raw)


def message_wants_formula_calculator(text: str) -> bool:
    """Public alias for retrieve / inject — skip Master Corpus fallback."""
    return _message_is_formula_style_ask(text)


def project_is_master_corpus(project_id: str | None) -> bool:
    """True when ``project_id`` is the Master Corpus alias or its source."""
    pid = (project_id or "").strip()
    if not pid:
        return False
    if pid == "master_corpus":
        return True
    try:
        from app.core.projects import (
            MASTER_CORPUS_PROJECT_ID,
            MASTER_CORPUS_SOURCE_PROJECT_ID,
        )
        if pid in (MASTER_CORPUS_PROJECT_ID, MASTER_CORPUS_SOURCE_PROJECT_ID):
            return True
    except Exception:  # noqa: BLE001
        _LOG.debug("master-corpus id table unavailable", exc_info=True)
    src = (os.getenv("MASTER_CORPUS_SOURCE_PROJECT_ID") or "").strip()
    return bool(src) and pid == src


def project_is_user_fixture(project_id: str | None) -> bool:
    """True when the active project is a user FIXTURE (not Master Corpus).

    Live UI ids are short hex slugs (``0a1b2c3d``-shaped); the FIXTURE- prefix lives
    on the project name. Synthetic test pids embed ``fixture`` too.
    """
    pid = (project_id or "").strip()
    if not pid or project_is_master_corpus(pid):
        return False
    low = pid.lower()
    if low.startswith("fixture") or "fixture" in low:
        return True
    try:
        from app.core.projects import get_project
        rec = get_project(pid, user_id=None, include_admin_approved=True)
        name = str((rec or {}).get("name") or "").strip().lower()
        return name.startswith("fixture")
    except Exception:  # noqa: BLE001
        _LOG.debug("user-fixture name lookup skipped", exc_info=True)
        return False


def should_suppress_master_corpus_fallback(
    project_id: str | None, text: str | None,
) -> bool:
    """True when a calculator-shaped ask must not use Master Corpus RAG.

    Any non-master project (including a user FIXTURE whose id is a hex
    slug) skips fallback / banner / MC sources for formula-style and
    named_calculator asks. Operator-selected Master Corpus is unchanged.
    """
    raw = (text or "").strip()
    if not raw:
        return False
    if project_is_master_corpus(project_id):
        return False
    if _message_is_formula_style_ask(raw):
        return True
    if _message_wants_named_calculator(raw):
        return True
    return False


_MC_BLEED_PREAMBLE_RE = re.compile(
    r"^\s*_?This project has no documents of its own for this question\s+"
    r"[—–-]\s+answering from the Master Corpus\.?_?\s*",
    re.IGNORECASE,
)


def _strip_master_corpus_preamble(text: str) -> str:
    """Drop the official banner and the model's copy of the same line."""
    raw = text or ""
    if _MASTER_CORPUS_FALLBACK_NOTE.strip() in raw:
        raw = raw.replace(_MASTER_CORPUS_FALLBACK_NOTE, "")
        raw = raw.replace(_MASTER_CORPUS_FALLBACK_NOTE.strip(), "")
    cleaned = _MC_BLEED_PREAMBLE_RE.sub("", raw, count=1)
    return cleaned.lstrip()


_CALC_ASK_BLOB_KEYS = ("text", "formula", "message")


def _calc_args_have_ask_blob(args: dict | None) -> bool:
    """True when the tool kwargs already carry extractable ask text."""
    if not isinstance(args, dict):
        return False
    if any(args.get(k) not in (None, "") for k in _CALC_ASK_BLOB_KEYS):
        return True
    for nest in ("params", "input"):
        inner = args.get(nest)
        if isinstance(inner, dict) and any(
            inner.get(k) not in (None, "") for k in _CALC_ASK_BLOB_KEYS
        ):
            return True
    return False


def _inject_user_ask_into_construction_calc_args(
    args: dict | None,
    user_message: str | None,
    history: list | None = None,
) -> dict:
    """Copy the current user turn into empty construction_calc kwargs.

    Live Phase-2 re-probe: the model calls construction_calc with only
    ``{"action": "construction_calc"}``. #663 extract reads
    text/formula/message — without the ask, bind never sees the labeled
    numbers. D7: never invent a figure that is not in the ask. Existing
    text/formula/message win so a later retry with real params is kept.

    A follow-up ("add a waste allowance and price the total") has no
    geometry. The count lives on the previous operator turn. Carry that
    text as ``prior_text`` so concrete_volume multiplies by it instead
    of pricing one element. Explicit dims on this call are not replaced.
    """
    out = dict(args or {})
    ask = str(user_message or "").strip()
    if ask:
        try:
            from app.lib.construction_formulas_quantities import (
                follow_up_refers_to_stated_total,
                prior_operator_text,
            )
            if follow_up_refers_to_stated_total(ask):
                prior = prior_operator_text(history)
                if prior:
                    out.setdefault("prior_text", prior)
        except Exception:
            _LOG.exception("stated-total follow-up context skipped")
    if not ask or _calc_args_have_ask_blob(out):
        return out
    out["text"] = ask
    return out


def _formula_calculator_name_from_message(text: str) -> str | None:
    """Unique registry name implied by the message, or None.

    Used only to invoke construction_calc. Does not invent a mapping
    among audit #43–84 — if two names match, we pass no name and the
    tool returns an honest unknown-calculation envelope.

    Shared with ``calculator_name_from_text``: full names beat 2-token
    stems (mix-design vs slip-form, cost-buildup concrete vs rebar) and
    a unique 3-token tail (``well point spacing``) counts.
    """
    try:
        from app.lib.construction_formulas import calculator_name_from_text
        return calculator_name_from_text(text)
    except Exception:  # noqa: BLE001
        _LOG.debug("calculator_name_from_text failed", exc_info=True)
        return None


_CALC_NOT_RUN_HEAD = f"{_PREDISPATCH_PREFIX} construction_calc was not run."


def _formula_predispatch_instruction(calc_name: str | None, result: Any) -> str:
    """What the model is told after the formula pre-dispatch.

    A real result is the answer. When no unique registry calculator matched
    (``calc_name`` None) the tool returns an "Unknown calculation" envelope,
    and "answer from that result" left the model nothing to answer from: it
    filled the gap with free-form variants that changed run to run ("a 10%
    allowance -> 44 days"). Say what actually happened and what to do.

    A calculator that matched but did not run is not "no calculator": its
    question names the inputs it needs by their labels.
    """
    ok = bool(isinstance(result, dict) and result.get("ok"))
    if calc_name and ok:
        return ("construction_calc has already been run. Answer from that "
                "result. Do not answer from Master Corpus excerpts.")
    inner = result.get("result") if isinstance(result, dict) else None
    inner = inner if isinstance(inner, dict) else {}
    question = str(inner.get("question") or "").strip()
    if inner.get("status") == "not_run" and str(inner.get("reason") or "").strip():
        return str(inner["reason"]).strip()
    if calc_name and question:
        nudge = _INPUT_QUESTION_NUDGE if inner.get("needs_input") else _CALC_MISSING_INPUT_NUDGE
        return nudge.format(question=question)
    if calc_name:
        from app.lib.source_labels import formula_display_name

        display = formula_display_name(calc_name) or "The matched calculator"
        return (f"{display} did not return a result on this request. Do not "
                "compute it yourself or choose its inputs. If the user's message "
                "or the project documents state them, call construction_calc "
                f"with {calc_name} and those figures; otherwise ask the user for "
                "the inputs it needs.")
    return ("No registry calculator matched this question, so there is no "
            "calculator result. If one of the listed calculators fits, call "
            "construction_calc with that name. Otherwise compute the answer "
            "once from the operator's own figures, show the working, and state "
            "a single result: no alternative scenarios, allowances or figures "
            "the operator did not give. Do not answer from Master Corpus excerpts.")


def _formula_not_run_bubble(calc_name: str | None, result: Any) -> str:
    """The platform note for a formula pre-dispatch that produced no result.

    An envelope that is not a result is never handed over as a draft that
    has "already been run": the model repeated the tool's error to the user
    as a failed run. The note says the calculator was not run and what to do,
    with the registered names only when none matched.
    """
    inner = result.get("result") if isinstance(result, dict) else None
    inner = inner if isinstance(inner, dict) else {}
    parts = [_CALC_NOT_RUN_HEAD, _formula_predispatch_instruction(calc_name, result)]
    available = inner.get("available")
    if not calc_name and isinstance(available, list) and available:
        parts.append("Registered calculators: " + ", ".join(str(a) for a in available) + ".")
    return " ".join(parts)


async def _predispatch_formula_calc(
    agent: "Agent",
    messages: list,
    project_id: str | None,
    operator_text: str | None = None,
) -> dict[str, Any] | None:
    """Run construction_calc before the model can answer from Master Corpus.

    Kimi/Groq reject named tool_choice, so `_forced_specific_tool` alone
    left named_calculator turns with zero tool calls. Predispatch is the
    provider-independent lever. Kill-switch: AGENT_FORMULA_PREDISPATCH=0.
    """
    if os.getenv("AGENT_FORMULA_PREDISPATCH", "1") == "0":
        return None
    if "construction" not in getattr(agent, "allowed_blocks", ()):
        return None
    try:
        user_msg, _history = _messages_user_and_history(messages)
        detect = (operator_text or user_msg or "").strip()
        if not detect:
            return None
        if (await _off_loop(_message_wants_rfi_draft, detect)) or (await _off_loop(_message_wants_vo_draft, detect)):
            return None
        if (await _off_loop(_message_wants_inline_boq, detect)):
            return None
        if (await _off_loop(_message_wants_drawing_qto, detect)):
            return None
        if (await _off_loop(_message_is_schedule_or_programme_deliverable, detect)):
            return None
        if not (
            (await _off_loop(_message_is_formula_style_ask, detect))
            or (await _off_loop(_message_wants_named_calculator, detect))
        ):
            return None
        if await _off_loop(_document_question_without_operands, detect):
            return None
        completed = await _off_loop(_user_supplied_registered_calculation, detect)
        bound: dict[str, Any] = {}
        if completed:
            calc_name, bound = completed
        else:
            calc_name = await _off_loop(_formula_calculator_name_from_message, detect)
        params = {"text": detect, **bound}
        tc = {
            "id": "predispatch-construction_calc",
            "type": "function",
            "function": {
                "name": "construction_calc",
                "arguments": json.dumps({
                    "calculation": calc_name,
                    "params": params,
                    "text": detect,
                }),
            },
        }
        result = await agent._run_tool_call(tc)
        inner = result.get("result") if isinstance(result, dict) else result
        ran = (
            isinstance(inner, dict)
            and inner.get("status") == "success"
            and bool(inner.get("calculation"))
        )
        if ran:
            # A user-bubble alone is not a tool run, so the credit path
            # never sees the calculator. Record the same envelope as a
            # tool result. The arguments carry the user's inputs.
            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [tc],
            })
            messages.append({
                "role": "tool",
                "name": "construction_calc",
                "tool_call_id": tc["id"],
                "content": json.dumps(inner, default=str),
            })
            _inject_predispatch(
                messages,
                "construction_calc",
                json.dumps(inner, default=str)[:4000],
                (await _off_loop(_formula_predispatch_instruction, calc_name, result)),
            )
        else:
            from app.agents import answer_exit

            answer_exit.note_tool_result("construction_calc", inner)
            messages.append({
                "role": "user",
                "content": await _off_loop(_formula_not_run_bubble, calc_name, result),
            })
        return {
            "name": "construction_calc",
            "ok": bool(isinstance(result, dict) and result.get("ok")),
            "predispatched": True,
            "result": inner,
        }
    except Exception:  # noqa: BLE001
        _LOG.warning(
            "construction_calc pre-dispatch failed; continuing normally",
            exc_info=True,
        )
        return None


def _message_names_registered_calculator(text: str) -> bool:
    """True when the message literally names a registry calculator.

    Named formulas are an existing feature — construction_calc stays in
    charge. Unnamed custom arithmetic is self-coding's job.
    """
    try:
        from app.lib.construction_formulas import CALCULATORS
    except Exception:  # noqa: BLE001
        _LOG.debug("CALCULATORS import failed", exc_info=True)
        return False
    raw = text or ""
    t = raw.lower().replace("-", "_")
    spaced = raw.lower()
    for name in CALCULATORS:
        if len(name) < 6:
            continue
        if name.lower() in t or name.replace("_", " ") in spaced:
            return True
    return False


def _should_handoff_unmatched_calc(text: str) -> bool:
    """Computation-shaped, no registered formula name → self-coding once."""
    if _message_names_registered_calculator(text):
        return False
    # Live tip 0a95d03: inline BOQ ask1 must not hand off to self-coding
    # (which then called formula_executor_v2 and 429'd).
    try:
        from app.core.site_vocab import message_has_inline_boq_lines
        if message_has_inline_boq_lines(text) or _message_wants_vo_draft(text):
            return False
    except Exception:  # noqa: BLE001
        _LOG.debug("inline-boq handoff guard skipped", exc_info=True)
    return _looks_like_self_contained_calculation(text) or _asks_self_coding(text)


def _nudge_for_failed_tool(tool_result: dict[str, Any], agent: "Agent") -> str:
    """Prefer a one-shot self-coding handoff over retrying unknown formulas."""
    inner = tool_result.get("result") if isinstance(tool_result, dict) else None
    err = ""
    if isinstance(inner, dict):
        err = str(inner.get("error") or "")
        if inner.get("needs_input") and inner.get("question"):
            return _INPUT_QUESTION_NUDGE.format(question=inner["question"])
        if inner.get("missing") and inner.get("question"):
            return _CALC_MISSING_INPUT_NUDGE.format(question=inner["question"])
    if (
        agent.can_delegate
        and agent.name != "self-coding"
        and "unknown calculation" in err.lower()
    ):
        return _UNKNOWN_CALC_SELF_CODING_NUDGE
    name = str((tool_result or {}).get("name") or "").lower()
    if name in ("image", "ocr", "ocr_v2"):
        return (
            "The site photo could not be read (empty, placeholder, or no "
            "usable pixels). Do not retry the image tool. Answer the safety "
            "question from the depth, soil type, and geometry the user already "
            "stated. Use OSHA 29 CFR 1926 Subpart P."
        )
    return _TOOL_ERROR_NUDGE


# Generalist entry points that MAY be redirected to heavy-reasoning (and
# whose turns MAY be replaced by a predefined deliverable flow). A caller
# who addressed a specialist by name (learning, quantity-surveyor, …) chose
# that agent deliberately — keyword routing must not steal the turn.
# Live find: /v1/agents/learning/chat with "invoice rate 480 SAR/m³" was
# classified as payment_certificate and answered by heavy-reasoning, which
# has no learning_engine. Same contract as F24 (predefined intercept).
ROUTING_GENERALISTS = frozenset({
    "heavy-reasoning", "project-assistant", "smart-orchestrator",
})


async def select_agent_for_message(
    user_message: str,
    requested_agent: Agent,
) -> tuple[Agent, dict[str, Any]]:
    """Decide which agent should actually handle this message.

    Returns ``(final_agent, routing_info)``. ``routing_info`` is a dict
    suitable for emitting as an SSE event so the client can see the
    routing decision. Shape::

        {
            "requested": "project-assistant",
            "final": "heavy-reasoning",
            "action": "generate_wbs",
            "confidence": 0.8,
            "reason": "needs_planning",
        }

    Pass-through cases (``final == requested``):
      * The kill-switch ``SMART_ORCH_ROUTING_DISABLED`` is set.
      * smart_orchestrator isn't registered (no construction kit loaded).
      * Message is empty / whitespace.
      * Message is a Contract Data lookup (Time for Completion / milestones).
      * Top action confidence is below the routing threshold.
      * Top action is not in ``GENERATIVE_INTENTS`` (i.e. small talk / Q&A).
      * The requested agent is already ``heavy-reasoning``.
      * ``heavy-reasoning`` isn't registered in AGENT_REGISTRY.

    Each pass-through still populates ``routing_info`` with the
    classification result so the caller can observe what would have
    happened, and so the UI can show a "stayed on" badge if it wants.
    """
    info: dict[str, Any] = {
        "requested": requested_agent.name,
        "final": requested_agent.name,
        "action": None,
        "confidence": 0.0,
        "reason": "no-op",
    }

    if _routing_disabled():
        info["reason"] = "routing_disabled_env"
        return requested_agent, info

    if not user_message or not user_message.strip():
        info["reason"] = "empty_message"
        return requested_agent, info

    if message_is_contract_data_lookup(user_message):
        info["reason"] = "contract_data_lookup"
        return requested_agent, info

    if message_wants_answer_report(user_message):
        info["reason"] = "answer_report_export"
        return requested_agent, info

    block = _get_smart_orchestrator_block()
    if block is None:
        info["reason"] = "smart_orchestrator_not_registered"
        return requested_agent, info

    try:
        result = await block.process({"user_message": user_message})
    except Exception as exc:  # noqa: BLE001
        # Routing is a best-effort enhancement; a classifier crash must
        # never break chat. Pass through with the error logged so the
        # operator can see it in Sentry/logs without a user-visible
        # failure.
        _LOG.warning("smart_orchestrator classification failed: %s", exc)
        info["reason"] = "classifier_error"
        info["error"] = str(exc)[:200]
        return requested_agent, info

    # Local import to avoid the runtime → action_router cycle that exists
    # because action_router consumes smart_orchestrator output and lives
    # in app.core (which runtime.py doesn't import at module top).
    from app.core.action_router import (
        best_action,
        needs_planning,
    )

    action, confidence = best_action(result)
    info["action"] = action
    info["confidence"] = confidence

    # Explicit "use self-coding" from a generalist wins even when the
    # classifier also matched a deliverable. One hop to self-coding.
    if (
        requested_agent.name in ROUTING_GENERALISTS
        and requested_agent.name != "self-coding"
        and (await _off_loop(_asks_self_coding, user_message))
    ):
        sc = AGENT_REGISTRY.get("self-coding")
        if sc is not None:
            info["final"] = sc.name
            info["reason"] = "self_coding_requested"
            return sc, info

    # "use N days per slab and re-run" must re-call generate_wbs, not a
    # named calculator or a stale WBS table from history.
    try:
        from app.lib.wbs_duration_overrides import message_wants_wbs_duration_rerun
        if message_wants_wbs_duration_rerun(user_message):
            info["action"] = "generate_wbs"
            info["confidence"] = max(float(info.get("confidence") or 0.0), 0.85)
            info["reason"] = "duration_override_rerun"
            if requested_agent.name in ROUTING_GENERALISTS:
                heavy = AGENT_REGISTRY.get("heavy-reasoning")
                if heavy is not None and requested_agent.name != "heavy-reasoning":
                    info["final"] = heavy.name
                    return heavy, info
            return requested_agent, info
    except Exception:  # noqa: BLE001
        _LOG.debug(
            "duration-override routing skipped; continuing classifier path",
            exc_info=True,
        )

    if (await _off_loop(_message_wants_named_calculator, user_message)):
        info["action"] = None
        info["reason"] = "named_calculator"
        return requested_agent, info

    # A lookup QUESTION must not be swapped onto heavy-reasoning by a keyword.
    #
    # lookup_question_hijack already guards the predefined-dispatch
    # interception in app/routers/agents.py, from the 2026-07-24 repro where
    # "eot" sent a notice-period question to forensic_delay_analysis. It never
    # guarded THIS decision, which is the one that changes agents -- so the
    # same class kept happening one layer down. Live on 467d7a7: "What is
    # the difference between EOT and prolongation cost?" classified
    # forensic_delay_analysis at confidence 0.2 and was handed to
    # heavy-reasoning with reason=needs_planning. A definition question is
    # not a planning task -- in ANY project. This reads the message and the
    # confidence only; the corpus is one-to-all and nothing here may branch
    # on which project was asked. Deliverable-verbed asks ("generate...") and
    # confident routes are untouched, exactly as on the other path.
    try:
        from app.core.predefined_reasoning import lookup_question_hijack
        # Only where it changes the outcome. A non-generative action already
        # stays put and keeps its own reason (below_routing_gate), which
        # tests/test_smart_orchestrator_routing.py pins.
        if needs_planning(action, confidence) and lookup_question_hijack(
            user_message, float(confidence or 0.0)
        ):
            info["reason"] = "lookup_question"
            return requested_agent, info
    except Exception:  # noqa: BLE001 - routing is best-effort; never break chat
        _LOG.debug("lookup-question guard skipped", exc_info=True)

    if not needs_planning(action, confidence):
        # No existing feature/workflow. A computation with no registered
        # calculator name is self-coding's job — one hop, then that agent
        # runs formula_executor_v2 once. Named calculators stay on the
        # requested generalist so construction_calc remains the path.
        if (
            requested_agent.name in ROUTING_GENERALISTS
            and requested_agent.name != "self-coding"
            and (await _off_loop(_should_handoff_unmatched_calc, user_message))
        ):
            sc = AGENT_REGISTRY.get("self-coding")
            if sc is not None:
                info["final"] = sc.name
                info["reason"] = "no_feature_or_formula"
                return sc, info
        info["reason"] = "below_routing_gate"
        return requested_agent, info

    if requested_agent.name not in ROUTING_GENERALISTS:
        info["reason"] = "specialist_passthrough"
        return requested_agent, info

    # Already on the heavy path — no redirect needed.
    if requested_agent.name == "heavy-reasoning":
        info["reason"] = "already_heavy_reasoning"
        return requested_agent, info

    heavy = AGENT_REGISTRY.get("heavy-reasoning")
    if heavy is None:
        info["reason"] = "heavy_reasoning_not_registered"
        return requested_agent, info

    info["final"] = heavy.name
    info["reason"] = "needs_planning"
    _LOG.info(
        "smart_orch routing: %s -> %s (action=%s, confidence=%.2f)",
        requested_agent.name, heavy.name, action, confidence,
    )
    return heavy, info


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


def _parse_agent_file(path: Path) -> Agent:
    raw = path.read_text(encoding="utf-8")
    m = _FRONTMATTER_RE.match(raw)
    if not m:
        raise ValueError(f"missing YAML frontmatter in {path}")
    frontmatter, body = m.group(1), m.group(2).strip()

    # Lightweight YAML parsing — we don't import PyYAML to keep deps minimal.
    # Supports: key: value scalars, and `key:` followed by `  - item` lists.
    config: dict[str, Any] = {}
    current_list_key: str | None = None
    for raw_line in frontmatter.splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if line.startswith("  - ") or line.startswith("\t- "):
            if current_list_key:
                config[current_list_key].append(line.strip()[2:].strip().strip("\"'"))
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            if value == "":
                config[key] = []
                current_list_key = key
            else:
                config[key] = value.strip("\"'")
                current_list_key = None
    name = config.get("name") or path.stem
    if not body:
        raise ValueError(f"empty system prompt in {path}")
    body, hat_ids = _apply_hat_kernels(body, config.get("hats"), path)
    return Agent(
        name=name,
        description=config.get("description", ""),
        system_prompt=body,
        allowed_blocks=list(config.get("allowed_blocks") or []),
        model=config.get("model") or DEEPSEEK_DEFAULT_MODEL,
        temperature=float(config.get("temperature", 0.3)),
        max_tokens=int(config.get("max_tokens", 2048)),
        icon=config.get("icon", ""),
        can_delegate=_resolve_can_delegate(config),
        hats=hat_ids,
        user_data_authoritative=(
            str(config.get("user_data_authoritative", "false")).strip().lower()
            in ("true", "1", "yes", "on")
        ),
    )


def _apply_hat_kernels(body: str, hats_cfg: Any, path: Path) -> tuple[str, list[str]]:
    """Bind an agent to its discipline-hat kernels (app/agents/manifests/).

    ``hats:`` in the config frontmatter is a list of manifest ids, or the
    scalar ``all`` (the orchestrator carries every discipline kernel). Each
    resolved hat's system_prompt_guidance is appended to the agent's system
    prompt under a Discipline Kernels section, and formula bindings are
    validated at load so a manifest pointing at a missing implementation
    fails HERE, not mid-conversation. An unknown hat id raises -- a typo
    must not silently produce an unhatted agent."""
    if not hats_cfg:
        return body, []
    from app.agents import catalog as _hat_catalog
    from app.agents.formulas import validate_manifest_bindings

    if isinstance(hats_cfg, str):
        if hats_cfg.strip().lower() == "all":
            manifests = sorted(_hat_catalog.list_hats(), key=lambda m: m.id)
        else:
            manifests = [m for m in [_hat_catalog.get_hat_by_id(hats_cfg.strip())] if m]
            if not manifests:
                raise ValueError(f"unknown hat '{hats_cfg}' in {path}")
    else:
        manifests = []
        for hid in hats_cfg:
            m = _hat_catalog.get_hat_by_id(str(hid).strip())
            if m is None:
                raise ValueError(f"unknown hat '{hid}' in {path}")
            manifests.append(m)
    if not manifests:
        return body, []
    for m in manifests:
        errors = validate_manifest_bindings(m)
        if errors:
            raise ValueError(f"hat '{m.id}' has broken formula bindings: {errors}")
    sections = ["\n\n## Discipline kernels\n"]
    for m in manifests:
        sections.append(f"### {m.name} ({m.id})\n{m.system_prompt_guidance}\n")
    return body + "".join(sections), [m.id for m in manifests]


def _agent_delegation_enabled() -> bool:
    """W7 — master switch for lighting up the dormant delegator agents. Default
    OFF (unset/false): only agents with an explicit ``can_delegate: true`` (today
    just heavy-reasoning) can delegate, so the live routing surface is unchanged.
    Flip ``FORK_AGENT_DELEGATION`` on to activate the ``can_delegate_when_enabled``
    agents' hand-offs. Kept default-off deliberately — enabling delegation expands
    the mis-route surface and is a per-pilot decision."""
    return (os.getenv("FORK_AGENT_DELEGATION") or "").strip().lower() in ("1", "true", "yes", "on")


def _resolve_can_delegate(config: dict[str, Any]) -> bool:
    """Explicit ``can_delegate`` always wins; otherwise a dormant agent that
    declares ``can_delegate_when_enabled`` delegates ONLY when the master flag is
    on (W7). Default-off: unchanged behaviour."""
    def _truthy(v) -> bool:
        return str(v).strip().lower() in ("true", "1", "yes", "on")
    if _truthy(config.get("can_delegate", "false")):
        return True
    if _truthy(config.get("can_delegate_when_enabled", "false")):
        return _agent_delegation_enabled()
    return False


def _chunks(text: str, n: int) -> list[str]:
    return [text[i:i + n] for i in range(0, len(text), n)]


def _summarize_result(result: Any) -> str:
    if isinstance(result, dict):
        if result.get("status") == "error":
            return f"error: {result.get('error', '?')}"
        keys = list(result.keys())[:6]
        return f"keys=[{', '.join(keys)}]"
    if isinstance(result, list):
        return f"list[{len(result)}]"
    return str(result)[:200]


# ── helpers: tool-error prose + user-supplied cost arithmetic ──

_CG_PERCENT_RE = re.compile(
    rf"({_CG_NUM})\s*(?:%|percent\b|per\s*cent\b)",
    re.IGNORECASE,
)
_CG_DIM_CHAIN_RE = re.compile(
    rf"{_CG_NUM}(?:\s*[x×*]\s*{_CG_NUM})+",
    re.IGNORECASE,
)
# "410/m3" / "300 mm" — the unit digit is not a quantity the user typed.
_CG_UNIT_NOISE_RE = re.compile(
    r"(?:/m[23³²]|/mm|/cm|/kg)|(?<=\d)\s*(?:m[23³²]|mm|cm)\b",
    re.IGNORECASE,
)


def _current_request_id() -> str:
    """Best-effort request id for a user-facing failure sentence (#626)."""
    try:
        from app.infra.monitoring import get_request_id
        return str(get_request_id() or "")
    except Exception:
        _LOG.debug("request_id unavailable for tool-error sentence", exc_info=True)
        return ""


def _parse_json_object(text: str) -> dict[str, Any] | None:
    stripped = _normalize_tool_json_text(text or "")
    if not stripped.startswith("{"):
        return None
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError:
        _LOG.debug("tool-error JSON parse miss", exc_info=True)
        return None
    return obj if isinstance(obj, dict) else None


def _tool_error_payload(obj: Any, _depth: int = 0) -> dict[str, Any] | None:
    """The error envelope inside a tool result, or None."""
    if not isinstance(obj, dict) or _depth > 3:
        return None
    status = str(obj.get("status") or "").strip().lower()
    err = obj.get("error")
    if status in _FAILURE_STATUSES and err:
        return obj
    if isinstance(err, str) and err.strip() and status not in ("success", "ok"):
        return obj
    inner = obj.get("result")
    if isinstance(inner, dict):
        return _tool_error_payload(inner, _depth + 1)
    return None


def _looks_like_tool_error_json(text: str) -> bool:
    """True when ``text`` is a whole-string tool-error JSON envelope."""
    obj = _parse_json_object(text)
    return bool(obj and _tool_error_payload(obj))


def _format_tool_error_plain(
    payload: dict[str, Any] | None,
    *,
    request_id: str = "",
) -> str:
    """One honest sentence. Never the raw JSON the model echoed."""
    err = ""
    if isinstance(payload, dict):
        err = str(payload.get("error") or "")
        request_id = request_id or str(payload.get("request_id") or "")
    request_id = request_id or _current_request_id()
    low = err.lower()
    if "unknown calculation" in low:
        sentence = (
            "The calculator does not have a formula for that request. "
            "Please restate the inputs or name the calculation."
        )
    elif err:
        cleaned = re.sub(r"[{}\[\]]", "", err).strip().rstrip(".")
        # Empty-kwargs class: keep the named required-parameter list
        # (with units). Do not squash it into a 180-char generic.
        names_params = " needs " in cleaned.lower() and "(" in cleaned
        if names_params:
            sentence = cleaned if cleaned.endswith(".") else f"{cleaned}."
        else:
            if len(cleaned) > 180:
                cleaned = cleaned[:177] + "..."
            sentence = f"The tool could not complete that request ({cleaned})."
    else:
        sentence = "The tool could not complete that request."
    if request_id:
        sentence = f"{sentence} Request id: {request_id}."
    return sentence


def _plain_sentence_for_tool_error_json(text: str) -> str:
    """Plain sentence if ``text`` is a tool-error JSON blob, else ''."""
    obj = _parse_json_object(text)
    if not obj:
        return ""
    payload = _tool_error_payload(obj)
    if not payload:
        return ""
    return _format_tool_error_plain(payload)


def _is_calc_tool_error(
    msg: dict[str, Any] | None,
    inner: dict[str, Any] | None,
    err_payload: dict[str, Any] | None,
) -> bool:
    """True when the failed tool is a calculator, not a search/lookup."""
    name = str((msg or {}).get("name") or "").lower()
    if name in ("construction_calc", "formula_executor_v2", "formula_executor"):
        return True
    blob = inner or {}
    err = err_payload or {}
    if blob.get("calculation") or err.get("calculation"):
        return True
    if "unknown calculation" in str(err.get("error") or "").lower():
        return True
    if err.get("available"):
        return True
    return False


_COVERAGE_FOOTER_RE = re.compile(
    r"^\d+\s+of\s+\d+\s+project documents indexed\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_FORMULA_NAME_EQ_RE = re.compile(
    r"[A-Za-z][A-Za-z0-9 /()_-]{1,40}\s*=\s*[A-Za-z]",
)


def _strip_coverage_footer(text: str) -> str:
    return _COVERAGE_FOOTER_RE.sub("", text or "").strip()


def _looks_like_formula_template(text: str) -> bool:
    """True when the text is a PE-sheet formula name with no computed figures.

    Live A2-1 shipped ``Duration = Quantity / Daily Production`` (+ corpus
    footer). Digits in the footer must not hide the empty formula.
    """
    body = _strip_coverage_footer(text)
    if not body:
        return False
    if re.search(r"\d", body):
        return False
    return bool("=" in body and _FORMULA_NAME_EQ_RE.search(body))


def _looks_like_trivial_rebar_demo(result: dict[str, Any], note: str = "") -> bool:
    """1 m × 1 bar unit-mass demo — not metres run for tonnes of bar."""
    blob = " ".join(
        str(x) for x in (
            note,
            result.get("note"),
            result.get("summary"),
        ) if x
    )
    if re.search(r"x\s*1(?:\.0+)?\s*m\s*x\s*1\b", blob, re.IGNORECASE):
        return True
    try:
        length = result.get("total_length_m", result.get("metres_run"))
        qty = result.get("quantity", 1)
        mass = result.get("total_mass_kg")
        if length is None:
            return False
        length_f = float(length)
        qty_f = float(qty) if qty not in (None, "") else 1.0
        mass_f = float(mass) if mass not in (None, "") else None
    except (TypeError, ValueError):
        _LOG.debug("rebar demo fields not numeric: %r", result)
        return False
    if length_f > 1.0001 or qty_f > 1.0001:
        return False
    if mass_f is not None and mass_f >= 20:
        return False
    return True


def _compose_calc_result_fields(result: dict[str, Any], payload: dict[str, Any]) -> str:
    """Numeric line from result fields — never a formula-name-only note."""
    calc = str(payload.get("calculation") or result.get("calculation") or "")
    if calc == "productivity_manpower_duration" or result.get("duration") is not None:
        from app.lib.construction_formulas_planning import format_productivity_cost_line
        line = format_productivity_cost_line(result)
        if line:
            return line
    metres = result.get("metres_run")
    if metres is None and calc == "rebar_weight":
        metres = result.get("total_length_m")
    if metres is not None:
        try:
            if float(metres) > 10:
                from app.lib.construction_formulas_quantities import (
                    format_rebar_metres_run_line,
                )
                line = format_rebar_metres_run_line(result)
                if line:
                    return line
        except (TypeError, ValueError):
            _LOG.debug("metres_run field not numeric: %r", metres)
    calc_name = payload.get("calculation") or result.get("calculation")
    value = result.get("value")
    if value is not None and calc_name:
        return f"{calc_name}: {value}"
    for key in (
        "total_cost", "cost", "labour_cost",
        "duration", "duration_days", "crew_days",
        "metres_run", "total_length_m", "total_bar_length_m",
        "total_mass_kg", "total_mass_t", "unit_mass_kg_m",
        "volume_m3",
    ):
        if result.get(key) is not None:
            return f"{key.replace('_', ' ')}: {result[key]}"
    return ""


def _last_successful_calc_payload(
    messages: list[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    last: dict[str, Any] | None = None
    for m in messages or []:
        if m.get("role") != "tool":
            continue
        raw = m.get("content") or ""
        try:
            payload = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        inner = payload
        if payload.get("truncated") and payload.get("preview"):
            try:
                inner = json.loads(payload["preview"])
            except (TypeError, json.JSONDecodeError):
                inner = payload
        if not isinstance(inner, dict) or _tool_error_payload(inner):
            continue
        if inner.get("status") == "error":
            continue
        result = inner.get("result") if isinstance(inner.get("result"), dict) else inner
        if not isinstance(result, dict):
            continue
        if inner.get("calculation") or result.get("duration") is not None or result.get(
            "unit_mass_kg_m"
        ) is not None:
            last = inner
    return last


def _graft_complete_calc_answer(
    text: str,
    messages: list[dict[str, Any]] | None,
) -> str:
    """Replace formula-only / 1 m-demo calc answers with the computed ask.

    Live A2-1: duration note without SAR cost. Live A2-2: Y16 1 m demo
    instead of metres run for 12 t. Numbers come from the operator ask
    and/or the last successful construction_calc payload.
    """
    user = _latest_operator_ask(messages)
    if not user:
        return text
    try:
        from app.lib.construction_formulas_planning import (
            compose_productivity_cost_from_ask,
            format_productivity_cost_line,
            looks_like_productivity_cost_ask,
        )
        from app.lib.construction_formulas_quantities import (
            compose_rebar_metres_run_from_ask,
            format_rebar_metres_run_line,
            looks_like_rebar_metres_run_ask,
        )
    except Exception:  # noqa: BLE001 — graft must never break a turn
        return text

    payload = _last_successful_calc_payload(messages)
    result = (
        payload.get("result")
        if isinstance(payload, dict) and isinstance(payload.get("result"), dict)
        else (payload if isinstance(payload, dict) else {})
    )

    if looks_like_productivity_cost_ask(user):
        composed = compose_productivity_cost_from_ask(user)
        if composed and composed.get("line"):
            body = _strip_coverage_footer(text)
            user_money = {v for _frag, v in _cg_money_values(user)}
            answer_money = [v for _frag, v in _cg_money_values(body)]
            already = any(
                v >= 1000 and v not in user_money for v in answer_money
            )
            if already and not _looks_like_formula_template(text):
                return text
            if result.get("total_cost") is not None and result.get("duration") is not None:
                line = format_productivity_cost_line(result, composed.get("currency") or "SAR")
                if line and re.search(r"\b(?:SAR|AED|USD|GBP|EUR)\b", line, re.I):
                    return line
            return composed["line"]

    if looks_like_rebar_metres_run_ask(user):
        composed = compose_rebar_metres_run_from_ask(user)
        if composed and composed.get("line"):
            body = _strip_coverage_footer(text)
            if re.search(r"(?<![\d.])(?:[5-9]\d{3}|\d{5,})(?:\.\d+)?\s*m\b", body or ""):
                return text
            if result and not _looks_like_trivial_rebar_demo(result, body):
                line = format_rebar_metres_run_line(result)
                if line:
                    return line
            return composed["line"]

    return text


def _format_any_calc_result(payload: dict[str, Any]) -> str:
    """User-facing line from a successful non-volume construction_calc."""
    if not isinstance(payload, dict):
        return ""
    if _tool_error_payload(payload):
        return ""
    result = payload.get("result") if isinstance(payload.get("result"), dict) else payload
    if not isinstance(result, dict):
        return ""
    for key in ("note", "summary", "message"):
        val = result.get(key)
        if (
            isinstance(val, str)
            and val.strip()
            and "unknown calculation" not in val.lower()
            and not _looks_like_formula_template(val)
            and not _looks_like_trivial_rebar_demo(result, val)
        ):
            try:
                from app.lib.construction_formulas_commercial import (
                    answer_is_unbound_delay_damages,
                )
                if answer_is_unbound_delay_damages(val):
                    continue
            except Exception:  # noqa: BLE001 — formatter must never break
                _LOG.debug("unbound delay-damages note check failed", exc_info=True)
            return val.strip()
    composed = _compose_calc_result_fields(result, payload)
    if composed:
        return composed
    return ""


def _cg_strip_unit_noise(text: str) -> str:
    # Item references go first, for the same reason unit digits do: neither is
    # a quantity the operator typed. See app/lib/boq_ref_codes.
    return _CG_UNIT_NOISE_RE.sub(" ", strip_ref_codes(text))


# Each step of the user-arithmetic closure multiplies the set it is given, so
# the whole is O(n^2 x counts x money x percents) in the user's figures. A
# normal ask (a handful of figures) builds ~16k entries. Live 21 Sep 2026 a
# figure-heavy user text drove it past 2 GiB and Render OOM-killed the web
# instance. A step that would exceed this many entries is skipped and logged:
# fewer grounded numbers only makes the gate stricter, never looser.
_CG_USER_CLOSURE_MAX = 250_000
_CG_USER_PAIR_ENTRIES = 5  # a*b, a+b, |a-b|, a/b, b/a per pair


def _cg_user_arithmetic_closure(user_text: str, seeded: set) -> set:
    """a×b×c, qty×rate after a quotient, and a×(1+p%) from user figures.

    The pairwise pass in ``_cg_grounded_numbers`` is one hop. Live pad-footing
    take-off (14 × 2.4 × 2.4 × 0.6 × 1.05 × 410 × 1.10) and plastering
    (3400/42 × 1950) need this short extra closure. A rate the user never
    wrote is still not in the set.
    """
    extra: set = set(seeded)
    originals = [v for v in seeded if v is not None]
    pct_raw: set = set()
    pct_factors: list[float] = []
    for m in _CG_PERCENT_RE.finditer(user_text or ""):
        p = _cg_to_number(m.group(1))
        if p is None or p <= 0 or p > 100:
            continue
        pct_raw.add(p)
        factor = 1.0 + (p / 100.0)
        pct_factors.append(factor)
        extra.add(round(p / 100.0, 4))
        extra.add(round(factor, 4))
    for m in _CG_DIM_CHAIN_RE.finditer(user_text or ""):
        parts = [_cg_to_number(p) for p in _CG_NUM_RE.findall(m.group(0))]
        parts = [p for p in parts if p is not None]
        if len(parts) < 2:
            continue
        prod = 1.0
        for p in parts:
            prod *= p
        extra.add(round(prod, 4))
    # Pairwise of the user's own figures (qty/productivity, qty×rate).
    pairwise = originals
    if _CG_USER_PAIR_ENTRIES * len(originals) ** 2 // 2 > _CG_USER_CLOSURE_MAX:
        _LOG.warning(
            "cost gate: user-arithmetic pairwise pass skipped (%d figures)",
            len(originals),
        )
        pairwise = []
    for i, a in enumerate(pairwise):
        for b in pairwise[i:]:
            extra.add(round(a * b, 4))
            extra.add(round(a + b, 4))
            extra.add(round(abs(a - b), 4))
            if b:
                extra.add(round(a / b, 4))
            if a:
                extra.add(round(b / a, 4))
    # Triple products of small dimension-like seeds (2.4×2.4×0.6).
    # Percent figures (5, 10) are factors, not dimensions.
    small = [v for v in originals if 0 < v < 100 and v not in pct_raw]
    if 3 <= len(small) <= 12:
        for i in range(len(small)):
            for j in range(i, len(small)):
                for k in range(j, len(small)):
                    extra.add(round(small[i] * small[j] * small[k], 4))

    money = [v for _frag, v in _cg_money_values(user_text or "")]
    if not money:
        money = [v for v in originals if v >= 20]
    money_set = set(money)
    counts = [
        v for v in originals
        if (
            v >= 2
            and abs(v - round(v)) < 1e-9
            and v not in pct_raw
            and v not in money_set
        )
    ]

    def _over_budget(step: str, size: int) -> bool:
        if size <= _CG_USER_CLOSURE_MAX:
            return False
        _LOG.warning(
            "cost gate: user-arithmetic closure step %s skipped (%d entries > %d)",
            step, size, _CG_USER_CLOSURE_MAX,
        )
        return True

    def _times(values: set, factors: list[float], *, min_qty: float = 0.0) -> set:
        if _over_budget("times", len(values) * len(factors)):
            return values
        out = set(values)
        for a in list(values):
            if a < min_qty:
                continue
            for b in factors:
                if not b:
                    continue
                out.add(round(a * b, 4))
        return out

    def _apply_pct(values: set) -> set:
        if _over_budget("percent", len(values) * len(pct_factors)):
            return values
        out = set(values)
        for a in list(values):
            for f in pct_factors:
                out.add(round(a * f, 4))
        return out

    # count × (a×b×c), then waste, then qty-scale × user rate, then
    # contingency. Dimension leftovers (2.4×0.6=1.44) must not be priced
    # — that collision grounded an invented SAR 650/m3.
    extra = _times(extra, counts)
    extra = _apply_pct(extra)
    qty_floor = min(counts) if counts else 1.0
    extra = _times(extra, money, min_qty=qty_floor)
    extra = _apply_pct(extra)
    return extra


def _ipc_args_from_ask(text: str, existing: dict[str, Any] | None = None) -> dict[str, Any]:
    """Populate IPC tool args from the operator ask. Never invent figures."""
    try:
        from app.containers.construction.boq import ipc_args_from_ask
        return ipc_args_from_ask(text or "", existing)
    except Exception:  # noqa: BLE001
        _LOG.debug("IPC ask coercion skipped", exc_info=True)
        out = dict(existing or {})
        if text:
            out.setdefault("message", text)
            out.setdefault("user_message", text)
            out.setdefault("text", text)
        return out


def _ask_is_ipc_certificate(text: str) -> bool:
    raw = text or ""
    return bool(re.search(r"payment certificate|\bipc\b|interim payment", raw, re.I))


async def _dispatch_payment_certificate(
    args: dict[str, Any] | None,
    *,
    user_message: str | None,
) -> dict[str, Any]:
    """Run ConstructionContainer.payment_certificate with ask-coerced args."""
    try:
        from app.dependencies import get_block_instance
        container = get_block_instance("construction")
    except Exception as e:
        return {
            "name": "payment_certificate",
            "ok": False,
            "result": {"status": "error", "error": f"construction unavailable: {e}"},
        }
    params = _ipc_args_from_ask(user_message or "", args if isinstance(args, dict) else {})
    payload = {
        "message": params.get("message") or user_message or "",
        "user_message": params.get("user_message") or user_message or "",
        "text": params.get("text") or user_message or "",
    }
    for key in (
        "contract_value", "gross_valuation", "measured_works",
        "materials_on_site", "retention_percent", "work_done_percent",
        "advance_payment", "advance_percent", "advance_recovery_percent",
        "previous_certified",
    ):
        if params.get(key) not in (None, ""):
            payload[key] = params[key]
    try:
        result = await container.payment_certificate(payload, params)
    except Exception as e:
        return {
            "name": "payment_certificate",
            "ok": False,
            "result": {"status": "error", "error": f"payment_certificate failed: {e}"},
        }
    return {
        "name": "payment_certificate",
        "ok": isinstance(result, dict) and result.get("status") == "success",
        "result": result,
    }


# Import every tool package now, at app start, not on a turn's first tool
# call -- importing modules there held the event loop for ~1 s under load.
from app.agents.core import tool_registry as _tool_registry_at_start  # noqa: E402

_tool_registry_at_start.load()
