"""Conversation-answer report export — not an RFP / attachment lookup.

A live answer-report export ask (Master Corpus / theshovel.ai), exactly::

    Export Q1-Q9 answers as a docx report

Expected: a downloadable Word file of the Q1–Q9 *chat answers*, with a
real page footer carrying the live URL, figures intact.

Actual on ``567147a``: no docx landed. ``Q1-Q9`` is identifier-shaped
(``extract_query_identifiers`` keeps ``q1-q9``), retrieval treated it as
RFP appendix / attachment codes, and the turn listed those documents
instead of compiling the conversation.

This module is the steal-guard + pair collector. Rendering lives in
``app/routers/exports.py``. Kill-switch: ``ANSWER_REPORT_EXPORT=0``.
"""

from __future__ import annotations

import os
import re
from typing import Any, Iterable, Optional

# Q1-Q5 / B2 to B4 / K1–K3: the user's own one-letter label for the turns,
# any letter, the same letter (or none) on the second number. Not a file code.
_ANSWER_RANGE_RE = re.compile(
    r"\b(?P<label>[A-Z])\s*(?P<start>\d{1,2})\s*(?:[-–—]|to)\s*(?:(?P=label))?\s*(?P<end>\d{1,2})\b",
    re.IGNORECASE,
)

_EXPORT_VERB_RE = re.compile(
    r"\b(export|download|save|send\s+me|give\s+me)\b",
    re.IGNORECASE,
)

_DOCX_OR_REPORT_RE = re.compile(
    r"\b(docx|word|doc|report)\b",
    re.IGNORECASE,
)

_ANSWERS_OR_CONVO_RE = re.compile(
    r"\b(answers?|conversation|thread|chat)\b",
    re.IGNORECASE,
)

# RFP / tender attachment listing — the live misroute. An export that
# names those packages is not an answer-range report, unless it also
# explicitly asks for chat *answers*.
_RFP_ATTACH_RE = re.compile(
    r"\b(rfp|request\s+for\s+proposal|invitation\s+to\s+tender|"
    r"tender\s+package|attachments?)\b",
    re.IGNORECASE,
)

# Confirmation the short-circuit itself persisted — skip on re-export.
_REPORT_READY_RE = re.compile(
    r"prepared a word report of answers|conversation export, not an rfp",
    re.IGNORECASE,
)

# Keyword-router actions that steal this ask onto RFP / document listing.
ANSWER_REPORT_BLOCKED_ACTIONS = frozenset({
    "rfp_draft",
    "rfp_management",
    "tender_bid_analysis",
    "job_requisition",
    "process_document",
    "daily_site_report",
    "as_built_deviation_report",
    "esg_sustainability_report",
    "submittal_log_generator",
})


def answer_report_export_enabled() -> bool:
    """Kill-switch. Default on; ``ANSWER_REPORT_EXPORT=0`` restores 567147a."""
    return (os.getenv("ANSWER_REPORT_EXPORT") or "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


def parse_answer_report_range(text: str) -> Optional[tuple[int, int]]:
    """Return 1-based ``(start, end)`` for ``Q1-Q5`` / ``1-5``, or None."""
    raw = text or ""
    m = _ANSWER_RANGE_RE.search(raw)
    if m:
        start, end = int(m.group("start")), int(m.group("end"))
    else:
        m = re.search(r"\b(\d{1,2})\s*[-–—]\s*(\d{1,2})\b", raw)
        if not m:
            return None
        start, end = int(m.group(1)), int(m.group(2))
    if start > end:
        start, end = end, start
    start = max(1, min(start, 99))
    end = max(1, min(end, 99))
    return start, end


def message_wants_answer_report(text: str) -> bool:
    """True when the turn wants a docx of conversation answers, not files.

    Positive: the live export ask; 'export this conversation as a Word
    document'; 'save answers B2 to B4 as docx'.

    Negative: RFP/attachment listing; a single-letter code lookup
    ('what does Q1 of the RFP say'); generic 'give me a docx copy'
    (last-message UI button); a Contract Data question (who is the Engineer).
    """
    raw = text or ""
    if not raw.strip():
        return False
    if not _EXPORT_VERB_RE.search(raw):
        return False
    if not _DOCX_OR_REPORT_RE.search(raw) and not (
        _ANSWER_RANGE_RE.search(raw) and _ANSWERS_OR_CONVO_RE.search(raw)
    ):
        return False
    if _RFP_ATTACH_RE.search(raw) and not _ANSWERS_OR_CONVO_RE.search(raw):
        return False
    # A labelled range alone ("S1-S5") may be sheets or items; it is an answer
    # report only when the turn also speaks of answers / the conversation.
    return bool(_ANSWERS_OR_CONVO_RE.search(raw))


def export_workspace_project_id(
    project_id: str,
    conversation_id: str = "",
) -> str:
    """UI/workspace project id for an export URL — never the RAG backing id.

    Chat remaps ``master_corpus`` → the master-corpus source project
    before the agent runs. Stamping that remapped id into
    ``/v1/projects/{id}/conversations/.../export`` 404s
    (``Project '<source id>' not found``). Prefer the ``ws-{ui_pid}``
    conversation prefix, then reverse-map the master-corpus source.
    """
    from app.core.projects import MASTER_CORPUS_PROJECT_ID, ui_project_id

    cid = conversation_id or ""
    if cid.startswith("ws-"):
        rest = cid[3:]
        if rest == MASTER_CORPUS_PROJECT_ID:
            return MASTER_CORPUS_PROJECT_ID
        if "-" in rest:
            pid, suffix = rest.rsplit("-", 1)
            if pid == MASTER_CORPUS_PROJECT_ID and suffix:
                return MASTER_CORPUS_PROJECT_ID
    return ui_project_id(project_id) or project_id


def answer_report_export_descriptor(
    project_id: str,
    conversation_id: str,
    start: Optional[int],
    end: Optional[int],
) -> dict[str, Any]:
    """SSE ``exports`` offer the frontend already knows how to click."""
    scoped = export_workspace_project_id(project_id, conversation_id)
    label = range_label(start, end)
    qs = "format=docx&scope=answers"
    if start is not None and end is not None:
        qs += f"&range=A{start}-A{end}"
    return {
        "label": f"{label} answers (Word)" if label != "Answers" else "Conversation answers (Word)",
        "format": "docx",
        "method": "POST",
        "endpoint": (
            f"/v1/projects/{scoped}/conversations/{conversation_id}/export?{qs}"
        ),
        "payload": {},
    }


def compose_answer_report_reply(
    pairs: list[dict[str, str]],
    start: Optional[int],
    end: Optional[int],
) -> str:
    """Honest confirmation — this is a conversation export, not RFP files."""
    if not pairs:
        return (
            "There are no prior answers in this conversation to export. "
            "Ask the questions first, then export the answers (e.g. Q1–Q5) as a docx report."
        )
    first = pairs[0]["label"]
    last = pairs[-1]["label"]
    wanted = range_label(start, end)
    have = f"{first}–{last}" if first != last else first
    extra = ""
    if start is not None and end is not None and len(pairs) < (end - start + 1):
        extra = (
            f" This conversation has {len(pairs)} assistant answer(s), so "
            f"the report includes {have} ({wanted} was asked)."
        )
    return (
        f"Prepared a Word report of answers {have} from this conversation. "
        "Download it with the button below. Figures are copied as answered; "
        "the page footer carries the live site URL.\n\n"
        "This is a conversation export, not an RFP or attachment listing."
        + extra
    )


def range_label(start: Optional[int], end: Optional[int]) -> str:
    if start is None or end is None:
        return "Answers"
    if start == end:
        return f"A{start}"
    return f"A{start}–A{end}"


def collect_answer_pairs(
    messages: Iterable[dict[str, Any]],
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> list[dict[str, str]]:
    """Pair user questions with assistant answers; skip export asks.

    Labels are ``A1``… in conversation order (then sliced to ``start``–``end``).
    The current H1 export ask and a prior 'report is ready' confirmation
    are not answers.
    """
    pairs: list[dict[str, str]] = []
    pending_q = ""
    for raw in messages or []:
        role = (raw.get("role") or "").strip()
        content = (raw.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            if message_wants_answer_report(content):
                pending_q = ""
                continue
            pending_q = content
            continue
        if role != "assistant":
            continue
        if _REPORT_READY_RE.search(content):
            pending_q = ""
            continue
        pairs.append({"question": pending_q, "answer": content})
        pending_q = ""

    origin = start or 1
    if start is not None:
        i0 = max(start - 1, 0)
        i1 = end if end is not None else len(pairs)
        pairs = pairs[i0:i1]
    else:
        origin = 1

    labelled: list[dict[str, str]] = []
    for i, pair in enumerate(pairs):
        labelled.append({
            "label": f"A{origin + i}",
            "question": pair["question"],
            "answer": pair["answer"],
        })
    return labelled
