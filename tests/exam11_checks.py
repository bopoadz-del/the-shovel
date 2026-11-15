"""Exam 11 (chat workspace) checkers: pure functions over what a person sees.

Test code only. The browser exam (tests/browser/test_exam11_chat_workspace.py)
feeds them the visible text of the page; tests/test_exam11_checks.py proves
each checker passes one known-good input and fails one known-bad input
(SPEC_EXAM_PLAN rule 3: the exam is itself tested).

Each checker returns a list of findings; an empty list means the check holds.
Nothing here names a document, a row, a question or an exam case of the
product: the patterns are the kinds of text W3/W4 forbid.
"""
from __future__ import annotations

import re

# W3: citations live in the side panel, never in the answer text.
_CITATION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("source line", re.compile(r"(?im)^\s*\(?sources?\s*:")),
    ("inline source", re.compile(r"(?i)\(\s*sources?\s*:[^)]*\)")),
    ("file name", re.compile(
        r"(?i)\b[\w.\-]+\.(?:pdf|docx?|xlsx?|csv|md|txt|dwg|ifc|xer|pptx?)\b")),
    ("document id", re.compile(
        r"(?i)\b(?:doc(?:ument)?_?id\s*[:=]\s*\S+"
        r"|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})")),
    ("coverage count", re.compile(
        r"(?i)\b\d+\s+(?:of\s+\d+\s+)?documents?\s+(?:are\s+|were\s+)?"
        r"(?:indexed|searchable|searched)\b")),
)

# W4: the answer never narrates the machinery.
_PROCESS_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("machine identifier", re.compile(r"\b[a-z]+(?:_[a-z0-9]+)+\b")),
    ("machinery word", re.compile(
        r"(?i)\b(?:deterministic(?:ally)?|certified|certifying|pre-?dispatch"
        r"|tool call|calculator returned|the search returned)\b")),
    ("search narration", re.compile(
        r"(?i)\bI\s+(?:searched|looked\s+through|queried)\s+the\s+"
        r"(?:project\s+)?(?:documents|index|corpus)\b")),
)


def _scan(text: str, patterns) -> list[str]:
    found: list[str] = []
    for label, pattern in patterns:
        for match in pattern.finditer(text or ""):
            found.append(f"{label}: {match.group(0).strip()!r}")
    return found


def citation_leaks(answer_text: str) -> list[str]:
    """Citation material that appears in the answer text instead of the side panel."""
    return _scan(answer_text, _CITATION_PATTERNS)


def process_talk(answer_text: str) -> list[str]:
    """Machinery narration in the answer text."""
    return _scan(answer_text, _PROCESS_PATTERNS)


def activity_mismatch(shown_lines: list[str], ran_labels: list[str]) -> list[str]:
    """W4 activity rule over one turn.

    ``shown_lines`` is every status line the UI showed during the turn, in
    order (consecutive repeats are one line); ``ran_labels`` is the label of
    every backend event that actually ran in the turn. A shown line that
    names no event of this turn is a false line. More than one line at
    once is caught by the caller (it samples the page); here a status line
    still shown after the answer arrived is passed as the sentinel
    ``"<after answer>"`` and reported.
    """
    ran = {label.strip().lower() for label in ran_labels if label.strip()}
    findings: list[str] = []
    for line in shown_lines:
        if line == "<after answer>":
            findings.append("status line still shown after the answer arrived")
            continue
        shown = line.strip().lower().rstrip(".…")
        if not any(shown == label or (len(label) >= 4 and label in shown) for label in ran):
            findings.append(f"status line with no backend event this turn: {line!r}")
    return findings


def sse_events(stream_body: str) -> list[dict]:
    """Parsed events, in order, from a captured /v1/chat/stream body."""
    import json

    events: list[dict] = []
    for raw in (stream_body or "").splitlines():
        line = raw.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except ValueError:
            continue
        if isinstance(obj, dict) and obj.get("type"):
            events.append(obj)
    return events


def ran_labels(events: list[dict]) -> list[str]:
    """What actually ran this turn, in words a status line may use.

    A ``status`` event contributes its label; a ``tool_call`` event its
    tool name with underscores read as spaces.
    """
    labels: list[str] = []
    for evt in events:
        if evt.get("type") == "status" and isinstance(evt.get("label"), str):
            labels.append(evt["label"])
        elif evt.get("type") == "tool_call" and isinstance(evt.get("tool"), str):
            labels.append(evt["tool"].replace("_", " "))
    return labels
