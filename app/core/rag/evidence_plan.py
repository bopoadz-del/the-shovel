"""Evidence-plan + coverage check for project-figure retrieval (item 2).

The answer agent writes a short evidence plan (which documents / sections /
rows would answer the question), then retrieval actions fetch evidence in
several shapes. ``missing_plan_items`` compares what came back against the
plan so a plan item with no evidence triggers another round, bounded by
``ROUND_LIMIT`` (round 4 never runs — the spec's hard bound).

These are the PURE, unit-tested pieces. The LLM loop that generates a plan
and dispatches the retrieval actions lives in the runtime route and is built
on top of these; nothing here knows a project, an item name or a question.
"""
from __future__ import annotations

from dataclasses import dataclass, field

#: Hard bound on retrieval rounds (spec): re-query up to this many times, then
#: proceed with what was found. ``round_number`` is 1-based.
ROUND_LIMIT = 3

#: Words that carry no retrieval signal and are dropped from term overlap.
#: ``item``/``row``/``section`` and the like are plan vocabulary, not evidence.
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "for", "in", "on", "at", "to",
    "with", "from", "by", "as", "is", "are", "be", "was", "were", "no",
    "not", "that", "this", "these", "those", "which", "what", "where",
    "would", "should", "could", "item", "items", "section", "sections",
    "row", "rows", "document", "documents", "sheet", "sheets", "clause",
    "clauses", "need", "needs", "needed", "answer", "answers",
})


@dataclass(frozen=True)
class EvidencePlan:
    """One parsed evidence plan: the raw text and its items, in order."""
    text: str
    items: tuple[str, ...] = field(default_factory=tuple)


def _strip_marker(line: str) -> str:
    """Drop a leading bullet / number marker: ``-``, ``*``, ``1.``, ``2)``."""
    text = line.strip()
    if text.startswith(("- ", "* ")):
        return text[2:].strip()
    for i, ch in enumerate(text):
        if ch in ".)" and text[:i].isdigit():
            return text[i + 1:].strip()
        if not (ch.isdigit() or ch in ".()"):
            break
    return text


def parse_evidence_plan(text: str) -> EvidencePlan:
    """Split a plan into items.

    A plan is one item per line (bulleted, numbered, or plain); a single
    plain line may itself be a comma-separated list of needs, which is split
    so a plan item maps to one thing to check. Markers and empty lines are
    dropped.
    """
    lines = []
    for raw in (text or "").splitlines():
        item = _strip_marker(raw)
        if item:
            lines.append(item)
    # A single line whose needs are comma-separated is one item per comma.
    if len(lines) == 1 and "," in lines[0]:
        lines = [part.strip() for part in lines[0].split(",") if part.strip()]
    return EvidencePlan(text=(text or "").strip(), items=tuple(lines))


def _terms(phrase: str) -> frozenset[str]:
    """The non-stopword words of ``phrase``, lowercased, for overlap."""
    out = set()
    for token in phrase.lower().split():
        word = token.strip(" ,;:()[]\"'#")
        if word and word not in _STOPWORDS:
            out.add(word)
    return frozenset(out)


def missing_plan_items(plan: EvidencePlan, evidence_text: str) -> tuple[str, ...]:
    """Plan items with no evidence in ``evidence_text``.

    Coverage is lexical and conservative: an item is covered only when EVERY
    one of its salient terms appears in the retrieved evidence. If any term is
    absent the item is a gap and triggers another round. This errs toward
    re-querying (a bounded extra round), never toward silently accepting a
    missing row — a false negative on coverage would drop the very row the
    loop exists to fetch.
    """
    evidence = " ".join((evidence_text or "").lower().split())
    missing = []
    for item in plan.items:
        terms = _terms(item)
        if not terms:
            continue  # an empty plan item is not a gap; it carries no signal
        if not all(term in evidence for term in terms):
            missing.append(item)
    return tuple(missing)


def round_allowed(round_number: int) -> bool:
    """True when ``round_number`` may still run: 1..ROUND_LIMIT. Round 4 never."""
    return 1 <= round_number <= ROUND_LIMIT
