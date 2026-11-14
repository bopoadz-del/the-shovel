"""Route hook for agentic figure retrieval (item 2).

The runtime asks two questions of this module per project-scoped turn:

1. ``wants_agentic_figure_retrieval`` — is this a project-figure or
   published-standard question (one that asks for a value, a rate or a
   measured quantity), as opposed to a self-contained calculation, an export
   or a deliverable? Pure: both the measured-quantity detector and the
   exclusion predicate are injected by the runtime, so no case name or
   question shape is repeated here.

2. ``augment_figure_retrieval`` — run the #163 evidence loop over the first
   pass and return the merged evidence text. The primitives (semantic /
   lexical fetch), the plan call and the sync-loop executor are injected, so
   it unit-tests with fakes and the runtime supplies ``retrieve_with_filter``,
   ``store.bm25_search``, the plan LLM call and ``off_loop``.

Nothing here names a project, a work item or a column.
"""
from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from typing import Any

from app.core.rag.evidence_loop import run_project_figure_retrieval
from app.core.rag.evidence_plan import is_platform_note

PlanCaller = Callable[[str], Awaitable[dict[str, Any]]]

# One excerpt marker as ``inject.chunk_marker`` prints it. Text before the
# first marker is the header the platform wrote (instructions, notes); the
# marker itself is the platform's label. Only the text after each marker is
# what retrieval returned.
_EXCERPT_MARKER_RE = re.compile(r"\[doc_id=[^\]\n]*\]")

# A word as the first-pass query tokeniser reads one (``inject._content_terms``),
# so a question term and an excerpt word are cut the same way.
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-/\.]*")


def _norm_word(word: str) -> str:
    return word.lower().rstrip(".-/")


def excerpt_words(evidence_text: str) -> frozenset[str]:
    """The whole words of the retrieved excerpts in ``evidence_text``.

    The header before the first excerpt marker, the markers themselves and
    every line the platform wrote (``evidence_plan.is_platform_note``) are
    dropped: words the platform supplied never show that retrieval found
    something.
    """
    bodies = _EXCERPT_MARKER_RE.split(evidence_text or "")[1:]
    words: set[str] = set()
    for body in bodies:
        for line in body.splitlines():
            if is_platform_note(line):
                continue
            words.update(_norm_word(w) for w in _WORD_RE.findall(line))
    words.discard("")
    return frozenset(words)


def question_coverage(
    question: str,
    evidence_text: str,
    *,
    content_terms: Callable[[str], Iterable[str]],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """``(terms, missing)``: the question's content terms and those the
    retrieved excerpts do not contain as whole words.

    ``content_terms`` is the tokeniser the first-pass query was built with,
    injected so the gate asks exactly "did the first pass return the terms
    it searched on". A question with no content terms carries no signal and
    is never uncovered.
    """
    found = excerpt_words(evidence_text)
    terms: list[str] = []
    for raw in content_terms(question or ""):
        term = _norm_word(raw)
        if term and term not in terms:
            terms.append(term)
    return tuple(terms), tuple(t for t in terms if t not in found)

# An interrogative that asks for a value (as opposed to a procedure or a
# yes/no): "what is the rate", "how much cover", "how many lux".
_ASK_VALUE_RE = re.compile(r"(?i)\b(?:what|how\s+much|how\s+many)\b")

# Nouns that name the thing being asked FOR when the ask is a figure. These
# are general vocabulary — a value, rate, quantity or published-standard
# figure — not any one document's wording.
_FIGURE_NOUN_RE = re.compile(
    r"(?i)\b(?:rate|rates|value|values|quantity|quantities|figure|figures|"
    r"standard|standards|benchmark|benchmarks|productivity|norm|norms|"
    r"typical|normal|allowance|allowances|factor|factors|waste|cover|"
    r"thickness|density|level|levels)\b"
)

# A measured dimension, or a quantity / bill total, named as the thing asked
# for: "what is the billed area of", "how much is the total of". General
# measurement vocabulary only.
_DIMENSION_OR_TOTAL_NOUN_RE = re.compile(
    r"(?i)\b(?:area|areas|length|lengths|volume|volumes|weight|weights|mass|"
    r"tonnage|height|width|depth|perimeter|total|totals|subtotal|amount|"
    r"amounts|sum|number|count|cost|costs|price|prices)\b"
)

# An SI or trade measurement unit, only where it reads as a unit: after a
# number ("250 m2", "12 no.") or after "in"/"per" ("in m3", "per lm"). A bare
# letter elsewhere ("don't", "a t-beam") is not a unit.
_UNIT_TOKEN_RE = re.compile(
    r"(?i)(?:\d\s*|\b(?:in|per)\s+)"
    r"(?:m|m2|m3|m²|m³|lm|kg|t|nr|no\.|sqm|cum)(?![\w.\-])"
)

# A procedural question ("how do I measure the area") asks for a method,
# not a figure, even when it names a dimension.
_PROCEDURAL_RE = re.compile(
    r"(?i)^\s*how\s+(?:do|does|did|can|could|should|would|to)\b"
)


def wants_agentic_figure_retrieval(
    user_message: str,
    *,
    asks_measured_quantity: Callable[[str], bool],
    exclude: Callable[[str], bool],
) -> bool:
    """True when the question asks for a figure (a measured quantity, or a
    value/rate/standard named by an interrogative) and the caller's own
    exclusions (self-contained calculation, export, deliverable) do not apply.

    A value interrogative that names a measured dimension (area, length,
    volume ...), a quantity or bill total (total, amount, count, cost ...) or
    a measurement unit after a number ("250 m2", "in m3") is a figure ask
    too. A procedural "how do I ..." ask stays out of that second clause.

    ``asks_measured_quantity`` is the runtime's measured-quantity detector
    (cover / compaction / illuminance); ``exclude`` is the runtime's set of
    "not a retrieval ask" checks. Both are injected so this stays pure.
    """
    text = (user_message or "").strip()
    if not text or exclude(text):
        return False
    if asks_measured_quantity(text):
        return True
    if not _ASK_VALUE_RE.search(text):
        return False
    if _FIGURE_NOUN_RE.search(text):
        return True
    if _PROCEDURAL_RE.search(text):
        return False
    return bool(
        _DIMENSION_OR_TOTAL_NOUN_RE.search(text) or _UNIT_TOKEN_RE.search(text)
    )


async def augment_figure_retrieval(
    *,
    question: str,
    project_id: str,
    first_pass_evidence: str,
    plan_caller: PlanCaller,
    retrieve_fn: Callable[..., tuple],
    bm25_fn: Callable[..., Sequence],
    run_sync: Callable[..., Awaitable[Any]] | None = None,
    sheets: Mapping[str, Sequence[Sequence[str]]] | None = None,
    headers: Mapping[str, Sequence[str]] | None = None,
    outcome_sink: Callable[[Any], None] | None = None,
    typed_bills: Callable[[str], Iterable[str]] | None = None,
    bill_chunks: Callable[[str, list[str]], Sequence] | None = None,
    bill_total_status: Callable[[str], Mapping | None] | None = None,
) -> str:
    """Run the evidence loop and return the merged evidence text.

    The loop only re-queries plan items the first pass does not cover, and
    never runs round 4. On any failure the first pass is returned unchanged:
    augmentation must not break a turn.

    ``outcome_sink``, when supplied, is called once with the loop's
    ``EvidenceLoopOutcome`` (plan items, gaps, rounds) so a caller can record
    the evidence decision without changing what this returns.
    """
    try:
        outcome = await run_project_figure_retrieval(
            question=question,
            plan_caller=plan_caller,
            retrieve_fn=retrieve_fn,
            bm25_fn=bm25_fn,
            project_id=project_id,
            first_pass_evidence=first_pass_evidence,
            sheets=sheets,
            headers=headers,
            run_sync=run_sync,
            typed_bills=typed_bills,
            bill_chunks=bill_chunks,
            bill_total_status=bill_total_status,
        )
        if outcome_sink is not None:
            outcome_sink(outcome)
    except Exception:  # noqa: BLE001 -- augmentation must never break a turn
        # A retrieval-augmentation failure must never break the answer.
        return first_pass_evidence
    merged = outcome.evidence
    return merged if (merged or "").strip() else first_pass_evidence
