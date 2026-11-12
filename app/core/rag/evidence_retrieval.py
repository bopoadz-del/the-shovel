"""The evidence-fetch loop: plan -> actions -> coverage -> re-query (item 2).

Orchestrates the multi-shape retrieval over an evidence plan. It is PURE
orchestration over callables, so it is unit-testable without a live index:
the runtime supplies the retrieval actions (semantic / lexical / structural),
each ``(plan_item_text) -> evidence_text``. This module only decides which
plan items still lack evidence and bounds the rounds.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from app.core.rag.evidence_budget import (
    CONTEXT_TRUNCATED_NOTE,
    cap_evidence_text,
)
from app.core.rag.evidence_plan import (
    ROUND_LIMIT,
    EvidencePlan,
    missing_plan_items,
)

#: A retrieval action: a plan item (phrase) in, evidence text out.
RetrievalAction = Callable[[str], str]


@dataclass(frozen=True)
class RoundRecord:
    """One re-query round: what was still missing going in, what remained after."""
    number: int
    queried_items: tuple[str, ...]
    missing_after: tuple[str, ...]


@dataclass
class EvidenceRun:
    """The outcome of the loop: combined evidence, remaining gaps, and rounds."""
    plan: EvidencePlan
    evidence: str
    missing: tuple[str, ...]
    rounds: list[RoundRecord] = field(default_factory=list)
    #: Why the loop stopped: ``covered`` (nothing missing), ``no_new_evidence``
    #: (no still-missing item can yield a line not already held) or
    #: ``round_limit`` (``max_rounds`` exhausted while items still yielded).
    stop_reason: str = ""


_NOTE_LINES = frozenset(
    line.strip() for line in CONTEXT_TRUNCATED_NOTE.splitlines() if line.strip()
)


def _evidence_lines(text: str) -> set[str]:
    """The evidence lines of ``text`` (chunks/rows are newline-joined).

    The budget's truncation note is bookkeeping, not evidence, so it never
    counts as something a round fetched.
    """
    return {
        line.strip() for line in (text or "").splitlines()
        if line.strip() and line.strip() not in _NOTE_LINES
    }


def gather_evidence(
    plan: EvidencePlan,
    actions: Sequence[RetrievalAction],
    initial_evidence: str = "",
    max_rounds: int = ROUND_LIMIT,
) -> EvidenceRun:
    """Fetch evidence for a plan, re-querying missing items up to ``max_rounds``.

    ``initial_evidence`` is the first-pass retrieval text (round 1). For each
    subsequent round, every still-missing plan item is given to every action
    and the returned text is appended, then coverage is re-checked. The loop
    stops when nothing is missing, when a round fetches nothing new, or when
    ``max_rounds`` is exhausted — round ``max_rounds + 1`` never runs.

    "Nothing new" is measured against the evidence lines already held. The
    actions are deterministic over a fixed index, so an item whose actions
    returned no line not already held would return the same again: it is not
    re-queried in later rounds (it stays in ``missing``). When no missing
    item is left to query, the loop ends with ``stop_reason="no_new_evidence"``.
    """
    parts = [initial_evidence] if (initial_evidence or "").strip() else []
    missing = missing_plan_items(plan, initial_evidence or "")
    rounds: list[RoundRecord] = []
    held = _evidence_lines(initial_evidence or "")
    exhausted: set[str] = set()
    for number in range(1, max_rounds + 1):
        queried = tuple(item for item in missing if item not in exhausted)
        if not queried:
            break
        fetched: list[str] = []
        for item in queried:
            item_new = False
            for action in actions:
                text = (action(item) or "").strip()
                if text:
                    fetched.append(text)
                    lines = _evidence_lines(text)
                    item_new = item_new or bool(lines - held)
                    held |= lines
            if not item_new:
                exhausted.add(item)
        parts.extend(fetched)
        combined = "\n".join(parts)
        new_missing = missing_plan_items(plan, combined)
        rounds.append(RoundRecord(number=number, queried_items=queried,
                                  missing_after=new_missing))
        missing = new_missing
    if not missing:
        stop_reason = "covered"
    elif all(item in exhausted for item in missing):
        stop_reason = "no_new_evidence"
    else:
        stop_reason = "round_limit"
    combined = "\n".join(parts)
    # Whole-turn safety net: the merged evidence (first pass + every fetched
    # round) must never exceed the injection budget, even if a single action's
    # own budget was raised or bypassed. Keep the front (relevance order) and
    # record the truncation so the answer says the table was sampled.
    combined, truncated = cap_evidence_text(combined)
    if truncated:
        combined = combined.rstrip() + CONTEXT_TRUNCATED_NOTE
    return EvidenceRun(plan=plan, evidence=combined, missing=missing,
                       rounds=rounds, stop_reason=stop_reason)
