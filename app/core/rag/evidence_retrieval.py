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
    """
    parts = [initial_evidence] if (initial_evidence or "").strip() else []
    missing = missing_plan_items(plan, initial_evidence or "")
    rounds: list[RoundRecord] = []
    for number in range(1, max_rounds + 1):
        if not missing:
            break
        fetched: list[str] = []
        for item in missing:
            for action in actions:
                text = (action(item) or "").strip()
                if text:
                    fetched.append(text)
        parts.extend(fetched)
        combined = "\n".join(parts)
        new_missing = missing_plan_items(plan, combined)
        rounds.append(RoundRecord(number=number, queried_items=missing,
                                  missing_after=new_missing))
        missing = new_missing
    return EvidenceRun(plan=plan, evidence="\n".join(parts), missing=missing,
                       rounds=rounds)
