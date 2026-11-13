"""Route seam for agentic retrieval of project figures (item 2).

Composes the #163 pure modules — evidence plan, the three retrieval actions
(semantic / lexical / structural), the coverage check and the bounded re-query
loop — into the one mechanism the runtime route calls when a project-figure or
published-standard question's first-pass retrieval leaves plan items
uncovered:

    plan -> retrieval actions -> coverage check -> bounded re-query (max 3)

The primitives (semantic / lexical / structural fetchers) and the plan call
are injected, so the loop is unit-testable with fakes and the route supplies
the real ``retrieve_with_filter``, ``store.bm25_search``, the WBS
``XLSXDocument.sheets`` and its LLM caller. The loop never runs unless the
first pass is uncovered, and it never runs round 4 (the spec's hard bound).

Nothing here names a project, an item or a column.
"""
from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.core.rag.evidence_actions import (
    bill_rows_action,
    lexical_action,
    semantic_action,
    structural_action,
)
from app.core.rag.evidence_plan import (
    ROUND_LIMIT,
    EvidencePlan,
    missing_plan_items,
)
from app.core.rag.evidence_retrieval import (
    EvidenceRun,
    RetrievalAction,
    gather_evidence,
)
from app.core.rag.plan_generation import parse_generated_plan

logger = logging.getLogger(__name__)

#: The plan call takes a question and returns the ``_call_llm``-style response
#: dict the pure parser already accepts.
PlanCaller = Callable[[str], Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class EvidenceLoopOutcome:
    """The loop's result for the route: merged evidence, gaps and round log.

    ``plan_items`` is the parsed plan (one phrase per item, in order); it is
    carried so a dataset logger can record what the agent decided to fetch.
    """

    evidence: str
    missing: tuple[str, ...]
    rounds: int
    plan_items: tuple[str, ...] = ()


def build_actions(
    *,
    retrieve_fn: Callable[..., tuple],
    bm25_fn: Callable[..., Sequence],
    project_id: str,
    sheets: Mapping[str, Sequence[Sequence[str]]] | None = None,
    headers: Mapping[str, Sequence[str]] | None = None,
    typed_bills: Callable[[str], Iterable[str]] | None = None,
    bill_chunks: Callable[[str, list[str]], Sequence] | None = None,
    question: str = "",
) -> list[RetrievalAction]:
    """The three retrieval actions over the real primitives.

    ``retrieve_fn`` matches ``retrieve_with_filter(query, project_id, k=...)``
    and returns ``(chunks, noise_filtered)``. ``bm25_fn`` matches
    ``store.bm25_search(project_id, query, k=...)``. ``sheets``/``headers``
    are the WBS ``XLSXDocument.sheets`` shape; a structural action is only
    built when sheets are supplied. The bill-rows action scopes the same
    lexical primitive to documents whose chunks print priced rows, so a long
    PDF bill's rows are not crowded out of the global ranking;
    typed_bills(project_id) (optional) names the documents the ingest
    typed as a bill, which that action ranks the item inside and prefers.
    ``bill_chunks(project_id, doc_ids)`` (optional) returns every chunk of
    those bills, from which the action reads each bill's sections; the
    ``question`` is where an ask names one.
    """
    from app.core.rag.retriever import priced_row_count

    actions: list[RetrievalAction] = [
        semantic_action(retrieve_fn, project_id),
        lexical_action(bm25_fn, project_id),
        bill_rows_action(
            bm25_fn, project_id, row_count=priced_row_count, typed_bills=typed_bills,
            bill_chunks=bill_chunks, ask=question,
        ),
    ]
    if sheets is not None:
        actions.append(structural_action(sheets, headers))
    return actions


def first_pass_uncovered(plan: EvidencePlan, evidence_text: str) -> tuple[str, ...]:
    """Plan items the first-pass evidence does not cover."""
    return missing_plan_items(plan, evidence_text)


def run_loop(
    plan: EvidencePlan,
    actions: Sequence[RetrievalAction],
    initial_evidence: str = "",
    max_rounds: int = ROUND_LIMIT,
) -> EvidenceRun:
    """Run the bounded fetch loop over ``plan``; round ``max_rounds + 1`` never."""
    return gather_evidence(
        plan, actions, initial_evidence=initial_evidence, max_rounds=max_rounds,
    )


async def run_project_figure_retrieval(
    *,
    question: str,
    plan_caller: PlanCaller,
    retrieve_fn: Callable[..., tuple],
    bm25_fn: Callable[..., Sequence],
    project_id: str,
    first_pass_evidence: str,
    sheets: Mapping[str, Sequence[Sequence[str]]] | None = None,
    headers: Mapping[str, Sequence[str]] | None = None,
    run_sync: Callable[..., Awaitable[Any]] | None = None,
    typed_bills: Callable[[str], Iterable[str]] | None = None,
    bill_chunks: Callable[[str, list[str]], Sequence] | None = None,
) -> EvidenceLoopOutcome:
    """Plan, then fetch evidence over the plan with the bounded loop.

    ``first_pass_evidence`` is the retrieval text already in hand (round 1);
    the loop only re-queries the plan items it does not cover. An empty or
    already-covered plan skips the loop entirely and returns the first pass.

    ``run_sync`` is an async executor for the synchronous fetch loop (the
    runtime passes ``off_loop`` so embedding/SQL never blocks the event loop);
    ``None`` runs it inline for unit tests.
    """
    response = await plan_caller(question)
    plan = parse_generated_plan(response)
    if not plan.items or not first_pass_uncovered(plan, first_pass_evidence):
        logger.info(
            "rag_evidence_loop_trace project=%s plan_items=%d rounds=0 "
            "reason=%s", project_id, len(plan.items),
            "empty_plan" if not plan.items else "first_pass_covered",
        )
        return EvidenceLoopOutcome(
            evidence=first_pass_evidence,
            missing=first_pass_uncovered(plan, first_pass_evidence),
            rounds=0,
            plan_items=plan.items,
        )
    actions = build_actions(
        retrieve_fn=retrieve_fn,
        bm25_fn=bm25_fn,
        project_id=project_id,
        sheets=sheets,
        headers=headers,
        typed_bills=typed_bills,
        bill_chunks=bill_chunks,
        question=question,
    )
    if run_sync is None:
        run = run_loop(plan, actions, initial_evidence=first_pass_evidence)
    else:
        run = await run_sync(run_loop, plan, actions, initial_evidence=first_pass_evidence)
    logger.info(
        "rag_evidence_loop_trace project=%s plan_items=%d rounds=%d "
        "actions=%d missing=%d reason=%s", project_id, len(plan.items),
        len(run.rounds), len(actions), len(run.missing), run.stop_reason,
    )
    return EvidenceLoopOutcome(
        evidence=run.evidence,
        missing=run.missing,
        rounds=len(run.rounds),
        plan_items=plan.items,
    )
