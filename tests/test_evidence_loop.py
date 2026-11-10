"""Tests for the agentic-retrieval route seam (item 2)."""
from __future__ import annotations

import pytest
from types import SimpleNamespace

from app.core.rag.evidence_loop import (
    build_actions,
    first_pass_uncovered,
    run_loop,
    run_project_figure_retrieval,
)
from app.core.rag.evidence_plan import parse_evidence_plan


def _chunk(text):
    return SimpleNamespace(text=text)


def _fake_retrieve(chunks_by_item):
    def _retrieve(query, project_id, k=5):
        return [_chunk(chunks_by_item.get(query, ""))], 0
    return _retrieve


def _fake_bm25(chunks_by_item):
    def _bm25(project_id, query, k=50):
        return [_chunk(chunks_by_item.get(query, ""))]
    return _bm25


async def _plan_caller(plan_text):
    return {"choice": {"message": {"content": plan_text}}}


@pytest.mark.parametrize("plan_text,evidence", [
    ("bulk excavation, blinding", "bulk excavation 75 mm blinding to slab"),
    ("formwork, painting", "formwork and painting quantities"),
])
def test_a_covered_first_pass_skips_the_loop(plan_text, evidence):
    plan = parse_evidence_plan(plan_text)
    assert first_pass_uncovered(plan, evidence) == ()


@pytest.mark.parametrize("plan_text,evidence,missing", [
    ("bulk excavation, blinding", "bulk excavation in all classes", ("blinding",)),
    ("formwork, painting", "formwork to all floors", ("painting",)),
])
def test_an_uncovered_item_is_named(plan_text, evidence, missing):
    plan = parse_evidence_plan(plan_text)
    assert first_pass_uncovered(plan, evidence) == missing


@pytest.mark.parametrize("plan_text,initial,rounds_want", [
    ("bulk excavation, blinding", "bulk excavation only", 1),
    ("formwork, painting", "formwork only", 1),
])
def test_an_uncovered_plan_runs_the_loop_and_recovers(plan_text, initial, rounds_want):
    plan = parse_evidence_plan(plan_text)
    actions = build_actions(
        retrieve_fn=_fake_retrieve({}),
        bm25_fn=_fake_bm25({
            "blinding": "75 mm blinding to concrete",
            "painting": "painting to all walls",
        }),
        project_id="p",
    )
    run = run_loop(plan, actions, initial_evidence=initial)
    assert run.missing == ()
    assert len(run.rounds) == rounds_want


@pytest.mark.parametrize("plan_text,missing", [
    ("bulk excavation, lift walls", ("bulk excavation", "lift walls")),
    ("formwork, staircases", ("formwork", "staircases")),
])
def test_round_four_never_runs(plan_text, missing):
    plan = parse_evidence_plan(plan_text)
    actions = build_actions(
        retrieve_fn=_fake_retrieve({}),
        bm25_fn=_fake_bm25({"anything": "unrelated sheet rows"}),
        project_id="p",
    )
    run = run_loop(plan, actions, initial_evidence="raft concrete")
    assert [r.number for r in run.rounds] == [1, 2, 3]
    assert run.missing == missing


@pytest.mark.parametrize("plan_text,initial", [
    ("bulk excavation, blinding", "bulk excavation 75 mm blinding to slab"),
    ("formwork, painting", "formwork and painting quantities"),
])
async def test_route_seam_returns_zero_rounds_when_covered(plan_text, initial):
    outcome = await run_project_figure_retrieval(
        question="what are the quantities?",
        plan_caller=lambda q: _plan_caller(plan_text),
        retrieve_fn=_fake_retrieve({}),
        bm25_fn=_fake_bm25({}),
        project_id="p",
        first_pass_evidence=initial,
    )
    assert outcome.rounds == 0
    assert outcome.missing == ()


@pytest.mark.parametrize("plan_text,initial,item_fill", [
    ("bulk excavation, blinding", "bulk excavation only",
     {"blinding": "75 mm blinding to concrete"}),
    ("formwork, painting", "formwork only",
     {"painting": "painting to all walls"}),
])
async def test_route_seam_loops_to_cover_a_gap(plan_text, initial, item_fill):
    outcome = await run_project_figure_retrieval(
        question="what are the quantities?",
        plan_caller=lambda q: _plan_caller(plan_text),
        retrieve_fn=_fake_retrieve({}),
        bm25_fn=_fake_bm25(item_fill),
        project_id="p",
        first_pass_evidence=initial,
    )
    assert outcome.rounds == 1
    assert outcome.missing == ()
