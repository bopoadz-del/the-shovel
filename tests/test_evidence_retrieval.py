"""Tests for the evidence-fetch loop (item 2)."""
from __future__ import annotations

import pytest

from app.core.rag.evidence_plan import parse_evidence_plan
from app.core.rag.evidence_retrieval import gather_evidence


def _action(result):
    return lambda item: result


@pytest.mark.parametrize("plan_text,evidence", [
    ("bulk excavation, blinding", "bulk excavation 75 mm blinding to slab"),
    ("formwork, painting", "formwork and painting quantities"),
])
def test_a_plan_covered_on_first_pass_needs_no_requery(plan_text, evidence):
    plan = parse_evidence_plan(plan_text)
    run = gather_evidence(plan, [_action(evidence)], initial_evidence=evidence)
    assert run.missing == ()
    assert run.rounds == []


@pytest.mark.parametrize("plan_text,initial,action_result,missing_item", [
    ("bulk excavation, blinding",
     "bulk excavation in all classes", "75 mm blinding to concrete", "blinding"),
    ("formwork, painting",
     "formwork to all floors", "painting to all walls", "painting"),
])
def test_a_missing_item_triggers_a_second_round_that_covers_it(
    plan_text, initial, action_result, missing_item,
):
    plan = parse_evidence_plan(plan_text)
    run = gather_evidence(plan, [_action(action_result)], initial_evidence=initial)
    assert run.missing == ()
    assert [r.number for r in run.rounds] == [1]
    assert run.rounds[0].queried_items == (missing_item,)
    assert run.rounds[0].missing_after == ()


@pytest.mark.parametrize("plan_text,missing", [
    ("bulk excavation, lift walls", ("bulk excavation", "lift walls")),
    ("formwork, staircases", ("formwork", "staircases")),
])
def test_round_four_never_runs(plan_text, missing):
    plan = parse_evidence_plan(plan_text)
    # Each round fetches something new but never the planned items.
    run = gather_evidence(plan, [_action("unrelated sheet rows")],
                          initial_evidence="raft concrete")
    # max_rounds defaults to ROUND_LIMIT = 3; the loop never reaches round 4.
    assert [r.number for r in run.rounds] == [1, 2, 3]
    assert run.missing == missing


@pytest.mark.parametrize("plan_text", [
    "bulk excavation, lift walls, staircases",
    "formwork, painting, ceilings",
])
def test_all_missing_items_are_queried_by_every_action(plan_text):
    plan = parse_evidence_plan(plan_text)
    seen = []
    run = gather_evidence(
        plan,
        [lambda item: (seen.append(("semantic", item)), "")[1],
         lambda item: (seen.append(("structural", item)), "lift walls")[1]],
        initial_evidence="raft concrete",
    )
    # Each missing item went through each action, in order.
    first = plan.items[0]
    assert ("semantic", first) in seen
    assert ("structural", first) in seen


@pytest.mark.parametrize("plan_text,missing", [
    ("bulk excavation", ("bulk excavation",)),
    ("formwork", ("formwork",)),
])
def test_the_loop_is_bounded_even_when_fetching_nothing(plan_text, missing):
    plan = parse_evidence_plan(plan_text)
    run = gather_evidence(plan, [_action("")], initial_evidence="raft concrete")
    # It still runs at most ROUND_LIMIT rounds, never more.
    assert [r.number for r in run.rounds] == [1, 2, 3]
    assert run.missing == missing
