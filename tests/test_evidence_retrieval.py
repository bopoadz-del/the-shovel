"""Tests for the evidence-fetch loop (item 2)."""
from __future__ import annotations

from app.core.rag.evidence_plan import parse_evidence_plan
from app.core.rag.evidence_retrieval import gather_evidence


def _action(result):
    return lambda item: result


def test_a_plan_covered_on_first_pass_needs_no_requery():
    plan = parse_evidence_plan("bulk excavation, blinding")
    run = gather_evidence(plan, [_action("bulk excavation 75 mm blinding")],
                          initial_evidence="bulk excavation 75 mm blinding to slab")
    assert run.missing == ()
    assert run.rounds == []


def test_a_missing_item_triggers_a_second_round_that_covers_it():
    plan = parse_evidence_plan("bulk excavation, blinding")
    # First pass has excavation only; the action answers blinding.
    run = gather_evidence(
        plan,
        [_action("75 mm blinding to concrete")],
        initial_evidence="bulk excavation in all classes",
    )
    assert run.missing == ()
    assert [r.number for r in run.rounds] == [1]
    assert run.rounds[0].queried_items == ("blinding",)
    assert run.rounds[0].missing_after == ()


def test_round_four_never_runs():
    plan = parse_evidence_plan("bulk excavation, lift walls")
    # Each round fetches something new but never the planned items.
    run = gather_evidence(plan, [_action("unrelated sheet rows")], initial_evidence="raft concrete")
    # max_rounds defaults to ROUND_LIMIT = 3; the loop never reaches round 4.
    assert [r.number for r in run.rounds] == [1, 2, 3]
    assert run.missing == ("bulk excavation", "lift walls")


def test_all_missing_items_are_queried_by_every_action():
    plan = parse_evidence_plan("bulk excavation, lift walls, staircases")
    seen = []
    run = gather_evidence(
        plan,
        [lambda item: (seen.append(("semantic", item)), "")[1],
         lambda item: (seen.append(("structural", item)), "lift walls")[1]],
        initial_evidence="raft concrete",
    )
    # Each missing item went through each action, in order.
    assert ("semantic", "bulk excavation") in seen
    assert ("structural", "lift walls") in seen
    # "lift walls" is covered by the structural action's constant answer.
    assert run.missing == ("bulk excavation", "staircases")


def test_the_loop_is_bounded_even_when_fetching_nothing():
    plan = parse_evidence_plan("bulk excavation")
    run = gather_evidence(plan, [_action("")], initial_evidence="raft concrete")
    # It still runs at most ROUND_LIMIT rounds, never more.
    assert [r.number for r in run.rounds] == [1, 2, 3]
    assert run.missing == ("bulk excavation",)
