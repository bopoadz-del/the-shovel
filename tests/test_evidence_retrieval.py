"""Tests for the evidence-fetch loop (item 2)."""
from __future__ import annotations

import pytest

from app.core.rag.evidence_budget import CONTEXT_TRUNCATED_NOTE
from app.core.rag.evidence_plan import parse_evidence_plan
from app.core.rag.evidence_retrieval import gather_evidence


def _action(result):
    return lambda item: result


def _fresh_action(prefix):
    """An action whose every call returns a line never returned before."""
    counter = iter(range(10_000))
    return lambda item: f"{prefix} {next(counter)}"


@pytest.mark.parametrize("plan_text,evidence", [
    ("bulk excavation, blinding", "bulk excavation 75 mm blinding to slab"),
    ("formwork, painting", "formwork and painting quantities"),
])
def test_a_plan_covered_on_first_pass_needs_no_requery(plan_text, evidence):
    plan = parse_evidence_plan(plan_text)
    run = gather_evidence(plan, [_action(evidence)], initial_evidence=evidence)
    assert run.missing == ()
    assert run.rounds == []
    assert run.stop_reason == "covered"


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
    assert run.stop_reason == "covered"


@pytest.mark.parametrize("plan_text,missing", [
    ("bulk excavation, lift walls", ("bulk excavation", "lift walls")),
    ("formwork, staircases", ("formwork", "staircases")),
])
def test_round_four_never_runs(plan_text, missing):
    plan = parse_evidence_plan(plan_text)
    # Each call fetches a line not held before, but never the planned items.
    run = gather_evidence(plan, [_fresh_action("unrelated sheet row")],
                          initial_evidence="raft concrete")
    # max_rounds defaults to ROUND_LIMIT = 3; the loop never reaches round 4.
    assert [r.number for r in run.rounds] == [1, 2, 3]
    assert run.missing == missing
    assert run.stop_reason == "round_limit"


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
def test_a_round_fetching_nothing_ends_the_loop(plan_text, missing):
    plan = parse_evidence_plan(plan_text)
    run = gather_evidence(plan, [_action("")], initial_evidence="raft concrete")
    # Round 1 returned nothing, so re-running the same queries is not tried.
    assert [r.number for r in run.rounds] == [1]
    assert run.missing == missing
    assert run.stop_reason == "no_new_evidence"


@pytest.mark.parametrize("plan_text,repeat,missing", [
    ("bulk excavation, lift walls", "unrelated sheet rows",
     ("bulk excavation", "lift walls")),
    ("formwork, staircases", "raft concrete\nlevel 2 slab",
     ("formwork", "staircases")),
])
def test_a_round_that_repeats_held_evidence_ends_the_loop(plan_text, repeat, missing):
    plan = parse_evidence_plan(plan_text)
    calls = []

    def _same(item):
        calls.append(item)
        return repeat

    run = gather_evidence(plan, [_same], initial_evidence="raft concrete")
    # Round 1 adds the text once (only the first item counts it as new);
    # round 2 re-queries that item, gets nothing new, and round 3 never runs.
    assert [r.number for r in run.rounds] == [1, 2]
    assert run.rounds[1].queried_items == (plan.items[0],)
    assert len(calls) == len(plan.items) + 1
    assert run.missing == missing
    assert run.stop_reason == "no_new_evidence"


@pytest.mark.parametrize("plan_text,missing", [
    ("bulk excavation, lift walls", ("bulk excavation", "lift walls")),
    ("formwork, staircases", ("formwork", "staircases")),
])
def test_a_round_that_adds_a_new_line_continues(plan_text, missing):
    plan = parse_evidence_plan(plan_text)
    run = gather_evidence(plan, [_fresh_action("schedule row")],
                          initial_evidence="raft concrete")
    assert len(run.rounds) > 1
    assert run.missing == missing


@pytest.mark.parametrize("plan_text,dry,live", [
    ("bulk excavation, lift walls", "bulk excavation", "lift walls"),
    ("formwork, staircases", "staircases", "formwork"),
])
def test_an_item_whose_actions_all_return_nothing_is_not_requeried(plan_text, dry, live):
    plan = parse_evidence_plan(plan_text)
    seen = []
    fresh = _fresh_action("unrelated row")

    def _semantic(item):
        seen.append(item)
        return ""

    def _lexical(item):
        return "" if item == dry else fresh(item)

    run = gather_evidence(plan, [_semantic, _lexical],
                          initial_evidence="raft concrete")
    assert [r.number for r in run.rounds] == [1, 2, 3]
    assert seen.count(dry) == 1
    assert seen.count(live) == 3
    assert run.rounds[1].queried_items == (live,)


@pytest.mark.parametrize("plan_text,fill", [
    ("bulk excavation", "A" * 4000),
    ("formwork", "B" * 4000),
])
def test_truncation_note_survives_the_whole_turn_merge(monkeypatch, plan_text, fill):
    monkeypatch.setenv("AGENTIC_EVIDENCE_MAX_TOKENS", "2")
    plan = parse_evidence_plan(plan_text)
    run = gather_evidence(plan, [_action(fill)], initial_evidence="first pass " * 50)
    assert CONTEXT_TRUNCATED_NOTE in run.evidence
