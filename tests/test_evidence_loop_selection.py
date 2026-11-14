"""S-R10 slice 1: the evidence loop selects, it does not concatenate.

Every test builds its own plan, first pass and fake actions. The fetched text
uses the excerpt marker the real actions print (``inject.chunk_marker``), so
a block's identity is its document and chunk position.
"""
from __future__ import annotations

import pytest

from app.core.rag.evidence_budget import (
    CONTEXT_TRUNCATED_NOTE,
    loop_item_char_budget,
    loop_total_char_budget,
)
from app.core.rag.evidence_plan import (
    EvidencePlan,
    is_platform_note,
    missing_plan_items,
)
from app.core.rag.evidence_retrieval import gather_evidence


def _excerpt(doc: str, idx: int, text: str) -> str:
    return f"[doc_id={doc} chunk={idx} score=0.500 class=document] {text}"


def _by_item(table: dict[str, str]):
    """An action that returns a fixed text per plan item."""
    return lambda item: table.get(item, "")


@pytest.mark.parametrize("doc,idx,body", [
    ("d1", 3, "anchor bolt grade value"),
    ("d7", 0, "membrane lap width value"),
])
def test_a_chunk_fetched_twice_is_handed_once(doc, idx, body):
    plan = EvidencePlan("p", ("alpha first", "beta second"))
    shared = _excerpt(doc, idx, f"{body} alpha first beta second")
    run = gather_evidence(
        plan, [_by_item({"alpha first": shared, "beta second": shared}),
               _by_item({"alpha first": shared})],
        initial_evidence="unrelated opening text",
    )
    assert run.evidence.count(f"[doc_id={doc} chunk={idx} ") == 1
    assert run.added_blocks == 1
    assert run.duplicate_blocks == 2


@pytest.mark.parametrize("doc,idx", [("d2", 4), ("d9", 11)])
def test_a_chunk_already_in_the_first_pass_is_not_handed_again(doc, idx):
    plan = EvidencePlan("p", ("gamma third",))
    first = _excerpt(doc, idx, "opening excerpt text")
    again = _excerpt(doc, idx, "opening excerpt text")
    other = _excerpt(doc + "x", idx + 1, "gamma third found here")
    run = gather_evidence(plan, [lambda item: again + "\n" + other],
                          initial_evidence=first)
    assert run.evidence.count(f"[doc_id={doc} chunk={idx} ") == 1
    assert f"[doc_id={doc}x chunk={idx + 1} " in run.evidence
    assert run.missing == ()


@pytest.mark.parametrize("items,item_budget,total_budget", [
    (("alpha", "beta", "gamma", "delta"), 600, 1500),
    (("one", "two", "three", "four", "five", "six"), 400, 1000),
])
def test_additions_stay_inside_the_item_and_total_budgets(
    monkeypatch, items, item_budget, total_budget,
):
    monkeypatch.setenv("AGENTIC_EVIDENCE_ITEM_CHARS", str(item_budget))
    monkeypatch.setenv("AGENTIC_EVIDENCE_LOOP_CHARS", str(total_budget))
    plan = EvidencePlan("p", tuple(f"{item} missing" for item in items))

    def flood(item):
        return "\n".join(_excerpt(f"{item}-doc", i, "filler " * 20) for i in range(30))

    run = gather_evidence(plan, [flood], initial_evidence="opening")
    assert run.added_chars <= total_budget
    assert all(spent <= item_budget for spent in run.item_chars.values())
    assert run.withheld_blocks > 0
    assert run.evidence.rstrip().endswith(CONTEXT_TRUNCATED_NOTE.strip())
    assert not run.backstop_cut


@pytest.mark.parametrize("big_item,small_items", [
    ("first heavy", ("second light", "third light")),
    ("lead item", ("tail item",)),
])
def test_every_item_with_a_hit_keeps_a_block_when_the_first_floods(
    monkeypatch, big_item, small_items,
):
    monkeypatch.setenv("AGENTIC_EVIDENCE_ITEM_CHARS", "100000")
    monkeypatch.setenv("AGENTIC_EVIDENCE_LOOP_CHARS", "3000")
    plan = EvidencePlan("p", (big_item, *small_items))
    table = {big_item: "\n".join(
        _excerpt("heavy", i, f"{big_item} " + "x" * 200) for i in range(40))}
    for n, item in enumerate(small_items):
        table[item] = _excerpt(f"light{n}", 0, f"{item} stated here")
    run = gather_evidence(plan, [_by_item(table)], initial_evidence="opening")
    for n, _item in enumerate(small_items):
        assert f"[doc_id=light{n} chunk=0 " in run.evidence
    assert run.added_chars <= 3000


@pytest.mark.parametrize("item", ["quartz lintel spacing", "zeta flange torque"])
def test_a_note_that_echoes_the_item_does_not_cover_it(item):
    plan = EvidencePlan("p", (item,))
    note = f'[bill item rows: every priced row matching "{item}" (1 rows, not sampled).]'
    assert is_platform_note(note)
    assert missing_plan_items(plan, note + "\nunrelated row 1 2 3") == (item,)
    excerpt = _excerpt("d3", 2, f"the {item} is stated")
    assert missing_plan_items(plan, excerpt) == ()


@pytest.mark.parametrize("term,longer", [("pump", "pumping"), ("bolt", "bolts")])
def test_a_term_inside_a_longer_word_does_not_cover_it(term, longer):
    plan = EvidencePlan("p", (term,))
    assert missing_plan_items(plan, f"the {longer} schedule") == (term,)
    assert missing_plan_items(plan, f"the {term}. schedule") == ()


@pytest.mark.parametrize("line,is_note", [
    ("[context_truncated: rows were sampled]", True),
    ("[doc_id=d1 chunk=0 score=0.1 class=document] text", False),
    ("[Section 3] priced row 1 2 3", False),
    ("plain retrieved text", False),
])
def test_platform_note_shape(line, is_note):
    assert is_platform_note(line) is is_note


@pytest.mark.parametrize("item", ["alpha present", "beta present"])
def test_a_run_inside_budget_carries_no_sampling_note(item):
    plan = EvidencePlan("p", (item,))
    run = gather_evidence(
        plan, [lambda _i: _excerpt("d4", 1, f"{item} value")],
        initial_evidence="opening",
    )
    assert run.withheld_blocks == 0
    assert CONTEXT_TRUNCATED_NOTE.strip() not in run.evidence
    assert run.missing == ()


@pytest.mark.parametrize("env,value", [
    ("AGENTIC_EVIDENCE_ITEM_CHARS", 1234),
    ("AGENTIC_EVIDENCE_LOOP_CHARS", 5678),
])
def test_loop_budgets_are_declared_and_configurable(monkeypatch, env, value):
    read = (loop_item_char_budget if env.endswith("ITEM_CHARS")
            else loop_total_char_budget)
    default = read()
    assert default > 0
    monkeypatch.setenv(env, str(value))
    assert read() == value
    monkeypatch.setenv(env, "0")
    assert read() == default


@pytest.mark.parametrize("doc,idx,scores", [
    ("d5", 2, ("0.910", "0.120")),
    ("d8", 6, ("0.333", "0.777")),
])
def test_the_same_chunk_printed_by_two_legs_with_different_scores_is_one_block(
    doc, idx, scores,
):
    plan = EvidencePlan("p", ("omega item",))
    body = "omega item stated"
    a = f"[doc_id={doc} chunk={idx} score={scores[0]} class=document] {body}"
    b = f"[doc_id={doc} chunk={idx} score={scores[1]} class=document] {body}"
    run = gather_evidence(plan, [lambda _i: a, lambda _i: b],
                          initial_evidence="opening")
    assert run.evidence.count(f"[doc_id={doc} chunk={idx} ") == 1
    assert run.added_blocks == 1
    assert run.duplicate_blocks == 1


@pytest.mark.parametrize("item", ["sigma item", "tau item"])
def test_an_actions_own_truncation_note_is_stated_once_not_handed(item):
    plan = EvidencePlan("p", (item,))
    fetched = _excerpt("d6", 0, f"{item} stated") + CONTEXT_TRUNCATED_NOTE
    run = gather_evidence(plan, [lambda _i: fetched], initial_evidence="opening")
    assert run.added_blocks == 1
    assert run.withheld_blocks == 1
    assert run.evidence.count(CONTEXT_TRUNCATED_NOTE.strip()) == 1
    assert run.evidence.rstrip().endswith(CONTEXT_TRUNCATED_NOTE.strip())
