"""Exam C1 is itself tested (SPEC_EXAM_PLAN rule 3): every case has a
known-good output that must pass and known-bad outputs that must fail, and
the fixture's expectations are grounded in its own invented passages.

These tests do not assert the product's score; ``python -m
tests.conflict_exam.checker`` measures that."""

from __future__ import annotations

import copy

import pytest

from tests.conflict_exam.checker import (
    check,
    good_output,
    load_cases,
    product_output,
    score,
)

CASES = load_cases()
DIFFER = [c for c in CASES if c["expected"]["conflict"]]
NO_CONFLICT = [c for c in CASES if not c["expected"]["conflict"]]


def _ids(cases):
    return [c["id"] for c in cases]


# ── the fixture ──────────────────────────────────────────────────────────

def test_ids_unique_and_every_kind_has_two_cases():
    assert len({c["id"] for c in CASES}) == len(CASES)
    kinds = {}
    for c in CASES:
        kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
    assert set(kinds) == {"differ", "agree", "looks_different", "other_thing"}
    assert all(n >= 2 for n in kinds.values()), kinds
    with_winner = [c for c in DIFFER if c["expected"]["governs"]]
    no_winner_but_statement = [
        c for c in DIFFER if not c["expected"]["governs"] and len(c["rows"]) > len(c["expected"]["values"])]
    assert len(with_winner) >= 2 and len(no_winner_but_statement) >= 1


@pytest.mark.parametrize("case", DIFFER, ids=_ids(DIFFER))
def test_expected_values_are_written_in_their_documents(case):
    snippets = {r["doc_name"]: r["snippet"] for r in case["rows"]}
    for v in case["expected"]["values"]:
        assert v["document"] in snippets
        assert v["value"] in snippets[v["document"]], (v, snippets[v["document"]])
    assert len({v["value"] for v in case["expected"]["values"]}) >= 2
    exp = case["expected"]
    if exp["governs"]:
        value_docs = {v["document"] for v in exp["values"]}
        assert exp["governs"] in value_docs
        assert exp["basis_document"] in snippets and exp["basis_document"] not in value_docs


@pytest.mark.parametrize("case", CASES, ids=_ids(CASES))
def test_model_answer_states_one_value_only(case):
    """The input answer is the failure the exam targets: a single pick."""
    if case["expected"]["conflict"]:
        said = {v["value"] for v in case["expected"]["values"] if v["value"] in case["model_answer"]}
        assert len(said) == 1


# ── the checker, must pass ───────────────────────────────────────────────

@pytest.mark.parametrize("case", CASES, ids=_ids(CASES))
def test_checker_passes_the_correct_output(case):
    ok, why = check(case, good_output(case))
    assert ok, why


def test_checker_ignores_case_and_spacing_only():
    case = DIFFER[0]
    out = good_output(case)
    out["text"] = out["text"].upper().replace(" ", "  ")
    for v in out["conflicts"][0]["values"]:
        v["value"] = v["value"].lower()
    assert check(case, out)[0]


# ── the checker, must fail ───────────────────────────────────────────────

@pytest.mark.parametrize("case", DIFFER, ids=_ids(DIFFER))
def test_checker_fails_a_silent_pick(case):
    assert not check(case, {"text": case["model_answer"], "conflicts": []})[0]
    assert not check(case, {"text": case["model_answer"]})[0]


@pytest.mark.parametrize("case", DIFFER, ids=_ids(DIFFER))
def test_checker_fails_a_dropped_value(case):
    out = good_output(case)
    out["conflicts"][0]["values"].pop()
    assert not check(case, out)[0]


@pytest.mark.parametrize("case", DIFFER, ids=_ids(DIFFER))
def test_checker_fails_a_value_credited_to_the_wrong_document(case):
    out = good_output(case)
    vals = out["conflicts"][0]["values"]
    i = next(i for i in range(1, len(vals)) if vals[i]["value"] != vals[0]["value"])
    vals[0]["document"], vals[i]["document"] = vals[i]["document"], vals[0]["document"]
    assert not check(case, out)[0]


@pytest.mark.parametrize("case", DIFFER, ids=_ids(DIFFER))
def test_checker_fails_when_the_text_hides_a_value(case):
    out = good_output(case)
    hidden = case["expected"]["values"][-1]["value"]
    out["text"] = out["text"].replace(hidden, "")
    assert not check(case, out)[0]


@pytest.mark.parametrize("case", DIFFER, ids=_ids(DIFFER))
def test_checker_fails_a_wrong_winner(case):
    out = good_output(case)
    c = out["conflicts"][0]
    exp = case["expected"]
    if exp["governs"]:
        c["governs"] = None
        assert not check(case, out)[0]
        out = good_output(case)
        other = next(v["document"] for v in exp["values"] if v["document"] != exp["governs"])
        out["conflicts"][0]["governs"] = other
        assert not check(case, out)[0]
        out = good_output(case)
        out["conflicts"][0]["basis_document"] = None
        assert not check(case, out)[0]
    else:
        c["governs"] = exp["values"][0]["document"]
        assert not check(case, out)[0]


@pytest.mark.parametrize("case", DIFFER, ids=_ids(DIFFER))
def test_checker_fails_a_duplicated_conflict(case):
    out = good_output(case)
    out["conflicts"].append(copy.deepcopy(out["conflicts"][0]))
    assert not check(case, out)[0]


@pytest.mark.parametrize("case", NO_CONFLICT, ids=_ids(NO_CONFLICT))
def test_checker_fails_a_false_conflict(case):
    rows = case["rows"]
    fake = {"values": [{"value": "x", "document": r["doc_name"]} for r in rows],
            "governs": None, "basis_document": None}
    assert not check(case, {"text": case["model_answer"], "conflicts": [fake]})[0]


# ── the exam as a whole ──────────────────────────────────────────────────

def test_exam_neither_passes_nor_fails_everything():
    perfect = score(CASES, good_output)
    assert perfect["passed"] == perfect["total"] == len(CASES)
    silent = score(CASES, lambda c: {"text": c["model_answer"], "conflicts": []})
    assert silent["passed"] == len(NO_CONFLICT)
    flag_all = score(CASES, lambda c: {
        "text": c["model_answer"],
        "conflicts": [{"values": [{"value": "x", "document": r["doc_name"]} for r in c["rows"]]}]})
    assert flag_all["passed"] == 0
    crash = score(CASES[:1], lambda c: 1 / 0)
    assert crash["passed"] == 0 and "crashed" in crash["results"][0]["reasons"][0]


def test_runner_reads_the_product_contract():
    """The product adapter runs on every case and yields the exam's shape;
    the score itself is measured, not asserted, here."""
    report = score(CASES, product_output)
    assert report["total"] == len(CASES)
    assert not any("crashed" in "".join(r["reasons"]) for r in report["results"])
