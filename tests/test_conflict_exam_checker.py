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

@pytest.mark.parametrize("kind", ["differ", "agree", "looks_different", "other_thing"])
def test_every_kind_has_two_cases_with_unique_ids(kind):
    assert len({c["id"] for c in CASES}) == len(CASES)
    assert {c["kind"] for c in CASES} == {"differ", "agree", "looks_different", "other_thing"}
    assert sum(c["kind"] == kind for c in CASES) >= 2


@pytest.mark.parametrize("decides, minimum", [(True, 2), (False, 1)])
def test_precedence_statements_cover_both_decisions(decides, minimum):
    """A precedence statement in the evidence that names a winner, and one
    that does not cover the documents in conflict (no winner)."""
    with_statement = [c for c in DIFFER if len(c["rows"]) > len({v["document"] for v in c["expected"]["values"]})]
    assert sum(bool(c["expected"]["governs"]) == decides for c in with_statement) >= minimum


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

def _silent(c):
    return {"text": c["model_answer"], "conflicts": []}


def _flag_all(c):
    return {"text": c["model_answer"],
            "conflicts": [{"values": [{"value": "x", "document": r["doc_name"]} for r in c["rows"]]}]}


def _crash(c):
    raise RuntimeError("producer failed")


@pytest.mark.parametrize("name, produce, expected_passed", [
    ("perfect", good_output, len(CASES)),
    ("silent pick", _silent, len(NO_CONFLICT)),
    ("flag everything", _flag_all, 0),
    ("crash", _crash, 0),
])
def test_exam_neither_passes_nor_fails_everything(name, produce, expected_passed):
    report = score(CASES, produce)
    assert report["total"] == len(CASES)
    assert report["passed"] == expected_passed, name
    if name == "crash":
        assert all("crashed" in r["reasons"][0] for r in report["results"])


@pytest.mark.parametrize("subset", [DIFFER, NO_CONFLICT], ids=["differ", "no_conflict"])
def test_runner_reads_the_product_contract(subset):
    """The product adapter runs on every case and yields the exam's shape;
    the score itself is measured, not asserted, here."""
    report = score(subset, product_output)
    assert report["total"] == len(subset)
    assert not any("crashed" in "".join(r["reasons"]) for r in report["results"])
