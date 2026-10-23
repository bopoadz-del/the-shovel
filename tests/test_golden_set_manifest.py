"""Contract tests for tests/golden_set.yaml and its gate runner.

The golden set is the TASK 4e pilot gate's test plan; these tests defend the
invariants the gate relies on so a hand-edit cannot silently weaken it:
parseable yaml, 25-30 queries, every entry with a prompt and answer_expect,
the mandatory EOT project-vs-GK collision case with both expect and reject,
compilable regexes, and full coverage of the 12 pilot-critical features.
load_golden_set() itself raises on the structural defects (missing prompt,
bad regex, duplicate id), so a successful parse here already proves those.
"""
from __future__ import annotations

import re

import pytest

from scripts.golden_set_gate import (
    DEFAULT_GOLDEN_SET,
    REGEX_FLAGS,
    GoldenSetError,
    load_golden_set,
    pass_bar,
)

# The 12 pilot-critical capabilities the golden set must exercise (mirrors
# the must-cover block of tests/feature_matrix_manifest.yaml; feature values
# in golden_set.yaml use the same action names for traceability).
PILOT_CRITICAL = {
    "process_document",            # document Q&A (the client project PEP)
    "boq_process",                 # BOQ total
    "spec_analyze",                # specs extraction
    "document_metadata",           # metadata listing
    "generate_wbs",                # WBS / schedule generation
    "resource_histogram",          # manpower histogram
    "parse_primavera_schedule",    # milestones
    "cash_flow_forecast",          # S-curve
    "procurement_list_generator",  # procurement list
    "rfp_management",              # RFP production
    "construction_advisor",        # KB calculation (mass concrete)
    "drawing_qto",                 # QTO from drawings
}

FRESH_UPLOAD_PROJECT = "FIXTURE — Fresh Upload Eval"  # resolved BY NAME to the seeded fixture project


@pytest.fixture(scope="module")
def golden() -> dict:
    return load_golden_set(DEFAULT_GOLDEN_SET)


def test_golden_set_parses(golden):
    assert golden["queries"], "golden set has no queries"


def test_query_count_between_25_and_30(golden):
    n = len(golden["queries"])
    assert 25 <= n <= 30, f"golden set must hold 25-30 queries, has {n}"


def test_composition(golden):
    fresh = [q for q in golden["queries"]
             if q["project"] == FRESH_UPLOAD_PROJECT]
    demo = [q for q in golden["queries"] if q["feature"] == "demo_flow"]
    assert len(fresh) == 12, \
        f"expected 12 fresh-upload queries vs {FRESH_UPLOAD_PROJECT}, " \
        f"got {len(fresh)}"
    assert len(demo) == 4, f"expected 4 demo-flow queries, got {len(demo)}"


def test_ids_unique(golden):
    ids = [q["id"] for q in golden["queries"]]
    assert len(ids) == len(set(ids))


def test_every_entry_has_prompt_and_answer_expect(golden):
    for q in golden["queries"]:
        assert q["prompt"].strip(), f"{q['id']}: empty prompt"
        assert q["answer_expect"], f"{q['id']}: no answer_expect patterns"
        assert q["project"], f"{q['id']}: no project"


def test_all_regexes_compile(golden):
    # load_golden_set already rejects bad regexes; assert directly anyway so
    # the invariant survives a refactor of the loader's validation.
    for q in golden["queries"]:
        for p in q["answer_expect"] + q["answer_reject"]:
            re.compile(p, REGEX_FLAGS)


def test_eot_case_present_with_expect_and_reject(golden):
    """The mandatory project-vs-GK collision case (RAG_AUDIT_V2 section 4):
    the project's PC-20.2 says 21 days, GK's FIDIC notes say 28 days."""
    by_id = {q["id"]: q for q in golden["queries"]}
    assert "fresh_eot_notice_period" in by_id, "EOT golden case missing"
    eot = by_id["fresh_eot_notice_period"]
    assert eot["project"] == FRESH_UPLOAD_PROJECT
    assert re.search(r"\b(eot|extension of time)\b", eot["prompt"],
                     re.IGNORECASE) and "notice" in eot["prompt"].lower()
    # The expect patterns must accept the project truth (21 days) and the
    # reject patterns must catch the colliding GK value (28 days).
    assert any(re.search(p, "notice within 21 days", REGEX_FLAGS)
               for p in eot["answer_expect"]), \
        "EOT answer_expect does not match '21 days'"
    assert eot["answer_reject"], "EOT case has no answer_reject"
    assert any(re.search(p, "notice within 28 days", REGEX_FLAGS)
               for p in eot["answer_reject"]), \
        "EOT answer_reject does not catch '28 days'"
    # And they must not cross-fire.
    assert not any(re.search(p, "notice within 28 days", REGEX_FLAGS)
                   for p in eot["answer_expect"])
    assert not any(re.search(p, "notice within 21 days", REGEX_FLAGS)
                   for p in eot["answer_reject"])
    # A CORRECT grounded answer may quote the source note's own context
    # ("reduced from the standard 28 days") — the reject targets 28 as the
    # operative deadline, not any mention of 28.
    correct_with_context = ("Notice must be given within 21 days, reduced "
                            "from the standard 28 days by PC-20.2.")
    assert not any(re.search(p, correct_with_context, REGEX_FLAGS)
                   for p in eot["answer_reject"]), \
        "EOT reject fires on a correct answer that cites 28 as context"


def test_feature_coverage_includes_all_pilot_critical(golden):
    features = {q["feature"] for q in golden["queries"]}
    missing = PILOT_CRITICAL - features
    assert not missing, f"pilot-critical features not covered: {missing}"


def test_kb_calculation_pins_verified_thermal_limits(golden):
    """The construction_advisor case is the one pilot-critical query whose
    ground truth is verifiable from the repo (construction_kb.json: 70 degC
    core cap, 20 degC differential) — keep those numbers pinned."""
    q = next(x for x in golden["queries"]
             if x["feature"] == "construction_advisor")
    assert any(re.search(p, "core temperature limit 70 degC", REGEX_FLAGS)
               for p in q["answer_expect"])
    assert any(re.search(p, "differential limit 20 degC", REGEX_FLAGS)
               for p in q["answer_expect"])


def test_pass_bar_is_at_least_90_percent(golden):
    n = len(golden["queries"])
    bar = pass_bar(n)
    assert bar / n >= 0.9
    if n == 28:
        assert bar == 26


def test_bad_regex_is_rejected_at_load(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "queries:\n"
        "  - id: x\n"
        "    feature: f\n"
        "    project: p\n"
        "    prompt: hi\n"
        "    answer_expect: ['[unclosed']\n",
        encoding="utf-8",
    )
    with pytest.raises(GoldenSetError, match="bad regex"):
        load_golden_set(bad)
