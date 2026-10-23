"""S5 — eval-battery harness: NO_OUTPUT classification + battery summary.

The 2026-07 results audit found most `answer_chars < min_chars` FAILs were
near-empty husks (60–250 chars), i.e. "produced ~nothing", and that reading
them as quality FAILs made the headline pass-rate untriageable. The sweep now
tags a clean-but-near-empty turn as ``execution_subtype: NO_OUTPUT`` (triage
class 5) — an explicit error signal always keeps priority.
"""
from __future__ import annotations

import json

from scripts.feature_matrix_sweep import (
    execution_subtype,
    judge_execution,
    triage_class,
)
from scripts.run_eval_battery import _fmt_counts, _jsonl_counts

FEAT = {"min_chars": 800}


def test_near_empty_clean_turn_is_no_output():
    res = {"answer": "x" * 93, "errors": [], "http_error": None}
    verdict, reasons = judge_execution(FEAT, res)
    assert verdict == "FAIL"
    assert execution_subtype(FEAT, res) == "NO_OUTPUT"


def test_terse_but_substantial_answer_is_not_no_output():
    # 700/800 chars: a genuine near-miss, not a husk.
    res = {"answer": "x" * 700, "errors": [], "http_error": None}
    assert judge_execution(FEAT, res)[0] == "FAIL"
    assert execution_subtype(FEAT, res) is None


def test_error_event_keeps_priority_over_no_output():
    res = {"answer": "x" * 10, "errors": ["Response timeout"], "http_error": None}
    assert execution_subtype(FEAT, res) is None


def test_http_error_keeps_priority_over_no_output():
    res = {"answer": "", "errors": [], "http_error": "HTTP 502"}
    assert execution_subtype(FEAT, res) is None


def test_no_min_chars_never_no_output():
    res = {"answer": "", "errors": [], "http_error": None}
    assert execution_subtype({"min_chars": 0}, res) is None


def test_boundary_is_thirty_percent_of_min_chars():
    just_below = {"answer": "x" * 239, "errors": [], "http_error": None}
    at_line = {"answer": "x" * 240, "errors": [], "http_error": None}
    assert execution_subtype(FEAT, just_below) == "NO_OUTPUT"
    assert execution_subtype(FEAT, at_line) is None


def test_triage_class_5_for_no_output_fail():
    rec = {"verdict": "FAIL", "routing": "PASS", "execution_subtype": "NO_OUTPUT"}
    assert triage_class(rec) == "5"


def test_triage_class_routing_miss_beats_no_output():
    rec = {"verdict": "FAIL", "routing": "FAIL", "execution_subtype": "NO_OUTPUT"}
    assert triage_class(rec) == "1"


def test_triage_class_2_for_plain_execution_fail():
    rec = {"verdict": "FAIL", "routing": "PASS", "execution_subtype": None}
    assert triage_class(rec) == "2"


# ── battery summary helpers ──────────────────────────────────────────────────

def test_jsonl_counts_last_wins(tmp_path):
    p = tmp_path / "r.jsonl"
    rows = [
        {"action": "a", "prompt_index": 0, "run_index": 0, "verdict": "FAIL"},
        {"action": "a", "prompt_index": 0, "run_index": 0, "verdict": "PASS"},  # rerun
        {"action": "b", "prompt_index": 0, "run_index": 0, "verdict": "PARTIAL"},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert _jsonl_counts(p) == {"PASS": 1, "PARTIAL": 1}


def test_jsonl_counts_missing_file():
    from pathlib import Path
    assert _jsonl_counts(Path("/nonexistent/x.jsonl")) == {}


def test_fmt_counts_ordering_and_table_safety():
    out = _fmt_counts({"FAIL": 3, "PASS": 25, "PARTIAL": 7})
    assert out == "PASS 25 · PARTIAL 7 · FAIL 3"
    assert "|" not in out  # lands inside a markdown table cell
    assert _fmt_counts({}) == "no results file"


# ── honest-refusal → BLOCKED heuristic ───────────────────────────────────────

def test_blocked_re_matches_honest_refusals():
    from scripts.feature_matrix_sweep import _BLOCKED_RE

    refusals = [
        "I don't have that information. The retrieved project context does "
        "not contain any RFI register, RFI log, or tracking data.",
        "The retrieved context does not contain specific basement options.",
        "If you have one, upload it and I'll extract the open items for you.",
        "To answer this, I would need an RFI register or log document uploaded "
        "and indexed in the project.",
        "please upload the priced BOQ first",
        "No baseline schedule has been uploaded for this project.",
    ]
    for text in refusals:
        assert _BLOCKED_RE.search(text), f"not matched: {text!r}"


def test_blocked_re_does_not_match_real_answers():
    from scripts.feature_matrix_sweep import _BLOCKED_RE

    answers = [
        "The total BOQ value is SAR 6,938,130 across 13 line items.",
        "Open VOs: VO-03 (SAR -220,000) and VO-04 (SAR 640,000), both Pending.",
        "CPI = 0.87 and SPI = 0.91 indicate the project is over budget and behind schedule.",
    ]
    for text in answers:
        assert not _BLOCKED_RE.search(text), f"false positive: {text!r}"


def test_manifest_fixture_projects_and_needs_are_declared():
    """Every data-dependent feature names a canonical FIXTURE project and a
    needs line — hardcoded live project ids are a defect (they die on every
    fixture reseed)."""
    import yaml
    from scripts.feature_matrix_sweep import DEFAULT_MANIFEST

    m = yaml.safe_load(open(DEFAULT_MANIFEST, encoding="utf-8"))
    feats = {f["action"]: f for f in m["features"]}
    for action in (
        "boq_process", "estimate_costs", "extract_quantities",
        "procurement_list_generator", "procurement_optimizer",
        "value_engineering", "spec_analyze", "process_specification_full",
        "rfi_management", "variation_order_manager", "change_order_impact",
        "document_metadata", "parse_primavera_schedule", "progress_tracker",
        "forensic_delay_analysis",
    ):
        f = feats[action]
        assert str(f.get("project", "")).startswith("FIXTURE —"), (
            f"{action}: project must be a canonical FIXTURE name, got {f.get('project')!r}"
        )
        assert f.get("needs"), f"{action}: missing needs declaration"
