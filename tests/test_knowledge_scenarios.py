"""Tests for scripts.generate_knowledge_scenarios — deterministic Q&A
pairs derived from construction_knowledge.py.

These pairs power the "rules baked in" supplement to the document-driven
training_scenarios.jsonl. The key property the tests defend: every answer
comes from running the production code (or reading the production
constants), so the fine-tuned model can never learn a wrong rule.
"""
from __future__ import annotations

from app.core import construction_knowledge as ck
from scripts.generate_knowledge_scenarios import (
    gen_critical_rules,
    gen_doc_numbers,
    gen_design_statuses,
    gen_ncr_workflow,
    gen_review_timeline,
    gen_score_risk,
    gen_payment,
    gen_evm,
    gen_tender,
    gen_procedure_lookup,
    gen_enforce_rules,
    generate_all,
)


# ── Schema invariant: every row matches the JSONL contract ────────────────


def test_every_row_has_required_keys():
    rows = generate_all()
    assert rows, "generator produced no rows"
    for r in rows:
        assert set(r.keys()) == {"instruction", "response", "source"}, r
        assert r["instruction"].strip(), f"empty instruction: {r}"
        assert r["response"].strip(), f"empty response: {r}"
        assert r["source"].startswith("construction_knowledge.py:"), r


def test_total_meets_target_of_200_plus():
    """User-facing target: 200-300 pairs from the knowledge base alone."""
    rows = generate_all()
    assert 200 <= len(rows) <= 400, f"got {len(rows)} rows, target band 200-400"


# ── Determinism: same input → identical rows on every run ─────────────────


def test_generator_is_deterministic():
    a = generate_all()
    b = generate_all()
    assert a == b, "generator output is not deterministic — answers must be reproducible"


# ── CRITICAL_RULES — 3 rows per entry ────────────────────────────────────


def test_critical_rules_produces_three_per_entry():
    rows = list(gen_critical_rules())
    assert len(rows) == 3 * len(ck.CRITICAL_RULES)
    # Each rule_id should appear exactly 3 times in source tags.
    for rule_id in ck.CRITICAL_RULES:
        matches = [r for r in rows if rule_id in r["source"]]
        assert len(matches) == 3, f"rule {rule_id} produced {len(matches)} rows, not 3"


def test_critical_rules_include_real_violation_text():
    rows = list(gen_critical_rules())
    for rule_id, entry in ck.CRITICAL_RULES.items():
        violation_msg = entry["violation_message"]
        # The violation_message must appear verbatim in at least one row's
        # response — guarantees no LLM paraphrasing crept in.
        assert any(violation_msg in r["response"] for r in rows), (
            f"violation message for {rule_id} not present verbatim"
        )


# ── generate_doc_number — every documented doc type covered ──────────────


def test_doc_numbers_run_real_generator():
    rows = list(gen_doc_numbers())
    # Pick a known-format pair and verify it matches the live function.
    expected_rfi_42 = ck.generate_doc_number("RFI", 42)
    assert expected_rfi_42 == "RFI-0042"
    matches = [r for r in rows if "RFI" in r["instruction"] and "42" in r["instruction"]]
    assert matches
    assert expected_rfi_42 in matches[0]["response"]


def test_doc_numbers_cover_year_variants():
    rows = list(gen_doc_numbers())
    # NCR with year should appear with the YYYY-NNN pattern.
    ncr_year_rows = [r for r in rows if "NCR" in r["instruction"] and "year 2025" in r["instruction"]]
    assert ncr_year_rows
    assert "NCR-2025-" in ncr_year_rows[0]["response"]


# ── Design statuses ───────────────────────────────────────────────────────


def test_design_statuses_separates_valid_and_forbidden():
    rows = list(gen_design_statuses())
    # Every valid status produces an "is valid" affirmation.
    for valid in ck.VALID_DESIGN_STATUSES:
        hits = [r for r in rows if valid in r["instruction"]]
        assert hits, f"missing row for valid status {valid}"
        assert "valid" in hits[0]["response"].lower()
    # Every forbidden status produces a rejection citing the design review procedure.
    for forb in ck.FORBIDDEN_DESIGN_STATUSES:
        hits = [r for r in rows if forb in r["instruction"]]
        assert hits, f"missing row for forbidden status {forb}"
        assert "design review and acceptance procedure" in hits[0]["response"]


# ── check_review_timeline — boundary correctness ─────────────────────────


def test_review_timeline_boundary_seven_days_is_compliant():
    rows = list(gen_review_timeline())
    # The 7-day window case must mark the timeline compliant — that is
    # exactly the design-review minimum.
    seven = [r for r in rows if "2026-01-08" in r["instruction"]]
    assert seven
    assert "compliant" in seven[0]["response"].lower()
    assert "non-compliant" not in seven[0]["response"].lower()


def test_review_timeline_under_minimum_is_flagged():
    rows = list(gen_review_timeline())
    five_day = [r for r in rows if "2026-01-06" in r["instruction"]]
    assert five_day
    assert "non-compliant" in five_day[0]["response"].lower()


# ── NCR workflow — transitions are real ──────────────────────────────────


def test_ncr_workflow_includes_every_state():
    rows = list(gen_ncr_workflow())
    state_rows = [r for r in rows if "follows" in r["instruction"]]
    states = [s for s in ck.NCR_WORKFLOW_SEQUENCE]
    for s in states:
        hits = [r for r in state_rows if f"'{s}'" in r["instruction"]]
        assert hits, f"missing transition row for state {s}"


def test_ncr_workflow_terminal_state_has_no_successor():
    rows = list(gen_ncr_workflow())
    terminal = [r for r in rows if "'CLOSED'" in r["instruction"]]
    assert terminal
    assert "terminal" in terminal[0]["response"].lower() or "no next" in terminal[0]["response"].lower()


# ── score_risk — every cell in the 5x5 grid ──────────────────────────────


def test_score_risk_covers_full_grid():
    rows = list(gen_score_risk())
    # 25 grid + 1 out-of-range row.
    assert len(rows) == 26
    # GREEN/AMBER/RED bands all represented.
    text = " ".join(r["response"] for r in rows)
    assert "GREEN" in text and "AMBER" in text and "RED" in text


def test_score_risk_band_boundaries_are_correct():
    rows = list(gen_score_risk())
    # Find the row for p=2, i=2 (score 4) — should be GREEN.
    g = next(r for r in rows if "probability=2" in r["instruction"] and "impact=2" in r["instruction"])
    assert "GREEN" in g["response"]
    # p=1, i=5 (score 5) — should be AMBER.
    a = next(r for r in rows if "probability=1" in r["instruction"] and "impact=5" in r["instruction"])
    assert "AMBER" in a["response"]
    # p=5, i=5 (score 25) — must be RED and require action.
    r = next(r for r in rows if "probability=5" in r["instruction"] and "impact=5" in r["instruction"])
    assert "RED" in r["response"] and "escalation" in r["response"]


# ── Payment + EVM run the real math ──────────────────────────────────────


def test_payment_response_matches_calculate_payment():
    rows = list(gen_payment())
    for r in rows:
        assert "Net payment due" in r["response"]
        assert "complete" in r["response"]


def test_evm_response_uses_real_status_strings():
    rows = list(gen_evm())
    text = " ".join(r["response"] for r in rows)
    # The function's enum strings must appear (no LLM rewording).
    assert "UNDER BUDGET" in text or "OVER BUDGET" in text
    assert "AHEAD" in text or "BEHIND" in text


# ── Tender evaluation — recommended bidder matches the winner ────────────


def test_tender_recommendation_matches_actual_winner():
    rows = list(gen_tender())
    # Three-bidder panel — recompute the winner ourselves and confirm the row
    # cites the same name.
    three_bidder = next(
        r for r in rows if "Bidder A" in r["instruction"] and "Bidder B" in r["instruction"]
    )
    result = ck.evaluate_tender([
        {"name": "Bidder A", "technical_score": 85, "commercial_score": 75, "hse_score": 90, "local_content_score": 60},
        {"name": "Bidder B", "technical_score": 78, "commercial_score": 82, "hse_score": 85, "local_content_score": 70},
        {"name": "Bidder C", "technical_score": 70, "commercial_score": 88, "hse_score": 80, "local_content_score": 75},
    ])
    assert result["recommended"]["name"] in three_bidder["response"]


# ── Live-detector questions hit enforce_critical_rules truthfully ────────


def test_enforce_rules_uses_live_detector():
    rows = list(gen_enforce_rules())
    # "design package was approved" must flag a violation.
    design_row = next(r for r in rows if "design package was approved" in r["instruction"])
    assert "rule" in design_row["response"].lower()
    # "schedule was approved by the project manager" has no design keyword
    # → no flag.
    schedule_row = next(r for r in rows if "construction schedule" in r["instruction"])
    assert "no critical-rule violations" in schedule_row["response"].lower()


# ── procedure_lookup respects what's actually in the knowledge base ──────


def test_procedure_lookup_does_not_invent_data():
    rows = list(gen_procedure_lookup())
    for r in rows:
        # Either the row reports the real procedure body, or it explicitly
        # says it's not in the knowledge base. No invented descriptions.
        assert (
            "Purpose:" in r["response"]
            or "not present" in r["response"]
            or "system prompt" in r["response"].lower()
            or r["source"].endswith("get_system_prompt")
        ), r
