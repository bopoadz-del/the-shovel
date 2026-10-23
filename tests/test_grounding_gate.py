"""Phase-2 grounding gate — confidence/validation stamp gating (increment 1).

The predefined-dispatch synthesis prompt TELLS the LLM to ground every figure in
the tool result, but nothing ENFORCES it. This gate FLAGS (never blocks) a
validation/confidence stamp asserted in the prose that the computed result does
not back with a corresponding field. It must fire on an unbacked stamp and must
NOT fire on (a) a stamp the result actually backs, or (b) legitimate checklist /
instruction language ("verify the torque"), which is the dominant false-positive
risk.
"""

from __future__ import annotations

from app.core.predefined_reasoning import _gate_confidence_stamps as gate

_CAVEAT = "treat any"


def _flagged(text, result):
    return _CAVEAT in gate(text, result)


class TestStampFiresWhenUnbacked:
    def test_unbacked_confidence_stamp_flags(self):
        assert _flagged("Analysis complete. Confidence: High.", {}) is True

    def test_unbacked_confidence_percent_flags(self):
        assert _flagged("Result computed with 95% confidence.", {"status": "success"}) is True

    def test_unbacked_validation_stamp_flags(self):
        assert _flagged("Validation passed on all line items.", {"total": 5}) is True

    def test_unbacked_quality_assured_flags(self):
        assert _flagged("This is a quality-assured deliverable.", {}) is True


class TestStampNotFlaggedWhenBacked:
    def test_backed_confidence_field(self):
        assert _flagged("Confidence: High.", {"confidence": 0.92}) is False

    def test_backed_validation_field(self):
        assert _flagged("Verified against the calculation engine.",
                        {"validation": {"pass": True}}) is False

    def test_backed_checks_field(self):
        assert _flagged("Quality-assured result.", {"checks": [{"pass": True}]}) is False

    def test_backed_source_note_field(self):
        # historical_benchmark carries source_note/confidence — a real backing.
        assert _flagged("Confidence: medium.", {"source_note": "2024 rate book"}) is False

    def test_backing_field_nested_deep(self):
        assert _flagged("Confidence: high.",
                        {"a": {"b": [{"confidence_score": 0.8}]}}) is False


class TestZeroFalsePositiveOnLegitLanguage:
    """The dominant FP risk: checklist / instruction prose that uses
    'verify'/'validate' as verbs, and plain figures — none are stamps."""

    def test_instructional_verify_not_flagged(self):
        assert _flagged("Verify the torque settings before energizing.", {}) is False

    def test_validated_by_person_not_flagged(self):
        assert _flagged("Ensure all connections are validated by the QA engineer.", {}) is False

    def test_verification_noun_not_flagged(self):
        # Real commissioning-checklist language observed live.
        assert _flagged("Perform verification of relay settings and record results.", {}) is False

    def test_plain_and_money_figures_not_flagged(self):
        assert _flagged("Total: 700 man-hours across 42 trades. Peak Week 1.", {}) is False
        assert _flagged("Net due this period: SAR 120,000.", {}) is False


class TestGateNeverBlocks:
    def test_flag_appends_never_replaces(self):
        original = "Analysis complete. Confidence: High."
        out = gate(original, {})
        # The original text is preserved in full; only a caveat is appended.
        assert out.startswith(original)
        assert len(out) > len(original)


# ── increment 2: money/rate figure grounding ────────────────────────────────

import pytest  # noqa: E402
from app.core.predefined_reasoning import (  # noqa: E402
    _gate_money_figures as mgate,
    _collect_result_numbers,
)


def _mflagged(text, result):
    return "could not be traced" in mgate(text, result)


class TestMoneyFiguresFlaggedWhenUngrounded:
    def test_fabricated_rate_flagged(self):
        assert _mflagged("The rate is SAR 1,200/day.", {"total": 100}) is True

    def test_fabricated_currency_amount_flagged(self):
        assert _mflagged("Estimated cost: SAR 5,000,000.", {"subtotal": 42}) is True

    def test_multiple_ungrounded_listed(self):
        out = mgate("Day rate SAR 999,999/day and material USD 12,345.", {"x": 1})
        assert "999,999" in out and "12,345" in out


class TestMoneyFiguresNotFlaggedWhenGrounded:
    def test_grounded_amount_not_flagged(self):
        # 120000 is in the result → the SAR figure grounds.
        assert _mflagged("Net due: SAR 120,000.",
                         {"payment": {"net_due": 120000.0}}) is False

    def test_rounding_tolerance(self):
        # LLM renders 300000.0 as 'SAR 300,000' → still grounds.
        assert _mflagged("Gross: SAR 300,000.", {"gross_valuation": 300000.0}) is False

    def test_number_embedded_in_result_string_grounds(self):
        assert _mflagged("IPC total SAR 220,000.",
                         {"summary": "Cumulative certified SAR 220,000 to date"}) is False

    def test_bare_non_currency_numbers_never_checked(self):
        # counts / weeks / years are not currency-shaped → never flagged
        assert _mflagged("700 man-hours across 42 trades, 34 months.", {}) is False


class TestMoneyGateZeroFalsePositivesOnRealDeliverables:
    """The real acceptance (per plan): render every money number in a real
    financial result as currency and confirm the gate flags NONE of them."""

    @pytest.mark.asyncio
    async def test_payment_certificate_zero_fp(self):
        from app.containers.construction import ConstructionContainer
        c = ConstructionContainer()
        r = await c.payment_certificate(
            {"contract_value": 1_000_000, "work_done_percent": 30,
             "previous_certified": 100_000, "retention_percent": 10,
             "advance_payment": 50_000, "advance_recovery_percent": 20},
            {"payment_period": "Month 3"})
        answer = "Payment certificate. " + "; ".join(
            f"SAR {n:,.2f}" for n in sorted(_collect_result_numbers(r)) if n >= 1)
        assert _mflagged(answer, r) is False  # every figure is result-derived

    @pytest.mark.asyncio
    async def test_cash_flow_forecast_zero_fp(self):
        from app.containers.construction import ConstructionContainer
        c = ConstructionContainer()
        r = await c.cash_flow_forecast(
            {"contract_value": 5_000_000, "duration_months": 12}, {})
        answer = "Cash flow. " + "; ".join(
            f"SAR {n:,.2f}" for n in sorted(_collect_result_numbers(r)) if n >= 1)
        assert _mflagged(answer, r) is False
