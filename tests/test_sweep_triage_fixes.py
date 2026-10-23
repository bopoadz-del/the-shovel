"""Two real bugs surfaced by the 2026-07-25 fresh sweep triage.

Both had the answer's numbers sitting in the prompt, same family as the
payment_certificate / cash_flow figures-from-message fixes:

1. carbon_footprint_calculator computed ZERO for "12,000 m3 of C40
   concrete and 900t of rebar" — predefined dispatch sent empty params,
   so the quantities in the message were never parsed.
2. sympy_reason variance question ("planned SAR 4.2M vs actual SAR 5.1M")
   got the cost-grounding refusal — the gate didn't count the USER's own
   supplied figures as grounded, so it refused to echo them back.
"""

from __future__ import annotations

import pytest


class TestCarbonQuantitiesFromMessage:
    def test_extracts_concrete_and_rebar(self):
        from app.containers.construction.boq import _carbon_quantities_from_message
        q = _carbon_quantities_from_message(
            "calculate the embodied carbon for 12,000 m3 of C40 concrete and 900t of rebar"
        )
        assert q["concrete_m3"] == 12000.0
        assert q["rebar_kg"] == 900_000.0  # 900 tonnes -> kg

    @pytest.mark.asyncio
    async def test_carbon_action_nonzero_from_message(self):
        from app.containers.construction import ConstructionContainer
        r = await ConstructionContainer().carbon_footprint_calculator(
            {"message": "embodied carbon for 12,000 m3 of C40 concrete and 900t of rebar"},
            {},
        )
        assert r["status"] == "success"
        assert r["total_embodied_carbon_kg"] > 0
        # concrete 12000*250 + rebar 900000*1.9 = 3,000,000 + 1,710,000
        assert r["total_embodied_carbon_kg"] == pytest.approx(4_710_000.0, rel=1e-6)

    @pytest.mark.asyncio
    async def test_explicit_quantities_still_win(self):
        from app.containers.construction import ConstructionContainer
        r = await ConstructionContainer().carbon_footprint_calculator(
            {"quantities": {"concrete_m3": 100}, "message": "999999 m3 concrete"}, {},
        )
        assert r["total_embodied_carbon_kg"] == pytest.approx(25_000.0)  # 100*250


class TestCostGateGroundsUserFigures:
    def test_user_supplied_numbers_are_grounded(self):
        from app.agents.runtime import _cg_grounded_numbers
        messages = [{"role": "user",
                     "content": "variance analysis: planned SAR 4.2M vs actual SAR 5.1M"}]
        grounded = _cg_grounded_numbers("", messages)
        # 4.2M and 5.1M the user typed must be in the grounded set.
        assert any(abs(g - 4_200_000) <= 21000 for g in grounded)
        assert any(abs(g - 5_100_000) <= 25500 for g in grounded)

    def test_variance_answer_passes_the_gate(self):
        from app.agents.runtime import _cost_grounding_gate
        messages = [{"role": "user",
                     "content": "planned SAR 4.2M vs actual SAR 5.1M, what drove the overrun?"}]
        answer = ("The overrun is SAR 0.9M: actual SAR 5.1M against planned "
                  "SAR 4.2M, a 21% variance driven by rate escalation.")
        out = _cost_grounding_gate(answer, None, messages)
        assert out == answer  # echoing the user's own figures is not refused

    def test_fabricated_rate_still_refused(self, monkeypatch):
        from app.agents import runtime as rt
        # A figure the user NEVER supplied, no RAG, no tool -> still refused.
        messages = [{"role": "user", "content": "what's the rate for C40 concrete?"}]
        answer = "The rate for C40 concrete is SAR 465 per cubic metre."
        out = rt._cost_grounding_gate(answer, None, messages)
        assert out == rt._CG_REFUSAL

    def test_english_number_words_ground_percent_of_product(self):
        from app.agents.runtime import _cg_grounded_numbers, _cost_grounding_gate
        messages = [{"role": "user", "content": (
            "LD cap is ten percent of four million eight hundred thousand"
        )}]
        grounded = _cg_grounded_numbers("", messages)
        assert any(abs(g - 480_000) <= 2400 for g in grounded)
        answer = "Cap = 480,000 SAR."
        assert _cost_grounding_gate(answer, None, messages) == answer


class TestQ12PaymentPhrasing:
    """Live 2026-08-02 verification: Q2 (guardrail height) now passes via
    construction_calc, but Q12 STILL refused — 'Calculate the interim payment
    for work valued at 900,000 with 10 percent retention' routed to
    payment_certificate and the extractor missed BOTH figures: 'work valued
    at <n>' is not a 'gross valuation' label, and '10 percent retention' puts
    the percent BEFORE the label. Same figures-in-the-prompt family as above."""

    def test_work_valued_at_and_percent_before_retention(self):
        from app.containers.construction.boq import _payment_figures_from_message
        fig = _payment_figures_from_message(
            "Calculate the interim payment for work valued at 900,000 "
            "with 10 percent retention."
        )
        assert fig["gross_valuation"] == 900_000.0
        assert fig["retention_percent"] == 10.0

    def test_value_of_work_done_phrasing(self):
        from app.containers.construction.boq import _payment_figures_from_message
        fig = _payment_figures_from_message(
            "issue an IPC — value of work done SAR 2,500,000, retention 5%"
        )
        assert fig["gross_valuation"] == 2_500_000.0
        assert fig["retention_percent"] == 5.0

    def test_label_forms_still_win_and_do_not_regress(self):
        from app.containers.construction.boq import _payment_figures_from_message
        fig = _payment_figures_from_message(
            "gross valuation SAR 10,000,000, retention 10%, advance recovery 20%"
        )
        assert fig["gross_valuation"] == 10_000_000.0
        assert fig["retention_percent"] == 10.0
        assert fig["advance_recovery_percent"] == 20.0

    @pytest.mark.asyncio
    async def test_q12_certificate_issues_end_to_end(self):
        from app.containers.construction import ConstructionContainer
        r = await ConstructionContainer().payment_certificate(
            {"message": "Calculate the interim payment for work valued at "
                        "900,000 with 10 percent retention."},
            {},
        )
        assert r["status"] == "success"
        assert r["valuation"]["gross_valuation"] == pytest.approx(900_000.0)
        assert r["deductions"]["retention_held"] == pytest.approx(90_000.0)
        # FIDIC 14.3 shape: 900,000 - 10% retention = 810,000 net (no advance,
        # no previous certified in this prompt)
        assert r["payment"]["net_due_this_period"] == pytest.approx(810_000.0)
