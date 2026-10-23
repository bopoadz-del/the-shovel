"""Deterministic unit/integration tests for ConstructionContainer
BOQ / commercial / contract-text actions.

These tests exercise the public actions and pure helpers in
``app.containers.construction.boq`` without calling live LLMs, external
network services, or real PDF/BIM/CAD/Primavera files.

Tests that rely on the ``historical_benchmark`` block use
``@requires_construction_kit`` and expect ``CEREBRUM_DOMAIN_KITS`` to
include ``construction`` so the block is registered.
"""

from __future__ import annotations

import pytest

from app.containers.construction import ConstructionContainer
from tests.conftest import requires_construction_kit


@pytest.fixture
def container():
    return ConstructionContainer()


# ══════════════════════════════════════════════════════════════════════════════
# Pure helpers
# ══════════════════════════════════════════════════════════════════════════════


class TestCalculateQuantities:
    def test_area_and_volume_measurements(self, container):
        measurements = [
            {"type": "area", "value": 500.0, "item": "floor"},
            {"type": "volume", "value": 120.0, "item": "concrete"},
        ]
        q = container._calculate_quantities(measurements)
        assert q["floor_area_m2"] == 500.0
        assert q["concrete_volume_m3"] == 120.0
        assert q["steel_weight_kg"] == 120.0 * 120  # DEFAULT_REBAR_RATIO

    def test_area_fallback_slab_volume(self, container):
        measurements = [{"type": "area", "value": 1000.0, "item": "floor"}]
        q = container._calculate_quantities(measurements)
        assert q["floor_area_m2"] == 1000.0
        assert q["concrete_volume_m3"] == 1000.0 * 0.15  # DEFAULT_SLAB_THICKNESS

    def test_count_whitelisted_items(self, container):
        measurements = [
            {"type": "count", "value": 42, "item": "fire door"},
            {"type": "count", "value": 7, "item": "Server hall"},
        ]
        q = container._calculate_quantities(measurements)
        assert q["fire_door_count"] == 42
        assert "server_hall_count" not in q


class TestEstimateCosts:
    def test_requires_rates(self, container):
        result = container._estimate_costs({"concrete_volume_m3": 10}, rates={})
        assert result["status"] == "error"
        assert "concrete_usd_per_m3" in result["error"]

    def test_with_rates(self, container):
        result = container._estimate_costs(
            {"concrete_volume_m3": 10, "steel_weight_kg": 500},
            rates={"concrete_usd_per_m3": 150, "steel_usd_per_kg": 1.2},
        )
        assert result["concrete_cost"] == 1500.0
        assert result["steel_cost"] == 600.0
        assert result["subtotal"] == 2100.0
        assert result["total"] == 2100.0


class TestClassifyProcurementItem:
    def test_structural_steel(self, container):
        cat, lead, supplier = container._classify_procurement_item("Structural steel frame")
        assert cat == "Structural Steel"
        assert lead == 16
        assert supplier == "Steel Fabricator"

    def test_concrete(self, container):
        cat, lead, supplier = container._classify_procurement_item("Ready-mix concrete")
        assert cat == "Concrete"
        assert lead == 2
        assert supplier == "Ready-Mix Supplier"

    def test_general_fallback(self, container):
        cat, lead, supplier = container._classify_procurement_item("Office supplies")
        assert cat == "General Materials"
        assert lead == 4


class TestGenerateProcurementRecommendations:
    def test_critical_long_lead_and_spend(self, container):
        items = [
            {"priority": "critical", "category": "Mechanical / HVAC", "total_cost": 6_000_000, "lead_time_weeks": 16},
            {"priority": "normal", "category": "Electrical", "total_cost": 200_000, "lead_time_weeks": 8},
        ]
        recs = container._generate_procurement_recommendations(items)
        assert any("Immediate action" in r for r in recs)
        assert any("combined MEP package" in r for r in recs)
        assert any("Spend > $5M" in r for r in recs)


class TestCalculateCoCostImpact:
    def test_scope_addition(self, container):
        analysis = {"category": "scope_addition", "complexity": "medium"}
        result = container._calculate_co_cost_impact(100_000, analysis)
        assert result["direct_cost"] == 100_000.0
        assert result["overhead"] == 20_000.0
        assert result["profit"] == 10_000.0
        assert result["risk_allowance"] == 10_000.0
        assert result["total"] == 140_000.0

    def test_omission_no_profit(self, container):
        analysis = {"category": "scope_omission", "complexity": "low"}
        result = container._calculate_co_cost_impact(50_000, analysis)
        assert result["profit"] == 0.0
        assert result["total"] == 50_000.0 + 10_000.0 + 2_500.0  # direct + overhead + risk


class TestCategorizeVariation:
    def test_design_change(self, container):
        assert container._categorize_variation("Update drawings for lobby") == "design_change"

    def test_unforeseen_condition(self, container):
        assert container._categorize_variation("Unforeseen rock encountered") == "unforeseen_condition"

    def test_scope_addition_default(self, container):
        assert container._categorize_variation("Extra facade works") == "scope_addition"


class TestCalculateVariationPrice:
    def test_addition(self, container):
        vo_data = {"direct_cost": 10_000, "quantity": 2, "overhead_percent": 0.10, "profit_percent": 0.08}
        result = container._calculate_variation_price(vo_data, "addition")
        direct = 20_000.0
        indirect = direct * 0.15
        overhead = (direct + indirect) * 0.10
        profit = (direct + indirect) * 0.08
        assert result["direct"] == round(direct, 2)
        assert result["indirect"] == round(indirect, 2)
        assert result["total"] == round(direct + indirect + overhead + profit, 2)

    def test_omission_negative_overhead(self, container):
        vo_data = {"direct_cost": 10_000, "quantity": 1}
        result = container._calculate_variation_price(vo_data, "omission")
        assert result["direct"] == 10_000.0
        assert result["indirect"] == 0.0
        assert result["overhead"] == -1_000.0
        assert result["profit"] == -800.0
        assert result["total"] == 8_200.0


# ══════════════════════════════════════════════════════════════════════════════
# Public actions
# ══════════════════════════════════════════════════════════════════════════════


class TestPaymentCertificate:
    @pytest.mark.asyncio
    async def test_payment_certificate_from_contract_value(self, container):
        result = await container.payment_certificate(
            {
                "contract_value": 1_000_000,
                "work_done_percent": 30,
                "previous_certified": 100_000,
                "retention_percent": 10,
                "advance_payment": 50_000,
                "advance_recovery_percent": 20,
            },
            {"payment_period": "Month 3"},
        )
        assert result["status"] == "success"
        assert result["action"] == "payment_certificate"
        assert result["certificate"]["period"] == "Month 3"
        assert result["valuation"]["contract_value"] == 1_000_000.0
        assert result["valuation"]["gross_valuation"] == 300_000.0
        assert result["deductions"]["retention_held"] == 30_000.0
        assert result["deductions"]["advance_recovery"] == 50_000.0
        assert result["payment"]["net_due_this_period"] == 120_000.0
        assert result["payment"]["cumulative_certified"] == 220_000.0

    @pytest.mark.asyncio
    async def test_payment_certificate_from_gross_valuation(self, container):
        result = await container.payment_certificate(
            {"gross_valuation": 250_000},
            {"retention_percent": 5},
        )
        assert result["status"] == "success"
        assert result["valuation"]["gross_valuation"] == 250_000.0
        assert result["payment"]["net_due_this_period"] == 237_500.0

    @pytest.mark.asyncio
    async def test_payment_certificate_figures_from_message(self, container):
        """Predefined dispatch sends the chat envelope as input_data with empty
        params (2026-07-24 live probe): the figures exist only in `message`.
        The action must parse them instead of erroring."""
        result = await container.payment_certificate(
            {
                "action": "payment_certificate",
                "project_id": "p1",
                "document_ids": [],
                "message": (
                    "Issue a payment certificate: gross valuation SAR 10,000,000, "
                    "retention 10%, advance recovery SAR 500,000, "
                    "previous certificates SAR 6,000,000."
                ),
            },
            {},
        )
        assert result["status"] == "success"
        assert result["valuation"]["gross_valuation"] == 10_000_000.0
        assert result["deductions"]["retention_held"] == 1_000_000.0
        assert result["deductions"]["advance_recovery"] == 500_000.0
        assert result["deductions"]["previous_payments"] == 6_000_000.0
        assert result["payment"]["net_due_this_period"] == 2_500_000.0

    @pytest.mark.asyncio
    async def test_payment_certificate_contract_value_from_message(self, container):
        result = await container.payment_certificate(
            {
                "message": (
                    "issue an interim payment certificate for a contract value of "
                    "SAR 2,000,000 with 25% work done and retention 5 percent"
                ),
            },
            {},
        )
        assert result["status"] == "success"
        assert result["valuation"]["contract_value"] == 2_000_000.0
        assert result["valuation"]["gross_valuation"] == 500_000.0
        assert result["deductions"]["retention_held"] == 25_000.0

    @pytest.mark.asyncio
    async def test_payment_certificate_message_without_figures_still_errors(self, container):
        result = await container.payment_certificate(
            {"message": "issue a payment certificate"},
            {},
        )
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_payment_certificate_explicit_params_beat_message(self, container):
        """Explicit params/data figures always win over message parsing."""
        result = await container.payment_certificate(
            {
                "gross_valuation": 250_000,
                "message": "gross valuation SAR 999,999",
            },
            {"retention_percent": 5},
        )
        assert result["valuation"]["gross_valuation"] == 250_000.0

    @pytest.mark.asyncio
    async def test_payment_certificate_bare_gross_amount_from_message(self, container):
        """Live F5: 'gross 750000 with 5 percent retention' had no 'valuation' word."""
        result = await container.payment_certificate(
            {
                "message": (
                    "Interim payment certificate gross 750000 with 5 percent retention. "
                    "Net payable?"
                ),
            },
            {},
        )
        assert result["status"] == "success"
        assert result["valuation"]["gross_valuation"] == 750_000.0
        assert result["deductions"]["retention_held"] == 37_500.0
        assert result["payment"]["net_due_this_period"] == 712_500.0

    @pytest.mark.asyncio
    async def test_cash_flow_figures_from_message_with_m_suffix(self, container):
        """2026-07-24 battery FAIL repro: 'SAR 60M, 18-month project' errored
        with 'contract_value required' — figures lived only in the message."""
        result = await container.cash_flow_forecast(
            {"message": "build a cash flow forecast with an S-curve for a SAR 60M, 18-month project"},
            {},
        )
        assert result["status"] == "success"
        assert result["project_parameters"]["contract_value"] == 60_000_000.0
        assert len(result["s_curve_data"]) == 18

    @pytest.mark.asyncio
    async def test_cash_flow_plain_amount_from_message(self, container):
        result = await container.cash_flow_forecast(
            {"message": "cash flow please, contract value SAR 24,000,000 over a 12-month duration"},
            {},
        )
        assert result["status"] == "success"
        assert result["project_parameters"]["contract_value"] == 24_000_000.0
        assert len(result["s_curve_data"]) == 12

    @pytest.mark.asyncio
    async def test_cash_flow_still_refuses_without_figures(self, container):
        result = await container.cash_flow_forecast(
            {"message": "what does the cumulative spend curve look like month by month?"},
            {},
        )
        assert result["status"] == "error"
        assert "contract_value" in result["error"]

    @pytest.mark.asyncio
    async def test_cash_flow_explicit_params_beat_message(self, container):
        result = await container.cash_flow_forecast(
            {"contract_value": 10_000_000, "message": "for a SAR 99M project"},
            {"duration_months": 6},
        )
        assert result["status"] == "success"
        assert result["project_parameters"]["contract_value"] == 10_000_000
        assert len(result["s_curve_data"]) == 6

    @pytest.mark.asyncio
    async def test_payment_certificate_remaining_balance_no_double_retention(self, container):
        """Regression: remaining_contract_balance must not subtract retention twice."""
        result = await container.payment_certificate(
            {
                "contract_value": 1_000_000,
                "work_done_percent": 50,
                "previous_certified": 0,
                "retention_percent": 10,
            },
            {},
        )
        gross = 500_000.0
        retention = 50_000.0
        net = gross - retention
        cumulative = net
        # Before fix this was contract_value - cumulative - retention (wrong).
        expected_remaining = 1_000_000.0 - cumulative
        assert result["payment"]["cumulative_certified"] == cumulative
        assert result["payment"]["remaining_contract_balance"] == expected_remaining

    @pytest.mark.asyncio
    async def test_payment_certificate_missing_value_error(self, container):
        result = await container.payment_certificate({}, {})
        assert result["status"] == "error"
        assert "contract value" in result["error"].lower()


class TestCashFlowForecast:
    @pytest.mark.asyncio
    async def test_cash_flow_forecast_shape_and_retention(self, container):
        result = await container.cash_flow_forecast(
            {"contract_value": 2_400_000},
            {
                "duration_months": 12,
                "project_start_date": "2026-01-15",
                "payment_terms": {
                    "advance_payment": 0.10,
                    "retention": 0.10,
                    "payment_delay_days": 30,
                    "mobilization_duration": 2,
                },
            },
        )
        assert result["status"] == "success"
        assert result["action"] == "cash_flow_forecast"
        forecast = result["s_curve_data"]
        assert len(forecast) == 12
        assert forecast[0]["month"] == 1
        # Advance paid in month 0 and recovered over first ~80% of duration.
        assert forecast[0]["net_cash_in"] > 0
        assert result["summary_metrics"]["total_planned_revenue"] > 0
        assert result["summary_metrics"]["peak_month"] is not None
        # Retention should be released exactly once when progress crosses 95%.
        releases = [m["retention_release"] for m in forecast if m["retention_release"] > 0]
        assert len(releases) == 1

    @pytest.mark.asyncio
    async def test_cash_flow_forecast_requires_contract_value(self, container):
        result = await container.cash_flow_forecast({}, {})
        assert result["status"] == "error"
        assert "contract_value" in result["error"]


class TestGenerateCarbonReport:
    @pytest.mark.asyncio
    async def test_generate_carbon_report(self, container):
        result = await container.generate_carbon_report(
            {
                "quantities": {
                    "concrete_m3": {"quantity": 100},
                    "steel_kg": {"quantity": 1000},
                    "unknown_item": {"quantity": 5},
                }
            },
            {},
        )
        assert result["status"] == "success"
        assert result["action"] == "carbon_report"
        assert result["total_embodied_carbon_kg"] == 100 * 250 + 1000 * 2.3 + 5 * 100
        assert result["total_tonnes_co2"] == round(result["total_embodied_carbon_kg"] / 1000, 2)
        assert len(result["breakdown"]) == 3


class TestRiskRegisterAutoPopulate:
    @pytest.mark.asyncio
    async def test_risk_register_auto_populate(self, container):
        result = await container.risk_register_auto_populate(
            {
                "risks": [
                    {"category": "Safety", "description": "Fall hazard", "severity": "high", "mitigation": "Guardrails"},
                    {"category": "Schedule", "description": "Material delay", "severity": "medium"},
                ]
            },
            {"default_owner": "Construction Manager"},
        )
        assert result["status"] == "success"
        assert result["action"] == "risk_register"
        assert result["total_risks"] == 2
        assert result["high_risks"] == 1
        assert all(r["id"].startswith("RISK-") for r in result["risk_register"])
        assert result["risk_register"][0]["risk_score"] >= result["risk_register"][-1]["risk_score"]


class TestSubmittalLogGenerator:
    @pytest.mark.asyncio
    async def test_submittal_log_generator(self, container):
        result = await container.submittal_log_generator(
            {
                "spec_sections": [{"value": "Concrete mix design"}, {"value": "Waterproofing membrane"}],
                "boq": [
                    {"description": "Structural steel beams", "quantity": 50, "unit": "t"},
                    {"description": "PVC pipe", "quantity": 200, "unit": "lm"},
                ],
                "project_name": "Test Tower",
                "contract_start_date": "2026-03-01",
            },
            {},
        )
        assert result["status"] == "success"
        assert result["action"] == "submittal_log"
        assert result["project"] == "Test Tower"
        assert result["total_submittals"] > 0
        assert "Material Submittal" in result["by_type"]
        assert "Shop Drawing" in result["by_type"]
        # Ref numbers must be deterministic for the same description.
        refs = [s["ref"] for s in result["submittal_register"]]
        assert len(refs) == len(set(refs))


class TestRFIGenerator:
    @pytest.mark.asyncio
    async def test_rfi_generator(self, container):
        result = await container.rfi_generator(
            {
                "issues": [
                    {"description": "Column grid mismatch at A-1", "type": "structural", "severity": "high"},
                    {"description": "Missing ceiling finish spec", "type": "specification", "severity": "medium"},
                ],
                "project_name": "Test Tower",
            },
            {"contractor_name": "Test Contractor", "engineer_name": "Test Engineer"},
        )
        assert result["status"] == "success"
        assert result["action"] == "rfi_generator"
        assert result["total_rfis"] == 2
        assert result["rfis"][0]["rfi_number"] == "RFI-0001"
        assert "Structural" in result["rfis"][0]["discipline"]
        assert result["rfis"][0]["priority"] == "high"

    @pytest.mark.asyncio
    async def test_rfi_generator_no_issues(self, container):
        result = await container.rfi_generator({}, {})
        assert result["status"] == "error"
        assert result["rfis"] == []
        assert "issues" in result["error"].lower()


class TestChangeOrderImpact:
    @pytest.mark.asyncio
    async def test_change_order_impact(self, container):
        result = await container.change_order_impact(
            {"change_type": "additional excavation work", "direct_cost": 50_000},
            {},
        )
        assert result["status"] == "success"
        assert result["action"] == "change_order_analysis"
        assert result["change_type"] == "additional excavation work"
        assert result["category"] == "scope_addition"
        assert result["cost_impact"]["total"] > 50_000
        assert result["schedule_impact_days"] >= 0


class TestValueEngineering:
    @pytest.mark.asyncio
    async def test_value_engineering(self, container):
        boq = [
            {"id": "C-01", "material_type": "concrete_c30", "quantity": 1000, "total_cost": 150_000, "carbon_impact": 25_000},
            {"id": "S-01", "material_type": "structural_steel", "quantity": 500, "total_cost": 250_000, "carbon_impact": 40_000},
        ]
        result = await container.value_engineering(
            {"boq": boq},
            {"target_reduction": 0.15, "carbon_priority": False},
        )
        assert result["status"] == "success"
        assert result["action"] == "value_engineering_analysis"
        assert result["alternatives_identified"] > 0
        assert "scenarios" in result
        assert "recommended_scenario" in result
        assert result["impact_summary"]["cost_savings"] >= 0


class TestTenderBidAnalysis:
    @pytest.mark.asyncio
    async def test_tender_bid_analysis_requires_two_bids(self, container):
        # <2 bids must NEVER fabricate bidders or an award recommendation — the
        # previous behaviour injected 3 fictitious contractors. Honest error only.
        for payload in (
            {},
            {"bids": []},
            {"bids": [{"contractor_name": "A", "total_price": 1_000_000}]},
        ):
            result = await container.tender_bid_analysis(payload, {})
            assert result["status"] == "error", payload
            assert "at least two" in result["error"].lower()
            assert "recommendation" not in result
            assert "ranking" not in result

    @pytest.mark.asyncio
    async def test_tender_bid_analysis_custom_bids(self, container):
        bids = [
            {"contractor_name": "A", "total_price": 1_000_000, "duration_days": 300, "experience_score": 90, "financial_stability": 95, "safety_rating": 92, "quality_score": 88},
            {"contractor_name": "B", "total_price": 950_000, "duration_days": 320, "experience_score": 80, "financial_stability": 85, "safety_rating": 80, "quality_score": 82},
        ]
        result = await container.tender_bid_analysis({"bids": bids}, {})
        assert result["status"] == "success"
        assert result["bids_received"] == 2
        # A has higher weighted score (better non-price factors) despite higher price.
        assert result["ranking"]["first"]["contractor"] == "A"
        assert result["price_analysis"]["lowest_bid"] == 950_000


class TestVariationOrderManager:
    @pytest.mark.asyncio
    async def test_variation_order_manager(self, container):
        result = await container.variation_order_manager(
            {
                "variation_data": {
                    "vo_number": "VO-001",
                    "description": "Additional blockwork to plant room",
                    "type": "addition",
                    "direct_cost": 20_000,
                    "quantity": 1,
                    "schedule_impact_days": 5,
                    "critical_path": True,
                },
                "existing_vos": [{"total": 50_000}],
                "contract_value": 1_000_000,
            },
            {},
        )
        assert result["status"] == "success"
        assert result["action"] == "variation_order_processed"
        assert result["vo_number"] == "VO-001"
        assert result["category"] == "scope_addition"
        assert result["cumulative_impact"]["cumulative_value"] == 50_000.0 + result["pricing"]["total_value"]
        assert result["approval_workflow"]["level"] in {"project_manager", "contracts_manager", "director", "board_client"}


class TestProcurementOptimizer:
    @pytest.mark.asyncio
    async def test_procurement_optimizer(self, container):
        suppliers = [
            {
                "name": "Steel-R-Us",
                "price_score": 85,
                "delivery_score": 80,
                "quality_score": 90,
                "financial_score": 95,
                "esg_score": 70,
                "support_score": 75,
                "lead_time": 8,
                "capabilities": ["structural_steel"],
            },
            {
                "name": "Conco",
                "price_score": 90,
                "delivery_score": 85,
                "quality_score": 85,
                "lead_time": 4,
                "capabilities": ["concrete"],
            },
        ]
        boq = [
            {"id": "ST-01", "material_type": "structural_steel", "quantity": 100, "unit": "t", "required_date": "2026-06-01", "quality_critical": True},
            {"id": "CO-01", "material_type": "concrete", "quantity": 500, "unit": "m3", "required_date": "2026-05-01"},
        ]
        result = await container.procurement_optimizer(
            {"boq": boq, "suppliers": suppliers},
            {"constraints": {"max_suppliers": 5}},
        )
        assert result["status"] == "success"
        assert result["action"] == "procurement_optimization"
        assert result["suppliers_evaluated"] == 2
        assert len(result["procurement_plan"]["plan"]) == 2
        assert result["procurement_plan"]["critical_path_items"] == 1


class TestESGSustainabilityReport:
    @pytest.mark.asyncio
    async def test_esg_sustainability_report(self, container):
        result = await container.esg_sustainability_report(
            {
                "project_data": {"contract_value": 5_000_000},
                "boq": [
                    {"material_type": "concrete", "quantity": 1000, "unit": "m3", "total_cost": 150_000},
                    {"material_type": "steel", "quantity": 5000, "unit": "kg", "total_cost": 50_000},
                ],
                "manpower": {"total_workers": 50},
                "safety_records": [],
            },
            {"period": "annual"},
        )
        assert result["status"] == "success"
        assert result["action"] == "esg_sustainability_report"
        assert result["reporting_period"] == "annual"
        assert "esg_scores" in result
        assert "environmental" in result
        assert "social" in result
        assert "governance" in result
        assert result["esg_scores"]["rating"] in {"A", "B", "C", "D"}


class TestProcurementListGenerator:
    @requires_construction_kit
    @pytest.mark.asyncio
    async def test_from_boq_with_rates(self, container):
        boq = [
            {"item": "Concrete", "quantity": 100, "unit": "m3", "adjusted_rate": 150.0, "total": 15_000},
            {"item": "Structural steel", "quantity": 50, "unit": "t", "adjusted_rate": 2500.0, "total": 125_000},
        ]
        result = await container.procurement_list_generator({"boq": boq}, {"budget": 200_000})
        assert result["status"] == "success"
        assert result["action"] == "procurement_list"
        assert result["total_items"] == 2
        assert result["total_procurement_cost"] == 140_000.0
        assert result["budget_variance"] == 60_000.0
        assert any(i["category"] == "Concrete" for i in result["procurement_list"])

    @requires_construction_kit
    @pytest.mark.asyncio
    async def test_from_quantities_uses_benchmark(self, container):
        result = await container.procurement_list_generator(
            {
                "quantities": {
                    "concrete": {"quantity": 50, "unit": "m3"},
                    "steel": {"quantity": 1000, "unit": "kg"},
                }
            },
            {"location": "US National Average", "project_type": "general_building"},
        )
        assert result["status"] == "success"
        assert result["total_items"] == 2
        assert result["total_procurement_cost"] > 0
        assert result["procurement_list"][0]["lead_time_weeks"] >= result["procurement_list"][-1]["lead_time_weeks"]


class TestProcurementAnalysis:
    @requires_construction_kit
    @pytest.mark.asyncio
    async def test_procurement_analysis(self, container):
        suppliers = [
            {
                "name": "Conco",
                "price_score": 90,
                "delivery_score": 85,
                "quality_score": 85,
                "lead_time": 4,
                "capabilities": ["concrete"],
            }
        ]
        boq = [
            {"id": "CO-01", "material_type": "concrete", "quantity": 500, "unit": "m3", "required_date": "2026-05-01"},
        ]
        result = await container.procurement_analysis(
            {"boq": boq, "suppliers": suppliers},
            {},
        )
        assert result["status"] == "success"
        assert result["action"] == "procurement_analysis"
        assert result["procurement_list"]["status"] == "success"
        assert result["optimization"]["status"] == "success"


class TestAsBuiltDeviationHonestGate:
    """F1 — as_built must never return APPROVED without a real comparison
    (was a hardcoded 'Column grid A-1, 18mm' deviation / a false 0-deviation APPROVED)."""

    @pytest.mark.asyncio
    async def test_no_inputs_errors_not_approved(self, container):
        result = await container.as_built_deviation_report({}, {})
        assert result["status"] == "error"
        assert "sign_off_status" not in result
        assert "deviation_summary" not in result

    @pytest.mark.asyncio
    async def test_measurements_path_real_compare(self, container):
        result = await container.as_built_deviation_report(
            {"measurements": [{"type": "length", "value": 3.6, "unit": "m"}]},
            {"design_measurements": [{"type": "length", "value": 3.618, "unit": "m"}]},
        )
        assert result["status"] == "success"
        assert result["deviation_summary"]["total_deviations"] >= 1


    @pytest.mark.asyncio
    async def test_file_path_real_extraction_and_compare(self, container, monkeypatch):
        # File path: _compare_as_built_to_design extracts via drawing_qto and
        # compares like-typed dimensions. Monkeypatch drawing_qto so the full
        # extraction->compare path runs deterministically (no real DXF/CAD).
        class _FakeQTO:
            async def process(self, data):
                if "asbuilt" in (data.get("file_path") or ""):
                    return {"measurements": [{"type": "length", "value": 3.618, "unit": "m"}]}
                return {"measurements": [{"type": "length", "value": 3.6, "unit": "m"}]}
        monkeypatch.setattr(container, "_resolve_block",
                            lambda name: _FakeQTO() if name == "drawing_qto" else None)
        result = await container.as_built_deviation_report(
            {"as_built_file": "asbuilt.dxf", "design_file": "design.dxf"}, {})
        assert result["status"] == "success"
        # 18mm diff > 10mm tolerance -> a real, source-derived deviation
        assert result["deviation_summary"]["total_deviations"] >= 1
        # never the old hardcoded stub
        assert "column grid a-1" not in str(result).lower()

    @pytest.mark.asyncio
    async def test_file_path_no_extractable_dimensions_errors(self, container, monkeypatch):
        class _EmptyQTO:
            async def process(self, data):
                return {"measurements": []}
        monkeypatch.setattr(container, "_resolve_block",
                            lambda name: _EmptyQTO() if name == "drawing_qto" else None)
        result = await container.as_built_deviation_report(
            {"as_built_file": "a.dxf", "design_file": "b.dxf"}, {})
        assert result["status"] == "error"
        assert "comparable dimensions" in result["error"].lower()
        assert "sign_off_status" not in result

    @pytest.mark.asyncio
    async def test_file_path_no_qto_block_errors(self, container, monkeypatch):
        # drawing_qto unavailable (e.g. kit not loaded) -> None -> honest error.
        monkeypatch.setattr(container, "_resolve_block", lambda name: None)
        result = await container.as_built_deviation_report(
            {"as_built_file": "a.dxf", "design_file": "b.dxf"}, {})
        assert result["status"] == "error"
        assert "sign_off_status" not in result

    @pytest.mark.asyncio
    async def test_measurements_without_design_errors(self, container):
        result = await container.as_built_deviation_report(
            {"measurements": [{"type": "length", "value": 3.6, "unit": "m"}]}, {})
        assert result["status"] == "error"
        assert "design_measurements" in result["error"]


    @pytest.mark.asyncio
    async def test_file_path_qto_raises_errors(self, container, monkeypatch):
        class _BoomQTO:
            async def process(self, data):
                raise RuntimeError("qto boom")
        monkeypatch.setattr(container, "_resolve_block", lambda name: _BoomQTO())
        result = await container.as_built_deviation_report(
            {"as_built_file": "a.dxf", "design_file": "b.dxf"}, {})
        assert result["status"] == "error"
        assert "sign_off_status" not in result

    @pytest.mark.asyncio
    async def test_file_path_resolve_raises_errors(self, container, monkeypatch):
        def _boom(name):
            raise RuntimeError("resolve boom")
        monkeypatch.setattr(container, "_resolve_block", _boom)
        result = await container.as_built_deviation_report(
            {"as_built_file": "a.dxf", "design_file": "b.dxf"}, {})
        assert result["status"] == "error"
        assert "sign_off_status" not in result


class TestEsgAvailabilityGatedScoring:
    """V2 — esg must not invent a rating for pillars with no supplied data.
    Social/governance metrics are placeholder zeros; scoring them anyway fabricated
    a composite + letter grade from nothing (social 80 for 'ltifr==0'=no-data,
    governance a flat 70)."""

    _BOQ = [{"description": "Concrete", "total_cost": 1_000_000, "quantity": 500, "unit": "m3"}]

    @pytest.mark.asyncio
    async def test_no_data_yields_no_score_no_rating(self, container):
        r = await container.esg_sustainability_report({}, {})
        s = r["esg_scores"]
        assert s["overall"] is None
        assert s["rating"] is None
        assert s["social"] is None and s["governance"] is None
        assert set(s["data_status"]["pillars_missing_data"]) == {
            "environmental", "social", "governance"}
        assert s["data_status"]["complete"] is False

    @pytest.mark.asyncio
    async def test_env_only_is_honest_partial(self, container):
        r = await container.esg_sustainability_report({"boq": self._BOQ}, {})
        s = r["esg_scores"]
        assert s["environmental"] is not None
        assert s["social"] is None and s["governance"] is None
        # Overall reflects the single assessed pillar; rating is a labelled partial.
        assert s["overall"] == s["environmental"]
        assert s["rating"].startswith("partial")
        assert "environmental" in s["rating"]

    @pytest.mark.asyncio
    async def test_all_pillars_present_gives_full_letter_rating(self, container):
        r = await container.esg_sustainability_report({
            "boq": self._BOQ,
            "manpower": {"total_workers": 200, "local_percent": 85},
            "safety_records": [{"type": "near_miss"}],
            "project_data": {"anti_corruption": True, "ethics_training": 95},
        }, {})
        s = r["esg_scores"]
        assert s["data_status"]["complete"] is True
        assert all(s[p] is not None for p in ("environmental", "social", "governance"))
        assert s["rating"] in ("A", "B", "C", "D")
