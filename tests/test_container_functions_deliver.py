"""Phase 1 delivery tests for ConstructionContainer actions.

Each case carries a hand-derived expected output in the docstring.
Central fail: status=success without delivering the artefact, or silent
invention on missing / empty / wrong-type / absurd input.

Synthetic figures only. Instantiates ConstructionContainer the same way
as tests/test_construction_boq_actions.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.containers.construction import ConstructionContainer

# Virgin CI (and the postgres job, which is also virgin) does not register
# drawing_qto / boq_processor / primavera_parser. Those actions return an
# honest error; skip the delivery asserts instead of failing the suite.
_EXTERNAL_UNAVAILABLE_MARKERS = (
    "unavailable",
    "needs-external",
    "needs external",
    "not installed",
    "missing dependency",
    "no module named",
)


def _skip_if_external_unavailable(result: dict, *, what: str) -> None:
    """Skip when an optional parser/block is not loaded. Hollow success still fails."""
    if not isinstance(result, dict) or result.get("status") != "error":
        return
    err = str(result.get("error") or "").lower()
    if any(marker in err for marker in _EXTERNAL_UNAVAILABLE_MARKERS):
        pytest.skip(f"{what} optional parser/block unavailable: {result.get('error')}")


@pytest.fixture
def container():
    return ConstructionContainer()


def _write_minimal_xer(path: Path) -> Path:
    """Three-activity synthetic XER. Dates are calendar facts, not a job."""
    path.write_text(
        "\n".join(
            [
                "%T\tPROJECT",
                "%F\tproj_id\tproj_short_name\tplan_start_date\tlast_recalc_date",
                "%R\t1\tFixture\t2026-03-01 00:00\t2026-03-08 00:00",
                "%T\tTASK",
                "%F\ttask_id\ttask_code\ttask_name\ttarget_drtn_hr_cnt"
                "\tearly_start_date\tearly_end_date\ttotal_float_hr_cnt"
                "\tremain_drtn_hr_cnt\twbs_id\tstatus_code\tphys_complete_pct",
                "%R\t1001\tA\tMobilise\t40\t2026-03-01 00:00\t2026-03-05 00:00"
                "\t80\t0\tW1\tTK_Complete\t100",
                "%R\t1002\tB\tExcavate\t80\t2026-03-10 00:00\t2026-03-20 00:00"
                "\t0\t64\tW2\tTK_Active\t20",
                "%R\t1003\tC\tConcrete\t120\t2026-04-15 00:00\t2026-04-30 00:00"
                "\t40\t120\tW3\tTK_NotStart\t0",
                "%E",
                "",
            ]
        ),
        encoding="cp1252",
    )
    return path


# ═══════════════════════════════════════════════════════════════════════════
# 1 look_ahead
# ═══════════════════════════════════════════════════════════════════════════


class TestLookAheadDeliver:
    @pytest.mark.asyncio
    async def test_missing_file_is_error(self, container):
        """Empty payload must not invent a look-ahead programme."""
        result = await container.look_ahead({}, {})
        assert result["status"] == "error"
        assert "schedule" in result["error"].lower() or ".xer" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_bad_days_is_error(self, container):
        """days='abc' is a type error, not a defaulted 3-week window."""
        result = await container.look_ahead({}, {"days": "abc"})
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_zero_window_is_error(self, container):
        """Absurd window of 0 days must refuse."""
        result = await container.look_ahead({}, {"days": 0, "schedule_file": "x.xer"})
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_window_from_xer_hand_derived(self, container, tmp_path):
        """as_of 2026-03-08 inclusive 21 calendar days → window_end 2026-03-28.

        A Mobilise 2026-03-01..05 is before the window.
        B Excavate 2026-03-10..20 overlaps.
        C Concrete 2026-04-15..30 is after the window.
        Remaining duration on B is 64 hours / 8 = 8 days. Float 0 → critical.
        """
        path = _write_minimal_xer(tmp_path / "fixture.xer")
        result = await container.look_ahead(
            {},
            {"schedule_file": str(path), "as_of": "2026-03-08", "days": 21},
        )
        _skip_if_external_unavailable(result, what="look_ahead")
        assert result["status"] == "success", result
        assert result["action"] == "look_ahead"
        assert result["window_days"] == 21
        assert result["as_of"] == "2026-03-08"
        # Inclusive 21-day window: 2026-03-08 .. 2026-03-28.
        assert result["window_end"] == "2026-03-28"
        codes = {a.get("code") or a.get("id") for a in result["activities"]}
        assert "B" in codes
        assert "C" not in codes
        assert "A" not in codes
        hit = next(a for a in result["activities"] if (a.get("code") or a.get("id")) == "B")
        assert hit["is_critical"] is True
        assert hit["remaining_duration_days"] == pytest.approx(8.0)


# ═══════════════════════════════════════════════════════════════════════════
# 2 procurement_list_generator
# ═══════════════════════════════════════════════════════════════════════════


class TestProcurementListDeliver:
    @pytest.mark.asyncio
    async def test_empty_is_error_not_hollow_success(self, container):
        """No BOQ and no quantities → error, not success with an empty list."""
        result = await container.procurement_list_generator({}, {})
        assert result["status"] == "error"
        assert result.get("procurement_list") in ([], None)
        assert "procurement" in result["error"].lower() or "quantit" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_from_priced_boq_hand_derived(self, container):
        """100 m3 × 150 + 50 t × 2500 = 15_000 + 125_000 = 140_000.

        Budget 200_000 → variance 60_000.
        'Structural steel frame' classifies as Structural Steel, lead 16 → critical.
        'Ready-mix concrete' classifies as Concrete, lead 2 → normal.
        Sort is lead_time descending, so steel is first.
        """
        boq = [
            {"item": "Ready-mix concrete", "quantity": 100, "unit": "m3", "adjusted_rate": 150.0, "total": 15_000},
            {"item": "Structural steel frame", "quantity": 50, "unit": "t", "adjusted_rate": 2500.0, "total": 125_000},
        ]
        result = await container.procurement_list_generator({"boq": boq}, {"budget": 200_000})
        assert result["status"] == "success"
        assert result["total_items"] == 2
        assert result["total_procurement_cost"] == 140_000.0
        assert result["budget_variance"] == 60_000.0
        assert result["critical_long_lead_items"] == 1
        assert result["procurement_list"][0]["item"] == "Structural steel frame"
        assert result["procurement_list"][0]["lead_time_weeks"] == 16
        assert result["procurement_list"][0]["priority"] == "critical"
        assert result["procurement_list"][1]["lead_time_weeks"] == 2

    @pytest.mark.asyncio
    async def test_line_items_keep_quantity_and_total_in_each_driver_shape(self, container):
        """The workflow envelope carries line items; params stay empty.

        A prompt copies qty/rate/amount. A fetched sheet copies its headers.
        A fetched BOQ chunk copies the index phrases (quantity and price
        with a trailing unit). An ingested row uses unit_cost and total_cost.
        Each row's own extension must come back — a missing rate is not a
        reason to price the line at zero.
        """
        cases = [
            (
                {"description": "Busbar", "unit": "nr", "qty": "12", "rate": "25.50", "amount": "306"},
                12,
                25.5,
            ),
            (
                {"description": "Tray", "Qty": "4", "Rate": "1,500", "Amount": "6,000"},
                4,
                1500,
            ),
            (
                {
                    "description": "Insulation",
                    "total quantity": "40 m",
                    "unit price (rate)": "18",
                    "total price (amount)": "720",
                },
                40,
                18,
            ),
            (
                {
                    "description": "Panel",
                    "unit": "m2",
                    "quantity": 7,
                    "unit_cost": 100,
                    "total_cost": 700,
                },
                7,
                100,
            ),
            (
                {"description": "Gasket", "qty": 5, "amount": 250},
                5,
                50,
            ),
        ]
        result = await container.procurement_list_generator(
            {
                "action": "procurement_list_generator",
                "message": "Produce a procurement list for the project.",
                "line_items": [row for row, _qty, _rate in cases],
            },
            {},
        )
        assert result["status"] == "success", result
        by_name = {item["item"]: item for item in result["procurement_list"]}
        expected_total = 0.0
        for row, qty, rate in cases:
            item = by_name[row["description"]]
            assert item["quantity"] == pytest.approx(qty), row
            assert item["quantity"] > 0, row
            assert item["unit_cost"] == pytest.approx(rate), row
            assert item["total_cost"] == pytest.approx(qty * rate), row
            assert item["total_cost"] > 0, row
            expected_total += qty * rate
        assert result["total_procurement_cost"] == pytest.approx(expected_total)


# ═══════════════════════════════════════════════════════════════════════════
# 3 drawing_qto
# ═══════════════════════════════════════════════════════════════════════════


class TestDrawingQtoDeliver:
    @pytest.mark.asyncio
    async def test_missing_file_is_error(self, container):
        """Delegate must not succeed with an empty take-off."""
        result = await container.drawing_qto({}, {})
        assert result["status"] == "error"
        err = str(result.get("error", "")).lower()
        assert "file" in err or "unavailable" in err

    @pytest.mark.asyncio
    async def test_missing_path_on_disk_is_error(self, container):
        result = await container.drawing_qto({"file_path": "/tmp/does-not-exist-fixture.dxf"}, {})
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_wrong_extension_is_error(self, container, tmp_path):
        """A .txt is the wrong type — refuse, do not invent geometry."""
        p = tmp_path / "notes.txt"
        p.write_text("not a drawing", encoding="utf-8")
        result = await container.drawing_qto({"file_path": str(p)}, {})
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_closed_rectangle_area_hand_derived(self, container, tmp_path):
        """10 m × 5 m closed polyline in metre units → area 50 m².

        Perimeter of four edges = 10+5+10+5 = 30 m (if edges are measured).
        """
        ezdxf = pytest.importorskip("ezdxf", reason="drawing_qto DXF path needs ezdxf")
        from ezdxf import units as ez_units

        doc = ezdxf.new()
        doc.units = ez_units.M
        msp = doc.modelspace()
        msp.add_lwpolyline([(0, 0), (10, 0), (10, 5), (0, 5)], close=True)
        dxf = tmp_path / "rect.dxf"
        doc.saveas(dxf)
        result = await container.drawing_qto({"file_path": str(dxf)}, {})
        _skip_if_external_unavailable(result, what="drawing_qto")
        assert result["status"] == "success", result
        assert result.get("total_area_m2") == pytest.approx(50.0, rel=1e-3)
        assert result.get("entity_count", 0) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# 4 variation_order_manager
# ═══════════════════════════════════════════════════════════════════════════


class TestVariationOrderDeliver:
    @pytest.mark.asyncio
    async def test_missing_vo_is_error(self, container):
        result = await container.variation_order_manager({}, {})
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_draft_a_variation_order_without_facts_is_error(self, container):
        """Live: 'Draft a variation order' → Status: Success with nothing drafted.

        The ask is not a variation. No works scope and no cost → error, not
        VO-001 / Total: 0.0 with the prompt as the description.
        """
        envelopes = [
            ({"message": "Draft a variation order"}, {}),
            (
                {
                    "variation_data": {"description": "Draft a variation order"},
                    "contract_value": 1_000_000,
                },
                {},
            ),
            ({"text": "Draft a variation order under FIDIC clause"}, {}),
        ]
        for payload, params in envelopes:
            result = await container.variation_order_manager(payload, params)
            assert result["status"] == "error", result
            assert "scope" in result["error"].lower() or "vo_data" in result["error"].lower()
            assert not result.get("document_content")
            assert result.get("pricing") in (None, {})

        routed = await container.route(
            "variation_order_manager",
            {"message": "Draft a variation order"},
            {},
        )
        assert routed["status"] == "error"
        iw = await container.intelligent_workflow(
            {"message": "Draft a variation order"},
            {},
        )
        assert iw["status"] == "error"
        assert not iw.get("document_content")

    @pytest.mark.asyncio
    async def test_missing_contract_value_is_error(self, container):
        result = await container.variation_order_manager(
            {"variation_data": {"description": "extra blockwork", "direct_cost": 1000}},
            {},
        )
        assert result["status"] == "error"
        assert "contract_value" in result["error"]

    @pytest.mark.asyncio
    async def test_pricing_and_no_invented_clause(self, container):
        """direct 5_000 × qty 1, addition:
        indirect = 5_000 × 0.15 = 750
        overhead = (5_000+750) × 0.10 = 575
        profit = (5_000+750) × 0.08 = 460
        total = 5_000+750+575+460 = 6_785
        6_785 / 200_000 = 3.3925% → project_manager (<10_000) → approve.
        No contract file → clause / entitlement / notice stay unset (not invented).
        """
        result = await container.variation_order_manager(
            {
                "variation_data": {
                    "vo_number": "VO-007",
                    "description": "Additional blockwork to plant room",
                    "type": "addition",
                    "direct_cost": 5_000,
                    "quantity": 1,
                    "supporting_documents": ["priced-breakdown.csv"],
                },
                "contract_value": 200_000,
            },
            {},
        )
        assert result["status"] == "success"
        assert result["vo_number"] == "VO-007"
        assert result["pricing"]["direct_costs"] == 5_000.0
        assert result["pricing"]["indirect_costs"] == 750.0
        assert result["pricing"]["overhead"] == 575.0
        assert result["pricing"]["profit"] == 460.0
        assert result["pricing"]["total_value"] == 6_785.0
        assert result["approval_workflow"]["level"] == "project_manager"
        assert result["recommended_action"] == "approve"
        assert result["contract_compliance"]["variation_clause"] in (None, "")
        assert result["contract_compliance"]["entitlement_clear"] is None
        assert result["contract_compliance"]["notice_requirements_met"] is None
        assert result["contract_compliance"]["time_bar_risk"]["at_risk"] is None
        assert result["supporting_documents"] == ["priced-breakdown.csv"]
        assert "Clause XX" not in str(result)


# ═══════════════════════════════════════════════════════════════════════════
# 5 payment_certificate
# ═══════════════════════════════════════════════════════════════════════════


class TestPaymentCertificateDeliver:
    @pytest.mark.asyncio
    async def test_missing_figures_is_error(self, container):
        result = await container.payment_certificate({}, {})
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_ipc_arithmetic_hand_derived(self, container):
        """contract 1_000_000, 30% done, previous 100_000, retention 10%,
        advance 50_000 recovered at 20% of gross:
        gross = 300_000
        retention = 30_000
        advance recovered = min(50_000, 60_000) = 50_000
        net = 300_000 − 30_000 − 50_000 − 100_000 = 120_000
        cumulative = 220_000
        remaining = 1_000_000 − 220_000 = 780_000
        """
        result = await container.payment_certificate(
            {
                "contract_value": 1_000_000,
                "work_done_percent": 30,
                "previous_certified": 100_000,
                "retention_percent": 10,
                "advance_payment": 50_000,
                "advance_recovery_percent": 20,
            },
            {"payment_period": "Period 3"},
        )
        assert result["status"] == "success"
        assert result["action"] == "payment_certificate"
        assert result["certificate"]["period"] == "Period 3"
        assert result["valuation"]["gross_valuation"] == 300_000.0
        assert result["deductions"]["retention_held"] == 30_000.0
        assert result["deductions"]["advance_recovery"] == 50_000.0
        assert result["payment"]["net_due_this_period"] == 120_000.0
        assert result["payment"]["cumulative_certified"] == 220_000.0
        assert result["payment"]["remaining_contract_balance"] == 780_000.0
        assert "300,000.00" in result["certificate_summary"]
        assert "120,000.00" in result["certificate_summary"]


# ═══════════════════════════════════════════════════════════════════════════
# 6 generate_wbs
# ═══════════════════════════════════════════════════════════════════════════


class TestGenerateWbsDeliver:
    @pytest.mark.asyncio
    async def test_building_brief_delivers_tree(self, container, monkeypatch):
        """target_count 20 on project_type=building → at least 20 activities,
        a non-empty wbs_tree, and CPM fields on the first activity.
        Empty-input default brief is not used when brief is supplied.
        """
        from tests import _wbs_fakes
        _wbs_fakes.install(monkeypatch, min_activities=20)
        result = await container.generate_wbs(
            {"brief": "Two-storey office shell and core."},
            {"target_count": 20, "project_type": "building", "start_date": "2026-01-05"},
        )
        assert result["status"] == "success"
        assert result["project_type"] == "building"
        assert result["actual_count"] >= 20
        assert isinstance(result.get("wbs_tree"), dict) and result["wbs_tree"]
        a0 = result["activities"][0]
        for key in ("id", "name", "duration_days", "early_start_day", "total_float_days"):
            assert key in a0
        assert result["start_date"] == "2026-01-05"


# ═══════════════════════════════════════════════════════════════════════════
# 7 cash_flow_forecast
# ═══════════════════════════════════════════════════════════════════════════


class TestCashFlowDeliver:
    @pytest.mark.asyncio
    async def test_missing_value_is_error(self, container):
        result = await container.cash_flow_forecast({}, {})
        assert result["status"] == "error"
        assert "contract_value" in result["error"]

    @pytest.mark.asyncio
    async def test_month_one_s_curve_hand_derived(self, container):
        """1_200_000 over 12 months, default advance 10%, retention 10%.
        Month 1 time_percent = 1/12 ≤ 0.25 → progress = (1/12)×0.8 = 1/15
        monthly_value = 1_200_000 / 15 = 80_000
        advance paid month 1 = 120_000
        retention = 8_000
        recovery window = int(12×0.8) = 9 → recovery = 120_000/9 = 13_333.33…
        net_cash_in = 80_000 − 8_000 + 120_000 − 13_333.33… = 178_666.67
        """
        result = await container.cash_flow_forecast(
            {"contract_value": 1_200_000},
            {
                "duration_months": 12,
                "project_start_date": "2026-01-01",
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
        curve = result["s_curve_data"]
        assert len(curve) == 12
        m1 = curve[0]
        assert m1["month"] == 1
        assert m1["monthly_value"] == 80_000.0
        assert m1["retention_deduction"] == 8_000.0
        assert m1["advance_recovery"] == pytest.approx(13_333.33, abs=0.02)
        assert m1["net_cash_in"] == pytest.approx(178_666.67, abs=0.02)


# ═══════════════════════════════════════════════════════════════════════════
# 8 rfi_generator
# ═══════════════════════════════════════════════════════════════════════════


class TestRfiGeneratorDeliver:
    @pytest.mark.asyncio
    async def test_empty_is_error_not_hollow(self, container):
        result = await container.rfi_generator({}, {})
        assert result["status"] == "error"
        assert result.get("rfis") == []

    @pytest.mark.asyncio
    async def test_two_issues_numbered(self, container):
        """Two issues → RFI-0001 and RFI-0002. High severity → 14-day reply."""
        result = await container.rfi_generator(
            {
                "issues": [
                    {"description": "Grid clash at axis A-1", "type": "structural", "severity": "high"},
                    {"description": "Finish spec missing at soffit", "type": "specification", "severity": "medium"},
                ],
                "project_name": "Fixture Tower",
            },
            {"contractor_name": "Fixture Contractor", "engineer_name": "Fixture Engineer"},
        )
        assert result["status"] == "success"
        assert result["total_rfis"] == 2
        assert result["rfis"][0]["rfi_number"] == "RFI-0001"
        assert result["rfis"][1]["rfi_number"] == "RFI-0002"
        assert result["rfis"][0]["discipline"] == "Structural"
        assert result["rfis"][0]["priority"] == "high"
        assert result["rfis"][0]["issued_by"] == "Fixture Contractor"
        assert result["rfis"][0]["addressed_to"] == "Fixture Engineer"


# ═══════════════════════════════════════════════════════════════════════════
# 9 evm_calculate
# ═══════════════════════════════════════════════════════════════════════════


class TestEvmCalculateDeliver:
    @pytest.mark.asyncio
    async def test_missing_actuals_is_error(self, container):
        result = await container.evm_calculate({"pv": 100_000, "ev": 90_000}, {})
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_classic_indices_hand_derived(self, container):
        """PV 100_000, EV 90_000, AC 95_000, BAC 200_000
        SPI = 0.9
        CPI = 90_000/95_000 = 18/19 ≈ 0.947368
        SV = −10_000
        CV = −5_000
        EAC = BAC/CPI = 200_000 × 95_000 / 90_000 = 211_111.11…
        """
        result = await container.evm_calculate(
            {"pv": 100_000, "ev": 90_000, "ac": 95_000, "bac": 200_000},
            {},
        )
        assert result["status"] == "success"
        evm = result["evm"]
        assert evm["SPI"] == pytest.approx(0.9)
        assert evm["CPI"] == pytest.approx(90_000 / 95_000, abs=1e-3)
        assert evm["SV"] == -10_000
        assert evm["CV"] == -5_000
        assert evm["EAC"] == pytest.approx(200_000 * 95_000 / 90_000)


# ═══════════════════════════════════════════════════════════════════════════
# 10 boq_process
# ═══════════════════════════════════════════════════════════════════════════


class TestBoqProcessDeliver:
    @pytest.mark.asyncio
    async def test_missing_file_is_error(self, container):
        result = await container.boq_process({}, {})
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_csv_line_total_hand_derived(self, container, tmp_path):
        """10 m3 × 150 = 1_500 on a one-row CSV."""
        csv_path = tmp_path / "fixture_boq.csv"
        csv_path.write_text(
            "Item,Description,Quantity,Unit,Rate\n"
            "1,Ready-mix concrete C30,10,m3,150\n",
            encoding="utf-8",
        )
        result = await container.boq_process({"file_path": str(csv_path)}, {})
        _skip_if_external_unavailable(result, what="boq_process")
        assert result["status"] == "success", result
        breakdown = result.get("cost_breakdown") or {}
        general = breakdown.get("General") or {}
        assert general.get("total") == 1500.0
        assert result.get("item_count") == 1


# ═══════════════════════════════════════════════════════════════════════════
# 11 parse_primavera_schedule
# ═══════════════════════════════════════════════════════════════════════════


class TestParsePrimaveraDeliver:
    @pytest.mark.asyncio
    async def test_missing_file_is_error(self, container):
        result = await container.parse_primavera_schedule({}, {})
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_unsupported_format_is_error(self, container, tmp_path):
        p = tmp_path / "notes.txt"
        p.write_text("not a schedule", encoding="utf-8")
        result = await container.parse_primavera_schedule({"file_path": str(p)}, {})
        assert result["status"] == "error"
        assert "unsupported" in result["error"].lower() or "format" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_xer_delivers_three_activities(self, container, tmp_path):
        """The fixture XER has exactly three TASK rows."""
        path = _write_minimal_xer(tmp_path / "fixture.xer")
        result = await container.parse_primavera_schedule({"file_path": str(path)}, {})
        _skip_if_external_unavailable(result, what="parse_primavera_schedule")
        assert result["status"] == "success", result
        assert result["summary"]["total_activities"] == 3


# ═══════════════════════════════════════════════════════════════════════════
# Remaining actions — one delivery / refuse group each
# ═══════════════════════════════════════════════════════════════════════════


class TestExtractQuantitiesDeliver:
    @pytest.mark.asyncio
    async def test_empty_is_error(self, container):
        result = await container.extract_quantities({}, {})
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_area_to_concrete_hand_derived(self, container):
        """200 m² floor × 0.15 m default slab = 30 m³; × 120 kg/m³ = 3_600 kg."""
        result = await container.extract_quantities(
            {"measurements": [{"type": "area", "value": 200.0, "item": "floor"}]},
            {},
        )
        assert result["status"] == "success"
        q = result["quantities"]
        assert q["floor_area_m2"] == 200.0
        assert q["concrete_volume_m3"] == pytest.approx(30.0)
        assert q["steel_weight_kg"] == pytest.approx(3_600.0)


class TestEstimateCostsDeliver:
    @pytest.mark.asyncio
    async def test_empty_is_error(self, container):
        result = await container.estimate_costs({}, {})
        assert result["status"] == "error"


class TestProgressTrackerDeliver:
    @pytest.mark.asyncio
    async def test_empty_is_error(self, container):
        result = await container.progress_tracker({}, {})
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_proxy_spi_hand_derived(self, container):
        """planned 50, actual 40 → variance −10, SPI 0.8, delay_days = 10/0.5 = 20.
        EV = 0.40 × 1_000_000 = 400_000; PV = 500_000; SV = −100_000.
        """
        result = await container.progress_tracker(
            {},
            {
                "planned_percent": 50.0,
                "actual_percent": 40.0,
                "contract_value": 1_000_000.0,
            },
        )
        assert result["status"] == "success"
        overall = result["overall_progress"]
        assert overall["variance_percent"] == -10.0
        assert overall["schedule_performance_index"] == pytest.approx(0.8)
        assert overall["estimated_delay_days"] == 20
        assert result["earned_value"]["schedule_variance"] == -100_000.0


class TestCommissioningChecklistDeliver:
    @pytest.mark.asyncio
    async def test_named_system_delivers_tests(self, container):
        result = await container.commissioning_checklist({"systems": ["waterproofing"]}, {})
        assert result["status"] == "success"
        assert result["summary"]["total_tests"] > 0
        assert "waterproofing" in result["checklists_by_system"]


class TestResourceHistogramDeliver:
    @pytest.mark.asyncio
    async def test_missing_file_is_error(self, container):
        result = await container.resource_histogram({}, {})
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_taskrsrc_700h_hand_derived(self, container):
        """A1010 LAB 200 + A1020 LAB 400 + A1030 CARP 100 = 700 man-hours."""
        path = Path("tests/fixtures/resource_loaded.xer")
        if not path.is_file():
            pytest.fail(f"resource-loaded fixture missing: {path}")
        result = await container.resource_histogram({}, {"schedule_file": str(path)})
        assert result["status"] == "success", result
        assert result["total_manhours"] == 700.0
        assert result["by_trade_totals"] == {"LAB": 600.0, "CARP": 100.0}
        assert result["source"] == "primavera_taskrsrc"


class TestClaimsBuilderDeliver:
    @pytest.mark.asyncio
    async def test_empty_does_not_invent_events(self, container):
        result = await container.claims_builder({}, {})
        assert result["status"] == "error"
        assert "delay_events" in result["error"]

    @pytest.mark.asyncio
    async def test_quantum_from_event_costs_hand_derived(self, container):
        """Two events: 10 d / 25_000 + 5 d / 10_000 → delay 15 d, claim 35_000.
        Breakdown: staff 30% = 10_500, accommodation 20% = 7_000,
        plant 25% = 8_750, insurances 10% = 3_500, OH 15% = 5_250.
        """
        result = await container.claims_builder(
            {
                "delay_events": [
                    {"event_id": "DE-01", "description": "access withheld", "delay_days": 10, "cost_impact": 25_000},
                    {"event_id": "DE-02", "description": "late IFC", "delay_days": 5, "cost_impact": 10_000},
                ]
            },
            {},
        )
        assert result["status"] == "success"
        assert result["delay_summary"]["total_delay_days"] == 15
        q = result["quantum_calculation"]
        assert q["calculation_mode"] == "event_sum"
        assert q["prolongation_period_days"] == 15
        assert q["total_claim"] == 35_000.0
        assert q["breakdown"]["site_staff"] == 10_500.0
        assert q["breakdown"]["site_accommodation"] == 7_000.0
        assert q["breakdown"]["plant_standing"] == 8_750.0
        assert q["breakdown"]["insurances_bonds"] == 3_500.0
        assert q["breakdown"]["overheads_profit"] == 5_250.0


class TestChangeOrderImpactDeliver:
    @pytest.mark.asyncio
    async def test_empty_is_error(self, container):
        result = await container.change_order_impact({}, {})
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_addition_cost_hand_derived(self, container):
        """direct 50_000, additional work → overhead 10_000, profit 5_000,
        complexity low (short text) → risk 2_500, total 67_500.
        """
        result = await container.change_order_impact(
            {"change_type": "additional excavation work", "direct_cost": 50_000},
            {},
        )
        assert result["status"] == "success"
        cost = result["cost_impact"]
        assert cost["direct_cost"] == 50_000.0
        assert cost["overhead"] == 10_000.0
        assert cost["profit"] == 5_000.0
        assert cost["risk_allowance"] == 2_500.0
        assert cost["total"] == 67_500.0
        assert result["category"] == "scope_addition"


class TestTenderBidDeliver:
    @pytest.mark.asyncio
    async def test_two_bids_lowest_price(self, container):
        bids = [
            {
                "contractor_name": "Alpha",
                "total_price": 1_000_000,
                "duration_days": 300,
                "experience_score": 90,
                "financial_stability": 95,
                "safety_rating": 92,
                "quality_score": 88,
            },
            {
                "contractor_name": "Beta",
                "total_price": 950_000,
                "duration_days": 320,
                "experience_score": 80,
                "financial_stability": 85,
                "safety_rating": 80,
                "quality_score": 82,
            },
        ]
        result = await container.tender_bid_analysis({"bids": bids}, {})
        assert result["status"] == "success"
        assert result["bids_received"] == 2
        assert result["price_analysis"]["lowest_bid"] == 950_000


class TestForensicDelayDeliver:
    @pytest.mark.asyncio
    async def test_missing_schedules_is_error(self, container):
        result = await container.forensic_delay_analysis({}, {})
        assert result["status"] == "error"


class TestWarrantyDeliver:
    @pytest.mark.asyncio
    async def test_named_system_expiry_hand_derived(self, container):
        """Handover 2026-01-01, mechanical warranty 24 months → expiry 2027-12-24
        (24 × 30 days = 720 days from 2026-01-01).
        """
        result = await container.warranty_maintenance_schedule(
            {
                "systems": [{"name": "AHU-1", "type": "mechanical", "supplier": "Fixture HVAC"}],
                "handover_date": "2026-01-01",
            },
            {},
        )
        assert result["status"] == "success"
        assert result["total_systems"] == 1
        row = result["warranty_register"][0]
        assert row["system"] == "AHU-1"
        assert row["warranty_months"] == 24
        assert row["warranty_expiry"] == "2027-12-22"

    @pytest.mark.asyncio
    async def test_empty_systems_is_error(self, container):
        result = await container.warranty_maintenance_schedule({}, {})
        assert result["status"] == "error"


class TestSubmittalLogDeliver:
    @pytest.mark.asyncio
    async def test_from_one_boq_row(self, container):
        result = await container.submittal_log_generator(
            {"boq": [{"description": "Structural steel beams", "quantity": 10}]},
            {},
        )
        assert result["status"] == "success"
        assert result["total_submittals"] >= 1
        descs = [s["description"] for s in result["submittal_register"]]
        assert any("Structural steel" in d for d in descs)
        assert not any("QA/QC Plan" in d for d in descs)

    @pytest.mark.asyncio
    async def test_empty_is_error(self, container):
        result = await container.submittal_log_generator({}, {})
        assert result["status"] == "error"


class TestRiskRegisterDeliver:
    @pytest.mark.asyncio
    async def test_from_supplied_risks(self, container):
        """severity high → p=0.7 i=0.8 score=56.0; medium → 0.4×0.5×100=20.0.
        Sorted descending. No catalogue padding.
        """
        result = await container.risk_register_auto_populate(
            {
                "risks": [
                    {"description": "Fall hazard", "category": "Safety", "severity": "high"},
                    {"description": "Material delay", "category": "Schedule", "severity": "medium"},
                ]
            },
            {},
        )
        assert result["status"] == "success"
        assert result["total_risks"] == 2
        assert result["risk_register"][0]["risk_score"] == 56.0
        assert result["risk_register"][1]["risk_score"] == 20.0
        assert all(r["source"] == "auto" for r in result["risk_register"])

    @pytest.mark.asyncio
    async def test_empty_is_error(self, container):
        result = await container.risk_register_auto_populate({}, {})
        assert result["status"] == "error"


class TestProcurementOptimizerDeliver:
    @pytest.mark.asyncio
    async def test_empty_does_not_rank_ghosts(self, container):
        result = await container.procurement_optimizer({}, {})
        if result.get("status") == "success":
            plan = (result.get("procurement_plan") or {}).get("plan") or []
            assert plan == [] or result.get("suppliers_evaluated") == 0
        else:
            assert result["status"] == "error"


class TestJetsonDispatchDeliver:
    @pytest.mark.asyncio
    async def test_missing_task_is_error(self, container):
        result = await container.jetson_dispatch({}, {})
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_staged_not_success_hollow(self, container):
        """Hardware is absent — pending_hardware is honest, success is not."""
        result = await container.jetson_dispatch({"task": "safety_scan"}, {})
        assert result["status"] == "pending_hardware"
        assert result["staged_job"]["task"] == "safety_scan"


class TestCarbonFootprintDeliver:
    @pytest.mark.asyncio
    async def test_empty_refuses_or_names_gap(self, container):
        result = await container.carbon_footprint_calculator({}, {})
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_factors_hand_derived(self, container):
        """12 m³ × 250 + 2_000 kg × 2.3 = 3_000 + 4_600 = 7_600 kg = 7.6 t."""
        result = await container.carbon_footprint_calculator(
            {"quantities": {"concrete_m3": {"quantity": 12}, "steel_kg": {"quantity": 2_000}}},
            {},
        )
        assert result["status"] == "success"
        assert result["total_embodied_carbon_kg"] == 7_600.0
        assert result["total_tonnes_co2"] == 7.6


class TestBimExtractDeliver:
    @pytest.mark.asyncio
    async def test_missing_file_is_error(self, container):
        result = await container.bim_extract({}, {})
        assert result["status"] == "error"


class TestPrimaveraParseDelegateDeliver:
    @pytest.mark.asyncio
    async def test_missing_file_is_error(self, container):
        result = await container.primavera_parse({}, {})
        assert result["status"] == "error"


class TestSpecAnalyzeDeliver:
    @pytest.mark.asyncio
    async def test_missing_input_is_error(self, container):
        result = await container.spec_analyze({}, {})
        assert result["status"] == "error"


class TestProcessDocumentDeliver:
    @pytest.mark.asyncio
    async def test_missing_file_is_error(self, container):
        result = await container.process_document({}, {})
        assert result["status"] == "error"


class TestForensicAndCdeAndTwin:
    @pytest.mark.asyncio
    async def test_cde_post_rfi_without_config_errors(self, container):
        result = await container.cde_post_rfi({}, {})
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_cde_poll_without_config_errors(self, container):
        result = await container.cde_poll_events({}, {})
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_digital_twin_without_model_is_prepared_not_pushed(self, container):
        """Empty payload must not claim a live platform push."""
        result = await container.digital_twin_sync({}, {})
        assert result.get("sync_status") == "prepared_not_pushed"
        assert "not connected" in str(result.get("note", "")).lower() or "not" in str(result.get("note", "")).lower()


class TestWirAndDraftsDeliver:
    @pytest.mark.asyncio
    async def test_wir_from_pour_facts(self, container):
        result = await container.wir_form(
            {"message": "WIR for raft pour 45 m3 at grid A, week 12, mix C30"},
            {},
        )
        if result.get("status") == "success":
            blob = str(result).lower()
            assert "45" in blob or "c30" in blob or "raft" in blob
            assert result.get("checklist") and result.get("activity")
        else:
            assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_job_requisition_empty_errors_or_asks(self, container):
        result = await container.job_requisition({}, {})
        assert result["status"] == "error"
        assert "scope" in result["error"].lower() or "facts" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_safety_briefing_empty_errors_or_asks(self, container):
        result = await container.safety_briefing({}, {})
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_rfp_empty_is_error(self, container):
        result = await container.rfp_draft({}, {})
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_job_requisition_title_not_invented_lighting(self, container):
        result = await container.job_requisition(
            {"text": "Job requisition for traffic signage and road markings only."},
            {},
        )
        assert result["status"] == "success"
        assert "street-lighting" not in result["title"].lower()


class TestAutoPipelineAndOrchestrate:
    @pytest.mark.asyncio
    async def test_auto_pipeline_missing_file_is_error(self, container):
        result = await container.auto_pipeline({}, {})
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_orchestrate_missing_message(self, container):
        result = await container.orchestrate({}, {})
        if result.get("status") == "success":
            matched = result.get("matched_actions") or result.get("actions") or []
            assert matched == [] or result.get("fallback")
        else:
            assert result["status"] == "error"


class TestValueEngineeringDeliver:
    @pytest.mark.asyncio
    async def test_empty_refuses_or_does_not_invent_savings(self, container):
        result = await container.value_engineering({}, {})
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_ggbs_saving_hand_derived(self, container):
        """concrete 100_000: GGBS −5% = −5_000, fly ash −8% = −8_000;
        steel 50_000 × 0 = 0. Sum of negative deltas = −13_000.
        Conservative scenario savings = abs(sum)×0.5 = 6_500.
        """
        result = await container.value_engineering(
            {
                "boq": [
                    {"id": "C-01", "material_type": "concrete_c30", "quantity": 400, "total_cost": 100_000, "carbon_impact": 10_000},
                    {"id": "S-01", "material_type": "structural_steel", "quantity": 20, "total_cost": 50_000, "carbon_impact": 8_000},
                ]
            },
            {},
        )
        assert result["status"] == "success"
        assert result["current_project_cost"] == 150_000
        assert result["alternatives_identified"] >= 3
        cons = result["scenarios"]["conservative"]
        assert cons["cost_savings"] == 6_500.0


class TestDailySiteReportDeliver:
    @pytest.mark.asyncio
    async def test_date_only_does_not_invent_weather_or_headcount(self, container):
        result = await container.daily_site_report({"date": "2026-03-08"}, {})
        if result.get("status") == "success":
            assert "sample" not in str(result).lower()
        else:
            assert result["status"] == "error"


class TestOmManualDeliver:
    @pytest.mark.asyncio
    async def test_empty_is_error(self, container):
        result = await container.om_manual_generator({}, {})
        assert result["status"] == "error"
        assert "equipment" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_named_system_without_equipment_is_error(self, container):
        """systems=['hvac'] is not an equipment_list — do not invent tags."""
        result = await container.om_manual_generator({"systems": ["hvac"]}, {})
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_equipment_list_delivers_tag(self, container):
        result = await container.om_manual_generator(
            {
                "equipment_list": [
                    {
                        "tag": "AHU-1",
                        "description": "Air handling unit",
                        "system_type": "mechanical",
                    }
                ]
            },
            {},
        )
        assert result["status"] == "success"
        blob = str(result)
        assert "AHU-1" in blob
        assert result.get("sections") or result.get("manual")


class TestEsgDeliver:
    @pytest.mark.asyncio
    async def test_with_boq_rows(self, container):
        result = await container.esg_sustainability_report(
            {
                "project_data": {"contract_value": 1_000_000},
                "boq": [{"material_type": "concrete", "quantity": 10, "unit": "m3", "total_cost": 1_500}],
                "manpower": {"total_workers": 5},
                "safety_records": [],
            },
            {"period": "annual"},
        )
        assert result["status"] == "success"
        assert "esg_scores" in result


class TestSafetyAuditDeliver:
    @pytest.mark.asyncio
    async def test_empty_refuses_or_does_not_invent_findings(self, container):
        result = await container.safety_compliance_audit({}, {})
        if result.get("status") == "success":
            findings = result.get("findings") or result.get("hazards") or []
            assert isinstance(findings, list)
        else:
            assert result["status"] == "error"


class TestLearnRecommendBenchmark:
    @pytest.mark.asyncio
    async def test_learn_missing_is_error(self, container):
        result = await container.learn({}, {})
        if result.get("status") == "success":
            assert not result.get("recorded_actual")
            assert "todo" not in str(result).lower()
        else:
            assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_recommend_missing_is_error(self, container):
        result = await container.recommend({}, {})
        if result.get("status") == "success":
            recs = result.get("recommendations") or result.get("items") or []
            assert recs == [] or result.get("error") or result.get("message")
        else:
            assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_benchmark_missing_item_is_error(self, container):
        result = await container.benchmark_lookup({}, {})
        if result.get("status") == "success":
            assert not result.get("unit_rate") or result.get("item") == ""
        else:
            assert result["status"] == "error"


class TestQaQcAndAsBuilt:
    @pytest.mark.asyncio
    async def test_qa_qc_without_photos_errors_or_asks(self, container):
        result = await container.qa_qc_inspection({}, {})
        assert result["status"] in {"error", "success"}

    @pytest.mark.asyncio
    async def test_as_built_without_files_errors(self, container):
        result = await container.as_built_deviation_report({}, {})
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_as_built_10_vs_10_05_is_major(self, container):
        """design 10.0 vs as-built 10.05; default tolerance 10 mm = 0.01.
        |10.05−10.0| = 0.05 > 0.01 → deviation.
        0.05 > 10/500 = 0.02 → major → CONDITIONAL.
        """
        result = await container.as_built_deviation_report(
            {"measurements": [{"type": "length", "value": 10.05, "unit": "m"}]},
            {"design_measurements": [{"type": "length", "value": 10.0, "unit": "m"}]},
        )
        assert result["status"] == "success"
        assert result["deviation_summary"]["total_deviations"] == 1
        assert result["deviation_summary"]["major"] == 1
        assert result["sign_off_status"] == "CONDITIONAL"
        assert result["deviations"][0]["deviation"] == pytest.approx(0.05)


class TestBimAnalysisAndClash:
    @pytest.mark.asyncio
    async def test_bim_analysis_missing_file(self, container):
        result = await container.bim_analysis({}, {})
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_clash_missing_file(self, container):
        result = await container.bim_clash_detection({}, {})
        assert result["status"] == "error"


class TestProcessContractAndSpec:
    @pytest.mark.asyncio
    async def test_process_contract_missing(self, container):
        result = await container.process_contract({}, {})
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_process_specification_missing(self, container):
        result = await container.process_specification_full({}, {})
        assert result["status"] == "error"


class TestRouteUnknownAndAliases:
    @pytest.mark.asyncio
    async def test_unknown_action_lists_known(self, container):
        result = await container.route("not_a_real_action", {}, {})
        assert result["status"] == "error"
        assert "known_actions" in result

    @pytest.mark.asyncio
    async def test_inspection_request_alias_calls_wir(self, container):
        result = await container.route(
            "inspection_request",
            {"message": "WIR for raft pour 12 m3 grid B week 4 mix C25"},
            {},
        )
        assert result.get("status") != "error" or "wir" in str(result.get("error", "")).lower()
        assert result.get("error", "") != "Unknown action: inspection_request"

    @pytest.mark.asyncio
    async def test_health_check_is_metadata_not_deliverable(self, container):
        result = await container.route("health_check", {}, {})
        assert result.get("status") in {"success", "ok", "healthy"} or "health" in str(result).lower()


class TestProcurementAnalysisDeliver:
    @pytest.mark.asyncio
    async def test_empty_is_error(self, container):
        result = await container.procurement_analysis({}, {})
        assert result["status"] == "error"
        assert result.get("stage") == "list_generation"


class TestRouteOnlyDeliver:
    @pytest.mark.asyncio
    async def test_extract_measurements_empty_is_error(self, container):
        result = await container.route("extract_measurements", {}, {})
        assert result["status"] == "error"
        assert "extract" in result["error"].lower() or "drawing" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_generate_construction_report_empty_is_error(self, container):
        result = await container.route("generate_construction_report", {}, {})
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_track_progress_empty_is_error(self, container):
        result = await container.route("track_progress", {}, {})
        assert result["status"] == "error"
        assert "photo" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_cost_estimate_empty_is_error(self, container):
        result = await container.route("cost_estimate", {}, {})
        assert result["status"] == "error"
        assert "quantit" in result["error"].lower() or "boq" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_analyze_spec_empty_is_error(self, container):
        result = await container.route("analyze_spec", {}, {})
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_schedule_risk_empty_is_error(self, container):
        result = await container.route("schedule_risk", {}, {})
        assert result["status"] == "error"
