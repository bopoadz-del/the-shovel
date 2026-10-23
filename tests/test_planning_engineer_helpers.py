"""Planning-engineer helpers: look-ahead, classic EVM, PE productivity, mix/units."""
from __future__ import annotations

from datetime import date

import pytest

from app.core.construction_knowledge import calculate_evm
from app.containers.construction import ConstructionContainer
from app.lib.construction_formulas_planning import (
    concrete_mix_proportions,
    material_consumption,
    pe_unit_convert,
    productivity_manpower_duration,
    progress_quantity,
)
from app.lib.pm_computations import activity_overlaps_window, select_look_ahead
from tests.conftest import requires_construction_kit


@pytest.fixture
def container():
    return ConstructionContainer()


# ---------------------------------------------------------------------------
# Look-ahead window selection
# ---------------------------------------------------------------------------

def test_activity_overlaps_window_inclusive():
    assert activity_overlaps_window(
        date(2026, 3, 1), date(2026, 3, 10),
        date(2026, 3, 5), date(2026, 3, 25),
    )
    assert not activity_overlaps_window(
        date(2026, 2, 1), date(2026, 2, 10),
        date(2026, 3, 5), date(2026, 3, 25),
    )


def test_select_look_ahead_filters_by_calendar_window():
    acts = [
        {"id": "A", "name": "Before", "start": "2026-03-01", "finish": "2026-03-05",
         "total_float_days": 10},
        {"id": "B", "name": "Overlap", "start": "2026-03-10", "finish": "2026-03-20",
         "total_float_days": 0, "remaining_duration_days": 8, "wbs_id": "1.2"},
        {"id": "C", "name": "After", "start": "2026-04-10", "finish": "2026-04-20",
         "total_float_days": 5},
        {"id": "D", "name": "No dates", "total_float_days": 0},
    ]
    out = select_look_ahead(acts, as_of=date(2026, 3, 8), window_days=21)
    assert out["as_of"] == "2026-03-08"
    assert out["window_end"] == "2026-03-28"  # inclusive 21-day window
    ids = [a["id"] for a in out["activities"]]
    assert ids == ["B"]
    assert out["activities"][0]["is_critical"] is True
    assert out["activities"][0]["remaining_duration_days"] == 8
    assert out["activities"][0]["wbs"] == "1.2"


def test_select_look_ahead_empty_schedule_returns_empty_not_fabricated():
    out = select_look_ahead([], as_of=date(2026, 1, 1), window_days=28)
    assert out["count"] == 0
    assert out["activities"] == []


def test_select_look_ahead_rejects_bad_window():
    with pytest.raises(ValueError):
        select_look_ahead([], window_days=0)


@requires_construction_kit
@pytest.mark.asyncio
async def test_look_ahead_container_refuses_without_xer(container):
    result = await container.look_ahead({}, {})
    assert result["status"] == "error"
    assert "schedule" in result["error"].lower() or ".xer" in result["error"].lower()


@requires_construction_kit
@pytest.mark.asyncio
async def test_look_ahead_from_minimal_xer(container, tmp_path):
    xer = "\n".join([
        "%T\tPROJECT",
        "%F\tproj_id\tproj_short_name\tplan_start_date\tlast_recalc_date",
        "%R\t1\tDemo\t2026-03-01 00:00\t2026-03-08 00:00",
        "%T\tTASK",
        "%F\ttask_id\ttask_code\ttask_name\ttarget_drtn_hr_cnt\tearly_start_date\tearly_end_date\ttotal_float_hr_cnt\tremain_drtn_hr_cnt\twbs_id\tstatus_code\tphys_complete_pct",
        "%R\t1001\tA\tMobilise\t40\t2026-03-01 00:00\t2026-03-05 00:00\t80\t0\tW1\tTK_Complete\t100",
        "%R\t1002\tB\tExcavate\t80\t2026-03-10 00:00\t2026-03-20 00:00\t0\t64\tW2\tTK_Active\t20",
        "%R\t1003\tC\tConcrete\t120\t2026-04-15 00:00\t2026-04-30 00:00\t40\t120\tW3\tTK_NotStart\t0",
        "%E",
        "",
    ])
    path = tmp_path / "demo.xer"
    path.write_text(xer, encoding="cp1252")

    result = await container.look_ahead(
        {},
        {"schedule_file": str(path), "as_of": "2026-03-08", "days": 21},
    )
    assert result["status"] == "success", result
    assert result["action"] == "look_ahead"
    assert result["window_days"] == 21
    codes = {a["code"] or a["id"] for a in result["activities"]}
    assert "B" in codes
    assert "C" not in codes  # after window
    hit = next(a for a in result["activities"] if (a["code"] or a["id"]) == "B")
    assert hit["is_critical"] is True
    assert hit["remaining_duration_days"] == pytest.approx(8.0)


@requires_construction_kit
@pytest.mark.asyncio
async def test_look_ahead_empty_activities_is_error(container, tmp_path):
    xer = "%T\tTASK\n%F\ttask_id\ttask_code\ttask_name\ttarget_drtn_hr_cnt\n%E\n"
    path = tmp_path / "empty.xer"
    path.write_text(xer, encoding="cp1252")
    result = await container.look_ahead({}, {"schedule_file": str(path)})
    assert result["status"] == "error"
    assert "no activities" in result["error"].lower()


# ---------------------------------------------------------------------------
# Classic EVM — SV vs CV
# ---------------------------------------------------------------------------

def test_calculate_evm_sv_vs_cv_and_aliases():
    r = calculate_evm(pv=100_000, ev=90_000, ac=95_000, bac=200_000)
    assert r["SPI"] == pytest.approx(0.9)
    assert r["CPI"] == pytest.approx(90_000 / 95_000, rel=1e-3)
    assert r["SV"] == -10_000  # EV − PV (schedule)
    assert r["CV"] == -5_000   # EV − AC (cost)
    assert r["EAC"] == pytest.approx(200_000 / (90_000 / 95_000), rel=1e-3)
    assert r["ETC"] == pytest.approx(r["EAC"] - 95_000, rel=1e-3)
    assert r["VAC"] == pytest.approx(200_000 - r["EAC"], rel=1e-3)
    assert r["formulas"]["EAC"] == "BAC / CPI"


def test_calculate_evm_refuses_missing_actuals():
    r = calculate_evm(pv=100_000, ev=90_000)  # AC missing
    assert isinstance(r.get("error"), str)
    assert "AC" in r["error"] or "missing" in r["error"].lower()


def test_calculate_evm_without_bac_skips_forecasts():
    r = calculate_evm(bcws=100, bcwp=90, acwp=95)
    assert "error" not in r
    assert r["SPI"] == pytest.approx(0.9)
    assert r["CV"] == -5
    assert r["SV"] == -10
    assert r["EAC"] is None
    assert r["ETC"] is None
    assert r["VAC"] is None
    assert "BAC omitted" in r.get("note", "")


@requires_construction_kit
@pytest.mark.asyncio
async def test_evm_calculate_container_path(container):
    ok = await container.evm_calculate({}, {"pv": 500, "ev": 400, "ac": 450, "bac": 1000})
    assert ok["status"] == "success"
    assert ok["evm"]["SV"] == -100
    assert ok["evm"]["CV"] == -50
    bad = await container.evm_calculate({}, {"pv": 500, "ev": 400})
    assert bad["status"] == "error"


@requires_construction_kit
@pytest.mark.asyncio
async def test_progress_tracker_renames_sv_not_cv(container):
    result = await container.progress_tracker(
        {},
        {"planned_percent": 50, "actual_percent": 51, "contract_value": 1_000_000},
    )
    ev = result["earned_value"]
    assert ev["schedule_variance"] == 10_000
    assert ev["cost_variance"] is None
    # With AC supplied, true CV appears.
    with_ac = await container.progress_tracker(
        {},
        {
            "planned_percent": 50, "actual_percent": 51,
            "contract_value": 1_000_000, "actual_cost": 480_000,
        },
    )
    assert with_ac["earned_value"]["cost_variance"] == 30_000  # 510k − 480k


# ---------------------------------------------------------------------------
# Qty → productivity → manpower → duration
# ---------------------------------------------------------------------------

def test_progress_quantity_formulas():
    r = progress_quantity(total_qty=1000, planned_qty=400, actual_qty=350)
    assert r["planned_percent"] == pytest.approx(40.0)
    assert r["actual_percent"] == pytest.approx(35.0)
    assert r["remaining_qty"] == pytest.approx(650.0)
    assert r["progress_variance_percent"] == pytest.approx(-5.0)


def test_progress_quantity_refuses_without_qty():
    r = progress_quantity(total_qty=1000)
    assert isinstance(r.get("error"), str)


def test_productivity_manpower_duration_chain():
    r = productivity_manpower_duration(
        quantity_executed=100, man_hours=50,
        quantity=200, daily_production=20,
        remaining_manhours=40, available_hours=8,
    )
    assert r["productivity"] == pytest.approx(2.0)
    assert r["manhours_required"] == pytest.approx(100.0)  # 200 / 2
    assert r["duration"] == pytest.approx(10.0)
    assert r["required_manpower"] == pytest.approx(5.0)


def test_productivity_refuses_without_pairs():
    r = productivity_manpower_duration(quantity=100)
    assert isinstance(r.get("error"), str)


def test_productivity_rate_plus_crew_cost_computes_sar():
    """Live A2-1 first call: productivity_rate + crew_cost_per_day."""
    r = productivity_manpower_duration(
        quantity=3400,
        productivity_rate=42,
        rate_unit="m2 per gang-day",
        crew_cost_per_day=1950,
    )
    assert r.get("error") is None
    assert r["duration"] == pytest.approx(3400 / 42, abs=0.01)
    assert r["total_cost"] == pytest.approx(3400 / 42 * 1950, abs=0.5)


# ---------------------------------------------------------------------------
# Unit conversions + material / concrete mix refuse-without-inputs
# ---------------------------------------------------------------------------

def test_pe_unit_convert_m_to_ft():
    r = pe_unit_convert(1, "m", "ft")
    assert r["value_out"] == pytest.approx(3.2808399, rel=1e-5)


def test_pe_unit_convert_m3_ft3_and_day_hour():
    r = pe_unit_convert(1, "m3", "ft3")
    assert r["value_out"] == pytest.approx(35.3146667, rel=1e-5)
    t = pe_unit_convert(1, "day", "hour")
    assert t["value_out"] == pytest.approx(24.0)


def test_material_consumption_refuses_without_inputs():
    r = material_consumption(quantity_of_work=100)
    assert isinstance(r.get("error"), str)


def test_material_consumption_with_waste():
    r = material_consumption(quantity_of_work=100, output_per_unit=10, waste_percent=5)
    assert r["base_without_waste"] == pytest.approx(10.0)
    assert r["material_required"] == pytest.approx(10.5)


def test_concrete_mix_refuses_without_ratios():
    r = concrete_mix_proportions(wet_volume=10)
    assert isinstance(r.get("error"), str)
    assert "refuse" in r["error"].lower() or "requires" in r["error"].lower()


def test_concrete_mix_1_2_4():
    r = concrete_mix_proportions(
        wet_volume=10, cement_parts=1, sand_parts=2, aggregate_parts=4,
    )
    assert r["dry_volume"] == pytest.approx(15.4)
    assert r["cement_volume"] == pytest.approx(15.4 / 7)
    assert r["sand_volume"] == pytest.approx(15.4 * 2 / 7)
    assert r["aggregate_volume"] == pytest.approx(15.4 * 4 / 7)


def test_new_calculators_registered_in_construction_calc():
    from app.lib import construction_formulas as cf
    for name in (
        "progress_quantity",
        "productivity_manpower_duration",
        "pe_unit_convert",
        "material_consumption",
        "concrete_mix_proportions",
        "calculate_evm",
    ):
        assert name in cf.CALCULATORS
    # Via run_calculation envelope
    env = cf.run_calculation(
        "calculate_evm",
        {"pv": 100, "ev": 90, "ac": 95, "bac": 200},
    )
    assert env["status"] == "success"
    assert env["result"]["SV"] == -10
    assert env["result"]["CV"] == -5
