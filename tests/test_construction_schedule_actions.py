"""Deterministic tests for construction schedule non-file actions.

Covers the public actions:
  - progress_tracker
  - warranty_maintenance_schedule
  - commissioning_checklist

And the pure helpers:
  - _calculate_duration_days
  - _calculate_date_diff
  - _analyze_schedule_risks
  - _generate_recovery_options

No real Primavera files or external services are used; the container is
exercised directly with synthetic inputs.
"""

from __future__ import annotations

import pytest

from app.containers.construction import ConstructionContainer
from tests.conftest import requires_construction_kit


@pytest.fixture
def container():
    return ConstructionContainer()


# ---------------------------------------------------------------------------
# Pure helper tests
# ---------------------------------------------------------------------------

@requires_construction_kit
def test_calculate_duration_days_typical(container):
    assert container._calculate_duration_days("2024-01-01", "2024-01-10") == 9


@requires_construction_kit
def test_calculate_duration_days_same_day(container):
    assert container._calculate_duration_days("2024-06-15", "2024-06-15") == 0


@requires_construction_kit
def test_calculate_duration_days_negative_range_is_zero(container):
    """Finish before start is clamped to zero (duration cannot be negative)."""
    assert container._calculate_duration_days("2024-06-15", "2024-06-01") == 0


@requires_construction_kit
def test_calculate_duration_days_iso_with_z(container):
    assert container._calculate_duration_days("2024-01-01T00:00:00Z", "2024-01-10T00:00:00Z") == 9


@requires_construction_kit
def test_calculate_duration_days_invalid_returns_zero(container):
    assert container._calculate_duration_days("not-a-date", "2024-01-10") == 0
    assert container._calculate_duration_days("2024-01-01", "") == 0


@requires_construction_kit
def test_calculate_date_diff_typical(container):
    assert container._calculate_date_diff("2024-01-01", "2024-01-10") == 9


@requires_construction_kit
def test_calculate_date_diff_negative_is_zero(container):
    """Delay analysis only cares about positive slips; negative diffs are clamped."""
    assert container._calculate_date_diff("2024-06-15", "2024-06-01") == 0


@requires_construction_kit
def test_calculate_date_diff_iso_with_z(container):
    assert container._calculate_date_diff("2024-01-01T00:00:00Z", "2024-01-10T00:00:00Z") == 9


@requires_construction_kit
def test_calculate_date_diff_invalid_returns_zero(container):
    assert container._calculate_date_diff("bad", "2024-01-10") == 0


@requires_construction_kit
def test_analyze_schedule_risks_high_when_low_float(container):
    risks = container._analyze_schedule_risks({"average_float": 1.5})
    assert len(risks) == 1
    risk = risks[0]
    assert risk["category"] == "schedule"
    assert risk["impact"] == "high"
    assert "minimal overall float" in risk["description"]
    assert "mitigation" in risk and risk["mitigation"]


@requires_construction_kit
def test_analyze_schedule_risks_empty_when_healthy_float(container):
    assert container._analyze_schedule_risks({"average_float": 5.0}) == []
    assert container._analyze_schedule_risks({"average_float": 2.0}) == []


@requires_construction_kit
def test_generate_recovery_options_no_delay_analysis(container):
    assert container._generate_recovery_options(None, {}) == []


@requires_construction_kit
def test_generate_recovery_options_zero_delay(container):
    assert container._generate_recovery_options({"total_delay_days": 0}, {}) == []


@requires_construction_kit
def test_generate_recovery_options_returns_three_strategies(container):
    options = container._generate_recovery_options({"total_delay_days": 10}, {})
    assert len(options) == 3
    strategies = [o["strategy"] for o in options]
    assert "Crash Critical Path" in strategies
    assert "Fast Track" in strategies
    assert "Scope Reduction" in strategies

    crash = next(o for o in options if o["strategy"] == "Crash Critical Path")
    assert crash["potential_savings_days"] == 5.0
    fast = next(o for o in options if o["strategy"] == "Fast Track")
    assert fast["potential_savings_days"] == 3.0
    scope = next(o for o in options if o["strategy"] == "Scope Reduction")
    assert scope["potential_savings_days"] == 5.0


# ---------------------------------------------------------------------------
# Public action tests
# ---------------------------------------------------------------------------

@requires_construction_kit
@pytest.mark.asyncio
async def test_progress_tracker_on_track(container):
    result = await container.progress_tracker(
        input_data={},
        params={
            "planned_percent": 50.0,
            "actual_percent": 51.0,
            "contract_value": 1_000_000.0,
            "reporting_period": "June 2025",
        },
    )
    assert result["status"] == "success"
    assert result["action"] == "progress_tracker"
    assert result["reporting_period"] == "June 2025"

    overall = result["overall_progress"]
    assert overall["planned_percent"] == 50.0
    assert overall["actual_percent"] == 51.0
    assert overall["variance_percent"] == 1.0
    assert overall["schedule_performance_index"] == pytest.approx(1.02, rel=1e-3)
    assert overall["status"] == "on_track"
    assert overall["estimated_delay_days"] == 0

    ev = result["earned_value"]
    assert ev is not None
    assert ev["contract_value"] == 1_000_000.0
    assert ev["earned_value"] == 510_000.0
    assert ev["planned_value"] == 500_000.0
    # EV − PV is schedule variance (SV), not classic cost variance.
    assert ev["schedule_variance"] == 10_000.0
    assert ev["cost_variance"] is None  # AC not supplied
    assert overall["schedule_performance_index_kind"] == (
        "progress_proxy_actual_over_planned_pct"
    )

    assert result["recommendations"] == ["Maintain current momentum"]


@requires_construction_kit
@pytest.mark.asyncio
async def test_progress_tracker_delayed(container):
    result = await container.progress_tracker(
        input_data={
            "planned_percent": 50.0,
            "actual_percent": 42.0,
            "activities": [
                {"name": "Foundation", "planned_percent": 80.0, "actual_percent": 70.0},
                {"name": "Steel", "planned_percent": 30.0, "actual_percent": 20.0},
            ],
        },
        params={"contract_value": 2_000_000.0},
    )
    overall = result["overall_progress"]
    assert overall["variance_percent"] == -8.0
    assert overall["status"] == "delayed"
    assert overall["schedule_performance_index"] == pytest.approx(0.84, rel=1e-3)
    assert overall["estimated_delay_days"] == 16  # round(8 / 0.5)

    activities = result["activities"]
    assert len(activities) == 2
    assert activities[0]["status"] == "delayed"
    assert activities[0]["variance"] == -10.0

    assert result["key_risks"]  # variance < -5
    assert any("recovery plan" in r for r in result["key_risks"])


@requires_construction_kit
@pytest.mark.asyncio
async def test_progress_tracker_ahead(container):
    result = await container.progress_tracker(
        input_data={},
        params={"planned_percent": 40.0, "actual_percent": 48.0},
    )
    assert result["overall_progress"]["status"] == "ahead"
    assert result["overall_progress"]["variance_percent"] == 8.0
    assert result["earned_value"] is None  # no contract value


@requires_construction_kit
@pytest.mark.asyncio
async def test_progress_tracker_zero_planned_avoids_division_error(container):
    result = await container.progress_tracker(
        input_data={},
        params={"planned_percent": 0.0, "actual_percent": 5.0},
    )
    assert result["overall_progress"]["schedule_performance_index"] == 1.0


@requires_construction_kit
@pytest.mark.asyncio
async def test_warranty_maintenance_schedule_custom_systems(container):
    result = await container.warranty_maintenance_schedule(
        input_data={},
        params={
            "systems": [
                {"name": "HVAC System", "type": "mechanical", "supplier": "Acme Mech"},
                {"name": "Electrical Distribution", "type": "electrical", "supplier": "Volt Co"},
            ],
            "project_name": "Data Center A",
            "handover_date": "2025-01-01",
            "defects_liability_months": 12,
        },
    )
    assert result["status"] == "success"
    assert result["action"] == "warranty_maintenance_schedule"
    assert result["project"] == "Data Center A"
    assert result["handover_date"] == "2025-01-01"
    assert result["defects_liability_period_months"] == 12
    assert result["total_systems"] == 2

    register = result["warranty_register"]
    assert len(register) == 2
    assert register[0]["system"] == "HVAC System"
    assert register[0]["warranty_months"] == 24
    assert register[0]["warranty_expiry"] == "2026-12-22"
    assert register[0]["dlp_expiry"] == "2025-12-27"  # 12 months * 30 days

    assert register[1]["system"] == "Electrical Distribution"
    assert register[1]["warranty_months"] == 12
    assert register[1]["warranty_expiry"] == "2025-12-27"

    maintenance = result["maintenance_schedule"]
    assert len(maintenance) == 5  # 3 mechanical + 2 electrical
    assert all("next_due" in m for m in maintenance)

    early = result["early_expiries"]
    assert len(early) == 1
    assert early[0]["system"] == "Electrical Distribution"

    assert any("2025-12-27" in r for r in result["recommendations"])


@requires_construction_kit
@pytest.mark.asyncio
async def test_warranty_maintenance_schedule_defaults_when_empty(container):
    result = await container.warranty_maintenance_schedule(input_data={}, params={})
    assert result["status"] == "error"
    assert "systems" in result["error"].lower() or "equipment" in result["error"].lower()


@requires_construction_kit
@pytest.mark.asyncio
async def test_warranty_maintenance_schedule_respects_input_data(container):
    """Regression: handover_date, defects_liability_months and project_name
    must be read from input_data when not supplied in params."""
    result = await container.warranty_maintenance_schedule(
        input_data={
            "systems": [{"name": "Backup Generator", "type": "electrical", "supplier": "GenSet"}],
            "project_name": "From Input",
            "handover_date": "2025-06-15",
            "defects_liability_months": 18,
        },
        params={},
    )
    assert result["project"] == "From Input"
    assert result["handover_date"] == "2025-06-15"
    assert result["defects_liability_period_months"] == 18
    assert result["warranty_register"][0]["dlp_expiry"] == "2026-12-07"  # 18 months * 30 days


@requires_construction_kit
@pytest.mark.asyncio
async def test_commissioning_checklist_default_systems(container):
    result = await container.commissioning_checklist(
        input_data={},
        params={"substantial_completion_date": "2025-01-01", "equipment_list": ["chiller", "ups"]},
    )
    assert result["status"] == "success"
    assert result["action"] == "commissioning_checklist_generated"
    assert result["project_phase"] == "pre_handover"
    assert result["substantial_completion_target"] == "2025-01-01"

    # 5 default systems * 2 weeks + 0 extra equipment (2 // 10 == 0) = 10 weeks
    assert result["commissioning_period_weeks"] == 10
    assert result["completion_target"] == "2025-03-12T00:00:00"

    summary = result["summary"]
    assert summary["systems_covered"] == 5
    assert summary["total_tests"] > 0
    assert summary["passed"] == 0
    assert summary["failed"] == 0
    assert summary["pending"] == summary["total_tests"]
    assert summary["percent_complete"] == 0.0

    assert set(result["checklists_by_system"].keys()) == {
        "electrical", "hvac", "fire_protection", "elevators", "building_envelope"
    }
    assert len(result["master_test_schedule"]) == summary["total_tests"]
    assert len(result["witness_required"]) > 0
    assert len(result["third_party_testing"]) > 0  # fire smoke detector sensitivity
    assert len(result["documentation_required"]) == 5
    assert len(result["training_requirements"]) == 5

    sign_off = result["final_sign_off"]
    assert sign_off["mechanical_contractor"] == "pending"
    assert sign_off["commissioning_authority"] == "pending"


@requires_construction_kit
@pytest.mark.asyncio
async def test_commissioning_checklist_custom_systems(container):
    result = await container.commissioning_checklist(
        input_data={},
        params={
            "systems": ["electrical", "mechanical", "fire"],
            "equipment_list": list(range(15)),
            "substantial_completion_date": "2025-06-01",
        },
    )
    summary = result["summary"]
    assert summary["systems_covered"] == 3
    assert summary["total_tests"] == 20  # 7 electrical + 7 hvac + 6 fire_protection

    # 3 systems * 2 weeks + 15 // 10 = 6 + 1 = 7 weeks
    assert result["commissioning_period_weeks"] == 7
    assert result["completion_target"] == "2025-07-20T00:00:00"

    assert len(result["witness_required"]) == 12
    assert len(result["third_party_testing"]) == 1


@requires_construction_kit
@pytest.mark.asyncio
async def test_commissioning_checklist_respects_input_data(container):
    """Regression: systems and substantial_completion_date must be read from
    input_data when not supplied in params."""
    result = await container.commissioning_checklist(
        input_data={
            "systems": ["electrical"],
            "substantial_completion_date": "2025-09-01",
        },
        params={},
    )
    assert result["summary"]["systems_covered"] == 1
    assert result["substantial_completion_target"] == "2025-09-01"
    assert result["commissioning_period_weeks"] == 2
    assert result["completion_target"] == "2025-09-15T00:00:00"


class TestClaimsBuilderHonestGate:
    """F2 — claims_builder must never invent delay events (was DE-001/DE-002)."""

    @pytest.mark.asyncio
    async def test_no_delay_events_errors_not_fabricates(self, container):
        result = await container.claims_builder({}, {})
        assert result["status"] == "error"
        assert "delay_events" in result["error"]
        assert "quantum" not in result and "total_claim" not in result

    @pytest.mark.asyncio
    async def test_real_delay_events_build_claim(self, container):
        events = [{"event_id": "DE-01", "description": "employer delay",
                   "delay_days": 10, "responsibility": "employer", "cost_impact": 50000}]
        result = await container.claims_builder({"delay_events": events}, {})
        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_eot_notice_period_lookup_does_not_demand_delay_events(
        self, container,
    ):
        """S14 — Contract Data days-to-notice is not a claim-build gate."""
        ask = (
            "Within how many days must the Contractor give notice of an "
            "EOT claim on this project?"
        )
        result = await container.claims_builder(
            {"user_message": ask}, {"user_message": ask},
        )
        assert result["status"] == "error"
        err = (result.get("error") or "").lower()
        assert "contract data" in err
        assert "do not require" in err


# ---------------------------------------------------------------------------
# F4 — _fetch_weather: real Open-Meteo or honest 'unavailable', never fabricated
# ---------------------------------------------------------------------------

class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class _FakeClient:
    """Async-context httpx stand-in returning canned payloads in .get() order."""

    def __init__(self, payloads, raise_on_get=False):
        self._payloads = list(payloads)
        self._raise = raise_on_get

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, params=None):
        if self._raise:
            raise RuntimeError("network down")
        return _FakeResp(self._payloads.pop(0))


def _patch_httpx(monkeypatch, payloads=None, raise_on_get=False):
    import httpx
    monkeypatch.setattr(
        httpx, "AsyncClient",
        lambda *a, **k: _FakeClient(payloads or [], raise_on_get=raise_on_get),
    )


_GEO_OK = {"results": [{"latitude": 24.7, "longitude": 46.6}]}
_DAILY_OK = {"daily": {
    "time": ["2026-07-10"],
    "temperature_2m_max": [44.1], "temperature_2m_min": [29.3],
    "precipitation_sum": [0.0], "windspeed_10m_max": [18.0],
    "weathercode": [0],
}}


class TestFetchWeatherHonestOrReal:
    """F4 — weather must be a real Open-Meteo reading or an honest 'unavailable';
    it must NEVER return the old fabricated 35C/sunny/favorable block."""

    @pytest.mark.asyncio
    async def test_no_location_unavailable(self, container):
        r = await container._fetch_weather("", "2026-07-10")
        assert r["status"] == "unavailable"
        assert "location" in r["reason"]

    @pytest.mark.asyncio
    async def test_invalid_date_unavailable(self, container):
        r = await container._fetch_weather("Riyadh", "not-a-date")
        assert r["status"] == "unavailable"
        assert "invalid date" in r["reason"]

    @pytest.mark.asyncio
    async def test_geocode_miss_unavailable(self, container, monkeypatch):
        _patch_httpx(monkeypatch, payloads=[{"results": []}])
        r = await container._fetch_weather("Nowheresville XYZ", "2026-07-10")
        assert r["status"] == "unavailable"
        assert "geocode" in r["reason"]

    @pytest.mark.asyncio
    async def test_no_record_unavailable(self, container, monkeypatch):
        _patch_httpx(monkeypatch, payloads=[_GEO_OK, {"daily": {"time": []}}])
        r = await container._fetch_weather("Riyadh", "2026-07-10")
        assert r["status"] == "unavailable"
        assert "no weather record" in r["reason"]

    @pytest.mark.asyncio
    async def test_network_error_unavailable(self, container, monkeypatch):
        _patch_httpx(monkeypatch, raise_on_get=True)
        r = await container._fetch_weather("Riyadh", "2026-07-10")
        assert r["status"] == "unavailable"
        assert "weather service error" in r["reason"]

    @pytest.mark.asyncio
    async def test_success_returns_real_reading(self, container, monkeypatch):
        _patch_httpx(monkeypatch, payloads=[_GEO_OK, _DAILY_OK])
        r = await container._fetch_weather("Riyadh", "2026-07-10")
        assert r["status"] == "success"
        assert r["temperature_high"] == 44.1
        assert r["conditions"] == "clear"          # WMO 0, not fabricated "sunny"
        assert r["impact"] == "favorable"           # 0mm precip, 18 km/h wind
        assert r["source"] == "open-meteo"

    @pytest.mark.asyncio
    async def test_archive_path_for_old_date(self, container, monkeypatch):
        old = {"daily": {"time": ["2020-01-01"],
                         "temperature_2m_max": [12.0], "temperature_2m_min": [4.0],
                         "precipitation_sum": [15.0], "windspeed_10m_max": [45.0],
                         "weathercode": [65]}}
        _patch_httpx(monkeypatch, payloads=[_GEO_OK, old])
        r = await container._fetch_weather("London", "2020-01-01")
        assert r["status"] == "success"
        assert r["conditions"] == "heavy rain"      # WMO 65
        assert r["impact"] == "adverse"             # 15mm precip / 45 km/h wind

    def test_wmo_conditions_mapping(self, container):
        assert container._wmo_conditions(0) == "clear"
        assert container._wmo_conditions(95) == "thunderstorm"
        assert container._wmo_conditions(999) == "unknown"
        assert container._wmo_conditions(None) == "unknown"

    def test_weather_impact_bands(self, container):
        assert container._weather_impact(0.0, 10.0) == "favorable"
        assert container._weather_impact(2.0, 10.0) == "marginal"
        assert container._weather_impact(0.0, 30.0) == "marginal"
        assert container._weather_impact(20.0, 10.0) == "adverse"
        assert container._weather_impact(None, None) == "unknown"


# ---------------------------------------------------------------------------
# V1 — resource_histogram: real P6 TASKRSRC histogram or honest error,
# never the old fake-success all-zero histogram with hardcoded trade ratios.
# ---------------------------------------------------------------------------

# A minimal resource-loaded .xer (PROJECT/CALENDAR/TASK/TASKPRED/TASKRSRC):
#   A1010 Excavation (5d) -> A1020 Foundation (10d) -> A1030 Backfill (5d)
#   TASKRSRC: A1010 LAB 200h, A1020 LAB 400h, A1030 CARP 100h  => 700 man-hours
_RESLOADED_XER = "tests/fixtures/resource_loaded.xer"
# A real .xer with NO TASKRSRC rows (no resource loading).
_NORSRC_XER = "tests/fixtures/no_taskrsrc.xer"


class TestDailySiteReportLocation:
    """F4 — daily_site_report must read the weather location from input_data too,
    not only params: the generic predefined dispatcher merges resolved params
    into the envelope and passes it as input_data. A params-only read silently
    dropped the project location (regression caught in review)."""

    @pytest.mark.asyncio
    async def test_location_from_input_data_reaches_weather(self, container, monkeypatch):
        seen = {}

        async def fake_fetch(location, date):
            seen["location"] = location
            return {"location": location, "date": date, "status": "success",
                    "conditions": "clear", "temperature_high": 30, "impact": "favorable",
                    "source": "test"}
        monkeypatch.setattr(container, "_fetch_weather", fake_fetch)

        # location supplied via input_data (the envelope), params empty — the
        # exact shape the dispatcher produces.
        result = await container.daily_site_report(
            {"location": "Riyadh", "date": "2026-07-10"}, {})
        assert seen.get("location") == "Riyadh"
        w = result["report_metadata"]["weather_conditions"]
        assert w["status"] == "success"
        assert w["location"] == "Riyadh"

    @pytest.mark.asyncio
    async def test_no_location_leaves_weather_empty_not_fabricated(self, container, monkeypatch):
        async def fake_fetch(location, date):  # pragma: no cover — must NOT be called
            raise AssertionError("_fetch_weather called without a location")
        monkeypatch.setattr(container, "_fetch_weather", fake_fetch)
        result = await container.daily_site_report({"date": "2026-07-10"}, {})
        assert result["report_metadata"]["weather_conditions"] == {}


class TestResourceHistogramRealOrHonest:
    """V1 — the container action must produce a real time-phased histogram from
    the schedule's own TASKRSRC assignments, or an honest error — never a
    fabricated/zeroed histogram."""

    @pytest.mark.asyncio
    async def test_real_taskrsrc_produces_nonzero_histogram(self, container):
        import os
        if not os.path.exists(_RESLOADED_XER):
            pytest.skip("resource-loaded fixture missing")
        r = await container.resource_histogram({}, {"schedule_file": _RESLOADED_XER})
        assert r["status"] == "success"
        # Real man-hours from TASKRSRC target_qty (200+400+100), not zero.
        assert r["total_manhours"] == 700.0
        # Real per-resource totals keyed on P6 rsrc_id — not a hardcoded
        # 30/20/15/15/20 concrete/masonry/steel/... split.
        assert r["by_trade_totals"] == {"LAB": 600.0, "CARP": 100.0}
        assert r["period_count"] >= 1
        assert r["source"] == "primavera_taskrsrc"
        # The fabricated fixed-ratio trades must be gone.
        assert "concrete" not in r["by_trade_totals"]

    @pytest.mark.asyncio
    async def test_labor_filter_excludes_material_and_cost(self, container):
        # Regression for the real the client project baseline finding: a .xer whose TASKRSRC
        # mixes RT_Labor with an RT_Mat resource (STEEL, target_qty 999999) must
        # count man-hours from labor ONLY — never sum the material quantity.
        import os
        fx = "tests/fixtures/resource_loaded_mixed.xer"
        if not os.path.exists(fx):
            pytest.skip("mixed-resource fixture missing")
        r = await container.resource_histogram({}, {"schedule_file": fx})
        assert r["status"] == "success"
        assert r["resource_types_included"] == "RT_Labor"
        # 200 + 400 + 100 labor hours; the 999999 material qty is excluded.
        assert r["total_manhours"] == 700.0
        assert r["by_trade_totals"] == {"LAB": 600.0, "CARP": 100.0}
        assert "STEEL" not in r["by_trade_totals"]
        assert r["labor_resource_count"] == 3
        assert r["total_assignments_in_schedule"] == 4

    @pytest.mark.asyncio
    async def test_xer_without_resources_activity_count_histogram(self, container):
        import os
        if not os.path.exists(_NORSRC_XER):
            pytest.skip("no-resource fixture missing")
        r = await container.resource_histogram({}, {"schedule_file": _NORSRC_XER})
        assert r["status"] == "success"
        assert r["histogram_kind"] == "activity_count"
        assert r["source"] == "activity_cpm_counts"
        assert "TASKRSRC" in r["labor_unavailable_reason"] or "resource assignment" in r["labor_unavailable_reason"]
        assert r["period_count"] >= 1
        assert r["peak_total"] >= 1
        # Still not a fabricated labor split.
        assert r["by_trade_totals"] == {}
        assert "total_manhours" not in r
        assert "concrete" not in r["by_trade_totals"]

    @pytest.mark.asyncio
    async def test_no_file_errors(self, container):
        r = await container.resource_histogram({}, {})
        assert r["status"] == "error"
        assert "schedule file" in r["error"].lower()

    @pytest.mark.asyncio
    async def test_missing_file_errors(self, container):
        r = await container.resource_histogram(
            {}, {"schedule_file": "tests/fixtures/nope_does_not_exist.xer"})
        assert r["status"] == "error"
        assert "not found" in r["error"].lower()

    @pytest.mark.asyncio
    async def test_non_xer_errors(self, container):
        r = await container.resource_histogram(
            {}, {"schedule_file": "tests/fixtures/sample.txt"})
        assert r["status"] == "error"
        assert ".xer" in r["error"]
