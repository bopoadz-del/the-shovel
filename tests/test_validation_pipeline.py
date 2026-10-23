"""Tests for app.blocks.validation_pipeline.ValidationPipelineBlock.

Deterministic unit tests for the 5-stage validation pipeline using small JSON
inputs and no external services.
"""

import math

import pytest

from app.blocks.validation_pipeline import ValidationPipelineBlock
from tests._dev_key import DEV_KEY


@pytest.fixture
def block():
    return ValidationPipelineBlock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
async def _run(block, payload):
    result = await block.process(payload)
    assert result.get("status") == "success"
    return result


def _stage(result, name):
    return result["stages"][name]


# ---------------------------------------------------------------------------
# Stage 1: syntactic
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_syntactic_valid_number_passes(block):
    result = await _run(block, {"value": 5.9, "context": {"metric": "temperature_degc"}})
    assert _stage(result, "syntactic")["pass"] is True


@pytest.mark.asyncio
async def test_syntactic_none_fails_and_short_circuits(block):
    result = await _run(block, {"value": None, "context": {"metric": "temperature_degc"}})
    assert result["overall"] == "fail"
    assert result["first_failure"] == "syntactic"
    assert "None" in _stage(result, "syntactic")["reason"]
    assert "skipped" in _stage(result, "physical")["reason"]


@pytest.mark.asyncio
async def test_prose_claim_without_value_runs_physical(block):
    """Leftover-hat L4: worded 40 m / 50 mm beam must fail Physical, not skip."""
    result = await _run(block, {
        "value": None,
        "claim": (
            "forty metres span on a fifty millimetre steel section "
            "with eight hundred kilonewtons per metre"
        ),
    })
    assert _stage(result, "syntactic")["pass"] is True
    assert _stage(result, "physical")["pass"] is False
    assert result["first_failure"] == "physical"
    assert result["overall"] == "fail"
    assert result.get("tier") == 4


@pytest.mark.asyncio
async def test_syntactic_bool_fails(block):
    result = await _run(block, {"value": True})
    assert result["first_failure"] == "syntactic"


@pytest.mark.asyncio
async def test_syntactic_non_numeric_string_fails(block):
    result = await _run(block, {"value": "hello"})
    assert result["first_failure"] == "syntactic"


@pytest.mark.asyncio
async def test_syntactic_nan_fails(block):
    result = await _run(block, {"value": float("nan")})
    assert result["first_failure"] == "syntactic"
    assert "non-finite" in _stage(result, "syntactic")["reason"]


@pytest.mark.asyncio
async def test_syntactic_inf_fails(block):
    result = await _run(block, {"value": float("inf")})
    assert result["first_failure"] == "syntactic"


@pytest.mark.asyncio
async def test_syntactic_string_numeric_coerces_and_passes(block):
    result = await _run(block, {"value": "  42.5 ", "context": {"metric": "temperature_degc"}})
    assert _stage(result, "syntactic")["pass"] is True
    assert result["value"] == 42.5


# ---------------------------------------------------------------------------
# Stage 2: dimensional
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_dimensional_missing_unit_is_skipped(block):
    result = await _run(block, {"value": 5.9, "context": {"metric": "temperature_degc"}})
    assert _stage(result, "dimensional")["pass"] is True
    assert "skipped" in _stage(result, "dimensional")["reason"]


@pytest.mark.asyncio
async def test_dimensional_valid_unit_parsed(block):
    result = await _run(block, {"value": 5.9, "unit": "m3", "context": {"material_type": "concrete"}})
    assert _stage(result, "dimensional")["pass"] is True
    assert result["metric_inferred"] == "volume_m3"


@pytest.mark.asyncio
async def test_dimensional_unrecognised_unit_fails(block):
    result = await _run(block, {"value": 5.9, "unit": "gibberish_unit"})
    assert _stage(result, "dimensional")["pass"] is False
    assert result["first_failure"] == "dimensional"


@pytest.mark.asyncio
async def test_dimensional_currency_unit_stripped(block):
    result = await _run(block, {"value": 100, "unit": "USD/m3", "context": {"material_type": "concrete"}})
    assert _stage(result, "dimensional")["pass"] is True


@pytest.mark.asyncio
async def test_dimensional_currency_only_unit_skipped(block):
    result = await _run(block, {"value": 100, "unit": "USD"})
    assert _stage(result, "dimensional")["pass"] is True
    assert "currency-only" in _stage(result, "dimensional")["reason"]


@pytest.mark.asyncio
async def test_dimensional_degC_offset_parsed(block):
    result = await _run(block, {"value": 5.9, "unit": "degC", "context": {"material_type": "concrete"}})
    assert _stage(result, "dimensional")["pass"] is True


# ---------------------------------------------------------------------------
# Stage 3: physical
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_physical_default_bounds_non_negative(block):
    result = await _run(block, {"value": -1, "context": {"metric": "volume_m3", "material_type": "concrete"}})
    assert _stage(result, "physical")["pass"] is False
    assert "below physical_min" in _stage(result, "physical")["reason"]


@pytest.mark.asyncio
async def test_physical_negative_temperature_allowed(block):
    result = await _run(block, {"value": -20, "unit": "degC", "context": {"material_type": "concrete"}})
    assert _stage(result, "physical")["pass"] is True


@pytest.mark.asyncio
async def test_physical_custom_bounds(block):
    result = await _run(block, {"value": 150, "context": {"metric": "temperature_degc", "physical_min": 0, "physical_max": 100}})
    assert _stage(result, "physical")["pass"] is False
    assert "above physical_max" in _stage(result, "physical")["reason"]


@pytest.mark.asyncio
async def test_physical_default_ceiling(block):
    result = await _run(block, {"value": 1e31, "context": {"metric": "volume_m3", "material_type": "concrete"}})
    assert _stage(result, "physical")["pass"] is False


# ---------------------------------------------------------------------------
# Stage 4: empirical
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_empirical_concrete_rate_in_range_passes(block):
    result = await _run(block, {
        "value": 200,
        "unit": "USD/m3",
        "context": {"material_type": "concrete", "metric": "rate_usd_per_m3"},
    })
    assert _stage(result, "empirical")["pass"] is True


@pytest.mark.asyncio
async def test_empirical_sar_currency_lookup(block):
    result = await _run(block, {
        "value": 1000,
        "unit": "SAR/m3",
        "context": {"material_type": "concrete", "currency": "SAR"},
    })
    assert _stage(result, "empirical")["pass"] is True


@pytest.mark.asyncio
async def test_empirical_out_of_range_fails(block):
    result = await _run(block, {
        "value": 1_000_000,
        "unit": "USD/m3",
        "context": {"material_type": "concrete", "metric": "rate_usd_per_m3"},
    })
    assert _stage(result, "empirical")["pass"] is False
    assert result["first_failure"] == "empirical"


@pytest.mark.asyncio
async def test_empirical_borderline_flagged(block):
    result = await _run(block, {
        "value": 700,
        "unit": "USD/m3",
        "context": {"material_type": "concrete", "metric": "rate_usd_per_m3"},
    })
    assert _stage(result, "empirical")["pass"] is True
    assert _stage(result, "empirical").get("borderline") is True
    assert result.get("borderline") is True


@pytest.mark.asyncio
async def test_empirical_strict_mode_rejects_borderline(block):
    result = await _run(block, {
        "value": 700,
        "unit": "USD/m3",
        "context": {"material_type": "concrete", "metric": "rate_usd_per_m3", "strict": True},
    })
    assert _stage(result, "empirical")["pass"] is False
    assert "strict" in _stage(result, "empirical")["reason"]


@pytest.mark.asyncio
async def test_empirical_explicit_range_overrides_file(block):
    result = await _run(block, {
        "value": 999,
        "context": {"metric": "temperature_degc", "empirical_min": 0, "empirical_max": 100},
    })
    assert _stage(result, "empirical")["pass"] is False
    assert "empirical range [0, 100]" in _stage(result, "empirical")["reason"]


@pytest.mark.asyncio
async def test_empirical_slack_factor_clamped_below_one(block):
    result = await _run(block, {
        "value": 600,
        "unit": "USD/m3",
        "context": {"material_type": "concrete", "metric": "rate_usd_per_m3", "slack_factor": 0.5},
    })
    # slack_factor < 1 is clamped to 1.0, so 600 is within [500, 500] -> actually 600 fails.
    assert _stage(result, "empirical")["pass"] is False


@pytest.mark.asyncio
async def test_empirical_range_spanning_zero_uses_additive_slack(block):
    result = await _run(block, {
        "value": -80,
        "unit": "degC",
        "context": {"metric": "temperature_degc", "empirical_min": -40, "empirical_max": 100},
    })
    # Range [-40, 100] with 2x slack -> [-250, 310]; -80 is inside slack.
    assert _stage(result, "empirical")["pass"] is True
    assert _stage(result, "empirical").get("borderline") is True


@pytest.mark.asyncio
async def test_empirical_no_range_available_is_skipped(block):
    result = await _run(block, {"value": 123, "context": {"material_type": "foo", "metric": "bar"}})
    assert _stage(result, "empirical")["pass"] is True
    assert "skipped" in _stage(result, "empirical")["reason"]


# ---------------------------------------------------------------------------
# Stage 5: operational
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_operational_missing_fields_skipped(block):
    result = await _run(block, {"value": 5.9, "context": {"metric": "temperature_degc"}})
    assert _stage(result, "operational")["pass"] is True
    assert "skipped" in _stage(result, "operational")["reason"]


@pytest.mark.asyncio
async def test_operational_duration_fits_available(block):
    result = await _run(block, {"value": 10, "context": {"duration_weeks": 8, "available_weeks": 10}})
    assert _stage(result, "operational")["pass"] is True


@pytest.mark.asyncio
async def test_operational_duration_exceeds_available_fails(block):
    result = await _run(block, {"value": 10, "context": {"duration_weeks": 16, "available_weeks": 8}})
    assert _stage(result, "operational")["pass"] is False
    assert "exceeds" in _stage(result, "operational")["reason"]
    assert result["first_failure"] == "operational"


@pytest.mark.asyncio
async def test_operational_non_numeric_fields_fail(block):
    result = await _run(block, {"value": 10, "context": {"duration_weeks": "many", "available_weeks": 8}})
    assert _stage(result, "operational")["pass"] is False


# ---------------------------------------------------------------------------
# Metric inference
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_infer_metric_kg_steel(block):
    result = await _run(block, {"value": 500, "unit": "kg", "context": {"material_type": "steel"}})
    assert result["metric_inferred"] == "weight_kg"


@pytest.mark.asyncio
async def test_infer_metric_mpa_concrete(block):
    result = await _run(block, {"value": 35, "unit": "MPa", "context": {"material_type": "concrete"}})
    assert result["metric_inferred"] == "compressive_mpa"


@pytest.mark.asyncio
async def test_infer_metric_percent(block):
    result = await _run(block, {"value": 75, "unit": "%"})
    assert result["metric_inferred"] == "percent"


# ---------------------------------------------------------------------------
# Overall / integration
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_empty_input_returns_error(block):
    result = await block.process({})
    assert result.get("status") == "error"
    assert "Provide" in result.get("error", "")


@pytest.mark.asyncio
async def test_params_promoted_to_context(block):
    result = await _run(block, {
        "value": 200,
        "unit": "USD/m3",
        "material_type": "concrete",
        "metric": "rate_usd_per_m3",
    })
    assert result["overall"] == "pass"


@pytest.mark.asyncio
async def test_first_failure_order(block):
    # Multiple failures: syntactic wins, then dimensional, physical, empirical, operational.
    result = await _run(block, {"value": "bad", "unit": "bad_unit"})
    assert result["first_failure"] == "syntactic"

# ---------------------------------------------------------------------------
# Production regression coverage: unit parsing must never crash
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_dimensional_value_only_still_passes(block):
    result = await _run(block, {"value": 120})
    assert _stage(result, "syntactic")["pass"] is True
    assert _stage(result, "dimensional")["pass"] is True


@pytest.mark.asyncio
async def test_dimensional_m3_does_not_crash(block):
    result = await _run(block, {"value": 120, "unit": "m3"})
    assert result["status"] == "success"
    assert _stage(result, "dimensional")["pass"] is True


@pytest.mark.asyncio
async def test_dimensional_unicode_cube_does_not_crash(block):
    result = await _run(block, {"value": 120, "unit": "m³"})
    assert result["status"] == "success"
    assert _stage(result, "dimensional")["pass"] is True


@pytest.mark.asyncio
async def test_dimensional_invalid_unit_returns_controlled_failure(block):
    result = await _run(block, {"value": 120, "unit": "gibberish_xyz"})
    assert result["status"] == "success"
    assert _stage(result, "dimensional")["pass"] is False
    assert "not recognised" in _stage(result, "dimensional")["reason"]


@pytest.mark.asyncio
async def test_dimensional_graceful_when_ureg_raises(monkeypatch, block):
    """If Pint/UnitRegistry fails internally, the stage must not propagate."""
    import app.blocks.validation_pipeline as vp

    def _broken_ureg():
        raise SystemError("simulated Pint failure")

    monkeypatch.setattr(vp, "_get_ureg", _broken_ureg)
    result = await block.process({"value": 120, "unit": "m3"})
    assert result["status"] == "success"
    assert _stage(result, "dimensional")["pass"] is True
    assert "unavailable" in _stage(result, "dimensional")["reason"]


@pytest.mark.asyncio
async def test_execute_validation_pipeline_unit_returns_non_500():
    """POST /v1/execute with validation_pipeline + unit must not crash."""
    from httpx import ASGITransport, AsyncClient
    from app.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post(
            "/v1/execute",
            json={
                "block": "validation_pipeline",
                "input": {"value": 120, "unit": "m3"},
                "params": {},
            },
            headers={"Authorization": "Bearer " + DEV_KEY},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("status") != "error"
    result = body.get("result", body)
    assert result.get("status") == "success"
