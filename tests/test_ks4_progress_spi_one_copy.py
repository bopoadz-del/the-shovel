"""KS-4 one copy: the progress tracker's SPI is the progress_kpi kernel's.

The progress tracker used to divide actual % by planned % itself and report
1.0 when nothing was planned. It now takes the ratio from
``run_calculation("progress_kpi", ...)``: these tests check that the two agree
on several inputs, that the envelope is really called, that an undefined
ratio stays undefined, and that inputs the envelope refuses are reported
rather than turned into a number.
"""

from __future__ import annotations

import pytest

from app.containers.construction import ConstructionContainer
from app.lib import construction_formulas as cf
from tests.conftest import requires_construction_kit


@pytest.fixture
def container():
    return ConstructionContainer()


async def _overall(container, planned, actual):
    out = await container.progress_tracker(
        input_data={}, params={"planned_percent": planned, "actual_percent": actual},
    )
    assert out["status"] == "success", out
    return out["overall_progress"]


@requires_construction_kit
@pytest.mark.asyncio
@pytest.mark.parametrize("planned,actual,want", [
    (60.0, 45.0, 0.75),
    (40.0, 50.0, 1.25),
    (33.0, 10.0, 0.303),
])
async def test_tracker_spi_equals_kernel(container, planned, actual, want):
    overall = await _overall(container, planned, actual)
    kernel = cf.run_calculation(
        "progress_kpi", {"planned_cum": planned, "actual_cum": actual},
    )
    assert kernel["status"] == "success", kernel
    assert overall["schedule_performance_index"] == kernel["result"]["spi_progress"]
    assert overall["schedule_performance_index"] == pytest.approx(want, abs=1e-3)
    assert "schedule_performance_index_error" not in overall


@requires_construction_kit
@pytest.mark.asyncio
@pytest.mark.parametrize("planned,actual", [(55.0, 44.0), (20.0, 30.0)])
async def test_tracker_goes_through_the_envelope(container, monkeypatch, planned, actual):
    calls = []
    real = cf.run_calculation

    def spy(name, params=None):
        calls.append((name, dict(params or {})))
        return real(name, params)

    monkeypatch.setattr(cf, "run_calculation", spy)
    await _overall(container, planned, actual)
    assert ("progress_kpi", {"planned_cum": planned, "actual_cum": actual}) in calls


@requires_construction_kit
@pytest.mark.asyncio
@pytest.mark.parametrize("actual", [0.0, 5.0])
async def test_nothing_planned_is_undefined_not_on_track(container, actual):
    overall = await _overall(container, 0.0, actual)
    assert overall["schedule_performance_index"] is None
    assert "schedule_performance_index_error" not in overall


@requires_construction_kit
@pytest.mark.asyncio
@pytest.mark.parametrize("planned,actual", [(-10.0, 5.0), (50.0, -1.0)])
async def test_refused_inputs_are_reported_not_divided(container, planned, actual):
    overall = await _overall(container, planned, actual)
    assert overall["schedule_performance_index"] is None
    assert overall["schedule_performance_index_error"]


@requires_construction_kit
@pytest.mark.asyncio
@pytest.mark.parametrize("planned,actual", [(50.0, 40.0), (25.0, 30.0)])
async def test_kernel_failure_is_reported(container, monkeypatch, planned, actual):
    def broken(name, params=None):
        return {"status": "error", "error": f"refused {name}"}

    monkeypatch.setattr(cf, "run_calculation", broken)
    overall = await _overall(container, planned, actual)
    assert overall["schedule_performance_index"] is None
    assert overall["schedule_performance_index_error"] == "refused progress_kpi"
    # The rest of the tracker still answers from the stated figures.
    assert overall["variance_percent"] == pytest.approx(actual - planned)
