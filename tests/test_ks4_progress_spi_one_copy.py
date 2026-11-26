"""KS-4 one copy: the progress tracker's SPI is the progress_kpi kernel's.

The progress tracker used to divide actual % by planned % itself and report
1.0 when nothing was planned. It now takes the ratio from
``run_calculation("progress_kpi", ...)``: these tests check that the two agree
on several inputs, that the envelope is really called, that an undefined
ratio stays undefined, and that inputs the envelope refuses are reported
rather than turned into a number.

The tracker and the kernel need no domain kit, so these tests run under every
profile (the virgin profile included).
"""

from __future__ import annotations

import pytest

from app.containers.construction import ConstructionContainer
from app.containers.construction import schedule as schedule_mod
from app.lib import construction_formulas as cf


@pytest.fixture
def container():
    return ConstructionContainer()


async def _overall(container, planned, actual):
    out = await container.progress_tracker(
        input_data={}, params={"planned_percent": planned, "actual_percent": actual},
    )
    assert out["status"] == "success", out
    return out["overall_progress"]


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


@pytest.mark.asyncio
@pytest.mark.parametrize("actual", [0.0, 5.0])
async def test_nothing_planned_is_undefined_not_on_track(container, actual):
    overall = await _overall(container, 0.0, actual)
    assert overall["schedule_performance_index"] is None
    assert "schedule_performance_index_error" not in overall


@pytest.mark.asyncio
@pytest.mark.parametrize("planned,actual", [(-10.0, 5.0), (50.0, -1.0)])
async def test_refused_inputs_are_reported_not_divided(container, planned, actual):
    overall = await _overall(container, planned, actual)
    assert overall["schedule_performance_index"] is None
    assert overall["schedule_performance_index_error"]


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


@pytest.mark.parametrize("planned,actual,want", [(80.0, 60.0, 0.75), (10.0, 25.0, 2.5)])
def test_progress_spi_helper_returns_the_kernel_ratio(planned, actual, want):
    spi, err = schedule_mod._progress_spi(planned, actual)
    assert err is None
    assert spi == pytest.approx(want, abs=1e-3)


@pytest.mark.parametrize("actual", [0.0, 12.0])
def test_progress_spi_helper_nothing_planned_is_none_without_error(actual):
    assert schedule_mod._progress_spi(0.0, actual) == (None, None)


@pytest.mark.parametrize("envelope", [None, "not an envelope", {"status": "error"}])
def test_progress_spi_helper_reports_a_non_success_envelope(monkeypatch, envelope):
    monkeypatch.setattr(cf, "run_calculation", lambda name, params=None: envelope)
    spi, err = schedule_mod._progress_spi(50.0, 40.0)
    assert spi is None
    assert err == "progress_kpi did not return a result"


@pytest.mark.parametrize("message", ["planned_cum out of range", "actual_cum missing"])
def test_progress_spi_helper_passes_the_envelope_error_through(monkeypatch, message):
    monkeypatch.setattr(
        cf, "run_calculation",
        lambda name, params=None: {"status": "error", "error": message},
    )
    assert schedule_mod._progress_spi(50.0, 40.0) == (None, message)


@pytest.mark.asyncio
@pytest.mark.parametrize("input_data,params", [
    ({}, {}),
    ({"contract_value": 900000.0}, {"reporting_period": "March"}),
])
async def test_tracker_with_no_figures_is_an_error_not_a_ratio(container, input_data, params):
    out = await container.progress_tracker(input_data=input_data, params=params)
    assert out["status"] == "error"
    assert "overall_progress" not in out


@pytest.mark.asyncio
@pytest.mark.parametrize("params,want_spi", [
    ({"planned_percent": 40.0}, 0.0),
    ({"actual_percent": 30.0}, None),
])
async def test_tracker_with_one_figure_still_answers(container, params, want_spi):
    out = await container.progress_tracker(input_data={}, params=params)
    assert out["status"] == "success", out
    assert out["overall_progress"]["schedule_performance_index"] == want_spi


# --- Inside a chat turn: the turn's ask is set (as app.agents.core.loop sets it)
# and the tracker's own figures must still reach the kernel unchanged.


@pytest.fixture
def turn_ask():
    # An async test sets the ask in its own task's context, so the token cannot
    # be reset here; the ask is set back to the module default instead.
    yield cf.set_current_ask
    cf.set_current_ask("")


@pytest.mark.asyncio
@pytest.mark.parametrize("ask,planned,actual", [
    ("Track progress: planned 40%, nothing done", 40.0, 0.0),
    ("Where are we? 55 percent planned by now, the crew has not started", 55.0, 0.0),
])
async def test_in_a_turn_a_zero_actual_is_an_spi_of_zero(container, turn_ask, ask, planned, actual):
    turn_ask(ask)
    overall = await _overall(container, planned, actual)
    assert overall["schedule_performance_index"] == 0.0
    assert "schedule_performance_index_error" not in overall


@pytest.mark.asyncio
@pytest.mark.parametrize("ask,actual", [
    ("Track progress: planned 0%, actual 5%", 5.0),
    ("Nothing was planned this month but we did 12%", 12.0),
])
async def test_in_a_turn_nothing_planned_is_undefined_without_error(container, turn_ask, ask, actual):
    turn_ask(ask)
    overall = await _overall(container, 0.0, actual)
    assert overall["schedule_performance_index"] is None
    assert "schedule_performance_index_error" not in overall


@pytest.mark.asyncio
@pytest.mark.parametrize("ask,planned,actual", [
    ("Planned 60%, actual 45%, variance -15%", 60.0, 45.0),
    ("Track progress, planned 0 and actual 0", 0.0, 0.0),
])
async def test_the_kernel_runs_without_the_turn_ask_and_the_ask_comes_back(
        container, turn_ask, monkeypatch, ask, planned, actual):
    seen = []
    real = cf.run_calculation

    def spy(name, params=None):
        seen.append(cf._CURRENT_ASK.get())
        return real(name, params)

    monkeypatch.setattr(cf, "run_calculation", spy)
    turn_ask(ask)
    await _overall(container, planned, actual)
    assert seen == [""]
    assert cf._CURRENT_ASK.get() == ask


@pytest.mark.parametrize("ask", ["planned 30%, actual 10%", "Track progress please"])
def test_the_turn_ask_comes_back_when_the_kernel_raises(turn_ask, monkeypatch, ask):
    def boom(name, params=None):
        raise RuntimeError("kernel down")

    monkeypatch.setattr(cf, "run_calculation", boom)
    turn_ask(ask)
    with pytest.raises(RuntimeError):
        schedule_mod._progress_spi(30.0, 10.0)
    assert cf._CURRENT_ASK.get() == ask


@pytest.mark.asyncio
@pytest.mark.parametrize("ask,planned,actual", [
    ("planned -10%, actual 5%", -10.0, 5.0),
    ("planned 50%, actual minus 1%", 50.0, -1.0),
])
async def test_in_a_turn_a_refusal_names_no_internal_input(container, turn_ask, ask, planned, actual):
    turn_ask(ask)
    overall = await _overall(container, planned, actual)
    assert overall["schedule_performance_index"] is None
    err = overall["schedule_performance_index_error"]
    assert err and "_cum" not in err and "Expected:" not in err


@pytest.mark.parametrize("envelope", [{"status": "success"}, {"status": "success", "result": None}])
def test_progress_spi_helper_success_without_a_result_is_absent(monkeypatch, envelope):
    monkeypatch.setattr(cf, "run_calculation", lambda name, params=None: envelope)
    assert schedule_mod._progress_spi(50.0, 40.0) == (None, None)
