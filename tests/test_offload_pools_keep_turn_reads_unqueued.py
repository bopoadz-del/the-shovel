"""Calculator runs and background jobs never queue ahead of a turn's reads.

Order 0520Z: after #980 moved the calculator run and hydration onto the one
turn pool (sized to the database pool), long workflow asks slowed end to end
while the driver turn itself did not -- work waited for a worker. They now
run in their own pools, and any call that waits for a worker is logged.
"""
import asyncio
import logging
import threading
import time

import pytest

from app.core import offload


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages: list = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@pytest.fixture(autouse=True)
def _fresh_pools(monkeypatch):
    """Each test gets new named pools and an unexpired OFFLOAD_QUEUE rate
    limit: what an earlier test left behind can never decide this one."""
    monkeypatch.setattr(offload, "_last_queue_log", 0.0)
    old = dict(offload._NAMED_POOLS)
    offload._NAMED_POOLS.clear()
    yield
    for pool in offload._NAMED_POOLS.values():
        pool.shutdown(wait=False)
    offload._NAMED_POOLS.clear()
    offload._NAMED_POOLS.update(old)


@pytest.fixture
def offload_log():
    """A handler on the offload logger itself (not the root capture, which an
    earlier test's logging setup can detach), proven live before use: if it
    cannot see a warning, the test fails here instead of passing vacuously."""
    log = offload._LOG
    handler = _Capture()
    saved = (log.disabled, log.level, logging.root.manager.disable)
    log.disabled = False
    log.setLevel(logging.WARNING)
    logging.disable(logging.NOTSET)
    log.addHandler(handler)
    try:
        log.warning("OFFLOAD_CAPTURE_PROBE")
        assert handler.messages == ["OFFLOAD_CAPTURE_PROBE"], "the offload logger's warnings are not captured"
        handler.messages.clear()
        yield handler
    finally:
        log.removeHandler(handler)
        log.disabled, _, _ = saved
        log.setLevel(saved[1])
        logging.disable(saved[2])


def test_the_calculator_pool_is_not_the_turn_pool():
    async def run():
        turn = await offload.off_loop(lambda: threading.current_thread().name)
        calc = await offload.off_loop_in("calc", lambda: threading.current_thread().name)
        back = await offload.off_loop_in("background", lambda: threading.current_thread().name)
        return turn, calc, back

    turn, calc, back = asyncio.run(run())
    assert turn.startswith("turn-io") and calc.startswith("calc-work") and back.startswith("background-work")


def test_a_long_wait_for_a_worker_is_logged(monkeypatch, offload_log):
    monkeypatch.setattr(offload, "_QUEUE_WARN_S", 0.05)
    monkeypatch.setitem(offload._NAMED_SIZES, "background", ("BACKGROUND_WORKERS_TEST_UNSET", 1))

    async def run():
        slow = asyncio.ensure_future(offload.off_loop_in("background", time.sleep, 0.2))
        await asyncio.sleep(0.01)
        await offload.off_loop_in("background", lambda: None)
        await slow

    asyncio.run(run())
    assert any("OFFLOAD_QUEUE pool=background" in m for m in offload_log.messages), offload_log.messages


def test_context_variables_still_round_trip_in_a_named_pool():
    import contextvars

    var = contextvars.ContextVar("probe", default="")

    async def run():
        var.set("caller")
        seen = await offload.off_loop_in("calc", var.get)
        await offload.off_loop_in("calc", var.set, "set in thread")
        return seen, var.get()

    assert asyncio.run(run()) == ("caller", "set in thread")


def _calc_call():
    from app.agents.core.tool_registry import ToolCall

    return ToolCall(agent=None, name="construction_calc",
                    args={"calculation": "productivity_rate", "params": {"output_quantity": 120, "labor_hours": 40}},
                    user_message="productivity of 120 m2 in 40 man-hours")


def test_a_calculator_call_stays_prompt_while_the_background_pool_is_saturated(monkeypatch, offload_log):
    """LANE4: every background worker blocked -> a calculator call still runs
    at once, and no calc OFFLOAD_QUEUE warning is written."""
    from app.agents import runtime
    from app.agents.base import tools
    from app.lib import construction_formulas

    monkeypatch.setattr(runtime, "_defaulted_run_refusal", lambda *a, **k: None)
    monkeypatch.setattr(construction_formulas, "run_calculation",
                        lambda *a, **k: {"status": "success", "result": {"value": 1}})
    monkeypatch.setattr(offload, "_QUEUE_WARN_S", 0.05)
    release = threading.Event()
    size = offload.named_pool("background")._max_workers

    async def run():
        blockers = [asyncio.ensure_future(offload.off_loop_in("background", release.wait, 5)) for _ in range(size + 2)]
        await asyncio.sleep(0.05)
        # The queued background calls may have logged; the calc call gets a
        # fresh rate limit, so a calc wait could not hide behind theirs.
        offload._last_queue_log = 0.0
        started = time.monotonic()
        out = await tools.handle_construction_calc(_calc_call())
        took = time.monotonic() - started
        release.set()
        await asyncio.gather(*blockers)
        return out, took

    out, took = asyncio.run(run())
    assert out["ok"], out
    assert took < 0.5, f"the calculator waited {took:.2f}s behind background work"
    assert not any("OFFLOAD_QUEUE pool=calc" in m for m in offload_log.messages), offload_log.messages


def test_the_calculator_runs_on_calc_threads_and_hydration_on_background_threads(monkeypatch):
    """LANE4: the real entry points, not the pool helper, pick the pools."""
    from app.agents import runtime
    from app.agents.base import tools
    from app.core import hydration_store
    from app.core.learning import hydration
    from app.lib import construction_formulas

    seen: dict = {"calc": [], "hydration": []}

    def calc_probe(*_a, **_k):
        seen["calc"].append(threading.current_thread().name)
        return {"status": "success", "result": {"value": 1}}

    monkeypatch.setattr(runtime, "_defaulted_run_refusal",
                        lambda *a, **k: seen["calc"].append(threading.current_thread().name))
    monkeypatch.setattr(construction_formulas, "run_calculation", calc_probe)
    asyncio.run(tools.handle_construction_calc(_calc_call()))

    def probe(result):
        def fn(*_a, **_k):
            seen["hydration"].append(threading.current_thread().name)
            return result
        return fn

    for name, result in (("_projects_active_in_window", ["p1"]), ("_filter_existing_projects", ["p1"]),
                         ("_collect_project_messages", []), ("_heuristic_project_summary", "summary"),
                         ("_writeback_for_next_chat", {}), ("_heuristic_global_summary", "global")):
        monkeypatch.setattr(hydration, name, probe(result))
    monkeypatch.setattr(hydration_store, "record_run", probe(None))

    async def offline(*_a, **_k):
        return "", "offline_template"

    monkeypatch.setattr(hydration, "_summarize_project", offline)
    monkeypatch.setattr(hydration, "_summarize_global", offline)
    asyncio.run(hydration.run(target_date="2026-10-15"))
    assert seen["calc"] and all(n.startswith("calc-work") for n in seen["calc"]), seen["calc"]
    assert seen["hydration"] and all(n.startswith("background-work") for n in seen["hydration"]), seen["hydration"]
