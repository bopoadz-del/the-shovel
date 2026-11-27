"""The process-wide loop watchdog ends with the lifespan that started it.

Before: ``loop_watchdog.start()`` ran from the app lifespan and nothing
stopped it. Its threads lived until they found the loop closed, reading CPU
through the module-level ``cpu_snapshot`` all the while (main tests run
45386234062, ea2df39f: a test's fake of that function counted a second
monitor's readings), and ``_started`` stayed pinned to the dead monitor, so a
later lifespan in the same process got no watchdog.
"""
import asyncio

import pytest

from app.core import loop_watchdog


def _threads_alive(mon) -> bool:
    return mon._thread.is_alive() or mon._snap_thread.is_alive()


@pytest.fixture
def fresh(monkeypatch):
    monkeypatch.setattr(loop_watchdog, "_started", None)
    monkeypatch.setenv("LOOP_WATCHDOG_MS", "250")
    monkeypatch.setenv("LOOP_WATCHDOG_TICK_MS", "10")
    yield
    leftover = loop_watchdog._started
    if leftover is not None:
        leftover.stop()


@pytest.mark.parametrize("running", [True, False])
def test_stop_ends_the_running_monitor_and_forgets_it(fresh, running):
    async def run():
        mon = loop_watchdog.start() if running else None
        if running:
            await asyncio.sleep(0.05)
            assert mon is not None and _threads_alive(mon)
        await asyncio.to_thread(loop_watchdog.stop)
        return mon

    mon = asyncio.run(run())
    assert loop_watchdog._started is None
    if running:
        assert not _threads_alive(mon)
        assert mon._stop.is_set()


def test_a_second_lifespan_gets_its_own_watchdog_on_its_own_loop(fresh):
    async def one_lifespan():
        mon = loop_watchdog.start()
        loop = asyncio.get_running_loop()
        await asyncio.sleep(0.05)
        await asyncio.to_thread(loop_watchdog.stop)
        return mon, loop

    first, first_loop = asyncio.run(one_lifespan())
    second, second_loop = asyncio.run(one_lifespan())
    assert first is not None and second is not None
    assert second is not first
    assert first.loop is first_loop and second.loop is second_loop
    assert not _threads_alive(first) and not _threads_alive(second)


@pytest.mark.parametrize("threshold", ["250", "0"])
def test_the_app_lifespan_shutdown_stops_the_watchdog(fresh, monkeypatch, threshold):
    from fastapi.testclient import TestClient

    from app.main import app

    monkeypatch.setenv("LOOP_WATCHDOG_MS", threshold)
    with TestClient(app):
        mon = loop_watchdog._started
        if threshold == "0":
            assert mon is None
        else:
            assert mon is not None and _threads_alive(mon)
    assert loop_watchdog._started is None
    if mon is not None:
        assert not _threads_alive(mon)
