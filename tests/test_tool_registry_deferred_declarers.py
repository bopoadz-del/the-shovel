"""A deferred result declarer is never lost and never half-read.

It leaves the waiting list only after it succeeds (a failure is logged by its
class and retried on the next read), and one lock covers the loop so a second
thread's first read waits for the declarations. Every test builds its own
invented declarer; none relies on what an earlier test already drained.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from app.agents.core import tool_registry


@pytest.fixture
def invented():
    names: list[str] = []
    declarers: list = []

    def make(name, body):
        names.append(name)
        declarers.append(body)
        tool_registry.declare_results_when_read(body)
        return body

    yield make
    for n in names:
        tool_registry._RESULTS.pop(n, None)
    for d in declarers:
        if d in tool_registry._DEFERRED:
            tool_registry._DEFERRED.remove(d)
        tool_registry._RETRY_AT.pop(d, None)


class _FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def past_the_back_off(self):
        self.now += tool_registry.DEFERRED_RETRY_INTERVAL_S + 0.001


@pytest.fixture
def clock(monkeypatch):
    fake = _FakeClock()
    monkeypatch.setattr(tool_registry, "_clock", fake)
    return fake


def _declare(name, kind="search"):
    tool_registry.declare_result(name, owner="base", display_name="Invented", kind=kind)


@pytest.mark.parametrize("name, error", [
    ("zz_flaky_declarer_a", RuntimeError),
    ("zz_flaky_declarer_b", ImportError),
])
def test_a_declarer_that_fails_once_is_retried_and_then_declares(invented, clock, caplog, name, error):
    calls = []

    def declarer():
        calls.append(1)
        if len(calls) == 1:
            raise error("secret detail " + name)
        _declare(name)

    invented(name, declarer)
    with caplog.at_level(logging.WARNING, logger=tool_registry.__name__):
        assert tool_registry.result_spec(name) is None
    assert declarer in tool_registry._DEFERRED
    logged = " ".join(r.getMessage() for r in caplog.records)
    assert error.__name__ in logged and "secret detail" not in logged

    clock.past_the_back_off()
    assert tool_registry.result_spec(name).kind == "search"
    assert len(calls) == 2
    assert declarer not in tool_registry._DEFERRED
    tool_registry.result_spec(name)
    assert len(calls) == 2


@pytest.mark.parametrize("failing, healthy", [
    ("zz_fail_first_a", "zz_ok_second_a"),
    ("zz_fail_first_b", "zz_ok_second_b"),
])
def test_one_failing_declarer_does_not_stop_the_others(invented, failing, healthy):
    def bad():
        raise ValueError(failing)

    invented(failing, bad)
    invented(healthy, lambda: _declare(healthy))
    assert healthy in tool_registry.tools_of_kind("search")
    assert bad in tool_registry._DEFERRED


@pytest.mark.parametrize("name", ["zz_slow_declarer_a", "zz_slow_declarer_b"])
def test_a_second_threads_first_read_waits_for_the_declarations(invented, name):
    started, release = threading.Event(), threading.Event()
    runs = []

    def declarer():
        runs.append(1)
        started.set()
        release.wait(10)
        _declare(name)

    invented(name, declarer)
    seen = {}
    first = threading.Thread(target=lambda: seen.setdefault("first", tool_registry.result_spec(name)))
    second = threading.Thread(target=lambda: seen.setdefault("second", tool_registry.result_spec(name)))
    first.start()
    assert started.wait(10)
    second.start()
    time.sleep(0.2)          # the second read is now waiting on the lock
    assert "second" not in seen
    release.set()
    first.join(10)
    second.join(10)
    assert seen["first"].kind == "search" and seen["second"].kind == "search"
    assert runs == [1]


@pytest.mark.parametrize("name", ["zz_reentrant_declarer_a", "zz_reentrant_declarer_b"])
def test_a_read_from_inside_a_declarer_does_not_rerun_it(invented, name):
    runs = []

    def declarer():
        runs.append(1)
        tool_registry.tools_of_kind("search")   # a read made while declaring
        _declare(name)

    invented(name, declarer)
    assert tool_registry.result_spec(name).kind == "search"
    assert runs == [1]


def test_a_calculator_reads_computed_in_a_fresh_process():
    """No earlier test in this process can have drained the deferred list:
    a new interpreter imports the app and reads one calculator's kind."""
    code = (
        "import app.main\n"
        "from app.agents.core import tool_registry as r\n"
        "from app.agents.runtime import _HAT_CALC_ALIASES\n"
        "assert r._DEFERRED, 'the declarations must still be waiting before the first read'\n"
        "for action in (sorted(_HAT_CALC_ALIASES)[0], sorted(_HAT_CALC_ALIASES)[-1]):\n"
        "    assert r.result_spec(action).kind == 'computed', action\n"
        "assert not r._DEFERRED\n"
    )
    env = dict(os.environ)
    env.setdefault("ENV", "development")
    env.setdefault("RAG_EMBEDDING_MODEL", "fake")
    subprocess.run([sys.executable, "-c", code], check=True, timeout=300,
                   cwd=str(Path(__file__).resolve().parents[1]), env=env)


class _CountingLock:
    """Stands in for the declarers' lock and counts every acquisition."""

    def __init__(self):
        self._inner = threading.RLock()
        self.taken = 0

    def __enter__(self):
        self.taken += 1
        return self._inner.__enter__()

    def __exit__(self, *exc):
        return self._inner.__exit__(*exc)


@pytest.mark.parametrize("read, name", [
    (lambda n: tool_registry.result_spec(n), "zz_lockfree_read_a"),
    (lambda n: tool_registry.tools_of_kind("search"), "zz_lockfree_read_b"),
])
def test_once_nothing_is_pending_a_read_never_takes_the_lock(invented, monkeypatch, read, name):
    """Every tool call reads the declarations: after they are declared a read
    must not queue on the lock (fifteen users would serialise on it)."""
    lock = _CountingLock()
    monkeypatch.setattr(tool_registry, "_DEFERRED_LOCK", lock)
    invented(name, lambda: _declare(name))
    read(name)                                  # pending: the lock is taken once
    assert lock.taken == 1 and not tool_registry._DEFERRED
    for _ in range(5):
        read(name)
    assert lock.taken == 1


@pytest.mark.parametrize("name", ["zz_warmup_declarer_a", "zz_warmup_declarer_b"])
def test_the_startup_warm_up_declares_off_the_event_loop(invented, monkeypatch, name):
    """The app declares at warm-up, in a worker thread, so the first read
    under load finds nothing pending and runs no import on the loop."""
    import asyncio

    from app import main

    for step in ("_warm_safety_detector", "_warm_embedder", "_check_embedder_identity",
                 "_seed_knowledge", "_warm_parse_libraries"):
        monkeypatch.setattr(main, step, lambda: None)
    threads = []

    def declarer():
        threads.append(threading.get_ident())
        _declare(name)

    invented(name, declarer)

    async def run():
        loop_thread = threading.get_ident()
        await main._warm_models()
        return loop_thread

    loop_thread = asyncio.run(run())
    assert threads and threads[0] != loop_thread
    assert declarer not in tool_registry._DEFERRED
    assert tool_registry._RESULTS[name].kind == "search"


@pytest.mark.parametrize("names", [("zz_direct_warm_a",), ("zz_direct_warm_b", "zz_direct_warm_c")])
def test_the_app_warm_step_drains_every_pending_declarer(invented, names):
    from app import main

    declarers = [invented(n, lambda n=n: _declare(n)) for n in names]
    main._declare_deferred_results()
    for n in names:
        assert tool_registry._RESULTS[n].kind == "search"
    assert not any(d in tool_registry._DEFERRED for d in declarers)


_WARM_STEPS = ("_declare_deferred_results", "_warm_safety_detector", "_warm_embedder",
               "_check_embedder_identity", "_seed_knowledge", "_warm_parse_libraries")


@pytest.mark.parametrize("blocked_at", [0, 2])
def test_cancelling_the_warm_up_stops_it_before_the_next_step(monkeypatch, blocked_at):
    """At shutdown the warm-up task is cancelled: it must end cancelled and
    must not go on loading the remaining models."""
    import asyncio

    from app import main

    ran, entered, release = [], threading.Event(), threading.Event()

    def make(i):
        def step():
            ran.append(i)
            if i == blocked_at:
                entered.set()
                release.wait(10)
        return step

    for i, step in enumerate(_WARM_STEPS):
        monkeypatch.setattr(main, step, make(i))

    async def run():
        task = asyncio.create_task(main._warm_models())
        while not entered.is_set():
            await asyncio.sleep(0.01)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            return True
        return False

    try:
        assert asyncio.run(run()) is True
    finally:
        release.set()
    assert ran == list(range(blocked_at + 1))


# --- a declarer that fails part-way, back-off, and reads on the event loop ---

@pytest.mark.parametrize("first, second, error", [
    ("zz_partial_first_a", "zz_partial_second_a", RuntimeError),
    ("zz_partial_first_b", "zz_partial_second_b", ImportError),
])
def test_a_declarer_that_fails_after_its_first_name_declares_both_on_retry(
        invented, clock, first, second, error):
    """Nothing of a failed run is committed, so the retry does not hit
    'declared twice' on the name the failed run had already declared."""
    calls = []

    def declarer():
        calls.append(1)
        # A computed result's statement is built per run (as a calculator's
        # is): a new function, so a second run's spec is not equal to the first.
        tool_registry.declare_result(first, owner="base", display_name="Invented",
                                     kind="computed", statement=lambda inputs: [])
        if len(calls) == 1:
            raise error("after the first name")
        _declare(second)

    invented(first, declarer)
    invented(second, lambda: None)      # registers the name for clean-up only
    assert tool_registry.result_spec(first) is None
    assert first not in tool_registry._RESULTS and declarer in tool_registry._DEFERRED

    clock.past_the_back_off()
    assert tool_registry.result_spec(first).kind == "computed"
    assert tool_registry.result_spec(second).kind == "search"
    assert len(calls) == 2 and declarer not in tool_registry._DEFERRED


@pytest.mark.parametrize("name, reads", [("zz_always_fails_a", 3), ("zz_always_fails_b", 7)])
def test_a_declarer_that_always_fails_is_retried_once_per_interval(
        invented, clock, monkeypatch, name, reads):
    """Between retries a read returns what is declared and never takes the
    lock: a broken heavy import must not be paid by every tool call."""
    lock = _CountingLock()
    monkeypatch.setattr(tool_registry, "_DEFERRED_LOCK", lock)
    calls = []

    def declarer():
        calls.append(1)
        raise RuntimeError(name)

    invented(name, declarer)
    for _ in range(reads):
        assert tool_registry.result_spec(name) is None
        assert name not in tool_registry.tools_of_kind("search")
    assert len(calls) == 1 and lock.taken == 1

    clock.now += tool_registry.DEFERRED_RETRY_INTERVAL_S / 2       # not yet due
    tool_registry.result_spec(name)
    assert len(calls) == 1 and lock.taken == 1

    clock.past_the_back_off()
    tool_registry.result_spec(name)
    assert len(calls) == 2 and lock.taken == 2
    assert declarer in tool_registry._DEFERRED


def _loop_lag_while(read, hold_s):
    """Run ``read`` on the event loop every 20 ms for ``hold_s`` seconds under
    the loop watchdog; return the worst lag (ms) the watchdog measured."""
    import asyncio

    from app.core.loop_watchdog import LagMonitor

    async def run():
        monitor = LagMonitor(asyncio.get_running_loop(), threshold_ms=250, tick_ms=20,
                             log=False).start()
        try:
            end = time.monotonic() + hold_s
            while time.monotonic() < end:
                read()
                await asyncio.sleep(0.02)
        finally:
            monitor.stop()
        return monitor.max_lag_ms

    return asyncio.run(run())


@pytest.mark.parametrize("name", ["zz_loop_during_warm_a", "zz_loop_during_warm_b"])
def test_a_loop_read_during_the_warm_up_does_not_wait_for_the_import(invented, name):
    """The warm-up thread holds the lock through a slow declaration (the
    import); a read on the loop meanwhile must not block the loop on it."""
    started, release = threading.Event(), threading.Event()

    def declarer():
        started.set()
        release.wait(10)          # the import, held by the warm-up thread
        _declare(name)

    invented(name, declarer)
    warm = threading.Thread(target=tool_registry.declare_deferred_results)
    warm.start()
    assert started.wait(10)
    try:
        lag = _loop_lag_while(lambda: tool_registry.result_spec(name), 1.0)
    finally:
        release.set()
        warm.join(10)
    assert lag < 250
    assert tool_registry.result_spec(name).kind == "search"


@pytest.mark.parametrize("name, hold_s", [("zz_loop_first_read_a", 0.6), ("zz_loop_first_read_b", 1.0)])
def test_a_first_read_on_the_loop_declares_off_the_loop(invented, name, hold_s):
    """With nothing having declared yet, the first read made ON the loop hands
    the declarer to a worker thread: the loop keeps serving, and the result is
    declared once the worker finishes."""
    threads = []

    def declarer():
        threads.append(threading.get_ident())
        time.sleep(hold_s)        # the import
        _declare(name)

    invented(name, declarer)
    loop_thread = []

    def read():
        loop_thread.append(threading.get_ident())
        tool_registry.result_spec(name)

    lag = _loop_lag_while(read, hold_s + 0.4)
    assert lag < 250
    deadline = time.monotonic() + 10
    while declarer in tool_registry._DEFERRED and time.monotonic() < deadline:
        time.sleep(0.02)
    assert threads == [threads[0]] and threads[0] != loop_thread[0]
    assert tool_registry._RESULTS[name].kind == "search"


@pytest.mark.parametrize("names", [("zz_ready_a",), ("zz_ready_b", "zz_ready_c")])
def test_the_app_declares_before_it_serves(invented, monkeypatch, names):
    """Readiness waits for the declarations even when the model warm-up runs
    in the background (production leaves WARM_MODELS_BLOCKING unset)."""
    from fastapi.testclient import TestClient

    from app import main

    async def no_warm_up():
        return None

    monkeypatch.delenv("WARM_MODELS_BLOCKING", raising=False)
    monkeypatch.setattr(main, "_warm_models", no_warm_up)
    declarers = [invented(n, lambda n=n: _declare(n)) for n in names]
    with TestClient(main.app):
        assert not any(d in tool_registry._DEFERRED for d in declarers)
        for n in names:
            assert tool_registry._RESULTS[n].kind == "search"
