"""A loop stall is captured WHILE it happens, and its CPU figures share one clock.

RED MAIN (CI e312ea2, 06e541d, PR #394): the watchdog's stacks were taken when
the stall ENDED -- the sampler is a Python thread and needs the GIL to wake, so
a thread holding the GIL across one long call (turn-io_3 burned 630 ms) was
already in its next call by the time anything was sampled. And the same stall
logged threads=820ms against proc=190ms: two clocks read at two instants.

The monitor tests drive ``LagMonitor._run`` (the monitor thread, started by
``start()``): a late heartbeat, the live timer it arms, the oversleep.

Every test builds its own input; no document text.
"""
from __future__ import annotations

import asyncio
import ctypes
import functools
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from app.core import loop_watchdog

# Never in a process shared with the gated shards: these tests arm the
# process-wide faulthandler timer and hold the GIL for up to 0.6 s against
# wall-clock thresholds. ``serial`` without ``timing`` runs them in CI's own
# "serial and not timing" step -- a pytest process of its own, after the xdist
# shards and apart from the timing job (pytest.ini, .github/workflows/test.yml).
pytestmark = [pytest.mark.serial]

_REPO = Path(__file__).resolve().parents[1]


def _gil_holding_call(n: int) -> None:
    """One C call that never releases the GIL: a backtracking regex."""
    re.fullmatch(r"(x+x+)+y", "x" * n)


#: The OS sleep, resolved once: loading a library releases the GIL, so it
#: never happens inside a hold. ``ctypes.PyDLL`` (unlike ``CDLL``) keeps the
#: GIL across each call.
if sys.platform == "win32":
    _os_sleep_ms = ctypes.PyDLL("kernel32").Sleep
else:
    _os_usleep = ctypes.PyDLL(None).usleep


def _gil_holding_wait(seconds: float) -> None:
    """One C call that never releases the GIL and lasts ``seconds`` on any
    runner. ``_size_for`` only bounds a hold from BELOW -- the regex's cost
    roughly doubles per character, so a size calibrated to 0.12 s can run
    0.3 s on a slower runner -- which is enough for a hold that must outlast
    a delay, never for one that must stay under it."""
    if sys.platform == "win32":
        _os_sleep_ms(ctypes.c_uint32(round(seconds * 1000)))
    else:
        _os_usleep(ctypes.c_uint32(round(seconds * 1_000_000)))


def _size_for(seconds: float) -> int:
    n = 16
    while n < 32:
        t = time.perf_counter()
        _gil_holding_call(n)
        if time.perf_counter() - t >= seconds:
            return n
        n += 1
    return n


#: A switch interval no hold in this file comes near: while it is set, a
#: thread that has the GIL keeps it until it blocks, and is never preempted.
_NO_FORCED_SWITCH_S = 30.0
#: Voluntary GIL releases the holder makes after raising the interval and
#: before it may start the hold. A thread already waiting for the GIL when the
#: interval changed keeps the old (short) interval for that wait; each of
#: these sleeps hands the GIL to such a waiter, so they are all served first.
_SETTLE_SLEEPS = 8


def hold_while_timer_in_force(monkeypatch):
    """A worker's hold that begins ONLY while the live timer is in force.

    The monitor has two stretches with no timer pending (advisor 1315Z, 21
    Nov; documented blind spot): from the ``cancel`` after a served heartbeat
    to the sleep's ``arm``, and from the sleep's ``cancel`` to the next
    post's ``arm``. A hold that begins there is invisible by design (max_lag
    0, stalls 0, no dump), so a test that lets its hold land anywhere fails
    on timing, not on the product. This places the hold, it does not move
    the windows: every ``arm`` and ``cancel`` is recorded (an arm after it
    returns, a cancel before it starts, so a doubt always reads "not in
    force"), and the hold begins with the GIL held continuously from the
    check that the last event was an ``arm``. Holding it continuously is the
    switch interval's job: raised for the placement and the hold, restored
    the moment the hold returns. The hold's length, the monitor and every
    assertion are untouched."""
    events: list = []
    real_arm, real_cancel = loop_watchdog.LiveDump.arm, loop_watchdog.LiveDump.cancel

    def arm(self, after_s=None):
        real_arm(self, after_s)
        if self.active:
            events.append("arm")

    def cancel(self):
        if self.active:
            events.append("cancel")
        real_cancel(self)

    monkeypatch.setattr(loop_watchdog.LiveDump, "arm", arm)
    monkeypatch.setattr(loop_watchdog.LiveDump, "cancel", cancel)

    def hold(n: int, timeout_s: float = 5.0) -> None:
        old = sys.getswitchinterval()
        sys.setswitchinterval(_NO_FORCED_SWITCH_S)
        try:
            for _ in range(_SETTLE_SLEEPS):
                time.sleep(0.001)
            deadline = time.monotonic() + timeout_s
            while not events or events[-1] != "arm":
                assert time.monotonic() < deadline, f"the timer was never in force: {events[-5:]}"
                time.sleep(0.001)
            # No call between the check and the hold releases the GIL.
            _gil_holding_call(n)
        finally:
            sys.setswitchinterval(old)

    return hold


def unjustified_reads(events: list, after_s: float) -> list:
    """Each dump-file read (``"take"``) whose round cannot have lasted
    ``after_s``. ``events`` is the monitor thread's ``("cancel"|"take", t)``
    in order, opened by ``("cancel", start)``. A round begins after the
    cancel that ended the round before it, so a read less than ``after_s``
    past the second-last cancel before it came from a round no timer could
    fire in. Bounded from below, never guessed: a read in a round a loaded
    runner really made late is justified, however many there are."""
    bad, cancels = [], []
    for kind, t in events:
        if kind == "cancel":
            cancels.append(t)
        elif len(cancels) >= 2 and t - cancels[-2] < after_s:
            bad.append(round((t - cancels[-2]) * 1000, 1))
    return bad


def rounds_past(events: list, limit_s: float) -> int:
    """How many gaps between consecutive cancels last ``limit_s`` or more:
    an upper bound on the rounds that really were that late."""
    cancels = [t for kind, t in events if kind == "cancel"]
    return sum(b - a >= limit_s for a, b in zip(cancels, cancels[1:]))


#: Set in the child pytest process that runs one C-timer test by itself.
_OWN_PROCESS_ENV = "LOOP_CAPTURE_OWN_PROCESS"
#: Wall-clock cap for one C-timer test in its own process.
_OWN_PROCESS_TIMEOUT_S = 60


def child_verdict(rc, out: str) -> str:
    """'' when the child pytest passed, else why it did not. A negative return
    code is a signal (the C timer reading a running thread's frame can crash
    the interpreter: #396 9370c0c, exit 139); it is named, never retried."""
    if rc == 0:
        return ""
    tail = out[-3000:]
    if rc is None:
        return f"timed out after {_OWN_PROCESS_TIMEOUT_S}s: {tail}"
    if rc < 0:
        try:
            name = signal.Signals(-rc).name
        except ValueError:
            name = f"signal {-rc}"
        return f"child crashed: {name}: {tail}"
    return f"child pytest rc={rc}: {tail}"


def run_in_own_process(nodeid: str) -> tuple:
    """Run one test node in a child pytest; ``(rc, combined output)``."""
    env = {**os.environ, _OWN_PROCESS_ENV: "1"}
    try:
        proc = subprocess.run([sys.executable, "-m", "pytest", nodeid, "-q", "-p", "no:cacheprovider"], check=False,
                              cwd=_REPO, env=env, capture_output=True, text=True,
                              timeout=_OWN_PROCESS_TIMEOUT_S)
        return proc.returncode, proc.stdout + proc.stderr
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or b"") + (exc.stderr or b"")
        return None, out.decode("utf-8", "replace") if isinstance(out, bytes) else out


def own_process(test):
    """A test that arms the real faulthandler C timer while other threads run
    is run in a child pytest of its own. A crash of that timer then fails THIS
    test (signal named) and cannot end the serial step and every unrelated test
    after it. Same body, same assertions: the child runs the test itself."""
    @functools.wraps(test)
    def wrapper(*args, **kwargs):
        if os.environ.get(_OWN_PROCESS_ENV):
            return test(*args, **kwargs)
        nodeid = os.environ["PYTEST_CURRENT_TEST"].rsplit(" (", 1)[0]
        why = child_verdict(*run_in_own_process(nodeid))
        assert not why, why
    return wrapper


@pytest.mark.parametrize("code,crashed", [
    ("import faulthandler; faulthandler.disable(); faulthandler._sigsegv()", True),
    ("pass", False),
])
def test_a_crash_in_a_child_is_named_and_the_parent_survives(code, crashed):
    """The guard's two sides, on a real process: a child that dies of a
    signal (or a non-zero exit) is a named failure; a clean child is none.
    This process is still here to assert it."""
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30, check=False)
    why = child_verdict(proc.returncode, proc.stdout + proc.stderr)
    assert bool(why) is crashed, (proc.returncode, why)
    if crashed and proc.returncode < 0:
        assert "child crashed: SIG" in why, why


@pytest.mark.parametrize("rc,expected", [
    (0, ""), (-11, "child crashed: SIGSEGV"), (1, "child pytest rc=1"), (None, "timed out after"),
])
def test_the_child_verdict_names_each_way_a_child_can_end(rc, expected):
    why = child_verdict(rc, "tail")
    assert why.startswith(expected) and (why == "" or why.endswith("tail")), why


@pytest.mark.parametrize("hold", [True, False])
@own_process
def test_the_live_dump_names_the_call_that_holds_the_gil_during_the_stall(monkeypatch, hold):
    """A worker thread holds the GIL inside one C call while the loop waits.
    The live dump (C timer, no GIL) shows the worker IN that call; with no
    such call there is no live dump at all."""
    n = _size_for(0.6) if hold else 0
    held = hold_while_timer_in_force(monkeypatch)

    async def run():
        mon = loop_watchdog.LagMonitor(asyncio.get_running_loop(), threshold_ms=250,
                                       tick_ms=10, log=False, live_ms=150).start()
        await asyncio.sleep(0.1)
        # Like a pool thread, the worker outlives its call, so it still has
        # its name when the stall ends and the dump is formatted.
        done, release = threading.Event(), threading.Event()

        def work():
            held(n)
            done.set()
            release.wait(5)

        if hold:
            worker = threading.Thread(target=work, name="gil-holder")
            worker.start()
            while not done.is_set():
                await asyncio.sleep(0.01)
        else:
            await asyncio.sleep(0.6)
        await asyncio.sleep(0.2)
        mon.stop()
        release.set()
        return mon

    mon = asyncio.run(run())
    if hold:
        assert mon.live_dumps, f"no live dump; max_lag={mon.max_lag_ms:.0f} stalls={len(mon.stalls)}"
        _, during = max(mon.live_dumps)
        assert "[gil-holder]" in during and "_gil_holding_call" in during, during
        # The hold is SEEN whichever side of a heartbeat it began on: as a
        # late heartbeat (a stall carrying the live dump) or, when it began
        # while the monitor slept, as the monitor's own late wake.
        if mon.stalls:
            worst = max(mon.stalls)[1]
            assert "DURING" in worst and "_gil_holding_call" in worst
        else:
            assert mon.max_wake_late_ms >= 250, mon.max_wake_late_ms
            assert during.startswith("WAKE_LATE ")
    else:
        assert mon.live_dumps == [] and mon.stalls == []
        assert mon.max_wake_late_ms < 150, mon.max_wake_late_ms


@pytest.mark.parametrize("hold", [True, False])
@own_process
def test_a_gil_hold_that_begins_while_the_monitor_sleeps_is_not_invisible(hold):
    """The blind spot this closes: the hold starts inside the monitor's tick
    sleep, so no heartbeat is pending and ``max_lag_ms`` stays ~0. The late
    wake and the live dump still name it."""
    n = _size_for(0.6) if hold else 0
    loop = asyncio.new_event_loop()
    mon = loop_watchdog.LagMonitor(loop, threshold_ms=250, tick_ms=400, log=False, live_ms=150)
    # No loop running: drive the sleep path directly on this thread while a
    # worker holds the GIL from inside that sleep.
    release = threading.Event()

    def work():
        time.sleep(0.05)
        if hold:
            _gil_holding_call(n)
        release.wait(5)

    worker = threading.Thread(target=work, name="gil-holder")
    worker.start()
    try:
        mon._sleep_watched(mon._live, {})
    finally:
        release.set()
        worker.join()
        mon._live.close()
        loop.close()
    if hold:
        assert mon.max_wake_late_ms >= 150, mon.max_wake_late_ms
        assert mon.live_dumps and "[gil-holder]" in mon.live_dumps[0][1], mon.live_dumps
        assert "_gil_holding_call" in mon.live_dumps[0][1]
    else:
        assert mon.max_wake_late_ms < 150 and mon.live_dumps == []


@pytest.mark.parametrize("cancelled,starts_at_once", [(False, True), (True, False)])
@own_process
def test_a_placed_hold_begins_only_while_the_timer_is_in_force(monkeypatch, cancelled, starts_at_once):
    """The placement the hold tests rely on: after a cancel the hold waits
    for the next arm; after an arm it starts at once. The switch interval
    is the caller's again once the hold returns."""
    held = hold_while_timer_in_force(monkeypatch)
    started = threading.Event()
    monkeypatch.setattr(sys.modules[__name__], "_gil_holding_call", lambda n: started.set())
    interval = sys.getswitchinterval()
    live = loop_watchdog.LiveDump(30_000.0)  # in force, never fires here
    try:
        assert live.active
        live.arm()
        if cancelled:
            live.cancel()
        worker = threading.Thread(target=held, args=(0,))
        worker.start()
        assert started.wait(0.5) is starts_at_once
        live.arm()
        assert started.wait(5)
        worker.join(5)
        assert not worker.is_alive()
    finally:
        live.close()
    assert sys.getswitchinterval() == interval


@pytest.mark.parametrize("first_closed,second_active", [(False, False), (True, True)])
def test_only_one_monitor_owns_the_process_wide_dump_timer(first_closed, second_active):
    """faulthandler has one timer per process: ``LiveDump.__init__`` refuses a
    second owner while the first holds it, and grants it once the first lets
    it go."""
    first = loop_watchdog.LiveDump(150)
    assert first.active
    if first_closed:
        first.close()
    second = loop_watchdog.LiveDump(150)
    try:
        assert second.active is second_active
        if not second_active:
            assert second.take() == ""
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize("text,names,expected", [
    ('Thread 0x00000000000000ff (most recent call first):\n'
     '  File "/w/app/x.py", line 7 in slow\n'
     '  File "/w/app/y.py", line 3 in run\n\n'
     'Current thread 0x0000000000000010 (most recent call first):\n'
     '  File "/usr/lib/selectors.py", line 468 in select\n',
     {255: "turn-io_3", 16: "MainThread"},
     "[turn-io_3] /w/app/x.py:7 slow <- /w/app/y.py:3 run || [*MainThread] /usr/lib/selectors.py:468 select"),
    ('Thread 0xabc (most recent call first):\n  File "a.py", line 1 in f\n',
     {}, "[0xabc] a.py:1 f"),
])
def test_the_live_dump_is_one_line_of_named_threads(text, names, expected):
    assert loop_watchdog.format_live_dump(text, names) == expected


def _stat_line(pid: int, utime: int, stime: int) -> bytes:
    rest = ["S", "1", "1", "1", "0", "-1", "0", "0", "0", "0", "0", str(utime), str(stime), "0", "0"]
    return f"{pid} (py) thing) {' '.join(rest)}\n".encode()


@pytest.mark.parametrize("proc_ticks,present", [(70, True), (0, False)])
def test_process_and_thread_cpu_are_read_on_one_clock(tmp_path, proc_ticks, present):
    tasks = tmp_path / "task"
    (tasks / "11").mkdir(parents=True)
    (tasks / "11" / "stat").write_bytes(_stat_line(11, 30, 10))
    stat = tmp_path / "stat"
    if present:
        stat.write_bytes(_stat_line(11, proc_ticks, 0))
    tick = loop_watchdog.os.sysconf("SC_CLK_TCK") if hasattr(loop_watchdog.os, "sysconf") else 100
    proc, threads = loop_watchdog.cpu_snapshot(str(tasks), str(stat))
    assert threads == {11: 40 * 1000.0 / tick}
    if present:
        assert proc == proc_ticks * 1000.0 / tick
    else:  # no /proc/self/stat: the os.times() fallback, never None
        assert isinstance(proc, float) and proc >= 0


@pytest.mark.parametrize("threads_after,expected", [
    # a thread absent from the earlier snapshot is counted apart, not summed
    ({1: 60.0, 2: 630.0}, "sum window=400ms proc=50ms threads=50ms/1 new=1"),
    ({1: 60.0}, "sum window=400ms proc=50ms threads=50ms/1"),
])
def test_a_thread_born_inside_the_window_is_not_summed_as_a_delta(threads_after, expected):
    got = loop_watchdog.stall_cpu_summary({1: 10.0}, threads_after, 50.0, {}, {}, 400.0)
    assert got == expected


@pytest.mark.parametrize("snapshot_s,hold", [(0.35, False), (0.35, True)])
@own_process
def test_the_heartbeat_is_stamped_at_the_post_and_the_snapshot_runs_after_it(monkeypatch, snapshot_s, hold):
    """Ruling (advisor 10:10Z 18 Nov; it replaces 09:17Z and 04:18Z): the
    heartbeat is stamped at the post and the CPU snapshot runs in its own
    thread, never on the heartbeat's path. A slow snapshot is therefore not
    loop lag (no stall, lag near zero, its duration its own figure), and
    the live dump still fires on a real GIL hold while that slow snapshot
    keeps running."""
    def snapshot(*_a, **_k):
        time.sleep(snapshot_s)
        return 0.0, {}

    monkeypatch.setattr(loop_watchdog, "cpu_snapshot", snapshot)
    n = _size_for(0.6) if hold else 0
    held = hold_while_timer_in_force(monkeypatch)

    async def run():
        mon = loop_watchdog.LagMonitor(asyncio.get_running_loop(), threshold_ms=250,
                                       tick_ms=10, log=False, live_ms=150).start()
        await asyncio.sleep(0.9)
        done, release = threading.Event(), threading.Event()

        def work():
            held(n)
            done.set()
            release.wait(5)

        if hold:
            worker = threading.Thread(target=work, name="gil-holder")
            worker.start()
            while not done.is_set():
                await asyncio.sleep(0.01)
            await asyncio.sleep(0.2)
        mon.stop()
        release.set()
        return mon

    mon = asyncio.run(run())
    assert mon.max_snapshot_ms >= snapshot_s * 1000 * 0.9, mon.max_snapshot_ms
    if hold:
        assert mon.live_dumps, f"no live dump; max_lag={mon.max_lag_ms:.0f} stalls={len(mon.stalls)}"
        _, during = max(mon.live_dumps)
        assert "[gil-holder]" in during and "_gil_holding_call" in during, during
    else:
        assert mon.stalls == [], (mon.max_lag_ms, mon.stalls[:1])
        assert mon.max_lag_ms < 100, mon.max_lag_ms
        assert mon.live_dumps == [], mon.live_dumps[:1]


@pytest.mark.parametrize("pre_post_s", [0.0, 0.4])
@own_process
def test_the_monitors_own_pre_post_work_is_never_charged_to_the_loop(monkeypatch, pre_post_s):
    """Arming the live timer cancels the C timer thread in force and waits for
    it to stop, and on a loaded runner that wait can outlast the live delay
    while the monitor holds the GIL. The wait is the MONITOR's own work, so the
    heartbeat is stamped and posted after it and the wait is reported as
    ``stamp_to_post``. The loop served every heartbeat at once here: that is
    never a stall, and a stall no timer had been armed for the live delay is
    never recorded (so a recorded stall always carries the dump its own window
    produced)."""
    real_arm = loop_watchdog.LiveDump.arm

    def arm_after_a_wait(self, after_s=None):
        if threading.current_thread().name == "loop-watchdog":
            time.sleep(pre_post_s)
        real_arm(self, after_s)

    monkeypatch.setattr(loop_watchdog.LiveDump, "arm", arm_after_a_wait)

    async def run():
        mon = loop_watchdog.LagMonitor(asyncio.get_running_loop(), threshold_ms=250,
                                       tick_ms=10, log=False, live_ms=150).start()
        await asyncio.sleep(2.0)
        mon.stop()
        return mon

    mon = asyncio.run(run())
    assert mon.max_stamp_to_post_ms >= pre_post_s * 1000 * 0.9, mon.max_stamp_to_post_ms
    assert mon.stalls == [], (mon.max_lag_ms, mon.stalls[:1])
    assert mon.max_lag_ms < 250, mon.max_lag_ms


@pytest.mark.parametrize("live_ms", [150.0, 0.0])
@own_process
def test_stop_ends_both_threads_and_hands_the_dump_timer_back(live_ms):
    """``stop`` joins the heartbeat and snapshot threads and releases the
    process-wide dump timer, so the next monitor can own it."""
    async def run():
        mon = loop_watchdog.LagMonitor(asyncio.get_running_loop(), threshold_ms=250,
                                       tick_ms=10, log=False, live_ms=live_ms).start()
        await asyncio.sleep(0.1)
        mon.stop()
        return mon

    mon = asyncio.run(run())
    assert not mon._thread.is_alive() and not mon._snap_thread.is_alive()
    nxt = loop_watchdog.LiveDump(150)
    try:
        assert nxt.active, "the stopped monitor still owns the dump timer"
    finally:
        nxt.close()


@pytest.mark.parametrize("lags,kept", [
    ([5.0, 1.0, 9.0, 3.0], [9.0, 5.0, 3.0]),
    ([2.0, 4.0], [4.0, 2.0]),
])
def test_live_dumps_keep_only_the_worst(lags, kept):
    """A monitor lives as long as the process; its dump list is bounded and
    keeps the latest-served (worst) entries."""
    got: list = []
    for lag in lags:
        loop_watchdog.keep_worst(got, (lag, f"d{lag}"), cap=3)
    assert sorted((lag for lag, _ in got), reverse=True) == kept


@pytest.mark.parametrize("live_ms,armed", [(0.0, False), (150.0, True)])
def test_live_capture_is_off_unless_asked(live_ms, armed):
    async def run():
        mon = loop_watchdog.LagMonitor(asyncio.get_running_loop(), log=False, live_ms=live_ms)
        try:
            return mon._live is not None and mon._live.active
        finally:
            if mon._live is not None:
                mon._live.close()

    assert asyncio.run(run()) is armed


@pytest.mark.parametrize("cancelled,fired", [(True, False), (False, True)])
@own_process
def test_a_cancelled_timer_writes_nothing(cancelled, fired):
    """``cancel`` is what makes a dump mean "still unserved": a timer
    cancelled before its delay writes nothing; one left armed writes."""
    live = loop_watchdog.LiveDump(40)
    try:
        live.arm()
        if cancelled:
            live.cancel()
        time.sleep(0.25)
        assert bool(live.take()) is fired
    finally:
        live.close()


# Every dump the C timer writes opens with this line.
_DUMP_START = "Timeout ("


@pytest.mark.parametrize("live_ms,after_s", [(40.0, None), (30_000.0, 0.03)])
def test_the_timer_armed_in_this_process_repeats_until_cancel(live_ms, after_s):
    """``arm`` runs in THIS process, not only inside a forked monitor: armed
    here directly, the timer writes one dump per delay -- so a stall longer
    than one delay is captured and a fire lost to signal jitter costs nothing
    -- and ``cancel`` stops it for good.

    Two inputs: the default delay (40 ms), and an explicit override that beats
    a 30 s default. A single dump (a one-shot timer), or any dump written
    after the cancel, fails."""
    live = loop_watchdog.LiveDump(live_ms)
    assert live.active, "another LiveDump still owns the process-wide timer"
    try:
        live.arm(after_s)
        text, deadline = "", time.monotonic() + 10.0
        while text.count(_DUMP_START) < 2 and time.monotonic() < deadline:
            text += live.take()
            time.sleep(0.01)
        assert text.count(_DUMP_START) >= 2, f"the timer did not repeat: {text[:200]!r}"
        live.cancel()
        time.sleep(0.2)  # far past a 30-40 ms delay: a live timer would fire again
        live.take()  # drain what was written up to (or during) the cancel
        time.sleep(0.2)
        assert live.take() == ""
    finally:
        live.close()


@pytest.mark.parametrize("who_cancels,fired", [("refused", True), ("owner", False)])
@own_process
def test_a_refused_owner_cannot_cancel_the_owners_timer(who_cancels, fired):
    """The timer is process-wide: a monitor refused ownership must not touch it."""
    owner = loop_watchdog.LiveDump(40)
    refused = loop_watchdog.LiveDump(40)
    try:
        assert owner.active and not refused.active
        owner.arm()
        (refused if who_cancels == "refused" else owner).cancel()
        time.sleep(0.25)
        assert bool(owner.take()) is fired
    finally:
        refused.close()
        owner.close()


@pytest.mark.parametrize("monitor_slow,loop_late_ms,near", [(True, 0, False), (False, 190, True)])
@own_process
def test_a_served_heartbeat_is_never_a_near_miss(monkeypatch, monitor_slow, loop_late_ms, near):
    """Serving a heartbeat cancels its timer (``_serve``). The monitor itself
    may be slow after that -- held 300 ms after every heartbeat here -- and a
    heartbeat already served still never counts as late. A loop really late
    past live_ms (150) but under the limit (250) is a near miss."""
    if monitor_slow:
        real = loop_watchdog.heartbeat_lag_ms

        def slow_after_serve(posted, served_at, now):
            time.sleep(0.3)
            return real(posted, served_at, now)

        monkeypatch.setattr(loop_watchdog, "heartbeat_lag_ms", slow_after_serve)

    async def run():
        mon = loop_watchdog.LagMonitor(asyncio.get_running_loop(), threshold_ms=250,
                                       tick_ms=10, log=False, live_ms=150).start()
        await asyncio.sleep(1.0)
        for _ in range(5 if loop_late_ms else 0):
            time.sleep(loop_late_ms / 1000)  # the loop itself, late under the limit
            await asyncio.sleep(0.05)
        mon.stop()
        return mon

    mon = asyncio.run(run())
    assert mon.stalls == [], mon.stalls[:1]
    # A near miss is counted; its dump is dropped, never kept.
    assert mon.live_dumps == [], mon.live_dumps[:1]
    assert (mon.live_near_misses > 0) is near, mon.live_near_misses


@pytest.mark.parametrize("hold_s,wake_late", [(0.2, False), (0.45, True)])
@own_process
def test_the_timer_runs_from_the_serve_not_from_the_post(hold_s, wake_late):
    """Serving a heartbeat re-arms the timer (``_serve``), so the monitor's
    late return is measured from the serve. Here the loop serves ~200 ms after
    the post (under live_ms 300) and then holds the GIL, which keeps the
    monitor from returning. A hold shorter than live_ms after the serve is
    nothing at all: a timer still running from the post fired inside it and
    counted a near miss the loop never had. A hold longer than live_ms after
    the serve is a WAKE_LATE dump naming the holder.

    CI 06:49Z 22 Nov (c878987c): the short case dumped WAKE_LATE. Its hold
    was a regex sized to "at least 0.12 s", and one size step more is ~0.3 s
    -- past live_ms. The hold is now exactly ``hold_s`` (``_gil_holding_wait``)
    and each instant sits >= 80 ms from the 300 ms delay it is judged
    against: serve ~0.2 s after the post; a timer left running from the post
    fires ~0.1 s into the 0.2 s hold; the serve's timer is due 0.1 s after
    the short hold ends and 0.15 s before the long one does."""

    async def run():
        loop = asyncio.get_running_loop()
        mon = loop_watchdog.LagMonitor(loop, threshold_ms=1000, tick_ms=10,
                                       log=False, live_ms=300).start()
        await asyncio.sleep(0.2)
        for _ in range(2):
            done = asyncio.Event()

            def late_then_hold():
                time.sleep(0.2)  # late, the GIL free: the post lands now
                # Runs after the heartbeat's _serve, queued during the sleep.
                loop.call_soon(lambda: (_gil_holding_wait(hold_s), done.set()))

            loop.call_soon(late_then_hold)
            await done.wait()
            await asyncio.sleep(0.2)
        mon.stop()
        return mon

    mon = asyncio.run(run())
    assert mon.stalls == [], mon.stalls[:1]
    if wake_late:
        assert any(d.startswith("WAKE_LATE ") and "_gil_holding_wait" in d
                   for _, d in mon.live_dumps), mon.live_dumps
    else:
        assert mon.live_near_misses == 0, mon.live_near_misses
        assert mon.live_dumps == [], mon.live_dumps[:1]



@pytest.mark.parametrize("hold", [True, False])
@own_process
def test_the_monitor_reads_its_dump_file_only_when_a_timer_could_have_fired(monkeypatch, hold):
    """CI 076bcbf (virgin, 18 Nov): a 0.6 s hold gave no dump, max_lag 0, no
    stall. The monitor read its dump file after every heartbeat and every
    sleep, right after ``cancel``: ``os.fstat`` releases the GIL with no
    timer pending, no heartbeat in flight and no sleep to oversleep, and a
    thread waiting for the GIL takes it there (reproduced 3 of 3 by starting
    the hold inside that read). Now a round too short for a timer to fire
    reads nothing, so a quiet monitor never hands the GIL over unwatched; a
    round the hold made late is read and names the holder."""
    n = _size_for(0.6) if hold else 0
    held = hold_while_timer_in_force(monkeypatch)
    events: list = []
    armed = threading.Event()
    real_take, real_cancel, real_arm = (
        loop_watchdog.LiveDump.take,
        loop_watchdog.LiveDump.cancel,
        loop_watchdog.LiveDump.arm,
    )

    def take(self):
        if threading.current_thread().name == "loop-watchdog":
            events.append(("take", time.monotonic()))
        return real_take(self)

    def cancel(self):
        real_cancel(self)
        if threading.current_thread().name == "loop-watchdog":
            events.append(("cancel", time.monotonic()))

    def arm(self, after_s=None):
        real_arm(self, after_s)
        # Signalled AFTER the arm, never before: ``dump_traceback_later`` can be
        # delayed past the next GIL handoff, so a barrier raised before the call
        # released the worker into its hold while NO timer was in force. The
        # hold then ran in an unarmed gap, the monitor correctly had no dump to
        # read, and the test failed with a real stall (max_lag 656-860) and no
        # dump -- measured: one ``arm_enter`` at the hold's start, its return
        # 780-860 ms later, after the hold. The barrier now means what it says.
        if threading.current_thread().name == "loop-watchdog":
            armed.set()

    monkeypatch.setattr(loop_watchdog.LiveDump, "take", take)
    monkeypatch.setattr(loop_watchdog.LiveDump, "cancel", cancel)
    monkeypatch.setattr(loop_watchdog.LiveDump, "arm", arm)

    async def run():
        events.append(("cancel", time.monotonic()))
        mon = loop_watchdog.LagMonitor(asyncio.get_running_loop(), threshold_ms=250,
                                       tick_ms=10, log=False, live_ms=150).start()
        await asyncio.sleep(0.3)
        done, release = threading.Event(), threading.Event()

        def work():
            # Start the hold only after the monitor has armed its timer, so
            # the hold overlaps the timer's fire window instead of racing it
            # (a hold begun in the unarmed gap between two heartbeats is
            # invisible: no heartbeat is in flight and no timer is armed).
            assert armed.wait(5), "the monitor never armed its timer"
            held(n)
            done.set()
            release.wait(5)

        if hold:
            armed.clear()
            worker = threading.Thread(target=work, name="gil-holder")
            worker.start()
            while not done.is_set():
                await asyncio.sleep(0.01)
        else:
            await asyncio.sleep(0.6)
        await asyncio.sleep(0.2)
        mon.stop()
        release.set()
        return mon

    mon = asyncio.run(run())
    reads = [t for kind, t in events if kind == "take"]
    # Every read in a round that lasted the live delay: a quiet round never
    # hands the GIL to a file read. Not "zero reads": a round a loaded runner
    # really made late may be read, and is justified by its own length.
    assert unjustified_reads(events, 0.150) == [], (unjustified_reads(events, 0.150), len(reads))
    assert len(mon.stalls) <= rounds_past(events, 0.250), mon.stalls[:1]
    if hold:
        assert reads, "the late round was never read"
        assert mon.live_dumps, f"no live dump; max_lag={mon.max_lag_ms:.0f} stalls={len(mon.stalls)}"
        _, during = max(mon.live_dumps)
        assert "[gil-holder]" in during and "_gil_holding_call" in during, during
    else:
        # A dump comes only from a read, and every read was justified above.
        assert len(mon.live_dumps) <= len(reads), mon.live_dumps[:1]


class _TimerlessDump:
    """``LiveDump`` without the C timer (so in this process): logs the
    monitor's cancels and reads, in order, for ``unjustified_reads``."""

    def __init__(self, after_ms: float) -> None:
        self.after_s = after_ms / 1000
        self.events: list = [("cancel", time.monotonic())]

    active = True

    def arm(self, after_s=None) -> None:
        pass

    def cancel(self) -> None:
        self.events.append(("cancel", time.monotonic()))

    def take(self) -> str:
        self.events.append(("take", time.monotonic()))
        return ""

    def close(self) -> None:
        pass


@pytest.mark.parametrize("loop_block_s, read", [(0.0, False), (0.2, True)])
def test_a_heartbeat_round_is_read_only_when_it_lasted_the_live_delay(monkeypatch, loop_block_s, read):
    """The read decision itself, with no C timer: a round shorter than the
    live delay (10 ms ticks, heartbeats served at once) is never read, so the
    monitor never releases the GIL to a file read with no timer pending; a
    heartbeat the loop serves 200 ms late (past live 150 ms, under the 250 ms
    limit) is read. Robust on a loaded runner without a wider limit: a read
    is allowed exactly where the round really lasted 150 ms."""
    monkeypatch.setattr(loop_watchdog, "LiveDump", _TimerlessDump)

    async def run():
        mon = loop_watchdog.LagMonitor(asyncio.get_running_loop(), threshold_ms=250,
                                       tick_ms=10, log=False, live_ms=150).start()
        await asyncio.sleep(0.3)
        if loop_block_s:
            time.sleep(loop_block_s)  # the loop is late; the GIL stays free
        await asyncio.sleep(0.1)
        mon.stop()
        return mon

    mon = asyncio.run(run())
    events = mon._live.events
    reads = [t for kind, t in events if kind == "take"]
    assert unjustified_reads(events, 0.150) == [], (unjustified_reads(events, 0.150), len(reads))
    assert len(mon.stalls) <= rounds_past(events, 0.250), mon.stalls[:1]
    if read:
        assert reads, "the 200 ms late heartbeat was never read"


@pytest.mark.parametrize("events, bad", [
    ([("cancel", 0.0), ("cancel", 0.01), ("take", 0.011)], [11.0]),
    ([("cancel", 0.0), ("cancel", 0.2), ("take", 0.201)], []),
    ([("cancel", 0.0), ("take", 0.001)], []),
])
def test_unjustified_reads_flags_a_read_only_in_a_round_shorter_than_the_delay(events, bad):
    assert unjustified_reads(events, 0.150) == bad
