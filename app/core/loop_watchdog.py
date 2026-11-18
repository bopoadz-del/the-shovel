"""Name the code that freezes the event loop.

Live 2026-10-07, 15 users holding 3-turn conversations: the I/O-free
``/livez`` waited up to 10.9 s. Something in a chat turn ran on the event loop
(or held the GIL from another thread long enough to starve it), and nothing in
the logs said what.

A daemon thread posts a heartbeat to the loop every ``LOOP_WATCHDOG_TICK_MS``
(default 50). When a heartbeat is still unserved ``LOOP_WATCHDOG_MS`` later
(default 250), it samples the stacks -- the loop thread's first, then every
other thread's, because a loop that is idle in ``select`` but late is being
starved of the GIL by a thread, and that thread is the culprit. When the
heartbeat finally runs, one ``LOOP_STALL lag=<ms>`` line is logged with the
sample. Code locations only (file, line, function), never local variables, so
no document text or client figures reach the logs. ``LOOP_WATCHDOG_MS=0``
turns it off.

``LagMonitor`` is the same measurement without logging, for tests.
"""
from __future__ import annotations

import asyncio
import faulthandler
import logging
import os
import re
import sys
import tempfile
import threading
import time
import traceback
from typing import Optional

_LOG = logging.getLogger(__name__)
_DEFAULT_MS = 250.0
_DEFAULT_TICK_MS = 50.0
_MAX_FRAMES = 25
_MIN_LOG_GAP_S = 2.0

_started: Optional["LagMonitor"] = None


def _env_float(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _stack(frame, max_frames: int = _MAX_FRAMES) -> str:
    lines = [f"{fs.filename}:{fs.lineno} {fs.name}" for fs in traceback.extract_stack(frame)]
    return " <- ".join(reversed(lines[-max_frames:]))


def sample_stacks(loop_ident: Optional[int]) -> str:
    """The loop thread's stack, then every other thread's (innermost first).

    When the loop thread id is not yet resolved (its ``call_soon`` to note it
    can itself be stuck behind the very stall we are sampling), fall back to the
    main thread, which is where the event loop runs. The loop stack is therefore
    always captured, so the worst-stall sample is never empty."""
    names = {t.ident: t.name for t in threading.enumerate()}
    me = threading.get_ident()
    frames = sys._current_frames()
    target = loop_ident if loop_ident in frames else threading.main_thread().ident
    parts = []
    if target in frames:
        parts.append(f"[loop] {_stack(frames[target])}")
    for ident, frame in frames.items():
        if ident in (me, target):
            continue
        parts.append(f"[{names.get(ident, ident)}] {_stack(frame, 8)}")
    return " || ".join(parts)


#: Linux exposes each thread's CPU time; elsewhere the CPU delta is omitted.
_TASK_DIR = "/proc/self/task"
#: Per-thread CPU is snapshotted at most this often (each read costs a file
#: per thread), so a stall's delta spans the stall plus at most this much.
_CPU_SNAPSHOT_S = 0.1


def thread_cpu_ms(task_dir: str = _TASK_DIR) -> dict:
    """CPU milliseconds (user + system) used so far by each thread, keyed by the
    thread's native id; empty where the OS gives no per-thread figures."""
    try:
        tids = os.listdir(task_dir)
    except OSError as exc:
        _LOG.debug("per-thread CPU unavailable (%s): stalls carry no cpu figures", exc)
        return {}
    tick = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
    out = {}
    for tid in tids:
        try:
            with open(os.path.join(task_dir, tid, "stat"), "rb") as fh:
                fields = fh.read().rsplit(b")", 1)[1].split()
            out[int(tid)] = (int(fields[11]) + int(fields[12])) * 1000.0 / tick
        except (OSError, ValueError, IndexError):
            continue
    return out


def busiest_threads(before: dict, after: dict, names: dict, top: int = 4) -> str:
    """The threads that burned the most CPU between two snapshots, by name.

    A loop that is late while parked in ``select`` is waiting for the GIL;
    the thread that burned CPU across that window held it. Threads that used
    no CPU are left out."""
    deltas = sorted(((after[t] - before.get(t, 0.0), t) for t in after), reverse=True)
    parts = [f"{names.get(t, t)}={d:.0f}ms" for d, t in deltas[:top] if d > 0]
    return ("cpu " + " ".join(parts)) if parts else ""


#: Host-wide CPU counters (Linux). Read with the per-thread snapshot so a
#: stall can say whether the whole box was busy (the runner starved us) or
#: our own threads used the window (a GIL convoy).
_HOST_STAT = "/proc/stat"


def process_cpu_ms() -> float:
    """CPU milliseconds (user + system) used so far by the whole process,
    every thread included (also threads that have since exited)."""
    t = os.times()
    return (t.user + t.system) * 1000.0


#: The whole process's CPU on the SAME clock as the per-thread figures.
_PROC_STAT = "/proc/self/stat"


def _proc_stat_cpu_ms(stat_path: str = _PROC_STAT) -> Optional[float]:
    """Process CPU (user + system) from ``/proc/self/stat``, in the same clock
    ticks as ``/proc/self/task/*/stat``; None where the file is absent."""
    try:
        with open(stat_path, "rb") as fh:
            fields = fh.read().rsplit(b")", 1)[1].split()
        tick = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
        return (int(fields[11]) + int(fields[12])) * 1000.0 / tick
    except (OSError, ValueError, IndexError) as exc:
        # Not Linux, or an unreadable stat line: the caller falls back to
        # os.times(), and says nothing about the per-thread clock match.
        _LOG.debug("process cpu from %s unavailable: %r", stat_path, exc)
        return None


def cpu_snapshot(task_dir: Optional[str] = None, stat_path: str = _PROC_STAT) -> tuple:
    """``(process_ms, {native_tid: thread_ms})`` read back to back from one
    clock. CI e312ea2 logged threads=820ms against proc=190ms for one stall:
    the per-thread figures came from ``/proc`` ticks and the process figure
    from ``os.times()`` read at a different instant, so the two could not be
    compared. Both now come from ``/proc`` in one pass; ``os.times()`` is the
    fallback only where ``/proc`` is absent."""
    threads = thread_cpu_ms() if task_dir is None else thread_cpu_ms(task_dir)
    proc = _proc_stat_cpu_ms(stat_path)
    return (process_cpu_ms() if proc is None else proc), threads


def host_cpu(stat_path: str = _HOST_STAT) -> dict:
    """Host-wide CPU ticks since boot -- ``busy`` (user, nice, system, irq,
    softirq), ``steal`` and ``total`` -- plus ``running`` (runnable tasks now).
    Empty where the OS has no such file."""
    try:
        with open(stat_path, "r", encoding="ascii", errors="replace") as fh:
            text = fh.read()
    except OSError as exc:
        _LOG.debug("host CPU unavailable (%s): stalls carry no host figures", exc)
        return {}
    out: dict = {}
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            if parts[0] == "cpu":
                v = [int(x) for x in parts[1:9]] + [0] * (8 - len(parts[1:9]))
                user, nice, system, _idle, _iowait, irq, softirq, steal = v
                out.update(busy=user + nice + system + irq + softirq, steal=steal, total=sum(v))
            elif parts[0] == "procs_running":
                out["running"] = int(parts[1])
        except (ValueError, IndexError) as exc:
            _LOG.debug("host CPU line unreadable (%s): %s", exc, parts[0])
    return out


def stall_cpu_summary(threads_before: dict, threads_after: dict, proc_ms: Optional[float],
                      host_before: dict, host_after: dict, window_ms: float) -> str:
    """Where the CPU went across a stall window, process-wide and host-wide.

    ``threads`` is the sum over EVERY thread that used CPU and how many did
    (not just the busiest four); ``proc`` the process total. Against the
    window: a process total near the window with many threads is a GIL
    convoy; a process total far below it while the host is busy (or stolen
    from) is the runner starving the whole process."""
    # A thread absent from the earlier snapshot started inside the window;
    # its lifetime total is not a delta, so it is counted apart (new=) and
    # kept out of the sum the process total is compared against.
    fresh = [t for t in threads_after if threads_before and t not in threads_before]
    used = [after - threads_before.get(t, 0.0) for t, after in threads_after.items()
            if t not in fresh]
    used = [d for d in used if d > 0]
    parts = [f"window={window_ms:.0f}ms"]
    if proc_ms is not None:
        parts.append(f"proc={proc_ms:.0f}ms")
    if threads_after:
        parts.append(f"threads={sum(used):.0f}ms/{len(used)}")
    if fresh:
        parts.append(f"new={len(fresh)}")
    total = host_after.get("total", 0) - host_before.get("total", 0)
    if total > 0:
        busy = host_after.get("busy", 0) - host_before.get("busy", 0)
        steal = host_after.get("steal", 0) - host_before.get("steal", 0)
        parts.append(f"host_busy={100.0 * busy / total:.0f}% steal={100.0 * steal / total:.0f}%")
    if "running" in host_after:
        parts.append(f"running={host_before.get('running', '?')}/{host_after['running']}")
    return "sum " + " ".join(parts)


def heartbeat_lag_ms(posted: float, served_at: Optional[float], now: float) -> float:
    """How late the LOOP served a heartbeat posted at ``posted``.

    ``served_at`` is stamped by the heartbeat callback itself, on the loop
    thread, so the figure is the loop's lateness only. The watchdog thread
    must re-take the GIL to wake from its wait; under load that wake-up can
    itself be hundreds of ms late, and timing the heartbeat at the watchdog's
    wake-up charged that delay to the loop (a lag above the threshold with no
    stall recorded, because the heartbeat had in fact been served in time).
    A heartbeat not yet served (``served_at`` None) is late by ``now - posted``.
    """
    end = served_at if served_at is not None else now
    return max(0.0, (end - posted) * 1000)


#: How late a heartbeat may be before the live dump fires. Below the stall
#: limit on purpose, so the dump lands WHILE the loop is still blocked.
DEFAULT_LIVE_MS = 150.0
_LIVE_FRAMES = 12
#: At most this many live dumps are kept per monitor (the worst ones); each is
#: a whole-process stack text, and a monitor runs for the life of the process.
LIVE_KEEP = 16
#: A watchdog wake-up at least this late is kept WITH ITS TIME
#: (``LagMonitor.wake_late_at``), so a report can say WHEN in a window the
#: thread could not run, not only how late it was at worst.
WAKE_LATE_NOTE_MS = 200.0
_FH_THREAD = re.compile(r"^(Current thread|Thread) (0x[0-9a-fA-F]+)")
_FH_FRAME = re.compile(r'^\s+File "(.*)", line (\d+) in (.*)$')
#: faulthandler has ONE process-wide timer; one monitor at a time may own it.
_live_owner_lock = threading.Lock()


def format_live_dump(text: str, names: dict, max_frames: int = _LIVE_FRAMES) -> str:
    """faulthandler's all-threads dump as one line, ``[thread] file:line func
    <- ...`` innermost first. Code locations only, like ``sample_stacks``.
    ``*`` marks the thread faulthandler names as current."""
    parts: list = []
    head, frames = None, []

    def flush() -> None:
        if head is not None:
            parts.append(f"[{head}] " + " <- ".join(frames[:max_frames]))

    for line in text.splitlines():
        m = _FH_THREAD.match(line)
        if m:
            flush()
            ident = int(m.group(2), 16)
            star = "*" if m.group(1).startswith("Current") else ""
            head, frames = f"{star}{names.get(ident, m.group(2))}", []
            continue
        f = _FH_FRAME.match(line)
        if f and head is not None:
            frames.append(f"{f.group(1)}:{f.group(2)} {f.group(3).strip()}")
    flush()
    return " || ".join(parts)


def keep_worst(kept: list, item: tuple, cap: int = LIVE_KEEP) -> None:
    """Add ``(lag_ms, text)`` to ``kept``, holding at most ``cap`` entries:
    when full, the least late one is dropped, so the worst stalls survive."""
    kept.append(item)
    if len(kept) > cap:
        kept.remove(min(kept, key=lambda e: e[0]))


class LiveDump:
    """Dumps every thread's stack DURING a stall, not after it.

    The sampler thread (``sample_stacks``) is Python: it needs the GIL to wake,
    so when a thread holds the GIL across a long call the sampler wakes only
    once that call has returned and records the loop back in ``select`` (CI
    e312ea2: turn-io_3 burned 630 ms, and the stacks showed it already in the
    next call). ``faulthandler.dump_traceback_later`` runs its timer on a C
    thread that reads the frames without the GIL, so a timer armed for each
    heartbeat and cancelled by the heartbeat itself fires only while that
    heartbeat is late, and shows what was running then."""

    def __init__(self, after_ms: float) -> None:
        self.after_s = after_ms / 1000
        self._fh = None
        if after_ms > 0 and _live_owner_lock.acquire(blocking=False):
            self._fh = tempfile.TemporaryFile()

    @property
    def active(self) -> bool:
        return self._fh is not None

    def arm(self, after_s: Optional[float] = None) -> None:
        """(Re)arm the one timer; arming again replaces the pending one."""
        if self._fh is not None:
            delay = self.after_s if after_s is None else after_s
            faulthandler.dump_traceback_later(delay, repeat=False, file=self._fh)

    def cancel(self) -> None:
        if self._fh is not None:
            faulthandler.cancel_dump_traceback_later()

    def take(self) -> str:
        """What the timer wrote since the last take (empty if it never fired)."""
        if self._fh is None:
            return ""
        fd = self._fh.fileno()
        size = os.fstat(fd).st_size
        if not size:
            return ""
        os.lseek(fd, 0, os.SEEK_SET)
        raw = os.read(fd, size)
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        return raw.decode("utf-8", "replace")

    def close(self) -> None:
        if self._fh is not None:
            faulthandler.cancel_dump_traceback_later()
            self._fh.close()
            self._fh = None
            _live_owner_lock.release()


def loop_depth(sample: str) -> int:
    """How many frames the ``[loop]`` stack in a ``sample_stacks`` string has
    (0 when the sample carries no loop stack)."""
    if not sample.startswith("[loop] "):
        return 0
    loop_part = sample[len("[loop] "):].split(" || ", 1)[0]
    return loop_part.count(" <- ") + 1 if loop_part else 0


def keep_deeper_sample(kept: str, snap: str) -> str:
    """Which of two samples taken during ONE stall to keep.

    A stall ends when the loop gets back to its own scheduler, so the last
    sample before the heartbeat is served is often the loop already idle in
    ``select`` / ``_read_from_self``: the frames that blocked it are gone. The
    loop stack is deepest while it is inside the call that blocks it, so keep
    the deeper loop stack; on a tie keep the newer one (most recent frames)."""
    if not snap:
        return kept
    return snap if loop_depth(snap) >= loop_depth(kept) else kept


class LagMonitor:
    """Measures how late the loop serves a heartbeat; optionally logs stalls."""

    def __init__(self, loop: asyncio.AbstractEventLoop, threshold_ms: float = _DEFAULT_MS,
                 tick_ms: float = _DEFAULT_TICK_MS, log: bool = True,
                 live_ms: float = 0.0) -> None:
        self.loop, self.threshold, self.tick = loop, threshold_ms / 1000, tick_ms / 1000
        self.log = log
        self.max_lag_ms = 0.0
        self.stalls: list[tuple[float, str]] = []
        # The loop runs on the main thread; _note_loop_thread refines this once
        # it is served. Seeding it (never None) means a stall sampled before
        # that call still resolves the loop stack instead of dropping it.
        self._loop_ident: Optional[int] = threading.main_thread().ident
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="loop-watchdog", daemon=True)
        self._last_log = 0.0
        #: Set each time a stall is recorded. Lets a waiter observe detection
        #: deterministically (event, not a wall-clock guess) instead of racing
        #: the sampler thread -- see tests/test_offload_and_loop_watchdog.
        self.stall_recorded = threading.Event()
        self._cpu_at = 0.0
        self._cpu_before: dict = {}
        self._proc_before = 0.0
        self._host_before: dict = {}
        #: In-stall capture (``LiveDump``); off unless ``live_ms`` > 0.
        self._live = LiveDump(live_ms) if live_ms > 0 else None
        #: Stalls whose live dump fired: (lag_ms, dump) -- what ran DURING it.
        #: Bounded to the LIVE_KEEP worst (``keep_worst``).
        self.live_dumps: list[tuple[float, str]] = []
        #: Worst oversleep of this thread past its tick: a GIL hold that began
        #: between heartbeats, which ``max_lag_ms`` cannot see (see
        #: ``_sleep_watched``). Reported beside the lag, never folded into it.
        self.max_wake_late_ms = 0.0
        #: ``(late_ms, monotonic_s_at_wake)`` of each wake-up at least
        #: ``WAKE_LATE_NOTE_MS`` late, the LIVE_KEEP worst. Plain arithmetic,
        #: no capture, so it is on in the gated measurement too.
        self.wake_late_at: list[tuple[float, float]] = []
        #: Heartbeats late past ``live_ms`` but served under the threshold:
        #: near misses, counted (their dump is not kept). A heartbeat already
        #: served never adds one -- serving it cancels the timer.
        self.live_near_misses = 0

    def start(self) -> "LagMonitor":
        self.loop.call_soon_threadsafe(self._note_loop_thread)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        if self._live is not None:
            self._live.close()

    def _note_loop_thread(self) -> None:
        self._loop_ident = threading.get_ident()

    def _run(self) -> None:
        while not self._stop.is_set():
            served = threading.Event()
            served_at: list[float] = []

            live = self._live

            def _serve(served=served, served_at=served_at, live=live) -> None:
                served_at.append(time.monotonic())
                if live is not None:
                    # Re-armed from the serve, not cancelled: the monitor needs
                    # the GIL to arm its sleep timer, so a hold that begins
                    # between this serve and that arm was invisible (CI #396
                    # 9f775c1: a 0.6 s hold, max_lag 1, no dump, no wake-late).
                    live.arm()
                served.set()

            # The heartbeat is stamped BEFORE the monitor's CPU snapshot, as
            # #380 shipped it: what the timing test measures is unchanged here
            # (advisor ruling 04:18Z; moving the stamp is its own proposal).
            # The live timer is armed with the stamp, so a GIL hold that
            # begins while the snapshot reads /proc -- each read releases the
            # GIL -- is dumped too. Armed after the snapshot, that window was
            # blind (CI #396 d3c7ef0: a 0.6 s hold, max_lag 2, no dump).
            posted = time.monotonic()
            if live is not None:
                # Names now: a thread in the dump may have exited by the time
                # the stall ends and the dump is formatted.
                armed_names = {t.ident: t.name for t in threading.enumerate()}
                live.arm()
            if posted - self._cpu_at >= _CPU_SNAPSHOT_S:
                self._proc_before, self._cpu_before = cpu_snapshot()
                self._cpu_at, self._host_before = posted, host_cpu()
            try:
                self.loop.call_soon_threadsafe(_serve)
            except RuntimeError:
                _LOG.info("loop watchdog stopped: the event loop is closed")
                self._stop.set()
                break
            sample = ""
            stalled = not served.wait(self.threshold)
            resumed = time.monotonic()
            if live is not None and not stalled:
                # Back with the GIL: the gap the serve re-armed for is over.
                # Cancelled here, the monitor's own work below never fires it.
                live.cancel()
            if stalled:
                # Sample now and keep re-sampling while the stall persists, so
                # the stack we keep is the loop caught in the act (the deepest,
                # most recent frames), not just the instant the threshold passed.
                sample = sample_stacks(self._loop_ident)
                while not served.wait(0.05):
                    if self._stop.is_set():
                        return
                    sample = keep_deeper_sample(sample, sample_stacks(self._loop_ident))
            lag_ms = heartbeat_lag_ms(posted, served_at[0] if served_at else None,
                                      time.monotonic())
            self.max_lag_ms = max(self.max_lag_ms, lag_ms)
            # The loop's own lateness decides: a heartbeat served past the
            # threshold is a stall even if this thread woke too late to sample
            # it during the stall (it then samples below).
            stalled = stalled or lag_ms >= self.threshold * 1000
            if stalled:
                # A detected stall is ALWAYS recorded with its stack. sample is
                # never empty (sample_stacks falls back to the main thread), but
                # resample once more as a belt-and-braces guard so the worst
                # stall can never carry an empty trace.
                if not sample:
                    sample = sample_stacks(self._loop_ident)
                # Which threads burned CPU across the stall: the GIL holder
                # when the loop itself was idle-but-late. Leads the sample, so
                # a truncated message still names it.
                names = {t.native_id: t.name for t in threading.enumerate()}
                proc_now, threads_now = cpu_snapshot()
                cpu = busiest_threads(self._cpu_before, threads_now, names)
                # Process- and host-wide totals over the same window, so the
                # line itself tells a GIL convoy from a starved runner.
                total = stall_cpu_summary(
                    self._cpu_before, threads_now, proc_now - self._proc_before,
                    self._host_before, host_cpu(), (time.monotonic() - self._cpu_at) * 1000)
                sample = f"{cpu} {total} :: {sample}" if cpu else f"{total} :: {sample}"
                dump = live.take() if live is not None else ""
                if dump:
                    idents = {**armed_names, **{t.ident: t.name for t in threading.enumerate()}}
                    during = format_live_dump(dump, idents)
                    keep_worst(self.live_dumps, (lag_ms, during))
                    sample = f"{sample} :: DURING {during}"
                self.stalls.append((lag_ms, sample))
                now = time.monotonic()
                if self.log and now - self._last_log >= _MIN_LOG_GAP_S:
                    self._last_log = now
                    _LOG.warning("LOOP_STALL lag=%dms %s", lag_ms, sample)
                # Recorded and (if logging) logged: signal last so a waiter that
                # sees the event also sees both.
                self.stall_recorded.set()
            elif live is not None:
                dump = live.take()
                # How long past the serve this thread got the GIL back: a hold
                # that began after the serve shows here, not in lag_ms.
                gap_ms = (resumed - served_at[0]) * 1000 if served_at else 0.0
                if dump and gap_ms >= live.after_s * 1000:
                    self.max_wake_late_ms = max(self.max_wake_late_ms, gap_ms)
                    self._note_wake_late(gap_ms, resumed)
                    idents = {**armed_names, **{t.ident: t.name for t in threading.enumerate()}}
                    keep_worst(self.live_dumps, (gap_ms, "WAKE_LATE " + format_live_dump(dump, idents)))
                elif dump:  # late past live_ms, under the limit
                    self.live_near_misses += 1
            self._sleep_watched(live, armed_names if live is not None else {})

    def _note_wake_late(self, late_ms: float, at: float) -> None:
        """Keep a wake-up at least ``WAKE_LATE_NOTE_MS`` late with its time."""
        if late_ms >= WAKE_LATE_NOTE_MS:
            keep_worst(self.wake_late_at, (round(late_ms, 1), at))

    def _sleep_watched(self, live: Optional[LiveDump], armed_names: dict) -> None:
        """Sleep one tick, and catch a GIL hold that starts during the sleep.

        This thread is Python: a call that holds the GIL from inside this
        sleep keeps it asleep too, and when the call returns the next
        heartbeat is posted and served at once -- lag 0, a stall the
        heartbeat never sees. So the C timer stays armed across the sleep
        (tick + live), and the oversleep itself is measured: ``wake_late``
        is how long past its tick this thread could not run."""
        if live is not None:
            live.arm(self.tick + live.after_s)
        slept_from = time.monotonic()
        self._stop.wait(self.tick)
        woke = time.monotonic()
        late_ms = max(0.0, (woke - slept_from - self.tick) * 1000)
        self.max_wake_late_ms = max(self.max_wake_late_ms, late_ms)
        self._note_wake_late(late_ms, woke)
        if live is not None:
            live.cancel()
            dump = live.take()
            if dump:
                idents = {**armed_names, **{t.ident: t.name for t in threading.enumerate()}}
                keep_worst(self.live_dumps, (late_ms, "WAKE_LATE " + format_live_dump(dump, idents)))


def start(loop: Optional[asyncio.AbstractEventLoop] = None) -> Optional[LagMonitor]:
    """Start the process-wide watchdog once (from the running loop's lifespan)."""
    global _started
    threshold = _env_float("LOOP_WATCHDOG_MS", _DEFAULT_MS)
    if threshold <= 0 or _started is not None:
        return _started
    loop = loop or asyncio.get_running_loop()
    _started = LagMonitor(loop, threshold, _env_float("LOOP_WATCHDOG_TICK_MS", _DEFAULT_TICK_MS)).start()
    _LOG.info("loop watchdog on: logs LOOP_STALL when the event loop is %dms late", threshold)
    return _started
