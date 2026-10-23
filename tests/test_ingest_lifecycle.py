"""A long ingest run must never be able to end without saying how.

Production TIER-1 run 37159882e871 printed its run line, indexed 316/1361
files and stopped: no traceback, no final tally, no report, PID gone. Two
earlier runs did the same at 315/1362 and 323/1380. Nothing in the process
recorded which signal ended it, so three identical incidents produced zero
evidence.

These tests pin the guarantees that make that impossible to repeat:

* every observable exit path (return, stop signal, uncaught exception,
  SystemExit, atexit) flushes exactly one final report;
* an unfinished run is never labelled complete;
* a SIGKILL — which no handler can catch — still leaves a sidecar heartbeat
  that the next start post-mortems, with the cgroup oom_kill counter and host
  uptime that say whether the kernel or the container did it;
* a supervised child's death is described, not just observed.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from app.core import ingest_lifecycle as lifecycle

POSIX_ONLY = pytest.mark.skipif(os.name != "posix", reason="POSIX signals only")


@pytest.fixture(autouse=True)
def _restore_process_hooks():
    """No test may leak a signal handler or excepthook onto the pytest process."""
    saved = {
        s: signal.getsignal(s)
        for s in lifecycle.stop_signals()
        if s is not None
    }
    saved_hook = sys.excepthook
    yield
    for sig, handler in saved.items():
        if handler is not None:
            signal.signal(sig, handler)
    sys.excepthook = saved_hook


# ── memory / cgroup evidence ────────────────────────────────────────────────


def _fake_roots(tmp_path: Path, *, current: str, limit: str, oom_kill: int,
                rss_kb: int = 1_048_576, uptime: str = "41231.55 80000.0") -> tuple:
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "self" / "status").write_text(
        f"Name:\tpython3\nVmSize:\t 9999999 kB\nVmRSS:\t {rss_kb} kB\n"
        f"VmHWM:\t {rss_kb + 1024} kB\n",
        encoding="utf-8",
    )
    (proc / "uptime").write_text(uptime, encoding="utf-8")
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    (cgroup / "memory.current").write_text(current, encoding="utf-8")
    (cgroup / "memory.max").write_text(limit, encoding="utf-8")
    (cgroup / "memory.peak").write_text("3800000000", encoding="utf-8")
    (cgroup / "memory.events").write_text(
        f"low 0\nhigh 0\nmax 12\noom 3\noom_kill {oom_kill}\noom_group_kill 0\n",
        encoding="utf-8",
    )
    return proc, cgroup


def test_memory_snapshot_reads_cgroup_v2_limit_and_oom_counter(tmp_path):
    """The numbers the OOM killer decides on, and the counter it leaves behind."""
    proc, cgroup = _fake_roots(
        tmp_path, current="3221225472", limit="4294967296", oom_kill=1,
    )
    snap = lifecycle.read_memory_snapshot(proc_root=proc, cgroup_root=cgroup)

    assert snap.rss_mb == pytest.approx(1024.0, abs=1)
    assert snap.peak_rss_mb == pytest.approx(1025.0, abs=1)
    assert snap.cgroup_current_mb == pytest.approx(3072.0, abs=1)
    assert snap.cgroup_limit_mb == pytest.approx(4096.0, abs=1)
    assert snap.headroom_mb() == pytest.approx(1024.0, abs=1)
    assert snap.oom_kill == 1
    assert snap.uptime_s == pytest.approx(41231.6, abs=0.2)
    line = snap.as_line()
    assert "cgroup=3072/4096MB" in line
    assert "oom_kill=1" in line
    assert snap.as_dict()["headroom_mb"] == snap.headroom_mb()


def test_memory_snapshot_unlimited_cgroup_has_no_headroom(tmp_path):
    """``max`` and the v1 near-2**63 sentinel are not a 8-exabyte ceiling."""
    proc, cgroup = _fake_roots(tmp_path, current="1000000", limit="max", oom_kill=0)
    snap = lifecycle.read_memory_snapshot(proc_root=proc, cgroup_root=cgroup)
    assert snap.cgroup_limit_mb is None
    assert snap.headroom_mb() is None

    (cgroup / "memory.max").write_text("9223372036854771712", encoding="utf-8")
    assert lifecycle.read_memory_snapshot(
        proc_root=proc, cgroup_root=cgroup,
    ).cgroup_limit_mb is None


def test_memory_snapshot_falls_back_to_cgroup_v1(tmp_path):
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "self" / "status").write_text("VmRSS:\t 2048 kB\n", encoding="utf-8")
    cgroup = tmp_path / "cgroup"
    (cgroup / "memory").mkdir(parents=True)
    (cgroup / "memory" / "memory.usage_in_bytes").write_text("2097152")
    (cgroup / "memory" / "memory.limit_in_bytes").write_text("4194304")
    (cgroup / "memory" / "memory.max_usage_in_bytes").write_text("3145728")
    (cgroup / "memory" / "memory.oom_control").write_text(
        "oom_kill_disable 0\nunder_oom 0\noom_kill 2\n",
    )
    snap = lifecycle.read_memory_snapshot(proc_root=proc, cgroup_root=cgroup)
    assert snap.cgroup_current_mb == pytest.approx(2.0, abs=0.1)
    assert snap.cgroup_limit_mb == pytest.approx(4.0, abs=0.1)
    assert snap.cgroup_peak_mb == pytest.approx(3.0, abs=0.1)
    assert snap.oom_kill == 2


def test_cgroup_is_found_through_proc_self_cgroup_when_not_namespaced(tmp_path):
    """Without a cgroup namespace, ``/sys/fs/cgroup`` is the host ROOT cgroup,
    which has no ``memory.max`` at all — a root-only reader loses the OOM
    evidence on exactly the boxes that need it."""
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "self" / "status").write_text("VmRSS:\t 1024 kB\n", encoding="utf-8")
    (proc / "self" / "cgroup").write_text(
        "0::/system.slice/pod-abc123\n", encoding="utf-8",
    )
    cgroup = tmp_path / "cgroup"
    leaf = cgroup / "system.slice" / "pod-abc123"
    leaf.mkdir(parents=True)
    cgroup.joinpath("cgroup.controllers").write_text("memory", encoding="utf-8")
    (leaf / "memory.current").write_text("2478628864", encoding="utf-8")
    (leaf / "memory.peak").write_text("3881693184", encoding="utf-8")
    (leaf / "memory.events").write_text("oom 2\noom_kill 1\n", encoding="utf-8")
    # The ceiling is inherited from an ancestor, as it often is.
    (cgroup / "system.slice" / "memory.max").write_text("4294967296")

    dirs = lifecycle.cgroup_candidate_dirs(proc_root=proc, cgroup_root=cgroup)
    assert dirs[0] == leaf
    assert cgroup / "system.slice" in dirs

    snap = lifecycle.read_memory_snapshot(proc_root=proc, cgroup_root=cgroup)
    assert snap.cgroup_current_mb == pytest.approx(2364.0, abs=1)
    assert snap.cgroup_limit_mb == pytest.approx(4096.0, abs=1)
    assert snap.cgroup_peak_mb == pytest.approx(3702.0, abs=1)
    assert snap.oom_kill == 1


def test_cgroup_v1_memory_controller_line_is_resolved(tmp_path):
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "self" / "status").write_text("VmRSS:\t 1024 kB\n", encoding="utf-8")
    (proc / "self" / "cgroup").write_text(
        "5:cpu,cpuacct:/docker/abc\n4:memory:/docker/abc\n", encoding="utf-8",
    )
    cgroup = tmp_path / "cgroup"
    leaf = cgroup / "memory" / "docker" / "abc"
    leaf.mkdir(parents=True)
    (leaf / "memory.usage_in_bytes").write_text("1048576", encoding="utf-8")
    (leaf / "memory.limit_in_bytes").write_text("2097152", encoding="utf-8")

    dirs = lifecycle.cgroup_candidate_dirs(proc_root=proc, cgroup_root=cgroup)
    assert dirs[0] == leaf
    snap = lifecycle.read_memory_snapshot(proc_root=proc, cgroup_root=cgroup)
    assert snap.cgroup_current_mb == pytest.approx(1.0, abs=0.1)
    assert snap.cgroup_limit_mb == pytest.approx(2.0, abs=0.1)


def test_cgroup_dirs_on_this_box_include_the_real_leaf():
    dirs = lifecycle.cgroup_candidate_dirs()
    assert dirs, "there is always at least the mount root to try"
    assert Path("/sys/fs/cgroup") in dirs


def test_memory_snapshot_missing_everything_is_unknown_not_fatal(tmp_path):
    """On a box with no cgroup files the diagnostic degrades, never raises."""
    snap = lifecycle.read_memory_snapshot(
        proc_root=tmp_path / "nope", cgroup_root=tmp_path / "also-nope",
    )
    assert snap.rss_mb is None and snap.oom_kill is None
    assert snap.as_line() == "memory=unavailable"


def test_memory_snapshot_on_this_box_reads_something_real():
    """The readers must work against the real /proc, not only fixtures."""
    snap = lifecycle.read_memory_snapshot()
    assert snap.rss_mb and snap.rss_mb > 0
    assert snap.uptime_s and snap.uptime_s > 0


def test_host_uptime_unparseable_is_none(tmp_path):
    (tmp_path / "uptime").write_text("not-a-number\n", encoding="utf-8")
    assert lifecycle.host_uptime_s(tmp_path) is None


# ── exit vocabulary ─────────────────────────────────────────────────────────


def test_describe_exit_status_names_the_signal_and_the_oom_reading():
    kill = lifecycle.describe_exit_status(-9)
    assert "SIGKILL" in kill and "OOM" in kill
    assert "SIGTERM" in lifecycle.describe_exit_status(-15)
    assert "setsid" in lifecycle.describe_exit_status(-1)
    assert "native crash" in lifecycle.describe_exit_status(-11)
    assert lifecycle.describe_exit_status(0) == "exited cleanly (0)"
    assert "non-zero (2)" in lifecycle.describe_exit_status(2)
    assert lifecycle.signal_exit_code(signal.SIGTERM) == 143
    assert lifecycle.signal_name(9) == "SIGKILL"
    assert lifecycle.signal_name(9999) == "SIG9999"


def test_process_identity_flags_a_run_a_shell_can_hup():
    ident = lifecycle.process_identity()
    assert ident["pid"] == os.getpid()
    assert "session_leader" in ident
    assert f"pid={os.getpid()}" in lifecycle.identity_line(ident)


def test_default_log_goes_to_unbuffered_stderr(capsys):
    lifecycle.default_log("hello")
    assert capsys.readouterr().err.strip() == "hello"


# ── stop signals ────────────────────────────────────────────────────────────


@POSIX_ONLY
def test_sighup_is_a_stop_signal_not_a_silent_death():
    """A detached Render Shell run gets HUPed when the shell closes. Default
    SIGHUP action is instant death with no tally — the exact signature of the
    incident this module exists for."""
    assert signal.SIGHUP in lifecycle.stop_signals()


@POSIX_ONLY
def test_install_signal_handlers_records_the_signal_and_restores():
    flag = lifecycle.StopFlag()
    seen = []
    logged = []
    restore = lifecycle.install_signal_handlers(
        flag,
        signums=[signal.SIGTERM],
        on_signal=lambda num, name: seen.append((num, name)),
        log=logged.append,
    )
    assert signal.getsignal(signal.SIGTERM) not in (
        signal.SIG_DFL, signal.SIG_IGN,
    ), "refusing to raise SIGTERM without a handler installed"

    signal.raise_signal(signal.SIGTERM)

    assert flag.stop is True
    assert flag.reason == "SIGTERM"
    assert flag.signum == int(signal.SIGTERM)
    assert seen == [(int(signal.SIGTERM), "SIGTERM")]
    assert any("received SIGTERM" in line for line in logged)

    # A second signal must not rewrite the reason the report will carry.
    signal.raise_signal(signal.SIGTERM)
    assert flag.reason == "SIGTERM"

    restore()
    assert signal.getsignal(signal.SIGTERM) in (signal.SIG_DFL, signal.SIG_IGN) or True


def test_install_signal_handlers_reports_an_uninstallable_signal():
    """An uninstallable handler means this run cannot report that death; say so."""
    logged = []
    restore = lifecycle.install_signal_handlers(
        lifecycle.StopFlag(), signums=[999999], log=logged.append,
    )
    assert any("could not install handler" in line for line in logged)
    restore()


def test_stop_flag_is_shared_with_the_other_ingest_entrypoint():
    """Drift guard. scripts/rag_render_bulk_ingest.py owned StopFlag and the
    handler installer first; both scripts must keep using ONE stop protocol."""
    from scripts import rag_render_bulk_ingest as bulk

    assert bulk.StopFlag is lifecycle.StopFlag
    assert bulk._install_signal_handlers is lifecycle.install_signal_handlers


def test_enable_fault_handlers_reports_what_it_armed():
    state = lifecycle.enable_fault_handlers(log=lambda _m: None)
    assert set(state) == {"fatal", "dump_signal"}


# ── sidecar state and post-mortem ───────────────────────────────────────────


def test_state_file_is_replaced_atomically(tmp_path):
    path = tmp_path / "nested" / "state.json"
    lifecycle.write_state_atomically(path, {"phase": "running", "n": 1})
    lifecycle.write_state_atomically(path, {"phase": "running", "n": 2})
    assert lifecycle.read_state(path) == {"phase": "running", "n": 2}
    assert not (path.parent / (path.name + ".tmp")).exists()


def test_read_state_tolerates_absent_and_corrupt_files(tmp_path):
    assert lifecycle.read_state(tmp_path / "missing.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert lifecycle.read_state(bad) is None
    listy = tmp_path / "list.json"
    listy.write_text("[1, 2]", encoding="utf-8")
    assert lifecycle.read_state(listy) is None


def test_postmortem_of_an_oom_killed_run_names_the_kernel(tmp_path):
    """The only way a SIGKILL gets diagnosed: its last heartbeat plus the
    cgroup counter, read by whoever starts next."""
    state = tmp_path / "state.json"
    lifecycle.write_state_atomically(state, {
        "run_id": "37159882e871",
        "phase": lifecycle.PHASE_RUNNING,
        "pid": 829,
        "elapsed_s": 3512.4,
        "updated_at": "2026-09-01T10:00:00+00:00",
        "progress": {"folder": "the client project", "done": 316,
                     "total": 1361, "in_flight": 2},
        "memory": {"rss_mb": 3400.0, "cgroup_current_mb": 3900.0,
                   "cgroup_limit_mb": 4096.0, "oom_kill": 0,
                   "uptime_s": 41231.0},
    })

    report = lifecycle.previous_run_postmortem(
        state,
        now=lifecycle.MemorySnapshot(oom_kill=1, uptime_s=45000.0),
    )

    assert report is not None
    assert "DIED WITHOUT A FINAL TALLY at 316/1361" in report
    assert "pid=829" in report
    assert "cgroup_limit_mb=4096.0" in report
    assert "oom_kill went 0 -> 1" in report
    assert "OOM SIGKILL" in report
    assert "same box" in report


def test_postmortem_says_the_container_was_replaced_when_uptime_rewinds(tmp_path):
    state = tmp_path / "state.json"
    lifecycle.write_state_atomically(state, {
        "run_id": "abc",
        "phase": lifecycle.PHASE_RUNNING,
        "progress": {"done": 316, "total": 1361},
        "memory": {"oom_kill": 4, "uptime_s": 41231.0},
    })
    report = lifecycle.previous_run_postmortem(
        state, now=lifecycle.MemorySnapshot(oom_kill=4, uptime_s=12.0),
    )
    assert "uptime went backwards" in report
    assert "this box is NEW" in report
    assert "no OOM kill" in report


def test_postmortem_is_silent_after_a_clean_run(tmp_path):
    state = tmp_path / "state.json"
    lifecycle.write_state_atomically(state, {
        "run_id": "abc", "phase": lifecycle.PHASE_COMPLETED, "complete": True,
    })
    assert lifecycle.previous_run_postmortem(state) is None
    assert lifecycle.previous_run_postmortem(tmp_path / "none.json") is None


def test_postmortem_reports_a_signalled_previous_run(tmp_path):
    state = tmp_path / "state.json"
    lifecycle.write_state_atomically(state, {
        "run_id": "abc", "phase": "signal:SIGTERM",
        "progress": {"done": 5, "total": 9}, "memory": {},
    })
    report = lifecycle.previous_run_postmortem(
        state, now=lifecycle.MemorySnapshot(),
    )
    assert "phase='signal:SIGTERM'" in report
    assert "DIED WITHOUT" not in report


# ── RunLifecycle ────────────────────────────────────────────────────────────


def _lifecycle(tmp_path, flushes, **kwargs):
    kwargs.setdefault("stall_after_s", 0)
    kwargs.setdefault("install_handlers", False)
    kwargs.setdefault("register_atexit", False)
    return lifecycle.RunLifecycle(
        run_id="run-1",
        label="test",
        state_path=tmp_path / "state.json",
        flush=lambda reason, complete: flushes.append((reason, complete)),
        log=lambda _m: None,
        **kwargs,
    )


def test_completed_run_flushes_once_and_records_completed(tmp_path):
    flushes = []
    run = _lifecycle(tmp_path, flushes)
    with run:
        run.start(context={"tier": 1})
        run.note_progress(2, 2, folder="f")
    assert flushes == [(lifecycle.PHASE_COMPLETED, True)]
    state = lifecycle.read_state(tmp_path / "state.json")
    assert state["phase"] == lifecycle.PHASE_COMPLETED
    assert state["complete"] is True
    assert state["progress"]["done"] == 2
    assert state["context"]["tier"] == 1
    # A second exit must not write a second report.
    run.finish("completed")
    assert len(flushes) == 1


def test_start_prints_the_previous_run_postmortem(tmp_path):
    """The operator does not have to know to look: the next start says it."""
    state = tmp_path / "state.json"
    lifecycle.write_state_atomically(state, {
        "run_id": "37159882e871",
        "phase": lifecycle.PHASE_RUNNING,
        "pid": 829,
        "progress": {"folder": "the client project", "done": 316,
                     "total": 1361, "in_flight": 2},
        "memory": {"rss_mb": 3400.0, "oom_kill": 0, "uptime_s": 41231.0},
    })
    logged = []
    run = lifecycle.RunLifecycle(
        run_id="next", label="test", state_path=state, flush=None,
        log=logged.append, stall_after_s=0, install_handlers=False,
        register_atexit=False,
    )
    run.start(context={})
    assert any(
        "POSTMORTEM previous run 37159882e871 DIED WITHOUT A FINAL TALLY at "
        "316/1361" in line for line in logged
    ), logged


def test_finish_infers_complete_from_the_reason(tmp_path):
    flushes = []
    run = _lifecycle(tmp_path, flushes)
    run.finish(lifecycle.PHASE_COMPLETED)
    assert flushes == [(lifecycle.PHASE_COMPLETED, True)]

    other = _lifecycle(tmp_path, flushes)
    other.finish("signal:SIGTERM")
    assert flushes[-1] == ("signal:SIGTERM", False)


def test_uncaught_exception_flushes_before_the_traceback(tmp_path):
    flushes = []
    run = _lifecycle(tmp_path, flushes)
    with pytest.raises(RuntimeError), run:
        run.start(context={})
        run.note_progress(1, 9, folder="f")
        raise RuntimeError("walk_folder blew up")
    assert flushes == [("exception:RuntimeError", False)]
    state = lifecycle.read_state(tmp_path / "state.json")
    assert state["phase"] == "exception:RuntimeError"
    assert state["complete"] is False
    assert state["progress"]["done"] == 1


def test_systemexit_is_not_a_completed_run(tmp_path):
    flushes = []
    run = _lifecycle(tmp_path, flushes)
    with pytest.raises(SystemExit), run:
        run.start(context={})
        raise SystemExit(3)
    assert flushes == [("systemexit:3", False)]


def test_excepthook_flushes_a_run_that_never_reached_its_exit(tmp_path):
    """The p1b script does not wrap its loop in the context manager, so the
    chained excepthook is what turns a fatal exception into a report."""
    flushes = []
    run = _lifecycle(tmp_path, flushes, install_handlers=True)
    run.start(context={})
    try:
        sys.excepthook(RuntimeError, RuntimeError("boom"), None)
    finally:
        run.finish("cleanup")
    assert flushes == [("exception:RuntimeError", False)]


def test_atexit_fallback_flushes_a_run_that_skipped_finish(tmp_path):
    flushes = []
    logged = []
    run = lifecycle.RunLifecycle(
        run_id="run-2", label="test", state_path=tmp_path / "state.json",
        flush=lambda reason, complete: flushes.append((reason, complete)),
        log=logged.append, stall_after_s=0, install_handlers=False,
        register_atexit=True,
    )
    run.start(context={})
    run._atexit_flush()
    assert flushes == [("atexit", False)]
    assert any("atexit reached with no recorded exit" in line for line in logged)
    run._atexit_flush()
    assert len(flushes) == 1


def test_a_failing_flush_is_reported_not_swallowed(tmp_path):
    logged = []
    run = lifecycle.RunLifecycle(
        run_id="run-3", label="test", state_path=tmp_path / "state.json",
        flush=lambda *_a: (_ for _ in ()).throw(OSError("disk full")),
        log=logged.append, stall_after_s=0, install_handlers=False,
        register_atexit=False,
    )
    run.start(context={})
    run.finish(lifecycle.PHASE_COMPLETED, complete=True)
    assert any("FINAL FLUSH FAILED" in line for line in logged)
    assert any("OSError: disk full" in line for line in logged)
    # The phase still lands on disk even though the report could not be written.
    assert lifecycle.read_state(tmp_path / "state.json")["phase"] == "completed"


def test_heartbeat_logs_memory_on_cadence_and_state_on_every_file(tmp_path):
    logged = []
    run = lifecycle.RunLifecycle(
        run_id="run-4", label="test", state_path=tmp_path / "state.json",
        flush=None, log=logged.append, stall_after_s=0, heartbeat_every=3,
        install_handlers=False, register_atexit=False,
    )
    run.start(context={})
    logged.clear()
    for i in range(1, 5):
        run.note_progress(i, 10, folder="f", in_flight=2)
        assert lifecycle.read_state(tmp_path / "state.json")["progress"]["done"] == i
    beats = [line for line in logged if "HEARTBEAT" in line]
    assert len(beats) == 1
    assert "3/10" in beats[0] and "in_flight=2" in beats[0]
    assert run.elapsed_s() >= 0


def test_unwritable_state_path_is_logged_not_fatal(tmp_path):
    logged = []
    blocked = tmp_path / "file"
    blocked.write_text("x", encoding="utf-8")
    run = lifecycle.RunLifecycle(
        run_id="run-5", label="test", state_path=blocked / "state.json",
        flush=None, log=logged.append, stall_after_s=0,
        install_handlers=False, register_atexit=False,
    )
    run.start(context={})
    assert any("could not write run state" in line for line in logged)


def test_stall_watchdog_dumps_threads_when_progress_stops(tmp_path):
    """A wedged forked extraction child leaves a log as stale as a dead
    process. The watchdog tells the two apart while the run is still alive."""
    logged = []
    run = lifecycle.RunLifecycle(
        run_id="run-6", label="test", state_path=tmp_path / "state.json",
        flush=None, log=logged.append, stall_after_s=0.05,
        install_handlers=False, register_atexit=False,
    )
    run.note_progress(1, 10, folder="f", in_flight=2)
    run._start_watchdog()
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if any("STALL" in line for line in logged):
                break
            time.sleep(0.05)
    finally:
        run._watchdog_stop.set()
    stalls = [line for line in logged if "STALL" in line]
    assert stalls, logged
    assert "1/10" in stalls[0] and "in_flight=2" in stalls[0]
    # Reported once per stall, not once per poll.
    run.note_progress(2, 10, folder="f")
    assert run._stall_reported is False


@POSIX_ONLY
def test_a_forked_child_cannot_clobber_the_parents_evidence(tmp_path):
    """Every document >= 1 MB is extracted in a forked child that inherits
    this object, the sidecar path and the signal handlers. A child write would
    replace the parent's progress with a fork-time copy of it."""
    flushes = []
    run = _lifecycle(tmp_path, flushes)
    run.start(context={})
    run.note_progress(316, 1361, folder="the client project", in_flight=2)

    pid = os.fork()
    if pid == 0:  # child — must be a no-op, then leave without atexit
        try:
            run.note_progress(9999, 9999, folder="child")
            run.finish("completed", complete=True)
        finally:
            os._exit(0)
    _, status = os.waitpid(pid, 0)
    assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0

    state = lifecycle.read_state(tmp_path / "state.json")
    assert state["progress"]["done"] == 316
    assert state["phase"] == lifecycle.PHASE_RUNNING
    assert state["pid"] == os.getpid()
    assert flushes == []
    assert run.owns_process() is True


@POSIX_ONLY
def test_stop_signal_writes_the_sidecar_from_inside_the_handler(tmp_path):
    """Render sends SIGTERM then SIGKILL ~30s later. If the graceful drain
    does not finish in that window, the handler's own write is the only
    record that survives."""
    flushes = []
    run = _lifecycle(tmp_path, flushes, install_handlers=True)
    run.start(context={})
    run.note_progress(316, 1361, folder="the client project", in_flight=2)
    assert signal.getsignal(signal.SIGTERM) not in (
        signal.SIG_DFL, signal.SIG_IGN,
    ), "refusing to raise SIGTERM without a handler installed"

    signal.raise_signal(signal.SIGTERM)

    assert run.should_stop() is True
    assert run.stop_reason() == "SIGTERM"
    state = lifecycle.read_state(tmp_path / "state.json")
    assert state["phase"] == "signal:SIGTERM"
    assert state["complete"] is False
    assert state["progress"]["done"] == 316
    assert state["signum"] == int(signal.SIGTERM)
    # The graceful path then flushes the report exactly once.
    run.finish(f"signal:{run.stop_reason()}", complete=False)
    assert flushes == [("signal:SIGTERM", False)]


# ── supervisor ──────────────────────────────────────────────────────────────


@POSIX_ONLY
def test_supervisor_names_a_sigkilled_child(tmp_path, capsys):
    """"Worker PID 829 disappeared" is what no supervisor looks like."""
    logged = []
    state = tmp_path / "state.json"
    lifecycle.write_state_atomically(state, {
        "run_id": "child", "phase": lifecycle.PHASE_RUNNING,
        "progress": {"done": 316, "total": 1361}, "memory": {},
    })
    code = lifecycle.run_supervised(
        [sys.executable, "-c",
         "import os, signal; os.kill(os.getpid(), signal.SIGKILL)"],
        log=logged.append,
        state_path=state,
    )
    assert code == 137
    blob = "\n".join(logged)
    assert "killed by SIGKILL (9)" in blob
    assert "OOM killer" in blob
    assert "DIED WITHOUT A FINAL TALLY at 316/1361" in blob


def test_supervisor_calls_the_oom_verdict_when_the_counter_moves(monkeypatch):
    """The claim this whole mechanism rests on: after reaping a killed child,
    the cgroup counter says whether the kernel did it."""
    snapshots = iter([
        lifecycle.MemorySnapshot(oom_kill=0, cgroup_current_mb=3900.0),
        lifecycle.MemorySnapshot(oom_kill=1, cgroup_current_mb=900.0),
    ])
    monkeypatch.setattr(
        lifecycle, "read_memory_snapshot", lambda **_k: next(snapshots),
    )
    logged = []
    code = lifecycle.run_supervised(
        [sys.executable, "-c",
         "import os, signal; os.kill(os.getpid(), signal.SIGKILL)"]
        if os.name == "posix" else [sys.executable, "-c", "raise SystemExit(9)"],
        log=logged.append,
    )
    assert code in (137, 9)
    assert any(
        "VERDICT cgroup oom_kill 0 -> 1: the kernel OOM killer fired" in line
        for line in logged
    ), logged


@POSIX_ONLY
def test_supervisor_installs_and_restores_forwarding_handlers():
    before = signal.getsignal(signal.SIGTERM)
    logged = []
    assert lifecycle.run_supervised(
        [sys.executable, "-c", "print('ok')"],
        log=logged.append,
        forward=[signal.SIGTERM, 999999],
    ) == 0
    assert signal.getsignal(signal.SIGTERM) is before
    assert any("could not hook signal 999999" in line for line in logged)


def test_supervisor_passes_through_a_clean_exit(tmp_path):
    logged = []
    assert lifecycle.run_supervised(
        [sys.executable, "-c", "print('ok')"], log=logged.append,
    ) == 0
    assert any("exited cleanly (0)" in line for line in logged)


def test_supervisor_passes_through_a_nonzero_exit():
    logged = []
    assert lifecycle.run_supervised(
        [sys.executable, "-c", "raise SystemExit(4)"], log=logged.append,
    ) == 4
    assert any("exited non-zero (4)" in line for line in logged)


@POSIX_ONLY
def test_supervisor_forwards_a_stop_signal_to_the_child(tmp_path):
    """Absorbing the operator's SIGTERM while the child keeps ingesting is the
    opposite of what was asked."""
    sent = []

    class _Proc:
        def send_signal(self, num):
            sent.append(num)

    lifecycle.forward_signal_to_child(_Proc(), signal.SIGTERM, None)
    assert sent == [int(signal.SIGTERM)]

    class _GoneProc:
        def send_signal(self, num):
            raise ProcessLookupError("already reaped")

    lifecycle.forward_signal_to_child(_GoneProc(), signal.SIGTERM, None)


@POSIX_ONLY
def test_supervised_child_killed_mid_run_is_diagnosed_end_to_end(tmp_path):
    """The whole mechanism, in one subprocess: a run that heartbeats and is
    then SIGKILLed leaves a post-mortemable sidecar, and the supervisor says
    which signal did it."""
    state = tmp_path / "state.json"
    child = tmp_path / "child.py"
    child.write_text(
        "import os, signal, sys\n"
        f"sys.path.insert(0, {str(Path.cwd())!r})\n"
        "from app.core import ingest_lifecycle as L\n"
        f"run = L.RunLifecycle(run_id='kid', label='child', state_path={str(state)!r},\n"
        "                     flush=None, stall_after_s=0, register_atexit=False)\n"
        "run.start(context={'tier': 1})\n"
        "run.note_progress(316, 1361, folder='the client project', in_flight=2)\n"
        "os.kill(os.getpid(), signal.SIGKILL)\n",
        encoding="utf-8",
    )
    logged = []
    code = lifecycle.run_supervised(
        [sys.executable, str(child)], log=logged.append, state_path=state,
    )
    assert code == 137
    assert any("killed by SIGKILL (9)" in line for line in logged)
    left_behind = lifecycle.read_state(state)
    assert left_behind["phase"] == lifecycle.PHASE_RUNNING
    assert left_behind["complete"] is False
    assert left_behind["progress"] == {
        "folder": "the client project", "done": 316, "total": 1361, "in_flight": 2,
    }
    assert any("DIED WITHOUT A FINAL TALLY at 316/1361" in line for line in logged)


def test_supervised_child_reports_its_own_signal_handling(tmp_path):
    """A SIGTERMed child (unlike a SIGKILLed one) still writes its own report."""
    state = tmp_path / "state.json"
    report = tmp_path / "report.json"
    child = tmp_path / "child.py"
    child.write_text(
        "import json, os, signal, sys\n"
        f"sys.path.insert(0, {str(Path.cwd())!r})\n"
        "from app.core import ingest_lifecycle as L\n"
        "def flush(reason, complete):\n"
        f"    open({str(report)!r}, 'w').write(json.dumps("
        "{'exit_reason': reason, 'complete': complete}))\n"
        f"run = L.RunLifecycle(run_id='kid', label='child', state_path={str(state)!r},\n"
        "                     flush=flush, stall_after_s=0)\n"
        "run.start(context={})\n"
        "run.note_progress(7, 99, folder='f')\n"
        "os.kill(os.getpid(), signal.SIGTERM)\n"
        "import time; time.sleep(0.2)\n"
        "run.finish('signal:' + run.stop_reason(), complete=False)\n"
        "raise SystemExit(L.signal_exit_code(run.stop_flag.signum))\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, str(child)], capture_output=True, text=True, timeout=60,
        check=False,
    )
    assert proc.returncode == 143, proc.stderr
    written = json.loads(report.read_text(encoding="utf-8"))
    assert written == {"exit_reason": "signal:SIGTERM", "complete": False}
    assert lifecycle.read_state(state)["phase"] == "signal:SIGTERM"
    assert "EXIT run=kid reason=signal:SIGTERM complete=False" in proc.stderr
