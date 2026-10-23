"""Make a long ingest run's death observable.

WHY THIS EXISTS (production TIER-1 ingest on live #480, 567147a):

    [p1b-server] run_id=37159882e871 tier=1 folders=1 shard_index=0 ...
    [p1b-server] the client project 316/1361 [ok] ...
    <nothing, ever>

No traceback. No final tally line. No report JSON. Worker PID 829 was simply
gone and the log was stale. Two earlier runs stopped the same way at 315/1362
and 323/1380 on different images.

``scripts/p1b_ingest_drive_server.py`` has no path that can produce that.
Every early ``return`` in ``main()`` prints ``ERROR: ...`` first and all of
them are ABOVE the ``run_id=`` line; after that line the only ways out are the
end of the function (which prints the tally) and an exception (which the
interpreter prints as a traceback). A run that emits the run line and then
nothing did not exit — it was ENDED, from outside the interpreter or from
below it, and nothing in the process recorded which.

That is the actual defect this module fixes: the termination was
unobservable, so three runs produced three identical mysteries. The four ways
a Python process ends with no traceback, and what is recorded for each:

1. ``SIGTERM`` / ``SIGINT`` / ``SIGHUP`` — container recycle, a deploy of the
   web service the shell is attached to, or the Render Shell session closing
   on a run that was not a session leader. Handleable: ``install_signal_handlers``
   names the signal, the run drains what is in flight, and the caller's flush
   callback still writes a report marked ``complete: false``.
2. ``SIGKILL`` from the cgroup OOM killer. Not handleable — no handler runs, no
   atexit, no buffer flush. Recorded instead by a per-file sidecar heartbeat
   carrying RSS, ``memory.current``/``memory.max`` and the cgroup ``oom_kill``
   counter, plus ``previous_run_postmortem`` on the NEXT start. This is the
   leading suspect: ``app/core/extract_isolated`` measured ONE 4.4 MB PDF
   taking the live box from 790 MB to 3.9 GB of 4 GB before a SIGKILL, and the
   three deaths cluster at the same position in a size-sorted queue (files of
   4.7-4.9 MB), not at a random wall-clock moment.
3. A native crash — SIGSEGV/SIGABRT out of PyMuPDF, tesseract or torch — which
   prints nothing at all by default. ``enable_fault_handlers`` turns that into
   a C-level traceback for every thread.
4. The whole container being replaced. Distinguishable from 1-3 by host
   uptime: the sidecar records ``/proc/uptime`` on every heartbeat, so a
   post-mortem where uptime went BACKWARDS says the box is new, and one where
   it went forward says the process was killed on a box that kept running.

Nothing here fakes a completed run. ``phase``/``exit_reason`` carry the truth
and ``complete`` is false unless the work actually finished.
"""
from __future__ import annotations

import atexit
import faulthandler
import json
import logging
import os
import platform
import signal
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

_MB = 1024.0 * 1024.0

# A cgroup "no limit" is written either as the literal "max" (v2) or as a
# near-2**63 sentinel (v1). Anything above this is not a real ceiling.
_NO_LIMIT_BYTES = 1 << 50

logger = logging.getLogger(__name__)

LogFn = Callable[[str], None]


def default_log(message: str) -> None:
    """Unbuffered stderr. A diagnostic line that is still sitting in a buffer
    when SIGKILL arrives never existed."""
    print(message, file=sys.stderr, flush=True)


# ── /proc and cgroup readers ─────────────────────────────────────────────────


def _read_text(path: Path) -> Optional[str]:
    """File contents, or None when the path is absent or unreadable.

    Every reader here is best-effort by design: this module runs on Render
    (cgroup v2), on Fargate (no cgroup-v1 memory files), on a dev box, and
    under pytest with fake roots. A missing file is not an error — the
    snapshot logs once when no memory counter exists. A present file that
    cannot be read logs one line and does not raise into the ingest loop.
    """
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        logger.warning("could not read %s", path)
        return None


def _read_int(path: Path) -> Optional[int]:
    """One integer from a single-value cgroup file. ``max`` reads as None."""
    raw = _read_text(path)
    if raw is None:
        return None
    token = raw.strip()
    if not token or token == "max":
        return None
    try:
        value = int(token.split()[0])
    except ValueError:
        logger.warning("cgroup integer unparseable in %r", token, exc_info=True)
        return None
    if value >= _NO_LIMIT_BYTES:
        return None
    return value


def _read_kv_ints(path: Path) -> Dict[str, int]:
    """``key value`` lines (``memory.events``, ``memory.oom_control``)."""
    raw = _read_text(path)
    if raw is None:
        return {}
    out: Dict[str, int] = {}
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            out[parts[0]] = int(parts[1])
        except ValueError:
            continue
    return out


def _read_proc_kb(status_path: Path, key: str) -> Optional[float]:
    """A ``VmRSS:``/``VmHWM:`` style kB field from /proc/<pid>/status, in MB."""
    raw = _read_text(status_path)
    if raw is None:
        return None
    prefix = f"{key}:"
    for line in raw.splitlines():
        if not line.startswith(prefix):
            continue
        parts = line.split()
        if len(parts) < 2:
            return None
        try:
            return round(int(parts[1]) / 1024.0, 1)
        except ValueError:
            logger.warning(
                "proc status %s field unparseable: %r", key, parts[1], exc_info=True
            )
            return None
    return None


def host_uptime_s(proc_root: Path = Path("/proc")) -> Optional[float]:
    """Seconds since the box booted.

    The discriminator between "our process was killed" and "the whole
    container was replaced": uptime only ever goes up on one box, so a
    post-mortem that sees it go DOWN is looking at a different box.
    """
    raw = _read_text(proc_root / "uptime")
    if raw is None:
        return None
    try:
        return round(float(raw.split()[0]), 1)
    except (ValueError, IndexError):
        logger.warning("host uptime unparseable: %r", raw, exc_info=True)
        return None


@dataclass(frozen=True)
class MemorySnapshot:
    """The box at one instant, in the terms the OOM killer actually uses."""

    rss_mb: Optional[float] = None
    peak_rss_mb: Optional[float] = None
    cgroup_current_mb: Optional[float] = None
    cgroup_peak_mb: Optional[float] = None
    cgroup_limit_mb: Optional[float] = None
    oom_kill: Optional[int] = None
    oom_group_kill: Optional[int] = None
    uptime_s: Optional[float] = None
    #: Non-reclaimable (anonymous) memory of the whole container -- every
    #: process in it, copy-on-write pages counted once. What the OOM killer
    #: actually weighs.
    anon_mb: Optional[float] = None
    #: Page cache. Counted in cgroup usage, but the kernel reclaims it before
    #: it kills anything.
    file_mb: Optional[float] = None

    def headroom_mb(self) -> Optional[float]:
        """MB left before the cgroup ceiling, or None when either is unknown."""
        if self.cgroup_limit_mb is None or self.cgroup_current_mb is None:
            return None
        return round(self.cgroup_limit_mb - self.cgroup_current_mb, 1)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "rss_mb": self.rss_mb,
            "peak_rss_mb": self.peak_rss_mb,
            "cgroup_current_mb": self.cgroup_current_mb,
            "cgroup_peak_mb": self.cgroup_peak_mb,
            "cgroup_limit_mb": self.cgroup_limit_mb,
            "headroom_mb": self.headroom_mb(),
            "oom_kill": self.oom_kill,
            "oom_group_kill": self.oom_group_kill,
            "uptime_s": self.uptime_s,
            "anon_mb": self.anon_mb,
            "file_mb": self.file_mb,
        }

    def as_line(self) -> str:
        """One greppable line. Unknown fields are omitted, not guessed."""
        bits: List[str] = []
        if self.rss_mb is not None:
            bits.append(f"rss={self.rss_mb:.0f}MB")
        if self.peak_rss_mb is not None:
            bits.append(f"rss_peak={self.peak_rss_mb:.0f}MB")
        if self.cgroup_current_mb is not None:
            limit = (
                f"/{self.cgroup_limit_mb:.0f}MB"
                if self.cgroup_limit_mb is not None
                else "MB"
            )
            bits.append(f"cgroup={self.cgroup_current_mb:.0f}{limit}")
        headroom = self.headroom_mb()
        if headroom is not None:
            bits.append(f"headroom={headroom:.0f}MB")
        if self.cgroup_peak_mb is not None:
            bits.append(f"cgroup_peak={self.cgroup_peak_mb:.0f}MB")
        if self.anon_mb is not None:
            bits.append(f"anon={self.anon_mb:.0f}MB")
        if self.file_mb is not None:
            bits.append(f"page_cache={self.file_mb:.0f}MB")
        if self.oom_kill is not None:
            bits.append(f"oom_kill={self.oom_kill}")
        if self.oom_group_kill:
            bits.append(f"oom_group_kill={self.oom_group_kill}")
        if self.uptime_s is not None:
            bits.append(f"host_uptime={self.uptime_s:.0f}s")
        return " ".join(bits) if bits else "memory=unavailable"


def cgroup_candidate_dirs(
    *,
    proc_root: Path = Path("/proc"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> List[Path]:
    """Directories that may hold this process's memory counters, closest first.

    ``/sys/fs/cgroup/memory.max`` only works when the container gets its own
    cgroup NAMESPACE. Without one — this dev box, and any host where the
    runtime does not namespace — ``/sys/fs/cgroup`` is the host ROOT cgroup,
    which by design has no ``memory.max`` at all, so a reader that only looks
    there reports "memory unavailable" and the OOM evidence is lost exactly
    where it matters. ``/proc/self/cgroup`` gives the real path
    (``0::/system.slice/pod-…``), and the ancestors are included because a
    leaf cgroup can inherit its ceiling from one of them.
    """
    dirs: List[Path] = []
    raw = _read_text(proc_root / "self" / "cgroup") or ""
    for line in raw.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        controllers, rel = parts[1], parts[2].strip().lstrip("/")
        # v2 lines have an empty controller field; v1 memory lines name it.
        if controllers and "memory" not in controllers.split(","):
            continue
        base = cgroup_root if not controllers else cgroup_root / "memory"
        node = base / rel if rel else base
        while True:
            if node not in dirs:
                dirs.append(node)
            if node == base or base not in node.parents:
                break
            node = node.parent
    for fallback in (cgroup_root, cgroup_root / "memory"):
        if fallback not in dirs:
            dirs.append(fallback)
    return dirs


def _first_int(dirs: Sequence[Path], *names: str) -> Optional[int]:
    for directory in dirs:
        for name in names:
            value = _read_int(directory / name)
            if value is not None:
                return value
    return None


def _first_kv(dirs: Sequence[Path], *names: str) -> Dict[str, int]:
    for directory in dirs:
        for name in names:
            values = _read_kv_ints(directory / name)
            if values:
                return values
    return {}


def read_memory_snapshot(
    *,
    proc_root: Path = Path("/proc"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    pid: str = "self",
) -> MemorySnapshot:
    """RSS + cgroup usage/limit + the cgroup OOM-kill counter.

    Reads cgroup v2 first (Render, modern Docker) and falls back to v1 paths.
    The ``oom_kill`` counter is the whole point: a SIGKILLed process cannot log
    its own death, but the counter it left behind survives it, so the next
    start — or a supervisor — can say "the kernel did this" instead of
    "the process vanished".
    """
    rss = _read_proc_kb(proc_root / pid / "status", "VmRSS")
    peak = _read_proc_kb(proc_root / pid / "status", "VmHWM")

    dirs = cgroup_candidate_dirs(proc_root=proc_root, cgroup_root=cgroup_root)
    # v2 names first (memory.current / memory.max). v1 names are the fallback.
    current = _first_int(dirs, "memory.current", "memory.usage_in_bytes")
    limit = _first_int(dirs, "memory.max", "memory.limit_in_bytes")
    cg_peak = _first_int(dirs, "memory.peak", "memory.max_usage_in_bytes")
    events = _first_kv(dirs, "memory.events", "memory.oom_control")
    anon_b, file_b = _anon_and_cache_bytes(dirs, proc_root)
    if current is None and limit is None and cg_peak is None and not events:
        logger.warning("cgroup memory path isn't available at %s", cgroup_root)

    def _mb(value: Optional[int]) -> Optional[float]:
        return None if value is None else round(value / _MB, 1)

    return MemorySnapshot(
        rss_mb=rss,
        peak_rss_mb=peak,
        cgroup_current_mb=_mb(current),
        cgroup_peak_mb=_mb(cg_peak),
        cgroup_limit_mb=_mb(limit),
        oom_kill=events.get("oom_kill"),
        oom_group_kill=events.get("oom_group_kill"),
        uptime_s=host_uptime_s(proc_root),
        anon_mb=_mb(anon_b),
        file_mb=_mb(file_b),
    )


def _meminfo_bytes(proc_root: Path) -> Dict[str, int]:
    """``/proc/meminfo`` as bytes (``MemTotal``, ``MemAvailable``, ``Cached`` ...)."""
    raw = _read_text(proc_root / "meminfo") or ""
    out: Dict[str, int] = {}
    for line in raw.splitlines():
        name, _, rest = line.partition(":")
        parts = rest.split()
        if parts and parts[0].isdigit():
            out[name.strip()] = int(parts[0]) * 1024
    return out


def _anon_and_cache_bytes(
    dirs: Sequence[Path], proc_root: Path,
) -> tuple[Optional[int], Optional[int]]:
    """(anonymous, page cache) bytes: cgroup ``memory.stat`` first, else meminfo.

    A Fargate task is its own micro-VM, so where the cgroup files hide the
    numbers, ``/proc/meminfo`` describes exactly this task.
    """
    stat = _first_kv(dirs, "memory.stat")
    anon = stat.get("anon", stat.get("total_rss"))
    cache = stat.get("file", stat.get("total_cache"))
    if anon is not None:
        return anon, cache
    info = _meminfo_bytes(proc_root)
    total, avail = info.get("MemTotal"), info.get("MemAvailable")
    if total is None or avail is None:
        return None, None
    return total - avail, info.get("Cached")


def memory_numbers(
    *,
    proc_root: Path = Path("/proc"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> tuple[Optional[int], Optional[int]]:
    """(non-reclaimable bytes in use, limit bytes) for this container.

    Anonymous memory -- page cache excluded, because the kernel drops cache
    before it OOM-kills. The limit is the cgroup's when it is readable and
    finite, else the box's MemTotal (on Fargate the cgroup reports ``max``
    and the micro-VM is the task). ``(None, None)`` where neither is known.
    """
    dirs = cgroup_candidate_dirs(proc_root=proc_root, cgroup_root=cgroup_root)
    info = _meminfo_bytes(proc_root)
    # The tightest limit anyone states: a finite cgroup limit, the ECS task's
    # declared memory (Fargate hides the cgroup limit, and its micro-VM can
    # report more MemTotal than the task is allowed), the box's MemTotal.
    known = [
        v for v in (
            _first_int(dirs, "memory.max", "memory.limit_in_bytes"),
            ecs_task_memory_limit_bytes(),
            info.get("MemTotal"),
        ) if v
    ]
    limit = min(known) if known else None
    total = info.get("MemTotal")
    stat = _first_kv(dirs, "memory.stat")
    used = stat.get("anon", stat.get("total_rss"))
    if used is None and total is not None and info.get("MemAvailable") is not None:
        used = total - info["MemAvailable"]
    if not limit or used is None:
        return None, None
    return used, limit


_ECS_LIMIT_CACHE: Dict[str, Optional[int]] = {}


def ecs_task_memory_limit_bytes() -> Optional[int]:
    """The ECS task's declared memory limit, from the task metadata endpoint.

    ``$ECS_CONTAINER_METADATA_URI_V4/task`` -> ``Limits.Memory`` (MiB), the
    container's own ``Limits.Memory`` when the task has none. Read once per
    process; ``None`` off ECS or when the endpoint does not answer.
    """
    base = os.getenv("ECS_CONTAINER_METADATA_URI_V4", "").strip()
    if not base:
        return None
    if base in _ECS_LIMIT_CACHE:
        return _ECS_LIMIT_CACHE[base]
    import urllib.request

    found: Optional[int] = None
    for suffix in ("/task", ""):
        try:
            with urllib.request.urlopen(base + suffix, timeout=2) as resp:
                data = json.load(resp)
        except (OSError, ValueError):
            logger.debug("ECS metadata %s%s unreadable", base, suffix, exc_info=True)
            continue
        mib = (data.get("Limits") or {}).get("Memory")
        if isinstance(mib, (int, float)) and mib > 0:
            found = int(mib) * _MB
            break
    _ECS_LIMIT_CACHE[base] = found
    return found


def release_freed_memory() -> None:
    """Hand freed heap back to the OS between files.

    CPython frees a file's buffers, but glibc keeps the pages in its arenas,
    so resident memory stays at the high-water mark and the next file's
    admission sees headroom that is not really there. ``malloc_trim`` (glibc
    only) returns them; elsewhere this is just a collection.
    """
    import gc

    gc.collect()
    if not sys.platform.startswith("linux"):
        return
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        logger.debug("malloc_trim unavailable", exc_info=True)


def memory_pressure(
    *,
    proc_root: Path = Path("/proc"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> Optional[float]:
    """Fraction of this container's memory in NON-reclaimable use, or None."""
    used, limit = memory_numbers(proc_root=proc_root, cgroup_root=cgroup_root)
    if not limit or used is None:
        return None
    return used / limit


# ── process identity ────────────────────────────────────────────────────────


def process_identity() -> Dict[str, Any]:
    """pid/ppid/pgid/sid and whether a shell exit can HUP this run.

    A job started from Render Shell without ``setsid``/``nohup`` is not a
    session leader: it shares the shell's session, so closing the shell
    delivers SIGHUP to it. ``session_leader=False`` in the start banner is
    therefore a live hazard the operator can see BEFORE the run dies, and a
    piece of evidence afterwards.
    """
    ident: Dict[str, Any] = {
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "python": platform.python_version(),
        "executable": sys.executable,
        "cwd": os.getcwd(),
    }
    if hasattr(os, "getpgid") and hasattr(os, "getsid"):
        try:
            pgid = os.getpgid(0)
            sid = os.getsid(0)
            ident["pgid"] = pgid
            ident["sid"] = sid
            ident["session_leader"] = sid == ident["pid"]
            ident["process_group_leader"] = pgid == ident["pid"]
        except OSError as exc:
            ident["session_error"] = str(exc)
    try:
        # os.ttyname is POSIX-only; a run started from a Windows shell has no tty name.
        ident["tty"] = os.ttyname(0) if hasattr(os, "ttyname") and os.isatty(0) else None
    except OSError:
        ident["tty"] = None
    return ident


def identity_line(ident: Dict[str, Any]) -> str:
    """The start banner. Ordered so the two operational hazards read first."""
    return (
        f"pid={ident.get('pid')} ppid={ident.get('ppid')} "
        f"pgid={ident.get('pgid')} sid={ident.get('sid')} "
        f"session_leader={ident.get('session_leader')} "
        f"tty={ident.get('tty')} python={ident.get('python')}"
    )


# ── exit status vocabulary ──────────────────────────────────────────────────


def signal_name(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except ValueError:
        return f"SIG{signum}"


def signal_exit_code(signum: int) -> int:
    """The shell convention: a process killed by signal N exits 128+N."""
    return 128 + int(signum)


def describe_exit_status(returncode: int) -> str:
    """Say what a child's exit code MEANS, including the OOM reading.

    ``subprocess`` reports a signalled child as a negative return code, which
    is exactly the information the three mystery runs lacked: ``-9`` is not
    "the script finished", it is "the kernel or an operator killed it".
    """
    if returncode == 0:
        return "exited cleanly (0)"
    if returncode < 0:
        signum = -returncode
        name = signal_name(signum)
        note = ""
        if signum == getattr(signal, "SIGKILL", 9):
            note = (
                " — SIGKILL runs no handler and flushes no buffer; on a memory"
                " limit this is the cgroup OOM killer, so check the cgroup"
                " oom_kill counter below"
            )
        elif signum == getattr(signal, "SIGTERM", 15):
            note = " — container recycle, deploy, or an operator stop"
        elif signum == getattr(signal, "SIGHUP", 1):
            note = " — controlling terminal closed (run it under setsid/nohup)"
        elif signum in {
            getattr(signal, "SIGSEGV", 11),
            getattr(signal, "SIGABRT", 6),
            getattr(signal, "SIGBUS", 7),
            getattr(signal, "SIGILL", 4),
            getattr(signal, "SIGFPE", 8),
        }:
            note = " — native crash below Python (faulthandler output above)"
        return f"killed by {name} ({signum}){note}"
    return f"exited non-zero ({returncode})"


# ── cooperative stop ────────────────────────────────────────────────────────


@dataclass
class StopFlag:
    """Cooperative stop request. Canonical home for both ingest scripts.

    ``scripts/rag_render_bulk_ingest.py`` defined this (and the handler
    installer below) first and still imports both from here, so the two ingest
    entrypoints cannot drift into two different stop protocols.
    """

    stop: bool = False
    reason: str = ""
    signum: Optional[int] = None


def stop_signals() -> List[int]:
    """The signals a batch job should treat as "stop and report".

    SIGHUP is in the list on purpose. A detached Render Shell run that is not
    a session leader is HUPed when the shell closes, and the default SIGHUP
    action is to die instantly with no tally — indistinguishable from the OOM
    kill this module exists to identify.
    """
    return [
        s
        for s in (
            getattr(signal, "SIGINT", None),
            getattr(signal, "SIGTERM", None),
            getattr(signal, "SIGHUP", None),
        )
        if s is not None
    ]


def install_signal_handlers(
    stop_flag: StopFlag,
    *,
    signums: Optional[Sequence[int]] = None,
    on_signal: Optional[Callable[[int, str], None]] = None,
    log: LogFn = default_log,
) -> Callable[[], None]:
    """Turn a stop signal into a named, logged, cooperative stop.

    Defaults to ``stop_signals()``. Returns a callable that restores the
    previous handlers — tests must not leave a handler installed on the pytest
    process, and a run that finishes normally should not keep one either.
    """
    if signums is None:
        signums = stop_signals()

    previous: List[Any] = []

    def _handler(signum, _frame):
        name = signal_name(signum)
        stop_flag.stop = True
        # First signal wins: SIGTERM followed by SIGKILL must still read as
        # SIGTERM, and the reason is what the report and the sidecar carry.
        if not stop_flag.reason:
            stop_flag.reason = name
            stop_flag.signum = int(signum)
        log(f"[lifecycle] received {name} ({signum}) — stopping after in-flight work")
        if on_signal is not None:
            on_signal(signum, name)

    for sig in signums:
        try:
            previous.append((sig, signal.signal(sig, _handler)))
        except (ValueError, OSError) as exc:
            # Not the main thread, or the platform has no such signal. Say so:
            # an uninstallable handler means this run CANNOT report that class
            # of death, which the operator needs to know up front.
            log(f"[lifecycle] could not install handler for signal {sig}: {exc}")

    def _restore() -> None:
        for sig, handler in previous:
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError) as exc:
                log(f"[lifecycle] could not restore handler for signal {sig}: {exc}")

    return _restore


def enable_fault_handlers(
    *,
    log: LogFn = default_log,
    dump_signal: Optional[int] = None,
) -> Dict[str, bool]:
    """Native crashes and stalls become readable stderr instead of silence.

    ``faulthandler.enable`` covers case 3 in the module docstring: a SIGSEGV in
    PyMuPDF/tesseract/torch prints a C-level traceback for every thread rather
    than ending the process with no output at all.

    Registering a dump signal (SIGUSR1) covers the opposite failure: a run that
    is alive but wedged — the documented fork-from-threads hazard in
    ``app/core/extract_isolated`` — can be asked for all thread stacks with
    ``kill -USR1 <pid>`` without killing it.
    """
    state = {"fatal": False, "dump_signal": False}
    try:
        faulthandler.enable(all_threads=True)
        state["fatal"] = True
    except (RuntimeError, ValueError, OSError, AttributeError) as exc:
        log(f"[lifecycle] faulthandler.enable unavailable: {exc}")
    sig = dump_signal if dump_signal is not None else getattr(signal, "SIGUSR1", None)
    if sig is not None:
        try:
            faulthandler.register(sig, all_threads=True, chain=True)
            state["dump_signal"] = True
        except (RuntimeError, ValueError, OSError, AttributeError) as exc:
            log(f"[lifecycle] faulthandler.register({sig}) unavailable: {exc}")
    return state


# ── sidecar state + post-mortem ─────────────────────────────────────────────

PHASE_RUNNING = "running"
PHASE_COMPLETED = "completed"


def write_state_atomically(path: Path, payload: Dict[str, Any]) -> None:
    """Replace ``path`` with ``payload`` using raw fds and a rename.

    Raw ``os.write`` rather than buffered IO because this also runs inside a
    signal handler, where re-entering a file object whose lock the interrupted
    code already holds can deadlock the very flush that is trying to record
    the death.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.write(fd, blob.encode("utf-8"))
    finally:
        os.close(fd)
    os.replace(str(tmp), str(path))


def read_state(path: Path) -> Optional[Dict[str, Any]]:
    raw = _read_text(path)
    if raw is None:
        return None
    try:
        state = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("ingest run-state JSON is corrupt at %s", path, exc_info=True)
        return None
    return state if isinstance(state, dict) else None


def previous_run_postmortem(
    state_path: Path,
    *,
    now: Optional[MemorySnapshot] = None,
) -> Optional[str]:
    """What happened to the PREVIOUS run, read off its last heartbeat.

    This is the only mechanism that can diagnose a SIGKILL: the dead process
    logged nothing, but its sidecar records where it was, how much memory it
    was holding, what the cgroup ceiling was, and what the OOM-kill counter
    read at that moment. Comparing those to now gives the verdict:

    * host uptime went backwards → the container was REPLACED (case 4)
    * oom_kill counter went up   → the kernel OOM killer fired (case 2)
    * neither, phase=running     → killed on a box that kept running: an
      external SIGKILL/lifecycle stop that this build could not name

    Returns None when there is no previous run, or when it ended cleanly.
    """
    state = read_state(state_path)
    if state is None:
        return None
    phase = str(state.get("phase") or "")
    if phase == PHASE_COMPLETED:
        return None

    progress = state.get("progress") or {}
    memory = state.get("memory") or {}
    done, total = progress.get("done"), progress.get("total")
    where = f"{done}/{total}" if done is not None and total else "an unknown position"
    lines: List[str] = []
    if phase == PHASE_RUNNING:
        lines.append(
            f"previous run {state.get('run_id')} DIED WITHOUT A FINAL TALLY at "
            f"{where} (folder={progress.get('folder')!r}, "
            f"in_flight={progress.get('in_flight')}, "
            f"elapsed={state.get('elapsed_s')}s, pid={state.get('pid')}, "
            f"last heartbeat {state.get('updated_at')})"
        )
    else:
        lines.append(
            f"previous run {state.get('run_id')} ended with phase={phase!r} at "
            f"{where} (last heartbeat {state.get('updated_at')})"
        )

    if memory:
        lines.append(
            "  last memory: "
            + " ".join(
                f"{k}={v}"
                for k, v in memory.items()
                if v is not None and k != "uptime_s"
            )
        )

    now = read_memory_snapshot() if now is None else now
    prev_uptime = memory.get("uptime_s")
    if prev_uptime is not None and now.uptime_s is not None:
        if now.uptime_s < prev_uptime:
            lines.append(
                f"  VERDICT: host uptime went backwards "
                f"({prev_uptime}s -> {now.uptime_s}s) — this box is NEW, so the "
                f"run died with its container (deploy/restart/recycle), not on "
                f"its own"
            )
        else:
            lines.append(
                f"  host uptime {prev_uptime}s -> {now.uptime_s}s — same box, so "
                f"the process was killed while the container kept running"
            )
    prev_oom = memory.get("oom_kill")
    if prev_oom is not None and now.oom_kill is not None:
        if now.oom_kill > prev_oom:
            lines.append(
                f"  VERDICT: cgroup oom_kill went {prev_oom} -> {now.oom_kill} — "
                f"the kernel OOM killer fired since that heartbeat. This was an "
                f"OOM SIGKILL, not a clean exit."
            )
        else:
            lines.append(
                f"  cgroup oom_kill unchanged at {now.oom_kill} — no OOM kill "
                f"recorded in this cgroup"
            )
    return "\n".join(lines)


class RunLifecycle:
    """One guaranteed final flush per run, plus the evidence to explain it.

    Usage::

        lifecycle = RunLifecycle(run_id=..., label=..., state_path=...,
                                 flush=_flush_report)
        with lifecycle:
            lifecycle.start(context={...})
            ...
            lifecycle.note_progress(done, total, folder=name, in_flight=n)

    ``flush(reason, complete)`` is the caller's single report writer. It runs
    exactly once, on every exit path this process can observe: normal return,
    exception, ``SystemExit``, a stop signal, and ``atexit`` for anything that
    bypasses the context manager. It does NOT run on SIGKILL — nothing does —
    which is what the sidecar heartbeat is for.
    """

    def __init__(
        self,
        *,
        run_id: str,
        label: str,
        state_path: Path,
        flush: Optional[Callable[[str, bool], None]] = None,
        log: LogFn = default_log,
        stall_after_s: float = 900.0,
        heartbeat_every: int = 10,
        install_handlers: bool = True,
        register_atexit: bool = True,
    ) -> None:
        self.run_id = run_id
        self.label = label
        self.state_path = Path(state_path)
        self.flush = flush
        self.log = log
        self.stall_after_s = float(stall_after_s)
        self.heartbeat_every = max(1, int(heartbeat_every))
        self._install_handlers = install_handlers
        self._register_atexit = register_atexit

        self.stop_flag = StopFlag()
        # Every document >= 1 MB is extracted in a FORKED child
        # (app/core/extract_isolated), and a fork inherits this object, the
        # signal handlers and the sidecar path. A child that ran the flush or
        # a state write would overwrite the parent's evidence with a copy of
        # it — with the child's pid and a snapshot of the parent's progress
        # frozen at fork time. Only the process that armed the lifecycle may
        # write for it.
        self._owner_pid = os.getpid()
        self.started_monotonic = time.monotonic()
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.context: Dict[str, Any] = {}
        self.identity: Dict[str, Any] = {}
        self.finished_reason: Optional[str] = None

        self._restore_handlers: Optional[Callable[[], None]] = None
        self._previous_excepthook: Optional[Callable[..., None]] = None
        self._atexit_registered = False
        self._progress: Dict[str, Any] = {
            "folder": None,
            "done": 0,
            "total": 0,
            "in_flight": 0,
        }
        self._last_progress_monotonic = time.monotonic()
        self._stall_reported = False
        self._watchdog_stop = threading.Event()
        self._watchdog: Optional[threading.Thread] = None

    # -- start ---------------------------------------------------------------

    def start(self, *, context: Optional[Dict[str, Any]] = None) -> None:
        """Post-mortem the last run, then arm everything for this one."""
        self.context = dict(context or {})
        postmortem = previous_run_postmortem(self.state_path)
        if postmortem:
            self.log(f"[lifecycle] POSTMORTEM {postmortem}")
        else:
            self.log(
                "[lifecycle] no unfinished previous run recorded at "
                f"{self.state_path}"
            )

        self.identity = process_identity()
        self.log(f"[lifecycle] {self.label} {identity_line(self.identity)}")
        if self.identity.get("session_leader") is False:
            self.log(
                "[lifecycle] WARNING this run is not a session leader — closing "
                "the shell that started it will SIGHUP it; use setsid/nohup for "
                "detached runs"
            )
        self.log(f"[lifecycle] memory at start: {self.snapshot().as_line()}")
        self.log(f"[lifecycle] fault handlers: {enable_fault_handlers(log=self.log)}")

        if self._install_handlers:
            self._restore_handlers = install_signal_handlers(
                self.stop_flag, on_signal=self._on_signal, log=self.log,
            )
            self.install_excepthook()
        if self._register_atexit:
            atexit.register(self._atexit_flush)
            self._atexit_registered = True

        self._write_state(PHASE_RUNNING)
        self._start_watchdog()

    def install_excepthook(self) -> None:
        """Flush the report BEFORE the interpreter prints the traceback.

        An uncaught exception would reach ``atexit`` anyway, but only as a
        generic "atexit" with the exception type already lost. Chaining the
        hook keeps the exit reason specific (``exception:ClientError``) and
        writes the report while the process is still healthy, instead of
        during interpreter teardown.
        """
        previous = sys.excepthook

        def _hook(exc_type, exc, tb) -> None:
            self.log(
                f"[lifecycle] FATAL {getattr(exc_type, '__name__', exc_type)}: "
                f"{exc} — flushing the report before the traceback"
            )
            self.finish(
                f"exception:{getattr(exc_type, '__name__', exc_type)}",
                complete=False,
            )
            previous(exc_type, exc, tb)

        self._previous_excepthook = previous
        sys.excepthook = _hook

    def snapshot(self) -> MemorySnapshot:
        return read_memory_snapshot()

    def owns_process(self) -> bool:
        """False in a forked child, which must not write for its parent."""
        return os.getpid() == self._owner_pid

    # -- progress ------------------------------------------------------------

    def note_progress(
        self,
        done: int,
        total: int,
        *,
        folder: Optional[str] = None,
        in_flight: int = 0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record one completed unit of work and heartbeat the sidecar.

        The sidecar is written on EVERY file, not on the heartbeat cadence: the
        whole value of the post-mortem is naming the exact file count the run
        reached, and a SIGKILL does not wait for a round number.
        """
        self._progress = {
            "folder": folder,
            "done": int(done),
            "total": int(total),
            "in_flight": int(in_flight),
        }
        if extra:
            self._progress.update(extra)
        self._last_progress_monotonic = time.monotonic()
        self._stall_reported = False
        memory = self._write_state(PHASE_RUNNING)
        if done % self.heartbeat_every == 0:
            elapsed = self.elapsed_s()
            rate = elapsed / done if done else 0.0
            self.log(
                f"[lifecycle] HEARTBEAT run={self.run_id} "
                f"{done}/{total} in_flight={in_flight} "
                f"elapsed={elapsed:.0f}s {rate:.1f}s/file {memory.as_line()}"
            )

    def elapsed_s(self) -> float:
        return round(time.monotonic() - self.started_monotonic, 1)

    def should_stop(self) -> bool:
        return self.stop_flag.stop

    def stop_reason(self) -> str:
        return self.stop_flag.reason

    # -- finish --------------------------------------------------------------

    def finish(self, reason: str, *, complete: Optional[bool] = None) -> None:
        """Flush the report and record the phase. Idempotent.

        Idempotency is what makes the guarantee safe to state: the context
        manager, the atexit hook and an explicit call can all fire for one run
        and the report is still written once, with the FIRST (most specific)
        reason rather than the last.
        """
        if self.finished_reason is not None or not self.owns_process():
            return
        self.finished_reason = reason
        if complete is None:
            complete = reason == PHASE_COMPLETED
        self._watchdog_stop.set()
        try:
            if self.flush is not None:
                self.flush(reason, bool(complete))
        except Exception as exc:  # noqa: BLE001 — a failed flush must still be visible
            self.log(
                f"[lifecycle] FINAL FLUSH FAILED ({type(exc).__name__}: {exc}) — "
                f"the run's own tally could not be written"
            )
            self.log(traceback.format_exc())
        self._write_state(reason, complete=bool(complete))
        self.log(
            f"[lifecycle] EXIT run={self.run_id} reason={reason} "
            f"complete={bool(complete)} progress="
            f"{self._progress.get('done')}/{self._progress.get('total')} "
            f"elapsed={self.elapsed_s():.0f}s {self.snapshot().as_line()}"
        )
        if self._restore_handlers is not None:
            self._restore_handlers()
            self._restore_handlers = None
        if self._previous_excepthook is not None:
            sys.excepthook = self._previous_excepthook
            self._previous_excepthook = None
        if self._atexit_registered:
            atexit.unregister(self._atexit_flush)
            self._atexit_registered = False
        sig = getattr(signal, "SIGUSR1", None)
        if sig is not None:
            try:
                faulthandler.unregister(sig)
            except (RuntimeError, ValueError, OSError) as exc:
                self.log(f"[lifecycle] faulthandler.unregister failed: {exc}")

    def __enter__(self) -> "RunLifecycle":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            reason = (
                f"signal:{self.stop_flag.reason}"
                if self.stop_flag.stop
                else PHASE_COMPLETED
            )
            self.finish(reason, complete=not self.stop_flag.stop)
            return False
        if issubclass(exc_type, SystemExit):
            self.finish(f"systemexit:{getattr(exc, 'code', None)}", complete=False)
            return False
        # An uncaught exception here would still print a traceback, but only
        # AFTER the interpreter unwinds; flush first so the report exists even
        # if the unwind itself is what gets killed.
        self.log(
            f"[lifecycle] FATAL {exc_type.__name__}: {exc} — flushing the report "
            f"before the traceback"
        )
        self.finish(f"exception:{exc_type.__name__}", complete=False)
        return False

    # -- internals -----------------------------------------------------------

    def _on_signal(self, signum: int, name: str) -> None:
        """Signal-handler-safe record of the stop.

        Runs in the main thread on top of whatever it was doing, so it takes no
        lock and touches no buffered file object — just a raw-fd sidecar write.
        Render sends SIGTERM and then SIGKILL ~30s later; if the graceful drain
        does not finish in that window, THIS is the record that survives.
        """
        self._write_state(f"signal:{name}", complete=False, extra={"signum": signum})

    def _atexit_flush(self) -> None:
        if self.finished_reason is None and self.owns_process():
            self.log(
                "[lifecycle] atexit reached with no recorded exit — something "
                "left the run without going through the lifecycle"
            )
            self.finish("atexit", complete=False)

    def _write_state(
        self,
        phase: str,
        *,
        complete: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> MemorySnapshot:
        memory = self.snapshot()
        if not self.owns_process():
            return memory
        payload: Dict[str, Any] = {
            "run_id": self.run_id,
            "label": self.label,
            "phase": phase,
            "complete": complete,
            "pid": os.getpid(),
            "started_at": self.started_at,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": self.elapsed_s(),
            "progress": dict(self._progress),
            "memory": memory.as_dict(),
            "identity": self.identity,
            "context": self.context,
        }
        if extra:
            payload.update(extra)
        try:
            write_state_atomically(self.state_path, payload)
        except OSError as exc:
            self.log(f"[lifecycle] could not write run state {self.state_path}: {exc}")
        return memory

    def _start_watchdog(self) -> None:
        if self.stall_after_s <= 0:
            return
        self._watchdog = threading.Thread(
            target=self._watchdog_loop, name="ingest-stall-watchdog", daemon=True,
        )
        self._watchdog.start()

    def _watchdog_loop(self) -> None:
        """Report a run that is alive but has stopped making progress.

        The forked extraction child can wedge when the fork happens while
        another pool thread holds a lock (documented in
        ``app/core/extract_isolated``); the parent then sits in
        ``concurrent.futures.wait`` for the full extraction timeout per file
        and the log looks exactly as stale as a dead process. Dumping every
        thread's stack tells the two apart.
        """
        interval = max(1.0, min(30.0, self.stall_after_s / 3.0))
        while not self._watchdog_stop.wait(interval):
            idle = time.monotonic() - self._last_progress_monotonic
            if idle < self.stall_after_s or self._stall_reported:
                continue
            self._stall_reported = True
            self.log(
                f"[lifecycle] STALL run={self.run_id} no completed file for "
                f"{idle:.0f}s at {self._progress.get('done')}/"
                f"{self._progress.get('total')} "
                f"in_flight={self._progress.get('in_flight')} "
                f"{self.snapshot().as_line()} — thread dump follows"
            )
            try:
                faulthandler.dump_traceback(all_threads=True)
            except (RuntimeError, ValueError, OSError) as exc:
                self.log(f"[lifecycle] thread dump unavailable: {exc}")


# ── supervisor ──────────────────────────────────────────────────────────────


def forward_signal_to_child(proc: Any, signum: int, _frame: Any = None) -> None:
    """Relay a stop signal to the supervised child.

    Without this the supervisor would absorb the operator's SIGTERM and the
    child would keep ingesting, which is the opposite of what was asked.
    """
    try:
        proc.send_signal(signum)
    except (OSError, ValueError, ProcessLookupError) as exc:
        default_log(f"[supervisor] could not forward signal {signum}: {exc}")


def run_supervised(
    argv: Sequence[str],
    *,
    log: LogFn = default_log,
    env: Optional[Dict[str, str]] = None,
    state_path: Optional[Path] = None,
    forward: Iterable[int] = (),
) -> int:
    """Run ``argv`` as a child and REPORT HOW IT DIED.

    A SIGKILLed process cannot describe its own death. Its parent can: the
    wait status carries the signal number, and by the time the child is reaped
    the cgroup ``oom_kill`` counter has already been incremented. That turns
    "worker PID 829 disappeared" into "killed by SIGKILL (9) ... oom_kill went
    0 -> 1", which is the whole difference between a mystery and a diagnosis.

    Returns the child's exit code, or 128+signal when it was signalled, so a
    supervised run still fails loudly in a shell or CI.
    """
    before = read_memory_snapshot()
    log(f"[supervisor] starting {' '.join(argv)}")
    log(f"[supervisor] memory before: {before.as_line()}")
    # Inherits stdio on purpose: the child's own log must keep going to the
    # same place the operator is already tailing.
    proc = subprocess.Popen(list(argv), env=env)  # noqa: S603 — operator-supplied argv
    restore: List[Any] = []
    for signum in forward:
        try:
            restore.append(
                (
                    signum,
                    signal.signal(
                        signum,
                        lambda s, f, _p=proc: forward_signal_to_child(_p, s, f),
                    ),
                )
            )
        except (ValueError, OSError) as exc:
            log(f"[supervisor] could not hook signal {signum}: {exc}")
    try:
        returncode = proc.wait()
    finally:
        for signum, handler in restore:
            try:
                signal.signal(signum, handler)
            except (ValueError, OSError) as exc:
                log(f"[supervisor] could not restore signal {signum}: {exc}")

    after = read_memory_snapshot()
    log(f"[supervisor] child pid={proc.pid} {describe_exit_status(returncode)}")
    log(f"[supervisor] memory after: {after.as_line()}")
    if (
        before.oom_kill is not None
        and after.oom_kill is not None
        and after.oom_kill > before.oom_kill
    ):
        log(
            f"[supervisor] VERDICT cgroup oom_kill {before.oom_kill} -> "
            f"{after.oom_kill}: the kernel OOM killer fired during this run"
        )
    if state_path is not None:
        postmortem = previous_run_postmortem(Path(state_path), now=after)
        if postmortem:
            log(f"[supervisor] POSTMORTEM {postmortem}")
    if returncode < 0:
        return signal_exit_code(-returncode)
    return returncode
