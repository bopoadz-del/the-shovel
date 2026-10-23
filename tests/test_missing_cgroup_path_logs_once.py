"""A missing cgroup memory path is one log line and no traceback.

Fargate has no cgroup-v1 memory files. Reading them must not dump a
stack into the ingest log, and the snapshot must still return.
"""
from __future__ import annotations

from pathlib import Path

from app.core import ingest_lifecycle as lifecycle


def _proc(tmp: Path) -> Path:
    proc = tmp / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "self" / "status").write_text(
        "Name:\tpython3\nVmRSS:\t 2048 kB\nVmHWM:\t 4096 kB\n",
        encoding="utf-8",
    )
    (proc / "uptime").write_text("100.0 1.0\n", encoding="utf-8")
    (proc / "self" / "cgroup").write_text("0::/\n", encoding="utf-8")
    return proc


def _spy(monkeypatch):
    """Capture warning() calls. caplog is empty once another test replaces root handlers."""
    calls: list[dict] = []

    def _capture(msg, *args, **kwargs):
        try:
            text = msg % args if args else str(msg)
        except (TypeError, ValueError):
            text = str(msg)
        calls.append({"text": text, "exc_info": kwargs.get("exc_info")})

    monkeypatch.setattr(lifecycle.logger, "warning", _capture)
    return calls


def test_a_missing_cgroup_path_logs_once_without_a_traceback(tmp_path, monkeypatch):
    """Several missing-path shapes. v2 and v1 files still read when present."""
    problems: list[str] = []
    calls = _spy(monkeypatch)

    def _check(label: str, root: Path, *, expect_warning: bool, expect_current):
        calls.clear()
        try:
            snap = lifecycle.read_memory_snapshot(
                proc_root=_proc(root), cgroup_root=root / "cgroup",
            )
        except Exception as exc:  # noqa: BLE001 — the assertion is "no raise"
            problems.append(f"{label}: raised {type(exc).__name__}: {exc}")
            return
        notes = list(calls)
        if any("Traceback" in note["text"] for note in notes):
            problems.append(f"{label}: traceback in log ({len(notes)} records)")
        if expect_warning:
            if len(notes) != 1:
                problems.append(
                    f"{label}: expected 1 log record, got {len(notes)}: "
                    + " | ".join(note["text"] for note in notes)
                )
            elif notes[0]["exc_info"] not in (None, False):
                problems.append(f"{label}: log record carries exc_info")
            elif "isn't available" not in notes[0]["text"]:
                problems.append(
                    f"{label}: message does not say the path isn't available: "
                    f"{notes[0]['text']!r}"
                )
        else:
            unavailable = [note for note in notes if "isn't available" in note["text"]]
            if unavailable:
                problems.append(f"{label}: warned that a present path isn't available")
            if any(note["exc_info"] not in (None, False) for note in notes):
                problems.append(f"{label}: traceback on a present path")
        if snap.cgroup_current_mb != expect_current and not (
            expect_current is not None
            and snap.cgroup_current_mb is not None
            and abs(snap.cgroup_current_mb - expect_current) < 0.2
        ):
            problems.append(
                f"{label}: cgroup_current_mb={snap.cgroup_current_mb!r} "
                f"expected {expect_current!r}"
            )
        if snap.rss_mb is None:
            problems.append(f"{label}: rss was not read")

    empty = tmp_path / "empty"
    (empty / "cgroup").mkdir(parents=True)
    _check("empty-cgroup-dir", empty, expect_warning=True, expect_current=None)

    missing = tmp_path / "missing"
    missing.mkdir()
    _check("cgroup-root-absent", missing, expect_warning=True, expect_current=None)

    v2 = tmp_path / "v2"
    cg = v2 / "cgroup"
    cg.mkdir(parents=True)
    (cg / "memory.current").write_text("1048576", encoding="utf-8")
    (cg / "memory.max").write_text("2097152", encoding="utf-8")
    _check("cgroup-v2-files", v2, expect_warning=False, expect_current=1.0)

    v1 = tmp_path / "v1"
    mem = v1 / "cgroup" / "memory"
    mem.mkdir(parents=True)
    (mem / "memory.usage_in_bytes").write_text("2097152", encoding="utf-8")
    (mem / "memory.limit_in_bytes").write_text("4194304", encoding="utf-8")
    _check("cgroup-v1-files", v1, expect_warning=False, expect_current=2.0)

    unlimited = tmp_path / "unlimited"
    ucg = unlimited / "cgroup"
    ucg.mkdir(parents=True)
    (ucg / "memory.current").write_text("1048576", encoding="utf-8")
    (ucg / "memory.max").write_text("max", encoding="utf-8")
    calls.clear()
    snap = lifecycle.read_memory_snapshot(
        proc_root=_proc(unlimited), cgroup_root=ucg,
    )
    if snap.cgroup_limit_mb is not None:
        problems.append(f"limit-max: limit parsed as {snap.cgroup_limit_mb}")
    if any("Traceback" in note["text"] for note in calls):
        problems.append("limit-max: traceback")

    assert not problems, "\n".join(problems)
