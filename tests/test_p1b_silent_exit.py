"""TIER-1 ingest must not be able to exit without a final tally.

The incident these tests exist for: production run 37159882e871 (live #480,
567147a) printed

    [p1b-server] run_id=37159882e871 tier=1 folders=1 ...
    [p1b-server] the client project 316/1361 [ok] ...

and then nothing at all — no traceback, no ``tier 1: N succeeded`` line, no
report JSON, PID 829 gone, log stale. Earlier runs stopped the same way at
315/1362 and 323/1380.

Reading ``main()`` shows there is no in-code path that does that: every early
``return`` prints ``ERROR: ...`` and sits ABOVE the run line, and an exception
prints a traceback. So the run was ended from outside the interpreter, and the
defect was that nothing recorded it. These tests hold the fix in place: a stop
signal, a fatal exception and a clean finish all produce exactly one report
with an honest ``exit_reason``/``complete`` pair, and an unfinished run is
never labelled complete.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

from app.core import ingest_lifecycle as lifecycle

REPO = Path(__file__).resolve().parent.parent
POSIX_ONLY = pytest.mark.skipif(os.name != "posix", reason="POSIX signals only")



@pytest.fixture(autouse=True)
def _names_follow_the_document_list(monkeypatch):
    """The name matchers read ``list_document_names``; these tests stub
    ``list_documents``. Route the former through whatever the latter is."""
    from app.core import projects as _p
    monkeypatch.setattr(_p, "list_document_names", lambda pid: _p.list_documents(pid))

@pytest.fixture(autouse=True)
def _restore_process_hooks():
    """``main()`` arms real signal handlers, an excepthook and an atexit hook.
    None of them may outlive a test on the pytest process."""
    saved = {s: signal.getsignal(s) for s in lifecycle.stop_signals()}
    saved_hook = sys.excepthook
    yield
    for sig, handler in saved.items():
        if handler is not None:
            signal.signal(sig, handler)
    sys.excepthook = saved_hook


def _drive_files(count: int, *, prefix: str = "Tier1") -> List[Dict[str, Any]]:
    return [
        {
            "id": f"drive-{i}",
            "name": f"doc-{i}.pdf",
            "_drive_path": f"{prefix}/doc-{i}.pdf",
            "mimeType": "application/pdf",
            "size": 1024 + i,
        }
        for i in range(count)
    ]


def _manifest(tmp_path: Path, folders: List[Dict[str, Any]]) -> Path:
    path = tmp_path / "priority.json"
    path.write_text(
        json.dumps({"tiers": {"1": {"folders": folders}}}), encoding="utf-8",
    )
    return path


class _Store:
    """Vector store stub: resume sees no existing chunks."""

    def count_by_doc(self, _project_id):
        return {}


def _install_fakes(monkeypatch, *, files, walk=None, list_documents=None):
    from app.core import gdrive_service, projects
    from app.core.rag import embeddings, vector_store

    monkeypatch.setattr(gdrive_service, "is_configured", lambda: True)
    monkeypatch.setattr(
        gdrive_service,
        "walk_folder",
        walk if walk is not None else (lambda *_a, **_k: (list(files), [])),
    )
    monkeypatch.setattr(
        projects,
        "get_or_create_project",
        lambda name, **_k: ({"id": "proj-test", "name": name}, False),
    )
    monkeypatch.setattr(
        projects,
        "list_documents",
        list_documents if list_documents is not None else (lambda _pid: []),
    )
    monkeypatch.setattr(embeddings, "reset_embedder_cache", lambda: None)
    monkeypatch.setattr(vector_store, "reset_store_cache", lambda: None)
    monkeypatch.setattr(vector_store, "get_store", lambda: _Store())


def _run_main(monkeypatch, tmp_path, *, argv_extra=(), files=None, parallelism="1",
              folders=None, holder=None, **fake_kwargs):
    """Run ``main()`` in-process with a recording lifecycle.

    ``install_handlers``/``register_atexit`` are off here so the flush
    guarantee is what is under test, not signal plumbing; the real handlers are
    exercised end-to-end in the subprocess tests below. ``holder`` receives the
    live ``RunLifecycle`` under ``"run"`` before the folder loop starts, which
    is how a fake ingest can request a stop mid-run deterministically.
    """
    import scripts.p1b_ingest_drive_server as p1b

    files = _drive_files(8) if files is None else files
    _install_fakes(monkeypatch, files=files, **fake_kwargs)

    created: List[Any] = []
    flush_calls: List[Any] = []
    real_cls = lifecycle.RunLifecycle

    class _Recording(real_cls):
        def __init__(self, **kwargs):
            kwargs["install_handlers"] = False
            kwargs["register_atexit"] = False
            kwargs["stall_after_s"] = 0
            inner = kwargs["flush"]

            def _spy(reason, complete):
                flush_calls.append((reason, complete))
                inner(reason, complete)

            kwargs["flush"] = _spy
            super().__init__(**kwargs)
            created.append(self)
            if holder is not None:
                holder["run"] = self

    monkeypatch.setattr(lifecycle, "RunLifecycle", _Recording)
    monkeypatch.setenv("P1B_PARALLELISM", str(parallelism))
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_VECTOR_NAMESPACE", "v2")

    report = tmp_path / "report.json"
    state = tmp_path / "state.json"
    manifest = _manifest(
        tmp_path,
        folders or [{"project_id": "proj-test",
                     "folder_name": "the client project",
                     "folder_id": "folder-1"}],
    )
    monkeypatch.setattr(sys, "argv", [
        "p1b_ingest_drive_server.py",
        "--tier", "1", "--resume",
        "--priority-manifest", str(manifest),
        "--output", str(report),
        "--run-state", str(state),
        *argv_extra,
    ])
    monkeypatch.chdir(tmp_path)
    code = p1b.main()
    return {
        "code": code,
        "report": json.loads(report.read_text(encoding="utf-8")) if report.exists() else None,
        "state": lifecycle.read_state(state),
        "flushes": flush_calls,
        "run": created[0] if created else None,
    }


# ── the guarantee ───────────────────────────────────────────────────────────


def test_finished_run_writes_one_report_marked_complete(monkeypatch, tmp_path, capsys):
    import scripts.p1b_ingest_drive_server as p1b

    monkeypatch.setattr(
        p1b, "_ingest_file",
        lambda fm, *a, **k: (fm["_drive_path"], {"status": "ok", "rag_indexed": 3}),
    )
    out = _run_main(monkeypatch, tmp_path)

    assert out["code"] == 0
    assert out["flushes"] == [("completed", True)]
    assert out["report"]["exit_reason"] == "completed"
    assert out["report"]["complete"] is True
    assert out["report"]["global_tally"]["succeeded"] == 8
    assert out["state"]["phase"] == "completed"
    err = capsys.readouterr().err
    assert err.count("[p1b-server] tier 1:") == 1
    assert "exit_reason=completed complete=True" in err
    assert "RUN INCOMPLETE" not in err


def test_stop_request_mid_run_writes_an_incomplete_final_tally(
    monkeypatch, tmp_path, capsys,
):
    """The 316/1361 shape: the run stops early and SAYS so, with the real
    counts. Nothing here promotes it to a finished tier."""
    import scripts.p1b_ingest_drive_server as p1b

    seen: List[str] = []
    holder: Dict[str, Any] = {}

    def _fake_ingest(fm, *_a, **_k):
        seen.append(fm["_drive_path"])
        if len(seen) == 3:
            # Stands in for the SIGTERM the container sends on a recycle; the
            # end-to-end signal path is covered by the subprocess test below.
            holder["run"].stop_flag.stop = True
            holder["run"].stop_flag.reason = "SIGTERM"
            holder["run"].stop_flag.signum = int(signal.SIGTERM)
        return fm["_drive_path"], {"status": "ok", "rag_indexed": 2}

    monkeypatch.setattr(p1b, "_ingest_file", _fake_ingest)

    out = _run_main(monkeypatch, tmp_path, holder=holder)

    assert out["code"] == 143, "a stopped run must not exit 0"
    assert out["flushes"] == [("signal:SIGTERM", False)]
    assert out["report"]["exit_reason"] == "signal:SIGTERM"
    assert out["report"]["complete"] is False
    assert out["report"]["global_tally"]["succeeded"] == 3
    assert len(seen) == 3, "no new file may be submitted after a stop request"
    assert out["state"]["phase"] == "signal:SIGTERM"
    assert out["state"]["progress"]["done"] == 3
    err = capsys.readouterr().err
    assert err.count("[p1b-server] tier 1:") == 1
    assert "RUN INCOMPLETE exit_reason=signal:SIGTERM" in err
    assert "stop requested (SIGTERM)" in err


def test_stop_drains_work_already_in_flight(monkeypatch, tmp_path):
    """The tally must cover every file actually attempted, so the in-flight
    files are drained (bounded by the pool size) rather than dropped."""
    import scripts.p1b_ingest_drive_server as p1b

    seen: List[str] = []
    holder: Dict[str, Any] = {}

    def _fake_ingest(fm, *_a, **_k):
        seen.append(fm["_drive_path"])
        if len(seen) == 1:
            holder["run"].stop_flag.stop = True
            holder["run"].stop_flag.reason = "SIGTERM"
        time.sleep(0.05)
        return fm["_drive_path"], {"status": "ok", "rag_indexed": 1}

    monkeypatch.setattr(p1b, "_ingest_file", _fake_ingest)

    out = _run_main(
        monkeypatch, tmp_path, files=_drive_files(9), parallelism="3",
        holder=holder,
    )

    # Three were in flight when the stop landed; all three are tallied and
    # none of the remaining six were started.
    assert len(seen) == 3
    assert out["report"]["global_tally"]["succeeded"] == 3
    assert sum(
        1 for r in out["report"]["results"] if r["result"]["status"] == "ok"
    ) == 3
    assert out["report"]["complete"] is False


def test_second_folder_is_not_started_after_a_stop(monkeypatch, tmp_path, capsys):
    import scripts.p1b_ingest_drive_server as p1b

    walked: List[str] = []
    holder: Dict[str, Any] = {}

    def _walk(folder_id, **_k):
        walked.append(folder_id)
        return _drive_files(2, prefix=folder_id), []

    def _fake_ingest(fm, *_a, **_k):
        holder["run"].stop_flag.stop = True
        holder["run"].stop_flag.reason = "SIGHUP"
        holder["run"].stop_flag.signum = int(signal.SIGHUP)
        return fm["_drive_path"], {"status": "ok", "rag_indexed": 1}

    monkeypatch.setattr(p1b, "_ingest_file", _fake_ingest)

    out = _run_main(
        monkeypatch,
        tmp_path,
        holder=holder,
        walk=_walk,
        folders=[
            {"project_id": "p1", "folder_name": "folder-one", "folder_id": "f1"},
            {"project_id": "p2", "folder_name": "folder-two", "folder_id": "f2"},
        ],
    )

    assert out["code"] == lifecycle.signal_exit_code(signal.SIGHUP)
    assert walked == ["f1"], "the second folder must not be walked after a stop"
    assert out["report"]["exit_reason"] == "signal:SIGHUP"
    assert out["report"]["complete"] is False
    assert "not starting further folders" in capsys.readouterr().err


def test_dry_run_keeps_its_own_state_file(monkeypatch, tmp_path):
    """A plan-only pass must not overwrite the evidence a dead live run left."""
    import scripts.p1b_ingest_drive_server as p1b

    args = p1b.build_parser().parse_args(
        ["--run-state", str(tmp_path / "s.json"), "--dry-run"],
    )
    live = p1b.run_state_path(args, dry_run=False)
    dry = p1b.run_state_path(args, dry_run=True)
    assert live != dry
    assert dry.name == "s_dry.json"


# ── end to end, real signals, real process ──────────────────────────────────


def _driver(tmp_path: Path, *, body: str, files: int = 40, folders=None,
            per_file_sleep: float = 0.25) -> Path:
    """A real subprocess that runs the real ``main()`` over faked Drive/DB."""
    folders = folders or [
        {"project_id": "proj-test", "folder_name": "the client project",
         "folder_id": "folder-1"},
    ]
    manifest = _manifest(tmp_path, folders)
    driver = tmp_path / "driver.py"
    driver.write_text(
        "import json, os, sys, time\n"
        f"sys.path.insert(0, {str(REPO)!r})\n"
        "from app.core import gdrive_service, projects\n"
        "from app.core.rag import embeddings, vector_store\n"
        "import scripts.p1b_ingest_drive_server as p1b\n"
        f"PROGRESS = {str(tmp_path / 'progress')!r}\n"
        "os.makedirs(PROGRESS, exist_ok=True)\n"
        "FILES = [{'id': 'drive-%d' % i, 'name': 'doc-%d.pdf' % i,\n"
        "          '_drive_path': 'Tier1/doc-%d.pdf' % i,\n"
        "          'mimeType': 'application/pdf', 'size': 1024 + i}\n"
        f"         for i in range({files})]\n"
        "class Store:\n"
        "    def count_by_doc(self, pid):\n"
        "        return {}\n"
        "gdrive_service.is_configured = lambda: True\n"
        "gdrive_service.walk_folder = lambda fid, **k: (list(FILES), [])\n"
        "projects.get_or_create_project = lambda name, **k: ({'id': 'proj', 'name': name}, False)\n"
        "projects.list_documents = lambda pid: []\n"
        "embeddings.reset_embedder_cache = lambda: None\n"
        "vector_store.reset_store_cache = lambda: None\n"
        "vector_store.get_store = lambda: Store()\n"
        "COUNT = {'n': 0}\n"
        + body
        + "\np1b._ingest_file = _ingest\n"
        f"SLEEP = {per_file_sleep}\n"
        "sys.argv = ['p1b_ingest_drive_server.py', '--tier', '1', '--resume',\n"
        f"            '--priority-manifest', {str(manifest)!r},\n"
        f"            '--output', {str(tmp_path / 'report.json')!r},\n"
        f"            '--run-state', {str(tmp_path / 'state.json')!r}]\n"
        "raise SystemExit(p1b.main())\n",
        encoding="utf-8",
    )
    return driver


_SLOW_OK = (
    "def _ingest(fm, *a, **k):\n"
    "    COUNT['n'] += 1\n"
    "    open(os.path.join(PROGRESS, str(COUNT['n'])), 'w').close()\n"
    "    time.sleep(SLEEP)\n"
    "    return fm['_drive_path'], {'status': 'ok', 'rag_indexed': 2}\n"
)


def _child_env(tmp_path: Path) -> Dict[str, str]:
    env = dict(os.environ)
    env.update({
        "ENV": "testing",
        "RAG_EMBEDDING_MODEL": "fake",
        "RAG_VECTOR_NAMESPACE": "v2",
        "P1B_PARALLELISM": "1",
        "DATA_DIR": str(tmp_path / "data"),
        "PYTHONUNBUFFERED": "1",
    })
    return env


def _wait_for_progress(progress_dir: Path, n: int, *, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if progress_dir.is_dir() and len(list(progress_dir.iterdir())) >= n:
            return
        time.sleep(0.05)
    raise AssertionError(f"ingest never reached {n} files")


@POSIX_ONLY
def test_real_sigterm_mid_run_leaves_a_final_tally_and_exit_143(tmp_path):
    """The incident, reproduced as a signal instead of a mystery.

    Before this change a SIGTERM here produced exactly what run 37159882e871
    produced: a truncated log, a stale partial report and no tally line.
    """
    driver = _driver(tmp_path, body=_SLOW_OK)
    proc = subprocess.Popen(
        [sys.executable, "-u", str(driver)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env=_child_env(tmp_path), cwd=str(tmp_path),
    )
    try:
        _wait_for_progress(tmp_path / "progress", 3)
        proc.send_signal(signal.SIGTERM)
        _, err = proc.communicate(timeout=120)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()

    assert proc.returncode == 143, err
    assert "received SIGTERM (15)" in err
    assert "stop requested (SIGTERM)" in err
    assert err.count("[p1b-server] tier 1:") == 1
    assert "exit_reason=signal:SIGTERM complete=False" in err
    assert "RUN INCOMPLETE" in err
    assert "EXIT run=" in err and "reason=signal:SIGTERM" in err

    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["exit_reason"] == "signal:SIGTERM"
    assert report["complete"] is False
    done = report["global_tally"]["succeeded"]
    assert 1 <= done < 40, done
    assert len(report["results"]) == done

    state = lifecycle.read_state(tmp_path / "state.json")
    assert state["phase"] == "signal:SIGTERM"
    assert state["complete"] is False
    assert state["progress"]["done"] == done


@POSIX_ONLY
def test_a_killed_run_is_diagnosed_on_the_next_start(tmp_path):
    """SIGKILL: no handler, no report — but the next start says where it died
    and whether the box or the kernel did it."""
    driver = _driver(tmp_path, body=_SLOW_OK)
    env = _child_env(tmp_path)
    proc = subprocess.Popen(
        [sys.executable, "-u", str(driver)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env=env, cwd=str(tmp_path),
    )
    try:
        _wait_for_progress(tmp_path / "progress", 3)
        proc.send_signal(signal.SIGKILL)
        _, err = proc.communicate(timeout=120)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()

    assert proc.returncode == -9
    assert "[p1b-server] tier 1:" not in err, (
        "SIGKILL cannot produce a tally — that is what the sidecar is for"
    )
    killed_at = lifecycle.read_state(tmp_path / "state.json")
    assert killed_at["phase"] == lifecycle.PHASE_RUNNING
    assert killed_at["complete"] is False
    assert killed_at["progress"]["done"] >= 1

    postmortem = lifecycle.previous_run_postmortem(tmp_path / "state.json")
    assert "DIED WITHOUT A FINAL TALLY" in postmortem

    # The next start prints it, unprompted.
    (tmp_path / "progress2").mkdir()
    second = subprocess.run(
        [sys.executable, "-u", str(_driver(tmp_path, body=_SLOW_OK, files=1,
                                           per_file_sleep=0.0))],
        capture_output=True, text=True, env=env, cwd=str(tmp_path), timeout=180,
        check=False,
    )
    assert second.returncode == 0, second.stderr
    assert "POSTMORTEM previous run" in second.stderr
    assert "DIED WITHOUT A FINAL TALLY" in second.stderr
    assert "exit_reason=completed complete=True" in second.stderr


@POSIX_ONLY
def test_fatal_exception_still_writes_a_report_and_prints_the_traceback(tmp_path):
    """A leaked exception (the #477/#480 shape) must produce a
    tally AND a traceback, never a truncated log."""
    body = (
        "def _ingest(fm, *a, **k):\n"
        "    COUNT['n'] += 1\n"
        "    open(os.path.join(PROGRESS, str(COUNT['n'])), 'w').close()\n"
        "    return fm['_drive_path'], {'status': 'ok', 'rag_indexed': 2}\n"
        "_real_walk = gdrive_service.walk_folder\n"
        "def _walk(fid, **k):\n"
        "    if fid == 'folder-2':\n"
        "        raise RuntimeError('drive walk exploded')\n"
        "    return (list(FILES), [])\n"
        "gdrive_service.walk_folder = _walk\n"
    )
    driver = _driver(
        tmp_path,
        body=body,
        files=4,
        per_file_sleep=0.0,
        folders=[
            {"project_id": "p1", "folder_name": "folder-one", "folder_id": "folder-1"},
            {"project_id": "p2", "folder_name": "folder-two", "folder_id": "folder-2"},
        ],
    )
    proc = subprocess.run(
        [sys.executable, "-u", str(driver)],
        capture_output=True, text=True, env=_child_env(tmp_path),
        cwd=str(tmp_path), timeout=180, check=False,
    )

    assert proc.returncode != 0
    assert "Traceback (most recent call last)" in proc.stderr
    assert "RuntimeError: drive walk exploded" in proc.stderr
    assert "FATAL RuntimeError" in proc.stderr
    assert proc.stderr.count("[p1b-server] tier 1:") == 1
    assert "exit_reason=exception:RuntimeError complete=False" in proc.stderr

    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["exit_reason"] == "exception:RuntimeError"
    assert report["complete"] is False
    assert report["global_tally"]["succeeded"] == 4, "work done before the fault counts"
    assert lifecycle.read_state(tmp_path / "state.json")["phase"] == (
        "exception:RuntimeError"
    )


@POSIX_ONLY
def test_start_banner_carries_the_lifecycle_evidence(tmp_path):
    driver = _driver(tmp_path, body=_SLOW_OK, files=1, per_file_sleep=0.0)
    proc = subprocess.run(
        [sys.executable, "-u", str(driver)],
        capture_output=True, text=True, env=_child_env(tmp_path),
        cwd=str(tmp_path), timeout=180, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "[lifecycle] p1b-server tier=1" in proc.stderr
    assert "session_leader=" in proc.stderr
    assert "memory at start:" in proc.stderr
    assert "HEARTBEAT run=" not in proc.stderr  # one file, cadence is 10
    assert "EXIT run=" in proc.stderr


@POSIX_ONLY
def test_supervise_reports_a_sigkilled_child(tmp_path):
    """--supervise is the only way the operator sees the signal that killed a
    detached run; the child itself cannot report a SIGKILL."""
    child = tmp_path / "kill_me.py"
    child.write_text(
        "import os, signal\nos.kill(os.getpid(), signal.SIGKILL)\n", encoding="utf-8",
    )
    logged: List[str] = []
    code = lifecycle.run_supervised(
        [sys.executable, str(child)], log=logged.append,
    )
    assert code == 137
    assert any("killed by SIGKILL (9)" in line for line in logged)


def test_supervise_flag_reexecs_without_itself():
    import scripts.p1b_ingest_drive_server as p1b

    argv = ["scripts/p1b_ingest_drive_server.py", "--tier", "1", "--supervise",
            "--resume"]
    child = p1b.supervised_child_argv(argv)
    assert child[0] == sys.executable
    assert child[1] == "-u"
    assert "--supervise" not in child
    assert child[2:] == ["scripts/p1b_ingest_drive_server.py", "--tier", "1",
                         "--resume"]


def test_supervise_is_skipped_inside_the_supervised_child(monkeypatch, tmp_path):
    """The child must ingest, not spawn a grandchild."""
    import scripts.p1b_ingest_drive_server as p1b

    called: List[Any] = []
    monkeypatch.setattr(
        p1b, "run_under_supervisor", lambda *a, **k: called.append(a) or 0,
    )
    monkeypatch.setattr(
        p1b, "_ingest_file",
        lambda fm, *a, **k: (fm["_drive_path"], {"status": "ok", "rag_indexed": 1}),
    )
    monkeypatch.setenv(p1b.SUPERVISED_ENV, "1")
    result = _run_main(monkeypatch, tmp_path, argv_extra=["--supervise"],
                       files=_drive_files(1))
    assert called == []
    assert result["code"] == 0
    assert result["report"]["complete"] is True


# ── memory headroom ────────────────────────────────────────────────────────


class _WeakPayload(bytearray):
    """A byte payload that can be weak-referenced, so a test can prove the
    ingest really let go of it. ``bytes`` subclasses cannot hold a weakref;
    ``bytearray`` ones can, and every consumer in ``_ingest_file`` (sha256,
    the transient-copy write, ``len``) takes either."""


def test_ingest_releases_the_downloaded_payload_before_indexing(tmp_path, monkeypatch):
    """Extraction is the memory peak; holding the whole download through it was
    up to two multi-hundred-MB buffers of pure waste on a 4 GB box."""
    import weakref

    from scripts.p1b_ingest_drive_server import _ingest_file

    # The payload is handed over exactly once, so after the download the
    # ingest's own local is the only strong reference left to it.
    box = [_WeakPayload(b"%PDF-1.4 " + b"x" * 4096)]
    ref = weakref.ref(box[0])
    alive_during_index: List[bool] = []

    monkeypatch.setattr(
        "app.core.projects.add_document", lambda **_k: {"id": "doc-1"},
    )

    def _index(_pid, _did, **_k):
        alive_during_index.append(ref() is not None)
        return {"status": "ok", "rag_indexed": 5}

    monkeypatch.setattr("app.core.doc_index.index_document", _index)

    class _Drive:
        def download_file_bytes(self, _fid):
            return box.pop(), None

    _, result = _ingest_file(
        {"id": "drive-1", "name": "big.pdf", "_drive_path": "Tier1/big.pdf",
         "mimeType": "application/pdf", "size": 4105},
        "proj", tmp_path, "run-mem", _Drive(),
    )

    assert result["rag_indexed"] == 5
    assert alive_during_index == [False], (
        "the downloaded payload was still resident during index_document"
    )


def test_verification_line_still_reports_the_downloaded_size(tmp_path, monkeypatch, capsys):
    """Releasing the buffer must not cost the provenance log its byte count."""
    from scripts.p1b_ingest_drive_server import _ingest_file

    monkeypatch.setattr(
        "app.core.projects.update_document_metadata", lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "app.core.projects.set_document_file_path", lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "app.core.doc_index.index_document",
        lambda *_a, **_k: {"status": "ok", "rag_indexed": 1},
    )

    class _Drive:
        def download_file_bytes(self, _fid):
            return b"1234567890", None

    _ingest_file(
        {"id": "drive-2", "name": "retry.pdf", "_drive_path": "Tier1/retry.pdf",
         "mimeType": "application/pdf", "size": 10},
        "proj", tmp_path, "run-mem", _Drive(), existing_doc={"id": "doc-9"},
    )
    assert "downloaded_bytes=10" in capsys.readouterr().err
