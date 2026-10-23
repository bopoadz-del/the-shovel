"""One file can no longer take the ingest down, and no file is retried forever.

Live 2026-10-03: every pass of the Drive ingest reached file 723 of 1,048
and was OOM-killed on the same drawing PDF. Text extraction was already
isolated; the table stages ran in the parent. And open outcomes
(ZERO_CHUNK, EXTRACT_FAILED, UNVERIFIED) were retried on every pass with no
bound, so a deterministic failure repeated forever.

These tests pin the mechanism, not the file:
  * the whole chunk-producing stage of a file runs in its own child; a child
    that exhausts memory, is killed, or runs past its time leaves the parent
    running and the file EXTRACT_FAILED;
  * one bad file in a batch does not stop the others;
  * an open outcome is retried only up to INGEST_MAX_ATTEMPTS, then closed
    for this extractor, and resume skips it.
The fork-based tests need POSIX fork (CI is Linux); the retry tests run
everywhere.
"""
from __future__ import annotations

import importlib
import os
import signal
import time

import pytest

from tests.conftest import _postgres_test_mode

pytestmark = pytest.mark.skipif(
    _postgres_test_mode(), reason="SQLite-backed ingest tests",
)

needs_fork = pytest.mark.skipif(
    os.name != "posix", reason="per-file isolation needs POSIX fork",
)


def _reload(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("DATA_ENCRYPTION_KEY", raising=False)
    import app.core.db as db_mod
    import app.core.users as users_mod
    from app.core import projects
    from app.core.rag.embeddings import reset_embedder_cache
    from app.core.rag.vector_store import reset_store_cache

    importlib.reload(db_mod)
    importlib.reload(users_mod)
    users_mod._initialized = False
    projects._initialized = False
    reset_store_cache()
    reset_embedder_cache()
    import app.core.doc_index as doc_index_mod

    doc_index_mod = importlib.reload(doc_index_mod)
    projects = importlib.reload(projects)
    projects.init_db()
    users_mod.ensure_user_exists("u1")
    pid = projects.create_project(name="P", client="C", user_id="u1")["id"]
    return projects, doc_index_mod, pid


def _add_txt(projects, tmp_path, pid, name, body: bytes):
    from app.core import file_crypto

    path = str(tmp_path / name)
    file_crypto.write_document(path, body)
    return projects.add_document(
        pid, name, file_path=path, size=len(body),
        content_sha256=f"sha-{name}",
    )


def _isolate_everything(monkeypatch):
    """Isolation threshold to 0 so every file takes the child path."""
    from app.core import extract_isolated

    monkeypatch.setattr(extract_isolated, "_MIN_MB", 0.0)


# ── per-file isolation ─────────────────────────────────────────────────────


@needs_fork
def test_child_over_memory_limit_parent_continues_file_extract_failed(monkeypatch, tmp_path):
    projects, doc_index, pid = _reload(monkeypatch, tmp_path)
    _isolate_everything(monkeypatch)
    monkeypatch.setenv("DOC_INDEX_FILE_MEM_MB", "64")
    real = doc_index._produce_chunks

    def _greedy(*args):
        hoard = []
        while True:  # well past a 64 MB address-space budget
            hoard.append(bytearray(32 * 1024 * 1024))
        return real(*args)  # pragma: no cover

    monkeypatch.setattr(doc_index, "_produce_chunks", _greedy)
    doc = _add_txt(projects, tmp_path, pid, "a.txt", b"some words " * 50)

    result = doc_index.index_document(pid, doc["id"])

    assert result["error"] == "ZERO_CHUNK", result
    assert "memory" in (result.get("extract_error") or ""), result
    assert projects.get_document(doc["id"])["ingest_status"] == "EXTRACT_FAILED"


@needs_fork
def test_child_killed_parent_continues_file_extract_failed(monkeypatch, tmp_path):
    """The kernel OOM killer's SIGKILL reaches the child, not the parent."""
    projects, doc_index, pid = _reload(monkeypatch, tmp_path)
    _isolate_everything(monkeypatch)

    def _killed(*args):
        os.kill(os.getpid(), signal.SIGKILL)

    monkeypatch.setattr(doc_index, "_produce_chunks", _killed)
    doc = _add_txt(projects, tmp_path, pid, "a.txt", b"some words " * 50)

    result = doc_index.index_document(pid, doc["id"])

    assert result["error"] == "ZERO_CHUNK", result
    assert "crash" in (result.get("extract_error") or ""), result
    assert projects.get_document(doc["id"])["ingest_status"] == "EXTRACT_FAILED"


@needs_fork
def test_child_timeout_parent_continues_file_extract_failed(monkeypatch, tmp_path):
    projects, doc_index, pid = _reload(monkeypatch, tmp_path)
    _isolate_everything(monkeypatch)
    monkeypatch.setenv("DOC_INDEX_FILE_TIMEOUT_S", "1")

    def _wedged(*args):
        time.sleep(60)

    monkeypatch.setattr(doc_index, "_produce_chunks", _wedged)
    doc = _add_txt(projects, tmp_path, pid, "a.txt", b"some words " * 50)

    t0 = time.monotonic()
    result = doc_index.index_document(pid, doc["id"])

    assert time.monotonic() - t0 < 30, "parent waited out the wedged child"
    assert "timeout" in (result.get("extract_error") or ""), result
    assert projects.get_document(doc["id"])["ingest_status"] == "EXTRACT_FAILED"


@needs_fork
def test_one_bad_file_in_a_batch_the_others_are_processed(monkeypatch, tmp_path):
    projects, doc_index, pid = _reload(monkeypatch, tmp_path)
    _isolate_everything(monkeypatch)
    real = doc_index._produce_chunks
    marker = "POISON"

    def _one_bad(file_path, filename, *rest):
        if marker in filename:
            os.kill(os.getpid(), signal.SIGKILL)
        return real(file_path, filename, *rest)

    monkeypatch.setattr(doc_index, "_produce_chunks", _one_bad)
    docs = [
        _add_txt(projects, tmp_path, pid, f"{'POISON' if i == 2 else 'good'}{i}.txt",
                 f"retaining wall drainage clause {i} ".encode() * 40)
        for i in range(5)
    ]

    results = [doc_index.index_document(pid, d["id"]) for d in docs]

    good = [r for i, r in enumerate(results) if i != 2]
    assert all(r.get("status") == "ok" and r.get("total_chunks", 0) > 0 for r in good), good
    assert results[2]["error"] == "ZERO_CHUNK"
    statuses = [projects.get_document(d["id"])["ingest_status"] for d in docs]
    assert statuses[2] == "EXTRACT_FAILED"
    assert all(s != "EXTRACT_FAILED" for i, s in enumerate(statuses) if i != 2), statuses


def test_isolated_child_drops_inherited_db_pool_without_closing_it(monkeypatch):
    """A forked child must never share the parent's DB sockets."""
    from app.core import extract_isolated

    calls = []

    class _Engine:
        def dispose(self, close=True):
            calls.append(close)

    monkeypatch.setattr("app.core.db.get_engine", lambda: _Engine())
    extract_isolated._forget_inherited_db_pools()
    assert calls == [False]


# ── bounded retries ────────────────────────────────────────────────────────


def test_open_outcome_closes_at_max_attempts_and_resume_skips_it(monkeypatch, tmp_path):
    projects, doc_index, pid = _reload(monkeypatch, tmp_path)
    from app.core import ingest_status as ist

    monkeypatch.setenv("INGEST_MAX_ATTEMPTS", "2")
    monkeypatch.setattr(doc_index, "_produce_chunks", lambda *a: ([], {}))
    doc = _add_txt(projects, tmp_path, pid, "empty.txt", b"x")

    doc_index.index_document(pid, doc["id"])
    row = projects.get_document(doc["id"])
    assert row["ingest_status"] == ist.ZERO_CHUNK
    assert ist.is_open(row["ingest_status"], row["ingest_status_reason"])
    assert not ist.resume_is_already_indexed(row, 0)  # 1 of 2: retried

    doc_index.index_document(pid, doc["id"])
    row = projects.get_document(doc["id"])
    assert row["ingest_status"] == ist.ZERO_CHUNK
    assert ist.attempts_exhausted(row["ingest_status_reason"])
    assert not ist.is_open(row["ingest_status"], row["ingest_status_reason"])
    assert ist.resume_is_already_indexed(row, 0)  # 2 of 2: skipped


def test_bound_applies_uniformly_to_every_recoverable_class(monkeypatch):
    from app.core import ingest_status as ist

    for status, reason in (
        (ist.UNVERIFIED, None),
        (ist.ZERO_CHUNK, None),
        (ist.EXTRACT_FAILED, None),
        (ist.UNSUPPORTED_TYPE, "dwg:recoverable"),
        (ist.TEXT_SPARSE, "single_window:recoverable"),
    ):
        assert ist.is_open(status, reason), (status, reason)
        closed = ist.exhausted_reason(reason, 1)
        assert not ist.is_open(status, closed), (status, closed)


def test_recoverable_class_keeps_its_reason_but_resume_bounds_it(monkeypatch):
    """A recoverable skip stays named as recoverable work on the ledger; the
    retry bound is applied by resume from the attempt count."""
    from app.core import ingest_status as ist

    monkeypatch.setenv("INGEST_MAX_ATTEMPTS", "1")
    dwg = {
        "original_name": "sheet.dwg", "ingest_status": ist.UNSUPPORTED_TYPE,
        "ingest_status_reason": "dwg:recoverable", "content_sha256": "abc",
    }
    parsers = {".pdf", ".dwg"}  # a build that can parse it: work, until tried
    assert not ist.resume_is_already_indexed(dwg, 0, parseable_exts=parsers)
    tried = {**dwg, "metadata": {"ingest_attempts": {"n": 1, "key": ist.attempt_key("abc")}}}
    assert ist.resume_is_already_indexed(tried, 0, parseable_exts=parsers)
    assert tried["ingest_status_reason"] == "dwg:recoverable"


def test_a_killed_attempt_still_counts_toward_the_bound(monkeypatch):
    """No outcome was stamped (the process died) -- the count alone bounds it."""
    from app.core import ingest_status as ist

    monkeypatch.setenv("INGEST_MAX_ATTEMPTS", "1")
    doc = {
        "original_name": "a.pdf", "ingest_status": ist.UNVERIFIED,
        "content_sha256": "abc",
        "metadata": {"ingest_attempts": {"n": 1, "key": ist.attempt_key("abc")}},
    }
    assert ist.resume_is_already_indexed(doc, 0)


def test_new_bytes_or_new_extractor_reopen_an_exhausted_row(monkeypatch):
    from app.core import ingest_status as ist

    monkeypatch.setenv("INGEST_MAX_ATTEMPTS", "1")
    doc = {
        "original_name": "a.pdf", "ingest_status": ist.UNVERIFIED,
        "content_sha256": "abc",
        "metadata": {"ingest_attempts": {"n": 1, "key": ist.attempt_key("abc")}},
    }
    assert ist.resume_is_already_indexed(doc, 0)
    changed = {**doc, "content_sha256": "def"}
    assert not ist.resume_is_already_indexed(changed, 0)
    monkeypatch.setattr(ist, "EXTRACTOR_VERSION", "next-extractor")
    assert not ist.resume_is_already_indexed(doc, 0)
    assert ist.is_open(ist.ZERO_CHUNK, "x|attempts_exhausted@older:1:terminal")


def test_attempt_counter_restarts_on_a_new_key(monkeypatch, tmp_path):
    projects, _doc_index, pid = _reload(monkeypatch, tmp_path)
    doc = _add_txt(projects, tmp_path, pid, "a.txt", b"words")
    assert projects.record_ingest_attempt(doc["id"], "k1") == 1
    assert projects.record_ingest_attempt(doc["id"], "k1") == 2
    assert projects.record_ingest_attempt(doc["id"], "k2") == 1
    assert projects.ingest_attempts(projects.get_document(doc["id"]), "k2") == 1


def test_duplicate_content_source_is_recorded_once(monkeypatch, tmp_path):
    projects, _doc_index, pid = _reload(monkeypatch, tmp_path)
    doc = _add_txt(projects, tmp_path, pid, "a.txt", b"words")
    projects.record_source_alias(doc["id"], "src-2", "tok")
    projects.record_source_alias(doc["id"], "src-2", "tok")
    meta = projects.get_document(doc["id"])["metadata"]
    assert meta["source_aliases"] == {"src-2": "tok"}


# ── uniform bound, per status class ────────────────────────────────────────

_CLASSES = [
    # (label, status, reason, extension, chunk_count, extractor_version)
    ("unverified", "UNVERIFIED", None, "a.pdf", 0, None),
    ("zero_chunk", "ZERO_CHUNK", None, "a.pdf", 0, None),
    ("extract_failed", "EXTRACT_FAILED", None, "a.pdf", 0, None),
    ("recoverable_skip", "UNSUPPORTED_TYPE", "dwg:recoverable", "sheet.dwg", 0, None),
    ("stale_extractor_docx", "TEXT_SPARSE", "single_window:terminal", "spec.docx", 1, "old-extractor"),
]


def _simulate_passes(doc, chunk_count, parsers, passes=10):
    """Run ``passes`` resume decisions; each one that assigns the file counts
    an attempt on the ledger row, exactly as index_document does."""
    from app.core import ingest_status as ist

    processed = 0
    for _ in range(passes):
        if ist.resume_is_already_indexed(doc, chunk_count, parseable_exts=parsers):
            continue
        processed += 1
        key = ist.attempt_key(doc.get("content_sha256"))
        prior = (doc.setdefault("metadata", {}).get("ingest_attempts") or {})
        n = prior.get("n", 0) if prior.get("key") == key else 0
        doc["metadata"]["ingest_attempts"] = {"n": n + 1, "key": key}
    return processed


@pytest.mark.parametrize("label,status,reason,name,chunks,extractor", _CLASSES, ids=[c[0] for c in _CLASSES])
@pytest.mark.parametrize("bound", [1, 3])
def test_each_class_is_processed_at_most_max_attempts_times(
    monkeypatch, label, status, reason, name, chunks, extractor, bound,
):
    from app.core import ingest_status as ist

    monkeypatch.setenv("INGEST_MAX_ATTEMPTS", str(bound))
    doc = {"original_name": name, "ingest_status": status, "ingest_status_reason": reason,
           "extractor_version": extractor, "content_sha256": "bytes-1"}
    parsers = {".pdf", ".docx", ".dwg"}  # a build that can parse every class here
    assert not ist.resume_is_already_indexed(doc, chunks, parseable_exts=parsers), "starts as work"

    assert _simulate_passes(doc, chunks, parsers) == bound
    assert ist.resume_is_already_indexed(doc, chunks, parseable_exts=parsers), "skipped afterwards"


@pytest.mark.parametrize("label,status,reason,name,chunks,extractor", _CLASSES, ids=[c[0] for c in _CLASSES])
def test_each_class_reopens_on_new_bytes_and_on_a_new_extractor(
    monkeypatch, label, status, reason, name, chunks, extractor,
):
    from app.core import ingest_status as ist

    monkeypatch.setenv("INGEST_MAX_ATTEMPTS", "1")
    parsers = {".pdf", ".docx", ".dwg"}
    doc = {"original_name": name, "ingest_status": status, "ingest_status_reason": reason,
           "extractor_version": extractor, "content_sha256": "bytes-1"}
    _simulate_passes(doc, chunks, parsers)
    assert ist.resume_is_already_indexed(doc, chunks, parseable_exts=parsers)

    changed = {**doc, "content_sha256": "bytes-2"}
    assert _simulate_passes(changed, chunks, parsers) == 1, "new source content reopens once"

    monkeypatch.setattr(ist, "EXTRACTOR_VERSION", "next-extractor")
    assert _simulate_passes(doc, chunks, parsers) == 1, "a new extractor reopens once"


@needs_fork
def test_child_process_does_not_share_parent_db_connection(monkeypatch, tmp_path):
    """The child that runs _produce_chunks starts with an EMPTY pool (the
    parent's pooled connections are detached, not closed), can still read the
    DB on its own connection, and the parent's pool is intact afterwards."""
    _projects, _doc_index, _pid = _reload(monkeypatch, tmp_path)
    from sqlalchemy import text

    from app.core.db import get_engine
    from app.core.extract_isolated import run_isolated

    eng = get_engine()
    with eng.connect() as conn:  # leave one connection checked in to the pool
        conn.execute(text("SELECT 1"))
    parent_pooled = eng.pool.checkedin()
    assert parent_pooled >= 1

    def _in_child():
        from app.core.db import get_engine as _ge

        e = _ge()
        inherited = e.pool.checkedin()
        with e.connect() as c:
            projects_seen = c.execute(text("SELECT count(*) FROM projects")).scalar()
        return inherited, projects_seen

    (inherited, projects_seen), diag = run_isolated(_in_child, (), fallback=(None, None))
    assert diag == {}
    assert inherited == 0, "child inherited the parent's pooled connections"
    assert projects_seen >= 1, "child could not read the DB on its own connection"
    assert eng.pool.checkedin() == parent_pooled, "parent pool was disturbed"
    with eng.connect() as conn:
        assert conn.execute(text("SELECT 1")).scalar() == 1


# ── container memory guard ─────────────────────────────────────────────────


def test_memory_pressure_reads_cgroup_anon_against_its_limit(tmp_path):
    from app.core.ingest_lifecycle import memory_pressure

    proc, cg = tmp_path / "proc", tmp_path / "cg"
    (proc / "self").mkdir(parents=True)
    (proc / "self" / "cgroup").write_text("0::/\n")
    (proc / "meminfo").write_text("MemTotal: 8388608 kB\nMemAvailable: 4194304 kB\n")
    cg.mkdir()
    (cg / "memory.max").write_text(str(4 * 1024**3))
    (cg / "memory.stat").write_text(f"anon {3 * 1024**3}\nfile {1024**3}\n")
    assert memory_pressure(proc_root=proc, cgroup_root=cg) == pytest.approx(0.75)


def test_memory_pressure_falls_back_to_meminfo_when_the_cgroup_hides_it(tmp_path):
    """Fargate: memory.max reads 'max' and memory.stat is absent; the micro-VM
    is the task, so MemTotal - MemAvailable over MemTotal is the pressure."""
    from app.core.ingest_lifecycle import memory_pressure, read_memory_snapshot

    proc, cg = tmp_path / "proc", tmp_path / "cg"
    (proc / "self").mkdir(parents=True)
    (proc / "self" / "cgroup").write_text("0::/\n")
    (proc / "self" / "status").write_text("VmRSS: 1024 kB\nVmHWM: 2048 kB\n")
    (proc / "meminfo").write_text(
        "MemTotal: 4194304 kB\nMemAvailable: 1048576 kB\nCached: 1572864 kB\n")
    cg.mkdir()
    (cg / "memory.max").write_text("max\n")
    assert memory_pressure(proc_root=proc, cgroup_root=cg) == pytest.approx(0.75)
    snap = read_memory_snapshot(proc_root=proc, cgroup_root=cg)
    assert snap.anon_mb == pytest.approx(3072.0) and snap.file_mb == pytest.approx(1536.0)
    assert "anon=3072MB" in snap.as_line() and "page_cache=1536MB" in snap.as_line()


@needs_fork
def test_container_guard_stops_the_child_before_the_container_dies(monkeypatch):
    """Copy-on-write growth is invisible to RLIMIT_AS; the parent's guard on
    container memory stops the child, and the parent carries on."""
    from app.core import extract_isolated

    monkeypatch.setenv("DOC_ISOLATE_MEM_GUARD_FRACTION", "0.85")
    monkeypatch.setattr(extract_isolated, "_memory_pressure", lambda: 0.97)

    t0 = time.monotonic()
    result, diag = extract_isolated.run_isolated(
        time.sleep, (60,), fallback="fallback", label="indexing of x", timeout_s=120,
    )
    assert time.monotonic() - t0 < 30, "guard did not stop the child"
    assert result == "fallback"
    assert diag["extract_failed"] == "memory_guard"
    assert "97%" in diag["extract_failed_detail"]


@needs_fork
def test_container_guard_off_at_zero(monkeypatch):
    from app.core import extract_isolated

    monkeypatch.setenv("DOC_ISOLATE_MEM_GUARD_FRACTION", "0")
    monkeypatch.setattr(extract_isolated, "_memory_pressure", lambda: 0.99)
    result, diag = extract_isolated.run_isolated(lambda: "done", (), fallback=None)
    assert result == "done" and diag == {}


# ── memory admission ───────────────────────────────────────────────────────


def test_memory_demand_is_one_plaintext_copy_each_side():
    """No archive and no encryption on the ingest path: one copy of the file."""
    from app.core.extract_isolated import memory_demand

    mb = 1024 * 1024
    assert memory_demand(300 * mb) == (300 * mb, 300 * mb)
    assert memory_demand(0) == (0, 0)


def test_admission_refuses_over_the_child_budget(monkeypatch):
    from app.core import extract_isolated, ingest_lifecycle

    monkeypatch.setattr(ingest_lifecycle, "memory_numbers", lambda: (None, None))
    mb = 1024 * 1024
    reason = extract_isolated.admission_refusal(1600 * mb, child_budget_mb=1536)
    assert reason and "child budget" in reason
    assert extract_isolated.admission_refusal(1500 * mb, child_budget_mb=1536) is None
    assert extract_isolated.admission_refusal(100 * mb, child_budget_mb=1536) is None


def test_admission_refuses_over_the_parent_headroom(monkeypatch):
    from app.core import extract_isolated, ingest_lifecycle

    gb = 1024 ** 3
    monkeypatch.setenv("DOC_ISOLATE_MEM_GUARD_FRACTION", "0.85")
    monkeypatch.setattr(ingest_lifecycle, "memory_numbers", lambda: (1 * gb, 4 * gb))
    mb = 1024 * 1024
    # headroom 4 GB x 0.85 - 1 GB = 2457 MB
    reason = extract_isolated.admission_refusal(2500 * mb, child_budget_mb=4096)
    assert reason and "container headroom" in reason
    assert extract_isolated.admission_refusal(2400 * mb, child_budget_mb=4096) is None


def test_memory_limit_is_the_tightest_one_stated(monkeypatch, tmp_path):
    """Fargate: cgroup says 'max', MemTotal is the micro-VM, the task declares
    less -- the task's declared limit wins."""
    from app.core import ingest_lifecycle as il

    proc, cg = tmp_path / "proc", tmp_path / "cg"
    (proc / "self").mkdir(parents=True)
    (proc / "self" / "cgroup").write_text("0::/\n")
    (proc / "meminfo").write_text("MemTotal: 8388608 kB\nMemAvailable: 7340032 kB\n")
    cg.mkdir()
    (cg / "memory.max").write_text("max\n")
    monkeypatch.setattr(il, "ecs_task_memory_limit_bytes", lambda: 4096 * 1024 * 1024)
    used, limit = il.memory_numbers(proc_root=proc, cgroup_root=cg)
    assert limit == 4096 * 1024 * 1024
    assert used == 1024 * 1024 * 1024


def test_ecs_task_memory_limit_reads_the_metadata_endpoint(monkeypatch):
    import io
    import json
    import urllib.request

    from app.core import ingest_lifecycle as il

    il._ECS_LIMIT_CACHE.clear()
    monkeypatch.setenv("ECS_CONTAINER_METADATA_URI_V4", "http://meta.invalid/v4/x")
    seen = []

    def _open(url, timeout=0):
        seen.append(url)
        return io.BytesIO(json.dumps({"Limits": {"CPU": 1, "Memory": 4096}}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", _open)
    assert il.ecs_task_memory_limit_bytes() == 4096 * 1024 * 1024
    assert seen == ["http://meta.invalid/v4/x/task"]
    monkeypatch.delenv("ECS_CONTAINER_METADATA_URI_V4")
    assert il.ecs_task_memory_limit_bytes() is None


def test_release_freed_memory_is_safe_everywhere():
    from app.core.ingest_lifecycle import release_freed_memory

    release_freed_memory()
