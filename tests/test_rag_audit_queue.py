"""The RAG audit append runs on one writer thread, not on the retrieval thread.

Every retrieval used to append its audit row to disk under a process-wide
lock on the calling thread, so concurrent retrievals queued behind each
other's file I/O. These tests hold the writer's disk append and check that
``write`` still returns at once, that rows reach disk in order once the
writer is released, that a full queue drops instead of blocking, and that
the row is the record as it was at the call.
"""
from __future__ import annotations

import json
import queue
import threading
import time

import pytest

from app.core.rag import audit


@pytest.fixture
def held_writer(monkeypatch):
    """Make the writer's disk append wait until the test releases it."""
    entered = threading.Event()
    release = threading.Event()
    real_append = audit._append

    def _held(path, lines):
        entered.set()
        assert release.wait(10.0)
        real_append(path, lines)

    monkeypatch.setattr(audit, "_append", _held)
    yield entered, release
    release.set()
    audit.flush(5.0)


def _rows(path):
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize("n_rows", [1, 25])
def test_write_returns_while_the_disk_append_is_held(monkeypatch, tmp_path, held_writer, n_rows):
    entered, release = held_writer
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    log = tmp_path / "logs" / "rag_audit.jsonl"

    audit.write({"i": -1})
    assert entered.wait(5.0)  # the writer thread is now stuck inside the append

    t0 = time.perf_counter()
    for i in range(n_rows):
        audit.write({"i": i})
    elapsed = time.perf_counter() - t0

    assert elapsed < 0.5, f"write blocked for {elapsed:.3f}s behind the disk append"
    assert threading.current_thread().name != "rag-audit-writer"
    assert _rows(log) == []  # nothing reached disk while the append is held

    release.set()
    assert audit.flush(5.0)
    assert [r["i"] for r in _rows(log)] == [-1] + list(range(n_rows))


@pytest.mark.parametrize("order", [("a", "b", "a"), ("b", "a", "b", "b")])
def test_rows_go_to_the_path_resolved_at_call_time(monkeypatch, tmp_path, order):
    for n, where in enumerate(order):
        monkeypatch.setenv("DATA_DIR", str(tmp_path / where))
        audit.write({"k": f"{where}{n}"})
    assert audit.flush(5.0)
    for where in set(order):
        expected = [f"{w}{n}" for n, w in enumerate(order) if w == where]
        assert [r["k"] for r in _rows(tmp_path / where / "logs" / "rag_audit.jsonl")] == expected


@pytest.mark.parametrize("capacity,extra", [(1, 3), (4, 2)])
def test_full_queue_drops_and_counts_instead_of_blocking(monkeypatch, tmp_path, held_writer, capacity, extra):
    entered, release = held_writer
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    log = tmp_path / "logs" / "rag_audit.jsonl"

    audit.write({"i": "first"})
    assert entered.wait(5.0)  # writer holds "first"; the queue is now empty

    monkeypatch.setattr(audit, "_QUEUE_MAX", capacity)
    before = audit.dropped()

    t0 = time.perf_counter()
    for i in range(capacity + extra):
        audit.write({"i": i})
    assert time.perf_counter() - t0 < 0.5
    assert audit.dropped() - before == extra

    release.set()
    assert audit.flush(5.0)
    assert [r["i"] for r in _rows(log)] == ["first"] + list(range(capacity))


@pytest.mark.parametrize("mutation", [
    lambda r: r.update({"injected_k": 99}),
    lambda r: r["chunks"].append({"id": "late"}),
])
def test_the_row_is_the_record_as_it_was_at_the_call(monkeypatch, tmp_path, held_writer, mutation):
    entered, release = held_writer
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    rec = {"injected_k": 2, "chunks": [{"id": "c1"}]}
    audit.write(rec)
    assert entered.wait(5.0)
    mutation(rec)  # the caller keeps using its dict after write()
    release.set()
    assert audit.flush(5.0)
    assert _rows(tmp_path / "logs" / "rag_audit.jsonl") == [{"injected_k": 2, "chunks": [{"id": "c1"}]}]


@pytest.mark.parametrize("bad", ["nul_byte", "parent_is_a_file"])
def test_a_failed_append_does_not_stop_the_writer(monkeypatch, tmp_path, bad):
    if bad == "nul_byte":
        bad_path = "\x00invalid/logs/rag_audit.jsonl"
    else:
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")
        bad_path = str(blocker / "logs" / "rag_audit.jsonl")
    good = tmp_path / "good"
    monkeypatch.setattr(audit, "_log_path", lambda: bad_path)
    audit.write({"k": "lost"})
    assert audit.flush(5.0)
    monkeypatch.setattr(audit, "_log_path", lambda: str(good / "logs" / "rag_audit.jsonl"))
    audit.write({"k": "kept"})
    assert audit.flush(5.0)
    assert [r["k"] for r in _rows(good / "logs" / "rag_audit.jsonl")] == ["kept"]


@pytest.mark.parametrize("record", [{"bad": object()}, {"s": {1, 2}}])
def test_unserialisable_values_still_log_via_str(monkeypatch, tmp_path, record):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    audit.write(record)
    assert audit.flush(5.0)
    rows = _rows(tmp_path / "logs" / "rag_audit.jsonl")
    assert len(rows) == 1 and set(rows[0]) == set(record)


@pytest.mark.parametrize("paths", [("one",), ("one", "two", "one")])
def test_drain_appends_a_batch_and_releases_its_flush(tmp_path, paths):
    """Drive the writer loop by name on a private queue."""
    q = queue.Queue()
    for n, p in enumerate(paths):
        q.put((audit._ROW, (str(tmp_path / p / "audit.jsonl"), json.dumps({"n": n}) + "\n")))
    done = threading.Event()
    q.put((audit._FLUSH, done))
    threading.Thread(target=audit._drain, args=(q,), daemon=True).start()
    assert done.wait(5.0)
    for p in set(paths):
        want = [n for n, x in enumerate(paths) if x == p]
        assert [r["n"] for r in _rows(tmp_path / p / "audit.jsonl")] == want


@pytest.mark.parametrize("bad_at", [0, 1])
def test_write_batch_groups_by_path_and_one_bad_path_spares_the_rest(tmp_path, bad_at):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    bad = str(blocker / "logs" / "audit.jsonl")
    good = str(tmp_path / "good" / "audit.jsonl")
    rows = [(good, '{"n": 0}\n'), (good, '{"n": 1}\n')]
    rows.insert(bad_at, (bad, '{"n": "lost"}\n'))
    audit._write_batch(rows)
    assert [r["n"] for r in _rows(tmp_path / "good" / "audit.jsonl")] == [0, 1]


@pytest.mark.parametrize("n_threads", [1, 8])
def test_ensure_writer_keeps_exactly_one_writer_thread(monkeypatch, tmp_path, n_threads):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    def _burst():
        for i in range(20):
            audit._ensure_writer()
            audit.write({"i": i})

    workers = [threading.Thread(target=_burst) for _ in range(n_threads)]
    for w in workers:
        w.start()
    for w in workers:
        w.join(10.0)
    assert audit.flush(5.0)
    writers = [t for t in threading.enumerate() if t.name == "rag-audit-writer" and t.is_alive()]
    assert len(writers) == 1
    assert len(_rows(tmp_path / "logs" / "rag_audit.jsonl")) == 20 * n_threads
