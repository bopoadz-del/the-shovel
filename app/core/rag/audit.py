"""Append-only JSONL audit log for RAG retrieval activity.

One row per turn at ``${DATA_DIR}/logs/rag_audit.jsonl``. Used by both
the runtime injection path and the chat block to record retrieval
outcomes for offline tuning of K, the confidence threshold, the daily
budget, and the noise filter regex.

Best-effort: write failures are swallowed with a logger.warning. The
runtime must never refuse a chat turn because the audit log couldn't
write.

Off the request path: ``write`` serialises the record on the caller's
thread (so a later mutation of the caller's dict cannot change or break
the logged row) and hands the line to ONE background writer thread
through a bounded queue. The caller never takes a file lock and never
touches the disk, so concurrent retrievals no longer queue behind each
other's file appends. When the queue is full the row is dropped and
counted (``dropped()``) rather than blocking the turn. ``flush`` waits
for every queued row to reach disk (tests, shutdown).
"""
from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import threading
from typing import Any, Dict, List, Optional, Tuple

_LOG = logging.getLogger(__name__)

_QUEUE_MAX = max(1, int(os.getenv("RAG_AUDIT_QUEUE_MAX", "10000") or "10000"))

_START_LOCK = threading.Lock()
_queue: "queue.Queue[Tuple[str, Any]]" = queue.Queue()
_writer: Optional[threading.Thread] = None
_writer_pid: Optional[int] = None
_dropped = 0

_ROW = "row"
_FLUSH = "flush"


def _log_path() -> str:
    base = os.getenv("DATA_DIR", "./data")
    return os.path.join(base, "logs", "rag_audit.jsonl")


def _append(path: str, lines: List[str]) -> None:
    """Append ``lines`` to ``path``. Runs on the writer thread only."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write("".join(lines))


def _write_batch(rows: List[Tuple[str, str]]) -> None:
    by_path: Dict[str, List[str]] = {}
    for path, line in rows:
        by_path.setdefault(path, []).append(line)
    for path, lines in by_path.items():
        try:
            _append(path, lines)
        except Exception as exc:
            _LOG.warning("rag_audit write failed: %s", exc)


def _drain() -> None:
    while True:
        kind, item = _queue.get()
        rows: List[Tuple[str, str]] = []
        flushes: List[threading.Event] = []
        while True:
            if kind == _ROW:
                rows.append(item)
            else:
                flushes.append(item)
            try:
                kind, item = _queue.get_nowait()
            except queue.Empty:
                break
        _write_batch(rows)
        for ev in flushes:
            ev.set()


def _ensure_writer() -> None:
    global _writer, _writer_pid
    pid = os.getpid()
    if _writer is not None and _writer_pid == pid and _writer.is_alive():
        return
    with _START_LOCK:
        if _writer is not None and _writer_pid == pid and _writer.is_alive():
            return
        t = threading.Thread(target=_drain, name="rag-audit-writer", daemon=True)
        t.start()
        _writer, _writer_pid = t, pid


def write(record: Dict[str, Any]) -> None:
    """Queue ``record`` as one JSON line. Best-effort, never raises, never blocks."""
    global _dropped
    try:
        line = json.dumps(record, default=str) + "\n"
        path = _log_path()
        _ensure_writer()
        if _queue.qsize() >= _QUEUE_MAX:
            _dropped += 1
            _LOG.warning("rag_audit queue full (%d); row dropped (total dropped %d)",
                         _QUEUE_MAX, _dropped)
            return
        _queue.put_nowait((_ROW, (path, line)))
    except Exception as exc:
        _LOG.warning("rag_audit write failed: %s", exc)


def dropped() -> int:
    """Rows dropped because the queue was full, since process start."""
    return _dropped


def flush(timeout: float = 5.0) -> bool:
    """Block until every row queued before this call is on disk.

    Returns False if the writer did not get there within ``timeout``.
    """
    try:
        _ensure_writer()
        ev = threading.Event()
        _queue.put((_FLUSH, ev), timeout=timeout)
        return ev.wait(timeout)
    except Exception as exc:
        _LOG.warning("rag_audit flush failed: %s", exc)
        return False


atexit.register(flush, 2.0)
