"""Log records are written by a background thread from a bounded queue; a
caller never waits on stdout."""
from __future__ import annotations

import logging
import queue
import time

from app.infra import monitoring


class _Collect(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def test_records_reach_the_writer_thread():
    sink = _Collect()
    h = monitoring._queued(sink)
    try:
        log = logging.getLogger("queued-probe")
        log.addHandler(h)
        log.propagate = False
        log.warning("hello %s", "world")
        for _ in range(50):
            if sink.lines:
                break
            time.sleep(0.02)
        assert sink.lines == ["hello world"]
    finally:
        log.removeHandler(h)
        monitoring._LISTENER.stop()
        monitoring._LISTENER = None


def test_a_full_queue_drops_and_counts_instead_of_blocking():
    q = queue.Queue(maxsize=1)
    h = monitoring._DropWhenFullQueueHandler(q)
    rec = logging.LogRecord("x", logging.INFO, __file__, 1, "m", None, None)
    t = time.monotonic()
    for _ in range(5):
        h.enqueue(rec)
    assert time.monotonic() - t < 0.5
    assert h.dropped == 4 and q.qsize() == 1
    q.get_nowait()
    h.enqueue(rec)  # room again: the drop count is reported first
    note = q.get_nowait()
    assert "dropped 4 log records" in note.getMessage()
    assert h.dropped == 1  # this record found the queue full again behind the note
