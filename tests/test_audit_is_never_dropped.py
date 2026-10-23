"""Audit entries are never dropped: not when the diagnostic log queue is
full, not when the audit file cannot be written. Synthetic data."""
from __future__ import annotations

import json
import logging
import queue

import pytest
from fastapi.testclient import TestClient

from app.core import audit
from app.infra import monitoring
from tests._dev_key import DEV_KEY

H = {"Authorization": "Bearer " + DEV_KEY}


@pytest.fixture
def full_log_queue():
    """Root logging goes to a full queue that nothing drains."""
    q = queue.Queue(maxsize=1)
    q.put_nowait(None)
    handler = monitoring._DropWhenFullQueueHandler(q)
    root = logging.getLogger()
    saved = root.handlers[:]
    root.handlers = [handler]
    try:
        yield handler
    finally:
        root.handlers = saved


def test_an_audited_action_produces_its_row_while_the_log_queue_is_full(full_log_queue):
    from app.main import app

    before = monitoring.LOG_RECORDS_DROPPED
    with TestClient(app) as client:
        logging.getLogger("synthetic").warning("this one is dropped")
        pid = client.post("/v1/projects", json={"name": "Audit Full Queue"}, headers=H).json()["id"]
    assert full_log_queue.dropped >= 1
    assert monitoring.LOG_RECORDS_DROPPED > before  # the drop is counted
    events = [e["event"] for e in audit.read_audit(project_id=pid)]
    assert "project.created" in events


def test_an_entry_the_file_cannot_take_goes_to_the_audit_events_table(monkeypatch):
    def _unwritable():
        return "/nonexistent-dir-for-audit/audit.log"

    monkeypatch.setattr(audit, "_audit_file", _unwritable)
    entry = audit.record("document.deleted", project_id="p-audit-db", document_id="d-1")
    assert entry["event"] == "document.deleted"
    rows = audit.read_audit(project_id="p-audit-db")
    assert any(r["event"] == "document.deleted" and r["document_id"] == "d-1" for r in rows)


def test_when_file_and_table_both_fail_the_entry_goes_to_stderr(monkeypatch, capfd):
    monkeypatch.setattr(audit, "_audit_file", lambda: "/nonexistent-dir-for-audit/audit.log")

    def _db_down(_entry):
        raise RuntimeError("database unreachable")

    monkeypatch.setattr(audit, "_db_insert", _db_down)
    audit.record("document.purged", project_id="p-audit-err", document_id="d-2")
    err = capfd.readouterr().err
    line = next(ln for ln in err.splitlines() if ln.startswith("AUDIT_UNWRITTEN "))
    kept = json.loads(line[len("AUDIT_UNWRITTEN "):])
    assert kept["event"] == "document.purged" and kept["document_id"] == "d-2"


def test_async_callers_write_off_the_event_loop(monkeypatch):
    import asyncio
    import threading

    seen = {}

    def _spy(event, **details):
        seen["thread"] = threading.current_thread()
        return {"event": event, **details}

    monkeypatch.setattr(audit, "record", _spy)

    async def run():
        seen["loop_thread"] = threading.current_thread()
        return await audit.arecord("project.updated", project_id="p-async")

    assert asyncio.run(run())["event"] == "project.updated"
    assert seen["thread"] is not seen["loop_thread"]
