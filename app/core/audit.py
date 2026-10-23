"""Append-only audit log — Roadmap V2 · Epic 6 (data governance).

Records who/what touched client documents — uploads, deletions, purges — so
the platform can answer "what happened to this data". Stored as JSONL in
DATA_DIR/audit.log (EFS on live).

An entry is never dropped and never passes through logging (whose queue
drops diagnostics when full). If the file append fails the entry is written
to the ``audit_events`` table; if that fails too it is written straight to
the process's stderr file descriptor as one ``AUDIT_UNWRITTEN`` line, which
the container's log driver keeps. Async callers use ``arecord`` so the write
runs off the event loop.
"""

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_lock = threading.Lock()


def _audit_file() -> str:
    data_dir = os.getenv("DATA_DIR", "./data")
    try:
        os.makedirs(data_dir, exist_ok=True)
    except OSError:
        import tempfile
        data_dir = tempfile.gettempdir()
    return os.path.join(data_dir, "audit.log")


_db_ready_for = None


def _db_insert(entry: Dict[str, Any]) -> None:
    global _db_ready_for
    from app.core.db import SessionLocal, engine, get_database_url
    from app.core.models import AuditEvent

    url = get_database_url()
    if _db_ready_for != url:
        AuditEvent.__table__.create(bind=engine, checkfirst=True)
        _db_ready_for = url
    with SessionLocal() as session:
        session.add(AuditEvent(id=str(uuid.uuid4()), ts=entry["ts"], event=entry["event"],
                               project_id=entry.get("project_id"),
                               body=json.dumps(entry, default=str)))
        session.commit()


def _last_resort(entry: Dict[str, Any]) -> None:
    """Straight to fd 2: unbuffered, and not through the logging queue."""
    line = "AUDIT_UNWRITTEN " + json.dumps(entry, default=str) + "\n"
    os.write(2, line.encode("utf-8", "replace"))


def record(event: str, **details: Any) -> Dict[str, Any]:
    """Append an audit entry. Never raises into the caller and never drops
    the entry: file, then the audit_events table, then stderr."""
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **details,
    }
    try:
        with _lock:
            with open(_audit_file(), "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        return entry
    except Exception:
        # A write failure must be loud, and the operation being audited must
        # not fail because of it (an upload must not fail because the file
        # did) -- so warn, then keep the entry somewhere durable.
        logger.warning(
            "AUDIT WRITE FAILED for event %r — writing it to the audit_events table",
            event, exc_info=True,
        )
    try:
        _db_insert(entry)
    except Exception:
        _last_resort(entry)
    return entry


async def arecord(event: str, **details: Any) -> Dict[str, Any]:
    """``record`` for async callers: the write runs off the event loop."""
    from app.core.offload import off_loop

    return await off_loop(record, event, **details)


def _db_entries() -> List[Dict[str, Any]]:
    try:
        from app.core.db import SessionLocal
        from app.core.models import AuditEvent

        if _db_ready_for is None:
            from sqlalchemy import inspect

            from app.core.db import engine
            if not inspect(engine).has_table(AuditEvent.__tablename__):
                return []
        with SessionLocal() as session:
            return [json.loads(r.body) for r in session.query(AuditEvent).all()]
    except Exception:
        logger.warning("audit_events unreadable; listing the file's entries only", exc_info=True)
        return []


def read_audit(
    limit: int = 200, project_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Return the most recent audit entries, optionally scoped to a project."""
    path = _audit_file()
    entries: List[Dict[str, Any]] = []
    db = [e for e in _db_entries() if not project_id or e.get("project_id") == project_id]
    if not os.path.exists(path):
        return sorted(db, key=lambda e: e.get("ts") or "")[-limit:]
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            if project_id and e.get("project_id") != project_id:
                continue
            entries.append(e)
    if db:
        entries = sorted(entries + db, key=lambda e: e.get("ts") or "")
    return entries[-limit:]
