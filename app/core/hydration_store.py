"""Hydration store — persistent record of nightly hydration runs.

Stores one row per (date, scope, project) tuple. ``scope`` is ``project`` for
per-project summaries and ``global`` for the cross-tenant rollup. Querying is
cheap: the index on (scope, project_id, run_date DESC) makes ``latest`` and
``history`` lookups direct.

SQLAlchemy-backed via app.core.db — unified The Fork schema.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import select

from app.core.db import SessionLocal, engine, get_database_url
from app.core.models import HydrationRun

_lock = threading.Lock()
_initialized = False
_initialized_for_url: str | None = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ensure_sqlite_parent_dir() -> None:
    url = get_database_url()
    if url.startswith("sqlite:///"):
        parent = os.path.dirname(url[len("sqlite:///") :])
        if parent:
            os.makedirs(parent, exist_ok=True)


def init_db() -> None:
    """Idempotent schema creation. Safe to call on every app startup.

    Cheap-idempotent: once created for the current database URL, repeated
    calls are a true no-op and re-issue no DDL. Startup calls this
    unconditionally on every boot (once per TestClient in the test suite);
    without this early-exit guard, DDL could re-run concurrently with a
    background task reading these tables and deadlock Postgres.
    """
    global _initialized, _initialized_for_url
    url = get_database_url()
    if _initialized and _initialized_for_url == url:
        return
    with _lock:
        if _initialized and _initialized_for_url == url:
            return
        from app.core.projects import init_db as init_projects_db

        init_projects_db()
        _ensure_sqlite_parent_dir()
        HydrationRun.__table__.create(bind=engine, checkfirst=True)
        _initialized = True
        _initialized_for_url = url


def _ensure_db() -> None:
    url = get_database_url()
    if not _initialized or _initialized_for_url != url:
        init_db()


def record_run(
    run_date: str,
    scope: str,
    project_id: Optional[str],
    summary_md: str,
    facts: Dict[str, Any],
    provider: str,
) -> str:
    """Insert one hydration row. Returns the generated id.

    ``scope`` must be ``"project"`` (project_id required) or ``"global"``
    (project_id should be None). The store does not enforce uniqueness on
    (run_date, scope, project_id) — re-running the same date appends a new
    row so audit history is preserved; ``get_latest`` returns the newest by
    created_at.
    """
    if scope not in ("project", "global"):
        raise ValueError(f"invalid scope: {scope!r}")
    if scope == "project" and not project_id:
        raise ValueError("project scope requires project_id")
    rid = str(uuid.uuid4())
    _ensure_db()
    with _lock:
        with SessionLocal() as session:
            session.add(
                HydrationRun(
                    id=rid,
                    run_date=run_date,
                    scope=scope,
                    project_id=project_id if scope == "project" else None,
                    summary_md=summary_md,
                    facts_json=facts,
                    provider=provider,
                    created_at=_now_iso(),
                )
            )
            session.commit()
    return rid


def _row_to_dict(row: HydrationRun) -> Dict[str, Any]:
    raw = row.facts_json
    try:
        if isinstance(raw, dict):
            facts = raw
        elif isinstance(raw, str):
            facts = json.loads(raw)
        else:
            facts = {}
    except Exception:
        facts = {}
    return {
        "id": row.id,
        "run_date": row.run_date,
        "scope": row.scope,
        "project_id": row.project_id,
        "summary_md": row.summary_md,
        "facts": facts,
        "provider": row.provider,
        "created_at": row.created_at,
    }


def get_latest(scope: str, project_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Return the most recently inserted run for the given scope/project."""
    _ensure_db()
    stmt = select(HydrationRun)
    if scope == "global":
        stmt = stmt.where(HydrationRun.scope == "global")
    else:
        stmt = stmt.where(
            HydrationRun.scope == "project",
            HydrationRun.project_id == project_id,
        )
    stmt = stmt.order_by(HydrationRun.created_at.desc()).limit(1)
    with SessionLocal() as session:
        row = session.scalars(stmt).first()
    return _row_to_dict(row) if row else None


def list_history(
    scope: Optional[str] = None,
    project_id: Optional[str] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    _ensure_db()
    stmt = select(HydrationRun)
    if scope is not None:
        stmt = stmt.where(HydrationRun.scope == scope)
    if project_id is not None:
        stmt = stmt.where(HydrationRun.project_id == project_id)
    stmt = stmt.order_by(HydrationRun.created_at.desc()).limit(
        max(1, min(int(limit), 200))
    )
    with SessionLocal() as session:
        rows = session.scalars(stmt).all()
    return [_row_to_dict(r) for r in rows]
