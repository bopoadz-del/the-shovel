"""Daily RAG injection token budget.

Layered on top of the per-turn ``MAX_RAG_TOKENS`` cap and the
``RAG_CONFIDENCE_THRESHOLD`` short-circuit so RAG token spend has
three independent degradation paths. When the day's consumed sum
reaches the budget INCLUSIVELY (``consumed >= budget``), the
injector degrades to ``K=2`` for the remaining turns of the day.

Day rollover is implicit: callers pass ``day=utc.strftime("%Y-%m-%d")``
and a fresh row materialises on the first read of a new date.

SQLAlchemy-backed via app.core.db — unified The Fork schema.
"""
from __future__ import annotations

import os
import threading
from typing import Dict

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from app.core.db import SessionLocal, engine, get_database_url
from app.core.models import RagBudget

_LOCK = threading.RLock()
_initialized = False
_initialized_for_url: str | None = None


def _ensure_sqlite_parent_dir() -> None:
    url = get_database_url()
    if url.startswith("sqlite:///"):
        parent = os.path.dirname(url[len("sqlite:///") :])
        if parent:
            os.makedirs(parent, exist_ok=True)


def _ensure_db() -> None:
    global _initialized, _initialized_for_url
    url = get_database_url()
    with _LOCK:
        if not _initialized or _initialized_for_url != url:
            _ensure_sqlite_parent_dir()
            RagBudget.__table__.create(bind=engine, checkfirst=True)
            _initialized = True
            _initialized_for_url = url


def _budget_value() -> int:
    try:
        v = int(os.getenv("RAG_DAILY_TOKEN_BUDGET", "2000000"))
    except ValueError:
        return 2000000
    return v if v >= 0 else 2000000


def _consume_stmt(day: str, tokens: int):
    values = {"day": day, "consumed": int(tokens)}
    if get_database_url().startswith("postgresql"):
        ins = pg_insert(RagBudget).values(**values)
    else:
        ins = sqlite_insert(RagBudget).values(**values)
    return ins.on_conflict_do_update(
        index_elements=[RagBudget.day],
        set_={"consumed": RagBudget.consumed + ins.excluded.consumed},
    )


def snapshot(day: str) -> Dict[str, object]:
    """Return the day's current budget state without mutating it."""
    _ensure_db()
    with _LOCK:
        with SessionLocal() as session:
            row = session.get(RagBudget, day)
    consumed = int(row.consumed) if row else 0
    budget = _budget_value()
    return {
        "day": day,
        "consumed": consumed,
        "budget": budget,
        "remaining": max(0, budget - consumed),
        "degraded": consumed >= budget,  # INCLUSIVE boundary (see spec)
    }


def consume(day: str, tokens: int) -> None:
    """Add ``tokens`` to the day's consumed counter atomically."""
    if tokens <= 0:
        return
    _ensure_db()
    with _LOCK:
        with SessionLocal() as session:
            session.execute(_consume_stmt(day, tokens))
            session.commit()
