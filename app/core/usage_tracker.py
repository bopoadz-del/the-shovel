"""LLM usage + soft-cap tracker.

Records every LLM round-trip (`prompt_tokens`, `completion_tokens`,
estimated cost) to the unified The Fork database and lets the runtime block
calls once a per-user daily budget is exceeded.

Wired in two places:

* ``app/agents/runtime.py::_call_llm`` — records every response's
  ``usage`` field and short-circuits when ``USAGE_DAILY_CAP_USD`` is
  set and the caller is over the cap.
* ``app/routers/usage.py`` — read endpoints for the UI: today, last 7
  days, per-agent + per-provider breakdown.

Cost model is a hardcoded per-provider+model rate table. Free-tier Groq
models are recorded at $0 — total_tokens still tracked so we can warn
before hitting Groq's daily TPD cap.

SQLAlchemy-backed via app.core.db — unified The Fork schema.
"""
from __future__ import annotations

import logging
import os
import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import func, select

from app.core.db import SessionLocal, engine, get_database_url
from app.core.models import UsageRun

logger = logging.getLogger(__name__)

_LOCK = threading.RLock()
_initialized = False
_initialized_for_url: str | None = None


def _db_path() -> str:
    """SQLite file path for the unified DB (test / debug helper)."""
    url = get_database_url()
    if url.startswith("sqlite:///"):
        return url[len("sqlite:///") :]
    raise RuntimeError("usage_tracker _db_path() requires a SQLite DATABASE_URL")


def _ensure_sqlite_parent_dir() -> None:
    url = get_database_url()
    if url.startswith("sqlite:///"):
        parent = os.path.dirname(url[len("sqlite:///") :])
        if parent:
            os.makedirs(parent, exist_ok=True)


def init_db() -> None:
    """Create the runs schema if absent. Idempotent — safe on every startup.

    Cheap-idempotent: once created for the current database URL, repeated
    calls (this is called from `record`/`daily_total`/`history` on every
    invocation) are a true no-op and re-issue no DDL — required so DDL
    never runs concurrently with a request/background task reading these
    tables (that ordering deadlocks Postgres).
    """
    global _initialized, _initialized_for_url
    if _initialized and _initialized_for_url == get_database_url():
        return
    with _LOCK:
        # The URL this run initialises: read once, so a DATA_DIR change mid-way
        # cannot mark another database initialised.
        url = get_database_url()
        if _initialized and _initialized_for_url == url:
            return
        _ensure_sqlite_parent_dir()
        UsageRun.__table__.create(bind=engine, checkfirst=True)
        _initialized = True
        _initialized_for_url = url


# Per-provider pricing per 1M tokens loaded from config/llm_pricing.json
# at first use. Operator-editable; missing models / providers cost $0
# (tokens still recorded so daily TPD usage stays visible).
_PRICING_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "config", "llm_pricing.json")
_PRICING_OVERRIDE_ENV = "LLM_PRICING_FILE"
_PRICING_CACHE: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None
_PRICING_MTIME: float = 0.0


def _load_pricing() -> Dict[str, Dict[str, Dict[str, float]]]:
    """Reload the pricing table when the on-disk JSON has changed. The
    operator can edit prices without a redeploy: next `record()` call
    picks up the new file."""
    global _PRICING_CACHE, _PRICING_MTIME
    import json
    path = os.getenv(_PRICING_OVERRIDE_ENV) or _PRICING_PATH
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        logger.warning("LLM pricing file missing or unreadable at %s", path, exc_info=True)
        return {}
    if _PRICING_CACHE is not None and mtime == _PRICING_MTIME:
        return _PRICING_CACHE
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _PRICING_CACHE = data.get("providers", {}) if isinstance(data, dict) else {}
        _PRICING_MTIME = mtime
    except (OSError, ValueError):
        _PRICING_CACHE = _PRICING_CACHE or {}
    return _PRICING_CACHE


def _estimate_cost(provider: str, model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pricing = _load_pricing()
    rates = pricing.get(provider, {}).get(model)
    if not rates:
        return 0.0
    return round(
        (prompt_tokens / 1_000_000) * rates.get("prompt", 0.0)
        + (completion_tokens / 1_000_000) * rates.get("completion", 0.0),
        6,
    )


def record(
    user_id: Optional[str],
    agent_name: str,
    provider: str,
    model: str,
    usage: Optional[Dict[str, Any]],
) -> None:
    """Persist one LLM round-trip's usage. Safe to call with usage=None
    (provider didn't return a usage block — nothing recorded)."""
    if not usage:
        return
    pt = int(usage.get("prompt_tokens") or 0)
    ct = int(usage.get("completion_tokens") or 0)
    tt = int(usage.get("total_tokens") or (pt + ct))
    cost = _estimate_cost(provider, model, pt, ct)
    init_db()
    with _LOCK:
        with SessionLocal() as session:
            session.add(
                UsageRun(
                    id=str(uuid.uuid4()),
                    user_id=user_id or "",
                    agent_name=agent_name or "",
                    provider=provider,
                    model=model,
                    prompt_tokens=pt,
                    completion_tokens=ct,
                    total_tokens=tt,
                    estimated_cost_usd=cost,
                    created_at=datetime.now(timezone.utc).isoformat(),
                )
            )
            session.commit()


def daily_total(user_id: Optional[str], day: Optional[str] = None) -> Dict[str, float]:
    """Return ``{tokens, cost_usd}`` for ``user_id`` on ``day`` (UTC date,
    default = today). ``user_id=None`` aggregates across all users."""
    if day is None:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    init_db()
    day_expr = func.substr(UsageRun.created_at, 1, 10)
    stmt = select(
        func.coalesce(func.sum(UsageRun.total_tokens), 0),
        func.coalesce(func.sum(UsageRun.estimated_cost_usd), 0.0),
    ).where(day_expr == day)
    if user_id is not None:
        stmt = stmt.where(UsageRun.user_id == user_id)
    with _LOCK:
        with SessionLocal() as session:
            tokens, cost = session.execute(stmt).one()
    return {"tokens": int(tokens), "cost_usd": float(cost)}


def is_over_cap(user_id: Optional[str], cap_usd: float) -> bool:
    """True iff today's spend for ``user_id`` already meets or exceeds
    ``cap_usd``. Caller is expected to short-circuit upstream."""
    if cap_usd is None or cap_usd <= 0:
        return False
    return daily_total(user_id)["cost_usd"] >= cap_usd


def history(user_id: Optional[str], days: int = 7) -> List[Dict[str, Any]]:
    """Return per-day totals + breakdowns for the last ``days`` days."""
    init_db()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    day_expr = func.substr(UsageRun.created_at, 1, 10)
    stmt = select(
        day_expr.label("day"),
        UsageRun.agent_name,
        UsageRun.provider,
        UsageRun.model,
        func.sum(UsageRun.total_tokens).label("tokens"),
        func.sum(UsageRun.estimated_cost_usd).label("cost"),
    ).where(day_expr >= cutoff)
    if user_id is not None:
        stmt = stmt.where(UsageRun.user_id == user_id)
    stmt = stmt.group_by(
        day_expr, UsageRun.agent_name, UsageRun.provider, UsageRun.model
    ).order_by(day_expr.desc())
    with _LOCK:
        with SessionLocal() as session:
            rows = session.execute(stmt).all()
    return [
        {
            "day": r.day,
            "agent_name": r.agent_name,
            "provider": r.provider,
            "model": r.model,
            "tokens": int(r.tokens),
            "cost_usd": round(float(r.cost), 6),
        }
        for r in rows
    ]
