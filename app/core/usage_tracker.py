"""LLM usage + daily cost guard.

Records every LLM round-trip (tokens and an estimated cost) and answers
whether today's account spend has met ``USAGE_DAILY_CAP_USD``.

The pre-call check lives with the callers that already invoke
``is_over_cap``. This module prices the call and applies the account-wide
ceiling. It does not itself open a provider connection.

An explicit ``0`` rate is free. A provider/model pair that is not in the
pricing file is unknown: it is billed at the configured fallback, never at
``0``, and the provider and model names are logged once. A missing fallback
makes ``is_over_cap`` return True so a capped caller refuses the call.

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


# USD per 1M tokens, loaded from config/llm_pricing.json at first use.
# An explicit 0 is free. An absent provider/model uses ``fallback`` and is
# logged once. It is never silently priced at 0.
_PRICING_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "config", "llm_pricing.json")
_PRICING_OVERRIDE_ENV = "LLM_PRICING_FILE"
_DOC_CACHE: Optional[Dict[str, Any]] = None
_DOC_MTIME: float = 0.0
_PRICE_KEYS = ("prompt", "completion", "cache_hit")
_CONDITION_KEYS = {"min_prompt_tokens", "utc_start", "utc_end", "utc_days"}
_UNKNOWN_LOGGED: set[tuple[str, str]] = set()
_ALERTED_DAYS: set[str] = set()
_ALERT_FRACTION = 0.8


def _pricing_document() -> Dict[str, Any]:
    """Reload the pricing document when the on-disk JSON changes."""
    global _DOC_CACHE, _DOC_MTIME
    import json
    path = os.getenv(_PRICING_OVERRIDE_ENV) or _PRICING_PATH
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        logger.warning("LLM pricing file missing or unreadable at %s", path)
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def _load_pricing() -> Dict[str, Dict[str, Dict[str, float]]]:
    """Provider map from the pricing document. Kept for existing readers."""
    providers = _pricing_document().get("providers", {})
    return providers if isinstance(providers, dict) else {}


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        negative = text.startswith("-")
        body = text.removeprefix("-").removeprefix("+")
        if body.count(".") > 1:
            return None
        digits = body.replace(".", "", 1)
        if digits.isdigit():
            return -float(body) if negative else float(body)
    return None


def _clock_hhmm(when: datetime) -> int:
    return when.hour * 100 + when.minute


def _window_matches(when: datetime, start: Any, end: Any) -> bool:
    """Inclusive start, exclusive end, as an HHMM clock. A later end that is
    not after the start wraps past midnight. An end of 0 means end of day."""
    start_n = _number(start)
    end_n = _number(end)
    if start_n is None or end_n is None:
        return False
    start_i = int(start_n)
    end_i = int(end_n)
    tick = _clock_hhmm(when)
    if end_i == 0:
        return tick >= start_i
    if end_i > start_i:
        return start_i <= tick < end_i
    return tick >= start_i or tick < end_i


def _override_applies(entry: Any, when: datetime, prompt_tokens: int) -> bool:
    """True when every recognised condition matches. An unknown condition
    field skips the entry so a newer tariff cannot be applied by guess."""
    if not isinstance(entry, dict):
        return False
    for key in entry:
        if key in _PRICE_KEYS or key == "cache_miss":
            continue
        if key not in _CONDITION_KEYS:
            return False
    if "min_prompt_tokens" in entry:
        threshold = _number(entry.get("min_prompt_tokens"))
        if threshold is None or prompt_tokens <= threshold:
            return False
    days = entry.get("utc_days")
    if days is not None:
        if not isinstance(days, list) or when.strftime("%A").lower() not in days:
            return False
    if "utc_start" in entry or "utc_end" in entry:
        if not _window_matches(when, entry.get("utc_start", 0), entry.get("utc_end", 0)):
            return False
    return True


def _resolved_rates(entry: Any, when: datetime, prompt_tokens: int) -> Optional[Dict[str, float]]:
    if not isinstance(entry, dict):
        return None
    prompt = _number(entry.get("prompt"))
    completion = _number(entry.get("completion"))
    if prompt is None or completion is None:
        return None
    cache_hit = _number(entry.get("cache_hit"))
    rates = {
        "prompt": prompt,
        "completion": completion,
        "cache_hit": prompt if cache_hit is None else cache_hit,
    }
    overrides = entry.get("overrides") or []
    if isinstance(overrides, list):
        for override in overrides:
            if not _override_applies(override, when, prompt_tokens):
                continue
            for key in _PRICE_KEYS:
                parsed = _number(override.get(key)) if isinstance(override, dict) else None
                if parsed is not None:
                    rates[key] = parsed
    rates["free"] = (
        rates["prompt"] == 0.0
        and rates["completion"] == 0.0
        and rates["cache_hit"] == 0.0
    )
    return rates


def _log_unknown_model(provider: str, model: str) -> None:
    key = (provider or "", model or "")
    with _LOCK:
        if key in _UNKNOWN_LOGGED:
            return
        _UNKNOWN_LOGGED.add(key)
    logger.warning(
        "unpriced model; using configured fallback rate provider=%s model=%s",
        key[0],
        key[1],
    )


def _split_input(usage: Optional[Dict[str, Any]], prompt_tokens: int) -> tuple[int, int]:
    """Cache-hit tokens and cache-miss tokens. Without a split, the whole
    prompt is a miss (the higher of the two input rates when they differ)."""
    prompt = max(int(prompt_tokens or 0), 0)
    if not isinstance(usage, dict):
        return 0, prompt
    hit = usage.get("prompt_cache_hit_tokens", usage.get("cache_hit_tokens"))
    miss = usage.get("prompt_cache_miss_tokens", usage.get("cache_miss_tokens"))
    hit_n = _number(hit)
    miss_n = _number(miss)
    if hit_n is None and miss_n is None:
        return 0, prompt
    hit_i = max(int(hit_n or 0), 0)
    if miss_n is None:
        return hit_i, max(prompt - hit_i, 0)
    return hit_i, max(int(miss_n), 0)


def price_call(
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    usage: Optional[Dict[str, Any]] = None,
    when: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Price one call.

    ``kind`` is ``free`` (explicit zero rates), ``priced`` (listed non-zero
    rates), ``fallback`` (unlisted model, configured conservative rates),
    or ``unpriced`` (no listing and no configured fallback). ``unpriced``
    has ``cost_usd`` None so it is not stored as zero.
    """
    if prompt_tokens is None and completion_tokens is None and usage is None:
        return {"kind": "unpriced", "cost_usd": None}
    moment = when or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    prompt_tokens = max(int(prompt_tokens or 0), 0)
    completion_tokens = max(int(completion_tokens or 0), 0)
    document = _pricing_document()
    providers = document.get("providers") if isinstance(document, dict) else None
    entry = None
    if isinstance(providers, dict) and isinstance(providers.get(provider), dict):
        entry = providers[provider].get(model)
    rates = _resolved_rates(entry, moment, prompt_tokens) if entry is not None else None
    kind = "priced"
    if rates is None:
        fallback = document.get("fallback") if isinstance(document, dict) else None
        rates = _resolved_rates(fallback, moment, prompt_tokens)
        kind = "fallback"
        _log_unknown_model(provider, model)
    if rates is None:
        return {"kind": "unpriced", "cost_usd": None}
    if rates["free"] and kind != "fallback":
        kind = "free"
    hit, miss = _split_input(usage, prompt_tokens)
    cost = (
        (hit / 1_000_000) * rates["cache_hit"]
        + (miss / 1_000_000) * rates["prompt"]
        + (completion_tokens / 1_000_000) * rates["completion"]
    )
    return {"kind": kind, "cost_usd": round(cost, 6)}


def _estimate_cost(
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    usage: Optional[Dict[str, Any]] = None,
    when: Optional[datetime] = None,
) -> Optional[float]:
    return price_call(
        provider, model, prompt_tokens, completion_tokens, usage=usage, when=when
    )["cost_usd"]


def _daily_cap() -> float:
    raw = os.getenv("USAGE_DAILY_CAP_USD")
    if raw is None or raw.strip() == "":
        return 0.0
    parsed = _number(raw)
    if parsed is None or parsed <= 0:
        return 0.0
    return parsed


def threshold_state(cost_usd: float, cap_usd: float) -> str:
    """``disabled``, ``ok``, ``alert`` (80% of the cap), or ``cap``."""
    if cap_usd is None or cap_usd <= 0:
        return "disabled"
    if cost_usd >= cap_usd:
        return "cap"
    if cost_usd >= cap_usd * _ALERT_FRACTION:
        return "alert"
    return "ok"


def note_daily_threshold(
    cost_usd: float,
    cap_usd: float,
    *,
    day: Optional[str] = None,
) -> str:
    """Log the 80% line once per UTC day. The cap itself is not a log line
    here; callers refuse the call from ``is_over_cap``."""
    state = threshold_state(cost_usd, cap_usd)
    if state != "alert":
        return state
    stamp = day or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with _LOCK:
        fresh = stamp not in _ALERTED_DAYS
        if fresh:
            _ALERTED_DAYS.add(stamp)
    if fresh:
        logger.warning(
            "USAGE_DAILY_CAP_USD 80%% line reached for utc day %s",
            stamp,
        )
    return state


def pricing_fail_closed() -> bool:
    """True when an unknown model would be billed at the configured fallback."""
    document = _pricing_document()
    fallback = document.get("fallback") if isinstance(document, dict) else None
    return _resolved_rates(fallback, datetime.now(timezone.utc), 0) is not None


def record(
    user_id: Optional[str],
    agent_name: str,
    provider: str,
    model: str,
    usage: Optional[Dict[str, Any]],
) -> None:
    """Persist one LLM round-trip's usage. A round-trip that returned no usage
    block is still recorded as a call with tokens unknown (0) — the meter
    counts the CALL; token/cost totals are unaffected by a zero row. (Before
    ORDERS_1920Z this no-oped on a falsy usage, so streamed/usage-less calls
    went uncounted.)"""
    # A round-trip with no usage block still counts as a CALL, but its tokens
    # are UNKNOWN, not zero — store NULL so an admin row can tell "we don't know"
    # apart from "a genuinely empty response" (ORDERS_2050Z). SUM() skips NULLs,
    # so token/cost totals are unaffected; COUNT() still counts the call.
    if usage:
        pt: int | None = int(usage.get("prompt_tokens") or 0)
        ct: int | None = int(usage.get("completion_tokens") or 0)
        tt: int | None = int(usage.get("total_tokens") or (pt + ct))
        cost = _estimate_cost(provider, model, pt, ct, usage=usage)
    else:
        pt = ct = tt = None
        cost = None
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
    cap = _daily_cap()
    if cap > 0:
        note_daily_threshold(daily_total(None)["cost_usd"], cap)


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
    """True when today's account spend, or this user's spend, meets ``cap_usd``.

    The ceiling is account-wide: one user's rows are part of
    ``daily_total(None)``. The per-user total is also checked so a caller
    that passes a user still refuses when that user alone has reached the
    cap. A non-positive cap disables the check. When the pricing file has
    no usable fallback, the result is True so a capped caller fails closed
    instead of treating unknown models as free.

    A database error still propagates. Callers that catch it and continue
    are outside this module.
    """
    if cap_usd is None or cap_usd <= 0:
        return False
    if not pricing_fail_closed():
        logger.warning("usage cap fail-closed: pricing fallback is not configured")
        return True
    account = daily_total(None)["cost_usd"]
    note_daily_threshold(account, float(cap_usd))
    if account >= cap_usd:
        return True
    if user_id is not None and daily_total(user_id)["cost_usd"] >= cap_usd:
        return True
    return False


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
        func.coalesce(func.sum(UsageRun.total_tokens), 0).label("tokens"),
        func.coalesce(func.sum(UsageRun.estimated_cost_usd), 0.0).label("cost"),
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


def totals(since_days: Optional[int] = None) -> Dict[str, Any]:
    """Global model-call meter: ``{calls, tokens, cost_usd}`` across ALL users.

    ``calls`` is the number of recorded LLM round-trips, ``tokens`` the summed
    ``total_tokens``, ``cost_usd`` the summed estimated cost. ``since_days=None``
    is all-time; an integer keeps only runs on or after that UTC-date cutoff.

    This is the name-free aggregate the public ``/health`` and the public
    Prometheus ``/metrics`` expose — it carries NO provider or model name (those
    stay on the admin ``/v1/metrics``; see ``totals_by_provider_model``). The
    meter must never break a health or metrics surface, so any error returns
    zeros rather than raising.
    """
    try:
        init_db()
        stmt = select(
            func.count(UsageRun.id),
            func.coalesce(func.sum(UsageRun.total_tokens), 0),
            func.coalesce(func.sum(UsageRun.estimated_cost_usd), 0.0),
        )
        if since_days is not None:
            cutoff = (
                datetime.now(timezone.utc) - timedelta(days=since_days)
            ).strftime("%Y-%m-%d")
            stmt = stmt.where(func.substr(UsageRun.created_at, 1, 10) >= cutoff)
        with _LOCK:
            with SessionLocal() as session:
                calls, tokens, cost = session.execute(stmt).one()
        return {
            "calls": int(calls or 0),
            "tokens": int(tokens or 0),
            "cost_usd": round(float(cost or 0.0), 6),
        }
    except Exception as exc:  # noqa: BLE001 — the meter never breaks a surface
        logger.debug("usage_tracker.totals swallowed %r", exc)
        return {"calls": 0, "tokens": 0, "cost_usd": 0.0}


def totals_by_provider_model(since_days: Optional[int] = None) -> List[Dict[str, Any]]:
    """Per provider+model meter rows: ``{provider, model, calls, tokens,
    cost_usd}``, highest call count first.

    This row set carries provider and model NAMES, so it is for ADMIN surfaces
    only (``/v1/metrics``) — never the public ``/health`` or public
    ``/metrics``. ``since_days`` behaves as in :func:`totals`. Never raises: an
    error returns an empty list.
    """
    try:
        init_db()
        calls_col = func.count(UsageRun.id)
        stmt = select(
            UsageRun.provider,
            UsageRun.model,
            calls_col.label("calls"),
            func.coalesce(func.sum(UsageRun.total_tokens), 0).label("tokens"),
            func.coalesce(func.sum(UsageRun.estimated_cost_usd), 0.0).label("cost"),
        )
        if since_days is not None:
            cutoff = (
                datetime.now(timezone.utc) - timedelta(days=since_days)
            ).strftime("%Y-%m-%d")
            stmt = stmt.where(func.substr(UsageRun.created_at, 1, 10) >= cutoff)
        stmt = stmt.group_by(UsageRun.provider, UsageRun.model).order_by(
            calls_col.desc()
        )
        with _LOCK:
            with SessionLocal() as session:
                rows = session.execute(stmt).all()
        return [
            {
                "provider": r.provider,
                "model": r.model,
                "calls": int(r.calls),
                "tokens": int(r.tokens),
                "cost_usd": round(float(r.cost), 6),
            }
            for r in rows
        ]
    except Exception as exc:  # noqa: BLE001 — the meter never breaks a surface
        logger.debug("usage_tracker.totals_by_provider_model swallowed %r", exc)
        return []
