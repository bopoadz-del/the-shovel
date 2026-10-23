"""Evaluated capability probes for health/readiness endpoints (audit §8.5/§8.6).

Health endpoints previously reported a HARDCODED ``status:"healthy"`` and a
provider leaderboard seeded at 100% reliability — so they could claim the DB,
embedder, or an LLM provider were up when the container could not actually
reach them. These probes report what is ACTUALLY observed.

Every probe is fast, bounded, and NEVER raises out to the caller — a probe that
can hang or crash the health endpoint is worse than the lie it replaces.
"""
from __future__ import annotations

import logging
import math
import os
import threading
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_corpus_cache_lock = threading.Lock()
_corpus_cache_value: Optional[Dict[str, Any]] = None
_corpus_cache_at: Optional[float] = None
_corpus_refreshing = False
_corpus_refresh_thread: Optional[threading.Thread] = None
_corpus_refresh_error: Optional[str] = None


_keep_warm_thread: Optional[threading.Thread] = None
_keep_warm_stop = threading.Event()


def _keep_warm_interval_s() -> float:
    """DB_KEEP_WARM_S: seconds between keep-warm round-trips; 0 turns it off.
    Default 240, inside the 300 s pool recycle window."""
    raw = (os.getenv("DB_KEEP_WARM_S") or "").strip()
    try:
        value = float(raw) if raw else 240.0
    except ValueError:
        return 240.0
    return value if math.isfinite(value) and value >= 0 else 240.0


def _keep_warm_once() -> None:
    """One keep-warm pass: touch every structure the first request after idle
    would otherwise rebuild. The DB pool (a recycled connection reconnects),
    and the corpus chunk-count cache (TTL 300 s, so on an idle box it expires
    between requests and the first ask pays the count query). Each warmer is
    best-effort and never raises -- a warm-up must not crash the thread."""
    result = probe_database()
    if not result.get("ok"):
        logger.warning("keep-warm: DB round-trip failed: %s", result.get("error"))
    try:
        # Refreshes the corpus cache when it is within the TTL; the tick runs
        # inside the TTL window, so the cache never goes cold on an idle box.
        probe_corpus_chunks()
    except Exception:  # noqa: BLE001 -- probe_corpus_chunks is non-raising, but the loop must survive anything
        logger.warning("keep-warm: corpus cache refresh failed", exc_info=True)


def _keep_warm_loop(interval_s: float) -> None:
    while not _keep_warm_stop.wait(interval_s):
        _keep_warm_once()


def start_db_keep_warm(interval_s: Optional[float] = None) -> Optional[threading.Thread]:
    """Keep one pooled database connection fresh: a SELECT 1 every interval,
    in a daemon thread off the event loop. After an idle spell the first
    request (and /health) otherwise paid a new connect + TLS (~0.6 s) because
    every pooled connection had passed the recycle window. One thread per
    process; returns it, or None when turned off."""
    global _keep_warm_thread
    interval = _keep_warm_interval_s() if interval_s is None else interval_s
    if interval <= 0:
        return None
    if _keep_warm_thread is not None and _keep_warm_thread.is_alive():
        return _keep_warm_thread
    _keep_warm_stop.clear()
    _keep_warm_thread = threading.Thread(target=_keep_warm_loop, args=(interval,),
                                         name="db-keep-warm", daemon=True)
    _keep_warm_thread.start()
    return _keep_warm_thread


def stop_db_keep_warm() -> None:
    _keep_warm_stop.set()


def probe_database(timeout_s: float = 2.0) -> Dict[str, Any]:
    """Real DB round-trip: ``SELECT 1`` via the app's session factory.

    Returns ``{"ok": bool, "latency_ms": int|None, "error": str|None}``.
    Bounded and non-raising: any failure (unreachable, auth, timeout) is
    reported as ``ok=False`` with a short reason, never an exception.
    """
    started = time.monotonic()
    try:
        from sqlalchemy import text
        from app.core.db import SessionLocal

        with SessionLocal() as session:
            # Best-effort per-statement timeout on Postgres; harmless no-op
            # (rolled back) elsewhere.
            try:
                session.execute(text("SET LOCAL statement_timeout = :ms"),
                                {"ms": int(timeout_s * 1000)})
            except Exception:  # noqa: BLE001 — SQLite/other: skip, still probe
                session.rollback()
            session.execute(text("SELECT 1"))
        latency_ms = int((time.monotonic() - started) * 1000)
        return {"ok": True, "latency_ms": latency_ms, "error": None}
    except Exception as exc:  # noqa: BLE001 — probe must never raise
        return {
            "ok": False,
            "latency_ms": int((time.monotonic() - started) * 1000),
            "error": f"{type(exc).__name__}: {exc}"[:200],
        }


def reset_corpus_chunk_cache() -> None:
    """Drop the cached corpus count. Tests use this so a prior probe cannot
    answer for a different database. Does not join a refresh already running.
    """
    global _corpus_cache_value, _corpus_cache_at, _corpus_refreshing, _corpus_refresh_error
    with _corpus_cache_lock:
        _corpus_cache_value = None
        _corpus_cache_at = None
        _corpus_refreshing = False
        _corpus_refresh_error = None


_DEFAULT_CORPUS_TTL_S = 300.0
_DEFAULT_CORPUS_NEGATIVE_TTL_S = 30.0
# Longer than a day is not used. A non-finite value, or one that is not
# above zero, also falls back to the default so the age comparison can expire.
_MAX_CORPUS_CACHE_TTL_S = 86400.0


def _env_float(name: str, default: float) -> float:
    """Read one cache TTL from the environment.

    Missing, blank, non-numeric, non-finite, non-positive, or above the cap
    keep ``default``. Zero, negative, nan, and inf must not reach the age
    checks: those either refresh on every probe or never expire a failure.
    """
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if (
        not math.isfinite(value)
        or value <= 0
        or value > _MAX_CORPUS_CACHE_TTL_S
    ):
        return default
    return value


def _corpus_cache_config() -> Dict[str, float]:
    """Tunables for the corpus-count probe.

    ``ttl_s`` is how long a successful count is served with no database work.
    ``negative_ttl_s`` is how long a failed refresh is remembered so a down
    database is not queried again on every /health hit.
    Both are read from the environment on each call. Unset, blank, or
    unusable values stay at ``CORPUS_CHUNK_CACHE_TTL_S`` 300 and
    ``CORPUS_CHUNK_CACHE_NEGATIVE_TTL_S`` 30. A usable value is finite and
    greater than zero, and not above one day.
    """
    return {
        "ttl_s": _env_float("CORPUS_CHUNK_CACHE_TTL_S", _DEFAULT_CORPUS_TTL_S),
        "negative_ttl_s": _env_float(
            "CORPUS_CHUNK_CACHE_NEGATIVE_TTL_S", _DEFAULT_CORPUS_NEGATIVE_TTL_S
        ),
    }


def _corpus_ttl_s() -> float:
    return float(_corpus_cache_config()["ttl_s"])


def _corpus_negative_ttl_s() -> float:
    return float(_corpus_cache_config()["negative_ttl_s"])


def _reltuples(session: Any, table: str) -> Optional[float]:
    """Planner estimate for ``table``, or None when Postgres has no row.

    ``reltuples`` is -1 until the table has been analyzed. A query failure
    (SQLite, missing catalog) is the same as a missing estimate.
    """
    try:
        from sqlalchemy import text

        value = session.execute(
            text(
                "SELECT c.reltuples FROM pg_class c "
                "JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE c.relname = :name AND c.relkind = 'r' "
                "AND n.nspname = current_schema()"
            ),
            {"name": table},
        ).scalar()
    except Exception:  # noqa: BLE001 — not Postgres, or the catalog read failed
        session.rollback()
        return None
    if value is None:
        return None
    return float(value)


def _refresh_corpus_count(timeout_s: float = 1.5) -> Dict[str, Any]:
    """Read the chunk-table size. Estimate first; exact COUNT only as fallback.

    The statement timeout is the same bound the old per-probe COUNT used.
    Never raises: a failure is an ``error`` field.
    """
    started = time.monotonic()
    table = None
    try:
        from sqlalchemy import text

        from app.core.db import SessionLocal
        from app.core.models import rag_chunk_table_name

        ns = os.getenv("RAG_VECTOR_NAMESPACE", "v2").strip()
        table = rag_chunk_table_name(ns)
        with SessionLocal() as session:
            try:
                session.execute(
                    text("SET LOCAL statement_timeout = :ms"),
                    {"ms": int(timeout_s * 1000)},
                )
            except Exception:  # noqa: BLE001 — SQLite: skip
                session.rollback()
            estimate = _reltuples(session, table)
            if estimate is None or estimate == -1:
                n = session.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                chunks = int(n or 0)
                source = "exact"
            else:
                chunks = int(round(estimate))
                source = "estimate"
        return {
            "chunks": chunks,
            "source": source,
            "table": table,
            "latency_ms": int((time.monotonic() - started) * 1000),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 — probe must never raise
        return {
            "chunks": None,
            "source": "exact",
            "table": table,
            "latency_ms": int((time.monotonic() - started) * 1000),
            "error": f"{type(exc).__name__}: {exc}"[:200],
        }


def _invoke_refresh(timeout_s: float) -> Dict[str, Any]:
    """Call the refresher. A raised error becomes a payload, never an exception."""
    started = time.monotonic()
    try:
        result = _refresh_corpus_count(timeout_s)
    except Exception as exc:  # noqa: BLE001 — a monkeypatched refresher may raise
        return {
            "chunks": None,
            "source": "exact",
            "table": None,
            "latency_ms": int((time.monotonic() - started) * 1000),
            "error": f"{type(exc).__name__}: {exc}"[:200],
        }
    if not isinstance(result, dict):
        return {
            "chunks": None,
            "source": "exact",
            "table": None,
            "latency_ms": int((time.monotonic() - started) * 1000),
            "error": "corpus refresh returned no payload",
        }
    return result


def _store_corpus_count(result: Dict[str, Any]) -> None:
    global _corpus_cache_value, _corpus_cache_at, _corpus_refresh_error
    _corpus_cache_value = dict(result)
    _corpus_cache_at = time.monotonic()
    _corpus_refresh_error = None


def _background_refresh(timeout_s: float) -> None:
    """One refresh off the request path. Failures stay recorded, never raised."""
    global _corpus_refreshing
    result = _invoke_refresh(timeout_s)
    with _corpus_cache_lock:
        _remember_corpus_result(result)
        _corpus_refreshing = False


def _schedule_background_refresh(timeout_s: float) -> None:
    """Start at most one refresh thread. Caller holds ``_corpus_cache_lock``."""
    global _corpus_refreshing, _corpus_refresh_thread
    if _corpus_refreshing:
        return
    _corpus_refreshing = True
    thread = threading.Thread(
        target=_background_refresh,
        args=(timeout_s,),
        name="corpus-chunk-count-refresh",
        daemon=True,
    )
    _corpus_refresh_thread = thread
    thread.start()


def _from_cache(payload: Dict[str, Any], age_s: float, *, stale: bool, latency_ms: int) -> Dict[str, Any]:
    out = dict(payload)
    out["age_s"] = round(float(age_s), 3)
    out["latency_ms"] = latency_ms
    if stale:
        out["stale"] = True
    else:
        out["source"] = "cached"
        out.pop("stale", None)
    return out


def _warming_payload(latency_ms: int) -> Dict[str, Any]:
    """A concurrent cold caller arrived while the one refresh is in flight."""
    return {
        "chunks": None,
        "source": "warming",
        "table": None,
        "latency_ms": latency_ms,
        "error": None,
        "warming": True,
    }


def _remember_corpus_result(result: Dict[str, Any]) -> None:
    """Store a success for the long TTL, or a failure for the negative TTL.

    Caller holds ``_corpus_cache_lock``. A failure does not replace a good count.
    """
    global _corpus_cache_value, _corpus_cache_at, _corpus_refresh_error
    ok = result.get("error") is None and result.get("chunks") is not None
    if ok:
        _store_corpus_count(result)
        return
    if _corpus_cache_value is not None and _corpus_cache_value.get("error") is None:
        _corpus_refresh_error = str(result.get("error") or "corpus refresh failed")[:200]
        logger.warning("corpus chunk count refresh failed: %s", _corpus_refresh_error)
        return
    _corpus_cache_value = dict(result)
    _corpus_cache_at = time.monotonic()
    _corpus_refresh_error = str(result.get("error") or "corpus refresh failed")[:200]
    logger.warning("corpus chunk count refresh failed: %s", _corpus_refresh_error)


def probe_corpus_chunks(timeout_s: float = 1.5) -> Dict[str, Any]:
    """Chunk-table size for /health — never documents.chunk_count.

    The first call reads Postgres ``pg_class.reltuples`` and falls back to
    ``COUNT(*)`` only when that estimate is missing or -1, under the same
    statement timeout as before. Later calls inside the TTL from the
    environment return the cached figure (``source="cached"``, ``age_s``)
    and do no database work. After that TTL the last figure is returned at
    once with ``stale: true`` while a single background thread refreshes it.
    Concurrent cold callers get one refresh; the rest receive a warming
    payload. A failed refresh is remembered for the negative TTL from the
    environment so a down
    database is not queried on every probe. A probe error is a field on the
    payload. This function does not raise.
    """
    global _corpus_refreshing
    started = time.monotonic()
    leader = False
    try:
        now = time.monotonic()
        with _corpus_cache_lock:
            cached = _corpus_cache_value
            cached_at = _corpus_cache_at
            latency_ms = int((time.monotonic() - started) * 1000)
            if cached is not None and cached_at is not None:
                age = now - cached_at
                failed = cached.get("error") is not None or cached.get("chunks") is None
                if failed and age < _corpus_negative_ttl_s():
                    return _from_cache(cached, age, stale=False, latency_ms=latency_ms)
                if not failed and age < _corpus_ttl_s():
                    return _from_cache(cached, age, stale=False, latency_ms=latency_ms)
                if not failed:
                    _schedule_background_refresh(timeout_s)
                    return _from_cache(cached, age, stale=True, latency_ms=latency_ms)
            if _corpus_refreshing:
                return _warming_payload(latency_ms)
            _corpus_refreshing = True
            leader = True
        try:
            result = _invoke_refresh(timeout_s)
            with _corpus_cache_lock:
                _remember_corpus_result(result)
            return result
        finally:
            with _corpus_cache_lock:
                _corpus_refreshing = False
    except Exception as exc:  # noqa: BLE001 — /health must never fail closed on this probe
        if leader:
            with _corpus_cache_lock:
                _corpus_refreshing = False
        return {
            "chunks": None,
            "source": "exact",
            "table": None,
            "latency_ms": int((time.monotonic() - started) * 1000),
            "error": f"{type(exc).__name__}: {exc}"[:200],
        }


def probe_embedder() -> Dict[str, Any]:
    """Report whether the RAG embedder is warm-loaded (NON-loading probe).

    ``loaded=False`` is not an error — the embedder lazy-loads on first use;
    it means retrieval will pay the cold-load cost on the next query. We do
    NOT trigger the load here (that would make /health slow on a cold box).
    """
    try:
        from app.core.rag.embeddings import embedder_is_loaded, loaded_embedder_identity

        loaded = embedder_is_loaded()
        return {"loaded": loaded, "identity": loaded_embedder_identity() if loaded else None}
    except Exception as exc:  # noqa: BLE001
        return {"loaded": False, "identity": None, "error": f"{type(exc).__name__}: {exc}"[:200]}


def probe_llm() -> Dict[str, Any]:
    """Whether the chat can reach a language model, and whether it can SURVIVE
    losing the primary one -- two booleans, nothing else.

    Live 2026-09-19: LLM_FALLBACK_PROVIDER named `kimi`, a provider the code had
    removed. `_cross_provider_fallback` resolved it back to the primary and
    returned None: no fallback at all, and nothing anywhere said so. An unset
    fallback key does the same. /health is public, so this names no provider,
    model, variable or key -- only whether each is usable.
    """
    try:
        import os

        from app.agents.runtime import _llm_config, _llm_fallback_config

        primary = _llm_config()
        key = primary.get("env_key")
        return {
            "primary_ready": bool(os.getenv(key)) if key else True,
            "fallback_ready": _llm_fallback_config(primary) is not None,
        }
    except Exception as exc:  # noqa: BLE001 - a probe never breaks /health
        return {"primary_ready": False, "fallback_ready": False,
                "error": type(exc).__name__}
