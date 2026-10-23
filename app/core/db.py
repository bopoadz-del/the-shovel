"""SQLAlchemy database engine and session helpers.

Phase 1 foundation: single DATABASE_URL for the unified The Fork schema.

Environment:
    DATABASE_URL  PostgreSQL (or other SQLAlchemy URL). When unset, falls back
                  to sqlite:///{DATA_DIR}/the_fork.db for local dev.
    DATA_DIR      Directory for the SQLite fallback file (default ./data).
"""

from __future__ import annotations

import os
from collections.abc import Generator
from functools import lru_cache
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker


def _data_dir() -> str:
    return os.getenv("DATA_DIR", "./data")


def to_psycopg_url(url: str) -> str:
    """Rewrite a bare Postgres URL to the installed psycopg v3 driver scheme.

    SQLAlchemy maps ``postgresql://`` / ``postgres://`` to psycopg2, which this
    project does NOT install (psycopg v3 only). Any code that builds an engine
    from a raw ``DATABASE_URL`` must route through here, or it crashes with
    ``ModuleNotFoundError: No module named 'psycopg2'``. Idempotent; non-Postgres
    URLs (e.g. sqlite) pass through unchanged.
    """
    if url.startswith("postgresql+psycopg://") or url.startswith("postgresql+psycopg2://"):
        return url
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url[len("postgresql://"):]
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url[len("postgres://"):]
    return url


def get_database_url() -> str:
    """Resolve DATABASE_URL from env, honoring DATA_DIR at call time.

    Render/Postgres URLs are typically `postgresql://...`. SQLAlchemy maps
    that dialect to psycopg2, but this project installs psycopg v3 only
    (`psycopg[binary]`), so the scheme is normalized to `postgresql+psycopg://`
    via ``to_psycopg_url``.
    """
    explicit = os.getenv("DATABASE_URL")
    if explicit:
        return to_psycopg_url(explicit)
    db_path = os.path.join(_data_dir(), "the_fork.db")
    return f"sqlite:///{db_path}"


# Evaluated at import for Alembic; runtime code should call get_database_url().
DATABASE_URL: str = get_database_url()


def _engine_kwargs(url: str) -> dict[str, Any]:
    if url.startswith("postgresql"):
        # Default pool_size=10 was fine for one web process, but five
        # identical Pro ingest workers × 10 = 50 connections and knocks
        # Render Postgres into recovery ("not yet accepting connections").
        # Ingest only needs 1–2 live connections (P1B_PARALLELISM≤2).
        # Override with DB_POOL_SIZE / DB_MAX_OVERFLOW on workers.
        pool_size = max(1, int(os.getenv("DB_POOL_SIZE", "10")))
        max_overflow = max(0, int(os.getenv("DB_MAX_OVERFLOW", "10")))
        # A connection older than the recycle window is replaced at checkout
        # (a new connect + TLS, ~0.6 s to the database). The pool hands out
        # the most recently used connection first (LIFO), so the one the
        # keep-warm timer (health_probes.start_db_keep_warm) touches is the
        # one the next request gets, not the oldest idle one.
        return {
            "pool_size": pool_size,
            "max_overflow": max_overflow,
            "pool_pre_ping": True,
            "pool_recycle": max(30, int(os.getenv("DB_POOL_RECYCLE_S", "300"))),
            "pool_use_lifo": True,
        }
    if url.startswith("sqlite"):
        return {"connect_args": {"timeout": 30.0}}
    return {}


@lru_cache(maxsize=8)
def _engine_for_url(url: str) -> Engine:
    return create_engine(url, **_engine_kwargs(url))


@event.listens_for(Engine, "connect")
def _sqlite_enable_foreign_keys(dbapi_conn: Any, _connection_record: Any) -> None:
    if dbapi_conn.__class__.__module__ == "sqlite3":
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def get_engine() -> Engine:
    """Return a cached engine for the current DATABASE_URL / DATA_DIR."""
    return _engine_for_url(get_database_url())


class _LazyEngine:
    """Proxy so ``from app.core.db import engine`` tracks DATA_DIR changes."""

    def __getattr__(self, name: str) -> Any:
        return getattr(get_engine(), name)

    def __repr__(self) -> str:
        return repr(get_engine())


engine = _LazyEngine()  # type: ignore[assignment]


def _session_factory_for_url(url: str) -> sessionmaker[Session]:
    return _session_factory_for_engine(_engine_for_url(url))


# Keyed by the engine, not the URL: once the engine cache evicts a URL and
# makes a new engine for it, a URL-keyed factory would keep the old pool open.
@lru_cache(maxsize=8)
def _session_factory_for_engine(bound: Engine) -> sessionmaker[Session]:
    return sessionmaker(autocommit=False, autoflush=False, bind=bound)


def SessionLocal() -> Session:
    """Open a new SQLAlchemy session bound to the current database URL."""
    return _session_factory_for_url(get_database_url())()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency: yield a request-scoped SQLAlchemy session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
