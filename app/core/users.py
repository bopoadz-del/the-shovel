"""User accounts store — Stream A (User Accounts & Multi-Tenancy).

SQLAlchemy-backed via app.core.db. Mirrors legacy store API.
A singleton 'system' user is auto-created; legacy API keys resolve to it.
"""
import hashlib
import hmac
import os
import secrets
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.core.db import SessionLocal, engine, get_database_url
from app.core.models import User

SYSTEM_USER_ID = "system"

_lock = threading.Lock()
_initialized = False
# Tracks WHICH database the schema was created in. `_initialized` alone is a
# global boolean, but the database is per-DATA_DIR / per-DATABASE_URL: after
# init runs against one database, `_ensure_db` short-circuits for every other
# one, so a later switch silently gets NO schema ("no such table: projects").
# Sibling stores (doc_index, hydration_store, rag.budget) already track the
# URL; these did not. Found via an order-dependent test failure, but the bug
# is real wherever the database can change after first use.
_initialized_for_url: str | None = None

_PBKDF2_ITERATIONS = 240_000


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_sqlite_parent_dir() -> None:
    url = get_database_url()
    if url.startswith("sqlite:///"):
        parent = os.path.dirname(url[len("sqlite:///") :])
        if parent:
            os.makedirs(parent, exist_ok=True)


def _as_dict(user: User) -> Dict[str, Any]:
    return {
        "id": user.id,
        "email": user.email,
        "password_hash": user.password_hash,
        "salt": user.salt,
        "display_name": user.display_name,
        "role": user.role,
        "created_at": user.created_at,
        "email_verified": bool(user.email_verified),
        "driver_mode": bool(getattr(user, "driver_mode", False)),
    }


def init_db() -> None:
    """Create the users schema if absent; auto-create the system user.

    Idempotent AND cheap: once the schema has been created for the current
    database URL, repeated calls (e.g. app startup firing on every
    TestClient instantiation in the test suite) are a true no-op — no DDL
    is re-issued. DDL running concurrently with a request/background task
    reading these tables is what deadlocks Postgres (AccessExclusiveLock
    vs. AccessShareLock taken in opposite orders on two connections), so
    this guard is checked before touching the engine at all — not only by
    callers that go through the private `_ensure_db` wrapper.
    """
    global _initialized, _initialized_for_url
    if _initialized and _initialized_for_url == get_database_url():
        return
    with _lock:
        # The URL this run initialises: read once, so a DATA_DIR change mid-way
        # cannot mark another database initialised.
        url = get_database_url()
        if _initialized and _initialized_for_url == url:
            return
        _ensure_sqlite_parent_dir()
        User.__table__.create(bind=engine, checkfirst=True)
        _add_missing_sqlite_columns(url)
        ensure_system_user()
        _initialized = True
        _initialized_for_url = url


def _add_missing_sqlite_columns(url: str) -> None:
    """SQLite dev/test databases made before a column existed (Postgres is
    migration-managed)."""
    if not url.startswith("sqlite"):
        return
    from sqlalchemy import text as sqla_text

    with engine.connect() as conn:
        cols = {row[1] for row in conn.execute(sqla_text("PRAGMA table_info(users)"))}
        if "driver_mode" not in cols:  # 0029
            conn.execute(sqla_text("ALTER TABLE users ADD COLUMN driver_mode BOOLEAN NOT NULL DEFAULT 0"))
            conn.commit()


def set_driver_mode(email: str, on: bool) -> Optional[Dict[str, Any]]:
    """Admin switch: this user's turns take driver mode (honoured when
    DRIVER_MODE=request). Returns the public user, or None if unknown."""
    _ensure_db()
    with _lock:
        session = SessionLocal()
        try:
            user = session.query(User).filter(User.email == (email or "").strip().lower()).one_or_none()
            if user is None:
                return None
            user.driver_mode = bool(on)
            session.commit()
            return _public(_as_dict(user))
        finally:
            session.close()


def driver_mode_users() -> List[Dict[str, Any]]:
    _ensure_db()
    with SessionLocal() as session:
        return [_public(_as_dict(u)) for u in session.query(User).filter(User.driver_mode.is_(True)).all()]


def driver_mode_for(user_id: Optional[str]) -> bool:
    if not user_id:
        return False
    row = get_user_by_id(user_id)
    return bool(row and row.get("driver_mode"))


def ensure_system_user() -> None:
    """Insert the system user row if it is missing. No DDL — safe to call
    any time the schema is known to already exist (e.g. re-seeding after a
    test-isolation TRUNCATE), including mid-request/background-task, since
    it never touches `Table.create`/`checkfirst` and takes no table lock
    beyond a normal row insert.

    Check-then-insert is not enough under Postgres: isolation TRUNCATE plus
    a live TestClient lifespan (knowledge-seed ``ensure_user_exists``) can
    both see a missing row and INSERT ``id='system'``. A UniqueViolation
    on ``users_pkey`` means the other caller won — treat that as success.
    """
    with SessionLocal() as session:
        if session.get(User, SYSTEM_USER_ID) is not None:
            return
        session.add(
            User(
                id=SYSTEM_USER_ID,
                email="system@local",
                password_hash=None,
                salt=None,
                display_name="System",
                role="admin",
                created_at=_now(),
            )
        )
        try:
            session.commit()
        except IntegrityError:
            session.rollback()


def _ensure_db() -> None:
    if not _initialized or _initialized_for_url != get_database_url():
        init_db()


def ensure_user_exists(
    user_id: str,
    *,
    email: Optional[str] = None,
    display_name: Optional[str] = None,
    role: str = "user",
) -> None:
    """Idempotently ensure a user row exists (satisfies projects.user_id FK)."""
    _ensure_db()
    with SessionLocal() as session:
        if session.get(User, user_id) is not None:
            return
        session.add(
            User(
                id=user_id,
                email=(email or f"{user_id}@local").lower(),
                password_hash=None,
                salt=None,
                display_name=display_name or user_id,
                role=role,
                created_at=_now(),
            )
        )
        session.commit()


def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    _ensure_db()
    with SessionLocal() as session:
        user = session.get(User, user_id)
    return _as_dict(user) if user else None


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    _ensure_db()
    with SessionLocal() as session:
        user = session.scalars(
            select(User).where(User.email == (email or "").lower())
        ).one_or_none()
    return _as_dict(user) if user else None


# ── password hashing (PBKDF2-HMAC-SHA256, stdlib only) ──────────────────────

def hash_password(password: str, salt: Optional[str] = None) -> Dict[str, str]:
    """Return {'salt', 'hash'} using PBKDF2-HMAC-SHA256 (stdlib, no native deps)."""
    if salt is None:
        salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"),
        bytes.fromhex(salt), _PBKDF2_ITERATIONS,
    )
    return {"salt": salt, "hash": dk.hex()}


def verify_password(password: str, password_hash: str, salt: str) -> bool:
    if not password_hash or not salt:
        return False
    candidate = hash_password(password, salt)["hash"]
    return hmac.compare_digest(candidate, password_hash)


def _public(row: Dict[str, Any]) -> Dict[str, Any]:
    """Strip secret columns from a user row before returning to callers."""
    return {k: v for k, v in row.items() if k not in ("password_hash", "salt")}


def create_user(
    email: str, password: str, display_name: Optional[str] = None,
    role: str = "user", email_verified: bool = False,
) -> Dict[str, Any]:
    """Create a user. ``email_verified`` defaults FALSE.

    Callers that provision a TRUSTED account out of band -- the env-driven
    bootstrap admin -- pass True explicitly. A fresh install runs the 0015
    backfill against an empty table, so that admin would otherwise be created
    unverified with no way to receive a link.
    """
    _ensure_db()
    email = (email or "").strip().lower()
    if not email:
        raise ValueError("email is required")
    if not password:
        raise ValueError("password is required")
    if get_user_by_email(email) is not None:
        raise ValueError(f"email '{email}' is already registered")
    uid = str(uuid.uuid4())[:8]
    creds = hash_password(password)
    with _lock:
        session = SessionLocal()
        try:
            session.add(
                User(
                    id=uid,
                    email=email,
                    password_hash=creds["hash"],
                    salt=creds["salt"],
                    display_name=display_name or email,
                    role=role,
                    created_at=_now(),
                    email_verified=email_verified,
                )
            )
            session.commit()
        except IntegrityError:
            session.rollback()
            raise ValueError(f"email '{email}' is already registered")
        finally:
            session.close()
    return _public(get_user_by_id(uid))


def password_fingerprint(user: Dict[str, Any]) -> str:
    """A short fingerprint of the user's current password hash. A reset link
    carries it, so the link stops working once the password changes -- it is
    single-use without a token table."""
    raw = f"{user.get('salt') or ''}:{user.get('password_hash') or ''}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def set_password(user_id: str, new_password: str) -> bool:
    """Replace a user's password (new salt). True if the user exists."""
    _ensure_db()
    creds = hash_password(new_password)
    with _lock:
        session = SessionLocal()
        try:
            user = session.get(User, user_id)
            if user is None:
                return False
            user.password_hash = creds["hash"]
            user.salt = creds["salt"]
            session.commit()
            return True
        finally:
            session.close()


def mark_email_verified(user_id: str) -> bool:
    """Flip a user's address to verified. True if a row changed.

    Idempotent by design: a verification link that is followed twice (mail
    clients prefetch them) must not error, so re-verifying an already-verified
    user is a no-op that still reports success to the caller.
    """
    _ensure_db()
    with _lock:
        session = SessionLocal()
        try:
            user = session.get(User, user_id)
            if user is None:
                return False
            user.email_verified = True
            session.commit()
            return True
        finally:
            session.close()
