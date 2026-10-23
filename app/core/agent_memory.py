"""Agent memory — persistent conversations, messages, and durable agent facts.

Phase C4 — Stream C: persistent agent memory.

SQLAlchemy-backed via app.core.db — unified The Fork schema.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy import delete, or_, select, update

from app.core.db import SessionLocal, engine, get_database_url
from app.core.models import AgentFact, Conversation, Message

logger = logging.getLogger(__name__)
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

# Monotonic insertion counter so rows created within the same system-clock
# tick still have a deterministic lexicographic order.
_NOW_LOCK = threading.Lock()
_NOW_COUNTER = 0


def _now() -> str:
    """ISO-8601 timestamp with a monotonic counter so consecutive inserts
    sort deterministically (fixes message-order flakiness in tests)."""
    global _NOW_COUNTER
    with _NOW_LOCK:
        _NOW_COUNTER += 1
        counter = _NOW_COUNTER
    ns = time.time_ns()
    dt = datetime.fromtimestamp(ns / 1e9, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S") + f".{ns % 1_000_000_000:09d}-{counter:010d}+00:00"


def _ensure_sqlite_parent_dir() -> None:
    url = get_database_url()
    if url.startswith("sqlite:///"):
        parent = os.path.dirname(url[len("sqlite:///") :])
        if parent:
            os.makedirs(parent, exist_ok=True)


def _project_id_to_db(project_id: Optional[str]) -> Optional[str]:
    """Map API project_id ('' or None) to NULL in the DB."""
    if project_id is None or project_id == "":
        return None
    return project_id


def _fact_project_id_to_db(project_id: Optional[str]) -> str:
    """Agent facts use '' for project-less scope (schema NOT NULL DEFAULT '')."""
    if project_id is None or project_id == "":
        return ""
    return project_id


def _fact_project_id_from_db(project_id: Optional[str]) -> str:
    """Map DB NULL/'' back to '' for agent-fact API compatibility."""
    return project_id if project_id else ""


# Same cap the first user message uses when it stamps a title.
SESSION_TITLE_MAX = 80
_MARKUP = re.compile(r"<[^>]*>")


class SessionTitleError(ValueError):
    """A rename the server will not store. ``code`` is empty, long, or HTML."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _conversation_as_dict(conversation: Conversation) -> Dict[str, Any]:
    return {
        "id": conversation.id,
        "agent_name": conversation.agent_name,
        "project_id": conversation.project_id,
        "owner_id": conversation.owner_id,
        "title": conversation.title,
        "created_at": conversation.created_at,
        "updated_at": conversation.updated_at,
    }


def _message_as_dict(message: Message) -> Dict[str, Any]:
    return {
        "id": message.id,
        "conversation_id": message.conversation_id,
        "role": message.role,
        "content": message.content,
        "created_at": message.created_at,
        "provenance": (json.loads(message.provenance) if getattr(message, "provenance", None) else None),
    }


def _agent_fact_as_dict(fact: AgentFact) -> Dict[str, Any]:
    return {
        "id": fact.id,
        "agent_name": fact.agent_name,
        "project_id": _fact_project_id_from_db(fact.project_id),
        "conversation_id": fact.conversation_id,
        "key": fact.key,
        "value": fact.value,
        "updated_at": fact.updated_at,
    }


def init_db() -> None:
    """Create the schema if absent. Idempotent — safe to call on every startup.

    Cheap-idempotent: once created for the current database URL, repeated
    calls are a true no-op and never re-issue DDL. App startup calls this
    unconditionally on every boot (once per TestClient in the test suite);
    without this early-exit guard, DDL could re-run concurrently with a
    background task reading these tables and deadlock Postgres.
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
        Conversation.__table__.create(bind=engine, checkfirst=True)
        Message.__table__.create(bind=engine, checkfirst=True)
        AgentFact.__table__.create(bind=engine, checkfirst=True)
        _patch_conversation_columns(url)
        _initialized = True
        _initialized_for_url = url


def _patch_conversation_columns(url: str) -> None:
    """SQLite databases created before owner tracking. Postgres uses Alembic."""
    if not url.startswith("sqlite"):
        return
    from sqlalchemy import text as sqla_text

    with engine.connect() as conn:
        cols = {row[1] for row in conn.execute(sqla_text("PRAGMA table_info(conversations)"))}
        if cols and "owner_id" not in cols:
            conn.execute(sqla_text("ALTER TABLE conversations ADD COLUMN owner_id TEXT"))
            conn.commit()


def _ensure_db() -> None:
    if not _initialized or _initialized_for_url != get_database_url():
        init_db()


# ── conversations ────────────────────────────────────────────────────────────

def get_or_create_conversation(
    conversation_id: str,
    agent_name: str,
    project_id: Optional[str] = None,
    owner_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the existing conversation row or create it with the given id. Idempotent.

    A NULL ``project_id`` or ``owner_id`` is filled when the caller now knows
    one. A value already stored is never overwritten.
    """
    _ensure_db()
    db_project_id = _project_id_to_db(project_id)
    db_owner_id = owner_id or None
    with SessionLocal() as session:
        conversation = session.get(Conversation, conversation_id)
        if conversation is not None:
            needs_project = conversation.project_id is None and db_project_id is not None
            needs_owner = conversation.owner_id is None and db_owner_id is not None
            if not needs_project and not needs_owner:
                return _conversation_as_dict(conversation)

    now = _now()
    with _lock:
        with SessionLocal() as session:
            conversation = session.get(Conversation, conversation_id)
            if conversation is None:
                session.add(
                    Conversation(
                        id=conversation_id,
                        agent_name=agent_name,
                        project_id=db_project_id,
                        owner_id=db_owner_id,
                        title=None,
                        created_at=now,
                        updated_at=now,
                    )
                )
                session.commit()
                return _conversation_as_dict(
                    session.get(Conversation, conversation_id)  # type: ignore[arg-type]
                )
            if conversation.project_id is None and db_project_id is not None:
                session.execute(
                    update(Conversation)
                    .where(
                        Conversation.id == conversation_id,
                        Conversation.project_id.is_(None),
                    )
                    .values(project_id=db_project_id, updated_at=now)
                )
            if conversation.owner_id is None and db_owner_id is not None:
                session.execute(
                    update(Conversation)
                    .where(
                        Conversation.id == conversation_id,
                        Conversation.owner_id.is_(None),
                    )
                    .values(owner_id=db_owner_id, updated_at=now)
                )
            session.commit()
            session.refresh(conversation)
            return _conversation_as_dict(conversation)


def get_conversation(conversation_id: str) -> Optional[Dict[str, Any]]:
    """Return the conversation row for the given id, or None if it does not exist."""
    _ensure_db()
    with SessionLocal() as session:
        conversation = session.get(Conversation, conversation_id)
    return _conversation_as_dict(conversation) if conversation else None


def list_conversations(
    agent_name: Optional[str] = None,
    project_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    _ensure_db()
    with SessionLocal() as session:
        stmt = select(Conversation).order_by(Conversation.updated_at.desc())
        if agent_name is not None:
            stmt = stmt.where(Conversation.agent_name == agent_name)
        if project_id is not None:
            stmt = stmt.where(
                Conversation.project_id == _project_id_to_db(project_id)
            )
        rows = session.scalars(stmt).all()
    return [_conversation_as_dict(c) for c in rows]


def session_visible_to(
    conversation: Dict[str, Any],
    user_id: Optional[str],
    project_owner_id: Optional[str],
) -> bool:
    """A session belongs to the user who wrote it.

    Rows that predate owner tracking have a NULL ``owner_id``. Those stay
    with the project row owner, because there is no record of the writer.
    """
    if not user_id:
        return False
    owner = conversation.get("owner_id")
    if owner:
        return owner == user_id
    return bool(project_owner_id) and project_owner_id == user_id


def workspace_id_binds(conversation_id: str, project_id: str) -> bool:
    """True when a ``ws-`` id names this project and not a longer neighbour id."""
    if not conversation_id or not project_id:
        return False
    legacy = f"ws-{project_id}"
    return conversation_id == legacy or conversation_id.startswith(f"{legacy}-")


def _like_literal(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def list_visible_sessions(
    project_ids: Iterable[str],
    user_id: str,
    project_owner_id: Optional[str],
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Sessions on these project ids that ``user_id`` may see, newest first.

    A row counts when its stored ``project_id`` is one of ``project_ids``,
    or its id is the legacy ``ws-{id}`` thread or ``ws-{id}-{suffix}``.
    Another user's sessions are omitted. A NULL owner is visible only when
    ``user_id`` is the project row owner.
    """
    _ensure_db()
    ids = [pid for pid in dict.fromkeys(project_ids) if pid]
    if not ids or not user_id:
        return []
    bound = [Conversation.project_id.in_(ids)]
    for pid in ids:
        legacy = f"ws-{pid}"
        bound.append(Conversation.id == legacy)
        bound.append(
            Conversation.id.like(f"ws-{_like_literal(pid)}-%", escape="\\")
        )
    if project_owner_id and project_owner_id == user_id:
        visible = or_(
            Conversation.owner_id == user_id,
            Conversation.owner_id.is_(None),
        )
    else:
        visible = Conversation.owner_id == user_id
    with SessionLocal() as session:
        rows = session.scalars(
            select(Conversation)
            .where(or_(*bound))
            .where(visible)
            .order_by(Conversation.updated_at.desc())
            .limit(limit)
        ).all()
    return [_conversation_as_dict(c) for c in rows]


def collapse_session_title(raw: str) -> str:
    """Whitespace-collapsed title, or ``SessionTitleError``.

    ``<`` and ``>`` are rejected so a stored name cannot carry markup.
    """
    text = "" if raw is None else str(raw)
    if "<" in text or ">" in text:
        raise SessionTitleError("HTML")
    collapsed = " ".join(text.split())
    if not collapsed:
        raise SessionTitleError("empty")
    if len(collapsed) > SESSION_TITLE_MAX:
        raise SessionTitleError("long")
    return collapsed


def title_from_message(content: str) -> str:
    """Default title from a user message: markup removed, then truncated."""
    text = _MARKUP.sub(" ", content or "")
    text = text.replace("<", " ").replace(">", " ")
    return " ".join(text.split())[:SESSION_TITLE_MAX]


def rename_conversation(conversation_id: str, title: str) -> Optional[Dict[str, Any]]:
    """Store a collapsed title. Returns None when the row does not exist."""
    stored = collapse_session_title(title)
    _ensure_db()
    now = _now()
    with _lock:
        with SessionLocal() as session:
            conversation = session.get(Conversation, conversation_id)
            if conversation is None:
                return None
            conversation.title = stored
            conversation.updated_at = now
            session.commit()
            session.refresh(conversation)
            return _conversation_as_dict(conversation)


def delete_conversation(conversation_id: str) -> bool:
    _ensure_db()
    with _lock:
        with SessionLocal() as session:
            conversation = session.get(Conversation, conversation_id)
            if not conversation:
                return False
            session.delete(conversation)
            session.commit()
            return True


def clear_conversation(conversation_id: str) -> Dict[str, int]:
    """Wipe the conversation's messages and agent_facts without dropping
    the conversation row itself. Used by the UI's "Clear history" button
    to escape a thread poisoned by prior hallucinated turns while keeping
    the conversation_id stable (so the React composer doesn't need to
    remount).

    Returns ``{"messages": N, "facts": M}`` so the caller can surface
    how much was removed. Idempotent — clearing an empty / nonexistent
    conversation returns zeros without raising. Also drops the staged
    conversation WBS snapshot so a later export cannot resurrect it.
    """
    try:
        from app.core.conversation_wbs import clear_conversation_wbs
        clear_conversation_wbs(conversation_id)
    except Exception:
        logger.warning(
            "could not clear staged WBS for conversation %s",
            conversation_id, exc_info=True,
        )
    _ensure_db()
    with _lock:
        with SessionLocal() as session:
            msgs = session.execute(
                delete(Message).where(Message.conversation_id == conversation_id)
            ).rowcount
            facts = session.execute(
                delete(AgentFact).where(
                    AgentFact.conversation_id == conversation_id
                )
            ).rowcount
            session.execute(
                update(Conversation)
                .where(Conversation.id == conversation_id)
                .values(updated_at=_now())
            )
            session.commit()
    return {"messages": int(msgs or 0), "facts": int(facts or 0)}


# ── messages ─────────────────────────────────────────────────────────────────

def append_message(
    conversation_id: str, role: str, content: str, provenance: Optional[list] = None,
) -> Dict[str, Any]:
    """Insert a message and bump the conversation's updated_at.

    The FIRST user message also stamps the conversation's title (when still
    NULL) so session lists read like the design's chat history ("pipe
    specs...") instead of raw conversation ids. Never overwritten after.

    An assistant message stores its provenance record (where each figure and
    fact came from): passed in, else the record of the answer just finished
    in this task.
    """
    _ensure_db()
    mid = str(uuid.uuid4())
    now = _now()
    if role == "assistant" and provenance is None:
        from app.agents.provenance_trail import LAST_PROVENANCE

        provenance = LAST_PROVENANCE.get()
    if role == "assistant" and isinstance(content, str):
        # A reopened conversation shows what was stored; it passes the same
        # exit check as the live answer. check_text never raises.
        from app.agents import answer_exit

        content = answer_exit.check_text_or_fallback(content)
    with _lock:
        with SessionLocal() as session:
            session.add(
                Message(
                    id=mid,
                    conversation_id=conversation_id,
                    role=role,
                    content=content,
                    created_at=now,
                    provenance=(json.dumps(provenance, default=str) if provenance else None),
                )
            )
            session.execute(
                update(Conversation)
                .where(Conversation.id == conversation_id)
                .values(updated_at=now)
            )
            if role == "user" and content:
                title = title_from_message(content)
                if title:
                    session.execute(
                        update(Conversation)
                        .where(
                            Conversation.id == conversation_id,
                            Conversation.title.is_(None),
                        )
                        .values(title=title)
                    )
            session.commit()
    with SessionLocal() as session:
        message = session.get(Message, mid)
    return _message_as_dict(message)  # type: ignore[arg-type]


def get_messages(conversation_id: str, limit: int = 40) -> List[Dict[str, Any]]:
    """Return messages oldest-first. If limit is set, return the most recent `limit` rows, still oldest-first."""
    _ensure_db()
    with SessionLocal() as session:
        rows = session.scalars(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc(), Message.id.desc())
            .limit(limit)
        ).all()
    return [_message_as_dict(m) for m in reversed(rows)]


# ── agent facts ───────────────────────────────────────────────────────────────

def set_agent_fact(
    agent_name: str,
    key: str,
    value: str,
    conversation_id: Optional[str] = None,
    project_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Upsert a durable fact for an agent, scoped to a project.

    Facts are keyed by (agent_name, project_id, key) so a fact remembered in
    one project is not visible to the same agent in another project. A
    project-less conversation uses the '' scope.
    """
    _ensure_db()
    db_project_id = _fact_project_id_to_db(project_id)
    now = _now()
    with _lock:
        with SessionLocal() as session:
            stmt = select(AgentFact).where(
                AgentFact.agent_name == agent_name,
                AgentFact.key == key,
                AgentFact.project_id == db_project_id,
            )
            existing = session.scalars(stmt).one_or_none()
            if existing:
                existing.value = value
                existing.conversation_id = conversation_id
                existing.updated_at = now
            else:
                session.add(
                    AgentFact(
                        id=str(uuid.uuid4()),
                        agent_name=agent_name,
                        project_id=db_project_id,
                        conversation_id=conversation_id,
                        key=key,
                        value=value,
                        updated_at=now,
                    )
                )
            session.commit()
    with SessionLocal() as session:
        fact = session.scalars(
            select(AgentFact).where(
                AgentFact.agent_name == agent_name,
                AgentFact.key == key,
                AgentFact.project_id == db_project_id,
            )
        ).one()
    return _agent_fact_as_dict(fact)


def list_agent_facts(
    agent_name: str, project_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """List an agent's facts for one project scope ('' = project-less)."""
    _ensure_db()
    db_project_id = _fact_project_id_to_db(project_id)
    with SessionLocal() as session:
        stmt = (
            select(AgentFact)
            .where(
                AgentFact.agent_name == agent_name,
                AgentFact.project_id == db_project_id,
            )
            .order_by(AgentFact.key)
        )
        rows = session.scalars(stmt).all()
    return [_agent_fact_as_dict(f) for f in rows]
