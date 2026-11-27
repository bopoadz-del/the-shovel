"""The model's own JSON memory: working notes it writes and reads by tool.

One table (``llm_memory``): project, name, data (JSON text), written_at,
written_by_turn. A note belongs to ONE project, and the project always comes
from the turn, never from the model's arguments. A write replaces the whole
document in one transaction. Nothing here is injected into a prompt: a note
reaches the model only as the result of its own ``read_memory`` /
``list_memory`` call, labelled as its note with the date it was written.

This store is separate from ``agent_memory``'s facts on purpose: a fact is a
key/value the app may hand to a prompt once verified; a note is the model's
own JSON on how to work, handed back only when it asks for it.

The owner calibrates the notes: ``review`` keeps or discards the current
version of a note. A discarded note stays in the table, marked, and is no
longer returned by the model's ``read`` / ``listing``; every decision is
recorded in ``llm_memory_review`` with the content it was made on. Code never
edits a note's content and holds no rule about what a good note is.

Limits are settings (environment): ``LLM_MEMORY_MAX_BYTES`` per document and
``LLM_MEMORY_MAX_DOCS`` per project.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import func, inspect, select, update
from sqlalchemy import text as sql_text
from sqlalchemy.exc import IntegrityError

from app.core.db import SessionLocal, engine, get_database_url
from app.core.models import Conversation, LlmMemory, LlmMemoryReview

DEFAULT_MAX_BYTES = 16384
DEFAULT_MAX_DOCS = 50
DISCARDED = "discarded"
#: The owner's two decisions and the mark each leaves on the note.
DECISIONS = {"keep": "kept", "discard": DISCARDED}
_MARK_COLUMNS = ("review", "reviewed_at", "reviewed_by")
#: Letters, digits, hyphen and underscore, 1-64 characters; ``.json`` optional.
_NAME_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")

_lock = threading.Lock()
_initialized_for_url: Optional[str] = None


class MemoryConflict(ValueError):
    """The note changed after the owner opened it; the decision was not made."""


class MemoryRefused(ValueError):
    """A request the model can correct: its message is returned to it."""


def _setting(env: str, default: int) -> int:
    try:
        value = int(os.getenv(env) or default)
    except ValueError:
        return default
    return value if value > 0 else default


def max_bytes() -> int:
    return _setting("LLM_MEMORY_MAX_BYTES", DEFAULT_MAX_BYTES)


def max_docs() -> int:
    return _setting("LLM_MEMORY_MAX_DOCS", DEFAULT_MAX_DOCS)


def init_db() -> None:
    """Create the table if absent (SQLite and tests; Postgres uses Alembic)."""
    global _initialized_for_url
    url = get_database_url()
    if _initialized_for_url == url:
        return
    with _lock:
        if _initialized_for_url == url:
            return
        if url.startswith("sqlite:///"):
            parent = os.path.dirname(url[len("sqlite:///"):])
            if parent:
                os.makedirs(parent, exist_ok=True)
        LlmMemory.__table__.create(bind=engine, checkfirst=True)
        LlmMemoryReview.__table__.create(bind=engine, checkfirst=True)
        # A table made before the owner's mark existed gets its columns.
        with engine.begin() as conn:
            have = {c["name"] for c in inspect(conn).get_columns("llm_memory")}
            for col in _MARK_COLUMNS:
                if col not in have:
                    conn.execute(sql_text(f"ALTER TABLE llm_memory ADD COLUMN {col} "
                                          "VARCHAR NOT NULL DEFAULT ''"))
        _initialized_for_url = url


def valid_name(name: Any) -> Optional[str]:
    """The stored name for ``name`` (``.json`` taken off), or None when the
    name is not text or has anything outside letters, digits, hyphen and
    underscore, or is longer than 64."""
    if not isinstance(name, str):
        return None
    stem = name[:-5] if name.lower().endswith(".json") else name
    return stem if _NAME_RE.fullmatch(stem) else None


def clean_name(name: Any) -> str:
    """The stored name for ``name``; a name ``valid_name`` rejects is refused
    with a message the model can act on."""
    if not isinstance(name, str):
        raise MemoryRefused("name must be a text: letters, digits, '-' or '_', at most 64 characters")
    stem = valid_name(name)
    if stem is None:
        raise MemoryRefused(
            f"name {name!r} is not allowed: use letters, digits, '-' or '_', at most 64 "
            "characters, optionally ending in .json")
    return stem


def _not_json_constant(token: str) -> Any:
    raise ValueError(f"{token} is not a JSON number")


def parse_data(data: Any) -> Tuple[str, Any]:
    """``(json_text, value)`` for a document: a JSON object or array, given
    as such or as JSON text. Anything else is refused."""
    value = data
    if isinstance(data, str):
        try:
            value = json.loads(data, parse_constant=_not_json_constant)
        except ValueError as exc:
            msg = getattr(exc, "msg", str(exc))
            where = f" at character {exc.pos}" if getattr(exc, "pos", None) is not None else ""
            raise MemoryRefused(f"data is not JSON ({msg}{where}); "
                               "send the whole document as a JSON object") from None
    if not isinstance(value, (dict, list)):
        raise MemoryRefused("data must be a JSON object or array (the whole document)")
    try:
        # Strict JSON only: NaN and Infinity are not JSON and would be stored
        # as text no JSON reader accepts.
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    except ValueError:
        raise MemoryRefused("data is not JSON (NaN and Infinity are not JSON numbers); "
                           "write the figure as a number or as text") from None
    return text, value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write(project_id: str, name: Any, data: Any, turn: str = "") -> Dict[str, Any]:
    """Replace the whole document ``name`` of ``project_id`` in one
    transaction. Refused (nothing written) for a bad name, data that is not a
    JSON object/array, a document over the size limit, or a new document past
    the per-project count."""
    if not project_id:
        raise MemoryRefused("memory needs a project; this conversation has none")
    stem = clean_name(name)
    text, _value = parse_data(data)
    size = len(text.encode("utf-8"))
    if size > max_bytes():
        raise MemoryRefused(f"document is {size} bytes; the limit is {max_bytes()} bytes. "
                           "Keep only what helps you work and write it again")
    init_db()
    # A write is a new version of the note: the owner's mark on the old one
    # does not carry over (the decision record keeps what it was made on).
    row = {"data": text, "size_bytes": size, "written_at": _now(), "written_by_turn": turn or "",
           "review": "", "reviewed_at": "", "reviewed_by": ""}
    with SessionLocal() as s:
        with s.begin():
            prev = s.scalar(select(LlmMemory.review)
                            .where(LlmMemory.project_id == project_id, LlmMemory.name == stem))
            if prev is None or prev == DISCARDED:
                _one_new_note_at_a_time(s, project_id)
                # Discarded notes are out of the model's sight, so they do not
                # take a place it could only free by replacing them; writing
                # one again brings it back into sight, so it needs a place.
                count = s.scalar(select(func.count()).select_from(LlmMemory)
                                 .where(LlmMemory.project_id == project_id,
                                        LlmMemory.review != DISCARDED)) or 0
                if count >= max_docs():
                    raise MemoryRefused(f"this project already holds {count} notes (the limit); "
                                       "replace or reuse one of them (list_memory shows them)")
            done = s.execute(update(LlmMemory)
                             .where(LlmMemory.project_id == project_id, LlmMemory.name == stem)
                             .values(**row)).rowcount
            if not done:
                s.add(LlmMemory(project_id=project_id, name=stem, **row))
        created = not done
    return {"name": stem, "bytes": size, "written_at": row["written_at"], "created": created}


def _one_new_note_at_a_time(session: Any, project_id: str) -> None:
    """Hold this project's new-note lock until the transaction ends, so the
    count that follows is the count the insert is made against. Without it,
    two turns creating two different notes on Postgres (READ COMMITTED) both
    read the same count and both insert, one past the limit. SQLite has one
    writer at a time, so its count is already the insert's count."""
    if session.get_bind().dialect.name == "postgresql":
        session.execute(sql_text("SELECT pg_advisory_xact_lock(hashtextextended(:project, 0))"),
                        {"project": f"llm_memory:{project_id}"})


def write_retrying(project_id: str, name: Any, data: Any, turn: str = "") -> Dict[str, Any]:
    """``write``; when a concurrent write inserted the same new document
    first, this one replaces it (the last whole document wins)."""
    try:
        return write(project_id, name, data, turn)
    except IntegrityError:
        return write(project_id, name, data, turn)


def read(project_id: str, name: Any) -> Dict[str, Any]:
    if not project_id:
        raise MemoryRefused("memory needs a project; this conversation has none")
    stem = clean_name(name)
    init_db()
    with SessionLocal() as s:
        row = s.get(LlmMemory, (project_id, stem))
        if row is None or row.review == DISCARDED:
            raise MemoryRefused(f"no note named {stem!r} in this project (list_memory shows them)")
        return {"name": row.name, "data": json.loads(row.data), "bytes": row.size_bytes,
                "written_at": row.written_at, "written_by_turn": row.written_by_turn}


def listing(project_id: str, with_data: bool = False,
            with_review: bool = False) -> List[Dict[str, Any]]:
    """The project's notes by name. The model's view (default) leaves out
    discarded notes; ``with_review`` (the owner's view) lists every note with
    its mark."""
    if not project_id:
        raise MemoryRefused("memory needs a project; this conversation has none")
    init_db()
    with SessionLocal() as s:
        query = select(LlmMemory).where(LlmMemory.project_id == project_id)
        if not with_review:
            query = query.where(LlmMemory.review != DISCARDED)
        rows = s.scalars(query.order_by(LlmMemory.name)).all()
        out = []
        for row in rows:
            entry = {"name": row.name, "bytes": row.size_bytes, "written_at": row.written_at,
                     "written_by_turn": row.written_by_turn}
            if with_data:
                entry["data"] = json.loads(row.data)
            if with_review:
                entry.update(review=row.review, reviewed_at=row.reviewed_at,
                             reviewed_by=row.reviewed_by, version=version_of(row),
                             turn=turn_label(s, project_id, row.written_by_turn))
            out.append(entry)
        return out


def turn_label(session: Any, project_id: str, ref: str) -> Dict[str, Any]:
    """The turn a note was written in, as a person reads it: the
    conversation's title (only when that conversation is in the same project)
    and the number of the user's message in it. The stored reference is an
    internal id and is not part of the label."""
    conv_id, sep, number = (ref or "").rpartition("#")
    message = int(number) if sep and number.isdigit() and int(number) > 0 else None
    title = None
    if conv_id:
        conv = session.get(Conversation, conv_id)
        if conv is not None and conv.project_id == project_id:
            title = (conv.title or "").strip() or None
    return {"conversation": title, "message": message}


def version_of(row: Any) -> str:
    """A token for one version of a note: it changes with every write (its
    content, time and turn), so a decision names the version it was made on."""
    raw = f"{row.written_at}\n{row.written_by_turn}\n{row.data}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def review(project_id: str, name: Any, decision: Any, version: Any,
           by: str = "") -> Dict[str, Any]:
    """The owner keeps or discards note ``name`` at the ``version`` they
    read (``version_of``, listed with the note): the note is marked and the
    decision recorded, with the content it was made on, in one transaction.
    When the note was written again since, nothing is marked or recorded and
    ``MemoryConflict`` says so. The note's content is never changed."""
    if not project_id:
        raise MemoryRefused("a review needs a project")
    if not isinstance(decision, str) or decision not in DECISIONS:
        raise MemoryRefused(f"decision must be one of {sorted(DECISIONS)}")
    if not isinstance(version, str) or not version:
        raise MemoryRefused("a review needs the version of the note that was read")
    stem = clean_name(name)
    init_db()
    at = _now()
    with SessionLocal() as s:
        with s.begin():
            row = s.get(LlmMemory, (project_id, stem))
            if row is None:
                raise LookupError(f"no note named {stem!r} in project {project_id!r}")
            if version_of(row) != version:
                raise MemoryConflict(f"note {stem!r} was written again after it was opened; "
                                     "reload and decide on the version shown")
            record = LlmMemoryReview(project_id=project_id, name=stem, decision=decision,
                                     data=row.data, note_written_at=row.written_at,
                                     note_written_by_turn=row.written_by_turn,
                                     decided_at=at, decided_by=by or "")
            # The mark lands only if the row is still the version just read:
            # a write that commits after the read leaves it untouched.
            marked = s.execute(update(LlmMemory)
                               .where(LlmMemory.project_id == project_id,
                                      LlmMemory.name == stem,
                                      LlmMemory.data == record.data,
                                      LlmMemory.written_at == record.note_written_at,
                                      LlmMemory.written_by_turn == record.note_written_by_turn)
                               .values(review=DECISIONS[decision], reviewed_at=at,
                                       reviewed_by=by or "")
                               .execution_options(synchronize_session=False)).rowcount
            if not marked:
                raise MemoryConflict(f"note {stem!r} was written again while the decision was "
                                     "being made; reload and decide on the version shown")
            s.add(record)
    return {"name": stem, "decision": decision, "review": DECISIONS[decision],
            "decided_at": at, "decided_by": by or ""}


def reviews(project_id: str) -> List[Dict[str, Any]]:
    """Every keep / discard made in the project, oldest first, each with the
    note's content as it was when decided."""
    init_db()
    with SessionLocal() as s:
        rows = s.scalars(select(LlmMemoryReview)
                         .where(LlmMemoryReview.project_id == project_id)
                         .order_by(LlmMemoryReview.id)).all()
        return [{"name": r.name, "decision": r.decision, "data": json.loads(r.data),
                 "note_written_at": r.note_written_at,
                 "note_written_by_turn": r.note_written_by_turn,
                 "decided_at": r.decided_at, "decided_by": r.decided_by} for r in rows]
