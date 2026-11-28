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

Limits are settings (environment): ``LLM_MEMORY_MAX_BYTES`` per document,
``LLM_MEMORY_MAX_DOCS`` per project and ``LLM_MEMORY_MAX_DEPTH`` levels of
nesting in a document.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

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
DEFAULT_MAX_DEPTH = 32
#: No setting raises the nesting limit above this. A document nested much
#: deeper can be parsed and stored and then not written out again from inside
#: a turn (the JSON writer runs out of stack), so every later read of it
#: would end the turn.
DEPTH_CEILING = 100
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


def max_depth() -> int:
    return min(_setting("LLM_MEMORY_MAX_DEPTH", DEFAULT_MAX_DEPTH), DEPTH_CEILING)


def nested_deeper_than(value: Any, limit: int) -> bool:
    """True when ``value`` holds objects or arrays more than ``limit`` deep.
    Counted from a list of what is left to visit, so the count cannot itself
    run out of stack on the document it measures."""
    left = [(value, 1)]
    while left:
        node, level = left.pop()
        if not isinstance(node, (dict, list, tuple)):
            continue
        if level > limit:
            return True
        left.extend((child, level + 1) for child in (node.values() if isinstance(node, dict) else node))
    return False


def _too_deep(limit: int) -> MemoryRefused:
    return MemoryRefused(f"data is nested more than {limit} levels deep (the limit is {limit}); "
                         "write the document with less nesting")


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


#: The field of a note's object that marks what it holds as told by the
#: user (true or false). A field of the note itself, no column: the note's
#: own record says it, and a read hands it back with the note.
TOLD_BY_USER = "told_by_user"


def told_by_user(data: Any) -> bool:
    """True when the note ``data`` is marked as told by the user."""
    return isinstance(data, dict) and data.get(TOLD_BY_USER) is True


#: A figure as every figure check reads one (the exit check's own reader,
#: citation_provenance._nums, uses this pattern): any run of digits, so the
#: 3 of "m3" and the 40 of "C40" count; a check may refuse more, never less.
FIGURE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def figure_values(text: Any) -> Set[float]:
    """The figures written in ``text``, by value: "1,850" and "1850" are one
    figure, and so are "4.10" and "4.1". Every check that compares a told
    figure with what was written reads figures through this one function."""
    out: Set[float] = set()
    for token in FIGURE.findall(text if isinstance(text, str) else ""):
        try:
            out.add(float(token.replace(",", "")))
        except ValueError:
            continue  # "1,,": digits with no figure in them
    return out


def note_figures(data: Any) -> Set[float]:
    """The figures a note's values hold, by value: a JSON number as it is,
    figures written inside text as ``figure_values`` reads them. Key names
    and true/false are not figures."""
    if isinstance(data, bool):
        return set()
    if isinstance(data, (int, float)):
        # Through its text, so a number past what a float holds reads as a
        # written one does (inf), never as an error.
        return {float(str(data))}
    if isinstance(data, str):
        return figure_values(data)
    if isinstance(data, dict):
        return set().union(*(note_figures(v) for v in data.values()))
    if isinstance(data, list):
        return set().union(*(note_figures(v) for v in data))
    return set()


def untold_figures(value: Any, told_from: Iterable[str], before: Any = None) -> List[float]:
    """The figures of a note marked as told by the user that the user did
    not write: not in ``told_from`` (the user's own words in the writing
    conversation) and not in ``before``, the note's earlier version when
    that version was itself told (each of its figures was checked when it
    was written). Empty for a note not marked as told."""
    if not told_by_user(value):
        return []
    backed = set().union(*(figure_values(t) for t in told_from))
    if told_by_user(before):
        backed |= note_figures(before)
    return sorted(note_figures(value) - backed)


def _shown(value: float) -> str:
    return str(int(value)) if value.is_integer() else repr(value)


def parse_data(data: Any) -> Tuple[str, Any]:
    """``(json_text, value)`` for a document: a JSON object or array, given
    as such or as JSON text. Anything else is refused."""
    value = data
    if isinstance(data, str):
        try:
            value = json.loads(data, parse_constant=_not_json_constant)
        except RecursionError:
            # Nested past what the JSON reader itself can follow.
            raise _too_deep(max_depth()) from None
        except ValueError as exc:
            msg = getattr(exc, "msg", str(exc))
            where = f" at character {exc.pos}" if getattr(exc, "pos", None) is not None else ""
            raise MemoryRefused(f"data is not JSON ({msg}{where}); "
                               "send the whole document as a JSON object") from None
    if not isinstance(value, (dict, list)):
        raise MemoryRefused("data must be a JSON object or array (the whole document)")
    if isinstance(value, dict) and TOLD_BY_USER in value and not isinstance(value[TOLD_BY_USER], bool):
        raise MemoryRefused(f"{TOLD_BY_USER} must be true or false (true: the user told you "
                           "what this note holds)")
    if nested_deeper_than(value, max_depth()):
        # Before it is written out: a document this deep is refused, never
        # stored as a note no later turn could read back.
        raise _too_deep(max_depth())
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


def write(project_id: str, name: Any, data: Any, turn: str = "",
          told_from: Iterable[str] = ()) -> Dict[str, Any]:
    """Replace the whole document ``name`` of ``project_id`` in one
    transaction. Refused (nothing written) for a bad name, data that is not a
    JSON object/array, a document over the size limit, a new document past
    the per-project count, or a note marked as told by the user holding a
    figure the user did not write (``told_from``: the user's own words in
    the conversation that writes it; see ``untold_figures``)."""
    if not project_id:
        raise MemoryRefused("memory needs a project; this conversation has none")
    stem = clean_name(name)
    text, value = parse_data(data)
    told_from = [t for t in told_from if isinstance(t, str)]
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
            # A note in the model's sight is replaced in place. This write
            # comes first: on SQLite it takes the one writer's lock, so the
            # count below is the count the insert is made against.
            revived = 0
            _check_told(s, project_id, stem, value, told_from)
            done = _replace(s, project_id, stem, row)
            if not done:
                _one_new_note_at_a_time(s, project_id)
                # Asked again now the lock is held: another turn may have
                # created this same note while this one waited, and then
                # this write is a replace, not a new note against the limit.
                done = _replace(s, project_id, stem, row)
            if not done:
                # Discarded notes are out of the model's sight, so they do not
                # take a place it could only free by replacing them; writing
                # one again brings it back into sight, so it needs a place.
                count = _notes_in_sight(s, project_id)
                if count >= max_docs():
                    raise MemoryRefused(f"this project already holds {count} notes (the limit); "
                                       "replace or reuse one of them (list_memory shows them)")
                revived = s.execute(update(LlmMemory)
                                    .where(LlmMemory.project_id == project_id, LlmMemory.name == stem)
                                    .values(**row)).rowcount
                if not revived:
                    s.add(LlmMemory(project_id=project_id, name=stem, **row))
        created = not (done or revived)
    return {"name": stem, "bytes": size, "written_at": row["written_at"], "created": created}


def _check_told(session: Any, project_id: str, stem: str, value: Any, told_from: List[str]) -> None:
    """Refuse a note marked as told by the user that holds a figure the user
    did not write. The mark is the model's word; this is the check on it,
    against the note's earlier version in the model's sight (read in the
    write's own transaction) and the user's words."""
    if not told_by_user(value):
        return
    row = session.get(LlmMemory, (project_id, stem))
    before = json.loads(row.data) if row is not None and row.review != DISCARDED else None
    untold = untold_figures(value, told_from, before)
    if untold:
        raise MemoryRefused(
            f"this note is marked {TOLD_BY_USER} but the user did not write "
            f"{', '.join(_shown(v) for v in untold)} in this conversation; nothing was written. "
            f"Keep in a told note only figures the user gave, or write it without the mark")


def _replace(session: Any, project_id: str, stem: str, row: Dict[str, Any]) -> int:
    """Replace the note ``stem`` of ``project_id`` with ``row`` when it is in
    the model's sight; the number of notes replaced (0 when there is no such
    note yet, or it is discarded)."""
    return session.execute(update(LlmMemory)
                           .where(LlmMemory.project_id == project_id, LlmMemory.name == stem,
                                  LlmMemory.review != DISCARDED)
                           .values(**row)).rowcount


def _notes_in_sight(session: Any, project_id: str) -> int:
    """How many of the project's notes the model can see: the places taken
    under ``LLM_MEMORY_MAX_DOCS``. A discarded note takes no place."""
    return session.scalar(select(func.count()).select_from(LlmMemory)
                          .where(LlmMemory.project_id == project_id,
                                 LlmMemory.review != DISCARDED)) or 0


def _one_new_note_at_a_time(session: Any, project_id: str) -> None:
    """Hold this project's new-note lock until the transaction ends, so the
    count that follows is the count the insert is made against. Without it,
    two turns creating two different notes on Postgres (READ COMMITTED) both
    read the same count and both insert, one past the limit. SQLite has one
    writer at a time, so its count is already the insert's count."""
    if session.get_bind().dialect.name == "postgresql":
        session.execute(sql_text("SELECT pg_advisory_xact_lock(hashtextextended(:project, 0))"),
                        {"project": f"llm_memory:{project_id}"})


def write_retrying(project_id: str, name: Any, data: Any, turn: str = "",
                   told_from: Iterable[str] = ()) -> Dict[str, Any]:
    """``write``; when a concurrent write inserted the same new document
    first, this one replaces it (the last whole document wins)."""
    told_from = list(told_from)
    try:
        return write(project_id, name, data, turn, told_from)
    except IntegrityError:
        return write(project_id, name, data, turn, told_from)


def verified_words(review: str) -> str:
    """Whether the owner verified a note, in words: only a note the owner
    kept is verified."""
    return "kept by the owner (verified)" if review == DECISIONS["keep"] else "not verified by the owner"


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
                "written_at": row.written_at, "written_by_turn": row.written_by_turn,
                "review": row.review}


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
    ``MemoryConflict`` says so. Keeping a discarded note needs a place under
    the count limit, as writing it again would; at the limit nothing is
    marked or recorded and ``MemoryRefused`` says so. The note's content is
    never changed."""
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
            back_in_sight = decision == "keep" and row.review == DISCARDED
            if back_in_sight:
                # Keeping a discarded note puts it back in the model's sight,
                # so it takes a place under the limit like a new note: the
                # same per-project lock as ``write``, and taken HERE, before
                # the mark touches the row. A write that revives this same
                # note takes the project lock and then the row; taking them
                # in the other order here would let the two wait on each
                # other for ever on Postgres.
                _one_new_note_at_a_time(s, project_id)
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
            if back_in_sight:
                # Counted under the project lock taken above, with this note
                # counted: the count the Keep is made against.
                count = _notes_in_sight(s, project_id)
                if count > max_docs():
                    raise MemoryRefused(f"keeping {stem!r} would put {count} notes in the model's "
                                       f"sight; the limit is {max_docs()}. Discard another note "
                                       "first, then keep this one")
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
