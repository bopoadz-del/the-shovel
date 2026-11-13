"""The ML loop's tables and repository.

All tables live in the block's own ``MetaData`` and carry the ``ml_``
prefix, so the block's schema is one namespace a Store deployment can
create, export and import without any host table. Every row is keyed by
project, conversation, message and build sha.

The repository never opens its own engine: it is handed ONE session
factory (see :class:`~app.core.ml_loop.settings.MLLoopSettings`).
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    func,
    insert,
    select,
)

TABLE_PREFIX = "ml_"

metadata = MetaData()


def _key_columns() -> list[Column]:
    return [
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("project_id", String(200), nullable=False),
        Column("conversation_id", String(200), nullable=True),
        Column("message_id", String(200), nullable=True),
        Column("build_sha", String(64), nullable=True),
        Column("created_at", DateTime(timezone=True), nullable=False),
    ]


ml_turns = Table(
    "ml_turns", metadata,
    *_key_columns(),
    Column("question", Text, nullable=False),
    Column("answer", Text, nullable=False),
    Column("verdict", String(64), nullable=False),
    Column("route", String(120), nullable=True),
    Column("evidence_plan", JSON, nullable=False),
    Column("retrieved_ids", JSON, nullable=False),
    Column("cited_ids", JSON, nullable=False),
    Index("ix_ml_turns_project_created", "project_id", "created_at"),
)

ml_feedback = Table(
    "ml_feedback", metadata,
    *_key_columns(),
    Column("kind", String(32), nullable=False),
    Column("correct_value", Float, nullable=True),
    Column("unit", String(32), nullable=True),
    Column("note", String(200), nullable=True),
    Column("cited_ids", JSON, nullable=False),
    Column("user_id", String(200), nullable=True),
    Index("ix_ml_feedback_project_created", "project_id", "created_at"),
)

ml_corrections = Table(
    "ml_corrections", metadata,
    *_key_columns(),
    Column("formula_id", String(200), nullable=False),
    Column("predicted", Float, nullable=False),
    Column("actual", Float, nullable=False),
    Column("source", String(64), nullable=False),
    Index("ix_ml_corrections_formula", "formula_id", "created_at"),
)

#: Tuned coefficients with version and provenance (written by ML-1 tuning).
ml_coefficients = Table(
    "ml_coefficients", metadata,
    *_key_columns(),
    Column("formula_id", String(200), nullable=False),
    Column("version", Integer, nullable=False),
    Column("coefficients", JSON, nullable=False),
    Column("mae", Float, nullable=True),
    Column("tier", String(32), nullable=False),
    Column("provenance", JSON, nullable=False),
    Index("ix_ml_coefficients_formula_version", "formula_id", "version"),
)

ml_route_corrections = Table(
    "ml_route_corrections", metadata,
    *_key_columns(),
    Column("message", Text, nullable=False),
    Column("correct_action", String(120), nullable=False),
    Column("original_action", String(120), nullable=True),
    Column("source", String(64), nullable=False),
    Index("ix_ml_route_corrections_project_created", "project_id", "created_at"),
)

ml_shadow_predictions = Table(
    "ml_shadow_predictions", metadata,
    *_key_columns(),
    Column("predictor", String(120), nullable=False),
    Column("live_label", String(120), nullable=False),
    Column("shadow_label", String(120), nullable=False),
    Column("confidence", Float, nullable=True),
    Index("ix_ml_shadow_predictor_created", "predictor", "created_at"),
)

#: Every table the block owns, by name. Export/import walk exactly these.
TABLES: dict[str, Table] = {t.name: t for t in metadata.sorted_tables}


def create_tables(bind: Any) -> None:
    """Create any of the block's tables that do not exist yet (idempotent).

    A host that runs migrations gets the same tables from its migration;
    this is for a host without one (a local SQLite store, a Store deploy).
    """
    metadata.create_all(bind, checkfirst=True)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_aware(value: Any) -> Any:
    """SQLite drops tzinfo on read; treat a naive stored stamp as UTC."""
    if isinstance(value, datetime) and value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _jsonable_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        value = _as_aware(value)
        out[key] = value.isoformat() if isinstance(value, datetime) else value
    return out


@dataclass(frozen=True)
class MLLoopRepository:
    """Reads and writes the block's tables through one session factory."""

    session_factory: Callable[[], Any]

    # ── writes ──────────────────────────────────────────────────────────
    def insert(self, table_name: str, values: Mapping[str, Any]) -> int:
        """Insert one row; returns its id. ``created_at`` defaults to now."""
        table = TABLES[table_name]
        row = dict(values)
        row.setdefault("created_at", utcnow())
        with self.session_factory() as session:
            result = session.execute(insert(table).values(**row))
            session.commit()
            return int(result.inserted_primary_key[0])

    def import_rows(self, table_name: str, rows: Iterable[Mapping[str, Any]]) -> int:
        """Append exported rows. Ids are reassigned; unknown columns refused."""
        table = TABLES[table_name]
        allowed = {c.name for c in table.columns} - {"id"}
        prepared: list[dict[str, Any]] = []
        for raw in rows:
            unknown = set(raw) - allowed - {"id"}
            if unknown:
                raise ValueError(
                    f"{table_name}: unknown column(s) {sorted(unknown)}"
                )
            row = {k: v for k, v in raw.items() if k in allowed}
            stamp = row.get("created_at")
            if isinstance(stamp, str):
                row["created_at"] = datetime.fromisoformat(stamp)
            row.setdefault("created_at", utcnow())
            prepared.append(row)
        if not prepared:
            return 0
        with self.session_factory() as session:
            session.execute(insert(table), prepared)
            session.commit()
        return len(prepared)

    # ── reads ───────────────────────────────────────────────────────────
    def rows(
        self,
        table_name: str,
        *,
        project_id: str | None = None,
        since: datetime | None = None,
        where: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        table = TABLES[table_name]
        stmt = select(table).order_by(table.c.id)
        if project_id is not None:
            stmt = stmt.where(table.c.project_id == project_id)
        if since is not None:
            stmt = stmt.where(table.c.created_at >= since)
        for key, value in (where or {}).items():
            stmt = stmt.where(table.c[key] == value)
        with self.session_factory() as session:
            result = session.execute(stmt).mappings().all()
        return [
            {k: _as_aware(v) for k, v in dict(r).items()} for r in result
        ]

    def counts(self, project_id: str | None = None) -> dict[str, int]:
        out: dict[str, int] = {}
        with self.session_factory() as session:
            for name, table in TABLES.items():
                stmt = select(func.count()).select_from(table)
                if project_id is not None:
                    stmt = stmt.where(table.c.project_id == project_id)
                out[name] = int(session.execute(stmt).scalar_one())
        return out

    def export_tables(self, project_id: str | None = None) -> dict[str, list[dict[str, Any]]]:
        return {
            name: [_jsonable_row(r) for r in self.rows(name, project_id=project_id)]
            for name in TABLES
        }


def window_start(days: int, now: datetime | None = None) -> datetime:
    return (now or utcnow()) - timedelta(days=days)
