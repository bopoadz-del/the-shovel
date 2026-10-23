#!/usr/bin/env python3
"""Migrate legacy SQLite stores under DATA_DIR into PostgreSQL.

Reads eight legacy ``.db`` files:

* ``users.db``
* ``projects.db`` (projects, documents, project_facts, workflows)
* ``agent_memory.db`` (conversations, messages, agent_facts)
* ``doc_index.db``
* ``hydration.db``
* ``usage.db``
* ``rag/vectors.db`` (chunks — embeddings unpacked dynamically from BLOB length)
* ``rag/budget.db``

Rows are inserted in foreign-key order with ``ON CONFLICT DO NOTHING`` so the
script is idempotent. Orphan ``projects.user_id`` values that do not exist in
``users`` are remapped to ``system``.

Requires ``alembic upgrade head`` (or equivalent) on the target database before
a real migration.

Usage::

    export DATA_DIR=./data
    export DATABASE_URL=postgresql+psycopg://user:pass@localhost:5432/thefork
    python scripts/migrate_sqlite_to_pg.py --dry-run
    python scripts/migrate_sqlite_to_pg.py --execute
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import struct
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Generator, Sequence

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, Engine

# Canonical Postgres pgvector dimension (model2vec / app.core.models.EMBEDDING_DIM).
EMBEDDING_DIM = 256
SYSTEM_USER_ID = "system"

# Insert order respects PostgreSQL FK constraints in the_fork_schema.sql.
MIGRATION_TABLES: tuple[str, ...] = (
    "users",
    "projects",
    "documents",
    "project_facts",
    "workflows",
    "conversations",
    "messages",
    "agent_facts",
    "doc_index",
    "runs",
    "hydration_runs",
    "rag_budget",
    "chunks",
)


def _data_dir(explicit: str | None) -> Path:
    return Path(explicit or os.getenv("DATA_DIR", "./data")).resolve()


def _sqlite_path(data_dir: Path, *parts: str) -> Path:
    return data_dir.joinpath(*parts)


def _unified_db_path(data_dir: Path) -> Path:
    """Unified SQLAlchemy-local store (may coexist with legacy split *.db files)."""
    return data_dir / "the_fork.db"


@contextmanager
def _sqlite_conn(path: Path) -> Generator[sqlite3.Connection | None, None, None]:
    if not path.is_file():
        yield None
        return
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _fetch_rows(conn: sqlite3.Connection, table: str) -> list[sqlite3.Row]:
    if not _table_exists(conn, table):
        return []
    return list(conn.execute(f"SELECT * FROM {table}"))


def _json_value(raw: Any) -> Any:
    if raw is None:
        return None
    if isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="replace")
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


def _unpack_embedding(blob: bytes | None) -> list[float]:
    """Unpack a float32 BLOB and fit to the Postgres ``vector(EMBEDDING_DIM)`` column."""
    if not blob:
        raise ValueError("chunk embedding BLOB is empty")
    if len(blob) % 4 != 0:
        raise ValueError(
            f"embedding BLOB length {len(blob)} is not a multiple of 4 (float32)"
        )
    dim = len(blob) // 4
    values = list(struct.unpack(f"<{dim}f", blob))
    if dim > EMBEDDING_DIM:
        values = values[:EMBEDDING_DIM]
    elif dim < EMBEDDING_DIM:
        values.extend([0.0] * (EMBEDDING_DIM - dim))
    return values


def _vector_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(f"{v:.8g}" for v in values) + "]"


def _existing_user_ids(pg: Connection) -> set[str]:
    rows = pg.execute(text("SELECT id FROM users")).fetchall()
    return {str(r[0]) for r in rows}


def _existing_users_by_email(pg: Connection) -> dict[str, str]:
    """Map lowercased email → canonical Postgres user id."""
    rows = pg.execute(text("SELECT id, email FROM users")).fetchall()
    return {str(email).strip().lower(): str(uid) for uid, email in rows}


def _resolve_user_id(
    user_id: str | None,
    known_users: set[str],
    user_id_remap: dict[str, str],
) -> str | None:
    if not user_id:
        return None
    uid = user_id_remap.get(str(user_id), str(user_id))
    if uid not in known_users:
        return None
    return uid


def _existing_project_ids(pg: Connection) -> set[str]:
    rows = pg.execute(text("SELECT id FROM projects")).fetchall()
    return {str(r[0]) for r in rows}


def _existing_document_ids(pg: Connection) -> set[str]:
    rows = pg.execute(text("SELECT id FROM documents")).fetchall()
    return {str(r[0]) for r in rows}


def _user_rows_from_sqlite(data_dir: Path) -> list[sqlite3.Row]:
    seen: dict[str, sqlite3.Row] = {}
    for path in (_sqlite_path(data_dir, "users.db"), _unified_db_path(data_dir)):
        with _sqlite_conn(path) as conn:
            if conn is None:
                continue
            for row in _fetch_rows(conn, "users"):
                seen[str(row["id"])] = row
    return list(seen.values())


def _migrate_users(
    pg: Connection,
    data_dir: Path,
    *,
    dry_run: bool,
    counts: dict[str, int],
    known_users: set[str],
    user_id_remap: dict[str, str],
) -> None:
    rows = _user_rows_from_sqlite(data_dir)
    if not rows:
        counts["users"] = 0
        return
    existing_by_email = _existing_users_by_email(pg)
    inserted = 0
    for row in rows:
        sqlite_id = str(row["id"])
        email_key = str(row["email"]).strip().lower()
        canonical_id = existing_by_email.get(email_key)
        if canonical_id and canonical_id != sqlite_id:
            # App may have created the same email under a different id (bootstrap).
            user_id_remap[sqlite_id] = canonical_id
            known_users.add(sqlite_id)
            known_users.add(canonical_id)
            if dry_run:
                inserted += 1
            continue
        known_users.add(sqlite_id)
        params = {
            "id": row["id"],
            "email": row["email"],
            "password_hash": row["password_hash"],
            "salt": row["salt"],
            "display_name": row["display_name"],
            "role": row["role"] or "user",
            "created_at": row["created_at"],
        }
        if dry_run:
            inserted += 1
            continue
        result = pg.execute(
            text(
                """
                INSERT INTO users (
                    id, email, password_hash, salt, display_name, role, created_at
                ) VALUES (
                    :id, :email, :password_hash, :salt, :display_name, :role, :created_at
                )
                ON CONFLICT (id) DO NOTHING
                """
            ),
            params,
        )
        if result.rowcount:
            inserted += 1
            existing_by_email[email_key] = sqlite_id
    counts["users"] = inserted


def _project_rows_from_sqlite(data_dir: Path) -> list[sqlite3.Row]:
    """Collect projects from legacy ``projects.db`` and unified ``the_fork.db``."""
    seen: dict[str, sqlite3.Row] = {}
    for path in (_sqlite_path(data_dir, "projects.db"), _unified_db_path(data_dir)):
        with _sqlite_conn(path) as conn:
            if conn is None:
                continue
            for row in _fetch_rows(conn, "projects"):
                seen[str(row["id"])] = row
    return list(seen.values())


def _migrate_projects(
    pg: Connection,
    data_dir: Path,
    *,
    dry_run: bool,
    counts: dict[str, int],
    known_users: set[str],
    user_id_remap: dict[str, str],
) -> None:
    rows = _project_rows_from_sqlite(data_dir)
    if not rows:
        counts["projects"] = 0
        return
    inserted = 0
    for row in rows:
        user_id = row["user_id"] if "user_id" in row.keys() else SYSTEM_USER_ID
        user_id = user_id or SYSTEM_USER_ID
        resolved = _resolve_user_id(str(user_id), known_users, user_id_remap)
        user_id = resolved if resolved is not None else SYSTEM_USER_ID
        params = {
            "id": row["id"],
            "name": row["name"],
            "client": row["client"],
            "status": row["status"] or "active",
            "aconex_connected": bool(row["aconex_connected"]),
            "user_id": user_id,
            "created_at": row["created_at"],
        }
        if dry_run:
            inserted += 1
            continue
        result = pg.execute(
            text(
                """
                INSERT INTO projects (
                    id, name, client, status, aconex_connected, user_id, created_at
                ) VALUES (
                    :id, :name, :client, :status, :aconex_connected, :user_id, :created_at
                )
                ON CONFLICT (id) DO NOTHING
                """
            ),
            params,
        )
        if result.rowcount:
            inserted += 1
    counts["projects"] = inserted


def _document_rows_from_sqlite(data_dir: Path) -> list[sqlite3.Row]:
    """Collect documents from legacy ``projects.db`` and unified ``the_fork.db``."""
    seen: dict[str, sqlite3.Row] = {}
    for path in (_sqlite_path(data_dir, "projects.db"), _unified_db_path(data_dir)):
        with _sqlite_conn(path) as conn:
            if conn is None:
                continue
            for row in _fetch_rows(conn, "documents"):
                seen[str(row["id"])] = row
    return list(seen.values())


def _migrate_documents(
    pg: Connection,
    data_dir: Path,
    *,
    dry_run: bool,
    counts: dict[str, int],
    **_kwargs: Any,
) -> None:
    rows = _document_rows_from_sqlite(data_dir)
    if not rows:
        counts["documents"] = 0
        return
    if dry_run:
        known_projects = {str(r["id"]) for r in _project_rows_from_sqlite(data_dir)}
    else:
        known_projects = _existing_project_ids(pg)
    inserted = 0
    for row in rows:
        project_id = row["project_id"]
        if project_id and str(project_id) not in known_projects:
            continue
        params = {
            "id": row["id"],
            "project_id": row["project_id"],
            "original_name": row["original_name"],
            "stored_as": row["stored_as"],
            "file_path": row["file_path"],
            "doc_type": row["doc_type"] or "document",
            "doc_role": row["doc_role"] or "other",
            "size": int(row["size"] or 0),
            "uploaded_at": row["uploaded_at"],
            "content_sha256": row["content_sha256"]
            if "content_sha256" in row.keys()
            else None,
        }
        if dry_run:
            inserted += 1
            continue
        result = pg.execute(
            text(
                """
                INSERT INTO documents (
                    id, project_id, original_name, stored_as, file_path,
                    doc_type, doc_role, size, uploaded_at, content_sha256
                ) VALUES (
                    :id, :project_id, :original_name, :stored_as, :file_path,
                    :doc_type, :doc_role, :size, :uploaded_at, :content_sha256
                )
                ON CONFLICT (id) DO NOTHING
                """
            ),
            params,
        )
        if result.rowcount:
            inserted += 1
    counts["documents"] = inserted


def _migrate_project_facts(
    pg: Connection,
    data_dir: Path,
    *,
    dry_run: bool,
    counts: dict[str, int],
    **_kwargs: Any,
) -> None:
    path = _sqlite_path(data_dir, "projects.db")
    with _sqlite_conn(path) as conn:
        if conn is None:
            counts["project_facts"] = 0
            return
        rows = _fetch_rows(conn, "project_facts")
        inserted = 0
        for row in rows:
            params = {
                "id": row["id"],
                "project_id": row["project_id"],
                "key": row["key"],
                "value": row["value"],
                "source_document": row["source_document"],
                "confidence": row["confidence"],
                "updated_at": row["updated_at"],
            }
            if dry_run:
                inserted += 1
                continue
            result = pg.execute(
                text(
                    """
                    INSERT INTO project_facts (
                        id, project_id, key, value, source_document, confidence, updated_at
                    ) VALUES (
                        :id, :project_id, :key, :value, :source_document, :confidence, :updated_at
                    )
                    ON CONFLICT (id) DO NOTHING
                    """
                ),
                params,
            )
            if result.rowcount:
                inserted += 1
        counts["project_facts"] = inserted


def _workflow_rows_from_sqlite(data_dir: Path) -> list[sqlite3.Row]:
    seen: dict[str, sqlite3.Row] = {}
    for path in (_sqlite_path(data_dir, "projects.db"), _unified_db_path(data_dir)):
        with _sqlite_conn(path) as conn:
            if conn is None:
                continue
            for row in _fetch_rows(conn, "workflows"):
                seen[str(row["id"])] = row
    return list(seen.values())


def _migrate_workflows(
    pg: Connection,
    data_dir: Path,
    *,
    dry_run: bool,
    counts: dict[str, int],
    known_users: set[str],
    user_id_remap: dict[str, str],
    **_kwargs: Any,
) -> None:
    rows = _workflow_rows_from_sqlite(data_dir)
    if not rows:
        counts["workflows"] = 0
        return
    known_projects = _existing_project_ids(pg)
    inserted = 0
    for row in rows:
        project_id = row["project_id"]
        if project_id and str(project_id) not in known_projects:
            continue
        owner_id = row["owner_id"] if "owner_id" in row.keys() else None
        owner_id = _resolve_user_id(str(owner_id) if owner_id else None, known_users, user_id_remap)
        steps = _json_value(row["steps"])
        if not isinstance(steps, list):
            steps = []
        params = {
            "id": row["id"],
            "name": row["name"],
            "project_id": project_id,
            "owner_id": owner_id,
            "steps": json.dumps(steps),
            "created_at": row["created_at"],
        }
        if dry_run:
            inserted += 1
            continue
        result = pg.execute(
            text(
                """
                INSERT INTO workflows (
                    id, name, project_id, owner_id, steps, created_at
                ) VALUES (
                    :id, :name, :project_id, :owner_id, CAST(:steps AS JSONB), :created_at
                )
                ON CONFLICT (id) DO NOTHING
                """
            ),
            params,
        )
        if result.rowcount:
            inserted += 1
    counts["workflows"] = inserted


def _migrate_conversations(
    pg: Connection,
    data_dir: Path,
    *,
    dry_run: bool,
    counts: dict[str, int],
    **_kwargs: Any,
) -> None:
    path = _sqlite_path(data_dir, "agent_memory.db")
    with _sqlite_conn(path) as conn:
        if conn is None:
            counts["conversations"] = 0
            return
        rows = _fetch_rows(conn, "conversations")
        inserted = 0
        for row in rows:
            params = {
                "id": row["id"],
                "agent_name": row["agent_name"],
                "project_id": row["project_id"],
                "title": row["title"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            if dry_run:
                inserted += 1
                continue
            result = pg.execute(
                text(
                    """
                    INSERT INTO conversations (
                        id, agent_name, project_id, title, created_at, updated_at
                    ) VALUES (
                        :id, :agent_name, :project_id, :title, :created_at, :updated_at
                    )
                    ON CONFLICT (id) DO NOTHING
                    """
                ),
                params,
            )
            if result.rowcount:
                inserted += 1
        counts["conversations"] = inserted


def _migrate_messages(
    pg: Connection,
    data_dir: Path,
    *,
    dry_run: bool,
    counts: dict[str, int],
    **_kwargs: Any,
) -> None:
    path = _sqlite_path(data_dir, "agent_memory.db")
    with _sqlite_conn(path) as conn:
        if conn is None:
            counts["messages"] = 0
            return
        rows = _fetch_rows(conn, "messages")
        inserted = 0
        for row in rows:
            params = {
                "id": row["id"],
                "conversation_id": row["conversation_id"],
                "role": row["role"],
                "content": row["content"],
                "created_at": row["created_at"],
            }
            if dry_run:
                inserted += 1
                continue
            result = pg.execute(
                text(
                    """
                    INSERT INTO messages (
                        id, conversation_id, role, content, created_at
                    ) VALUES (
                        :id, :conversation_id, :role, :content, :created_at
                    )
                    ON CONFLICT (id) DO NOTHING
                    """
                ),
                params,
            )
            if result.rowcount:
                inserted += 1
        counts["messages"] = inserted


def _migrate_agent_facts(
    pg: Connection,
    data_dir: Path,
    *,
    dry_run: bool,
    counts: dict[str, int],
    **_kwargs: Any,
) -> None:
    path = _sqlite_path(data_dir, "agent_memory.db")
    with _sqlite_conn(path) as conn:
        if conn is None:
            counts["agent_facts"] = 0
            return
        rows = _fetch_rows(conn, "agent_facts")
        inserted = 0
        for row in rows:
            project_id = row["project_id"] if "project_id" in row.keys() else ""
            if project_id is None:
                project_id = ""
            params = {
                "id": row["id"],
                "agent_name": row["agent_name"],
                "project_id": project_id,
                "conversation_id": row["conversation_id"],
                "key": row["key"],
                "value": row["value"],
                "updated_at": row["updated_at"],
            }
            if dry_run:
                inserted += 1
                continue
            result = pg.execute(
                text(
                    """
                    INSERT INTO agent_facts (
                        id, agent_name, project_id, conversation_id, key, value, updated_at
                    ) VALUES (
                        :id, :agent_name, :project_id, :conversation_id, :key, :value, :updated_at
                    )
                    ON CONFLICT (id) DO NOTHING
                    """
                ),
                params,
            )
            if result.rowcount:
                inserted += 1
        counts["agent_facts"] = inserted


def _migrate_doc_index(
    pg: Connection,
    data_dir: Path,
    *,
    dry_run: bool,
    counts: dict[str, int],
    **_kwargs: Any,
) -> None:
    path = _sqlite_path(data_dir, "doc_index.db")
    with _sqlite_conn(path) as conn:
        if conn is None:
            counts["doc_index"] = 0
            return
        rows = _fetch_rows(conn, "doc_index")
        inserted = 0
        for row in rows:
            index_json = _json_value(row["index_json"])
            if not isinstance(index_json, dict):
                continue
            params = {
                "project_id": row["project_id"],
                "index_json": json.dumps(index_json),
                "updated_at": row["updated_at"],
            }
            if dry_run:
                inserted += 1
                continue
            result = pg.execute(
                text(
                    """
                    INSERT INTO doc_index (project_id, index_json, updated_at)
                    VALUES (:project_id, CAST(:index_json AS JSONB), :updated_at)
                    ON CONFLICT (project_id) DO NOTHING
                    """
                ),
                params,
            )
            if result.rowcount:
                inserted += 1
        counts["doc_index"] = inserted


def _migrate_runs(
    pg: Connection,
    data_dir: Path,
    *,
    dry_run: bool,
    counts: dict[str, int],
    **_kwargs: Any,
) -> None:
    path = _sqlite_path(data_dir, "usage.db")
    with _sqlite_conn(path) as conn:
        if conn is None:
            counts["runs"] = 0
            return
        rows = _fetch_rows(conn, "runs")
        inserted = 0
        for row in rows:
            params = {
                "id": row["id"],
                "user_id": row["user_id"],
                "agent_name": row["agent_name"],
                "provider": row["provider"],
                "model": row["model"],
                "prompt_tokens": row["prompt_tokens"],
                "completion_tokens": row["completion_tokens"],
                "total_tokens": row["total_tokens"],
                "estimated_cost_usd": row["estimated_cost_usd"],
                "created_at": row["created_at"],
            }
            if dry_run:
                inserted += 1
                continue
            result = pg.execute(
                text(
                    """
                    INSERT INTO runs (
                        id, user_id, agent_name, provider, model,
                        prompt_tokens, completion_tokens, total_tokens,
                        estimated_cost_usd, created_at
                    ) VALUES (
                        :id, :user_id, :agent_name, :provider, :model,
                        :prompt_tokens, :completion_tokens, :total_tokens,
                        :estimated_cost_usd, :created_at
                    )
                    ON CONFLICT (id) DO NOTHING
                    """
                ),
                params,
            )
            if result.rowcount:
                inserted += 1
        counts["runs"] = inserted


def _migrate_hydration_runs(
    pg: Connection,
    data_dir: Path,
    *,
    dry_run: bool,
    counts: dict[str, int],
    **_kwargs: Any,
) -> None:
    path = _sqlite_path(data_dir, "hydration.db")
    with _sqlite_conn(path) as conn:
        if conn is None:
            counts["hydration_runs"] = 0
            return
        rows = _fetch_rows(conn, "hydration_runs")
        inserted = 0
        for row in rows:
            facts = _json_value(row["facts_json"])
            if not isinstance(facts, dict):
                facts = {}
            params = {
                "id": row["id"],
                "run_date": row["run_date"],
                "scope": row["scope"],
                "project_id": row["project_id"],
                "summary_md": row["summary_md"],
                "facts_json": json.dumps(facts),
                "provider": row["provider"],
                "created_at": row["created_at"],
            }
            if dry_run:
                inserted += 1
                continue
            result = pg.execute(
                text(
                    """
                    INSERT INTO hydration_runs (
                        id, run_date, scope, project_id, summary_md, facts_json, provider, created_at
                    ) VALUES (
                        :id, :run_date, :scope, :project_id, :summary_md,
                        CAST(:facts_json AS JSONB), :provider, :created_at
                    )
                    ON CONFLICT (id) DO NOTHING
                    """
                ),
                params,
            )
            if result.rowcount:
                inserted += 1
        counts["hydration_runs"] = inserted


def _migrate_rag_budget(
    pg: Connection,
    data_dir: Path,
    *,
    dry_run: bool,
    counts: dict[str, int],
    **_kwargs: Any,
) -> None:
    path = _sqlite_path(data_dir, "rag", "budget.db")
    with _sqlite_conn(path) as conn:
        if conn is None:
            counts["rag_budget"] = 0
            return
        rows = _fetch_rows(conn, "rag_budget")
        inserted = 0
        for row in rows:
            params = {
                "day": row["day"],
                "consumed": int(row["consumed"] or 0),
            }
            if dry_run:
                inserted += 1
                continue
            result = pg.execute(
                text(
                    """
                    INSERT INTO rag_budget (day, consumed)
                    VALUES (:day, :consumed)
                    ON CONFLICT (day) DO NOTHING
                    """
                ),
                params,
            )
            if result.rowcount:
                inserted += 1
        counts["rag_budget"] = inserted


def _migrate_chunks(
    pg: Connection,
    data_dir: Path,
    *,
    dry_run: bool,
    counts: dict[str, int],
    **_kwargs: Any,
) -> None:
    path = _sqlite_path(data_dir, "rag", "vectors.db")
    with _sqlite_conn(path) as conn:
        if conn is None:
            counts["chunks"] = 0
            return
        rows = _fetch_rows(conn, "chunks")
        if dry_run:
            known_docs = {str(r["id"]) for r in _document_rows_from_sqlite(data_dir)}
        else:
            known_docs = _existing_document_ids(pg)
        inserted = 0
        for row in rows:
            doc_id = row["doc_id"]
            if doc_id and str(doc_id) not in known_docs:
                continue
            embedding = _unpack_embedding(row["embedding"])
            params = {
                "chunk_id": row["chunk_id"],
                "project_id": row["project_id"],
                "doc_id": doc_id,
                "chunk_index": int(row["chunk_index"]),
                "text": row["text"],
                "embedding": _vector_literal(embedding),
                "created_at": row["created_at"],
            }
            if dry_run:
                inserted += 1
                continue
            result = pg.execute(
                text(
                    """
                    INSERT INTO chunks (
                        chunk_id, project_id, doc_id, chunk_index, text, embedding, created_at
                    ) VALUES (
                        :chunk_id, :project_id, :doc_id, :chunk_index, :text,
                        CAST(:embedding AS vector), :created_at
                    )
                    ON CONFLICT (chunk_id) DO NOTHING
                    """
                ),
                params,
            )
            if result.rowcount:
                inserted += 1
        counts["chunks"] = inserted


Migrator = Callable[..., None]

MIGRATORS: dict[str, Migrator] = {
    "users": _migrate_users,
    "projects": _migrate_projects,
    "documents": _migrate_documents,
    "project_facts": _migrate_project_facts,
    "workflows": _migrate_workflows,
    "conversations": _migrate_conversations,
    "messages": _migrate_messages,
    "agent_facts": _migrate_agent_facts,
    "doc_index": _migrate_doc_index,
    "runs": _migrate_runs,
    "hydration_runs": _migrate_hydration_runs,
    "rag_budget": _migrate_rag_budget,
    "chunks": _migrate_chunks,
}


def migrate(
    engine: Engine,
    data_dir: Path,
    *,
    dry_run: bool,
) -> dict[str, int]:
    """Run all table migrators in FK order. Returns per-table row counts."""
    counts: dict[str, int] = {table: 0 for table in MIGRATION_TABLES}
    known_users: set[str] = set()
    user_id_remap: dict[str, str] = {}

    with engine.begin() as pg:
        known_users.update(_existing_user_ids(pg))
        known_users.add(SYSTEM_USER_ID)

        for table in MIGRATION_TABLES:
            migrator = MIGRATORS[table]
            migrator(
                pg,
                data_dir,
                dry_run=dry_run,
                counts=counts,
                known_users=known_users,
                user_id_remap=user_id_remap,
            )
            if dry_run:
                continue
            # Refresh user ids after users migration for downstream orphan checks.
            if table == "users":
                known_users.update(_existing_user_ids(pg))
                known_users.add(SYSTEM_USER_ID)

    return counts


def _print_counts(counts: dict[str, int], *, dry_run: bool) -> None:
    label = "would migrate" if dry_run else "inserted"
    print(f"Migration summary ({label}):")
    for table in MIGRATION_TABLES:
        print(f"  {table}: {counts.get(table, 0)}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate legacy SQLite stores under DATA_DIR to PostgreSQL."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Count rows that would be migrated without writing to PostgreSQL.",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Perform the migration (ON CONFLICT DO NOTHING).",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Legacy DATA_DIR (default: $DATA_DIR or ./data).",
    )
    parser.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL"),
        help="Target PostgreSQL URL (default: $DATABASE_URL).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    database_url = args.database_url
    if not database_url:
        print("DATABASE_URL is required (env or --database-url).", file=sys.stderr)
        return 1
    if not database_url.startswith("postgresql"):
        print("Target DATABASE_URL must be a PostgreSQL URL.", file=sys.stderr)
        return 1

    data_dir = _data_dir(args.data_dir)
    if not data_dir.is_dir():
        print(f"DATA_DIR does not exist: {data_dir}", file=sys.stderr)
        return 1

    engine = create_engine(database_url, pool_pre_ping=True)
    dry_run = bool(args.dry_run)
    counts = migrate(engine, data_dir, dry_run=dry_run)
    _print_counts(counts, dry_run=dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
