"""Trigram indexes for the answer path's substring reads.

The answer path matches substrings with ``LOWER(...) LIKE '%needle%'`` in two
places, and neither had an index that can serve a leading-wildcard LIKE:

* chunk text -- ``chunks_containing_all`` / ``identifier_search`` (labelled-
  row, asked-quantity and term recall). The only usable index was the
  ``project_id`` btree, so a needle that is rare (the usual case: it is a
  phrase from the question) read the whole project's text to find a handful
  of rows. Measured on live: ~48k buffers (~380 MB) per call on the master
  corpus, several calls per question.
* document names -- ``documents_matching_filename_terms`` /
  ``documents_matching_title_phrase`` (titled-document and filename recall),
  a sequential scan of ``documents`` per call, ~10 calls per question.

``pg_trgm`` GIN indexes on exactly the expressions those queries filter on
let the planner find the matching rows from the index. Results are
unchanged -- an index is only an access path.

Like 0011/0021 this covers every chunk table present (legacy ``chunks`` and
any ``chunks_<ns>``; prod reads ``chunks_v2``). Namespaced tables created
later get the same index from ``vector_store._ensure_schema``. Idempotent.

Revision ID: 0022
Revises: 0021
"""
from __future__ import annotations

from alembic import op
from sqlalchemy import inspect, text

revision = "0022"
down_revision = "0021"
branch_labels = None
depends_on = None

_DOCUMENT_INDEXES = {
    "idx_documents_name_trgm": "lower(original_name)",
    "idx_documents_path_trgm": "lower(file_path)",
}


def _chunk_tables(conn) -> list[str]:
    """Every RAG chunk table present: legacy ``chunks`` + ``chunks_<ns>``."""
    out = []
    for name in inspect(conn).get_table_names():
        if name != "chunks" and not name.startswith("chunks_"):
            continue
        cols = {c["name"] for c in inspect(conn).get_columns(name)}
        # Only real chunk tables (SQLite FTS shadow tables share the prefix).
        if "chunk_index" in cols and "text" in cols:
            out.append(name)
    return out


def upgrade() -> None:
    conn = op.get_bind()
    if conn.dialect.name != "postgresql":
        # SQLite dev/test DBs scan small tables; LIKE needs no index there.
        return
    try:
        with conn.begin_nested():
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
    except Exception as exc:  # noqa: BLE001 — an access path must not block a deploy
        # pg_trgm is a trusted contrib extension, so this only fails on a
        # server built without contrib. The queries still answer correctly
        # without the index -- they read more -- so say so and carry on.
        print(f"0022: pg_trgm unavailable ({exc}); trigram indexes not created")
        return
    wanted = [
        (f"{tbl}_text_trgm", tbl, "lower(text)") for tbl in _chunk_tables(conn)
    ]
    if "documents" in inspect(conn).get_table_names():
        wanted += [(name, "documents", expr) for name, expr in _DOCUMENT_INDEXES.items()]
    # CONCURRENTLY, outside the migration transaction: the chunk table keeps
    # taking writes (ingest) while its index builds. A concurrent build that
    # died leaves an INVALID index that IF NOT EXISTS would skip forever, so
    # one of those is dropped and rebuilt.
    with op.get_context().autocommit_block():
        for name, tbl, expr in wanted:
            invalid = conn.execute(text(
                "SELECT 1 FROM pg_class c JOIN pg_index i ON i.indexrelid = c.oid "
                "WHERE c.relname = :n AND NOT i.indisvalid"
            ), {"n": name}).first()
            if invalid is not None:
                conn.execute(text(f'DROP INDEX CONCURRENTLY IF EXISTS "{name}"'))
            conn.execute(text(
                f'CREATE INDEX CONCURRENTLY IF NOT EXISTS "{name}" '
                f'ON "{tbl}" USING gin ({expr} gin_trgm_ops)'
            ))


def downgrade() -> None:
    conn = op.get_bind()
    if conn.dialect.name != "postgresql":
        return
    for name in _DOCUMENT_INDEXES:
        conn.execute(text(f"DROP INDEX IF EXISTS {name}"))
    for tbl in _chunk_tables(conn):
        conn.execute(text(f'DROP INDEX IF EXISTS "{tbl}_text_trgm"'))
