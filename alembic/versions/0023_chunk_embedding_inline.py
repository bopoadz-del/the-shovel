"""Chunk embeddings stay in the heap row (``STORAGE MAIN``), not in TOAST.

pgvector declares ``vector`` with EXTERNAL storage. A chunk row (text, the
generated ``text_search`` tsvector, the embedding) passes the ~2 KB TOAST
threshold, and the embedding is the column moved out of line. A distance
computed outside the HNSW index -- the exact ``ORDER BY embedding <=> q`` over
one project's rows that the planner picks for a small project -- then reads
each row's embedding through the TOAST index: measured ~6 buffers per row,
which made one search over 400 chunks touch 20 MB.

MAIN makes the toaster move ``text`` / ``text_search`` out of line first and
keep the vector inline. Metadata only: no rewrite, rows written before this
keep their old layout until they are rewritten. Covers every chunk table
present, like 0022; namespaced tables created later get the same setting from
``vector_store._ensure_schema``.

Revision ID: 0023
Revises: 0022
"""
from __future__ import annotations

from alembic import op
from sqlalchemy import text

revision = "0023"
down_revision = "0022"
branch_labels = None
depends_on = None


def _set_storage(mode: str) -> None:
    conn = op.get_bind()
    if conn.dialect.name != "postgresql":
        return
    # Every chunk table (legacy ``chunks`` and any ``chunks_<ns>``) whose
    # ``embedding`` is a pgvector column.
    tables = conn.execute(text(
        "SELECT c.relname FROM pg_class c "
        "JOIN pg_namespace n ON n.oid = c.relnamespace AND n.nspname = current_schema() "
        "JOIN pg_attribute a ON a.attrelid = c.oid AND a.attname = 'embedding' "
        "JOIN pg_type t ON t.oid = a.atttypid AND t.typname = 'vector' "
        "WHERE c.relkind = 'r' AND (c.relname = 'chunks' OR c.relname LIKE 'chunks!_%' ESCAPE '!')"
    )).scalars().all()
    for tbl in tables:
        conn.execute(text(f'ALTER TABLE "{tbl}" ALTER COLUMN embedding SET STORAGE {mode}'))


def upgrade() -> None:
    _set_storage("MAIN")


def downgrade() -> None:
    _set_storage("EXTERNAL")
