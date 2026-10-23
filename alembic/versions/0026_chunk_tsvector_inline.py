"""Chunk rows stay whole: BM25 ranking reads the tsvector from the heap row.

``ts_rank`` reads each candidate's ``text_search`` tsvector. Once the
embedding stayed inline (0023), an ordinary chunk row passed the default
~2 KB TOAST threshold and the tsvector was the column moved out of line, so
ranking ~1,000 candidates cost ~3,800 buffers instead of ~880 (measured).

Raises each chunk table's ``toast_tuple_target`` to 4080 so an ordinary row
stays whole, and sets ``text_search`` to STORAGE MAIN so an oversized row
moves its ``text`` out first. Metadata only: rows written earlier keep their
layout until rewritten. Namespaced tables created later get the same from
``vector_store._ensure_schema``.

Revision ID: 0026
Revises: 0025
"""
from __future__ import annotations

from alembic import op
from sqlalchemy import text

revision = "0026"
down_revision = "0025"
branch_labels = None
depends_on = None

_TARGET = 4080


def _chunk_tables(conn) -> list:
    return conn.execute(text(
        "SELECT c.relname FROM pg_class c "
        "JOIN pg_namespace n ON n.oid = c.relnamespace AND n.nspname = current_schema() "
        "JOIN pg_attribute a ON a.attrelid = c.oid AND a.attname = 'text_search' "
        "WHERE c.relkind = 'r' AND (c.relname = 'chunks' OR c.relname LIKE 'chunks!_%' ESCAPE '!')"
    )).scalars().all()


def upgrade() -> None:
    conn = op.get_bind()
    if conn.dialect.name != "postgresql":
        return
    for tbl in _chunk_tables(conn):
        conn.execute(text(f'ALTER TABLE "{tbl}" SET (toast_tuple_target = {_TARGET})'))
        conn.execute(text(f'ALTER TABLE "{tbl}" ALTER COLUMN text_search SET STORAGE MAIN'))


def downgrade() -> None:
    conn = op.get_bind()
    if conn.dialect.name != "postgresql":
        return
    for tbl in _chunk_tables(conn):
        conn.execute(text(f'ALTER TABLE "{tbl}" RESET (toast_tuple_target)'))
        conn.execute(text(f'ALTER TABLE "{tbl}" ALTER COLUMN text_search SET STORAGE EXTENDED'))
