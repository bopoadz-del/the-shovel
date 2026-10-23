"""chunks: nullable ``page`` column (source page a chunk's text starts on)

Citations showed "chunk #N" because a chunk carried no page. PDF extraction
now records where each page's text starts, and each chunk stores the 1-based
page of the source PDF on which its text begins. Nullable: non-PDF sources
and rows indexed before this revision have no page and keep the chunk label.

Like 0011, this alters every chunk table present -- the legacy ``chunks``
plus any ``chunks_<ns>`` (prod writes ``chunks_v2``) -- and is idempotent:
a table that already has the column is left alone.

Revision ID: 0021
Revises: 0020
"""
from __future__ import annotations

from alembic import op
from sqlalchemy import inspect, text

revision = "0021"
down_revision = "0020"
branch_labels = None
depends_on = None


def _chunk_tables(conn) -> list[str]:
    """Every RAG chunk table present: legacy ``chunks`` + ``chunks_<ns>``."""
    out = []
    for name in inspect(conn).get_table_names():
        if name != "chunks" and not name.startswith("chunks_"):
            continue
        # Only real chunk tables (SQLite FTS shadow tables share the prefix).
        if "chunk_index" in _columns(conn, name):
            out.append(name)
    return out


def _columns(conn, table: str) -> set[str]:
    return {c["name"] for c in inspect(conn).get_columns(table)}


def upgrade() -> None:
    conn = op.get_bind()
    for tbl in _chunk_tables(conn):
        if "page" not in _columns(conn, tbl):
            conn.execute(text(f'ALTER TABLE "{tbl}" ADD COLUMN page INTEGER'))


def downgrade() -> None:
    conn = op.get_bind()
    for tbl in _chunk_tables(conn):
        if "page" in _columns(conn, tbl):
            conn.execute(text(f'ALTER TABLE "{tbl}" DROP COLUMN page'))
