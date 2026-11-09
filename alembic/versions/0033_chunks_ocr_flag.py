"""OCR provenance on chunks: a chunk whose text came from OCR is marked.

Item 7 (advisor 06:12Z): scanned-PDF pages whose text layer is empty are
OCR'd at ingestion, and that text must carry an ``ocr=true`` provenance flag
so citations can tell OCR-derived text apart. This adds the ``ocr`` column to
every RAG chunk table (default false).

Like 0021, this alters every chunk table present -- the legacy ``chunks``
plus any ``chunks_<ns>`` (prod writes ``chunks_v2``) -- and is idempotent:
a table that already has the column (the 0001 unified schema ships ``ocr``
on ``chunks``) is left alone.

Revision ID: 0033
Revises: 0032
"""
from __future__ import annotations

from alembic import op
from sqlalchemy import inspect, text

revision = "0033"
down_revision = "0032"
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
        if "ocr" not in _columns(conn, tbl):
            conn.execute(text(
                f'ALTER TABLE "{tbl}" ADD COLUMN ocr BOOLEAN NOT NULL DEFAULT FALSE'
            ))


def downgrade() -> None:
    conn = op.get_bind()
    for tbl in _chunk_tables(conn):
        if "ocr" in _columns(conn, tbl):
            conn.execute(text(f'ALTER TABLE "{tbl}" DROP COLUMN ocr'))
