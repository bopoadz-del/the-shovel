"""Add generated tsvector column + GIN index for hybrid full-text search.

Revision ID: 0003
Revises: 0002
Create Date: 2026-06-12

Postgres only: ``text_search`` is a STORED generated column over ``text``.
SQLite deployments no-op here — FTS5 is handled in the application layer.
"""

from __future__ import annotations

from alembic import op
from sqlalchemy import text

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def _is_postgres() -> bool:
    return op.get_bind().dialect.name == "postgresql"


def _chunks_text_search_exists() -> bool:
    row = op.get_bind().execute(
        text(
            """
            SELECT 1
            FROM pg_attribute a
            JOIN pg_class c ON a.attrelid = c.oid
            JOIN pg_namespace n ON c.relnamespace = n.oid
            WHERE n.nspname = 'public'
              AND c.relname = 'chunks'
              AND a.attname = 'text_search'
              AND NOT a.attisdropped
            """
        )
    ).fetchone()
    return row is not None


def _chunks_fts_gin_exists() -> bool:
    row = op.get_bind().execute(
        text(
            """
            SELECT 1
            FROM pg_indexes
            WHERE schemaname = 'public'
              AND tablename = 'chunks'
              AND indexname = 'chunks_fts_gin'
            """
        )
    ).fetchone()
    return row is not None


def upgrade() -> None:
    if not _is_postgres():
        return
    if not _chunks_text_search_exists():
        op.execute(
            text(
                """
                ALTER TABLE chunks
                ADD COLUMN text_search tsvector
                GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
                """
            )
        )
    if not _chunks_fts_gin_exists():
        op.execute(
            text(
                "CREATE INDEX chunks_fts_gin ON chunks USING GIN (text_search)"
            )
        )


def downgrade() -> None:
    if not _is_postgres():
        return
    if _chunks_fts_gin_exists():
        op.execute(text("DROP INDEX IF EXISTS chunks_fts_gin"))
    if _chunks_text_search_exists():
        op.execute(text("ALTER TABLE chunks DROP COLUMN IF EXISTS text_search"))
