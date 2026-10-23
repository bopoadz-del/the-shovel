"""Align chunks.embedding with model2vec (256-dim production default).

Revision ID: 0002
Revises: 0001
Create Date: 2026-06-12

0001 created vector(384) for the legacy MiniLM path. The slim Render image
uses model2vec (minishlab/potion-base-8M) which emits 256-dim vectors.
pgvector 0.6 cannot truncate vectors in-place, so existing chunk rows are
deleted before the type change. Re-index project docs after cutover.
"""

from __future__ import annotations

from alembic import op
from sqlalchemy import text

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def _chunks_embedding_type() -> str | None:
    """Return formatted pg type for chunks.embedding (e.g. ``vector(256)``)."""
    row = op.get_bind().execute(
        text(
            """
            SELECT pg_catalog.format_type(a.atttypid, a.atttypmod)
            FROM pg_attribute a
            JOIN pg_class c ON a.attrelid = c.oid
            JOIN pg_namespace n ON c.relnamespace = n.oid
            WHERE n.nspname = 'public'
              AND c.relname = 'chunks'
              AND a.attname = 'embedding'
              AND NOT a.attisdropped
            """
        )
    ).fetchone()
    return str(row[0]) if row else None


def upgrade() -> None:
    if _chunks_embedding_type() == "vector(256)":
        return
    # Empty table avoids pgvector cast errors when shrinking 384 → 256.
    op.execute(text("DELETE FROM chunks"))
    op.execute(text("ALTER TABLE chunks ALTER COLUMN embedding TYPE vector(256)"))


def downgrade() -> None:
    if _chunks_embedding_type() == "vector(384)":
        return
    op.execute(text("DELETE FROM chunks"))
    op.execute(text("ALTER TABLE chunks ALTER COLUMN embedding TYPE vector(384)"))
