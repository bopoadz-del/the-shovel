"""drop photo_chunks table + FTS (V1 R4 detector retired)

The photo_chunks table held caption + photo_metadata rows derived from
the fine-tuned safety_qaqc_v1_r* detector. That detector was retired in
favour of a reparameterized YOLO-Worldv2 model whose detection results
flow through the chat composer's analyze-photo endpoint and never need
to be searchable as a corpus. The drive_archive RAG (text documents) is
untouched.

This migration drops:
  - photo_chunks (postgres + sqlite)
  - photo_chunks_fts virtual table + its triggers (sqlite only)

Revision ID: 0008
Revises: 0007
Create Date: 2026-06-25
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        # FTS5 triggers depend on photo_chunks -- drop them first, then the
        # virtual table, then the source table. IF EXISTS so re-runs are safe.
        for stmt in (
            "DROP TRIGGER IF EXISTS photo_chunks_au",
            "DROP TRIGGER IF EXISTS photo_chunks_ad",
            "DROP TRIGGER IF EXISTS photo_chunks_ai",
            "DROP TABLE IF EXISTS photo_chunks_fts",
            "DROP TABLE IF EXISTS photo_chunks",
        ):
            bind.execute(sa.text(stmt))
    else:
        bind.execute(sa.text("DROP TABLE IF EXISTS photo_chunks CASCADE"))


def downgrade() -> None:
    # Re-create the photo_chunks table at the 0006 shape so downgrade is
    # faithful. The FTS layer is intentionally NOT re-created -- ad-hoc
    # downgrade of an FTS5 setup with content/triggers is fragile and we
    # have no path that needs it.
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.create_table(
            "photo_chunks",
            sa.Column("chunk_id", sa.Text(), primary_key=True),
            sa.Column("project_id", sa.Text(), nullable=True),
            sa.Column("sha256", sa.Text(), nullable=False),
            sa.Column("caption", sa.Text(), nullable=False),
            sa.Column("photo_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
            sa.Column("created_at", sa.TIMESTAMP(), nullable=False,
                      server_default=sa.text("CURRENT_TIMESTAMP")),
            sa.UniqueConstraint("sha256", name="uq_photo_chunks_sha256"),
        )
    else:
        op.create_table(
            "photo_chunks",
            sa.Column("chunk_id", sa.Text(), primary_key=True),
            sa.Column("project_id", sa.Text(), nullable=True),
            sa.Column("sha256", sa.Text(), nullable=False),
            sa.Column("caption", sa.Text(), nullable=False),
            sa.Column("photo_metadata", sa.Text(), nullable=False),
            sa.Column("created_at", sa.TIMESTAMP(), nullable=False,
                      server_default=sa.text("CURRENT_TIMESTAMP")),
            sa.UniqueConstraint("sha256", name="uq_photo_chunks_sha256"),
        )
