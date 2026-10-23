"""photo_chunks + photos tables

Revision ID: 0006
Revises: 0005
Create Date: 2026-06-24
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name

    if dialect == "postgresql":
        op.create_table(
            "photo_chunks",
            sa.Column("chunk_id", sa.Text(), primary_key=True),
            sa.Column("project_id", sa.Text(), nullable=True),
            sa.Column("sha256", sa.Text(), nullable=False),
            sa.Column("caption", sa.Text(), nullable=False),
            sa.Column("photo_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
            sa.Column("created_at", sa.TIMESTAMP(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
            sa.UniqueConstraint("sha256", name="uq_photo_chunks_sha256"),
        )
        op.create_table(
            "photos",
            sa.Column("sha256", sa.Text(), primary_key=True),
            sa.Column("content_type", sa.Text(), nullable=False),
            sa.Column("size_bytes", sa.Integer(), nullable=False),
            sa.Column("bytes", postgresql.BYTEA(), nullable=False),
            sa.Column("uploaded_at", sa.TIMESTAMP(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        )
    else:
        op.create_table(
            "photo_chunks",
            sa.Column("chunk_id", sa.Text(), primary_key=True),
            sa.Column("project_id", sa.Text(), nullable=True),
            sa.Column("sha256", sa.Text(), nullable=False),
            sa.Column("caption", sa.Text(), nullable=False),
            sa.Column("photo_metadata", sa.Text(), nullable=False),
            sa.Column("created_at", sa.TIMESTAMP(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
            sa.UniqueConstraint("sha256", name="uq_photo_chunks_sha256"),
        )
        op.create_table(
            "photos",
            sa.Column("sha256", sa.Text(), primary_key=True),
            sa.Column("content_type", sa.Text(), nullable=False),
            sa.Column("size_bytes", sa.Integer(), nullable=False),
            sa.Column("bytes", sa.LargeBinary(), nullable=False),
            sa.Column("uploaded_at", sa.TIMESTAMP(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        )


def downgrade() -> None:
    op.drop_table("photos")
    op.drop_table("photo_chunks")
