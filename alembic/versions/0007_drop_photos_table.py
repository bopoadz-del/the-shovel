"""drop photos table (architecture correction)

V1 spec wrongly put raw photo bytes on Render. Photos belong at the user's
source (where they uploaded them); Render only stores detection metadata
in photo_chunks. This migration drops the photos table on deploy; any
existing rows are deleted with it.

The companion code change removes:
  - POST /v1/admin/photo-bytes/{sha256}
  - GET /v1/photos/{sha256}
  - the rejected_no_bytes check inside POST /v1/admin/photo-import

Revision ID: 0007
Revises: 0006
Create Date: 2026-06-25
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # IF EXISTS guard so re-runs on a fresh DB (without 0006) don't fail.
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        bind.execute(sa.text("DROP TABLE IF EXISTS photos"))
    else:
        bind.execute(sa.text("DROP TABLE IF EXISTS photos"))


def downgrade() -> None:
    # Re-create the photos table at the 0006 shape so downgrade is faithful.
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.create_table(
            "photos",
            sa.Column("sha256", sa.Text(), primary_key=True),
            sa.Column("content_type", sa.Text(), nullable=False),
            sa.Column("size_bytes", sa.Integer(), nullable=False),
            sa.Column("bytes", postgresql.BYTEA(), nullable=False),
            sa.Column("uploaded_at", sa.TIMESTAMP(), nullable=False,
                      server_default=sa.text("CURRENT_TIMESTAMP")),
        )
    else:
        op.create_table(
            "photos",
            sa.Column("sha256", sa.Text(), primary_key=True),
            sa.Column("content_type", sa.Text(), nullable=False),
            sa.Column("size_bytes", sa.Integer(), nullable=False),
            sa.Column("bytes", sa.LargeBinary(), nullable=False),
            sa.Column("uploaded_at", sa.TIMESTAMP(), nullable=False,
                      server_default=sa.text("CURRENT_TIMESTAMP")),
        )
