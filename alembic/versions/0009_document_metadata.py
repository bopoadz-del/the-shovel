"""Add metadata JSONB column to documents.

Revision ID: 0009
Revises: 0008
Create Date: 2026-07-08

Stores Drive provenance and other source metadata per document.
Postgres uses JSONB; SQLite gets the column from the ORM bootstrap,
so this migration is a no-op there.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add metadata JSONB column if it is not already present.

    Migration 0001 applies the_fork_schema.sql, which already includes the
    metadata column on fresh PostgreSQL databases. This migration must be
    idempotent so it also succeeds on databases created from that baseline
    (e.g. CI) and on existing databases that have not yet received it.
    """
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        # SQLite dev/test databases are created by the ORM, which already
        # includes the metadata column. No Alembic-managed migration needed.
        return
    op.execute(sa.text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS metadata JSONB"))


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        return
    op.execute(sa.text("ALTER TABLE documents DROP COLUMN IF EXISTS metadata"))
