"""conversations: who owns the chat session.

A session is visible to its writer. Rows created before this column
exist stay NULL and remain visible only to the project row owner.

Revision ID: 0030
Revises: 0029
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0030"
down_revision = "0029"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if op.get_bind().dialect.name == "sqlite":
        return
    op.execute(sa.text(
        "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS owner_id TEXT"
    ))


def downgrade() -> None:
    if op.get_bind().dialect.name == "sqlite":
        return
    op.execute(sa.text(
        "ALTER TABLE conversations DROP COLUMN IF EXISTS owner_id"
    ))
