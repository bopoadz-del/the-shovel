"""Add location column to projects.

Revision ID: 0010
Revises: 0009
Create Date: 2026-07-13

F4 / daily_site_report weather: a project carries a confirmed site location so the
daily site report's weather is read from project metadata — NEVER scraped from
the chat message (a mis-parsed token would put wrong-city weather in a formal
report). Nullable: a project without a set location falls back to the honest
"no location — weather not included" behaviour, never a fabricated reading.

Idempotent (matches 0009): SQLite dev/test databases are created by the ORM,
which already carries the column, so this is a Postgres-only ALTER guarded with
IF NOT EXISTS to also succeed on fresh baseline databases.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        return
    op.execute(sa.text("ALTER TABLE projects ADD COLUMN IF NOT EXISTS location VARCHAR"))


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        return
    op.execute(sa.text("ALTER TABLE projects DROP COLUMN IF EXISTS location"))
