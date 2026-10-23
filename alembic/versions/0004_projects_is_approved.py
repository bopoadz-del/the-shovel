"""Add is_approved column to projects.

Revision ID: 0004
Revises: 0003
Create Date: 2026-06-22

Operator's admin-approved-projects architecture (PR A):
  * Admin-created projects (via /v1/admin/projects/approve-from-drive)
    start with is_approved=true.
  * User-created projects (via /v1/projects POST) also start true —
    user-created is implicitly admin-approved for the user's own profile.
  * Future "detected but not yet approved" candidates will be created
    with is_approved=false; for now no endpoint creates them, but the
    column is in place so the admin UI can filter by approved status.

Backfill: every existing row → true (preserves current behaviour where
all projects are listed).

Backwards compatibility: column has a server-side default of true so
INSERTs that don't mention it (older code paths) still produce
approved projects. Application code passes the value explicitly going
forward.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add is_approved column with default true; backfill existing rows."""
    bind = op.get_bind()
    dialect = bind.dialect.name

    if dialect == "sqlite":
        # SQLite ALTER TABLE ADD COLUMN supports a constant DEFAULT;
        # the default applies to existing rows AND new INSERTs that
        # omit the column. NOT NULL with default = clean migration.
        op.add_column(
            "projects",
            sa.Column(
                "is_approved",
                sa.Boolean(),
                nullable=False,
                server_default=sa.true(),
            ),
        )
    else:
        # Postgres: add as nullable first, backfill, then enforce NOT NULL.
        # This pattern works on large tables too — no full-table lock with
        # the default applied in one shot.
        op.add_column(
            "projects",
            sa.Column("is_approved", sa.Boolean(), nullable=True),
        )
        op.execute("UPDATE projects SET is_approved = TRUE WHERE is_approved IS NULL")
        op.alter_column(
            "projects",
            "is_approved",
            existing_type=sa.Boolean(),
            nullable=False,
            server_default=sa.true(),
        )


def downgrade() -> None:
    op.drop_column("projects", "is_approved")
