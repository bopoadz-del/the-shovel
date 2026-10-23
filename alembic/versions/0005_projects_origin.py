"""Add origin column to projects.

Revision ID: 0005
Revises: 0004
Create Date: 2026-06-22

PR B — admin-approved-projects: distinguish admin-curated Drive projects
from user-created projects. ``is_approved`` from 0004 cannot do this on
its own — the operator's spec assigns is_approved=true to both buckets
(user-created defaults to true; admin-approve-from-drive sets true).

The admin page needs to list only projects the admin actively approved
from a Drive folder cascade (not personal-account-style user-created rows).
A discriminator column makes that filter unambiguous.

Values:
  * ``user_create``         — default; what /v1/projects POST emits and
                              what existing rows backfill to.
  * ``admin_drive_approved``— set by /v1/admin/projects/approve-from-drive.
  * ``user_drive_import``   — reserved for PR C ("From Drive folder" in
                              the user-facing NewProjectModal).

Stored as TEXT rather than an enum so adding future origins (Aconex,
ftp, etc.) doesn't require another migration.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name

    if dialect == "sqlite":
        op.add_column(
            "projects",
            sa.Column(
                "origin",
                sa.String(length=32),
                nullable=False,
                server_default=sa.text("'user_create'"),
            ),
        )
    else:
        op.add_column(
            "projects",
            sa.Column("origin", sa.String(length=32), nullable=True),
        )
        op.execute(
            "UPDATE projects SET origin = 'user_create' WHERE origin IS NULL"
        )
        op.alter_column(
            "projects",
            "origin",
            existing_type=sa.String(length=32),
            nullable=False,
            server_default=sa.text("'user_create'"),
        )


def downgrade() -> None:
    op.drop_column("projects", "origin")
