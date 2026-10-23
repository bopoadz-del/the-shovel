"""users: ``driver_mode`` flag (admin-set; driver mode for that user's turns)

With DRIVER_MODE=request, a user an admin has switched on takes driver
mode (F-DRIVER Phase B) on every turn, so it can be tried from the real UI
by one account without changing anyone else's turns. Off for every
existing and new user.

Revision ID: 0029
Revises: 0028
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0029"
down_revision = "0028"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if op.get_bind().dialect.name == "sqlite":
        return
    op.execute(sa.text("ALTER TABLE users ADD COLUMN IF NOT EXISTS driver_mode BOOLEAN NOT NULL DEFAULT FALSE"))


def downgrade() -> None:
    if op.get_bind().dialect.name == "sqlite":
        return
    op.execute(sa.text("ALTER TABLE users DROP COLUMN IF EXISTS driver_mode"))
