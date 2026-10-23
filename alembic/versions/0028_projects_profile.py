"""projects: nullable ``profile`` column (the project's profile, JSON text)

Driver mode (F-DRIVER Phase B) gives the model the project's profile every
turn -- discipline, contract form, phase, governing codes and the like --
filled in once by an admin. Nullable: a project without one says so.

Idempotent and Postgres-only, like 0010: SQLite dev/test databases get the
column from the ORM or from projects.init_db's column patch.

Revision ID: 0028
Revises: 0027
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0028"
down_revision = "0027"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if op.get_bind().dialect.name == "sqlite":
        return
    op.execute(sa.text("ALTER TABLE projects ADD COLUMN IF NOT EXISTS profile TEXT"))


def downgrade() -> None:
    if op.get_bind().dialect.name == "sqlite":
        return
    op.execute(sa.text("ALTER TABLE projects DROP COLUMN IF EXISTS profile"))
