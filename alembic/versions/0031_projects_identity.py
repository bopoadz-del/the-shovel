"""projects: nullable ``identity`` column (the project's identity set, JSON text)

Order 2210Z: a project's names, abbreviations, codes, contract numbers and
site names are derived from its own record and documents, stored with the
project, and replaced by a neutral project id wherever an answer, citation or
Sources line leaves (app.core.project_identity). Nullable: written when the record, profile or documents change.

Idempotent and Postgres-only, like 0028: SQLite dev/test databases get the
column from the ORM or from projects.init_db's column patch.

Revision ID: 0031
Revises: 0030
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0031"
down_revision = "0030"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if op.get_bind().dialect.name == "sqlite":
        return
    op.execute(sa.text("ALTER TABLE projects ADD COLUMN IF NOT EXISTS identity TEXT"))


def downgrade() -> None:
    if op.get_bind().dialect.name == "sqlite":
        return
    op.execute(sa.text("ALTER TABLE projects DROP COLUMN IF EXISTS identity"))
