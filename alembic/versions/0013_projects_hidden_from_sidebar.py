"""Add hidden_from_sidebar column to projects (layered RAG stage 5).

Lets a RAG corpus / general-knowledge / eval project be hidden from the user's
sidebar WITHOUT archiving it, so it stays fully retrievable. Distinct from
status='archived' (which also drops it from retrieval). Defaults false, so the
column is safe to add to every legacy row.

Idempotent (matches 0010): SQLite dev/test DBs are created by the ORM (which
already carries the column), so this is a Postgres-only ALTER guarded with
IF NOT EXISTS.

Revision ID: 0013
Revises: 0012
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0013"
down_revision = "0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        return
    op.execute(sa.text(
        "ALTER TABLE projects ADD COLUMN IF NOT EXISTS hidden_from_sidebar "
        "BOOLEAN NOT NULL DEFAULT FALSE"
    ))


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        return
    op.execute(sa.text("ALTER TABLE projects DROP COLUMN IF EXISTS hidden_from_sidebar"))
