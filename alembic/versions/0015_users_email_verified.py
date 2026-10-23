"""Add email_verified to users (signup email verification).

Open registration was enabled 2026-08-14, so anyone can sign up with any
address and nothing proves they own it. This column is the gate.

BACKFILL DIRECTION IS THE WHOLE POINT: existing rows get TRUE. They predate
the feature and were created by the operator (bootstrap admin) or by a
trusted path; defaulting them to FALSE would lock the only admins out of the
live service, and ``_bootstrap_first_user`` cannot rescue that because it
only seeds when NO users exist. New rows default FALSE.

Idempotent (matches 0013): SQLite dev/test DBs are created by the ORM, which
already carries the column, so this is a Postgres-only ALTER guarded with
IF NOT EXISTS.

Revision ID: 0015
Revises: 0014 (20260726_add_ingestion_jobs, which already claimed 0014)
"""
from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0015"
down_revision = "0014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        return
    # Added with DEFAULT FALSE so the column is NOT NULL-safe on legacy rows...
    op.execute(sa.text(
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS email_verified "
        "BOOLEAN NOT NULL DEFAULT FALSE"
    ))
    # ...then existing rows are flipped TRUE. Ordering matters: the ALTER has
    # to land first or there is nothing to update. Scoped to rows that exist
    # at migration time; anyone registering afterwards keeps the FALSE default.
    op.execute(sa.text("UPDATE users SET email_verified = TRUE"))


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        return
    op.execute(sa.text("ALTER TABLE users DROP COLUMN IF EXISTS email_verified"))
