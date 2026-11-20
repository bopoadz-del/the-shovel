"""Folder roles on documents (RQ-3): widen the doc_role vocabulary.

A document's role can now come from the folder-name table in
``app.core.doc_typing`` (contract, draft, tender, award, reporting, design),
next to the readiness roles. The CHECK constraint is re-emitted with the
widened vocabulary. No row is rewritten here; the backfill is a separate
plan/execute ops run.

Revision ID: 0035
Revises: 0034
"""
from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0035"
down_revision = "0034"
branch_labels = None
depends_on = None

_READINESS = ("baseline_schedule", "daily_report", "weekly_report", "other")
_FOLDER = ("contract", "draft", "tender", "award", "reporting", "design")


def _emit(roles) -> None:
    allowed = ", ".join(f"'{r}'" for r in roles)
    op.execute(sa.text(
        "ALTER TABLE documents DROP CONSTRAINT IF EXISTS ck_documents_doc_role"
    ))
    op.execute(sa.text(
        f"ALTER TABLE documents ADD CONSTRAINT ck_documents_doc_role "
        f"CHECK (doc_role IN ({allowed}))"
    ))


def upgrade() -> None:
    if op.get_bind().dialect.name == "sqlite":
        # SQLite dev/test DBs are built by the ORM, which carries the constraint.
        return
    _emit(_READINESS + _FOLDER)


def downgrade() -> None:
    if op.get_bind().dialect.name == "sqlite":
        return
    # Rows that carry a folder role go back to 'other' first, or the narrow
    # constraint would refuse them.
    allowed = ", ".join(f"'{r}'" for r in _FOLDER)
    op.execute(sa.text(
        f"UPDATE documents SET doc_role = 'other' WHERE doc_role IN ({allowed})"
    ))
    _emit(_READINESS)
