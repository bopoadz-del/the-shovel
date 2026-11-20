"""Widen the doc_role vocabulary with the contractor-submittal folder role.

A shop drawing or a drawing filed under a contractor-submittal folder now
carries doc_role ``contractor_submittal``; the doc-standing table then reads
it as a contractor's submittal, not a contract drawing. This revision widens
the single CHECK constraint on ``documents.doc_role`` the same way 0036 did:
find every CHECK on the column by its definition, drop it, re-add one widened
constraint. No row is rewritten here; the standing backfill is a separate
plan/execute ops run.

Revision ID: 0039
Revises: 0038
"""
from __future__ import annotations

import re
from collections.abc import Iterable

import sqlalchemy as sa

from alembic import op

revision = "0039"
down_revision = "0038"
branch_labels = None
depends_on = None

NAME = "ck_documents_doc_role"
ROLES = (
    "baseline_schedule", "daily_report", "weekly_report", "other",
    "contract", "draft", "tender", "award", "reporting", "design",
    "contractor_submittal",
)

_COLUMN = re.compile(r"\bdoc_role\b")


def doc_role_checks(rows: Iterable[tuple[str, str]]) -> list[str]:
    """Names of the CHECK constraints, given as ``(name, definition)``, on doc_role."""
    return [name for name, definition in rows if _COLUMN.search(definition or "")]


def _quoted(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _emit(roles: tuple[str, ...]) -> None:
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        # SQLite dev/test DBs are built by the ORM, which carries the constraint.
        return
    rows = bind.execute(sa.text(
        "SELECT conname, pg_get_constraintdef(oid) FROM pg_constraint "
        "WHERE conrelid = 'documents'::regclass AND contype = 'c'"
    )).fetchall()
    for name in doc_role_checks((r[0], r[1]) for r in rows):
        op.execute(sa.text(f"ALTER TABLE documents DROP CONSTRAINT {_quoted(name)}"))
    allowed = ", ".join(f"'{r}'" for r in roles)
    op.execute(sa.text(
        f"ALTER TABLE documents ADD CONSTRAINT {NAME} CHECK (doc_role IN ({allowed}))"
    ))


def upgrade() -> None:
    _emit(ROLES)


def downgrade() -> None:
    if op.get_bind().dialect.name == "sqlite":
        return
    # A row that carries the new role goes back to 'other' first, or the
    # narrower constraint would refuse it.
    op.execute(sa.text(
        "UPDATE documents SET doc_role = 'other' WHERE doc_role = 'contractor_submittal'"
    ))
    _emit(ROLES[:-1])
