"""Leave exactly one doc_role CHECK on documents, found by its definition (RQ-3).

0035 dropped the CHECK by the name ``ck_documents_doc_role`` and added the
widened one under that name. A table built by 0001 from the_fork_schema.sql
carries the inline column CHECK, which Postgres names on its own
(``documents_doc_role_check``), so 0035's DROP missed it: both constraints
applied and the narrow one refused every folder role. This revision reads the
table's CHECK constraints from ``pg_constraint`` and drops every one whose
definition names the ``doc_role`` column, whatever it is called, then adds the
single widened constraint.

Revision ID: 0036
Revises: 0035
"""
from __future__ import annotations

import re
from collections.abc import Iterable

import sqlalchemy as sa

from alembic import op

revision = "0036"
down_revision = "0035"
branch_labels = None
depends_on = None

NAME = "ck_documents_doc_role"
ROLES = (
    "baseline_schedule", "daily_report", "weekly_report", "other",
    "contract", "draft", "tender", "award", "reporting", "design",
)

_COLUMN = re.compile(r"\bdoc_role\b")


def doc_role_checks(rows: Iterable[tuple[str, str]]) -> list[str]:
    """Names of the CHECK constraints, given as ``(name, definition)``, on doc_role."""
    return [name for name, definition in rows if _COLUMN.search(definition or "")]


def _quoted(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _one_wide_check() -> None:
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
    allowed = ", ".join(f"'{r}'" for r in ROLES)
    op.execute(sa.text(
        f"ALTER TABLE documents ADD CONSTRAINT {NAME} CHECK (doc_role IN ({allowed}))"
    ))


def upgrade() -> None:
    _one_wide_check()


def downgrade() -> None:
    # 0035's intended state is this same single widened constraint; the narrow
    # leftover was the defect, so going back does not restore it.
    _one_wide_check()
