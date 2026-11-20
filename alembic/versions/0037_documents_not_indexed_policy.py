"""NOT_INDEXED_POLICY: a stored upload kept out of the RAG by policy.

A non-admin upload is stored but, by owner policy, never indexed into the
project RAG. Its ledger stayed at UNVERIFIED -- an OPEN status -- so an
inventory could not tell "not indexed by policy" from "an indexing job that
was lost". The upload route now stamps NOT_INDEXED_POLICY with the reason
``rag_admin_only:terminal``. It is a settled status (not in OPEN_STATUSES)
and holds no chunks, like every other non-indexed status.

Only the CHECK constraint changes: it is re-emitted with the new status in
the vocabulary. No row is rewritten on upgrade (no backfill).

Revision ID: 0037
Revises: 0036
"""
from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0037"
down_revision = "0036"
branch_labels = None
depends_on = None


_STATUSES = (
    "UNVERIFIED",
    "INDEXED",
    "TEXT_SPARSE",
    "ZERO_CHUNK",
    "UNSUPPORTED_TYPE",
    "EXTRACT_FAILED",
    "TOMBSTONED",
    "QUARANTINED",
    "PARTIAL_OCR",
    "NOT_INDEXED_POLICY",
)

_NEW = "NOT_INDEXED_POLICY"


def _emit_check(statuses) -> None:
    allowed = ", ".join(f"'{s}'" for s in statuses)
    op.execute(sa.text(
        "ALTER TABLE documents DROP CONSTRAINT IF EXISTS ck_documents_ingest_status"
    ))
    op.execute(sa.text(
        f"ALTER TABLE documents ADD CONSTRAINT ck_documents_ingest_status "
        f"CHECK (ingest_status IN ({allowed}))"
    ))


def upgrade() -> None:
    if op.get_bind().dialect.name == "sqlite":
        # SQLite dev/test DBs are built by the ORM, which has no status CHECK.
        return
    _emit_check(_STATUSES)


def downgrade() -> None:
    if op.get_bind().dialect.name == "sqlite":
        return
    # The older vocabulary has no such status: put those rows back where the
    # upload route used to leave them, or re-adding the CHECK would fail.
    op.execute(sa.text(
        "UPDATE documents SET ingest_status = 'UNVERIFIED', "
        "ingest_status_reason = NULL WHERE ingest_status = :new"
    ).bindparams(new=_NEW))
    _emit_check(tuple(s for s in _STATUSES if s != _NEW))
