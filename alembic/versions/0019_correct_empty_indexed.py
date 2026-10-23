"""Re-stamp documents recorded INDEXED that hold no text.

A file that yields no text is never recorded as indexed (owner ruling,
docs/INGEST_EXCLUSION_RULE.md). The bulk ``doc_index.index_project`` path
used to count such a file as indexed; it now stamps the ZERO_CHUNK class,
and this corrects rows already on the ledger. The rule itself is
``ingest_status.correct_empty_indexed`` -- dialect neutral and idempotent
(a second run changes 0 rows), unit-tested on SQLite. Deletes nothing.

Revision ID: 0019
Revises: 0018
"""
from __future__ import annotations

from alembic import op

revision = "0019"
down_revision = "0018"
branch_labels = None
depends_on = None


def upgrade() -> None:
    from app.core.ingest_status import correct_empty_indexed

    correct_empty_indexed(op.get_bind())


def downgrade() -> None:
    # Data-only revision: the schema is unchanged, so stepping back below 0019
    # needs no work, and re-recording an empty file as indexed would restore
    # the defect this revision removes. Logged so the step is visible.
    import logging

    logging.getLogger("alembic.runtime.migration").info(
        "0019 downgrade: empty-indexed status correction kept (data-only revision)"
    )
