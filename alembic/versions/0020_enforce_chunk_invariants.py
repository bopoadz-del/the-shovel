"""Make the document ledger and the chunk tables agree.

Under the standing ingest rules (docs/INGEST_EXCLUSION_RULE.md): a document
of a format no extractor reads is removed with its chunks and index entries;
a failed, empty, unsupported, quarantined or tombstoned document keeps no
chunks; a document recorded indexed with no chunk -- or not yet classified
while holding chunks -- is classified from what it holds. The rule is
``ingest_status.enforce_chunk_invariants``: dialect neutral, idempotent (a
second run changes nothing), unit-tested on SQLite. From here on the ledger
writer keeps the invariant itself (``projects.delete_document_chunks``).

Revision ID: 0020
Revises: 0019
"""
from __future__ import annotations

import logging

from alembic import op

revision = "0020"
down_revision = "0019"
branch_labels = None
depends_on = None


def upgrade() -> None:
    from app.core.ingest_status import enforce_chunk_invariants

    counts = enforce_chunk_invariants(op.get_bind())
    logging.getLogger("alembic.runtime.migration").info("0020: %s", counts)


def downgrade() -> None:
    # Data-only revision: the schema is unchanged and removed chunks of
    # excluded documents are not restored. Logged so the step is visible.
    logging.getLogger("alembic.runtime.migration").info(
        "0020 downgrade: chunk invariants kept (data-only revision)"
    )
