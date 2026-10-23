"""Index the documents hidden from retrieval, under the exact retrieval condition.

The retrieval filter hides a chunk whose document is ``retrieval_visible``
false or has an ingest status with no chunks. The only index on that table
covered the first half, so with the OR every BM25 call hashed the whole
documents table: live 2026-10-07, 1,547 pages (12.7 MB) per call, several
calls per question. A partial index under the same condition holds only the
hidden documents, so the anti-join reads a handful of index pages.

The condition comes from ``vector_store.hidden_document_predicate`` -- the
function the filter itself uses -- so the two cannot drift apart. Idempotent.

Revision ID: 0025
Revises: 0024
"""
from __future__ import annotations

from alembic import op
from sqlalchemy import text

revision = "0025"
down_revision = "0024"
branch_labels = None
depends_on = None

_INDEX = "idx_documents_hidden_from_retrieval"


def upgrade() -> None:
    conn = op.get_bind()
    if conn.dialect.name != "postgresql":
        return
    from app.core.rag.vector_store import hidden_document_predicate

    predicate = hidden_document_predicate(postgres=True)
    with op.get_context().autocommit_block():
        conn.execute(text(
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {_INDEX} ON documents (id) WHERE {predicate}"
        ))


def downgrade() -> None:
    conn = op.get_bind()
    if conn.dialect.name != "postgresql":
        return
    conn.execute(text(f"DROP INDEX IF EXISTS {_INDEX}"))
