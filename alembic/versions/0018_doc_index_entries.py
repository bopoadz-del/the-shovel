"""Split the per-project doc_index blob into one row per entry, text-free.

``doc_index.index_json`` carried every document / skipped entry of a project,
chunk text included, as ONE JSON blob (48 MB for a 3,191-document project),
and every single-document index update read and rewrote the whole blob.
Live 2026-10-03: an ECS service re-running the Drive ingest on restart
re-touched the same 1,048 files each pass and produced 4,479 full rewrites in
80 minutes -- roughly 430 GB over the wire and 1.1 TB of buffer traffic on a
1.6 GB database -- and those rewrites are where the retained WAL ("history")
came from.

Each entry is now a ``doc_index_entries`` row (``kind`` = document | skipped,
``seq`` keeps list order) holding identity and counts only: ``chunks_v2`` is
the only store of chunk text. ``doc_index.index_json`` keeps the header
(``project_id``, ``built_at``). ``app/core/doc_index.py`` assembles the
historical dict shape on read and writes one row per touched document.

The conversion itself is ``doc_index.migrate_blob_entries`` -- dialect
neutral and idempotent (a second run converts 0 rows), so it is unit-tested
on SQLite and runs here on PostgreSQL. It does not VACUUM: the dead blob
versions it leaves behind belong to this transaction and cannot be reclaimed
until it commits; reclaiming them (``VACUUM FULL doc_index``) is the
operator's post-deploy step, measured before/after.

Revision ID: 0018
Revises: 0017
"""
from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0018"
down_revision = "0017"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        # SQLite dev/test DBs are built by the ORM (doc_index.init_db).
        return

    op.execute(sa.text(
        "CREATE TABLE IF NOT EXISTS doc_index_entries ("
        "  project_id TEXT NOT NULL REFERENCES projects (id) ON DELETE CASCADE,"
        "  document_id TEXT NOT NULL,"
        "  kind TEXT NOT NULL,"
        "  seq INTEGER NOT NULL DEFAULT 0,"
        "  entry_json JSONB NOT NULL,"
        "  updated_at TEXT NOT NULL,"
        "  PRIMARY KEY (project_id, document_id)"
        ")"
    ))
    op.execute(sa.text(
        "CREATE INDEX IF NOT EXISTS idx_doc_index_entries_project "
        "ON doc_index_entries (project_id, seq)"
    ))

    from app.core.doc_index import migrate_blob_entries

    migrate_blob_entries(bind)


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        return
    # Fold the rows back into the blob (counts only -- the chunk text was
    # never in the rows and lives in chunks_v2), then drop the table.
    op.execute(sa.text(
        "UPDATE doc_index di SET index_json = di.index_json || jsonb_build_object("
        "  'documents', COALESCE((SELECT jsonb_agg(e.entry_json ORDER BY e.seq) "
        "     FROM doc_index_entries e "
        "     WHERE e.project_id = di.project_id AND e.kind = 'document'), '[]'::jsonb),"
        "  'skipped', COALESCE((SELECT jsonb_agg(e.entry_json ORDER BY e.seq) "
        "     FROM doc_index_entries e "
        "     WHERE e.project_id = di.project_id AND e.kind = 'skipped'), '[]'::jsonb)"
        ")"
    ))
    op.execute(sa.text("DROP TABLE IF EXISTS doc_index_entries"))
