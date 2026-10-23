"""chunks: layer + authority columns (layered RAG stage 1)

Adds the persisted layer (L1/L2A/L2B/L3) and authority-scoring columns.
Both nullable so pre-migration rows and the RAG_LAYERED=off path read as
unlayered. See docs/superpowers/plans/2026-07-23-layered-rag.md.

Prod writes to the namespaced table ``chunks_v2`` (RAG_VECTOR_NAMESPACE=v2),
not the legacy ``chunks`` table, so this migration alters every chunk table
that exists: the legacy ``chunks`` plus any ``chunks_<ns>``. Uses
IF [NOT] EXISTS so it is safe whether or not a given table is present.

Revision ID: 0011_chunks_layer_authority
Revises: 0010_projects_location
"""
from alembic import op
from sqlalchemy import text

revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None


def _chunk_tables(conn) -> list[str]:
    """Every RAG chunk table present: legacy ``chunks`` + ``chunks_<ns>``.

    Postgres path uses information_schema; SQLite uses sqlite_master. On any
    other backend, fall back to the two known names.
    """
    dialect = conn.dialect.name
    if dialect == "postgresql":
        rows = conn.execute(text(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = current_schema() "
            "AND (table_name = 'chunks' OR table_name LIKE 'chunks\\_%' ESCAPE '\\')"
        )).scalars().all()
        return list(rows)
    if dialect == "sqlite":
        rows = conn.execute(text(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND (name = 'chunks' OR name LIKE 'chunks\\_%' ESCAPE '\\')"
        )).scalars().all()
        return list(rows)
    return ["chunks", "chunks_v2"]


def upgrade() -> None:
    conn = op.get_bind()
    for tbl in _chunk_tables(conn):
        op.execute(f'ALTER TABLE "{tbl}" ADD COLUMN IF NOT EXISTS knowledge_layer TEXT')
        op.execute(f'ALTER TABLE "{tbl}" ADD COLUMN IF NOT EXISTS authority TEXT')
        op.execute(
            f'CREATE INDEX IF NOT EXISTS "idx_{tbl}_klayer" ON "{tbl}" (knowledge_layer)'
        )


def downgrade() -> None:
    conn = op.get_bind()
    for tbl in _chunk_tables(conn):
        op.execute(f'DROP INDEX IF EXISTS "idx_{tbl}_klayer"')
        op.execute(f'ALTER TABLE "{tbl}" DROP COLUMN IF EXISTS authority')
        op.execute(f'ALTER TABLE "{tbl}" DROP COLUMN IF EXISTS knowledge_layer')
