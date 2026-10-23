"""chunks.project_id: CASCADE -> SET NULL (decouple knowledge from workspaces)

The legacy ``chunks`` table's ``project_id`` FK was ``ON DELETE CASCADE`` — the
coupling that made deleting a project destroy its RAG. Soften it to SET NULL so
a chunk survives its workspace's deletion. Prod writes to the namespaced
``chunks_v2`` table, which the ORM builds with no FK, so this is belt-and-
suspenders on the retired legacy table; Postgres-only (SQLite dev has no such
FK to alter).

Revision ID: 0012_chunks_project_fk_set_null
Revises: 0011_chunks_layer_authority
"""
from alembic import op

revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    if conn.dialect.name != "postgresql":
        return  # SQLite dev DB has no such FK constraint to alter
    op.execute("ALTER TABLE chunks ALTER COLUMN project_id DROP NOT NULL")
    op.execute("ALTER TABLE chunks DROP CONSTRAINT IF EXISTS chunks_project_id_fkey")
    op.execute(
        "ALTER TABLE chunks ADD CONSTRAINT chunks_project_id_fkey "
        "FOREIGN KEY (project_id) REFERENCES projects (id) ON DELETE SET NULL"
    )


def downgrade() -> None:
    conn = op.get_bind()
    if conn.dialect.name != "postgresql":
        return
    op.execute("ALTER TABLE chunks DROP CONSTRAINT IF EXISTS chunks_project_id_fkey")
    op.execute(
        "ALTER TABLE chunks ADD CONSTRAINT chunks_project_id_fkey "
        "FOREIGN KEY (project_id) REFERENCES projects (id) ON DELETE CASCADE"
    )
    op.execute("ALTER TABLE chunks ALTER COLUMN project_id SET NOT NULL")
