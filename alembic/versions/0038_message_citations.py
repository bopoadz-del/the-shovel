"""messages: nullable ``citations`` column (the answer's structured citations)

The answer contract (``app.agents.answer_contract``) turns every citation in
an answer into a numbered marker with one structured entry (document, page or
section, tier). Those entries are stored with the assistant message here, as
JSON text, so a document produced from the answer later lists its sources from
them instead of re-reading the answer's wording (W2-a). Nullable: user turns
and messages written before this revision have none.

Idempotent: a table that already has the column is left alone.

Revision ID: 0038
Revises: 0037
"""
from __future__ import annotations

from alembic import op
from sqlalchemy import inspect, text

revision = "0038"
down_revision = "0037"
branch_labels = None
depends_on = None


def _has_column(conn, table: str, column: str) -> bool:
    return column in {c["name"] for c in inspect(conn).get_columns(table)}


def upgrade() -> None:
    conn = op.get_bind()
    if "messages" not in inspect(conn).get_table_names():
        return
    if not _has_column(conn, "messages", "citations"):
        conn.execute(text("ALTER TABLE messages ADD COLUMN citations TEXT"))


def downgrade() -> None:
    conn = op.get_bind()
    if "messages" in inspect(conn).get_table_names() and _has_column(conn, "messages", "citations"):
        conn.execute(text("ALTER TABLE messages DROP COLUMN citations"))
