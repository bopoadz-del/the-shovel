"""messages: nullable ``provenance`` column (where each figure/fact came from)

Every assistant answer carries a provenance record: for each figure and each
cited fact, its source kind -- user input, project document (doc and page),
general knowledge (document and page) or calculator (formula and inputs). The
record is streamed with the answer and stored with the message here, as JSON
text. Nullable: user turns and messages written before this revision have none.

Idempotent: a table that already has the column is left alone.

Revision ID: 0024
Revises: 0023
"""
from __future__ import annotations

from alembic import op
from sqlalchemy import inspect, text

revision = "0024"
down_revision = "0023"
branch_labels = None
depends_on = None


def _has_column(conn, table: str, column: str) -> bool:
    return column in {c["name"] for c in inspect(conn).get_columns(table)}


def upgrade() -> None:
    conn = op.get_bind()
    if "messages" not in inspect(conn).get_table_names():
        return
    if not _has_column(conn, "messages", "provenance"):
        conn.execute(text("ALTER TABLE messages ADD COLUMN provenance TEXT"))


def downgrade() -> None:
    conn = op.get_bind()
    if "messages" in inspect(conn).get_table_names() and _has_column(conn, "messages", "provenance"):
        conn.execute(text("ALTER TABLE messages DROP COLUMN provenance"))
