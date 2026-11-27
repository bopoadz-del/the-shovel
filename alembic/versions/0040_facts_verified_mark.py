"""project_facts / agent_facts: nullable ``verified_by`` and ``verified_at``

A fact the app saved by itself (a nightly brief, a value the model chose to
remember) used to be read into every later prompt as a known fact, with no
record of who stated it or whether anyone checked it. These two columns are
the mark: who confirmed the fact and when. NULL means not verified, and the
prompt readers skip such rows. Existing rows stay NULL, so none of them is
prompt material until someone confirms it.

Idempotent: a table that already has a column is left alone.

Revision ID: 0040
Revises: 0039
"""
from __future__ import annotations

from alembic import op
from sqlalchemy import inspect, text

revision = "0040"
down_revision = "0039"
branch_labels = None
depends_on = None

TABLES = ("project_facts", "agent_facts")
COLUMNS = ("verified_by", "verified_at")


def _has_column(conn, table: str, column: str) -> bool:
    return column in {c["name"] for c in inspect(conn).get_columns(table)}


def upgrade() -> None:
    conn = op.get_bind()
    present = set(inspect(conn).get_table_names())
    for table in TABLES:
        if table not in present:
            continue
        for column in COLUMNS:
            if not _has_column(conn, table, column):
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} TEXT"))


def downgrade() -> None:
    conn = op.get_bind()
    present = set(inspect(conn).get_table_names())
    for table in TABLES:
        if table not in present:
            continue
        for column in COLUMNS:
            if _has_column(conn, table, column):
                conn.execute(text(f"ALTER TABLE {table} DROP COLUMN {column}"))
