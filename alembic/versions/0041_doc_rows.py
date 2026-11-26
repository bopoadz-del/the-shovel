"""doc_rows: table rows of indexed documents, held as facts

One row per table row a document's reader returns at ingest (a priced bill,
a spreadsheet): the heading path, the field map, page or sheet row, and the
stored chunk made from it (``app.core.doc_rows``). Additive: no existing
table or row is touched.

Idempotent: an existing ``doc_rows`` table is left alone.

Revision ID: 0041
Revises: 0039
"""
from __future__ import annotations

from alembic import op
from sqlalchemy import inspect, text

revision = "0041"
down_revision = "0039"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    if "doc_rows" in inspect(conn).get_table_names():
        return
    conn.execute(text(
        "CREATE TABLE doc_rows ("
        " id VARCHAR PRIMARY KEY,"
        " project_id VARCHAR NOT NULL,"
        " doc_id VARCHAR NOT NULL,"
        " table_no INTEGER NOT NULL,"
        " row_no INTEGER NOT NULL,"
        " path TEXT NOT NULL DEFAULT '[]',"
        " fields TEXT NOT NULL DEFAULT '{}',"
        " extra TEXT NOT NULL DEFAULT '{}',"
        " page INTEGER,"
        " sheet VARCHAR,"
        " sheet_row INTEGER,"
        " chunk_index INTEGER,"
        " row_check VARCHAR NOT NULL DEFAULT '',"
        " created_at VARCHAR NOT NULL,"
        " CONSTRAINT uq_doc_rows_project_doc_row UNIQUE (project_id, doc_id, table_no, row_no)"
        ")"
    ))
    conn.execute(text("CREATE INDEX idx_doc_rows_doc ON doc_rows (project_id, doc_id)"))


def downgrade() -> None:
    conn = op.get_bind()
    if "doc_rows" in inspect(conn).get_table_names():
        conn.execute(text("DROP TABLE doc_rows"))
