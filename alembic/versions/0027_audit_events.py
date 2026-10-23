"""audit_events: the durable fallback for audit entries

An audit entry is appended to DATA_DIR/audit.log; when that append fails the
entry is written here instead (app.core.audit), so none is ever dropped. The
whole entry is kept as JSON in ``body``; ``project_id`` and ``ts`` are
indexed for the per-project audit listing.

Idempotent: an existing table is left alone.

Revision ID: 0027
Revises: 0026
"""
from __future__ import annotations

from alembic import op
from sqlalchemy import inspect, text

revision = "0027"
down_revision = "0026"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    if "audit_events" in inspect(conn).get_table_names():
        return
    conn.execute(text(
        "CREATE TABLE audit_events ("
        " id VARCHAR PRIMARY KEY,"
        " ts VARCHAR NOT NULL,"
        " event VARCHAR NOT NULL,"
        " project_id VARCHAR,"
        " body TEXT NOT NULL)"
    ))
    conn.execute(text("CREATE INDEX idx_audit_events_project_ts ON audit_events (project_id, ts)"))


def downgrade() -> None:
    conn = op.get_bind()
    if "audit_events" in inspect(conn).get_table_names():
        conn.execute(text("DROP TABLE audit_events"))
