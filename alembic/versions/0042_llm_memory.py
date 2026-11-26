"""llm_memory: the model's own JSON notes, one document per (project, name)

Written and read only by the model's memory tools (``app.core.llm_memory``):
project_id, name, data (JSON text), size_bytes, written_at, written_by_turn.
Primary key (project_id, name): a write replaces the whole document.

Idempotent: a database that already has the table is left alone.

Revision ID: 0042
Revises: 0039
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision = "0042"
down_revision = "0039"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if "llm_memory" in inspect(op.get_bind()).get_table_names():
        return
    op.create_table(
        "llm_memory",
        sa.Column("project_id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("data", sa.Text(), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("written_at", sa.String(), nullable=False),
        sa.Column("written_by_turn", sa.String(), nullable=False, server_default=""),
        sa.PrimaryKeyConstraint("project_id", "name"),
    )


def downgrade() -> None:
    if "llm_memory" in inspect(op.get_bind()).get_table_names():
        op.drop_table("llm_memory")
