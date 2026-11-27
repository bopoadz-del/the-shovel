"""llm_memory review: the owner keeps or discards the model's notes

Adds to ``llm_memory`` the owner's mark on the current version of a note
(review, reviewed_at, reviewed_by) and a ``llm_memory_review`` table that
records every keep / discard with the note's content as it was when decided.

Idempotent: columns or a table already present are left alone.

Revision ID: 0043
Revises: 0042
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision = "0043"
down_revision = "0042"
branch_labels = None
depends_on = None

_MARK_COLUMNS = ("review", "reviewed_at", "reviewed_by")


def upgrade() -> None:
    insp = inspect(op.get_bind())
    have = {c["name"] for c in insp.get_columns("llm_memory")}
    for col in _MARK_COLUMNS:
        if col not in have:
            op.add_column("llm_memory",
                          sa.Column(col, sa.String(), nullable=False, server_default=""))
    if "llm_memory_review" not in insp.get_table_names():
        op.create_table(
            "llm_memory_review",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("project_id", sa.String(), nullable=False),
            sa.Column("name", sa.String(), nullable=False),
            sa.Column("decision", sa.String(), nullable=False),
            sa.Column("data", sa.Text(), nullable=False),
            sa.Column("note_written_at", sa.String(), nullable=False),
            sa.Column("note_written_by_turn", sa.String(), nullable=False, server_default=""),
            sa.Column("decided_at", sa.String(), nullable=False),
            sa.Column("decided_by", sa.String(), nullable=False, server_default=""),
        )
        op.create_index("ix_llm_memory_review_project_id", "llm_memory_review", ["project_id"])


def downgrade() -> None:
    insp = inspect(op.get_bind())
    if "llm_memory_review" in insp.get_table_names():
        op.drop_table("llm_memory_review")
    have = {c["name"] for c in insp.get_columns("llm_memory")}
    for col in _MARK_COLUMNS:
        if col in have:
            op.drop_column("llm_memory", col)
