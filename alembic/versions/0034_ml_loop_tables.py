"""ML loop block tables (ML-1).

Revision ID: 0034
Revises: 0033

The six ``ml_`` tables of ``app.core.ml_loop.store``, spelled out here so
the migration does not change when the module does. A drift test
(``tests/test_ml_loop_ml1.py``) compares these columns and indexes with
the module's ``MetaData``.
"""
from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0034"
down_revision = "0033"
branch_labels = None
depends_on = None


def _keys() -> list[sa.Column]:
    return [
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("project_id", sa.String(200), nullable=False),
        sa.Column("conversation_id", sa.String(200), nullable=True),
        sa.Column("message_id", sa.String(200), nullable=True),
        sa.Column("build_sha", sa.String(64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    ]


def _turns() -> list[sa.Column]:
    return [
        sa.Column("question", sa.Text, nullable=False),
        sa.Column("answer", sa.Text, nullable=False),
        sa.Column("verdict", sa.String(64), nullable=False),
        sa.Column("route", sa.String(120), nullable=True),
        sa.Column("evidence_plan", sa.JSON, nullable=False),
        sa.Column("retrieved_ids", sa.JSON, nullable=False),
        sa.Column("cited_ids", sa.JSON, nullable=False),
    ]


def _feedback() -> list[sa.Column]:
    return [
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("correct_value", sa.Float, nullable=True),
        sa.Column("unit", sa.String(32), nullable=True),
        sa.Column("note", sa.String(200), nullable=True),
        sa.Column("cited_ids", sa.JSON, nullable=False),
        sa.Column("user_id", sa.String(200), nullable=True),
    ]


def _corrections() -> list[sa.Column]:
    return [
        sa.Column("formula_id", sa.String(200), nullable=False),
        sa.Column("predicted", sa.Float, nullable=False),
        sa.Column("actual", sa.Float, nullable=False),
        sa.Column("source", sa.String(64), nullable=False),
    ]


def _coefficients() -> list[sa.Column]:
    return [
        sa.Column("formula_id", sa.String(200), nullable=False),
        sa.Column("version", sa.Integer, nullable=False),
        sa.Column("coefficients", sa.JSON, nullable=False),
        sa.Column("mae", sa.Float, nullable=True),
        sa.Column("tier", sa.String(32), nullable=False),
        sa.Column("provenance", sa.JSON, nullable=False),
    ]


def _route_corrections() -> list[sa.Column]:
    return [
        sa.Column("message", sa.Text, nullable=False),
        sa.Column("correct_action", sa.String(120), nullable=False),
        sa.Column("original_action", sa.String(120), nullable=True),
        sa.Column("source", sa.String(64), nullable=False),
    ]


def _shadow() -> list[sa.Column]:
    return [
        sa.Column("predictor", sa.String(120), nullable=False),
        sa.Column("live_label", sa.String(120), nullable=False),
        sa.Column("shadow_label", sa.String(120), nullable=False),
        sa.Column("confidence", sa.Float, nullable=True),
    ]


#: (table, column builder, index name, index columns), in creation order.
TABLES = (
    ("ml_turns", _turns, "ix_ml_turns_project_created", ["project_id", "created_at"]),
    ("ml_feedback", _feedback, "ix_ml_feedback_project_created", ["project_id", "created_at"]),
    ("ml_corrections", _corrections, "ix_ml_corrections_formula", ["formula_id", "created_at"]),
    ("ml_coefficients", _coefficients, "ix_ml_coefficients_formula_version",
     ["formula_id", "version"]),
    ("ml_route_corrections", _route_corrections, "ix_ml_route_corrections_project_created",
     ["project_id", "created_at"]),
    ("ml_shadow_predictions", _shadow, "ix_ml_shadow_predictor_created",
     ["predictor", "created_at"]),
)


def upgrade() -> None:
    for name, columns, index_name, index_columns in TABLES:
        op.create_table(name, *_keys(), *columns())
        op.create_index(index_name, name, index_columns)


def downgrade() -> None:
    for name, _columns, index_name, _index_columns in reversed(TABLES):
        op.drop_index(index_name, table_name=name)
        op.drop_table(name)
