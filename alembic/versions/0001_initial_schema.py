"""Initial unified schema (the_fork_schema.sql).

Revision ID: 0001
Revises:
Create Date: 2026-06-12

Applies the canonical PostgreSQL schema reverse-engineered from the legacy
SQLite stores. Requires pgvector (CREATE EXTENSION vector).
"""

from __future__ import annotations

from pathlib import Path

from alembic import op
from sqlalchemy import text

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

_SCHEMA_FILE = Path(__file__).resolve().parents[2] / "the_fork_schema.sql"


def upgrade() -> None:
    sql = _SCHEMA_FILE.read_text(encoding="utf-8")
    op.get_bind().execute(text(sql))


def downgrade() -> None:
    op.get_bind().execute(
        text(
            """
            DROP TABLE IF EXISTS chunks CASCADE;
            DROP TABLE IF EXISTS rag_budget CASCADE;
            DROP TABLE IF EXISTS hydration_runs CASCADE;
            DROP TABLE IF EXISTS runs CASCADE;
            DROP TABLE IF EXISTS doc_index CASCADE;
            DROP TABLE IF EXISTS agent_facts CASCADE;
            DROP TABLE IF EXISTS messages CASCADE;
            DROP TABLE IF EXISTS conversations CASCADE;
            DROP TABLE IF EXISTS workflows CASCADE;
            DROP TABLE IF EXISTS project_facts CASCADE;
            DROP TABLE IF EXISTS documents CASCADE;
            DROP TABLE IF EXISTS projects CASCADE;
            DROP TABLE IF EXISTS users CASCADE;
            DROP EXTENSION IF EXISTS vector;
            """
        )
    )
