"""OCR provenance on chunks: a chunk whose text came from OCR is marked.

Item 7 (advisor 06:12Z): scanned-PDF pages whose text layer is empty are
OCR'd at ingestion, and that text must carry an ``ocr=true`` provenance flag
so citations can tell OCR-derived text apart. This adds the ``ocr`` column to
``chunks`` (default false). SQLite test/dev DBs are built by the ORM, which
carries the column.

Revision ID: 0033
Revises: 0032
"""
from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0033"
down_revision = "0032"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "chunks",
        sa.Column("ocr", sa.Boolean(), nullable=False, server_default=sa.text("FALSE")),
    )


def downgrade() -> None:
    op.drop_column("chunks", "ocr")
