"""OCR page ledger on documents: truncation is never silent.

A scanned PDF whose OCR is capped mid-file has been reported as INDEXED
because the ledger could only say "chunks came out". That hides the exact
defect the owner rules out: a photo/scan PDF is OCR'd into readable chunks,
never silently truncated. These columns make truncation visible on the row:

* ``pages_in_file``      -- pages the PDF actually has (fitz count).
* ``pages_ocrd``         -- pages whose text was captured by OCR.
* ``ocr_truncated``      -- True when the OCR page cap stopped before the end.
* ``ocr_skip_reason``    -- why OCR never ran (size ceiling); None otherwise.
* ``ocr_unusable_pages`` -- pages whose OCR output failed the quality floor.

A document with ``ocr_truncated = True`` is stamped PARTIAL_OCR, a status of
its own that is NOT INDEXED -- the UI and the chat both say "partially read,
pages 1-N of M" instead of looking complete. PARTIAL_OCR keeps its chunks
(they are real, readable pages) and is therefore not a no-chunk status.

The CHECK constraint is re-emitted with the new status in the vocabulary.

Revision ID: 0032
Revises: 0031
"""
from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0032"
down_revision = "0031"
branch_labels = None
depends_on = None


_STATUSES = (
    "UNVERIFIED",
    "INDEXED",
    "TEXT_SPARSE",
    "ZERO_CHUNK",
    "UNSUPPORTED_TYPE",
    "EXTRACT_FAILED",
    "TOMBSTONED",
    "QUARANTINED",
    "PARTIAL_OCR",
)


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        # SQLite dev/test DBs are built by the ORM, which carries these columns.
        return

    op.execute(sa.text(
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS pages_in_file INTEGER"
    ))
    op.execute(sa.text(
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS pages_ocrd INTEGER"
    ))
    op.execute(sa.text(
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS ocr_truncated "
        "BOOLEAN NOT NULL DEFAULT FALSE"
    ))
    op.execute(sa.text(
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS ocr_skip_reason TEXT"
    ))
    op.execute(sa.text(
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS ocr_unusable_pages "
        "INTEGER NOT NULL DEFAULT 0"
    ))

    allowed = ", ".join(f"'{s}'" for s in _STATUSES)
    op.execute(sa.text(
        "ALTER TABLE documents DROP CONSTRAINT IF EXISTS ck_documents_ingest_status"
    ))
    op.execute(sa.text(
        f"ALTER TABLE documents ADD CONSTRAINT ck_documents_ingest_status "
        f"CHECK (ingest_status IN ({allowed}))"
    ))


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        return
    allowed = ", ".join(
        f"'{s}'" for s in _STATUSES if s != "PARTIAL_OCR"
    )
    op.execute(sa.text(
        "ALTER TABLE documents DROP CONSTRAINT IF EXISTS ck_documents_ingest_status"
    ))
    op.execute(sa.text(
        f"ALTER TABLE documents ADD CONSTRAINT ck_documents_ingest_status "
        f"CHECK (ingest_status IN ({allowed}))"
    ))
    for col in (
        "ocr_unusable_pages",
        "ocr_skip_reason",
        "ocr_truncated",
        "pages_ocrd",
        "pages_in_file",
    ):
        op.execute(sa.text(f"ALTER TABLE documents DROP COLUMN IF EXISTS {col}"))
