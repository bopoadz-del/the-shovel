"""Per-document ingest status, chunk count, and Drive content identity.

The corpus has been re-ingested repeatedly, and every time the ledger said the
run had succeeded. It could not say otherwise: a ``documents`` row proves a file
was registered, never that anything usable came out of it. Measured 2026-09-04
against the live corpus, 1,873 of 2,691 PDFs in one project (69.6%) hold exactly
ONE chunk, and they sat under the same implicit "indexed" as a fully-extracted
specification. These columns are what makes that visible.

WHY ``chunk_count = 1`` IS THE SPARSE RULE, not a tuned threshold:
chunking uses ~500-word windows, so a document that produced exactly one chunk
is by construction a document whose entire extracted text fit in a single
window. Calibration run over the live corpus confirms the rule needs no magic
number -- at a 4,000-char cut, the count of one-chunk PDFs left classified
INDEXED was ZERO, i.e. the threshold added nothing that ``= 1`` had not already
caught. A chars/MB gate was measured and REJECTED: 935 one-chunk documents sit
above any defensible cut, because chars/MB conflates an image-heavy document
with a text-poor one.

BACKFILL DIRECTION, and why it is the safe direction (cf. 0015, where it was
the opposite): existing rows are demoted, never promoted. A row with chunks
becomes INDEXED only if it has MORE than one; one-chunk rows become TEXT_SPARSE
and zero-chunk rows become ZERO_CHUNK. Getting this backwards would re-assert
the exact false-green this column exists to end. Nothing is deleted and no
vector is touched -- this migration only labels.

TEXT_SPARSE is deliberately left UNQUALIFIED here. Splitting it into
TERMINAL (no recoverable source; closed, not backlog) and RECOVERABLE (a .dwg
source of the same name exists) requires the Drive manifest, which a migration
must not read. ``scripts/stamp_ingest_status.py`` refines it afterwards.

NOT a new ledger table, by design: these are columns on the row that already
exists, so ``content_sha256`` keeps its meaning as archive identity and is
never used as a skip key.

``drive_md5`` exists because resume compared the wrong things. Drive publishes
``md5Checksum``; this codebase stores ``sha256(raw_bytes)``. Those never compare
equal, so a content-aware resume built on the existing column could only ever
report "everything changed" and re-index the entire corpus on every pass. MD5 is
stored here purely as Drive's identity token for change detection -- it is not
used as a security primitive.

Revision ID: 0016
Revises: 0015
"""
from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0016"
down_revision = "0015"
branch_labels = None
depends_on = None


# Status vocabulary. Kept as a plain CHECK rather than a Postgres ENUM so a new
# state can ship in a follow-up migration without an ALTER TYPE dance.
_STATUSES = (
    "UNVERIFIED",              # backfill default; never written by the pipeline
    "INDEXED",                 # >1 chunk, text recovered
    "TEXT_SPARSE",             # exactly one chunk -- everything fit one window
    "ZERO_CHUNK",              # registered, produced nothing
    "UNSUPPORTED_TYPE",        # format cannot yield text (.dwg, raster, fonts)
    "EXTRACT_FAILED",          # extractor raised
    "TOMBSTONED",              # gone from Drive; chunks hidden, never deleted
    "QUARANTINED",             # orphan/suspect; owner-gated, never auto-purged
)


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        # SQLite dev/test DBs are built by the ORM, which carries these columns.
        return

    op.execute(sa.text(
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS ingest_status "
        "TEXT NOT NULL DEFAULT 'UNVERIFIED'"
    ))
    op.execute(sa.text(
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS ingest_status_reason TEXT"
    ))
    op.execute(sa.text(
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS chunk_count "
        "INTEGER NOT NULL DEFAULT 0"
    ))
    op.execute(sa.text(
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS drive_md5 TEXT"
    ))
    op.execute(sa.text(
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS last_verified_at TEXT"
    ))

    allowed = ", ".join(f"'{s}'" for s in _STATUSES)
    op.execute(sa.text(
        "ALTER TABLE documents DROP CONSTRAINT IF EXISTS ck_documents_ingest_status"
    ))
    op.execute(sa.text(
        f"ALTER TABLE documents ADD CONSTRAINT ck_documents_ingest_status "
        f"CHECK (ingest_status IN ({allowed}))"
    ))

    # The active chunk table is namespaced (chunks_v2 today). Resolve it rather
    # than hardcoding: the retired literal `chunks` table holds 0 rows, and
    # counting against it is precisely the false-zero fixed in #388.
    table = bind.execute(sa.text(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema='public' AND table_name LIKE 'chunks%' "
        "AND table_name <> 'chunks' ORDER BY table_name DESC LIMIT 1"
    )).scalar()
    if not table:
        # No namespaced chunk table: fresh DB. Columns exist; nothing to backfill.
        return

    # RE-APPLICATION MUST BE A NO-OP ON CLASSIFIED ROWS. This migration was
    # applied to production once out of band and stamped back, so the
    # entrypoint will run it again on deploy. Every write below is scoped to
    # rows still UNVERIFIED: anything the stamper has already refined (the
    # terminal/recoverable split, UNSUPPORTED_TYPE) is never touched. Without
    # this guard a deploy silently regressed 305 UNSUPPORTED_TYPE rows to
    # ZERO_CHUNK and erased every qualifier -- the ledger lying again, on the
    # very next release.
    op.execute(sa.text(
        f"UPDATE documents d SET chunk_count = COALESCE(("
        f"  SELECT COUNT(*) FROM {table} c WHERE c.doc_id = d.id), 0) "
        f"WHERE d.ingest_status = 'UNVERIFIED'"
    ))

    # Demote-only. Order matters: the widest bucket first, then narrow.
    op.execute(sa.text(
        "UPDATE documents SET ingest_status='INDEXED' "
        "WHERE ingest_status = 'UNVERIFIED' AND chunk_count > 1"
    ))
    op.execute(sa.text(
        "UPDATE documents SET ingest_status='TEXT_SPARSE', "
        "ingest_status_reason='single_window' "
        "WHERE ingest_status = 'UNVERIFIED' AND chunk_count = 1"
    ))
    op.execute(sa.text(
        "UPDATE documents SET ingest_status='ZERO_CHUNK' "
        "WHERE ingest_status = 'UNVERIFIED' AND chunk_count = 0"
    ))


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        return
    op.execute(sa.text(
        "ALTER TABLE documents DROP CONSTRAINT IF EXISTS ck_documents_ingest_status"
    ))
    for col in (
        "ingest_status",
        "ingest_status_reason",
        "chunk_count",
        "drive_md5",
        "last_verified_at",
    ):
        op.execute(sa.text(f"ALTER TABLE documents DROP COLUMN IF EXISTS {col}"))
