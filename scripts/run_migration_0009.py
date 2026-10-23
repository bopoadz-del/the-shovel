#!/usr/bin/env python3
"""One-off migration to add documents.metadata JSONB and stamp alembic head.

Run this on Render before the ingestion worker when the existing Postgres DB
is missing the metadata column (e.g. the web service was not yet upgraded to
the revision that adds it). It is idempotent.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sqlalchemy import text

from app.core.db import get_engine


def main() -> int:
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS metadata JSONB"))
        # Ensure alembic_version reflects this migration so future upgrades
        # don't try to re-add the column.
        conn.execute(text("CREATE TABLE IF NOT EXISTS alembic_version (version_num VARCHAR(32) PRIMARY KEY)"))
        conn.execute(text("DELETE FROM alembic_version"))
        conn.execute(text("INSERT INTO alembic_version (version_num) VALUES ('0009') ON CONFLICT (version_num) DO NOTHING"))
    print("migration 0009 applied (documents.metadata JSONB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
