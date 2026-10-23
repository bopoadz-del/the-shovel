#!/usr/bin/env python3
"""Refine the coarse statuses migration 0016 backfilled in SQL.

0016 could only see chunk counts -- a migration must not read a Drive manifest
or an extension policy. It therefore left two things undone, and this script
finishes them using ``app.core.ingest_status`` so the rule exists once:

1. ``UNSUPPORTED_TYPE`` for documents whose format cannot yield text at all.
   They sat at ZERO_CHUNK, which reads as a failure to fix. A ``.kmz`` or a
   font is not a failure; counting it as one is how a target of 6,206 files
   became a gate that could never go green.

2. The ``TEXT_SPARSE`` split. TERMINAL means no richer source exists anywhere
   -- a vector drawing sheet whose content is geometry, measured to yield
   nothing further under OCR at any DPI. RECOVERABLE means a ``.dwg`` of the
   same name exists, so the text is reachable via the F-DWG conversion route.
   Only RECOVERABLE is open work. Measured split: 1,747 terminal / 126
   recoverable in the largest project.

Sources for "a .dwg of the same name exists", in order: the Drive manifest when
given (it sees sources not yet ingested), plus sibling documents already in the
corpus. Dry run by default; ``--apply`` writes.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from app.core import ingest_status as ist


def _norm_stem(name: str) -> str:
    """Filename identity for sibling matching.

    Drops the extension, the ``(1)`` copy suffix Drive appends, and every
    separator, so ``FOO-BAR_01 (2).pdf`` and ``foo bar 01.dwg`` match.
    """
    s = name.rsplit(".", 1)[0].lower()
    s = re.sub(r"\s*\(\d+\)$", "", s)
    return re.sub(r"[^a-z0-9]+", "", s)


def _ext(name: str) -> str:
    i = name.rfind(".")
    return name[i:].lower() if i != -1 else ""


def _database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise SystemExit(
            "DATABASE_URL not set. Pass the Neon DSN explicitly -- never the "
            "retired Render dpg- host."
        )
    return re.sub(r"^postgresql\+psycopg://", "postgresql://", url)


def _convertible_stems(manifest: Path | None) -> set[str]:
    stems: set[str] = set()
    if manifest and manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        for folder in data.get("folders", []):
            for f in folder.get("files", []):
                if _ext(f.get("name", "")) in ist.RECOVERABLE_EXTS:
                    stems.add(_norm_stem(f["name"]))
    return stems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=None,
                    help="Drive manifest; adds CAD sources not yet ingested")
    ap.add_argument("--project", default=None, help="Limit to one project_id")
    ap.add_argument("--apply", action="store_true", help="Write (default: dry run)")
    args = ap.parse_args(argv)

    import psycopg

    url = _database_url()
    stems = _convertible_stems(args.manifest)
    print(f"[stamp] CAD stems from manifest: {len(stems)}")

    where = "WHERE project_id = %s" if args.project else ""
    params: tuple = (args.project,) if args.project else ()

    with psycopg.connect(url, connect_timeout=60) as cn:
        with cn.cursor() as cur:
            # Recompute chunk_count from the ACTIVE namespaced chunk table on
            # every run. The live pipeline does not stamp these columns yet, so
            # a document indexed after migration 0016 carries the default 0
            # until this runs. Without this step the stamper would read a
            # stale count and classify a freshly indexed file as ZERO_CHUNK --
            # the ledger lying in the opposite direction.
            cur.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema='public' AND table_name LIKE 'chunks%%' "
                "AND table_name <> 'chunks' ORDER BY table_name DESC LIMIT 1"
            )
            table = cur.fetchone()[0]
            cur.execute(
                f"UPDATE documents d SET chunk_count = COALESCE(("
                f"  SELECT COUNT(*) FROM {table} c WHERE c.doc_id = d.id), 0) "
                f"{where} AND chunk_count <> COALESCE(("
                f"  SELECT COUNT(*) FROM {table} c WHERE c.doc_id = d.id), 0)"
                if where else
                f"UPDATE documents d SET chunk_count = COALESCE(("
                f"  SELECT COUNT(*) FROM {table} c WHERE c.doc_id = d.id), 0) "
                f"WHERE chunk_count <> COALESCE(("
                f"  SELECT COUNT(*) FROM {table} c WHERE c.doc_id = d.id), 0)",
                params,
            )
            print(f"[stamp] chunk_count refreshed from {table}: {cur.rowcount} rows changed")
            if not args.apply:
                cn.rollback()
            else:
                cn.commit()
            cur.execute(
                f"SELECT id, original_name, chunk_count, ingest_status, "
                f"ingest_status_reason, size FROM documents {where}", params
            )
            rows = cur.fetchall()

            # Sibling CAD sources already registered as documents.
            cur.execute(
                f"SELECT original_name FROM documents {where}", params
            )
            for (nm,) in cur.fetchall():
                if _ext(nm or "") in ist.RECOVERABLE_EXTS:
                    stems.add(_norm_stem(nm))
        print(f"[stamp] CAD stems incl. corpus siblings: {len(stems)}")

        updates: list[tuple[str, str | None, str]] = []
        moves: Counter = Counter()
        for doc_id, name, chunk_count, status, reason, size in rows:
            name = name or ""
            c = ist.classify(
                chunk_count=chunk_count or 0,
                extension=_ext(name),
                size_bytes=size,
                has_convertible_source=_norm_stem(name) in stems,
            )
            if c.status != status or c.reason != reason:
                updates.append((c.status, c.reason, doc_id))
                moves[f"{status} -> {c.status}"] += 1

        print(f"\n[stamp] {len(updates)} of {len(rows)} rows change status/reason")
        for k, v in moves.most_common():
            print(f"   {k:<34}{v:>6}")

        if not args.apply:
            print("\nDRY RUN -- nothing written. Re-run with --apply.")
            return 0

        now = datetime.now(timezone.utc).isoformat()
        with cn.cursor() as cur:
            cur.executemany(
                "UPDATE documents SET ingest_status=%s, ingest_status_reason=%s, "
                f"last_verified_at='{now}' WHERE id=%s",
                updates,
            )
        cn.commit()
        print(f"\n[stamp] committed {len(updates)} updates")

        with cn.cursor() as cur:
            cur.execute(
                f"SELECT ingest_status, ingest_status_reason, COUNT(*) "
                f"FROM documents {where} GROUP BY 1,2 ORDER BY 3 DESC", params
            )
            print("\nfinal histogram:")
            for s, r, n in cur.fetchall():
                print(f"   {s:<18}{(r or '-'):<26}{n:>6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
