#!/usr/bin/env python3
"""Rewrite documents.chunk_count from the live chunk table.

Rows can sit at chunk_count=0 after a successful index because older
paths never stamped the ledger. /health already counts chunks with a
query; this script makes the column catch up so the two can be compared.

Never deletes a documents row. Dry-run by default; ``--apply`` writes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="Write (default: dry run)")
    ap.add_argument("--project", default=None, help="Limit to one project_id")
    args = ap.parse_args(argv)

    from app.core.projects import backfill_chunk_counts_from_table

    report = backfill_chunk_counts_from_table(
        project_id=args.project, apply=args.apply,
    )
    print(
        f"[backfill] scanned={report['scanned']} stale={report['stale']} "
        f"updated={report['updated']} apply={args.apply} "
        f"table={report['table']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
