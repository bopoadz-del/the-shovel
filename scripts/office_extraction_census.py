#!/usr/bin/env python3
"""S11: ledger census for .pptx / .xlsx / .pdf.

Read-only. Tallies INDEXED / TEXT_SPARSE / UNVERIFIED / ZERO_CHUNK /
EXTRACT_FAILED, thin and single-chunk rates, missing source (no local
bytes and no Drive pointer), and extractor_version.

Does not re-extract, fetch Drive, run p1b, or --apply-reingest.
The docx content-control walker is scripts/extraction_census.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main(argv: list[str] | None = None) -> int:
    from app.core.ingest_reconcile import (
        office_extraction_census,
        render_office_census_markdown,
    )

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", default=None)
    ap.add_argument(
        "--output",
        default=None,
        help="Markdown path (default artifacts/OFFICE_EXTRACTION_CENSUS.md)",
    )
    args = ap.parse_args(argv)
    report = office_extraction_census(project_id=args.project)
    md = render_office_census_markdown(report)
    dest = Path(args.output) if args.output else (REPO / "artifacts" / "OFFICE_EXTRACTION_CENSUS.md")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(md, encoding="utf-8")
    print(
        f"[office-census] docs={report['documents_total']} "
        f"missing_source={report['missing_source']} "
        f"indexed_zero_chunk={report['indexed_zero_chunk']} "
        f"out={dest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
