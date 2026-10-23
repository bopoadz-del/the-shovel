#!/usr/bin/env python3
"""Hide older uploads behind the newest copy of the same document.

Wraps ``app.core.projects.supersede_older_duplicates``: within a project,
visible documents whose names agree once case, spacing and copy markers are
ignored are copies of one document; the newest upload stays live and every
older copy gets ``superseded_by`` + ``retrieval_visible=false``. Never deletes;
reversible by flipping ``retrieval_visible`` back.

Dry-run by default (prints the planned pairs). ``--apply`` writes them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", default=None, help="limit to one project id")
    ap.add_argument("--apply", action="store_true", help="write the supersede pairs")
    args = ap.parse_args(argv)
    from app.core.projects import supersede_older_duplicates

    report = supersede_older_duplicates(args.project, apply=args.apply)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
