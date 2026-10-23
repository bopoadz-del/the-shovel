#!/usr/bin/env python3
"""Thin wrapper — prefer scripts/rag_render_bulk_ingest.py.

Kept so existing launchers that call _run_render_bulk_ingest.py keep working.
Defaults to --resume and NEVER wipes unless --destructive-wipe is passed.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.rag_render_bulk_ingest import main  # noqa: E402


if __name__ == "__main__":
    # If invoked with no args, default to --resume (safe).
    argv = list(sys.argv[1:])
    if not argv:
        argv = ["--resume"]
    raise SystemExit(main(argv))
