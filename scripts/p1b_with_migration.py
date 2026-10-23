#!/usr/bin/env python3
"""Wrapper: run migration 0009, then start the P1b ingestion job."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ultralytics (safety_world YOLO) tries to write under ~/.config/Ultralytics;
# the Render worker runs as a non-writable home user and spam-warns otherwise.
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts import run_migration_0009
from scripts.p1b_ingest_drive_server import main as p1b_main


if __name__ == "__main__":
    migration_rc = run_migration_0009.main()
    if migration_rc != 0:
        raise SystemExit(migration_rc)
    raise SystemExit(p1b_main())
