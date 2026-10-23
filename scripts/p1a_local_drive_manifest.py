#!/usr/bin/env python3
"""P1a fallback — enumerate the synced Google Drive folders locally.

When the service account cannot see the user's entire My Drive (no domain-wide
delegation), this script walks the local Google Drive for Desktop sync root
and emits a manifest of folder paths and file counts. The file list can be
used as the source-of-truth manifest for Phase 1b ingestion from local paths.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_ROOT = Path(os.environ.get("FORK_DRIVE_ROOT", "/data/example"))


def _walk_local(root: Path) -> Dict[str, Any]:
    folders: List[Dict[str, Any]] = []
    total_files = 0
    total_dirs = 0

    for dirpath, dirnames, filenames in os.walk(root):
        rel = Path(dirpath).relative_to(root)
        rel_str = str(rel).replace("\\", "/")
        if rel_str == ".":
            rel_str = "(root)"
        folders.append({
            "path": rel_str,
            "file_count": len(filenames),
            "subfolder_count": len(dirnames),
        })
        total_files += len(filenames)
        total_dirs += len(dirnames)

    return {
        "root": str(root),
        "total_files": total_files,
        "total_dirs": total_dirs,
        "folders": sorted(folders, key=lambda x: x["path"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Build local Drive sync manifest")
    ap.add_argument("--root", default=str(DEFAULT_ROOT), help="Local Google Drive root")
    ap.add_argument("--output", default="manifests/p1a_local_drive_manifest.json",
                    help="Output manifest path")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"ERROR: local Drive root not found: {root}", file=sys.stderr)
        return 1

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "local Google Drive sync",
        "root": str(root),
        **_walk_local(root),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    print(f"[p1a-local] manifest written to {out_path}", file=sys.stderr)
    print(
        f"[p1a-local] {manifest['total_files']} files in {manifest['total_dirs']} directories",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
