#!/usr/bin/env python3
"""Build a focused manifest for one local Drive folder.

Usage:
  python scripts/p1a_focused_manifest.py --folder "construction-3-001"
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


def _walk_folder(root: Path, folder: str) -> Dict[str, Any]:
    target = root / folder
    if not target.exists():
        raise FileNotFoundError(f"Folder not found: {target}")

    files: List[Dict[str, Any]] = []
    subfolders: List[Dict[str, Any]] = []

    for dirpath, dirnames, filenames in os.walk(target):
        rel = Path(dirpath).relative_to(target)
        rel_str = str(rel).replace("\\", "/")
        if rel_str == ".":
            rel_str = ""
        subfolders.append({
            "path": rel_str or "(root)",
            "file_count": len(filenames),
        })
        for name in filenames:
            fpath = Path(dirpath) / name
            files.append({
                "name": name,
                "relative_path": str((rel / name)).replace("\\", "/"),
                "size": fpath.stat().st_size,
                "extension": Path(name).suffix.lower(),
            })

    return {
        "folder_name": folder,
        "local_path": str(target),
        "total_files": len(files),
        "subfolders": sorted(subfolders, key=lambda x: x["path"]),
        "files": sorted(files, key=lambda x: x["relative_path"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Build focused local folder manifest")
    ap.add_argument("--folder", required=True, help="Folder name under the Drive root")
    ap.add_argument("--root", default=str(DEFAULT_ROOT), help="Local Drive root")
    ap.add_argument("--output", help="Output path (default: manifests/p1a_<folder>_manifest.json)")
    args = ap.parse_args()

    root = Path(args.root)
    out_path = Path(args.output or f"manifests/p1a_{args.folder.replace(' ', '_').replace('/', '_')}_manifest.json")

    try:
        summary = _walk_folder(root, args.folder)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "local Google Drive sync",
        **summary,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    print(f"[p1a-focus] manifest written to {out_path}", file=sys.stderr)
    print(
        f"[p1a-focus] {args.folder}: {manifest['total_files']} files",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
