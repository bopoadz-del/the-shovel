"""Archive a project by exact name (soft, reversible) — one-off operator script.

Run as a one-off ECS task against the live task definition (which already
carries DATABASE_URL + the RAG namespace env), the same shape as
scripts/project_inventory.py. It never deletes: ``archive_project`` sets
``status='archived'``, which hides the project everywhere while keeping its
row and RAG chunks.
"""
from __future__ import annotations

import argparse
import os

from app.core import projects as store


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Archive a project by exact name")
    parser.add_argument("--name", required=False, help="exact project name to archive (or ARCHIVE_PROJECT_NAME env)")
    args = parser.parse_args(argv)
    name = args.name or os.environ.get("ARCHIVE_PROJECT_NAME", "").strip()
    if not name:
        print("::error::no name given (--name or ARCHIVE_PROJECT_NAME)")
        return 1

    row = store._find_active_project_by_name(name)  # noqa: SLF001 — script helper
    if row is None:
        print(f"::error::no active project named {name!r}")
        return 1

    pid = row["id"]
    if not store.archive_project(pid):
        print(f"::error::archive_project returned False for id {pid}")
        return 1

    print(f"archived project {name!r} id={pid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
