#!/usr/bin/env python3
"""Print ``build_sha`` from a /livez JSON body, or a blank line.

A missing file, invalid JSON, a non-object body, and a missing or empty
``build_sha`` all print nothing and exit 0. The caller decides whether an
empty value is a failure.

This lives in a file so a workflow ``run: |`` block never has to embed a
multi-line ``python3 -c`` script. Continuation lines of that script sit at
column 0, and a literal block scalar cannot contain a non-empty line less
indented than its first content line. GitHub then rejects the workflow
before any job starts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def build_sha_from_livez(raw: str) -> str:
    """Return the stripped build_sha, or "" when the body has none."""
    try:
        payload = json.loads(raw or "{}")
    except json.JSONDecodeError:
        print("livez body is not JSON", file=sys.stderr)
        return ""
    if not isinstance(payload, dict):
        return ""
    value = payload.get("build_sha") or ""
    return str(value).strip()


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    path = Path(args[0] if args else "/tmp/livez.json")
    raw = path.read_text(encoding="utf-8") if path.is_file() else ""
    print(build_sha_from_livez(raw))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
