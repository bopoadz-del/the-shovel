"""Filter drive_inventory.jsonl to drawing PDFs and print a discipline summary.

Drawings are matched by:
  - path contains "02-Drawings" (case-insensitive), OR
  - path contains "/Drawings/" or "\\Drawings\\" (case-insensitive), OR
  - filename matches "*-DWG-*" (case-insensitive)

Discipline code parsed from EXC pattern:
  QZ-ENG-310-0000-EXC-DWG-<DISCIPLINE>-...
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter

SRC = r"data\logs\drive_inventory.jsonl"
DST = r"data\logs\drawings_pilot_inventory.jsonl"

# Repo-root relative paths
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ABS = os.path.join(_REPO_ROOT, SRC)
DST_ABS = os.path.join(_REPO_ROOT, DST)

DWG_NUM_RE = re.compile(
    r"QZ-ENG-\d+-\d+-EXC-DWG-([A-Z]{2,4})-",
    re.IGNORECASE,
)


def is_drawing(path: str) -> bool:
    norm = path.replace("\\", "/").lower()
    name = os.path.basename(norm)
    if "02-drawings" in norm:
        return True
    if "/drawings/" in norm:
        return True
    if "-dwg-" in name:
        return True
    if name.startswith("dwg-"):
        return True
    return False


def discipline_of(path: str) -> str:
    name = os.path.basename(path)
    m = DWG_NUM_RE.search(name)
    if m:
        return m.group(1).upper()
    return "UNKNOWN"


def sheet_series_of(path: str) -> str:
    """Return -NNN- 3-digit token after the discipline code, or empty."""
    name = os.path.basename(path)
    m = re.search(r"QZ-ENG-\d+-\d+-EXC-DWG-[A-Z]{2,4}-(\d{3})-", name, re.IGNORECASE)
    if m:
        return m.group(1)
    return ""


def main() -> int:
    if not os.path.exists(SRC_ABS):
        print(f"missing inventory: {SRC_ABS}", file=sys.stderr)
        return 2
    kept = []
    only_pdf_drops = 0
    with open(SRC_ABS, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            path = row.get("path") or ""
            ext = (row.get("ext") or "").lower()
            if not is_drawing(path):
                continue
            if ext != ".pdf":
                only_pdf_drops += 1
                continue
            kept.append(row)

    os.makedirs(os.path.dirname(DST_ABS) or ".", exist_ok=True)
    with open(DST_ABS, "w", encoding="utf-8") as fh:
        for row in kept:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"drawings_pilot_inventory written: {DST_ABS}")
    print(f"total drawing PDFs: {len(kept)}")
    print(f"non-pdf drawing-shaped entries dropped: {only_pdf_drops}")

    disc_counts = Counter(discipline_of(r["path"]) for r in kept)
    print("\ndiscipline breakdown:")
    for disc, n in disc_counts.most_common():
        print(f"  {disc:10s} {n}")

    series_counts = Counter(sheet_series_of(r["path"]) or "(none)" for r in kept)
    print("\nsheet-series breakdown (first 10):")
    for series, n in series_counts.most_common(10):
        print(f"  {series:10s} {n}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
