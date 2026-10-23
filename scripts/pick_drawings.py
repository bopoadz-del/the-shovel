"""Pick 10 drawings covering 6+ disciplines, mixing 200 and 600 series.

Strategy:
- Group rows by (discipline, series).
- Walk top disciplines by count, pull one 200-series sheet + one 600-series.
- Stop when we have >=10 and >=6 disciplines.
- Within a (discipline, series), prefer entries whose filename ends in -A.pdf,
  then -B.pdf, then -C.pdf, then anything.

Writes the chosen 10 (or fewer if subset too small) to:
  data/logs/drawings_pilot_chosen.jsonl
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ABS = os.path.join(_REPO_ROOT, r"data\logs\drawings_pilot_inventory.jsonl")
DST_ABS = os.path.join(_REPO_ROOT, r"data\logs\drawings_pilot_chosen.jsonl")

DWG_NUM_RE = re.compile(
    r"QZ-ENG-\d+-\d+-EXC-DWG-([A-Z]{2,4})-(\d{3})-",
    re.IGNORECASE,
)


def parse(name: str):
    m = DWG_NUM_RE.search(name)
    if m:
        return m.group(1).upper(), m.group(2)
    return "UNKNOWN", ""


def rev_rank(name: str) -> int:
    """Lower is preferred. Prefer -A,-B,-C suffixes to anchor 'revision' diversity."""
    n = name.upper().replace(".PDF", "")
    if n.endswith("-A"):
        return 0
    if n.endswith("-B"):
        return 1
    if n.endswith("-C"):
        return 2
    return 3


def main() -> int:
    rows = []
    with open(SRC_ABS, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    buckets: dict = defaultdict(list)
    for r in rows:
        name = os.path.basename(r["path"])
        disc, series = parse(name)
        buckets[(disc, series)].append(r)
    # Sort each bucket by rev_rank then size (prefer non-trivial files >50KB)
    for k, v in buckets.items():
        v.sort(key=lambda r: (rev_rank(os.path.basename(r["path"])), -r["size"]))

    # Discipline order: by total count desc, but bias to ensure coverage
    disc_counts: dict = defaultdict(int)
    for r in rows:
        d, _ = parse(os.path.basename(r["path"]))
        disc_counts[d] += 1

    disc_order = sorted(disc_counts.keys(), key=lambda d: (-disc_counts[d], d))

    # Series priority: 200 first, then 600, then 400, then 100, 300, 500, 700
    series_priority = ["200", "600", "400", "100", "300", "500", "700", ""]

    chosen = []
    chosen_disc = set()
    seen_paths = set()

    # Pass 1: one per discipline (prefer 200-series), to maximize discipline coverage
    for disc in disc_order:
        if len(chosen) >= 10:
            break
        for series in series_priority:
            bucket = buckets.get((disc, series), [])
            if bucket:
                row = bucket[0]
                if row["path"] in seen_paths:
                    continue
                chosen.append(row)
                chosen_disc.add(disc)
                seen_paths.add(row["path"])
                break

    # Pass 2: fill remaining slots with 600-series from already-picked disciplines
    if len(chosen) < 10:
        for disc in disc_order:
            if len(chosen) >= 10:
                break
            for series in ["600", "400", "100"]:
                bucket = buckets.get((disc, series), [])
                for row in bucket:
                    if row["path"] in seen_paths:
                        continue
                    chosen.append(row)
                    seen_paths.add(row["path"])
                    break
                if len(chosen) >= 10:
                    break

    # Pass 3: anything left
    if len(chosen) < 10:
        for r in rows:
            if r["path"] in seen_paths:
                continue
            chosen.append(r)
            seen_paths.add(r["path"])
            if len(chosen) >= 10:
                break

    chosen = chosen[:10]

    os.makedirs(os.path.dirname(DST_ABS) or ".", exist_ok=True)
    with open(DST_ABS, "w", encoding="utf-8") as fh:
        for r in chosen:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"chosen: {len(chosen)} drawings, disciplines covered: {len(chosen_disc)}")
    print(f"written: {DST_ABS}\n")
    for r in chosen:
        name = os.path.basename(r["path"])
        disc, series = parse(name)
        exists = os.path.exists(r["path"])
        print(f"  [{disc:6s} {series:3s}] size={r['size']:>9d} exists={exists}  {name}")
        print(f"          {r['path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
