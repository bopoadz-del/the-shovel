"""Hand-pick 10 drawings: 5 from 200-series, 5 from 600-series, 10 disciplines."""
from __future__ import annotations
import json
import os
import re

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ABS = os.path.join(_REPO_ROOT, r"data\logs\drawings_pilot_inventory.jsonl")
DST_ABS = os.path.join(_REPO_ROOT, r"data\logs\drawings_pilot_chosen.jsonl")

RE = re.compile(r"QZ-ENG-\d+-\d+-EXC-DWG-([A-Z]{2,4})-(\d{3})-", re.IGNORECASE)

# Explicit picks: 5 from 200-series, 5 from 600-series; 10 distinct disciplines;
# revision variety (-A, -B, -C, -04, -05).
TARGETS = [
    # discipline, series, preferred filename substring (revision/suffix anchor)
    ("TM", "200", "TM-200-0000001-A"),
    ("SW", "600", "SW-600-0000035-04"),
    ("SG", "200", "SG-200-1001000-A"),
    ("EL", "600", "EL-600-0200068-B"),
    ("LI", "200", "LI-200-1001000-A"),
    ("ST", "600", "ST-600-0000991-05"),
    ("WS", "600", "WS-600-0000001-C"),
    ("IR", "200", "IR-200-1001000-A"),
    ("TL", "600", "TL-600-0000002-D"),
    ("SE", "200", "SE-200-1001000-A"),
]


def main() -> int:
    rows = []
    with open(SRC_ABS, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    chosen = []
    for disc, series, anchor in TARGETS:
        hit = None
        for r in rows:
            name = os.path.basename(r["path"])
            if anchor.lower() in name.lower():
                hit = r
                break
        if hit is None:
            # fallback: first matching disc+series
            for r in rows:
                name = os.path.basename(r["path"])
                m = RE.search(name)
                if m and m.group(1).upper() == disc and m.group(2) == series:
                    hit = r
                    break
        if hit is None:
            print(f"NO MATCH for {disc}/{series}/{anchor}")
            continue
        chosen.append(hit)

    os.makedirs(os.path.dirname(DST_ABS) or ".", exist_ok=True)
    with open(DST_ABS, "w", encoding="utf-8") as fh:
        for r in chosen:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"chosen: {len(chosen)} drawings\nwritten: {DST_ABS}\n")
    for r in chosen:
        name = os.path.basename(r["path"])
        m = RE.search(name)
        disc = m.group(1).upper() if m else "?"
        series = m.group(2) if m else "?"
        exists = os.path.exists(r["path"])
        print(f"  [{disc:3s} {series}] size={r['size']:>10d} exists={exists}  {name}")
        print(f"          {r['path']}")
    return 0


if __name__ == "__main__":
    main()
