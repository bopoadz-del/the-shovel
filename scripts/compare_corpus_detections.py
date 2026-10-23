"""Compare two corpus-detection JSONLs (e.g. V_final vs V2 on the same 206 photos).

Shows what changed: new detections, lost detections, confidence shifts per photo.
Run after `test_model_on_corpus.py` produces a JSONL for each model.

Usage:
    python scripts/compare_corpus_detections.py --before data/training/corpus_detections_v1_final.jsonl --after data/training/corpus_detections_v2.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _load(path: Path) -> dict:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rows[r["filename"]] = r["detections"]
    return rows


def _key(d: dict) -> tuple:
    return (d["class"], tuple(int(x) for x in d["bbox"]))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--before", type=Path, required=True)
    p.add_argument("--after", type=Path, required=True)
    args = p.parse_args()

    before = _load(args.before)
    after = _load(args.after)

    all_files = sorted(set(before) | set(after))
    print(f"comparing {len(all_files)} photos across {args.before.name} -> {args.after.name}\n")

    before_class_total = Counter()
    after_class_total = Counter()
    new_per_class = Counter()
    lost_per_class = Counter()

    photos_with_changes = 0
    for fname in all_files:
        b = before.get(fname, [])
        a = after.get(fname, [])
        b_keys = {_key(d) for d in b}
        a_keys = {_key(d) for d in a}
        for d in b:
            before_class_total[d["class"]] += 1
        for d in a:
            after_class_total[d["class"]] += 1
        for d in b:
            if _key(d) not in a_keys:
                lost_per_class[d["class"]] += 1
        for d in a:
            if _key(d) not in b_keys:
                new_per_class[d["class"]] += 1
        if b_keys != a_keys:
            photos_with_changes += 1

    print(f"photos with any change: {photos_with_changes}/{len(all_files)}\n")
    print(f"{'class':35s}  {'before':>8s}  {'after':>8s}  {'delta':>8s}  {'new':>5s}  {'lost':>5s}")
    print("  ".join(["-" * 35, "-" * 8, "-" * 8, "-" * 8, "-" * 5, "-" * 5]))
    all_classes = sorted(set(before_class_total) | set(after_class_total))
    for cls in all_classes:
        b = before_class_total[cls]
        a = after_class_total[cls]
        delta = a - b
        new = new_per_class[cls]
        lost = lost_per_class[cls]
        print(f"{cls:35s}  {b:8d}  {a:8d}  {delta:+8d}  {new:5d}  {lost:5d}")
    print("\nDefinitions:")
    print("  before/after = total detections in each run")
    print("  delta        = net change (after - before)")
    print("  new          = detections in 'after' that weren't in 'before' (same bbox+class)")
    print("  lost         = detections in 'before' that aren't in 'after'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
