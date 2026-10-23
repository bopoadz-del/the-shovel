"""Merge the PostgreSQL shards' measured test durations into one file.

Each shard starts from the committed ``.test_durations`` and pytest-split
writes the times it measured for its own tests back into its copy. A test's
fresh value is the one that differs from (or is missing in) the committed
file; this keeps those and the committed value for everything else.

    python scripts/merge_test_durations.py durations-*.json > test_durations.json

Commit the output as ``.test_durations`` to rebalance the shards.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def merge(baseline: dict, shards: list[dict]) -> dict:
    out = dict(baseline)
    for shard in shards:
        for test, seconds in shard.items():
            if test not in baseline or seconds != baseline[test]:
                out[test] = seconds
    return dict(sorted(out.items()))


def main(argv: list[str]) -> int:
    root = Path(__file__).resolve().parent.parent
    committed = root / ".test_durations"
    baseline = json.loads(committed.read_text(encoding="utf-8")) if committed.is_file() else {}
    shards = [json.loads(Path(p).read_text(encoding="utf-8")) for p in argv]
    json.dump(merge(baseline, shards), sys.stdout, indent=1)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
