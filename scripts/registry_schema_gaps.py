#!/usr/bin/env python3
"""Registry schema gaps: list them, or shrink the baseline after filling some.

    python scripts/registry_schema_gaps.py            # what is still missing
    python scripts/registry_schema_gaps.py --shrink   # drop filled gaps from the baseline

The baseline (tests/fixtures/registry_schema_gaps.json) may only shrink:
--shrink removes entries that are no longer gaps and never adds one. A new
gap is fixed by declaring the field, not by listing it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASELINE = ROOT / "tests" / "fixtures" / "registry_schema_gaps.json"


def main(argv: list[str]) -> int:
    from app.lib.formula_registry import schema_gaps

    gaps = set(schema_gaps())
    base = set(json.loads(BASELINE.read_text(encoding="utf-8"))) if BASELINE.exists() else set()
    new = sorted(gaps - base)
    if "--shrink" in argv:
        kept = sorted(base & gaps)
        BASELINE.write_text(json.dumps(kept, indent=1) + "\n", encoding="utf-8")
        print(f"baseline {len(base)} -> {len(kept)}")
    print(f"gaps {len(gaps)}; not in baseline (fix these): {len(new)}")
    for g in new:
        print("  ", g)
    return 1 if new else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
