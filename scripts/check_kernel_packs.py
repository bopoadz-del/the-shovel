#!/usr/bin/env python3
"""Check every kernel pack (``catalogue_packs/<kernel>.json``) against the code.

    python scripts/check_kernel_packs.py            # exit 1 on any fault or stale line
    python scripts/check_kernel_packs.py --stamp    # re-stamp fingerprints; a changed line loses approved

The work is in ``app.agents.driver.kernel_packs``. A pack's words are
hand-written; this only checks them and stamps fingerprints.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    from app.agents.driver import kernel_packs

    if "--stamp" in argv:
        root = kernel_packs.packs_dir()
        for kernel, pack in kernel_packs.load(root).items():
            changed = kernel_packs.stamp(pack)
            for line in changed:
                print(f"{kernel}: re-stamped, no longer approved: {line}")
            if changed:
                path = os.path.join(root, kernel.partition(":")[2] + ".json")
                with open(path, "w", encoding="utf-8", newline="\n") as f:
                    f.write(json.dumps(pack, indent=2, ensure_ascii=False) + "\n")
        return 0
    found = kernel_packs.check()
    for line in found:
        print(line)
    print(f"{len(found)} fault(s) or stale line(s) in the kernel packs")
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
