#!/usr/bin/env python3
"""Write the facts of every catalogue file from the code.

    python scripts/build_catalogue_json.py            # write facts, keep hints
    python scripts/build_catalogue_json.py --check    # exit 1 where files and code disagree
    python scripts/build_catalogue_json.py --prune    # also remove files the code registers nothing for

One file per registered formula, discipline kernel, knowledge entry and
retrieval tool, under ``catalogue/``. Only ``facts`` is written here; a
file's ``hints`` are hand-written and are kept as they are. The work is in
``app.agents.driver.catalogue_files``.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    from app.agents.driver import catalogue_files

    if "--check" in argv:
        found = catalogue_files.drift()
        for line in found:
            print(line)
        print(f"{len(found)} difference(s) between the files and the code")
        return 1 if found else 0
    report = catalogue_files.write(prune="--prune" in argv)
    for rel in report["written"]:
        print("written", rel)
    for rel in report["extra"]:
        print("extra (the code registers nothing for it; --prune removes it)", rel)
    for rel in report["removed"]:
        print("removed", rel)
    gaps = catalogue_files.hint_gaps()
    print(f"{len(report['written'])} written, {report['unchanged']} unchanged, "
          f"{len(report['extra'])} extra, {len(report['removed'])} removed; "
          f"{len(gaps)} file(s) without a written hint")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
