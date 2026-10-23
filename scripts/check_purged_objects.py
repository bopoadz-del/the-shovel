"""Purge guard: reject commits that carry history removed by the 2026-10 rewrite.

The history rewrite removed real project documents and identifiers. An old
clone still holds them; pushing it back would bring them back. This check runs
on every push and pull request and fails when a commit in the pushed range

* is one of the pre-rewrite commits (config/purge/pre_rewrite_commits.txt), or
* contains a blob that the rewrite removed (config/purge/purged_blobs.txt), or
* adds a file at a path the rewrite removed (config/purge/purged_paths.txt,
  salted SHA-256 digests -- the paths themselves are not stored).

All three lists hold hashes only. ``main`` requires this check (protect-main
ruleset), so a branch carrying old history can never merge.

    python scripts/check_purged_objects.py <base-ref-or-sha> <head-ref-or-sha>
    (base all zeros -- a new branch -- checks every commit not on origin/main)
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LISTS = ROOT / "config" / "purge"
ZERO = "0" * 40


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True).stdout.decode("utf-8", "replace")


def _load(name: str) -> set[str]:
    p = LISTS / name
    if not p.is_file():
        return set()
    return {ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")}


def _salt() -> str:
    for ln in (LISTS / "purged_paths.txt").read_text(encoding="utf-8").splitlines():
        if ln.startswith("# salt "):
            return ln.split()[2]
    raise SystemExit("config/purge/purged_paths.txt: no salt line")


def path_digest(salt: str, path: str) -> str:
    return hashlib.sha256(f"{salt}|{path}".encode()).hexdigest()


def _base(base: str) -> str:
    """The push's old tip, or origin/main for a new branch or a tip this
    checkout cannot resolve (a force-push replaced it)."""
    if not base or base == ZERO:
        return "origin/main"
    ok = subprocess.run(["git", "cat-file", "-e", f"{base}^{{commit}}"], cwd=ROOT,
                        capture_output=True).returncode == 0
    return base if ok else "origin/main"


def commits_in_range(base: str, head: str) -> list[str]:
    return _git("rev-list", f"{_base(base)}..{head}").split()


def check(base: str, head: str) -> list[str]:
    old_commits = _load("pre_rewrite_commits.txt")
    blobs = _load("purged_blobs.txt")
    paths = _load("purged_paths.txt")
    salt = _salt() if paths else ""
    problems: list[str] = []
    commits = commits_in_range(base, head)
    # Any pre-rewrite commit anywhere in head's ancestry means an old clone,
    # whatever the range -- including a push straight to main.
    for c in _git("rev-list", head).split():
        if c in old_commits:
            problems.append(f"{c[:12]}: a pre-rewrite commit -- this is an old clone; re-clone and re-apply your change")
    if commits:
        # Objects the pushed commits introduce: reachable from head, not from base.
        for line in _git("rev-list", "--objects", head, "--not", _base(base)).splitlines():
            sha, _, path = line.partition(" ")
            if sha in blobs:
                problems.append(f"{sha[:12]} ({path or 'blob'}): content removed by the history rewrite")
            elif path and salt and path_digest(salt, path) in paths:
                problems.append(f"{path}: a path removed by the history rewrite")
    return problems


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    problems = check(*argv)
    for p in problems:
        print(f"::error::{p}")
    if problems:
        print(f"{len(problems)} purged object(s) in the pushed commits.", file=sys.stderr)
        return 1
    print("purge guard: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
