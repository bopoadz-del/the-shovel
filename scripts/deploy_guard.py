#!/usr/bin/env python3
"""Refuse a deploy that would ship anything but the head of main, or move live back.

deploy-aws fires on every green ``tests`` run on main. That run can belong to
an older commit, for example a ``--failed`` rerun that finishes after a newer
merge has already gone green and shipped. Without this check, the late run
ships the older commit and live goes backwards.

Two rules, both checked before anything is built or rolled:

1. The commit to deploy must be the current head of main.
2. It must not be a strict ancestor of the commit that is live now.
   Redeploying the live commit itself is allowed.

When /livez gives no build_sha, or the live commit is not in this repository's
history, rule 2 cannot be decided. The deploy then goes ahead on rule 1 alone,
with a warning: refusing would block the deploy that repairs a down site.

Usage (from the repository root, with origin/main fetched with history):

    python3 scripts/deploy_guard.py --deploy SHA --main-ref origin/main --live SHA

Exit 0 = deploy may proceed. Exit 1 = refused. Exit 2 = bad arguments.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable, Optional

# is_ancestor(a, b) -> True when a is an ancestor of b (or a == b), False when
# it is not, None when git cannot tell (an object is missing).
AncestorFn = Callable[[str, str], Optional[bool]]


@dataclass(frozen=True)
class Verdict:
    ok: bool
    reason: str
    warning: str = ""


def decide(deploy: str, main_head: str, live: str, is_ancestor: AncestorFn) -> Verdict:
    """Return whether ``deploy`` may ship, given main's head and the live sha."""
    deploy = deploy.strip().lower()
    main_head = main_head.strip().lower()
    live = live.strip().lower()
    if not deploy or not main_head:
        return Verdict(False, "the commit to deploy or the head of main is unknown")
    if deploy != main_head:
        return Verdict(
            False,
            f"{deploy} is not the head of main ({main_head}); only the head of main deploys",
        )
    if not live:
        return Verdict(True, "head of main", "live build_sha is unknown; the live check was not made")
    if live == deploy:
        return Verdict(True, "head of main, same commit as live")
    older = is_ancestor(deploy, live)
    if older is None:
        return Verdict(
            True,
            "head of main",
            f"live commit {live} is not in this repository's history; the live check was not made",
        )
    if older:
        return Verdict(False, f"{deploy} is older than the live commit {live}; live would go backwards")
    return Verdict(True, "head of main, not older than live")


def git_is_ancestor(a: str, b: str) -> Optional[bool]:
    proc = subprocess.run(
        ["git", "merge-base", "--is-ancestor", a, b],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0:
        return True
    if proc.returncode == 1:
        return False
    return None


def git_rev(ref: str) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--deploy", required=True, help="commit the run would ship")
    parser.add_argument("--main-ref", required=True, help="ref naming the head of main, e.g. origin/main")
    parser.add_argument("--live", default="", help="build_sha /livez reports now (may be empty)")
    args = parser.parse_args(argv)

    main_head = git_rev(args.main_ref)
    if not main_head:
        print(f"::error title=Deploy refused::cannot resolve {args.main_ref}")
        return 1
    verdict = decide(args.deploy, main_head, args.live, git_is_ancestor)
    if verdict.warning:
        print(f"::warning title=Deploy guard::{verdict.warning}")
    if not verdict.ok:
        print(f"::error title=Deploy refused::{verdict.reason}")
        return 1
    print(f"deploy guard: {args.deploy} may deploy ({verdict.reason})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
