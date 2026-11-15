#!/usr/bin/env python3
"""Fail a pull request that deletes test files or loses test functions.

A pull request FAILS this check when the change it would make to the base
branch does either of these things:

* deletes a file that lives under a ``tests`` directory (any path with a
  ``tests`` component; a rename that leaves the file under ``tests`` is not
  a deletion, a rename that moves it out is);
* removes more test functions than it adds, counted over the Python files
  under ``tests`` that the change touches. A test function is a ``def`` or
  ``async def`` whose name starts with ``test``, keyed by file and enclosing
  class, so renaming one test is one removed plus one added and passes.

It PASSES anyway when the pull request body carries a line

    TEST REMOVAL APPROVED <sha>

where ``<sha>`` is the pull request head commit (full, or a prefix of at
least 7 hex characters). A new push changes the head, so an approval is
for one exact head and must be renewed after it.

The change is measured between ``--base`` and ``--head``. In CI those are
the first parent of the pull request merge ref and the merge ref itself:
exactly what a squash merge would apply to the base branch, so a branch
cut from a stale base that silently drops files added to the base since is
caught as well.

Stdlib only. Exit 0 = pass, 1 = fail, 2 = could not evaluate (fails CI).
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import PurePosixPath

APPROVAL_RE = re.compile(
    r"^[ \t>*_`-]*TEST REMOVAL APPROVED[ \t]+([0-9a-fA-F]{7,40})\b",
    re.MULTILINE,
)
_DEF_RE = re.compile(r"^[ \t]*(?:async[ \t]+)?def[ \t]+(test\w*)", re.MULTILINE)


@dataclass(frozen=True)
class FileChange:
    """One entry of ``git diff --name-status -M``: A, M, D, R, C, T ..."""

    status: str
    old_path: str | None
    new_path: str | None


@dataclass
class Verdict:
    deleted_test_files: list[str] = field(default_factory=list)
    removed_tests: list[str] = field(default_factory=list)
    added_tests: list[str] = field(default_factory=list)
    approved: bool = False

    @property
    def removes_tests(self) -> bool:
        return bool(self.deleted_test_files) or len(self.removed_tests) > len(
            self.added_tests
        )

    @property
    def ok(self) -> bool:
        return self.approved or not self.removes_tests


def is_test_path(path: str | None) -> bool:
    if not path:
        return False
    return "tests" in PurePosixPath(path.replace("\\", "/")).parts[:-1]


def count_test_functions(source: str | None) -> Counter[str]:
    """Qualified names of test functions defined in ``source``."""
    if not source:
        return Counter()
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        # A file that does not parse still had tests in it; count the defs.
        return Counter(_DEF_RE.findall(source))
    names: Counter[str] = Counter()

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                walk(child, f"{prefix}{child.name}.")
            elif isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef)
            ) and child.name.startswith("test"):
                names[f"{prefix}{child.name}"] += 1

    walk(tree, "")
    return names


def approval_matches(body: str | None, head_sha: str | None) -> bool:
    if not body or not head_sha:
        return False
    head = head_sha.strip().lower()
    return any(head.startswith(token.lower()) for token in APPROVAL_RE.findall(body))


def evaluate(
    changes: Iterable[FileChange],
    read_old,
    read_new,
    body: str | None,
    head_sha: str | None,
) -> Verdict:
    """Pure decision. ``read_old`` / ``read_new`` map a path to its text."""
    verdict = Verdict()
    for ch in changes:
        kind = ch.status[:1]
        old_is_test = is_test_path(ch.old_path)
        new_is_test = is_test_path(ch.new_path)
        if old_is_test and (kind == "D" or (kind == "R" and not new_is_test)):
            verdict.deleted_test_files.append(ch.old_path)

        old_py = old_is_test and (ch.old_path or "").endswith(".py") and kind != "A"
        new_py = new_is_test and (ch.new_path or "").endswith(".py") and kind != "D"
        if not (old_py or new_py):
            continue
        before = count_test_functions(read_old(ch.old_path)) if old_py else Counter()
        after = count_test_functions(read_new(ch.new_path)) if new_py else Counter()
        where_old = ch.old_path or ch.new_path
        where_new = ch.new_path or ch.old_path
        for name, n in (before - after).items():
            verdict.removed_tests.extend([f"{where_old}::{name}"] * n)
        for name, n in (after - before).items():
            verdict.added_tests.extend([f"{where_new}::{name}"] * n)
    verdict.approved = approval_matches(body, head_sha)
    return verdict


# --- git plumbing -----------------------------------------------------------


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    ).stdout


def parse_name_status(text: str) -> list[FileChange]:
    out: list[FileChange] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        status = parts[0]
        if status[:1] in ("R", "C") and len(parts) >= 3:
            out.append(FileChange(status, parts[1], parts[2]))
        elif status[:1] == "D":
            out.append(FileChange(status, parts[1], None))
        elif status[:1] == "A":
            out.append(FileChange(status, None, parts[1]))
        else:
            out.append(FileChange(status, parts[1], parts[1]))
    return out


def git_changes(base: str, head: str) -> list[FileChange]:
    return parse_name_status(_git("diff", "--name-status", "-M", base, head))


def _reader(rev: str):
    def read(path: str | None) -> str | None:
        # Paths come from the diff of this same pair of revisions, so a
        # failed read is a broken checkout: it propagates and main() exits 2.
        return _git("show", f"{rev}:{path}") if path else None

    return read


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", required=True, help="revision the PR merges into")
    ap.add_argument("--head", required=True, help="revision of the merged result")
    ap.add_argument(
        "--head-sha",
        default=os.environ.get("PR_HEAD_SHA", ""),
        help="PR head commit the approval line must name (env PR_HEAD_SHA)",
    )
    ap.add_argument(
        "--body-file",
        help="file holding the PR body (default: env PR_BODY)",
    )
    args = ap.parse_args(argv)

    if args.body_file:
        with open(args.body_file, encoding="utf-8") as fh:
            body = fh.read()
    else:
        body = os.environ.get("PR_BODY", "")

    try:
        changes = git_changes(args.base, args.head)
        v = evaluate(
            changes, _reader(args.base), _reader(args.head), body, args.head_sha
        )
    except subprocess.CalledProcessError as exc:
        print(f"::error::could not read {args.base}..{args.head}: {exc.stderr}")
        return 2

    for path in v.deleted_test_files:
        print(f"deleted test file: {path}")
    for name in v.removed_tests:
        print(f"removed test: {name}")
    print(
        f"test files deleted: {len(v.deleted_test_files)}; "
        f"test functions removed: {len(v.removed_tests)}, added: {len(v.added_tests)}"
    )
    if not v.removes_tests:
        print("No test removal.")
        return 0
    if v.approved:
        print(f"Test removal APPROVED for head {args.head_sha}.")
        return 0
    print(
        "::error::This pull request deletes test files or removes more test "
        "functions than it adds. If that is intended, put the line "
        f"'TEST REMOVAL APPROVED {args.head_sha or '<head sha>'}' in the PR "
        "body (it must name the current head commit)."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
