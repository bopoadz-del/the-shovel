"""Purge guard: commits carrying history removed by the rewrite are rejected.

Synthetic: a throwaway git repository, invented file names and contents.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

import scripts.check_purged_objects as guard


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                          text=True).stdout.strip()


def _commit(repo: Path, name: str, content: str, msg: str) -> str:
    (repo / name).parent.mkdir(parents=True, exist_ok=True)
    (repo / name).write_bytes(content.encode("utf-8"))
    _git(repo, "add", name)
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-m", msg)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path, monkeypatch):
    r = tmp_path / "r"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    _git(r, "config", "core.autocrlf", "false")
    base = _commit(r, "README.md", "synthetic\n", "base")
    lists = r / "config" / "purge"
    lists.mkdir(parents=True)
    monkeypatch.setattr(guard, "ROOT", r)
    monkeypatch.setattr(guard, "LISTS", lists)
    return r, base, lists


def _lists(lists: Path, commits=(), blobs=(), paths=(), salt="5a1t"):
    (lists / "pre_rewrite_commits.txt").write_text("# test\n" + "\n".join(commits) + "\n", encoding="utf-8")
    (lists / "purged_blobs.txt").write_text("# test\n" + "\n".join(blobs) + "\n", encoding="utf-8")
    (lists / "purged_paths.txt").write_text(
        f"# test\n# salt {salt}\n" + "\n".join(hashlib.sha256(f"{salt}|{p}".encode()).hexdigest() for p in paths)
        + "\n", encoding="utf-8")


def test_a_clean_push_passes(repo):
    r, base, lists = repo
    _lists(lists)
    head = _commit(r, "docs/note.md", "an invented note\n", "add note")
    assert guard.check(base, head) == []


def test_a_pre_rewrite_commit_anywhere_in_the_ancestry_is_rejected(repo):
    r, base, lists = repo
    old = _commit(r, "docs/old.md", "invented old content\n", "old work")
    head = _commit(r, "docs/new.md", "invented new content\n", "new work")
    _lists(lists, commits=[old])
    problems = guard.check(old, head)  # the old commit is below the pushed range
    assert any("pre-rewrite commit" in p for p in problems)


def test_a_removed_blob_is_rejected_even_under_a_new_name(repo):
    r, base, lists = repo
    blob = subprocess.run(["git", "-C", str(r), "hash-object", "--stdin"], input=b"Morrow Varga drawing\n",
                          capture_output=True, check=True).stdout.decode().strip()
    _lists(lists, blobs=[blob])
    head = _commit(r, "renamed/anything.txt", "Morrow Varga drawing\n", "re-add")
    assert any("content removed" in p for p in guard.check(base, head))


def test_a_removed_path_is_rejected(repo):
    r, base, lists = repo
    _lists(lists, paths=["uploads/fenwick_reach_schedule.pdf"])
    head = _commit(r, "uploads/fenwick_reach_schedule.pdf", "different bytes\n", "re-add path")
    assert any("path removed" in p for p in guard.check(base, head))


def test_a_new_branch_is_checked_against_main(repo):
    r, base, lists = repo
    _lists(lists, paths=["uploads/fenwick_reach_schedule.pdf"])
    _git(r, "update-ref", "refs/remotes/origin/main", base)
    head = _commit(r, "uploads/fenwick_reach_schedule.pdf", "x\n", "on a new branch")
    assert guard.check(guard.ZERO, head)
    assert guard.main([base, head]) == 1
    assert guard.main([base, base]) == 0
