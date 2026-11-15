"""Guard against pull requests that delete tests (scripts/check_test_removal.py).

Every case builds its own throwaway git repository, makes a base commit and
a change commit, and runs the real script over ``base..head`` in a
subprocess, exactly as the pr-quality job does. Each decision is shown in
both directions (a removal fails, a normal change passes, the approval line
for the current head passes, an approval for another commit does not).

Each test names the mutation it kills.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "check_test_removal.py"
_SPEC = importlib.util.spec_from_file_location("check_test_removal", _SCRIPT)
guard = importlib.util.module_from_spec(_SPEC)
sys.modules["check_test_removal"] = guard
_SPEC.loader.exec_module(guard)


def _git(repo: Path, *args: str) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "guard-test",
        "GIT_AUTHOR_EMAIL": "guard-test@example.invalid",
        "GIT_COMMITTER_NAME": "guard-test",
        "GIT_COMMITTER_EMAIL": "guard-test@example.invalid",
    }
    return subprocess.run(
        ["git", "-c", "commit.gpgsign=false", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    ).stdout.strip()


def _write(repo: Path, files: dict[str, str | None]) -> None:
    for rel, text in files.items():
        p = repo / rel
        if text is None:
            p.unlink()
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")


def _repo(tmp_path: Path, base: dict[str, str], change: dict[str, str | None]):
    """Return (repo, base_sha, head_sha) with ``change`` applied on ``base``."""
    repo = tmp_path / "r"
    repo.mkdir()
    _git(repo, "init", "-q")
    _write(repo, base)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    base_sha = _git(repo, "rev-parse", "HEAD")
    _write(repo, change)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "change")
    head_sha = _git(repo, "rev-parse", "HEAD")
    return repo, base_sha, head_sha


def _run(repo: Path, base: str, head: str, body: str, head_sha: str):
    env = {**os.environ, "PR_BODY": body, "PR_HEAD_SHA": head_sha}
    return subprocess.run(
        [sys.executable, str(_SCRIPT), "--base", base, "--head", head],
        cwd=repo,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def _tests(*names: str) -> str:
    return "".join(f"def {n}():\n    assert 1\n\n" for n in names)


BASE = {
    "app/thing.py": "X = 1\n",
    "tests/test_alpha.py": _tests("test_a1", "test_a2"),
    "tests/test_beta.py": _tests("test_b1"),
    "tests/data/sample.txt": "fixture\n",
}


# -- deleting a file under tests/ ---------------------------------------------


@pytest.mark.parametrize("deleted", ["tests/test_beta.py", "tests/data/sample.txt"])
def test_deleting_a_file_under_tests_fails(tmp_path, deleted):
    """Mutation killed: a guard that only counts functions, or only .py files."""
    repo, b, h = _repo(tmp_path, BASE, {deleted: None})
    r = _run(repo, b, h, body="Ordinary change.", head_sha=h)
    assert r.returncode == 1, r.stdout + r.stderr
    assert f"deleted test file: {deleted}" in r.stdout
    assert "TEST REMOVAL APPROVED" in r.stdout


def test_deleting_a_file_outside_tests_passes(tmp_path):
    """Mutation killed: a guard that fails on ANY deleted file."""
    repo, b, h = _repo(tmp_path, BASE, {"app/thing.py": None})
    r = _run(repo, b, h, body="Ordinary change.", head_sha=h)
    assert r.returncode == 0, r.stdout + r.stderr


def test_moving_a_test_out_of_tests_counts_as_deletion(tmp_path):
    """Mutation killed: renames ignored, so a move out of tests/ slips by."""
    repo, b, h = _repo(
        tmp_path,
        BASE,
        {
            "tests/test_alpha.py": None,
            "attic/test_alpha.py": BASE["tests/test_alpha.py"],
        },
    )
    r = _run(repo, b, h, body="Ordinary change.", head_sha=h)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "deleted test file: tests/test_alpha.py" in r.stdout


def test_renaming_a_test_file_inside_tests_passes(tmp_path):
    """Mutation killed: a rename within tests/ treated as a deletion."""
    repo, b, h = _repo(
        tmp_path,
        BASE,
        {
            "tests/test_alpha.py": None,
            "tests/unit/test_alpha.py": BASE["tests/test_alpha.py"],
        },
    )
    r = _run(repo, b, h, body="Ordinary change.", head_sha=h)
    assert r.returncode == 0, r.stdout + r.stderr


# -- removing more test functions than are added -------------------------------


def test_removing_more_test_functions_than_added_fails(tmp_path):
    """Mutation killed: a guard that only looks at whole-file deletions."""
    repo, b, h = _repo(tmp_path, BASE, {"tests/test_alpha.py": _tests("test_a1")})
    r = _run(repo, b, h, body="Ordinary change.", head_sha=h)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "removed test: tests/test_alpha.py::test_a2" in r.stdout


def test_removed_tests_offset_by_new_tests_elsewhere_passes(tmp_path):
    """Mutation killed: 'any removal fails' instead of 'net removal fails'."""
    repo, b, h = _repo(
        tmp_path,
        BASE,
        {
            "tests/test_alpha.py": _tests("test_a1"),
            "tests/test_gamma.py": _tests("test_g1"),
        },
    )
    r = _run(repo, b, h, body="Ordinary change.", head_sha=h)
    assert r.returncode == 0, r.stdout + r.stderr


def test_renaming_a_test_function_passes(tmp_path):
    """Mutation killed: removed counted but added not counted."""
    repo, b, h = _repo(
        tmp_path, BASE, {"tests/test_alpha.py": _tests("test_a1", "test_a2_renamed")}
    )
    r = _run(repo, b, h, body="Ordinary change.", head_sha=h)
    assert r.returncode == 0, r.stdout + r.stderr


def test_class_methods_and_async_tests_are_counted(tmp_path):
    """Mutation killed: only module-level sync ``def test_*`` are counted."""
    base = {
        **BASE,
        "tests/test_cls.py": (
            "class TestThing:\n"
            "    def test_one(self):\n        assert 1\n"
            "    async def test_two(self):\n        assert 1\n"
        ),
    }
    change = {
        "tests/test_cls.py": "class TestThing:\n    def test_one(self):\n        assert 1\n"
    }
    repo, b, h = _repo(tmp_path, base, change)
    r = _run(repo, b, h, body="Ordinary change.", head_sha=h)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "TestThing.test_two" in r.stdout


def test_normal_pr_that_adds_tests_and_code_passes(tmp_path):
    """Mutation killed: a guard that fails every PR touching tests/."""
    repo, b, h = _repo(
        tmp_path,
        BASE,
        {
            "app/thing.py": "X = 2\n",
            "tests/test_alpha.py": _tests("test_a1", "test_a2", "test_a3"),
        },
    )
    r = _run(repo, b, h, body="Ordinary change.", head_sha=h)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "No test removal." in r.stdout


# -- the approval line ---------------------------------------------------------


@pytest.mark.parametrize("width", [40, 7])
def test_approval_line_naming_the_head_passes(tmp_path, width):
    """Mutation killed: an approval line that is never honoured."""
    repo, b, h = _repo(tmp_path, BASE, {"tests/test_beta.py": None})
    body = f"Why: superseded.\n\nTEST REMOVAL APPROVED {h[:width]}\n"
    r = _run(repo, b, h, body=body, head_sha=h)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "APPROVED for head" in r.stdout


@pytest.mark.parametrize(
    "body_for",
    [
        lambda b, h: f"TEST REMOVAL APPROVED {b}",  # names another commit
        lambda b, h: f"TEST REMOVAL APPROVED {h[:6]}",  # too short to bind
        lambda b, h: "TEST REMOVAL APPROVED",  # no sha at all
        lambda b, h: f"test removal approved {h}",  # not the agreed line
    ],
)
def test_approval_line_not_naming_the_head_still_fails(tmp_path, body_for):
    """Mutation killed: any approval-ish text, or any sha, unlocks the gate."""
    repo, b, h = _repo(tmp_path, BASE, {"tests/test_beta.py": None})
    r = _run(repo, b, h, body=body_for(b, h), head_sha=h)
    assert r.returncode == 1, r.stdout + r.stderr


def test_stale_base_merge_that_drops_newer_tests_fails(tmp_path):
    """The squash a stale branch would apply: the merge result loses a test
    file that the base gained after the branch was cut.

    Mutation killed: comparing the branch to its own fork point instead of
    the merged result to the current base.
    """
    repo = tmp_path / "r"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _write(repo, BASE)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    _git(repo, "checkout", "-q", "-b", "feature")
    _write(repo, {"app/thing.py": "X = 3\n"})
    _git(repo, "commit", "-qam", "feature")
    _git(repo, "checkout", "-q", "main")
    _write(repo, {"tests/test_new.py": _tests("test_n1")})
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "main gains a test")
    # The feature side "resolves" by keeping its own tree: the merge result
    # no longer has tests/test_new.py.
    _git(repo, "merge", "-q", "-s", "ours", "--no-edit", "feature")
    _git(repo, "checkout", "-q", "feature", "--", "app/thing.py")
    _git(repo, "rm", "-q", "tests/test_new.py")
    _git(repo, "commit", "-q", "--amend", "--no-edit")
    merge = _git(repo, "rev-parse", "HEAD")
    head = _git(repo, "rev-parse", "feature")
    r = _run(repo, f"{merge}^1", merge, body="Ordinary change.", head_sha=head)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "deleted test file: tests/test_new.py" in r.stdout


def test_bad_revision_is_an_error_not_a_pass(tmp_path):
    """Mutation killed: a diff failure swallowed into exit 0."""
    repo, _b, h = _repo(tmp_path, BASE, {})
    r = _run(repo, "0" * 40, h, body="", head_sha=h)
    assert r.returncode == 2, r.stdout + r.stderr


# -- pure helpers ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("tests/test_x.py", True),
        ("frontend/tests/a.spec.ts", True),
        ("app/tests_helpers.py", False),
        ("tests", False),
        (None, False),
    ],
)
def test_is_test_path(path, expected):
    """Mutation killed: substring match on 'tests' instead of a path part."""
    assert guard.is_test_path(path) is expected


def test_unparseable_test_file_still_counts_its_tests():
    """Mutation killed: a syntax error hides every test in the file."""
    src = "def test_a():\n    pass\n\ndef test_b(:\n"
    assert sum(guard.count_test_functions(src).values()) == 2
