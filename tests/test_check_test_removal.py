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
    "tests/test_cls.py": (
        "class TestThing:\n"
        "    def test_one(self):\n        assert 1\n"
        "    async def test_two(self):\n        assert 1\n"
    ),
}
_ALPHA = BASE["tests/test_alpha.py"]
_CLS_ONE = "class TestThing:\n    def test_one(self):\n        assert 1\n"


# -- one rule, both directions: (change, exit code, line the log must carry) ----
#
# Mutations killed, by row:
#   delete .py / delete data file -> a guard that only counts functions or .py
#   delete outside tests          -> a guard that fails on ANY deleted file
#   move out of tests             -> renames ignored, a move out slips by
#   rename inside tests           -> a rename within tests/ treated as deletion
#   net function loss             -> a guard that only sees whole-file deletes
#   loss offset elsewhere         -> "any removal fails" instead of net removal
#   rename a test function        -> removed counted but added not counted
#   class / async method lost     -> only module-level sync defs counted
#   normal code + tests PR        -> a guard that fails every PR touching tests/
SCENARIOS = [
    ({"tests/test_beta.py": None}, 1, "deleted test file: tests/test_beta.py"),
    ({"tests/data/sample.txt": None}, 1, "deleted test file: tests/data/sample.txt"),
    ({"app/thing.py": None}, 0, "No test removal."),
    (
        {"tests/test_alpha.py": None, "attic/test_alpha.py": _ALPHA},
        1,
        "deleted test file: tests/test_alpha.py",
    ),
    (
        {"tests/test_alpha.py": None, "tests/unit/test_alpha.py": _ALPHA},
        0,
        "No test removal.",
    ),
    (
        {"tests/test_alpha.py": _tests("test_a1")},
        1,
        "removed test: tests/test_alpha.py::test_a2",
    ),
    (
        {
            "tests/test_alpha.py": _tests("test_a1"),
            "tests/test_gamma.py": _tests("test_g1"),
        },
        0,
        "test functions removed: 1, added: 1",
    ),
    (
        {"tests/test_alpha.py": _tests("test_a1", "test_a2_renamed")},
        0,
        "test functions removed: 1, added: 1",
    ),
    (
        {"tests/test_cls.py": _CLS_ONE},
        1,
        "removed test: tests/test_cls.py::TestThing.test_two",
    ),
    (
        {
            "app/thing.py": "X = 2\n",
            "tests/test_alpha.py": _tests("test_a1", "test_a2", "test_a3"),
        },
        0,
        "No test removal.",
    ),
]


@pytest.mark.parametrize(("change", "code", "line"), SCENARIOS)
def test_change_without_approval(tmp_path, change, code, line):
    repo, b, h = _repo(tmp_path, BASE, change)
    r = _run(repo, b, h, body="Ordinary change.", head_sha=h)
    assert r.returncode == code, r.stdout + r.stderr
    assert line in r.stdout
    # A failing run always tells the author how to approve; a pass never does.
    assert ("TEST REMOVAL APPROVED" in r.stdout) == (code == 1)


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


# -- a stale branch merged onto a base that moved on -----------------------------


@pytest.mark.parametrize(("drop_newer_test", "code"), [(True, 1), (False, 0)])
def test_stale_base_merge(tmp_path, drop_newer_test, code):
    """Main gains a test after the branch was cut; the merge result either
    loses it (what a careless conflict resolution does) or keeps it.

    Mutation killed: comparing the branch to its own fork point instead of
    the merged result to the current base -- that sees no deletion at all.
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
    _git(repo, "merge", "-q", "--no-edit", "feature")
    if drop_newer_test:
        _git(repo, "rm", "-q", "tests/test_new.py")
        _git(repo, "commit", "-q", "--amend", "--no-edit")
    merge = _git(repo, "rev-parse", "HEAD")
    head = _git(repo, "rev-parse", "feature")
    r = _run(repo, f"{merge}^1", merge, body="Ordinary change.", head_sha=head)
    assert r.returncode == code, r.stdout + r.stderr
    assert ("deleted test file: tests/test_new.py" in r.stdout) == drop_newer_test


@pytest.mark.parametrize("bad_side", ["base", "head"])
def test_bad_revision_is_an_error_not_a_pass(tmp_path, bad_side):
    """Mutation killed: a diff failure swallowed into exit 0."""
    repo, b, h = _repo(tmp_path, BASE, {})
    missing = "0" * 40
    base, head = (missing, h) if bad_side == "base" else (b, missing)
    r = _run(repo, base, head, body="", head_sha=h)
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


@pytest.mark.parametrize(
    ("src", "expected"),
    [
        ("def test_a():\n    pass\n\ndef test_b(:\n", 2),  # does not parse
        ("def test_a():\n    pass\n\ndef helper():\n    pass\n", 1),  # parses
    ],
)
def test_test_functions_are_counted_parsed_or_not(src, expected):
    """Mutation killed: a syntax error hides every test in the file."""
    assert sum(guard.count_test_functions(src).values()) == expected
