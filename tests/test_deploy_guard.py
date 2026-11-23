"""deploy-aws ships only commits on main, never a commit older than live.

The failure this guards: a late ``--failed`` rerun of an older main commit's
tests finishes green after a newer commit has already shipped. deploy-aws
fires on that run and ships the older commit, so live goes backwards. Every
repository here is built by the test itself.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

import deploy_guard as dg  # noqa: E402

_WORKFLOW = _ROOT / ".github" / "workflows" / "deploy-aws.yml"
_GUARD_STEP = "Refuse a commit that is not on main or is older than live"

# ------------------------------------------------------------- the decision

def _graph(parents: dict[str, str]):
    """is_ancestor over a history given as child -> parent ("" for a root)."""

    def is_ancestor(a: str, b: str):
        if a not in parents or b not in parents:
            return None
        while b:
            if a == b:
                return True
            b = parents[b]
        return False

    return is_ancestor


# main is aaa -> bbb -> ccc; xxx is a branch commit off bbb, never merged.
_HISTORY = _graph({"aaa": "", "bbb": "aaa", "ccc": "bbb", "xxx": "bbb"})


@pytest.mark.parametrize("deploy,main_head,live,ok", [
    # Late rerun of an older commit while live is ahead: refused.
    ("aaa", "ccc", "bbb", False),
    ("bbb", "ccc", "ccc", False),
    # The head of main, ahead of live or equal to it: allowed.
    ("ccc", "ccc", "bbb", True),
    ("ccc", "ccc", "aaa", True),
    ("ccc", "ccc", "ccc", True),
    # On main but no longer the head (main moved while tests ran), and ahead
    # of live or equal to it: allowed.
    ("bbb", "ccc", "aaa", True),
    ("bbb", "ccc", "bbb", True),
    # A commit not on main: refused, whatever live is.
    ("xxx", "ccc", "aaa", False),
    ("xxx", "ccc", "bbb", False),
    # On main but older than live (live came from elsewhere): refused.
    ("bbb", "bbb", "ccc", False),
    ("aaa", "aaa", "ccc", False),
])
def test_only_a_commit_on_main_not_older_than_live_deploys(deploy, main_head, live, ok):
    verdict = dg.decide(deploy, main_head, live, _HISTORY)
    assert verdict.ok is ok


@pytest.mark.parametrize("deploy", ["ccc", "bbb"])
@pytest.mark.parametrize("live", ["", "zzz"])
def test_an_unknown_live_commit_does_not_block_a_commit_on_main(deploy, live):
    verdict = dg.decide(deploy, "ccc", live, _HISTORY)
    assert verdict.ok is True
    assert "not made" in verdict.warning


@pytest.mark.parametrize("live", ["", "zzz"])
def test_an_unknown_live_commit_does_not_let_a_commit_off_main_through(live):
    verdict = dg.decide("xxx", "ccc", live, _HISTORY)
    assert verdict.ok is False
    assert "not on main" in verdict.reason


@pytest.mark.parametrize("deploy", ["qqq", "rrr"])
def test_a_commit_git_cannot_place_against_main_is_refused(deploy):
    verdict = dg.decide(deploy, "ccc", "", _HISTORY)
    assert verdict.ok is False
    assert "cannot tell" in verdict.reason


@pytest.mark.parametrize("deploy,main_head", [("", "ccc"), ("ccc", "")])
def test_a_missing_sha_is_refused(deploy, main_head):
    assert dg.decide(deploy, main_head, "ccc", _HISTORY).ok is False


# ------------------------------------------------------------- real git

def _git(cwd: Path, *args: str) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid",
        "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1",
    }
    proc = subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, text=True, check=True)
    return proc.stdout.strip()


@pytest.fixture()
def origin(tmp_path: Path):
    """main is c1 -> c2 -> c3 (shas[0..2]); shas[3] is a branch commit off c2."""
    work = tmp_path / "work"
    work.mkdir()
    _git(work, "init", "-q", "-b", "main")
    (work / "scripts").mkdir()
    for name in ("deploy_guard.py", "read_livez_build_sha.py"):
        shutil.copy(_ROOT / "scripts" / name, work / "scripts" / name)
    shas = []
    for n in range(1, 4):
        (work / "n.txt").write_text(str(n), encoding="utf-8")
        _git(work, "add", "-A")
        _git(work, "commit", "-q", "-m", f"c{n}")
        shas.append(_git(work, "rev-parse", "HEAD"))
    # A commit that never reached main: a branch off c2.
    _git(work, "checkout", "-q", "-b", "side", shas[1])
    (work / "side.txt").write_text("side", encoding="utf-8")
    _git(work, "add", "-A")
    _git(work, "commit", "-q", "-m", "side")
    shas.append(_git(work, "rev-parse", "HEAD"))
    _git(work, "checkout", "-q", "main")
    return work, shas


def _cli(repo: Path, deploy: str, live: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(repo / "scripts" / "deploy_guard.py"),
         "--deploy", deploy, "--main-ref", "main", "--live", live],
        cwd=repo, capture_output=True, text=True, check=False,
    )


@pytest.mark.parametrize("deploy_index,live_index", [(0, 1), (0, 2), (1, 2)])
def test_the_late_rerun_of_an_older_commit_is_refused(origin, deploy_index, live_index):
    """22 Nov shape: an older commit's tests go green late; newer is live; main is at c3."""
    repo, shas = origin
    proc = _cli(repo, shas[deploy_index], shas[live_index])
    assert proc.returncode == 1
    assert "Deploy refused" in proc.stdout


@pytest.mark.parametrize("live_index", [0, 1, 2])
def test_the_head_of_main_deploys_over_any_older_or_equal_live(origin, live_index):
    repo, shas = origin
    proc = _cli(repo, shas[2], shas[live_index])
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.mark.parametrize("live_index", [0, 1])
def test_a_commit_main_has_moved_past_still_deploys_over_older_or_equal_live(origin, live_index):
    """Main moved on while c2's tests ran; c2 is not older than live, so it ships."""
    repo, shas = origin
    proc = _cli(repo, shas[1], shas[live_index])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "may deploy" in proc.stdout


@pytest.mark.parametrize("live_index", [0, 1])
def test_a_branch_commit_is_refused(origin, live_index):
    repo, shas = origin
    proc = _cli(repo, shas[3], shas[live_index])
    assert proc.returncode == 1
    assert "not on main" in proc.stdout


@pytest.mark.parametrize("main_index,live_index", [(1, 2), (0, 2), (0, 1)])
def test_main_head_older_than_live_is_refused(origin, main_index, live_index):
    """Live is ahead of main (shipped from a branch): main's head is older."""
    repo, shas = origin
    _git(repo, "branch", "-f", "main-old", shas[main_index])
    proc = subprocess.run(
        [sys.executable, str(repo / "scripts" / "deploy_guard.py"),
         "--deploy", shas[main_index], "--main-ref", "main-old", "--live", shas[live_index]],
        cwd=repo, capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 1
    assert "older than the live commit" in proc.stdout


@pytest.mark.parametrize("live", ["f" * 40, "0123456789abcdef" * 2 + "01234567"])
def test_a_live_commit_outside_the_history_warns_and_allows(origin, live):
    repo, shas = origin
    proc = _cli(repo, shas[2], live)
    assert proc.returncode == 0
    assert "::warning" in proc.stdout


# ------------------------------------------------------------- the workflow

def _steps() -> list[dict]:
    return yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))["jobs"]["deploy"]["steps"]


def _guard() -> dict:
    matches = [s for s in _steps() if s.get("name") == _GUARD_STEP]
    assert len(matches) == 1
    return matches[0]


@pytest.mark.parametrize("later", ["Build and push", "Roll the web service"])
def test_the_guard_runs_before_anything_is_built_or_rolled(later):
    names = [s.get("name") for s in _steps()]
    assert names.index(_GUARD_STEP) < names.index(later)


def test_the_guard_step_cannot_be_skipped():
    step = _guard()
    assert "if" not in step
    assert not step.get("continue-on-error")
    assert "scripts/deploy_guard.py" in step["run"]
    assert "--main-ref origin/main" in step["run"]


def _run_step(tmp_path: Path, origin_dir: Path, checkout_sha: str) -> subprocess.CompletedProcess:
    """Run the workflow's guard script in a depth-1 clone, as actions/checkout leaves it."""
    clone = tmp_path / f"clone-{checkout_sha[:7]}"
    _git(tmp_path, "clone", "-q", "--no-local", "--depth", "1", "--branch", "main",
         origin_dir.as_uri(), str(clone))
    _git(clone, "fetch", "-q", "--depth", "1", "origin", checkout_sha)
    _git(clone, "checkout", "-q", checkout_sha)
    script = _guard()["run"].replace("python3 ", f'"{Path(sys.executable).as_posix()}" ')
    env = {**os.environ, "DEPLOY_SHA": checkout_sha, "RUNNER_TEMP": str(tmp_path),
           "DEPLOY_BASE_URL": "", "GIT_CONFIG_GLOBAL": os.devnull}
    return subprocess.run([shutil.which("bash") or "bash", "-c", script], cwd=clone, env=env,
                          capture_output=True, text=True, check=False)


@pytest.mark.parametrize("index,ok", [(0, True), (1, True), (2, True), (3, False)])
def test_the_step_script_ships_only_commits_on_main(tmp_path, origin, index, ok):
    origin_dir, shas = origin
    proc = _run_step(tmp_path, origin_dir, shas[index])
    assert (proc.returncode == 0) is ok, proc.stdout + proc.stderr
    assert ("Deploy refused" in proc.stdout) is not ok
