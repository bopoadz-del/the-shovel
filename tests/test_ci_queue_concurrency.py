"""CI queue policy: superseded PR runs cancel, main runs never do, drafts run light.

Each workflow file is read as YAML and checked against the policy, and the
same checker runs on small synthetic workflows: one that keeps the policy and
ones that break it. The Docker skip classifier (scripts/ci_image_inputs.py)
is checked on file lists built here.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / ".github" / "workflows"
sys.path.insert(0, str(ROOT / "scripts"))

from ci_image_inputs import image_needed, is_non_image_file

PR_WORKFLOWS = [
    "test.yml",
    "lint.yml",
    "pr-quality.yml",
    "fast.yml",
    "mutation-gate.yml",
    "dependency-audit.yml",
    "docker-publish.yml",
]
PROBES = ["health-watch.yml", "slo-probe.yml"]
_EXPR = re.compile(r"^\$\{\{\s*github\.event_name\s*(==|!=)\s*'([a-z_]+)'\s*\}\}$")


def _load(text: str) -> dict:
    doc = yaml.safe_load(text)
    # YAML 1.1 reads the bare key `on` as True.
    if True in doc:
        doc["on"] = doc.pop(True)
    return doc


def _triggers(doc: dict) -> list[str]:
    on = doc["on"]
    if isinstance(on, str):
        return [on]
    return list(on)


def _cancels(value, event: str) -> bool:
    if isinstance(value, bool):
        return value
    m = _EXPR.match(str(value).strip())
    if not m:
        raise ValueError(f"unreadable cancel-in-progress: {value!r}")
    op, name = m.groups()
    return (event == name) if op == "==" else (event != name)


def concurrency_violations(text: str) -> list[str]:
    """Policy for a PR-triggered workflow; [] when it is kept."""
    doc = _load(text)
    conc = doc.get("concurrency")
    if not isinstance(conc, dict):
        return ["no concurrency block"]
    out = []
    if "github.ref" not in str(conc.get("group", "")):
        out.append("group is not per ref")
    cancel = conc.get("cancel-in-progress", False)
    triggers = _triggers(doc)
    if "pull_request" in triggers and not _cancels(cancel, "pull_request"):
        out.append("a superseded PR run is not cancelled")
    for ev in ("push", "schedule", "workflow_dispatch"):
        if ev in triggers and _cancels(cancel, ev):
            out.append(f"a {ev} run can be cancelled")
    return out


def _wf(name: str) -> str:
    return (WF / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("name", PR_WORKFLOWS)
def test_every_pr_workflow_keeps_the_concurrency_policy(name):
    assert concurrency_violations(_wf(name)) == []


def _group(text: str) -> str:
    return str(_load(text)["concurrency"]["group"])


def shared_group_prefixes(texts: list[str]) -> list[str]:
    """Group prefixes (the part before the first expression) used by more
    than one workflow: two workflows sharing one would cancel each other."""
    prefixes = [_group(t).split("$")[0] for t in texts]
    return sorted({p for p in prefixes if prefixes.count(p) > 1})


def _wf_text(prefix: str) -> str:
    return (
        "on:\n  pull_request:\nconcurrency:\n"
        f"  group: {prefix}${{{{ github.ref }}}}\n  cancel-in-progress: true\njobs: {{}}\n"
    )


@pytest.mark.parametrize(
    "texts,expected",
    [
        ([_wf(n) for n in PR_WORKFLOWS], []),
        ([_wf_text("lint-"), _wf_text("fast-"), _wf_text("tests-")], []),
        ([_wf_text("lint-"), _wf_text("fast-"), _wf_text("lint-")], ["lint-"]),
    ],
)
def test_groups_are_one_per_workflow(texts, expected):
    assert shared_group_prefixes(texts) == expected


_GOOD = (
    "on:\n  pull_request:\n  push:\n    branches: [main]\n"
    "concurrency:\n  group: x-${{ github.ref }}\n"
    "  cancel-in-progress: ${{ github.event_name == 'pull_request' }}\njobs: {}\n"
)


@pytest.mark.parametrize(
    "text,expected",
    [
        (_GOOD, []),
        (
            (
                "on:\n  pull_request:\n  push:\n    branches: [main]\n"
                "concurrency:\n  group: x-${{ github.ref }}\n  cancel-in-progress: true\n"
                "jobs: {}\n"
            ),
            ["a push run can be cancelled"],
        ),
        (
            (
                "on:\n  pull_request:\n  schedule:\n    - cron: '0 6 * * 1'\n"
                "concurrency:\n  group: x-${{ github.ref }}\n"
                "  cancel-in-progress: ${{ github.event_name != 'push' }}\njobs: {}\n"
            ),
            ["a schedule run can be cancelled"],
        ),
        (
            (
                "on:\n  pull_request:\n"
                "concurrency:\n  group: x-${{ github.ref }}\n  cancel-in-progress: false\n"
                "jobs: {}\n"
            ),
            ["a superseded PR run is not cancelled"],
        ),
        (
            (
                "on:\n  pull_request:\n"
                "concurrency:\n  group: x\n  cancel-in-progress: true\njobs: {}\n"
            ),
            ["group is not per ref"],
        ),
        ("on:\n  pull_request:\njobs: {}\n", ["no concurrency block"]),
    ],
)
def test_checker_on_kept_and_broken_policies(text, expected):
    assert concurrency_violations(text) == expected


def test_checker_refuses_an_expression_it_cannot_read():
    text = (
        "on:\n  pull_request:\nconcurrency:\n  group: x-${{ github.ref }}\n"
        "  cancel-in-progress: ${{ github.head_ref != '' }}\njobs: {}\n"
    )
    with pytest.raises(ValueError):
        concurrency_violations(text)


@pytest.mark.parametrize(
    "value,event,expected",
    [
        ("${{ github.event_name == 'pull_request' }}", "pull_request", True),
        ("${{ github.event_name == 'pull_request' }}", "push", False),
        ("${{ github.event_name != 'push' }}", "pull_request", True),
        ("${{ github.event_name != 'push' }}", "push", False),
        (True, "push", True),
        (False, "pull_request", False),
    ],
)
def test_cancel_expression_reading(value, event, expected):
    assert _cancels(value, event) is expected


# Draft PRs run only lint and fast; the heavy and gate jobs skip on draft and
# their workflow runs again on ready_for_review.
DRAFT_GATED = [
    ("mutation-gate.yml", "mutation-gate"),
    ("dependency-audit.yml", "pip-audit"),
    ("pr-quality.yml", "check-body-and-size"),
    ("pr-quality.yml", "test-removal-guard"),
    ("docker-publish.yml", "build"),
    ("docker-publish.yml", "changes"),
]


@pytest.mark.parametrize("name,job", DRAFT_GATED)
def test_heavy_job_skips_on_draft_and_reruns_on_ready(name, job):
    doc = _load(_wf(name))
    cond = str(doc["jobs"][job].get("if", ""))
    assert "! github.event.pull_request.draft" in cond
    assert "ready_for_review" in doc["on"]["pull_request"]["types"]


@pytest.mark.parametrize(
    "name,job", [("lint.yml", "ruff"), ("fast.yml", "two-input-lint")]
)
def test_draft_checks_still_run_on_draft(name, job):
    cond = str(_load(_wf(name))["jobs"][job].get("if", ""))
    assert "draft" not in cond


def test_docker_build_is_fail_closed_on_the_classifier():
    jobs = _load(_wf("docker-publish.yml"))["jobs"]
    cond = str(jobs["build"]["if"])
    assert "!cancelled()" in cond
    assert "github.event_name != 'pull_request'" in cond
    assert "needs.changes.result != 'success'" in cond
    assert "needs.changes.outputs.image != 'false'" in cond
    assert "needs.build.result == 'success'" in str(jobs["compose-health"]["if"])


@pytest.mark.parametrize("name", PROBES)
def test_probes_have_their_own_group_and_are_never_cancelled(name):
    doc = _load(_wf(name))
    conc = doc["concurrency"]
    assert "github.ref" not in str(conc["group"])
    assert name.split(".")[0] in str(conc["group"])
    assert conc["cancel-in-progress"] is False
    for job in doc["jobs"].values():
        for step in job["steps"]:
            if str(step.get("uses", "")).startswith("actions/checkout"):
                assert step.get("with", {}).get("sparse-checkout"), name


def probe_group_clashes(pr_texts: list[str], probe_texts: list[str]) -> list[str]:
    """Groups a probe shares with a PR workflow or with another probe; a
    shared group would queue the probe behind that run."""
    pr = {_group(t) for t in pr_texts}
    probes = [_group(t) for t in probe_texts]
    return sorted({g for g in probes if g in pr or probes.count(g) > 1})


def _probe_text(group: str) -> str:
    return (
        "on:\n  schedule:\n    - cron: '*/15 * * * *'\nconcurrency:\n"
        f"  group: {group}\n  cancel-in-progress: false\njobs: {{}}\n"
    )


@pytest.mark.parametrize(
    "pr_texts,probe_texts,expected",
    [
        ([_wf(n) for n in PR_WORKFLOWS], [_wf(n) for n in PROBES], []),
        ([_wf_text("lint-")], [_probe_text("a-probe"), _probe_text("b-probe")], []),
        (
            [_wf_text("lint-")],
            [_probe_text("a-probe"), _probe_text("a-probe")],
            ["a-probe"],
        ),
        (
            [_wf_text("lint-")],
            [_probe_text("lint-${{ github.ref }}"), _probe_text("b-probe")],
            ["lint-${{ github.ref }}"],
        ),
    ],
)
def test_probe_groups_differ_from_every_pr_group(pr_texts, probe_texts, expected):
    assert probe_group_clashes(pr_texts, probe_texts) == expected


@pytest.mark.parametrize(
    "paths,expected",
    [
        (["tests/test_a.py"], False),
        (["tests/test_a.py", "tests/fixtures/b.json"], False),
        (["docs/RUNBOOK.md", "README.md"], False),
        (["docs/diagram.png", "tests/test_b.py"], False),
        (["tests/test_a.py", "app/main.py"], True),
        (["docs/knowledge/concrete.md"], True),
        (["README.md", "requirements.txt"], True),
        (["Dockerfile"], True),
        ([".github/workflows/docker-publish.yml"], True),
        ([], True),
        (["", "  "], True),
        (["tests\\test_win.py"], False),
        (["frontend/src/App.tsx"], True),
    ],
)
def test_image_needed(paths, expected):
    assert image_needed(paths) is expected


@pytest.mark.parametrize(
    "path,expected",
    [
        ("tests/x.py", True),
        ("docs/x.txt", True),
        ("CHANGELOG.MD", True),
        ("docs/knowledge/x.md", False),
        ("app/tests_helper.py", False),
        ("scripts/tests/x.py", False),
        ("", False),
    ],
)
def test_is_non_image_file(path, expected):
    assert is_non_image_file(path) is expected


@pytest.mark.parametrize(
    "stdin,expected",
    [
        ("tests/test_a.py\ndocs/x.md\n", "image=false"),
        ("tests/test_a.py\napp/main.py\n", "image=true"),
        ("", "image=true"),
    ],
)
def test_cli_prints_a_github_output_line(stdin, expected):
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "ci_image_inputs.py")],
        input=stdin,
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == expected
