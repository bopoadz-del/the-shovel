"""Auto-deploy runs only for a push to this repository's main.

A fork pull request whose branch is named main must not satisfy the
workflow_run path. Manual dispatch stays. The post-deploy check waits for
ECS stability and for /livez to report the shipped commit, using a
repository variable for the origin. The workflow file parses without that
variable; an unset or empty value fails the confirm step and does not skip it.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

_WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "deploy-aws.yml"


def test_workflow_run_deploys_only_a_same_repo_push_to_main():
    text = _WORKFLOW.read_text(encoding="utf-8")
    assert "workflow_dispatch:" in text
    assert "github.event_name == 'workflow_dispatch'" in text
    assert "github.event.workflow_run.conclusion == 'success'" in text
    assert "github.event.workflow_run.event == 'push'" in text
    assert "github.event.workflow_run.head_repository.full_name == github.repository" in text
    assert "github.event.workflow_run.head_branch == 'main'" in text
    assert "aws ecs wait services-stable" in text
    assert "timeout 600 aws ecs wait services-stable" in text
    assert "vars.PUBLIC_BASE_URL" in text
    assert "/livez" in text
    assert "build_sha" in text
    assert "FACTORY_DEPLOY_KEY" not in text
    assert "https://theshovel.ai" not in text


def _confirm_step() -> dict:
    loaded = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    steps = loaded["jobs"]["deploy"]["steps"]
    matches = [
        step
        for step in steps
        if step.get("name") == "Confirm live build_sha matches the deployed commit"
    ]
    assert len(matches) == 1
    return matches[0]


def test_livez_confirm_parses_without_public_base_url_and_fails_when_missing():
    """The file parses with the variable unset. The step then fails closed."""
    step = _confirm_step()
    assert "if" not in step
    assert not step.get("continue-on-error")
    assert step["env"]["DEPLOY_BASE_URL"] == "${{ vars.PUBLIC_BASE_URL }}"
    script = step["run"]
    for env in ({}, {"DEPLOY_BASE_URL": ""}):
        proc = subprocess.run(
            ["bash", "-c", script],
            env=env,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        combined = proc.stdout + proc.stderr
        assert proc.returncode != 0
        assert combined.count("PUBLIC_BASE_URL") == 1
        assert "unset or empty" in combined
