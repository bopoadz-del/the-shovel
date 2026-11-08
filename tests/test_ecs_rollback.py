"""The web rollback restores the deployment that is serving, and stays failed.

Two service shapes per decision. No AWS calls. Revision N-1 is not the rule:
a gap between the desired revision and the one holding tasks must select the
one holding tasks.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts.ecs_rollback import (
    deploy_exit_code,
    main,
    recovery_matches,
    serving_task_definition,
)

_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOW = _ROOT / ".github" / "workflows" / "deploy-aws.yml"
_RUNBOOK = _ROOT / "deploy" / "ECS_ROLLBACK.md"

_REV4 = "arn:aws:ecs:us-west-2:111111111111:task-definition/the-fork:4"
_REV7 = "arn:aws:ecs:us-west-2:111111111111:task-definition/the-fork:7"
_REV8 = "arn:aws:ecs:us-west-2:111111111111:task-definition/the-fork:8"
_REV9 = "arn:aws:ecs:us-west-2:111111111111:task-definition/the-fork:9"


def _deployment(arn: str, *, state: str, running: int, status: str) -> dict:
    return {
        "status": status,
        "taskDefinition": arn,
        "rolloutState": state,
        "runningCount": running,
    }


def _service(*deployments: dict, desired: str) -> dict:
    return {"taskDefinition": desired, "deployments": list(deployments)}


@pytest.mark.parametrize(
    "service,expected,rejected",
    [
        (
            _service(
                _deployment(_REV9, state="IN_PROGRESS", running=0, status="PRIMARY"),
                _deployment(_REV7, state="COMPLETED", running=3, status="ACTIVE"),
                desired=_REV9,
            ),
            _REV7,
            (_REV8, _REV9),
        ),
        (
            _service(
                _deployment(_REV4, state="COMPLETED", running=1, status="PRIMARY"),
                desired=_REV9,
            ),
            _REV4,
            (_REV8, _REV9),
        ),
    ],
)
def test_serving_task_definition_is_not_the_neighbour_revision(service, expected, rejected):
    chosen = serving_task_definition(service)
    assert chosen == expected
    assert chosen not in rejected


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({"services": [_service(
            _deployment(_REV4, state="COMPLETED", running=2, status="PRIMARY"),
            desired=_REV9,
        )]}, _REV4),
        (_service(desired=_REV7), _REV7),
    ],
)
def test_capture_accepts_envelope_or_bare_service(payload, expected):
    assert serving_task_definition(payload) == expected


@pytest.mark.parametrize(
    "service,prior,matched",
    [
        (
            _service(
                _deployment(_REV7, state="COMPLETED", running=2, status="PRIMARY"),
                desired=_REV7,
            ),
            _REV7,
            True,
        ),
        (
            _service(
                _deployment(_REV7, state="COMPLETED", running=2, status="ACTIVE"),
                _deployment(_REV9, state="IN_PROGRESS", running=1, status="PRIMARY"),
                desired=_REV9,
            ),
            _REV7,
            False,
        ),
    ],
)
def test_recovery_requires_the_captured_definition_to_be_stable(service, prior, matched):
    assert recovery_matches(service, prior) is matched


@pytest.mark.parametrize("recovered", [True, False])
def test_rollback_keeps_the_deploy_failed(recovered):
    assert deploy_exit_code(recovered=recovered) == 1


@pytest.mark.parametrize(
    "argv,exit_code,out",
    [
        (["capture", "payload"], 0, _REV7 + "\n"),
        (["verify", "payload", "--prior", _REV9], 1, "not-recovered\n"),
    ],
)
def test_cli_capture_and_failed_verify(tmp_path, capsys, argv, exit_code, out):
    payload = {
        "services": [
            _service(
                _deployment(_REV9, state="IN_PROGRESS", running=0, status="PRIMARY"),
                _deployment(_REV7, state="COMPLETED", running=2, status="ACTIVE"),
                desired=_REV9,
            )
        ]
    }
    path = tmp_path / "payload"
    path.write_text(json.dumps(payload), encoding="utf-8")
    concrete = [str(path) if item == "payload" else item for item in argv]
    assert main(concrete) == exit_code
    assert capsys.readouterr().out == out


def _restore_step(text: str) -> dict:
    loaded = yaml.safe_load(text)
    steps = loaded["jobs"]["deploy"]["steps"]
    matches = [
        step
        for step in steps
        if step.get("name") == "Restore the previously serving web task definition"
    ]
    assert len(matches) == 1
    return matches[0]


def rollback_problems(text: str) -> list[str]:
    """Empty when a deploy workflow restores the captured web task definition and stays failed."""
    problems: list[str] = []
    if "scripts/ecs_rollback.py capture" not in text:
        problems.append("missing capture")
    if "scripts/ecs_rollback.py verify" not in text:
        problems.append("missing verify")
    if "web-roll-started" not in text:
        problems.append("missing roll marker")
    if "current-1" in text or "revision-1" in text:
        problems.append("uses neighbour revision")
    try:
        step = _restore_step(text)
    except (AssertionError, KeyError, yaml.YAMLError):
        problems.append("restore step missing")
        return problems
    if step.get("if") != "failure()":
        problems.append("restore does not run only after a failure")
    run = step.get("run") or ""
    verdict_kept = "deploy verdict FAILED" in run and run.rstrip().endswith("exit 1")
    if not verdict_kept:
        problems.append("restore can succeed")
    if "--service the-fork" not in run:
        problems.append("restore does not target the web service")
    if "the-fork-ingest" in run:
        problems.append("restore touches the ingest service")
    return problems


_BAD_WORKFLOW = """
jobs:
  deploy:
    steps:
      - name: Restore the previously serving web task definition
        if: success()
        run: |
          aws ecs update-service --service the-fork --task-definition current-1
          exit 0
"""


@pytest.mark.parametrize(
    "text,clean",
    [
        (_WORKFLOW.read_text(encoding="utf-8"), True),
        (_BAD_WORKFLOW, False),
    ],
)
def test_workflow_rollback_contract(text, clean):
    assert (rollback_problems(text) == []) is clean


@pytest.mark.parametrize(
    "path,needle",
    [
        (_RUNBOOK, "exit code 1"),
        (_RUNBOOK, "minus one"),
    ],
)
def test_runbook_states_failed_verdict_and_rejects_neighbour_revision(path, needle):
    text = path.read_text(encoding="utf-8")
    assert needle in text
    assert "N-1" in text
