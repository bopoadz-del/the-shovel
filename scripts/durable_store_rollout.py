#!/usr/bin/env python3
"""Render the next task definition revision for the durable document store rollout.

deploy-aws.yml reuses the task definition that is already on the service;
nothing in this repository writes container environment. This script is
the one place that does, for three settings only, in the order
deploy/aws/README.md gives:

  efs      DOCUMENT_BYTE_STORE=efs and DOCUMENT_STORE_ROOT=<the EFS mount>.
           REQUIRE_DURABLE_DOCUMENT_STORE must still be unset.
  enforce  REQUIRE_DURABLE_DOCUMENT_STORE=1. Refused unless the task already
           carries the efs stage.
  off      Removes all three, so the task is back to what it was before the
           rollout (the guard logs "not enforcing" and boots on local).

The decision is read from the task definition itself: the EFS volume, the
container path it is mounted at, and the environment and secret names. No
service, family or path is named here. A setting that is also defined as a
secret is refused, because the task would carry two values for one name.

render prints the register-task-definition payload; it never calls AWS.
Only setting names and these three non-secret values are printed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND = "DOCUMENT_BYTE_STORE"
ROOT = "DOCUMENT_STORE_ROOT"
GUARD = "REQUIRE_DURABLE_DOCUMENT_STORE"
MANAGED = (BACKEND, ROOT, GUARD)
STAGES = ("efs", "enforce", "off")

# Fields describe-task-definition returns that register-task-definition rejects.
_READ_ONLY = (
    "taskDefinitionArn", "revision", "status", "requiresAttributes",
    "compatibilities", "registeredAt", "registeredBy", "deregisteredAt",
)


class RolloutRefused(ValueError):
    """The task definition is not in a state this stage may change."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code


def _container(taskdef: dict) -> dict:
    containers = taskdef.get("containerDefinitions") or []
    if len(containers) != 1:
        raise RolloutRefused(
            "container_count", f"expected one container, found {len(containers)}"
        )
    return containers[0]


def efs_mount_path(taskdef: dict) -> str:
    """The container path of the one read-write EFS volume, or a refusal."""
    efs_volumes = {
        v.get("name") for v in taskdef.get("volumes") or []
        if v.get("efsVolumeConfiguration")
    }
    paths = [
        m.get("containerPath", "")
        for m in _container(taskdef).get("mountPoints") or []
        if m.get("sourceVolume") in efs_volumes and not m.get("readOnly")
    ]
    if not paths:
        raise RolloutRefused(
            "no_efs_mount", "the container has no read-write EFS mount"
        )
    if len(paths) > 1:
        raise RolloutRefused(
            "efs_mount_ambiguous", f"{len(paths)} read-write EFS mounts"
        )
    if not paths[0].startswith("/"):
        raise RolloutRefused("efs_mount_not_absolute", "mount path is not absolute")
    return paths[0]


def environment(taskdef: dict) -> dict[str, str]:
    return {
        e["name"]: e.get("value", "")
        for e in _container(taskdef).get("environment") or []
    }


def _refuse_secret_shadow(container: dict) -> None:
    shadowed = sorted(
        s["name"] for s in container.get("secrets") or [] if s.get("name") in MANAGED
    )
    if shadowed:
        raise RolloutRefused(
            "setting_is_a_secret", f"already defined as a secret: {', '.join(shadowed)}"
        )


def target_settings(taskdef: dict, stage: str) -> dict[str, str | None]:
    """What each managed setting must be after ``stage``. ``None`` means absent."""
    if stage not in STAGES:
        raise RolloutRefused("unknown_stage", f"stage must be one of {STAGES}")
    if stage == "off":
        return {BACKEND: None, ROOT: None, GUARD: None}
    mount = efs_mount_path(taskdef)
    env = environment(taskdef)
    if stage == "efs":
        if env.get(GUARD):
            raise RolloutRefused(
                "guard_already_set",
                f"{GUARD} is set; the efs stage boots with it unset first",
            )
        return {BACKEND: "efs", ROOT: mount, GUARD: None}
    if env.get(BACKEND) != "efs" or env.get(ROOT) != mount:
        raise RolloutRefused(
            "efs_stage_not_live",
            f"enforce needs {BACKEND}=efs and {ROOT} at the EFS mount first",
        )
    return {BACKEND: "efs", ROOT: mount, GUARD: "1"}


def render(described: dict, stage: str) -> tuple[dict, list[str]]:
    """A register-task-definition payload and the list of changed setting names."""
    taskdef = described.get("taskDefinition", described)
    container = _container(taskdef)
    _refuse_secret_shadow(container)
    targets = target_settings(taskdef, stage)
    current = environment(taskdef)

    kept = [
        e for e in container.get("environment") or [] if e.get("name") not in MANAGED
    ]
    added = [
        {"name": name, "value": value}
        for name, value in targets.items() if value is not None
    ]
    changed = sorted(n for n, v in targets.items() if current.get(n) != v)

    payload = {k: v for k, v in taskdef.items() if k not in _READ_ONLY}
    new_container = dict(container)
    new_container["environment"] = sorted(kept + added, key=lambda e: e["name"])
    payload["containerDefinitions"] = [new_container]
    if described.get("tags"):
        payload["tags"] = described["tags"]
    return payload, changed


def summary(described: dict, stage: str, changed: list[str]) -> str:
    taskdef = described.get("taskDefinition", described)
    after = target_settings(taskdef, stage)
    shown = ", ".join(f"{n}={after[n] if after[n] is not None else '<unset>'}" for n in MANAGED)
    family = taskdef.get("family", "?")
    rev = taskdef.get("revision", "?")
    return (
        f"{family}:{rev} stage={stage} changed={','.join(changed) or 'none'} "
        f"after: {shown}"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("described", type=Path, help="describe-task-definition JSON")
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--out", type=Path, required=True, help="register payload JSON")
    args = ap.parse_args(argv)
    described = json.loads(args.described.read_text(encoding="utf-8"))
    try:
        payload, changed = render(described, args.stage)
    except RolloutRefused as exc:
        print(f"REFUSED {exc}", file=sys.stderr)
        return 2
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(summary(described, args.stage, changed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
