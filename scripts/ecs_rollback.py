#!/usr/bin/env python3
"""Pick the task definition that is actually serving, and keep a failed deploy failed.

ECS keeps more than one deployment on a service. The service-level
taskDefinition is the desired one. During a rollout that desired value can
be a new revision that is not serving yet, while an older deployment still
holds the running tasks. Subtracting one from the current revision is not
an identity: a skipped number, a failed rollout, or a pin to a non-adjacent
revision all make N-1 the wrong task definition.

capture reads a describe-services payload and prints the taskDefinition of
the deployment that is holding tasks, preferring a COMPLETED rollout.
verify reports whether the service is back on that same value with a
completed rollout that has tasks. The workflow exits 1 after either result.
Recovery does not turn the deploy green.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _running(deployment: dict) -> int:
    try:
        return int(deployment.get("runningCount") or 0)
    except (TypeError, ValueError):
        return 0


def service_from_payload(payload: Any) -> dict:
    """A describe-services envelope or one service object."""
    if isinstance(payload, dict) and isinstance(payload.get("services"), list):
        services = [item for item in payload["services"] if isinstance(item, dict)]
        if len(services) != 1:
            raise ValueError("describe-services payload must contain one service")
        return services[0]
    if not isinstance(payload, dict):
        raise TypeError("service payload must be an object")
    return payload


def serving_task_definition(service: dict) -> str:
    """Task definition ARN of the deployment that is serving.

    Deployments with running tasks outrank empty ones. Among those, a
    COMPLETED rollout outranks an in-progress one. The service-level
    taskDefinition is used only when no deployment names a task definition.
    """
    body = service_from_payload(service)
    deployments = body.get("deployments") or []
    if not isinstance(deployments, list):
        raise TypeError("deployments must be a list")
    scored: list[tuple[tuple[int, int, int], str]] = []
    for deployment in deployments:
        if not isinstance(deployment, dict):
            continue
        arn = str(deployment.get("taskDefinition") or "").strip()
        if not arn:
            continue
        running = _running(deployment)
        completed = 1 if deployment.get("rolloutState") == "COMPLETED" else 0
        scored.append(((1 if running > 0 else 0, completed, running), arn))
    if scored:
        return max(scored)[1]
    fallback = str(body.get("taskDefinition") or "").strip()
    if fallback:
        return fallback
    raise ValueError("no serving task definition")


def recovery_matches(service: dict, prior: str) -> bool:
    """True when the service desired task definition is prior and it is up.

    prior must be the captured serving ARN, not a computed neighbour.
    """
    wanted = (prior or "").strip()
    if not wanted:
        return False
    body = service_from_payload(service)
    if str(body.get("taskDefinition") or "").strip() != wanted:
        return False
    for deployment in body.get("deployments") or []:
        if not isinstance(deployment, dict):
            continue
        if str(deployment.get("taskDefinition") or "").strip() != wanted:
            continue
        if deployment.get("rolloutState") == "COMPLETED" and _running(deployment) > 0:
            return True
    return False


def deploy_exit_code(*, recovered: bool) -> int:
    """The deploy stays failed whether or not the previous task definition is back."""
    del recovered
    return 1


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("payload must be a JSON object")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ECS web-service rollback decisions")
    sub = parser.add_subparsers(dest="cmd", required=True)

    capture = sub.add_parser("capture")
    capture.add_argument("payload", type=Path)

    verify = sub.add_parser("verify")
    verify.add_argument("payload", type=Path)
    verify.add_argument("--prior", required=True)

    args = parser.parse_args(argv)
    try:
        payload = _load(args.payload)
        if args.cmd == "capture":
            print(serving_task_definition(payload))
            return 0
        matched = recovery_matches(payload, args.prior)
        print("recovered" if matched else "not-recovered")
        return 0 if matched else 1
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
