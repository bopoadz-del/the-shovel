"""Register a new ``the-fork-ingest`` task-definition revision that moves
``WORKER_COMMAND`` from the ``secrets`` block to the plain ``environment``
block, with the SAME value it has today.

Why: ECS lets a Secrets Manager *secret* win over a same-name containerOverrides
``environment`` override, so ``ingest-run-once.yml`` could not reach the worker
command with argv flags. A plain ``environment`` value is overridable, so the
run-once workflow can pass the full invocation as ``WORKER_COMMAND``.

Safety contract (the "alias repoint" method):
- The value is read in-process and copied across. It is never printed, logged,
  or written anywhere except into the new task-definition ``environment``.
- Every other field is byte-identical. A SHA-256 of the container definition
  with the ``WORKER_COMMAND`` entry removed is computed before and after the
  move; the two digests must match, proving ONLY that one entry moved.
- The other secrets stay secrets (the remaining ``secrets`` list is untouched).

Prints only: the new revision number, the before/after digest match, and the
count of environment / secrets entries -- never any value.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from typing import Any, Dict, List

import boto3

from scripts.load_live_task_env import split_secret_ref

REGION = "us-west-2"
FAMILY = "the-fork-ingest"
WORKER_COMMAND = "WORKER_COMMAND"


def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha(obj: Any) -> str:
    return hashlib.sha256(_canonical(obj)).hexdigest()


def _without_entry(items: List[Dict[str, str]], name: str) -> List[Dict[str, str]]:
    return [i for i in items if i.get("name") != name]


def _container_without_worker(container: Dict[str, Any]) -> Dict[str, Any]:
    clone = dict(container)
    clone["environment"] = _without_entry(clone.get("environment") or [], WORKER_COMMAND)
    clone["secrets"] = _without_entry(clone.get("secrets") or [], WORKER_COMMAND)
    return clone


def _resolve_secret_value(client, value_from: str) -> str:
    """Resolve the secret JSON key named by a ``valueFrom`` ARN. Never printed."""
    secret_id, key = split_secret_ref(value_from)
    resp = client.get_secret_value(SecretId=secret_id)
    raw = resp.get("SecretString") or ""
    if key:
        return str(json.loads(raw).get(key, ""))
    return raw


def register(family: str = FAMILY, region: str = REGION) -> Dict[str, Any]:
    ecs = boto3.client("ecs", region_name=region)
    secrets = boto3.client("secretsmanager", region_name=region)

    td = ecs.describe_task_definition(taskDefinition=family)["taskDefinition"]
    container = td["containerDefinitions"][0]

    secrets_list = container.get("secrets") or []
    worker_secrets = [s for s in secrets_list if s.get("name") == WORKER_COMMAND]
    if not worker_secrets:
        # Already a plain environment value (or absent): nothing to move.
        env_has = any(e.get("name") == WORKER_COMMAND for e in (container.get("environment") or []))
        return {
            "already": True,
            "revision": td.get("revision"),
            "worker_in_environment": env_has,
            "digest_match": True,
            "env_count": len(container.get("environment") or []),
            "secrets_count": len(secrets_list),
        }

    before_digest = _sha(_container_without_worker(container))

    value = _resolve_secret_value(secrets, worker_secrets[0]["valueFrom"])
    if not value:
        raise RuntimeError("worker_command_empty")

    new_container = dict(container)
    new_container["secrets"] = _without_entry(secrets_list, WORKER_COMMAND)
    new_container["environment"] = (container.get("environment") or []) + [
        {"name": WORKER_COMMAND, "value": value},
    ]

    after_digest = _sha(_container_without_worker(new_container))
    digest_match = before_digest == after_digest
    if not digest_match:
        raise RuntimeError(
            f"integrity_failed before={before_digest} after={after_digest}"
        )

    # register-task-definition accepts the read fields minus read-only outputs.
    read_only = {
        "taskDefinitionArn", "revision", "status", "requiresAttributes",
        "compatibilities", "registeredAt", "registeredBy", "deregisteredAt",
    }
    payload = {k: v for k, v in td.items() if k not in read_only}
    payload["containerDefinitions"] = [new_container]

    registered = ecs.register_task_definition(**payload)["taskDefinition"]
    return {
        "already": False,
        "revision": registered.get("revision"),
        "worker_in_environment": True,
        "digest_match": digest_match,
        "env_count": len(new_container.get("environment") or []),
        "secrets_count": len(new_container.get("secrets") or []),
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", default=FAMILY)
    parser.add_argument("--region", default=REGION)
    args = parser.parse_args(argv)
    result = register(family=args.family, region=args.region)
    # Verification line: no value is ever included.
    if result["already"]:
        print(
            f"VERIFY family={args.family} revision={result['revision']} "
            f"already_plain_env={result['worker_in_environment']} "
            f"env={result['env_count']} secrets={result['secrets_count']}"
        )
    else:
        print(
            f"VERIFY family={args.family} new_revision={result['revision']} "
            f"digest_match={result['digest_match']} "
            f"env={result['env_count']} secrets={result['secrets_count']}"
        )
    return 0 if result["digest_match"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
