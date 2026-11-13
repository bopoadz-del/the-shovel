"""Flip one non-credential flag in the live service env (P-H1).

Called by `.github/workflows/service-env-flag.yml`. The live ECS task
definition maps service env vars from the Secrets Manager JSON with
`valueFrom = <secret-arn>:<json-key>::`. A flag is changed by rewriting
that one JSON key, then forcing a new deployment so the new tasks read it.

Safety, in order:
  1. The key must be on ALLOWLIST (non-credential, non-URL flags only) and
     the value must fit that key's kind. Anything else is refused before any
     AWS read.
  2. Plan prints ONLY `key=<current value>` and how the key is mapped in the
     live task definition. No other field of the secret is printed.
  3. Execute refuses when the key is not mapped from a Secrets Manager JSON
     key in the live task definition (the write would be a silent no-op), or
     when the mapping pins a version (the new version would not be read).
  4. The write is `json.loads` -> set the one key -> `json.dumps(
     ensure_ascii=False)`, sent as a str (UTF-8 on the wire, never a BOM).
     It is read back and compared before the service is touched.
  5. `update-service --force-new-deployment`, wait for services-stable, then
     read the value back again and search the new tasks' CloudWatch stream
     for the startup line that proves the flag took, when the key has one.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

_BOM = "﻿"

# Key -> value kind. Only flags that are neither credentials nor URLs.
ALLOWLIST: dict[str, str] = {
    "HYDRATION_ENABLED": "bool",
    "HYDRATION_HOUR_UTC": "hour",
    "HYDRATION_LLM_SUMMARY": "bool",
    "PDF_OCR_BOQ_PAGE_CAP": "posint",
    "PDF_OCR_PAGE_CAP": "posint",
    "PDF_OCR_FORCE_PAGE_CAP": "posint",
}

_BOOL_VALUES = {"true", "false"}


class Refused(Exception):
    """The request is not allowed; nothing was read or written."""


def validate_request(key: str, value: str) -> str:
    """Return the normalised value, or raise Refused. No AWS access."""
    key = (key or "").strip()
    if key not in ALLOWLIST:
        raise Refused(f"key {key!r} is not on the allowlist {sorted(ALLOWLIST)}")
    raw = (value or "").strip()
    kind = ALLOWLIST[key]
    if kind == "bool":
        norm = raw.lower()
        if norm not in _BOOL_VALUES:
            raise Refused(f"{key} takes true or false, got {raw!r}")
        return norm
    if not re.fullmatch(r"[0-9]{1,6}", raw):
        raise Refused(f"{key} takes a non-negative integer, got {raw!r}")
    num = int(raw)
    if kind == "hour" and num > 23:
        raise Refused(f"{key} takes an hour 0-23, got {num}")
    if kind == "posint" and num < 1:
        raise Refused(f"{key} takes a positive integer, got {num}")
    return str(num)


def proof_pattern(key: str, value: str) -> str | None:
    """The app.core.hydration_scheduler startup line that proves the flag
    took on a new task, or None when the app logs no line for this key."""
    if key == "HYDRATION_ENABLED":
        if value == "true":
            return "hydration scheduler started, target ="
        return "hydration scheduler disabled via HYDRATION_ENABLED=false"
    if key == "HYDRATION_HOUR_UTC":
        return f"hydration scheduler started, target = {int(value):02d}:00"
    return None


@dataclass(frozen=True)
class Mapping:
    where: str  # "secret", "environment" or "absent"
    container: str = ""
    secret_id: str = ""
    json_key: str = ""
    pinned: bool = False


def parse_value_from(value_from: str) -> tuple[str, str, bool]:
    """Split an ECS secrets valueFrom into (secret_arn, json_key, pinned).

    Form: arn:aws:secretsmanager:region:acct:secret:name[:json-key[:stage[:id]]]
    """
    parts = (value_from or "").split(":")
    if len(parts) < 7 or parts[2] != "secretsmanager":
        raise Refused("the key is not mapped from Secrets Manager")
    secret_id = ":".join(parts[:7])
    json_key = parts[7] if len(parts) > 7 else ""
    pinned = any(p for p in parts[8:10])
    return secret_id, json_key, pinned


def find_mapping(task_definition: dict, key: str) -> Mapping:
    for container in task_definition.get("containerDefinitions") or []:
        name = container.get("name", "")
        for env in container.get("environment") or []:
            if env.get("name") == key:
                return Mapping("environment", container=name)
        for sec in container.get("secrets") or []:
            if sec.get("name") == key:
                secret_id, json_key, pinned = parse_value_from(sec.get("valueFrom", ""))
                return Mapping("secret", name, secret_id, json_key, pinned)
    return Mapping("absent")


def refuse_unless_writable(mapping: Mapping) -> None:
    if mapping.where == "absent":
        raise Refused("the key is not mapped in the live task definition; a write would be a silent no-op")
    if mapping.where == "environment":
        raise Refused("the key is a plain environment value in the task definition, not a secret JSON key")
    if not mapping.json_key:
        raise Refused("the key maps the whole secret, not one JSON key")
    if mapping.pinned:
        raise Refused("the mapping pins a secret version; a new version would not be read")


def _load(secret_string: str) -> dict:
    doc = json.loads((secret_string or "").lstrip(_BOM))
    if not isinstance(doc, dict):
        raise Refused("the secret is not a JSON object")
    return doc


def read_value(secret_string: str, json_key: str) -> str:
    doc = _load(secret_string)
    if json_key not in doc:
        return "<absent>"
    return str(doc[json_key])


def rewrite(secret_string: str, json_key: str, value: str) -> str:
    """Set one key and keep every other key and value unchanged."""
    doc = _load(secret_string)
    doc[json_key] = value
    # A dict always serialises starting with "{", so the result never carries
    # a BOM, and a BOM read in is dropped by _load. AwsCli writes it as
    # UTF-8. The 11 Nov break was a BOM-prefixed file written from PowerShell.
    return json.dumps(doc, ensure_ascii=False)


# --------------------------------------------------------------------------
# AWS flow. Clients are injected so the flow is testable without AWS.


@dataclass
class Clients:
    ecs: object
    secrets: object
    logs: object


def _live_task_definition(clients: Clients, cluster: str, service: str) -> dict:
    svc = clients.ecs.describe_services(cluster=cluster, services=[service])["services"][0]
    arn = svc["taskDefinition"]
    return clients.ecs.describe_task_definition(taskDefinition=arn)["taskDefinition"]


def _secret_string(clients: Clients, secret_id: str) -> str:
    return clients.secrets.get_secret_value(SecretId=secret_id)["SecretString"]


def _epoch(started) -> float:
    """ECS startedAt as epoch seconds: a datetime (SDK), an ISO string (CLI
    JSON) or a number. A task that has not started yet counts as 0."""
    if isinstance(started, str):
        return datetime.fromisoformat(started).timestamp()
    if hasattr(started, "timestamp"):
        return started.timestamp()
    return float(started or 0)


def _new_task_streams(clients: Clients, cluster: str, service: str, td: dict,
                      container: str, since: float) -> tuple[str, list[str]]:
    opts = {}
    for c in td.get("containerDefinitions") or []:
        if c.get("name") == container:
            opts = (c.get("logConfiguration") or {}).get("options") or {}
    group = opts.get("awslogs-group", "")
    prefix = opts.get("awslogs-stream-prefix", "")
    if not group or not prefix:
        raise Refused(f"container {container!r} has no awslogs group/stream-prefix")
    arns = clients.ecs.list_tasks(cluster=cluster, serviceName=service,
                                  desiredStatus="RUNNING").get("taskArns") or []
    tasks = clients.ecs.describe_tasks(cluster=cluster, tasks=arns).get("tasks") if arns else []
    streams = []
    for t in tasks or []:
        started = t.get("startedAt")
        ts = _epoch(started)
        if ts >= since:
            streams.append(f"{prefix}/{container}/{t['taskArn'].rsplit('/', 1)[-1]}")
    return group, streams


def _find_proof(clients: Clients, group: str, streams: list[str], pattern: str,
                since: float, attempts: int, sleep: Callable[[float], None]) -> str:
    for i in range(attempts):
        resp = clients.logs.filter_log_events(
            logGroupName=group, logStreamNames=streams,
            filterPattern=json.dumps(pattern), startTime=int(since * 1000))
        for ev in resp.get("events") or []:
            if pattern in ev.get("message", ""):
                return ev["message"].strip()
        if i + 1 < attempts:
            sleep(15)
    return ""


def run(key: str, value: str, execute: bool, clients_factory: Callable[[], Clients],
        cluster: str, service: str, out=print, sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], float] = time.time, log_attempts: int = 20) -> int:
    try:
        value = validate_request(key, value)
    except Refused as exc:
        out(f"REFUSED: {exc}")
        return 2
    clients = clients_factory()
    td = _live_task_definition(clients, cluster, service)
    mapping = find_mapping(td, key)
    out(f"mapped_in_task_definition={mapping.where}"
        + (f" container={mapping.container} json_key={mapping.json_key} pinned={mapping.pinned}"
           if mapping.where == "secret" else ""))
    try:
        refuse_unless_writable(mapping)
    except Refused as exc:
        out(f"key={key} <not read>")
        out(f"REFUSED: {exc}")
        return 2
    current = read_value(_secret_string(clients, mapping.secret_id), mapping.json_key)
    out(f"plan: {key}={current} (requested {value})")
    if not execute:
        out("execute=false: nothing written")
        return 0

    new_string = rewrite(_secret_string(clients, mapping.secret_id), mapping.json_key, value)
    clients.secrets.put_secret_value(SecretId=mapping.secret_id, SecretString=new_string)
    back = read_value(_secret_string(clients, mapping.secret_id), mapping.json_key)
    out(f"secret read-back: {key}={back}")
    if back != value:
        out(f"FAILED: read-back {back!r} != requested {value!r}; service not rolled")
        return 1

    since = now()
    clients.ecs.update_service(cluster=cluster, service=service, forceNewDeployment=True)
    clients.ecs.get_waiter("services_stable").wait(
        cluster=cluster, services=[service], WaiterConfig={"Delay": 15, "MaxAttempts": 40})
    out("rollout: services-stable")

    td_after = _live_task_definition(clients, cluster, service)
    after = read_value(_secret_string(clients, mapping.secret_id), mapping.json_key)
    out(f"post-rollout read-back: {key}={after}")
    if after != value:
        out(f"FAILED: post-rollout value {after!r} != requested {value!r}")
        return 1

    pattern = proof_pattern(key, value)
    if pattern is None:
        out(f"proof: the app logs no startup line for {key}; read-back only")
        return 0
    group, streams = _new_task_streams(clients, cluster, service, td_after,
                                       mapping.container, since)
    if not streams:
        out("FAILED: no task started after the forced deployment")
        return 1
    line = _find_proof(clients, group, streams, pattern, since, log_attempts, sleep)
    if not line:
        out(f"FAILED: no new task logged {pattern!r}")
        return 1
    out(f"proof ({len(streams)} new task(s)): {line}")
    return 0


class AwsCli:
    """The AWS CLI behind the client methods `run` uses. The repository keeps
    no AWS SDK import (tests/test_ingest_no_archive.py), so the workflow's
    `aws` CLI does the calls and this class only shapes the JSON. stdout is
    parsed, never printed. A failed call raises with the CLI's stderr, which
    carries the AWS error text, not the secret."""

    def __init__(self, region: str, runner: Callable = subprocess.run):
        self.region, self.runner = region, runner

    def _call(self, *args: str, parse: bool = True, env: dict | None = None):
        cmd = ["aws", *args, "--region", self.region, "--output", "json"]
        proc = self.runner(cmd, capture_output=True, text=True, encoding="utf-8",
                           env={**os.environ, **(env or {})})
        if proc.returncode != 0:
            raise RuntimeError(f"aws {args[0]} {args[1]} failed: {proc.stderr.strip()[:500]}")
        return json.loads(proc.stdout) if parse and proc.stdout.strip() else {}

    # ECS
    def describe_services(self, cluster, services):
        return self._call("ecs", "describe-services", "--cluster", cluster, "--services", *services)

    def describe_task_definition(self, taskDefinition):
        return self._call("ecs", "describe-task-definition", "--task-definition", taskDefinition)

    def update_service(self, cluster, service, forceNewDeployment):
        flag = "--force-new-deployment" if forceNewDeployment else "--no-force-new-deployment"
        self._call("ecs", "update-service", "--cluster", cluster, "--service", service, flag,
                   parse=False)

    def get_waiter(self, name):
        cli = self

        class _Wait:
            def wait(self, cluster, services, WaiterConfig=None):
                cli._call("ecs", "wait", name.replace("_", "-"), "--cluster", cluster,
                          "--services", *services, parse=False)
        return _Wait()

    def list_tasks(self, cluster, serviceName, desiredStatus):
        return self._call("ecs", "list-tasks", "--cluster", cluster, "--service-name",
                          serviceName, "--desired-status", desiredStatus)

    def describe_tasks(self, cluster, tasks):
        return self._call("ecs", "describe-tasks", "--cluster", cluster, "--tasks", *tasks)

    # Secrets Manager
    def get_secret_value(self, SecretId):
        return self._call("secretsmanager", "get-secret-value", "--secret-id", SecretId)

    def put_secret_value(self, SecretId, SecretString):
        # Through a 0600 temp file, not argv. AWS_CLI_FILE_ENCODING makes the
        # CLI read it as UTF-8; the file is written without a BOM.
        fd, path = tempfile.mkstemp(suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
                fh.write(SecretString)
            self._call("secretsmanager", "put-secret-value", "--secret-id", SecretId,
                       "--secret-string", f"file://{path}",
                       env={"AWS_CLI_FILE_ENCODING": "UTF-8"})
        finally:
            os.unlink(path)

    # CloudWatch Logs
    def filter_log_events(self, logGroupName, logStreamNames, filterPattern, startTime):
        return self._call("logs", "filter-log-events", "--log-group-name", logGroupName,
                          "--log-stream-names", *logStreamNames, "--filter-pattern",
                          filterPattern, "--start-time", str(startTime))


def _cli_clients(region: str) -> Callable[[], Clients]:
    def make() -> Clients:
        cli = AwsCli(region)
        return Clients(ecs=cli, secrets=cli, logs=cli)
    return make


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--key", required=True)
    ap.add_argument("--value", required=True)
    ap.add_argument("--execute", default="false")
    ap.add_argument("--cluster", default="cerebrum")
    ap.add_argument("--service", default="the-fork")
    ap.add_argument("--region", default="us-west-2")
    args = ap.parse_args(argv)
    execute = str(args.execute).strip().lower() == "true"
    return run(args.key, args.value, execute, _cli_clients(args.region),
               args.cluster, args.service)


if __name__ == "__main__":
    sys.exit(main())
