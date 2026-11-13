"""P-H1: the service-env-flag ops helper. Every test builds its own fake
task definition, secret and AWS clients; nothing touches AWS."""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "service_env_flag", _ROOT / "scripts" / "service_env_flag.py")
sef = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = sef
_spec.loader.exec_module(sef)

_ARN = "arn:aws:secretsmanager:us-west-2:111122223333:secret:svc/app-AbCdEf"


# ---------------------------------------------------------------- validation

@pytest.mark.parametrize("key,value,want", [
    ("HYDRATION_ENABLED", "TRUE ", "true"),
    ("HYDRATION_LLM_SUMMARY", "false", "false"),
    ("HYDRATION_HOUR_UTC", "0", "0"),
    ("HYDRATION_HOUR_UTC", "23", "23"),
    ("PDF_OCR_PAGE_CAP", "080", "80"),
    ("PDF_OCR_FORCE_PAGE_CAP", "1", "1"),
])
def test_allowed_values_are_normalised(key, value, want):
    assert sef.validate_request(key, value) == want


@pytest.mark.parametrize("key,value", [
    ("DATABASE_URL", "x"),                 # not on the allowlist
    ("DEEPSEEK_API_KEY", "x"),             # credential
    ("hydration_enabled", "true"),         # allowlist is exact
    ("HYDRATION_ENABLED", "yes"),          # bool kind takes true/false only
    ("HYDRATION_ENABLED", ""),
    ("HYDRATION_HOUR_UTC", "24"),          # hour out of range
    ("HYDRATION_HOUR_UTC", "-1"),
    ("PDF_OCR_PAGE_CAP", "0"),             # posint must be >= 1
    ("PDF_OCR_BOQ_PAGE_CAP", "1e3"),
    ("PDF_OCR_BOQ_PAGE_CAP", "1234567"),
])
def test_refused_requests(key, value):
    with pytest.raises(sef.Refused):
        sef.validate_request(key, value)


@pytest.mark.parametrize("word", ["KEY", "SECRET", "TOKEN", "PASSWORD", "URL", "DSN"])
def test_allowlist_holds_no_credential_or_url_names(word):
    assert not [k for k in sef.ALLOWLIST if word in k]


@pytest.mark.parametrize("value", ["true", "false"])
def test_hydration_enabled_proof_line_exists_in_the_scheduler(value):
    src = (_ROOT / "app" / "core" / "hydration_scheduler.py").read_text(encoding="utf-8")
    pattern = sef.proof_pattern("HYDRATION_ENABLED", value)
    assert pattern and pattern in src


@pytest.mark.parametrize("key,value,want", [
    ("HYDRATION_HOUR_UTC", "3", "hydration scheduler started, target = 03:00"),
    ("HYDRATION_HOUR_UTC", "17", "hydration scheduler started, target = 17:00"),
    ("PDF_OCR_PAGE_CAP", "80", None),
    ("HYDRATION_LLM_SUMMARY", "true", None),
])
def test_proof_pattern_per_key(key, value, want):
    assert sef.proof_pattern(key, value) == want


# ---------------------------------------------------------------- mapping

def _td(secrets=(), environment=(), name="web", log=True):
    c = {"name": name, "secrets": list(secrets), "environment": list(environment)}
    if log:
        c["logConfiguration"] = {"options": {"awslogs-group": "/ecs/svc",
                                             "awslogs-stream-prefix": "ecs"}}
    return {"containerDefinitions": [c]}


@pytest.mark.parametrize("suffix,want_key,want_pinned", [
    (":FLAG_A::", "FLAG_A", False),
    (":FLAG_B", "FLAG_B", False),
    ("", "", False),
    (":FLAG_A:AWSPREVIOUS:", "FLAG_A", True),
    (":FLAG_A::v-123", "FLAG_A", True),
])
def test_parse_value_from_json_key_and_pins(suffix, want_key, want_pinned):
    assert sef.parse_value_from(_ARN + suffix) == (_ARN, want_key, want_pinned)


@pytest.mark.parametrize("value_from", [
    "arn:aws:ssm:us-west-2:111122223333:parameter/x",
    "not-an-arn",
])
def test_parse_value_from_refuses_non_secretsmanager(value_from):
    with pytest.raises(sef.Refused):
        sef.parse_value_from(value_from)


_SEC_TD = {"containerDefinitions": [{"name": "web", "secrets": [
    {"name": "OTHER", "valueFrom": f"{_ARN}:OTHER::"},
    {"name": "HYDRATION_ENABLED", "valueFrom": f"{_ARN}:HYDRATION_ENABLED::"}]}]}
_ENV_TD = {"containerDefinitions": [{"name": "api", "environment": [
    {"name": "PDF_OCR_PAGE_CAP", "value": "80"}]}]}


@pytest.mark.parametrize("td,key,want", [
    (_SEC_TD, "HYDRATION_ENABLED", ("secret", "web", _ARN, "HYDRATION_ENABLED", False)),
    (_SEC_TD, "PDF_OCR_PAGE_CAP", ("absent", "", "", "", False)),
    (_ENV_TD, "PDF_OCR_PAGE_CAP", ("environment", "api", "", "", False)),
    (_ENV_TD, "HYDRATION_ENABLED", ("absent", "", "", "", False)),
    ({}, "HYDRATION_ENABLED", ("absent", "", "", "", False)),
])
def test_find_mapping_variants(td, key, want):
    m = sef.find_mapping(td, key)
    assert (m.where, m.container, m.secret_id, m.json_key, m.pinned) == want


@pytest.mark.parametrize("mapping", [
    sef.Mapping("absent"),
    sef.Mapping("environment", "web"),
    sef.Mapping("secret", "web", _ARN, "", False),
    sef.Mapping("secret", "web", _ARN, "K", True),
])
def test_unwritable_mappings_are_refused(mapping):
    with pytest.raises(sef.Refused):
        sef.refuse_unless_writable(mapping)


@pytest.mark.parametrize("json_key", ["HYDRATION_ENABLED", "PDF_OCR_PAGE_CAP"])
def test_writable_mapping_passes(json_key):
    sef.refuse_unless_writable(sef.Mapping("secret", "web", _ARN, json_key, False))


# ---------------------------------------------------------------- rewrite

@pytest.mark.parametrize("prefix,doc,key,value,want", [
    ("", {"A_TOKEN": "s3crét", "HYDRATION_ENABLED": "false", "N": 1},
     "HYDRATION_ENABLED", "true",
     {"A_TOKEN": "s3crét", "HYDRATION_ENABLED": "true", "N": 1}),
    ("﻿", {"X": "مشروع"}, "PDF_OCR_PAGE_CAP", "90",
     {"X": "مشروع", "PDF_OCR_PAGE_CAP": "90"}),
])
def test_rewrite_sets_one_key_keeps_the_rest_and_writes_no_bom(prefix, doc, key, value, want):
    out = sef.rewrite(prefix + json.dumps(doc, ensure_ascii=False), key, value)
    assert json.loads(out) == want
    assert "\\u" not in out  # ensure_ascii=False: no escapes introduced
    assert not out.encode("utf-8").startswith(b"\xef\xbb\xbf")


@pytest.mark.parametrize("doc,want", [
    ({"K": "false"}, "false"),
    ({"K": 5}, "5"),
    ({"J": "x"}, "<absent>"),
])
def test_read_value_present_and_absent(doc, want):
    assert sef.read_value(json.dumps(doc), "K") == want


@pytest.mark.parametrize("doc", [["K"], "K"])
def test_read_value_refuses_a_non_object_secret(doc):
    with pytest.raises(sef.Refused):
        sef.read_value(json.dumps(doc), "K")


# ---------------------------------------------------------------- AWS flow

class _Waiter:
    def __init__(self, calls):
        self.calls = calls

    def wait(self, **kw):
        self.calls.append(("wait", kw["services"][0]))


class _Ecs:
    def __init__(self, td, tasks, calls):
        self.td, self.tasks, self.calls = td, tasks, calls

    def describe_services(self, cluster, services):
        return {"services": [{"taskDefinition": "arn:td/svc:7"}]}

    def describe_task_definition(self, taskDefinition):
        return {"taskDefinition": self.td}

    def update_service(self, **kw):
        self.calls.append(("update_service", kw["forceNewDeployment"]))

    def get_waiter(self, name):
        assert name == "services_stable"
        return _Waiter(self.calls)

    def list_tasks(self, **kw):
        return {"taskArns": [t["taskArn"] for t in self.tasks]}

    def describe_tasks(self, cluster, tasks):
        return {"tasks": self.tasks}


class _Secrets:
    def __init__(self, doc, calls, drop_write=False):
        self.value = json.dumps(doc)
        self.calls, self.drop_write = calls, drop_write

    def get_secret_value(self, SecretId):
        assert SecretId == _ARN
        return {"SecretString": self.value}

    def put_secret_value(self, SecretId, SecretString):
        assert isinstance(SecretString, str)
        self.calls.append(("put_secret_value", SecretId))
        if not self.drop_write:
            self.value = SecretString


class _Logs:
    def __init__(self, messages, calls):
        self.messages, self.calls = messages, calls

    def filter_log_events(self, **kw):
        self.calls.append(("filter_log_events", tuple(kw["logStreamNames"])))
        return {"events": [{"message": m} for m in self.messages]}


def _run(key, value, execute, *, td=None, doc=None, tasks=None, messages=(),
         drop_write=False, now=1000.0):
    calls, lines = [], []
    td = td if td is not None else _td(
        secrets=[{"name": key, "valueFrom": f"{_ARN}:{key}::"}])
    doc = doc if doc is not None else {key: "false", "API_TOKEN": "must-not-print"}
    if tasks is None:
        tasks = [{"taskArn": "arn:task/cerebrum/old1",
                  "startedAt": datetime.fromtimestamp(10, timezone.utc)},
                 {"taskArn": "arn:task/cerebrum/new1",
                  "startedAt": datetime.fromtimestamp(now + 5, timezone.utc)}]
    secrets = _Secrets(doc, calls, drop_write)
    clients = sef.Clients(ecs=_Ecs(td, tasks, calls), secrets=secrets,
                          logs=_Logs(list(messages), calls))
    rc = sef.run(key, value, execute, lambda: clients, "cerebrum", "svc",
                 out=lines.append, sleep=lambda s: None, now=lambda: now, log_attempts=2)
    return rc, lines, calls, secrets


@pytest.mark.parametrize("key,execute", [("DATABASE_URL", True), ("OPENROUTER_API_KEY", False)])
def test_refused_key_never_builds_clients(key, execute):
    built, lines = [], []
    rc = sef.run(key, "x", execute, lambda: built.append(1), "c", "s", out=lines.append)
    assert rc == 2
    assert built == []
    assert lines[0].startswith("REFUSED")


@pytest.mark.parametrize("key,value,current", [
    ("HYDRATION_ENABLED", "true", "false"),
    ("PDF_OCR_BOQ_PAGE_CAP", "200", "160"),
])
def test_plan_prints_only_the_one_key_and_writes_nothing(key, value, current):
    rc, lines, calls, secrets = _run(key, value, False,
                                     doc={key: current, "API_TOKEN": "must-not-print"})
    text = "\n".join(lines)
    assert rc == 0
    assert f"plan: {key}={current} (requested {value})" in lines
    assert "mapped_in_task_definition=secret" in text
    assert "must-not-print" not in text and "API_TOKEN" not in text
    assert calls == []
    assert json.loads(secrets.value)[key] == current


@pytest.mark.parametrize("execute", [False, True])
def test_unmapped_key_is_refused_without_reading_or_writing(execute):
    td = _td(secrets=[{"name": "OTHER", "valueFrom": f"{_ARN}:OTHER::"}])
    rc, lines, calls, _ = _run("HYDRATION_ENABLED", "true", execute, td=td)
    assert rc == 2 and calls == []
    assert "mapped_in_task_definition=absent" in lines[0]
    assert any(line.startswith("REFUSED") for line in lines)


@pytest.mark.parametrize("key,value,armed", [
    ("HYDRATION_ENABLED", "true",
     "INFO app.core.hydration_scheduler hydration scheduler started, target = 01:00 UTC"),
    ("HYDRATION_HOUR_UTC", "4",
     "INFO app.core.hydration_scheduler hydration scheduler started, target = 04:00 Asia/Dubai"),
])
def test_execute_writes_rolls_and_proves_the_scheduler_armed(key, value, armed):
    rc, lines, calls, secrets = _run(key, value, True, messages=["unrelated", armed])
    assert rc == 0, lines
    assert json.loads(secrets.value) == {key: value, "API_TOKEN": "must-not-print"}
    assert [c[0] for c in calls] == ["put_secret_value", "update_service", "wait",
                                     "filter_log_events"]
    assert calls[3][1] == ("ecs/web/new1",)  # only the task started after the roll
    assert f"post-rollout read-back: {key}={value}" in lines
    assert lines[-1].endswith(armed)
    assert "must-not-print" not in "\n".join(lines)


@pytest.mark.parametrize("key,value,messages", [
    ("HYDRATION_ENABLED", "true", ["hydration scheduler disabled via HYDRATION_ENABLED=false"]),
    ("HYDRATION_ENABLED", "false", []),
])
def test_execute_fails_when_no_new_task_logs_the_arm_line(key, value, messages):
    rc, lines, calls, _ = _run(key, value, True, messages=messages)
    assert rc == 1 and lines[-1].startswith("FAILED: no new task logged")
    assert [c[0] for c in calls].count("filter_log_events") == 2  # retried


@pytest.mark.parametrize("tasks", [
    [{"taskArn": "arn:task/cerebrum/old1", "startedAt": 10.0}],
    [],
])
def test_execute_fails_when_no_task_started_after_the_roll(tasks):
    rc, lines, _, _ = _run("HYDRATION_ENABLED", "true", True, tasks=tasks)
    assert rc == 1 and lines[-1] == "FAILED: no task started after the forced deployment"


@pytest.mark.parametrize("key,value", [("HYDRATION_ENABLED", "true"), ("PDF_OCR_PAGE_CAP", "99")])
def test_execute_stops_before_the_roll_when_the_write_did_not_land(key, value):
    rc, lines, calls, _ = _run(key, value, True, drop_write=True)
    assert rc == 1 and "service not rolled" in lines[-1]
    assert [c[0] for c in calls] == ["put_secret_value"]


@pytest.mark.parametrize("key,value", [
    ("PDF_OCR_PAGE_CAP", "120"),
    ("HYDRATION_LLM_SUMMARY", "true"),
])
def test_execute_for_a_key_without_a_startup_line_ends_on_read_back(key, value):
    rc, lines, calls, secrets = _run(key, value, True)
    assert rc == 0
    assert json.loads(secrets.value)[key] == value
    assert [c[0] for c in calls] == ["put_secret_value", "update_service", "wait"]
    assert "read-back only" in lines[-1]


@pytest.mark.parametrize("key", ["HYDRATION_ENABLED", "HYDRATION_HOUR_UTC"])
def test_execute_refuses_when_the_container_has_no_awslogs(key):
    td = _td(secrets=[{"name": key, "valueFrom": f"{_ARN}:{key}::"}], log=False)
    with pytest.raises(sef.Refused):
        _run(key, "1" if key == "HYDRATION_HOUR_UTC" else "true", True, td=td)


# ---------------------------------------------------------------- workflow

def _workflow(name: str) -> dict:
    return yaml.safe_load((_ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("path,want", [
    (("permissions",), {"contents": "read"}),
    (("concurrency", "group"), "deploy-aws"),
    (("concurrency", "cancel-in-progress"), False),
])
def test_workflow_permissions_and_deploy_lock(path, want):
    node = _workflow("service-env-flag.yml")
    for part in path:
        node = node[part]
    assert node == want


@pytest.mark.parametrize("workflow", ["service-env-flag.yml", "corpus-ghost-tombstone.yml"])
def test_ops_workflow_is_dispatch_only_with_execute_off_by_default(workflow):
    doc = _workflow(workflow)
    trigger = doc["on"] if "on" in doc else doc[True]
    assert set(trigger) == {"workflow_dispatch"}
    assert trigger["workflow_dispatch"]["inputs"]["execute"]["default"] is False
