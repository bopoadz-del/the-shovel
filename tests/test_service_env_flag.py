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


def test_allowlist_holds_no_credential_or_url_names():
    for key in sef.ALLOWLIST:
        assert not any(w in key for w in ("KEY", "SECRET", "TOKEN", "PASSWORD", "URL", "DSN"))


def test_proof_pattern_names_the_scheduler_lines():
    src = (_ROOT / "app" / "core" / "hydration_scheduler.py").read_text(encoding="utf-8")
    on = sef.proof_pattern("HYDRATION_ENABLED", "true")
    off = sef.proof_pattern("HYDRATION_ENABLED", "false")
    assert on and on in src
    assert off and off in src
    assert sef.proof_pattern("HYDRATION_HOUR_UTC", "3") == "hydration scheduler started, target = 03:00"
    assert sef.proof_pattern("HYDRATION_HOUR_UTC", "17").endswith("17:00")
    assert sef.proof_pattern("PDF_OCR_PAGE_CAP", "80") is None
    assert sef.proof_pattern("HYDRATION_LLM_SUMMARY", "true") is None


# ---------------------------------------------------------------- mapping

def _td(secrets=(), environment=(), name="web", log=True):
    c = {"name": name, "secrets": list(secrets), "environment": list(environment)}
    if log:
        c["logConfiguration"] = {"options": {"awslogs-group": "/ecs/svc",
                                             "awslogs-stream-prefix": "ecs"}}
    return {"containerDefinitions": [c]}


def test_parse_value_from_json_key_and_pins():
    assert sef.parse_value_from(f"{_ARN}:FLAG_A::") == (_ARN, "FLAG_A", False)
    assert sef.parse_value_from(f"{_ARN}:FLAG_B") == (_ARN, "FLAG_B", False)
    assert sef.parse_value_from(_ARN) == (_ARN, "", False)
    assert sef.parse_value_from(f"{_ARN}:FLAG_A:AWSPREVIOUS:") == (_ARN, "FLAG_A", True)
    assert sef.parse_value_from(f"{_ARN}:FLAG_A::v-123") == (_ARN, "FLAG_A", True)


@pytest.mark.parametrize("value_from", [
    "arn:aws:ssm:us-west-2:111122223333:parameter/x",
    "not-an-arn",
])
def test_parse_value_from_refuses_non_secretsmanager(value_from):
    with pytest.raises(sef.Refused):
        sef.parse_value_from(value_from)


def test_find_mapping_variants():
    sec = _td(secrets=[{"name": "OTHER", "valueFrom": f"{_ARN}:OTHER::"},
                       {"name": "HYDRATION_ENABLED", "valueFrom": f"{_ARN}:HYDRATION_ENABLED::"}])
    m = sef.find_mapping(sec, "HYDRATION_ENABLED")
    assert (m.where, m.container, m.secret_id, m.json_key, m.pinned) == (
        "secret", "web", _ARN, "HYDRATION_ENABLED", False)
    env = _td(environment=[{"name": "PDF_OCR_PAGE_CAP", "value": "80"}])
    assert sef.find_mapping(env, "PDF_OCR_PAGE_CAP").where == "environment"
    assert sef.find_mapping(env, "HYDRATION_ENABLED").where == "absent"
    assert sef.find_mapping({}, "HYDRATION_ENABLED").where == "absent"


@pytest.mark.parametrize("mapping", [
    sef.Mapping("absent"),
    sef.Mapping("environment", "web"),
    sef.Mapping("secret", "web", _ARN, "", False),
    sef.Mapping("secret", "web", _ARN, "K", True),
])
def test_unwritable_mappings_are_refused(mapping):
    with pytest.raises(sef.Refused):
        sef.refuse_unless_writable(mapping)


def test_writable_mapping_passes():
    sef.refuse_unless_writable(sef.Mapping("secret", "web", _ARN, "K", False))


# ---------------------------------------------------------------- rewrite

def test_rewrite_sets_one_key_and_keeps_the_rest_byte_safe():
    src = json.dumps({"A_TOKEN": "s3crét", "HYDRATION_ENABLED": "false", "N": 1},
                     ensure_ascii=False)
    out = sef.rewrite(src, "HYDRATION_ENABLED", "true")
    assert json.loads(out) == {"A_TOKEN": "s3crét", "HYDRATION_ENABLED": "true", "N": 1}
    assert "é" in out  # ensure_ascii=False: no \\u escapes introduced
    assert not out.encode("utf-8").startswith(b"\xef\xbb\xbf")


def test_rewrite_adds_a_missing_key_and_drops_an_inbound_bom():
    out = sef.rewrite("﻿" + json.dumps({"X": "1"}), "PDF_OCR_PAGE_CAP", "90")
    assert not out.startswith("﻿")
    assert json.loads(out) == {"X": "1", "PDF_OCR_PAGE_CAP": "90"}


def test_read_value_present_absent_and_non_object():
    assert sef.read_value(json.dumps({"K": "false"}), "K") == "false"
    assert sef.read_value(json.dumps({"K": 5}), "K") == "5"
    assert sef.read_value(json.dumps({"J": "x"}), "K") == "<absent>"
    with pytest.raises(sef.Refused):
        sef.read_value(json.dumps(["K"]), "K")


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
        tasks = [{"taskArn": "arn:task/cerebrum/old1", "startedAt": datetime.fromtimestamp(10, timezone.utc)},
                 {"taskArn": "arn:task/cerebrum/new1", "startedAt": datetime.fromtimestamp(now + 5, timezone.utc)}]
    secrets = _Secrets(doc, calls, drop_write)
    clients = sef.Clients(ecs=_Ecs(td, tasks, calls), secrets=secrets,
                          logs=_Logs(list(messages), calls))
    rc = sef.run(key, value, execute, lambda: clients, "cerebrum", "svc",
                 out=lines.append, sleep=lambda s: None, now=lambda: now, log_attempts=2)
    return rc, lines, calls, secrets


def test_refused_key_never_builds_clients():
    built = []
    lines = []
    rc = sef.run("DATABASE_URL", "x", True, lambda: built.append(1), "c", "s", out=lines.append)
    assert rc == 2 and built == [] and lines[0].startswith("REFUSED")


def test_plan_prints_only_the_one_key_and_writes_nothing():
    rc, lines, calls, secrets = _run("HYDRATION_ENABLED", "true", False)
    text = "\n".join(lines)
    assert rc == 0
    assert "HYDRATION_ENABLED=false" in text and "mapped_in_task_definition=secret" in text
    assert "must-not-print" not in text and "API_TOKEN" not in text
    assert calls == []
    assert json.loads(secrets.value)["HYDRATION_ENABLED"] == "false"


@pytest.mark.parametrize("execute", [False, True])
def test_unmapped_key_is_refused_without_reading_or_writing(execute):
    td = _td(secrets=[{"name": "OTHER", "valueFrom": f"{_ARN}:OTHER::"}])
    rc, lines, calls, _ = _run("HYDRATION_ENABLED", "true", execute, td=td)
    assert rc == 2 and calls == []
    assert "mapped_in_task_definition=absent" in lines[0]
    assert any(line.startswith("REFUSED") for line in lines)


def test_execute_writes_rolls_and_proves_the_scheduler_armed():
    armed = "INFO app.core.hydration_scheduler hydration scheduler started, target = 01:00 UTC"
    rc, lines, calls, secrets = _run("HYDRATION_ENABLED", "true", True,
                                     messages=["unrelated", armed])
    assert rc == 0, lines
    assert json.loads(secrets.value) == {"HYDRATION_ENABLED": "true", "API_TOKEN": "must-not-print"}
    assert [c[0] for c in calls] == ["put_secret_value", "update_service", "wait", "filter_log_events"]
    assert calls[3][1] == ("ecs/web/new1",)  # only the task started after the roll
    assert "post-rollout read-back: HYDRATION_ENABLED=true" in lines
    assert lines[-1].endswith(armed)
    assert "must-not-print" not in "\n".join(lines)


def test_execute_fails_when_no_new_task_logs_the_arm_line():
    rc, lines, calls, _ = _run("HYDRATION_ENABLED", "true", True,
                               messages=["hydration scheduler disabled via HYDRATION_ENABLED=false"])
    assert rc == 1 and lines[-1].startswith("FAILED: no new task logged")
    assert [c[0] for c in calls].count("filter_log_events") == 2  # retried


def test_execute_fails_when_no_task_started_after_the_roll():
    old = [{"taskArn": "arn:task/cerebrum/old1", "startedAt": 10.0}]
    rc, lines, _, _ = _run("HYDRATION_ENABLED", "true", True, tasks=old)
    assert rc == 1 and lines[-1] == "FAILED: no task started after the forced deployment"


def test_execute_stops_before_the_roll_when_the_write_did_not_land():
    rc, lines, calls, _ = _run("HYDRATION_ENABLED", "true", True, drop_write=True)
    assert rc == 1 and "service not rolled" in lines[-1]
    assert [c[0] for c in calls] == ["put_secret_value"]


def test_execute_for_a_key_without_a_startup_line_ends_on_read_back():
    rc, lines, calls, secrets = _run("PDF_OCR_PAGE_CAP", "120", True)
    assert rc == 0
    assert json.loads(secrets.value)["PDF_OCR_PAGE_CAP"] == "120"
    assert [c[0] for c in calls] == ["put_secret_value", "update_service", "wait"]
    assert "read-back only" in lines[-1]


def test_execute_refuses_when_the_container_has_no_awslogs():
    td = _td(secrets=[{"name": "HYDRATION_ENABLED", "valueFrom": f"{_ARN}:HYDRATION_ENABLED::"}],
             log=False)
    with pytest.raises(sef.Refused):
        _run("HYDRATION_ENABLED", "true", True, td=td)


# ---------------------------------------------------------------- workflow

def test_workflow_is_dispatch_only_and_shares_the_deploy_lock():
    doc = yaml.safe_load((_ROOT / ".github" / "workflows" / "service-env-flag.yml")
                         .read_text(encoding="utf-8"))
    trigger = doc["on"] if "on" in doc else doc[True]
    assert set(trigger) == {"workflow_dispatch"}
    assert set(trigger["workflow_dispatch"]["inputs"]) == {"key", "value", "execute"}
    assert trigger["workflow_dispatch"]["inputs"]["execute"]["default"] is False
    assert doc["permissions"] == {"contents": "read"}
    assert doc["concurrency"]["group"] == "deploy-aws"
    assert doc["concurrency"]["cancel-in-progress"] is False
