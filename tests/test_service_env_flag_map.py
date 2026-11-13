"""P-H1c: service-env-flag map mode. Every test builds its own fake live and
latest task definitions, secret and AWS clients; nothing touches AWS."""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "service_env_flag_map_under_test", _ROOT / "scripts" / "service_env_flag.py")
sef = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = sef
_spec.loader.exec_module(sef)

_ARN = "arn:aws:secretsmanager:us-west-2:111122223333:secret:svc/app-AbCdEf"
_OTHER_ARN = "arn:aws:secretsmanager:us-west-2:111122223333:secret:svc/other-XyZ"
_TD_ARN = "arn:aws:ecs:us-west-2:111122223333:task-definition/fam"


def _live(rev=7, secrets=None, environment=None, image="repo/app:latest", cpu="1024"):
    secrets = secrets if secrets is not None else [
        {"name": "SECRET_KEY", "valueFrom": f"{_ARN}:SECRET_KEY::"},
        {"name": "HYDRATION_ENABLED", "valueFrom": f"{_ARN}:HYDRATION_ENABLED::"}]
    return {
        "taskDefinitionArn": f"{_TD_ARN}:{rev}", "family": "fam", "revision": rev,
        "status": "ACTIVE", "registeredAt": f"2026-11-0{rev % 9 + 1}T00:00:00Z",
        "registeredBy": "arn:aws:iam::111122223333:user/someone",
        "requiresAttributes": [{"name": "ecs.capability.secrets.asm.environment-variables"}],
        "compatibilities": ["EC2", "FARGATE"],
        "executionRoleArn": "arn:aws:iam::111122223333:role/exec",
        "networkMode": "awsvpc", "requiresCompatibilities": ["FARGATE"],
        "cpu": cpu, "memory": "2048",
        "containerDefinitions": [{
            "name": "web", "image": image,
            "environment": environment if environment is not None else [
                {"name": "ENV", "value": "production"}],
            "secrets": secrets,
            "logConfiguration": {"logDriver": "awslogs", "options": {
                "awslogs-group": "/ecs/svc", "awslogs-stream-prefix": "ecs"}},
        }],
    }


class _Waiter:
    def __init__(self, calls):
        self.calls = calls

    def wait(self, **kw):
        self.calls.append(("wait", kw["services"][0]))


class _Ecs:
    """Service runs `live`; the family's latest revision is `latest`.
    register adds a revision; update_service(taskDefinition=) switches the
    service to it unless `ignore_roll`."""

    def __init__(self, live, latest, calls, ignore_roll=False):
        self.defs = {live["taskDefinitionArn"]: live, latest["taskDefinitionArn"]: latest}
        self.service_td = live["taskDefinitionArn"]
        self.latest = latest["taskDefinitionArn"]
        self.calls, self.ignore_roll = calls, ignore_roll
        self.registered = []

    def describe_services(self, cluster, services):
        return {"services": [{"taskDefinition": self.service_td}]}

    def describe_task_definition(self, taskDefinition):
        arn = self.latest if taskDefinition == "fam" else taskDefinition
        return {"taskDefinition": copy.deepcopy(self.defs[arn])}

    def register_task_definition(self, **spec):
        rev = max(d["revision"] for d in self.defs.values()) + 1
        arn = f"{_TD_ARN}:{rev}"
        self.defs[arn] = {**copy.deepcopy(spec), "taskDefinitionArn": arn, "revision": rev}
        self.latest = arn
        self.registered.append(copy.deepcopy(spec))
        self.calls.append(("register_task_definition", arn))
        return {"taskDefinition": self.defs[arn]}

    def update_service(self, **kw):
        self.calls.append(("update_service", kw.get("taskDefinition"), kw["forceNewDeployment"]))
        if kw.get("taskDefinition") and not self.ignore_roll:
            self.service_td = kw["taskDefinition"]

    def get_waiter(self, name):
        assert name == "services_stable"
        return _Waiter(self.calls)

    def list_tasks(self, **kw):
        return {"taskArns": ["arn:task/cerebrum/new1"]}

    def describe_tasks(self, cluster, tasks):
        return {"tasks": [{"taskArn": "arn:task/cerebrum/new1",
                           "startedAt": datetime.fromtimestamp(2000, timezone.utc)}]}


class _Secrets:
    def __init__(self, doc, calls):
        self.value, self.calls = json.dumps(doc), calls

    def get_secret_value(self, SecretId):
        assert SecretId == _ARN
        return {"SecretString": self.value}

    def put_secret_value(self, SecretId, SecretString):
        self.calls.append(("put_secret_value", SecretId))
        self.value = SecretString


class _Logs:
    def __init__(self, messages, calls):
        self.messages, self.calls = messages, calls

    def filter_log_events(self, **kw):
        self.calls.append(("filter_log_events", tuple(kw["logStreamNames"])))
        return {"events": [{"message": m} for m in self.messages]}


def _run(key, value, execute, *, live=None, latest=None, doc=None, messages=(),
         ignore_roll=False):
    calls, lines = [], []
    live = live if live is not None else _live()
    latest = latest if latest is not None else live
    doc = doc if doc is not None else {"SECRET_KEY": "must-not-print"}
    ecs = _Ecs(live, latest, calls, ignore_roll)
    secrets = _Secrets(doc, calls)
    clients = sef.Clients(ecs=ecs, secrets=secrets, logs=_Logs(list(messages), calls))
    rc = sef.run(key, value, execute, lambda: clients, "cerebrum", "svc",
                 out=lines.append, sleep=lambda s: None, now=lambda: 1000.0,
                 log_attempts=2, map_key=True)
    return rc, lines, calls, ecs, secrets


# ---------------------------------------------------------------- absent key

@pytest.mark.parametrize("key,value", [
    ("HYDRATION_LLM_SUMMARY", "true"),
    ("PDF_OCR_PAGE_CAP", "120"),
])
def test_plan_prints_the_one_entry_diff_and_registers_nothing(key, value):
    rc, lines, calls, ecs, secrets = _run(key, value, False)
    text = "\n".join(lines)
    assert rc == 0, lines
    assert calls == [] and ecs.registered == []
    want = (f"+ containerDefinitions[web].secrets: name={key} "
            f"valueFrom=arn:aws:secretsmanager:us-west-2:***:secret:svc/app-AbCdEf:{key}::")
    assert want in lines
    assert "111122223333" not in text and "must-not-print" not in text
    assert f"plan: {key}=<absent> (requested {value})" in lines
    assert lines[-1] == "execute=false: nothing registered or written"


@pytest.mark.parametrize("key,value", [
    ("HYDRATION_LLM_SUMMARY", "true"),
    ("PDF_OCR_BOQ_PAGE_CAP", "300"),
])
def test_execute_registers_one_added_entry_and_nothing_else_changed(key, value):
    live = _live()
    rc, lines, calls, ecs, secrets = _run(key, value, True, live=live)
    assert rc == 0, lines
    assert len(ecs.registered) == 1
    spec = ecs.registered[0]
    want = sef.register_input(live)
    want["containerDefinitions"][0]["secrets"].append(
        {"name": key, "valueFrom": f"{_ARN}:{key}::"})
    assert spec == want
    # ECS-assigned metadata is never sent back to register.
    for field in ("taskDefinitionArn", "revision", "status", "registeredAt",
                  "registeredBy", "requiresAttributes", "compatibilities"):
        assert field not in spec
    assert [c[0] for c in calls] == ["register_task_definition", "put_secret_value",
                                     "update_service", "wait"]
    assert calls[2] == ("update_service", f"{_TD_ARN}:8", True)  # rolled ONTO the new revision
    assert json.loads(secrets.value) == {"SECRET_KEY": "must-not-print", key: value}
    assert f"live task definition maps {key}: fam:8" in lines
    assert f"post-rollout read-back: {key}={value}" in lines
    assert "read-back only" in lines[-1]
    assert "must-not-print" not in "\n".join(lines)


@pytest.mark.parametrize("value,armed", [
    ("true", "INFO app.core.hydration_scheduler hydration scheduler started, target = 01:00 UTC"),
    ("false", "INFO app.core.hydration_scheduler hydration scheduler disabled via "
              "HYDRATION_ENABLED=false"),
])
def test_execute_with_a_proof_line_finds_it_in_a_new_task(value, armed):
    live = _live(secrets=[{"name": "SECRET_KEY", "valueFrom": f"{_ARN}:SECRET_KEY::"}])
    rc, lines, calls, _, _ = _run("HYDRATION_ENABLED", value, True, live=live,
                                  messages=["unrelated", armed])
    assert rc == 0, lines
    assert calls[-1] == ("filter_log_events", ("ecs/web/new1",))
    assert lines[-1].endswith(armed)


@pytest.mark.parametrize("key", ["HYDRATION_LLM_SUMMARY", "PDF_OCR_PAGE_CAP"])
def test_execute_fails_when_the_service_did_not_take_the_new_revision(key):
    rc, lines, calls, _, _ = _run(key, "true" if key.startswith("HYD") else "9", True,
                                  ignore_roll=True)
    assert rc == 1
    assert lines[-1] == "FAILED: the service does not run the registered revision fam:8"
    assert "post-rollout read-back" not in "\n".join(lines)


# ---------------------------------------------------------------- refusals

@pytest.mark.parametrize("live", [
    _live(secrets=[{"name": "SECRET_KEY", "valueFrom": f"{_ARN}:SECRET_KEY::"},
                   {"name": "HYDRATION_LLM_SUMMARY",
                    "valueFrom": f"{_ARN}:HYDRATION_LLM_SUMMARY::"}]),
    _live(environment=[{"name": "HYDRATION_LLM_SUMMARY", "value": "false"}]),
])
@pytest.mark.parametrize("execute", [False, True])
def test_an_already_mapped_key_is_refused_and_nothing_is_registered(live, execute):
    rc, lines, calls, ecs, _ = _run("HYDRATION_LLM_SUMMARY", "true", execute, live=live)
    assert rc == 2
    assert calls == [] and ecs.registered == []
    assert lines[-1].startswith("REFUSED: HYDRATION_LLM_SUMMARY is already mapped")


@pytest.mark.parametrize("latest,field", [
    (_live(rev=8, cpu="2048"), "cpu"),
    (_live(rev=8, image="repo/app:handmade"), "containerDefinitions"),
])
@pytest.mark.parametrize("execute", [False, True])
def test_a_drifted_definition_is_refused_and_nothing_is_registered(latest, field, execute):
    rc, lines, calls, ecs, _ = _run("HYDRATION_LLM_SUMMARY", "true", execute,
                                    live=_live(rev=7), latest=latest)
    assert rc == 2
    assert calls == [] and ecs.registered == []
    assert lines[-1].startswith("REFUSED: the live task definition (revision 7)")
    assert f"['{field}']" in lines[-1]
    assert "handmade" not in lines[-1] and "2048" not in lines[-1]  # names only


@pytest.mark.parametrize("latest_rev", [8, 12])
def test_a_later_revision_identical_but_for_metadata_is_not_drift(latest_rev):
    rc, lines, _, ecs, _ = _run("HYDRATION_LLM_SUMMARY", "true", True,
                                live=_live(rev=7), latest=_live(rev=latest_rev))
    assert rc == 0, lines
    assert len(ecs.registered) == 1


@pytest.mark.parametrize("secrets,why", [
    ([], "exactly one container mapping secrets, found 0"),
    ([{"name": "A", "valueFrom": f"{_ARN}:A::"},
      {"name": "B", "valueFrom": f"{_OTHER_ARN}:B::"}], "from one secret, found 2"),
    ([{"name": "WHOLE", "valueFrom": _ARN}], "from one secret, found 0"),
])
def test_add_secret_entry_refuses_without_one_source_secret(secrets, why):
    with pytest.raises(sef.Refused, match=why):
        sef.add_secret_entry(_live(secrets=secrets), "HYDRATION_LLM_SUMMARY")


@pytest.mark.parametrize("key", ["HYDRATION_LLM_SUMMARY", "PDF_OCR_PAGE_CAP"])
def test_add_secret_entry_does_not_mutate_the_live_definition(key):
    live = _live()
    before = copy.deepcopy(live)
    spec, container, entry = sef.add_secret_entry(live, key)
    assert live == before
    assert container == "web"
    assert entry == {"name": key, "valueFrom": f"{_ARN}:{key}::"}
    assert spec["containerDefinitions"][0]["secrets"][-1] == entry


@pytest.mark.parametrize("value_from,want", [
    (f"{_ARN}:K::", "arn:aws:secretsmanager:us-west-2:***:secret:svc/app-AbCdEf:K::"),
    (f"{_OTHER_ARN}:J::", "arn:aws:secretsmanager:us-west-2:***:secret:svc/other-XyZ:J::"),
])
def test_describe_entry_masks_the_account(value_from, want):
    line = sef.describe_entry("api", {"name": "K", "valueFrom": value_from})
    assert line == f"+ containerDefinitions[api].secrets: name=K valueFrom={want}"


# ---------------------------------------------------------------- CLI adapter

class _Proc:
    def __init__(self, stdout):
        self.returncode, self.stdout, self.stderr = 0, stdout, ""


class _Runner:
    def __init__(self, stdout="{}"):
        self.calls, self.files, self.stdout = [], [], stdout

    def __call__(self, cmd, **kw):
        self.calls.append(cmd)
        for arg in cmd:
            if arg.startswith("file://"):
                self.files.append(Path(arg[len("file://"):]).read_bytes())
        return _Proc(self.stdout)


@pytest.mark.parametrize("env_value", ["production", "é-utf8"])
def test_cli_register_goes_through_a_utf8_file_not_argv(env_value):
    runner = _Runner(stdout=json.dumps({"taskDefinition": {"taskDefinitionArn": "x:9"}}))
    cli = sef.AwsCli("us-west-2", runner=runner)
    spec = sef.register_input(_live(environment=[{"name": "ENV", "value": env_value}]))
    out = cli.register_task_definition(**spec)
    assert out["taskDefinition"]["taskDefinitionArn"] == "x:9"
    cmd = runner.calls[0]
    assert cmd[:4] == ["aws", "ecs", "register-task-definition", "--cli-input-json"]
    assert env_value not in " ".join(cmd)
    assert not runner.files[0].startswith(b"\xef\xbb\xbf")
    assert json.loads(runner.files[0].decode("utf-8")) == spec


@pytest.mark.parametrize("td,want", [
    ("arn:td/fam:8", ["--task-definition", "arn:td/fam:8", "--force-new-deployment"]),
    (None, ["--force-new-deployment"]),
])
def test_cli_update_service_pins_the_revision_only_when_given(td, want):
    runner = _Runner()
    sef.AwsCli("us-west-2", runner=runner).update_service(
        cluster="c", service="s", forceNewDeployment=True, taskDefinition=td)
    cmd = runner.calls[0]
    assert cmd[cmd.index("s") + 1: cmd.index("--region")] == want


@pytest.mark.parametrize("argv,want", [
    (["--map", "true"], True),
    ([], False),
])
def test_main_passes_map_through(monkeypatch, argv, want):
    seen = {}
    monkeypatch.setattr(sef, "run", lambda *a, **kw: seen.update(kw) or 0)
    assert sef.main(["--key", "HYDRATION_LLM_SUMMARY", "--value", "true", *argv]) == 0
    assert seen["map_key"] is want


@pytest.mark.parametrize("name,default", [("map", False), ("execute", False)])
def test_workflow_map_and_execute_inputs_default_off(name, default):
    wf = yaml.safe_load((_ROOT / ".github" / "workflows" / "service-env-flag.yml")
                        .read_text(encoding="utf-8"))
    inputs = wf[True]["workflow_dispatch"]["inputs"]  # yaml reads `on:` as True
    assert inputs[name]["type"] == "boolean" and inputs[name]["default"] is default
    run = wf["jobs"]["flag"]["steps"][-1]["run"]
    assert f'"$FLAG_{name.upper()}"' in run
