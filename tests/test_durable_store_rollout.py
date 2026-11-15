"""The durable store rollout renders one stage at a time and refuses the wrong order.

Every task definition here is built by the test. No AWS calls. Each
decision is exercised with at least two shapes (mount path, volume name,
extra settings), so a renderer that hardcodes one path or name fails.
"""
from __future__ import annotations

import json

import pytest

from scripts.durable_store_rollout import (
    BACKEND,
    GUARD,
    ROOT,
    RolloutRefused,
    efs_mount_path,
    environment,
    main,
    render,
)


def _described(
    *,
    mount: str = "/app/data",
    volume: str = "data",
    efs: bool = True,
    read_only: bool = False,
    env: dict[str, str] | None = None,
    secrets: tuple[str, ...] = ("DATA_DIR",),
    family: str = "fam-a",
    revision: int = 12,
) -> dict:
    vol: dict = {"name": volume}
    if efs:
        vol["efsVolumeConfiguration"] = {
            "fileSystemId": "fs-0", "rootDirectory": "/", "transitEncryption": "ENABLED",
        }
    else:
        vol["host"] = {}
    return {
        "taskDefinition": {
            "taskDefinitionArn": f"arn:aws:ecs:r:1:task-definition/{family}:{revision}",
            "family": family,
            "revision": revision,
            "status": "ACTIVE",
            "requiresAttributes": [{"name": "x"}],
            "compatibilities": ["EC2", "FARGATE"],
            "registeredAt": "2026-11-01T00:00:00Z",
            "registeredBy": "arn:aws:iam::1:user/u",
            "cpu": "1024",
            "memory": "4096",
            "volumes": [vol],
            "containerDefinitions": [{
                "name": "app",
                "image": "img:sha",
                "environment": [{"name": k, "value": v} for k, v in (env or {"KEEP_ME": "1"}).items()],
                "secrets": [{"name": s, "valueFrom": f"arn:secret:{s}"} for s in secrets],
                "mountPoints": [{"sourceVolume": volume, "containerPath": mount, "readOnly": read_only}],
            }],
        },
        "tags": [{"key": "team", "value": "t"}],
    }


def _env_of(payload: dict) -> dict[str, str]:
    return environment(payload)


@pytest.mark.parametrize(("mount", "volume"), [("/app/data", "data"), ("/mnt/efs/documents", "docs")])
def test_efs_stage_points_the_store_at_the_tasks_own_mount(mount, volume):
    payload, changed = render(_described(mount=mount, volume=volume), "efs")
    env = _env_of(payload)
    assert env[BACKEND] == "efs"
    assert env[ROOT] == mount
    assert GUARD not in env
    assert env["KEEP_ME"] == "1"
    assert changed == sorted([BACKEND, ROOT])


@pytest.mark.parametrize("extra", [{"KEEP_ME": "1"}, {"A": "x", "Z": "y", "KEEP_ME": "2"}])
def test_render_keeps_every_other_setting_and_drops_read_only_fields(extra):
    described = _described(env=extra)
    payload, _ = render(described, "efs")
    env = _env_of(payload)
    for k, v in extra.items():
        assert env[k] == v
    for field in ("taskDefinitionArn", "revision", "status", "requiresAttributes",
                  "compatibilities", "registeredAt", "registeredBy"):
        assert field not in payload
    assert payload["cpu"] == "1024"
    assert payload["tags"] == described["tags"]
    secrets = payload["containerDefinitions"][0]["secrets"]
    assert [s["name"] for s in secrets] == ["DATA_DIR"]


@pytest.mark.parametrize("mount", ["/app/data", "/srv/store"])
def test_enforce_turns_the_guard_on_only_after_efs(mount):
    after_efs, _ = render(_described(mount=mount), "efs")
    payload, changed = render({"taskDefinition": after_efs}, "enforce")
    env = _env_of(payload)
    assert env[GUARD] == "1"
    assert env[BACKEND] == "efs"
    assert env[ROOT] == mount
    assert changed == [GUARD]


@pytest.mark.parametrize("env", [
    {"KEEP_ME": "1"},
    {BACKEND: "local"},
    {BACKEND: "efs", ROOT: "/somewhere/else"},
])
def test_enforce_is_refused_before_the_efs_stage_is_on_the_task(env):
    with pytest.raises(RolloutRefused) as exc:
        render(_described(env=env), "enforce")
    assert exc.value.code == "efs_stage_not_live"


@pytest.mark.parametrize("kwargs", [
    {"efs": False},
    {"read_only": True},
])
def test_no_writable_efs_mount_is_refused_not_pointed_at_container_disk(kwargs):
    with pytest.raises(RolloutRefused) as exc:
        render(_described(**kwargs), "efs")
    assert exc.value.code == "no_efs_mount"


@pytest.mark.parametrize("stage", ["efs", "enforce"])
def test_a_managed_setting_held_as_a_secret_is_refused(stage):
    with pytest.raises(RolloutRefused) as exc:
        render(_described(secrets=("DATA_DIR", GUARD)), stage)
    assert exc.value.code == "setting_is_a_secret"


@pytest.mark.parametrize("value", ["1", "true"])
def test_efs_stage_refuses_when_the_guard_is_already_set(value):
    with pytest.raises(RolloutRefused) as exc:
        render(_described(env={GUARD: value}), "efs")
    assert exc.value.code == "guard_already_set"


@pytest.mark.parametrize("env", [
    {BACKEND: "efs", ROOT: "/app/data", GUARD: "1", "KEEP_ME": "1"},
    {BACKEND: "efs", ROOT: "/app/data", "KEEP_ME": "1"},
])
def test_off_removes_exactly_the_three_managed_settings(env):
    payload, changed = render(_described(env=env), "off")
    out = _env_of(payload)
    assert out == {"KEEP_ME": "1"}
    assert changed == sorted(n for n in (BACKEND, ROOT, GUARD) if n in env)


@pytest.mark.parametrize("off_without_mount", [{"efs": False}, {"read_only": True}])
def test_off_does_not_need_an_efs_mount(off_without_mount):
    payload, _ = render(_described(env={BACKEND: "efs", "KEEP_ME": "1"}, **off_without_mount), "off")
    assert _env_of(payload) == {"KEEP_ME": "1"}


@pytest.mark.parametrize("paths", [("/a", "/b"), ("/x/1", "/x/2")])
def test_two_writable_efs_mounts_are_ambiguous(paths):
    described = _described(mount=paths[0])
    td = described["taskDefinition"]
    td["volumes"].append({"name": "second", "efsVolumeConfiguration": {"fileSystemId": "fs-1"}})
    td["containerDefinitions"][0]["mountPoints"].append(
        {"sourceVolume": "second", "containerPath": paths[1], "readOnly": False}
    )
    with pytest.raises(RolloutRefused) as exc:
        efs_mount_path(td)
    assert exc.value.code == "efs_mount_ambiguous"


@pytest.mark.parametrize("count", [0, 2])
def test_not_exactly_one_container_is_refused(count):
    described = _described()
    c = described["taskDefinition"]["containerDefinitions"][0]
    described["taskDefinition"]["containerDefinitions"] = [c] * count
    with pytest.raises(RolloutRefused) as exc:
        render(described, "off")
    assert exc.value.code == "container_count"


@pytest.mark.parametrize(("stage", "rc"), [("efs", 0), ("enforce", 2)])
def test_cli_writes_payload_or_refuses_with_exit_2(tmp_path, capsys, stage, rc):
    src = tmp_path / "described.json"
    src.write_text(json.dumps(_described(mount="/mnt/m")), encoding="utf-8")
    out = tmp_path / "register.json"
    assert main([str(src), "--stage", stage, "--out", str(out)]) == rc
    captured = capsys.readouterr()
    if rc == 0:
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert _env_of(payload)[ROOT] == "/mnt/m"
        assert "fam-a:12 stage=efs" in captured.out
        assert "arn:secret" not in captured.out
    else:
        assert not out.exists()
        assert "REFUSED efs_stage_not_live" in captured.err
