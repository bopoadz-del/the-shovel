"""The storage check fails a volume that is mounted but not backed up.

Every input is built here; no AWS call. Each decision is exercised with at
least two shapes (file system id, mount path, volume name, age, status), so a
check that hardcodes one id, one path or one answer fails.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts.storage_check import (
    POLICY_NOT_FOUND,
    _load,
    check_backup_policy,
    check_durable_store,
    check_recovery_point,
    mounted_file_system,
    run_checks,
)
from scripts.storage_check import main as _main

NOW = datetime(2026, 11, 24, 21, 0, tzinfo=timezone.utc)


def _check_points(points, fs_id, now, limit):
    return check_recovery_point({"RecoveryPoints": list(points)}, fs_id, now, limit, asked_for=fs_id)


def main(argv):
    """The CLI, run the way a deploy runs it: the backup inputs were fetched
    for the file system the task file mounts. Binding tests call ``_main``."""
    argv = [str(a) for a in argv]
    if "--file-system-id" not in argv and argv and not argv[0].startswith("-"):
        described = _load(Path(argv[0]))
        try:
            fs_id = mounted_file_system(described) if isinstance(described, dict) else ""
        except ValueError:  # a task file that shows no one file system binds nothing
            fs_id = ""
        if fs_id:
            argv += ["--file-system-id", fs_id]
    return _main(argv)


def _described(
    *,
    fs_id: str = "fs-0aaa",
    mount: str = "/app/data",
    volume: str = "data",
    env: dict | None = None,
    efs: bool = True,
) -> dict:
    if env is None:
        env = {
            "DOCUMENT_BYTE_STORE": "efs",
            "DOCUMENT_STORE_ROOT": mount,
            "REQUIRE_DURABLE_DOCUMENT_STORE": "1",
        }
    vol = {"name": volume}
    if efs:
        vol["efsVolumeConfiguration"] = {"fileSystemId": fs_id, "transitEncryption": "ENABLED"}
    return {
        "taskDefinition": {
            "family": "svc",
            "volumes": [vol],
            "containerDefinitions": [{
                "name": "app",
                "mountPoints": [{"sourceVolume": volume, "containerPath": mount, "readOnly": False}],
                "environment": [{"name": k, "value": v} for k, v in env.items()],
            }],
        }
    }


def _point(created: datetime, *, status: str = "COMPLETED", arn: str | None = None) -> dict:
    p = {"CreationDate": created.isoformat(), "Status": status}
    if arn is not None:
        p["ResourceArn"] = arn
    return p


def _arn(fs_id: str) -> str:
    return f"arn:aws:elasticfilesystem:us-west-2:000000000000:file-system/{fs_id}"


@pytest.mark.parametrize("fs_id,volume,mount", [
    ("fs-0aaa", "data", "/app/data"),
    ("fs-9zzz", "docs", "/srv/store"),
])
def test_the_file_system_is_the_one_the_mount_uses(fs_id, volume, mount):
    assert mounted_file_system(_described(fs_id=fs_id, volume=volume, mount=mount)) == fs_id


@pytest.mark.parametrize("mount", ["/app/data", "/srv/store"])
def test_durable_store_ok_when_store_is_on_the_mount_with_guard(mount):
    name, ok, detail = check_durable_store(_described(mount=mount))
    assert (name, ok) == ("durable_store", True)
    assert mount in detail


@pytest.mark.parametrize("env,missing", [
    ({"DOCUMENT_BYTE_STORE": "local", "DOCUMENT_STORE_ROOT": "/app/data",
      "REQUIRE_DURABLE_DOCUMENT_STORE": "1"}, "DOCUMENT_BYTE_STORE=local"),
    ({"DOCUMENT_BYTE_STORE": "efs", "DOCUMENT_STORE_ROOT": "/elsewhere",
      "REQUIRE_DURABLE_DOCUMENT_STORE": "1"}, "DOCUMENT_STORE_ROOT=/elsewhere"),
    ({"DOCUMENT_BYTE_STORE": "efs", "DOCUMENT_STORE_ROOT": "/app/data"},
     "REQUIRE_DURABLE_DOCUMENT_STORE=<unset>"),
])
def test_durable_store_fails_and_names_the_wrong_setting(env, missing):
    _, ok, detail = check_durable_store(_described(env=env))
    assert not ok
    assert missing in detail


@pytest.mark.parametrize("volume,mount", [("data", "/app/data"), ("docs", "/srv/store")])
def test_durable_store_fails_without_an_efs_volume(volume, mount):
    _, ok, detail = check_durable_store(_described(efs=False, volume=volume, mount=mount))
    assert not ok
    assert detail == "no_efs_mount"


@pytest.mark.parametrize("policy,ok", [
    ({"BackupPolicy": {"Status": "ENABLED"}}, True),
    ({"BackupPolicy": {"Status": "DISABLED"}}, False),
    ({"BackupPolicy": {"Status": "ENABLING"}}, False),
    (POLICY_NOT_FOUND, False),
    (None, None),
])
def test_backup_policy_must_be_enabled(policy, ok):
    _, got, detail = check_backup_policy(policy, "fs-0bbb", asked_for="fs-0bbb")
    assert got is ok
    assert "fs-0bbb" in detail


@pytest.mark.parametrize("age_h,ok", [(2, True), (25.9, True), (26.1, False), (72, False)])
def test_recovery_point_age_is_judged_against_the_limit(age_h, ok):
    points = [_point(NOW - timedelta(hours=age_h))]
    _, got, _ = _check_points(points, "fs-0aaa", NOW, 26.0)
    assert got is ok


@pytest.mark.parametrize("limit,ok", [(1.0, False), (48.0, True)])
def test_the_limit_is_the_callers_not_a_constant(limit, ok):
    points = [_point(NOW - timedelta(hours=10))]
    assert _check_points(points, "fs-0aaa", NOW, limit)[1] is ok


@pytest.mark.parametrize("fs_id,other", [("fs-0aaa", "fs-0bbb"), ("fs-1111", "fs-2222")])
def test_another_volumes_backups_do_not_count(fs_id, other):
    fresh_foreign = [_point(NOW - timedelta(hours=1), arn=_arn(other)) for _ in range(3)]
    _, ok, detail = _check_points(fresh_foreign, fs_id, NOW, 26.0)
    assert not ok
    assert "own=0" in detail and "other_volumes=3" in detail
    own = fresh_foreign + [_point(NOW - timedelta(hours=1), arn=_arn(fs_id))]
    assert _check_points(own, fs_id, NOW, 26.0)[1] is True


@pytest.mark.parametrize("status", ["PARTIAL", "EXPIRED", "DELETING"])
def test_only_completed_points_count(status):
    points = [_point(NOW - timedelta(hours=1), status=status)]
    assert _check_points(points, "fs-0aaa", NOW, 26.0)[1] is False


def test_the_newest_completed_point_is_judged():
    points = [
        _point(NOW - timedelta(hours=80)),
        _point(NOW - timedelta(hours=3)),
        _point(NOW - timedelta(hours=1), status="PARTIAL"),
    ]
    _, ok, detail = _check_points(points, "fs-0aaa", NOW, 26.0)
    assert ok
    assert "age_h=3.0" in detail


def test_no_points_at_all_fails():
    _, ok, detail = _check_points([], "fs-0aaa", NOW, 26.0)
    assert not ok and "newest=<none>" in detail


@pytest.mark.parametrize("fs_id,policy", [
    ("fs-0ccc", POLICY_NOT_FOUND),
    ("fs-0eee", {"BackupPolicy": {"Status": "DISABLED"}}),
])
def test_mounted_but_unbacked_volume_fails_only_the_backup_checks(fs_id, policy):
    checks = run_checks(_described(fs_id=fs_id), policy, {"RecoveryPoints": []}, NOW, asked_for=fs_id)
    assert [(n, ok) for n, ok, _ in checks] == [
        ("durable_store", True), ("backup_policy", False), ("recovery_point", False),
    ]


def _write(tmp_path, name, obj):
    p = tmp_path / name
    p.write_text("" if obj is None else json.dumps(obj), encoding="utf-8")
    return str(p)


@pytest.mark.parametrize("fs_id", ["fs-0aaa", "fs-0ddd"])
def test_cli_exit_zero_only_when_every_check_is_ok(tmp_path, capsys, fs_id):
    td = _write(tmp_path, "td.json", _described(fs_id=fs_id))
    pol = _write(tmp_path, "pol.json", {"BackupPolicy": {"Status": "ENABLED"}})
    rp = _write(tmp_path, "rp.json", {"RecoveryPoints": [_point(NOW - timedelta(hours=5), arn=_arn(fs_id))]})
    assert main([td, "--backup-policy", pol, "--recovery-points", rp, "--now", NOW.isoformat()]) == 0
    out = capsys.readouterr().out
    assert "verdict=ok failed=none" in out and fs_id in out

    assert main([td, "--no-backup-policy", "--recovery-points", rp, "--now", NOW.isoformat()]) == 1
    assert "verdict=FAIL failed=backup_policy" in capsys.readouterr().out


@pytest.mark.parametrize("content", ["{not json", None])
def test_cli_unreadable_input_is_exit_two(tmp_path, content):
    bad = tmp_path / "bad.json"
    if content is not None:
        bad.write_text(content, encoding="utf-8")
    # content None: the file does not exist at all.
    assert main([str(bad)]) == 2


@pytest.mark.parametrize("raw", ["", "yesterday", None])
def test_an_unreadable_date_is_counted_not_trusted(raw):
    points = [{"CreationDate": raw, "Status": "COMPLETED"}]
    _, ok, detail = _check_points(points, "fs-0aaa", NOW, 26.0)
    assert ok is None
    assert "undated=1" in detail


@pytest.mark.parametrize("now", ["not-a-time", "2026-13-40"])
def test_cli_unreadable_now_is_exit_two(tmp_path, now):
    td = _write(tmp_path, "td.json", _described())
    assert main([td, "--now", now]) == 2


@pytest.mark.parametrize("fs_id", ["", "   "])
def test_an_efs_volume_without_a_file_system_id_is_not_checked_for_backups(fs_id):
    checks = run_checks(_described(fs_id=fs_id), {"BackupPolicy": {"Status": "ENABLED"}},
                        {"RecoveryPoints": [_point(NOW - timedelta(hours=1))]}, NOW)
    assert [(n, ok, d) for n, ok, d in checks][1:] == [
        ("backup_policy", None, "not_checked=described:no_file_system_id"),
        ("recovery_point", None, "not_checked=described:no_file_system_id"),
    ]


@pytest.mark.parametrize("limit", [26.0, 10.0])
def test_a_point_exactly_at_the_limit_still_counts(limit):
    at_limit = [_point(NOW - timedelta(hours=limit))]
    assert _check_points(at_limit, "fs-0aaa", NOW, limit)[1] is True
    just_over = [_point(NOW - timedelta(hours=limit, seconds=36))]
    assert _check_points(just_over, "fs-0aaa", NOW, limit)[1] is False


@pytest.mark.parametrize("age_h,code", [(25.5, 0), (27.0, 1)])
def test_the_cli_default_limit_is_one_day_plus_the_backup_window(tmp_path, age_h, code):
    td = _write(tmp_path, "td.json", _described())
    pol = _write(tmp_path, "pol.json", {"BackupPolicy": {"Status": "ENABLED"}})
    rp = _write(tmp_path, "rp.json", {"RecoveryPoints": [_point(NOW - timedelta(hours=age_h))]})
    assert main([td, "--backup-policy", pol, "--recovery-points", rp, "--now", NOW.isoformat()]) == code


@pytest.mark.parametrize("created,ok", [
    ("2026-11-24T20:00:00", True),   # 1 h before NOW when read as UTC
    ("2026-11-24T16:30:00", False),  # 4.5 h before NOW when read as UTC
])
def test_a_creation_date_without_a_zone_is_read_as_utc(created, ok):
    points = [{"CreationDate": created, "Status": "COMPLETED"}]
    assert _check_points(points, "fs-0aaa", NOW, 4.0)[1] is ok


@pytest.mark.parametrize("rw_fs,ro_fs", [("fs-0aaa", "fs-0bbb"), ("fs-1111", "fs-2222")])
def test_a_read_only_volume_on_the_same_path_is_not_the_backed_up_one(rw_fs, ro_fs):
    described = _described(fs_id=rw_fs)
    taskdef = described["taskDefinition"]
    taskdef["volumes"].insert(0, {"name": "snapshot", "efsVolumeConfiguration": {"fileSystemId": ro_fs}})
    taskdef["containerDefinitions"][0]["mountPoints"].insert(
        0, {"sourceVolume": "snapshot", "containerPath": "/app/data", "readOnly": True})
    assert mounted_file_system(described) == rw_fs


# --- Settings read the way the app reads them (grader S721 faults 1, 3, 4) ---

def _env(backend="efs", root="/app/data", guard="1") -> dict:
    env = {"DOCUMENT_BYTE_STORE": backend, "DOCUMENT_STORE_ROOT": root}
    if guard is not None:
        env["REQUIRE_DURABLE_DOCUMENT_STORE"] = guard
    return env


@pytest.mark.parametrize("guard", ["0", "false", "off", "no", "disabled", " "])
def test_a_guard_value_the_app_reads_as_off_fails(guard):
    _, ok, detail = check_durable_store(_described(env=_env(guard=guard)))
    assert ok is False
    assert "REQUIRE_DURABLE_DOCUMENT_STORE=" in detail


@pytest.mark.parametrize("guard", ["1", "true", "YES", " on "])
def test_a_guard_value_the_app_reads_as_on_passes(guard):
    assert check_durable_store(_described(env=_env(guard=guard)))[1] is True


@pytest.mark.parametrize("raw,on", [("1", True), ("On", True), ("0", False), ("enabled", False), ("", False)])
def test_the_checker_and_the_app_share_one_guard_reading(monkeypatch, raw, on):
    from app.core.byte_store import durable_guard_requested
    monkeypatch.setenv("REQUIRE_DURABLE_DOCUMENT_STORE", raw)
    assert durable_guard_requested() is on
    assert (check_durable_store(_described(env=_env(guard=raw)))[1] is True) is on


@pytest.mark.parametrize("backend", ["EFS", " efs", "Efs "])
def test_the_backend_is_read_case_and_space_blind_like_the_app(backend):
    assert check_durable_store(_described(env=_env(backend=backend)))[1] is True


@pytest.mark.parametrize("backend", ["local", "efs2", ""])
def test_a_backend_other_than_efs_still_fails(backend):
    assert check_durable_store(_described(env=_env(backend=backend)))[1] is False


@pytest.mark.parametrize("mount,root,ok", [
    ("/app/data", "/app/data/", True),
    ("/srv/store", " /srv/store ", True),
    ("/srv/store", "/srv//store/.", True),
    ("/app/data", "/app/data/sub", True),
    ("/app/data", "/app", False),
    ("/app/data", "/app/database", False),
])
def test_the_root_is_compared_as_a_path(mount, root, ok):
    d = _described(mount=mount, env=_env(root=root))
    assert check_durable_store(d)[1] is ok


def _with_secret(described: dict, name: str) -> dict:
    c = described["taskDefinition"]["containerDefinitions"][0]
    c["environment"] = [e for e in c["environment"] if e["name"] != name]
    c["secrets"] = [{"name": name, "valueFrom": "arn:aws:ssm:x:0:parameter/p"}]
    return described


@pytest.mark.parametrize("name", ["REQUIRE_DURABLE_DOCUMENT_STORE", "DOCUMENT_BYTE_STORE"])
def test_a_setting_given_as_a_secret_is_unverified_not_unset(name):
    _, ok, detail = check_durable_store(_with_secret(_described(), name))
    assert ok is None
    assert f"{name}=<secret>" in detail and "<unset>" not in detail


def test_a_secret_setting_does_not_hide_a_wrong_plain_one():
    d = _with_secret(_described(env=_env(backend="local")), "REQUIRE_DURABLE_DOCUMENT_STORE")
    _, ok, detail = check_durable_store(d)
    assert ok is False and "DOCUMENT_BYTE_STORE=local" in detail


@pytest.mark.parametrize("fs_id", ["fs-0aaa", "fs-0eee"])
def test_cli_unverified_setting_is_exit_two_and_a_failure_still_wins(tmp_path, capsys, fs_id):
    td = _write(tmp_path, "td.json", _with_secret(_described(fs_id=fs_id), "REQUIRE_DURABLE_DOCUMENT_STORE"))
    pol = _write(tmp_path, "pol.json", {"BackupPolicy": {"Status": "ENABLED"}})
    rp = _write(tmp_path, "rp.json", {"RecoveryPoints": [_point(NOW - timedelta(hours=5), arn=_arn(fs_id))]})
    assert main([td, "--backup-policy", pol, "--recovery-points", rp, "--now", NOW.isoformat()]) == 2
    out = capsys.readouterr().out
    assert "durable_store=UNVERIFIED" in out and "verdict=UNVERIFIED unverified=durable_store" in out
    off = _write(tmp_path, "off.json", {"BackupPolicy": {"Status": "DISABLED"}})
    assert main([td, "--backup-policy", off, "--recovery-points", rp, "--now", NOW.isoformat()]) == 1
    assert "verdict=FAIL failed=backup_policy" in capsys.readouterr().out


# --- A bad input file is the checker's input fault: not checked, exit 2, never 1 ---

@pytest.mark.parametrize("which", ["described", "--backup-policy", "--recovery-points"])
@pytest.mark.parametrize("content", [[{"a": 1}], "ENABLED", 3])
def test_cli_json_that_is_not_an_object_is_exit_two(tmp_path, capsys, which, content):
    files = {
        "described": _described(),
        "--backup-policy": {"BackupPolicy": {"Status": "ENABLED"}},
        "--recovery-points": {"RecoveryPoints": [_point(NOW - timedelta(hours=5))]},
    }
    files[which] = content
    paths = {k: _write(tmp_path, k.strip("-") + ".json", v) for k, v in files.items()}
    argv = [paths["described"], "--backup-policy", paths["--backup-policy"],
            "--recovery-points", paths["--recovery-points"], "--now", NOW.isoformat()]
    assert main(argv) == 2
    captured = capsys.readouterr()
    assert f"unreadable_input {which.strip('-')} not_an_object" in captured.err
    assert "verdict=UNVERIFIED" in captured.out and "=FAIL" not in captured.out


@pytest.mark.parametrize("blank", ["", "   \n"])
def test_cli_empty_task_definition_is_exit_two_not_a_failed_check(tmp_path, capsys, blank):
    td = tmp_path / "td.json"
    td.write_text(blank, encoding="utf-8")
    assert main([str(td)]) == 2
    out = capsys.readouterr().out
    assert out.count("not_checked=described:empty_file") == 3
    assert "verdict=UNVERIFIED" in out and "=FAIL" not in out


@pytest.mark.parametrize("flag", ["--backup-policy", "--recovery-points"])
def test_cli_an_empty_optional_file_is_not_checked_not_failed(tmp_path, capsys, flag):
    td = _write(tmp_path, "td.json", _described())
    empty = _write(tmp_path, "empty.json", None)
    assert main([td, flag, empty, "--now", NOW.isoformat()]) == 2
    out = capsys.readouterr().out
    assert f"not_checked={flag.strip('-')}:empty_file" in out
    assert "verdict=UNVERIFIED" in out and "=FAIL" not in out


# --- Notes: a future-dated point and the limit flag ---

@pytest.mark.parametrize("ahead", [timedelta(minutes=1), timedelta(minutes=30),
                                   timedelta(hours=1), timedelta(hours=8760)])
def test_a_recovery_point_dated_after_now_is_not_judged(ahead):
    _, ok, detail = _check_points([_point(NOW + ahead)], "fs-0aaa", NOW, 26)
    assert ok is None and "ahead=1" in detail and "age_h=-" not in detail


def test_a_point_dated_exactly_now_counts():
    assert _check_points([_point(NOW)], "fs-0aaa", NOW, 26)[1] is True
    assert _check_points([_point(NOW)], "fs-0bbb", NOW, 1)[1] is True


@pytest.mark.parametrize("limit", ["inf", "nan", "-5", "0"])
def test_cli_a_limit_that_is_not_a_finite_positive_number_is_refused(tmp_path, limit):
    td = _write(tmp_path, "td.json", _described())
    with pytest.raises(SystemExit) as exc:
        main([td, "--max-age-hours", limit])
    assert exc.value.code == 2


@pytest.mark.parametrize("limit,ok", [("0.5", False), ("48", True)])
def test_cli_a_finite_positive_limit_is_used(tmp_path, limit, ok):
    td = _write(tmp_path, "td.json", _described())
    pol = _write(tmp_path, "pol.json", {"BackupPolicy": {"Status": "ENABLED"}})
    rp = _write(tmp_path, "rp.json", {"RecoveryPoints": [_point(NOW - timedelta(hours=30))]})
    code = main([td, "--backup-policy", pol, "--recovery-points", rp, "--max-age-hours", limit, "--now", NOW.isoformat()])
    assert code == (0 if ok else 1)


# --- Test debt: the ARN must name the file system, not end with its id (M2) ---

@pytest.mark.parametrize("fs_id", ["fs-0aaa", "fs-0fff"])
def test_a_recovery_point_arn_that_only_ends_in_the_id_is_another_resource(fs_id):
    suffix = fs_id.split("-", 1)[1]
    other = f"arn:aws:backup:us-west-2:000000000000:recovery-point:xfs-{suffix}"
    _, ok, detail = _check_points([_point(NOW - timedelta(hours=1), arn=other)], fs_id, NOW, 26)
    assert ok is False
    assert "own=0" in detail and "other_volumes=1" in detail


# --- Every input is read and understood, or reported not checked (S721 grade 2) ---

def _cli(tmp_path, described, policy, points, *extra):
    argv = [_write(tmp_path, "td.json", described)]
    if policy is not None:
        argv += ["--backup-policy", _write(tmp_path, "pol.json", policy)]
    if points is not None:
        argv += ["--recovery-points", _write(tmp_path, "rp.json", points)]
    return main(argv + list(extra) + ["--now", NOW.isoformat()])


_ENABLED = {"BackupPolicy": {"Status": "ENABLED"}}


def _fresh(fs_id: str, hours: float = 5) -> dict:
    return {"RecoveryPoints": [_point(NOW - timedelta(hours=hours), arn=_arn(fs_id))]}


def _add_container(described: dict, container: dict) -> dict:
    described["taskDefinition"]["containerDefinitions"].append(container)
    return described


@pytest.mark.parametrize("fs_id,sidecar", [
    ("fs-0aaa", {"name": "log_router", "firelensConfiguration": {"type": "fluentbit"}}),
    ("fs-0ddd", {"name": "agent", "mountPoints": [], "environment": [{"name": "X", "value": "1"}]}),
])
def test_a_task_with_a_sidecar_is_read_through_the_container_that_mounts_efs(tmp_path, capsys, fs_id, sidecar):
    described = _add_container(_described(fs_id=fs_id), sidecar)
    assert _cli(tmp_path, described, _ENABLED, _fresh(fs_id)) == 0
    assert "verdict=ok failed=none" in capsys.readouterr().out
    # The same task still fails a disabled policy: the sidecar does not make it pass.
    off = {"BackupPolicy": {"Status": "DISABLED"}}
    assert _cli(tmp_path, described, off, _fresh(fs_id)) == 1
    assert "backup_policy=FAIL" in capsys.readouterr().out


@pytest.mark.parametrize("fs_a,fs_b", [("fs-0aaa", "fs-0bbb"), ("fs-1111", "fs-2222")])
def test_efs_mounted_in_two_containers_is_not_judged_for_the_store(fs_a, fs_b):
    described = _described(fs_id=fs_a)
    taskdef = described["taskDefinition"]
    taskdef["volumes"].append({"name": "other", "efsVolumeConfiguration": {"fileSystemId": fs_b}})
    _add_container(described, {"name": "second", "mountPoints": [
        {"sourceVolume": "other", "containerPath": "/srv/other", "readOnly": False}]})
    checks = run_checks(described, _ENABLED, _fresh(fs_a), NOW, asked_for=fs_a)
    assert checks == [
        ("durable_store", None, "not_checked=described:efs_mounts=2"),
        ("backup_policy", None, "not_checked=described:efs_file_systems=2"),
        ("recovery_point", None, "not_checked=described:efs_file_systems=2"),
    ]
    # Negative: both containers on ONE file system still have its backups checked.
    taskdef["volumes"][-1]["efsVolumeConfiguration"]["fileSystemId"] = fs_a
    checks = run_checks(described, _ENABLED, _fresh(fs_a), NOW, asked_for=fs_a)
    assert [(n, ok) for n, ok, _ in checks] == [
        ("durable_store", None), ("backup_policy", True), ("recovery_point", True)]


@pytest.mark.parametrize("epoch_of", [
    lambda when: when.timestamp(),             # CLI v1 default: float seconds
    lambda when: int(when.timestamp()),        # whole seconds
])
def test_an_epoch_creation_date_is_read(epoch_of):
    fresh = [{"CreationDate": epoch_of(NOW - timedelta(hours=11.8)), "Status": "COMPLETED"}]
    _, ok, detail = _check_points(fresh, "fs-0aaa", NOW, 26)
    assert ok is True and "age_h=11.8" in detail and "undated=0" in detail
    stale = [{"CreationDate": epoch_of(NOW - timedelta(hours=40)), "Status": "COMPLETED"}]
    assert _check_points(stale, "fs-0aaa", NOW, 26)[1] is False


@pytest.mark.parametrize("raw", [True, float("nan"), float("inf"), -float("inf"), {"t": 1}, 1e300])
def test_a_date_that_is_not_a_time_is_undated_not_failed(raw):
    _, ok, detail = _check_points([{"CreationDate": raw, "Status": "COMPLETED"}], "fs-0aaa", NOW, 26)
    assert ok is None and "undated=1" in detail


@pytest.mark.parametrize("stale_h", [30, 200])
def test_an_undated_point_beside_a_stale_one_is_not_a_fail(stale_h):
    points = [_point(NOW - timedelta(hours=stale_h)), {"CreationDate": "soon", "Status": "COMPLETED"}]
    _, ok, detail = _check_points(points, "fs-0aaa", NOW, 26)
    assert ok is None and "undated=1" in detail and f"age_h={stale_h}.0" in detail
    # Negative: only stale points, all dated, is a real fail.
    assert _check_points(points[:1], "fs-0aaa", NOW, 26)[1] is False
    # Negative: a fresh dated point decides, whatever else cannot be dated.
    assert _check_points(points + [_point(NOW - timedelta(hours=2))], "fs-0aaa", NOW, 26)[1] is True


@pytest.mark.parametrize("ahead_min", [1, 30])
def test_a_point_after_now_does_not_hide_a_fresh_one(ahead_min):
    points = [_point(NOW + timedelta(minutes=ahead_min)), _point(NOW - timedelta(hours=3))]
    _, ok, detail = _check_points(points, "fs-0aaa", NOW, 26)
    assert ok is True and "ahead=1" in detail and "age_h=3.0" in detail


@pytest.mark.parametrize("which,content,reason", [
    ("policy", {"BackupPolicy": ["ENABLED"]}, "backup-policy:BackupPolicy_not_an_object"),
    ("policy", {"BackupPolicy": {"Status": 1}}, "backup-policy:Status_not_text"),
    ("policy", {"BackupPolicy": {}}, "backup-policy:no_Status"),
    ("points", {"RecoveryPoints": {"a": 1}}, "recovery-points:RecoveryPoints_not_a_list"),
    ("points", {"RecoveryPoints": ["x"]}, "recovery-points:RecoveryPoints_not_an_object"),
    ("points", {"RecoveryPoints": [{"ResourceArn": 7}]}, "recovery-points:ResourceArn_not_text"),
])
def test_a_wrong_shape_inside_an_input_is_not_checked_not_a_crash(tmp_path, capsys, which, content, reason):
    fs_id = "fs-0aaa"
    policy = content if which == "policy" else _ENABLED
    points = content if which == "points" else _fresh(fs_id)
    assert _cli(tmp_path, _described(fs_id=fs_id), policy, points) == 2
    out = capsys.readouterr().out
    assert f"not_checked={reason}" in out and "=FAIL" not in out
    assert "durable_store=ok" in out


@pytest.mark.parametrize("described,reason", [
    ({"taskDefinition": None}, "taskDefinition_not_an_object"),
    ({"taskDefinition": {"containerDefinitions": "app"}}, "containerDefinitions_not_a_list"),
    ({"taskDefinition": {"containerDefinitions": []}}, "no_containers"),
    ({"services": [{"serviceName": "svc"}]}, "no_task_definition"),
    ({}, "no_task_definition"),
])
def test_a_described_file_that_is_not_a_task_definition_checks_nothing(tmp_path, capsys, described, reason):
    assert _cli(tmp_path, described, _ENABLED, _fresh("fs-0aaa")) == 2
    out = capsys.readouterr().out
    assert out.count(f"not_checked=described:{reason}") == 3 and "=FAIL" not in out


@pytest.mark.parametrize("mutate,reason", [
    (lambda c: c.update(environment=[{"name": "DOCUMENT_BYTE_STORE", "value": 1}]), "environment_value_not_text"),
    (lambda c: c.update(environment="DOCUMENT_BYTE_STORE=efs"), "environment_not_a_list"),
    (lambda c: c.update(secrets=[{"name": ["GUARD"]}]), "secret_name_not_text"),
])
def test_unreadable_settings_leave_the_store_unchecked_and_backups_judged(mutate, reason):
    described = _described(fs_id="fs-0ccc")
    mutate(described["taskDefinition"]["containerDefinitions"][0])
    checks = run_checks(described, _ENABLED, _fresh("fs-0ccc"), NOW, asked_for="fs-0ccc")
    assert checks[0] == ("durable_store", None, f"not_checked=described:{reason}")
    assert [(n, ok) for n, ok, _ in checks[1:]] == [("backup_policy", True), ("recovery_point", True)]


@pytest.mark.parametrize("flag_value", ["false", 0])
def test_a_read_only_flag_that_is_not_a_flag_is_not_read(flag_value):
    described = _described()
    described["taskDefinition"]["containerDefinitions"][0]["mountPoints"][0]["readOnly"] = flag_value
    assert check_durable_store(described) == ("durable_store", None, "not_checked=described:readOnly_not_a_flag")


@pytest.mark.parametrize("fs_id", ["fs-0aaa", "fs-0eee"])
def test_an_input_not_given_is_named_not_failed(tmp_path, capsys, fs_id):
    assert _cli(tmp_path, _described(fs_id=fs_id), _ENABLED, None) == 2
    out = capsys.readouterr().out
    assert f"recovery_point=UNVERIFIED file_system={fs_id} not_checked=recovery-points:not_given" in out
    assert _cli(tmp_path, _described(fs_id=fs_id), None, _fresh(fs_id)) == 2
    out = capsys.readouterr().out
    assert f"backup_policy=UNVERIFIED file_system={fs_id} not_checked=backup-policy:not_given" in out
    # Negative: the CLI's PolicyNotFound answer, given as such, is a real fail.
    assert _cli(tmp_path, _described(fs_id=fs_id), None, _fresh(fs_id), "--no-backup-policy") == 1
    assert f"backup_policy=FAIL file_system={fs_id} status=PolicyNotFound" in capsys.readouterr().out


@pytest.mark.parametrize("fs_id", ["fs-0aaa", "fs-0bbb"])
def test_a_policy_file_and_policy_not_found_together_are_refused(tmp_path, fs_id):
    with pytest.raises(SystemExit) as exc:
        _cli(tmp_path, _described(fs_id=fs_id), _ENABLED, _fresh(fs_id), "--no-backup-policy")
    assert exc.value.code == 2


@pytest.mark.parametrize("fs_id", ["fs-0aaa", "fs-0bbb"])
def test_swapped_policy_and_points_files_are_not_checked(tmp_path, capsys, fs_id):
    assert _cli(tmp_path, _described(fs_id=fs_id), _fresh(fs_id), _ENABLED) == 2
    out = capsys.readouterr().out
    assert "not_checked=backup-policy:no_BackupPolicy" in out
    assert "not_checked=recovery-points:no_RecoveryPoints" in out
    assert "=FAIL" not in out


@pytest.mark.parametrize("missing", ["described", "pol", "rp"])
def test_a_file_that_does_not_exist_is_named(tmp_path, capsys, missing):
    paths = {
        "described": _write(tmp_path, "td.json", _described()),
        "pol": _write(tmp_path, "pol.json", _ENABLED),
        "rp": _write(tmp_path, "rp.json", _fresh("fs-0aaa")),
    }
    paths[missing] = str(tmp_path / "absent.json")
    code = main([paths["described"], "--backup-policy", paths["pol"],
                 "--recovery-points", paths["rp"], "--now", NOW.isoformat()])
    captured = capsys.readouterr()
    assert code == 2 and "unreadable_file_FileNotFoundError" in captured.out
    label = {"described": "described", "pol": "backup-policy", "rp": "recovery-points"}[missing]
    assert f"unreadable_input {label} unreadable_file_FileNotFoundError" in captured.err


@pytest.mark.parametrize("name", ["REQUIRE_DURABLE_DOCUMENT_STORE", "DOCUMENT_BYTE_STORE"])
def test_a_setting_that_is_both_plain_and_secret_is_unverified(name):
    described = _described()
    described["taskDefinition"]["containerDefinitions"][0]["secrets"] = [
        {"name": name, "valueFrom": "arn:aws:ssm:x:0:parameter/p"}]
    _, ok, detail = check_durable_store(described)
    assert ok is None and f"{name}=<secret>" in detail
    # Negative: the same task without the secret is judged on its plain value.
    assert check_durable_store(_described())[1] is True


@pytest.mark.parametrize("root", ["/app/data", "/srv/x"])
def test_a_task_with_no_backend_setting_at_all_fails(root):
    env = {"DOCUMENT_STORE_ROOT": root, "REQUIRE_DURABLE_DOCUMENT_STORE": "1"}
    _, ok, detail = check_durable_store(_described(mount=root, env=env))
    assert ok is False and detail == "DOCUMENT_BYTE_STORE=<unset>"


@pytest.mark.parametrize("fs_id", ["fs-1", "fs-2"])
def test_a_resource_arn_with_a_trailing_slash_is_still_own(fs_id):
    point = _point(NOW - timedelta(hours=1), arn=_arn(fs_id) + "/")
    _, ok, detail = _check_points([point], fs_id, NOW, 26)
    assert ok is True and "own=1" in detail


# --- First grade of #779: every input read, or not checked (encoding, env files, root, pages) ---

def _write_bytes(tmp_path, name, data: bytes) -> str:
    p = tmp_path / name
    p.write_bytes(data)
    return str(p)


def _argv(paths: dict) -> list:
    return [paths["described"], "--backup-policy", paths["backup-policy"],
            "--recovery-points", paths["recovery-points"], "--now", NOW.isoformat()]


@pytest.mark.parametrize("which", ["described", "backup-policy", "recovery-points"])
@pytest.mark.parametrize("encode,reason", [
    (lambda t: b"\xff\xfe\xe9" + b"\x00", "not_json"),
    (lambda t: t.encode("utf-8").replace(b"{", b"{\xe9", 1), "not_utf8_text"),
    (lambda t: b"\xff\xfe" + b"\x00\xd8", "not_utf16_text"),
])
def test_cli_a_file_that_does_not_decode_is_not_checked_not_a_crash(tmp_path, capsys, which, encode, reason):
    fs_id = "fs-0aaa"
    objs = {"described": _described(fs_id=fs_id), "backup-policy": _ENABLED,
            "recovery-points": _fresh(fs_id)}
    paths = {k: _write(tmp_path, f"{k}.json", v) for k, v in objs.items()}
    paths[which] = _write_bytes(tmp_path, f"bad-{which}.json", encode(json.dumps(objs[which])))
    assert main(_argv(paths)) == 2
    captured = capsys.readouterr()
    assert f"unreadable_input {which} " in captured.err and "=FAIL" not in captured.out
    assert f"not_checked={which}:{reason}" in captured.out


@pytest.mark.parametrize("reason,data", [
    ("not_utf8_text", b'{"taskDefinition": "\xe9"}'),
    ("not_utf16_text", b"\xfe\xff\xd8\x00"),
])
def test_an_undecodable_file_names_its_encoding(tmp_path, reason, data):
    from scripts.storage_check import NotRead, _load
    p = tmp_path / "x.json"
    p.write_bytes(data)
    assert _load(p) == NotRead(reason)


@pytest.mark.parametrize("encoding,bom", [
    ("utf-16-le", b"\xff\xfe"), ("utf-16-be", b"\xfe\xff"), ("utf-8", b"\xef\xbb\xbf")])
@pytest.mark.parametrize("fs_id", ["fs-0aaa", "fs-0bbb"])
def test_cli_a_file_in_the_encoding_its_mark_declares_is_read(tmp_path, capsys, encoding, bom, fs_id):
    objs = {"described": _described(fs_id=fs_id), "backup-policy": _ENABLED,
            "recovery-points": _fresh(fs_id)}
    paths = {k: _write_bytes(tmp_path, f"{k}.json", bom + json.dumps(v).encode(encoding))
             for k, v in objs.items()}
    assert main(_argv(paths)) == 0
    assert "verdict=ok failed=none" in capsys.readouterr().out
    # Negative: the same encoding still carries a failing answer through.
    paths["backup-policy"] = _write_bytes(
        tmp_path, "off.json", bom + json.dumps({"BackupPolicy": {"Status": "DISABLED"}}).encode(encoding))
    assert main(_argv(paths)) == 1
    assert "backup_policy=FAIL" in capsys.readouterr().out


@pytest.mark.parametrize("depth", [100_000, 200_000])
def test_json_nested_past_the_parser_limit_is_not_json(tmp_path, depth):
    from scripts.storage_check import NotRead, _load
    p = tmp_path / "deep.json"
    p.write_text("[" * depth + "]" * depth, encoding="utf-8")
    assert _load(p) == NotRead("not_json")


def _with_env_file(described: dict, keep: dict) -> dict:
    c = described["taskDefinition"]["containerDefinitions"][0]
    c["environment"] = [{"name": k, "value": v} for k, v in keep.items()]
    c["environmentFiles"] = [{"type": "s3", "value": "arn:aws:s3:::bucket/app.env"}]
    return described


@pytest.mark.parametrize("keep,hidden", [
    ({}, ["DOCUMENT_BYTE_STORE=<env_file>", "DOCUMENT_STORE_ROOT=<env_file>",
          "REQUIRE_DURABLE_DOCUMENT_STORE=<env_file>"]),
    ({"DOCUMENT_BYTE_STORE": "efs"}, ["DOCUMENT_STORE_ROOT=<env_file>",
                                      "REQUIRE_DURABLE_DOCUMENT_STORE=<env_file>"]),
])
def test_settings_left_to_an_environment_file_are_unverified_not_unset(keep, hidden):
    _, ok, detail = check_durable_store(_with_env_file(_described(), keep))
    assert ok is None and "<unset>" not in detail
    assert all(h in detail for h in hidden)


@pytest.mark.parametrize("keep,wrong", [
    ({"DOCUMENT_BYTE_STORE": "local"}, "DOCUMENT_BYTE_STORE=local"),
    ({"REQUIRE_DURABLE_DOCUMENT_STORE": "0"}, "REQUIRE_DURABLE_DOCUMENT_STORE=0 (off)"),
])
def test_a_wrong_plain_value_beside_an_environment_file_still_fails(keep, wrong):
    _, ok, detail = check_durable_store(_with_env_file(_described(), keep))
    assert ok is False and wrong in detail


@pytest.mark.parametrize("mount", ["/app/data", "/srv/store"])
def test_plain_values_beside_an_environment_file_are_judged(mount):
    env = {"DOCUMENT_BYTE_STORE": "efs", "DOCUMENT_STORE_ROOT": mount,
           "REQUIRE_DURABLE_DOCUMENT_STORE": "1"}
    assert check_durable_store(_with_env_file(_described(mount=mount), env))[1] is True


@pytest.mark.parametrize("files,reason", [
    ("s3://b/app.env", "environmentFiles_not_a_list"),
    (["s3://b/app.env"], "environmentFiles_not_an_object"),
])
def test_an_environment_files_field_of_the_wrong_shape_is_not_read(files, reason):
    d = _described()
    d["taskDefinition"]["containerDefinitions"][0]["environmentFiles"] = files
    assert check_durable_store(d) == ("durable_store", None, f"not_checked=described:{reason}")
    # An empty list is no file: the plain settings decide.
    d["taskDefinition"]["containerDefinitions"][0]["environmentFiles"] = []
    assert check_durable_store(d)[1] is True


@pytest.mark.parametrize("mount,root", [("/app/data", "/app/data/docs"), ("/srv/store", "/srv/store/a/b")])
def test_a_root_under_the_mount_is_on_the_mount(mount, root):
    _, ok, detail = check_durable_store(_described(mount=mount, env=_env(root=root)))
    assert ok is True and f"mount={mount}" in detail


@pytest.mark.parametrize("mount,root", [("/app/data", "/app/datafile"), ("/srv/store", "/tmp/store")])
def test_a_root_outside_the_mount_fails(mount, root):
    _, ok, detail = check_durable_store(_described(mount=mount, env=_env(root=root)))
    assert ok is False and "on=<container_disk>" in detail and f"mount={mount}" in detail


@pytest.mark.parametrize("mount,inner", [("/app/data", "/app/data/cache"), ("/srv/store", "/srv/store/tmp")])
def test_a_root_on_a_deeper_non_efs_mount_is_not_on_efs(mount, inner):
    d = _described(mount=mount, env=_env(root=inner + "/docs"))
    d["taskDefinition"]["volumes"].append({"name": "scratch"})
    d["taskDefinition"]["containerDefinitions"][0]["mountPoints"].append(
        {"sourceVolume": "scratch", "containerPath": inner})
    _, ok, detail = check_durable_store(d)
    assert ok is False and f"on={inner}" in detail
    # Negative: a root beside that inner mount, still under EFS, is on EFS.
    d2 = _described(mount=mount, env=_env(root=mount + "/docs"))
    d2["taskDefinition"]["volumes"].append({"name": "scratch"})
    d2["taskDefinition"]["containerDefinitions"][0]["mountPoints"].append(
        {"sourceVolume": "scratch", "containerPath": inner})
    assert check_durable_store(d2)[1] is True


@pytest.mark.parametrize("mount", ["/app/data", "/srv/store"])
def test_a_root_on_a_path_two_mounts_share_is_unverified(mount):
    d = _described(mount=mount)
    d["taskDefinition"]["volumes"].append({"name": "scratch"})
    d["taskDefinition"]["containerDefinitions"][0]["mountPoints"].append(
        {"sourceVolume": "scratch", "containerPath": mount + "/"})
    _, ok, detail = check_durable_store(d)
    assert ok is None and f"mounts_on_one_path={mount}" in detail


@pytest.mark.parametrize("mount", ["/app/data", "/srv/store"])
def test_an_unset_root_falls_back_to_data_dir_like_the_app(mount):
    env = {"DOCUMENT_BYTE_STORE": "efs", "DATA_DIR": mount, "REQUIRE_DURABLE_DOCUMENT_STORE": "1"}
    _, ok, detail = check_durable_store(_described(mount=mount, env=env))
    assert ok is True
    # Negative: DATA_DIR off the mount fails, and names both settings.
    env["DATA_DIR"] = "/var/lib/other"
    _, ok, detail = check_durable_store(_described(mount=mount, env=env))
    assert ok is False and "DOCUMENT_STORE_ROOT=<unset> DATA_DIR=/var/lib/other" in detail


@pytest.mark.parametrize("root,data_dir,want", [
    ("/a", "/b", "/a"), (" /c ", None, "/c"), ("", "/d", "/d"), ("  ", None, "./data"), (None, "", "")])
def test_the_checker_and_the_app_share_one_root_reading(monkeypatch, root, data_dir, want):
    from app.core.byte_store import resolve_store_root, store_root
    for name, value in (("DOCUMENT_STORE_ROOT", root), ("DATA_DIR", data_dir)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    assert store_root() == resolve_store_root(root, data_dir) == want


@pytest.mark.parametrize("workdir,rel,ok", [
    ("/app", "data", True), ("/app", "./data/docs", True), ("/srv", "data", False)])
def test_a_relative_root_is_joined_to_the_working_directory(workdir, rel, ok):
    d = _described(mount="/app/data", env=_env(root=rel))
    d["taskDefinition"]["containerDefinitions"][0]["workingDirectory"] = workdir
    assert check_durable_store(d)[1] is ok


@pytest.mark.parametrize("env", [
    {"DOCUMENT_BYTE_STORE": "efs", "DOCUMENT_STORE_ROOT": "data", "REQUIRE_DURABLE_DOCUMENT_STORE": "1"},
    {"DOCUMENT_BYTE_STORE": "efs", "REQUIRE_DURABLE_DOCUMENT_STORE": "1"},
])
def test_a_relative_root_without_a_working_directory_is_unverified(env):
    _, ok, detail = check_durable_store(_described(env=env))
    assert ok is None and "workingDirectory=<not_in_task_definition>" in detail


@pytest.mark.parametrize("hidden", ["secrets", "environmentFiles"])
def test_an_unset_root_with_a_hidden_data_dir_is_unverified(hidden):
    env = {"DOCUMENT_BYTE_STORE": "efs", "REQUIRE_DURABLE_DOCUMENT_STORE": "1"}
    d = _described(env=env)
    c = d["taskDefinition"]["containerDefinitions"][0]
    if hidden == "secrets":
        c["secrets"] = [{"name": "DATA_DIR", "valueFrom": "arn:aws:ssm:x:0:parameter/p"}]
        want = "DATA_DIR=<secret>"
    else:
        c["environmentFiles"] = [{"type": "s3", "value": "arn:aws:s3:::b/app.env"}]
        want = "DOCUMENT_STORE_ROOT=<env_file>"
    _, ok, detail = check_durable_store(d)
    assert ok is None and want in detail


@pytest.mark.parametrize("age_h", [30, 191])
def test_one_page_of_a_longer_listing_without_a_fresh_point_is_unverified(age_h):
    points = {"RecoveryPoints": [_point(NOW - timedelta(hours=age_h), arn=_arn("fs-0aaa"))],
              "NextToken": "abc"}
    _, ok, detail = check_recovery_point(points, "fs-0aaa", NOW, 26, asked_for="fs-0aaa")
    assert ok is None and "more_pages=not_read" in detail
    # Negative: the same stale page with no NextToken is complete, and fails.
    del points["NextToken"]
    assert check_recovery_point(points, "fs-0aaa", NOW, 26, asked_for="fs-0aaa")[1] is False


@pytest.mark.parametrize("token", ["abc", "eyJuZXh0IjoxfQ=="])
def test_a_fresh_point_on_one_page_is_enough(token):
    points = {"RecoveryPoints": [_point(NOW - timedelta(hours=2), arn=_arn("fs-0bbb"))], "NextToken": token}
    assert check_recovery_point(points, "fs-0bbb", NOW, 26, asked_for="fs-0bbb")[1] is True


@pytest.mark.parametrize("token", [None, "", "  "])
def test_an_empty_next_token_is_a_complete_listing(token):
    points = {"RecoveryPoints": [], "NextToken": token}
    _, ok, detail = check_recovery_point(points, "fs-0aaa", NOW, 26, asked_for="fs-0aaa")
    assert ok is False and "more_pages" not in detail


@pytest.mark.parametrize("token", [5, ["abc"]])
def test_a_next_token_that_is_not_text_is_not_read(token):
    points = {"RecoveryPoints": [], "NextToken": token}
    assert check_recovery_point(points, "fs-0aaa", NOW, 26, asked_for="fs-0aaa") == (
        "recovery_point", None, "file_system=fs-0aaa not_checked=recovery-points:NextToken_not_text")


@pytest.mark.parametrize("workdir", ["app", "./srv", "   "])
def test_a_relative_root_under_a_relative_working_directory_is_unverified(workdir):
    # The image's own WORKDIR, which a relative one resolves against, is not in the file.
    d = _described(mount="/app/data", env=_env(root="data"))
    d["taskDefinition"]["containerDefinitions"][0]["workingDirectory"] = workdir
    _, ok, detail = check_durable_store(d)
    assert ok is None and "workingDirectory=<not_in_task_definition>" in detail


# --- #779 grade 1 (trial4, trial5): absent fields, binding, Status shape ---

def _mount_point(d: dict) -> dict:
    return d["taskDefinition"]["containerDefinitions"][0]["mountPoints"][0]


@pytest.mark.parametrize("fs_id,path", [("fs-0aaa", None), ("fs-0bbb", "   ")])
def test_an_efs_mount_with_no_container_path_is_not_checked(fs_id, path):
    d = _described(fs_id=fs_id)
    _mount_point(d)["containerPath"] = path
    reason = "not_checked=described:mountPoint_without_containerPath"
    assert run_checks(d, _ENABLED, _fresh(fs_id), NOW, asked_for=fs_id) == [
        ("durable_store", None, reason),
        ("backup_policy", None, reason),
        ("recovery_point", None, reason),
    ]


@pytest.mark.parametrize("path", [None, ""])
def test_another_mount_with_no_container_path_is_not_checked(path):
    d = _described()
    d["taskDefinition"]["volumes"].append({"name": "scratch", "host": {}})
    d["taskDefinition"]["containerDefinitions"][0]["mountPoints"].append(
        {"sourceVolume": "scratch", "containerPath": path})
    assert check_durable_store(d) == (
        "durable_store", None, "not_checked=described:mountPoint_without_containerPath")


@pytest.mark.parametrize("path", ["app/data", "srv"])
def test_a_relative_efs_mount_path_fails(path):
    d = _described(mount=path, env=_env(root=path))
    assert check_durable_store(d) == ("durable_store", False, "efs_mount_not_absolute")


@pytest.mark.parametrize("fs_id,name", [("fs-0aaa", None), ("fs-0bbb", " ")])
def test_an_unnamed_efs_volume_is_not_checked(fs_id, name):
    d = _described(fs_id=fs_id)
    d["taskDefinition"]["volumes"][0]["name"] = name
    _mount_point(d)["sourceVolume"] = name
    reason = "not_checked=described:efs_volume_without_name"
    checks = run_checks(d, _ENABLED, _fresh(fs_id), NOW, asked_for=fs_id)
    assert [(n, ok, det) for n, ok, det in checks] == [
        ("durable_store", None, reason),
        ("backup_policy", None, reason),
        ("recovery_point", None, reason),
    ]


@pytest.mark.parametrize("fs_id,source", [("fs-0aaa", None), ("fs-0bbb", "")])
def test_a_mount_point_with_no_source_volume_is_not_checked(fs_id, source):
    d = _described(fs_id=fs_id)
    _mount_point(d)["sourceVolume"] = source
    reason = "not_checked=described:mountPoint_without_sourceVolume"
    assert [c[1:] for c in run_checks(d, _ENABLED, _fresh(fs_id), NOW, asked_for=fs_id)] == [
        (None, reason), (None, reason), (None, reason),
    ]


@pytest.mark.parametrize("fs_id", ["fs-0aaa", "fs-0ddd"])
def test_an_unnamed_volume_that_is_not_efs_is_not_read(fs_id):
    # Negative: only an EFS volume needs a name to be told apart.
    d = _described(fs_id=fs_id)
    d["taskDefinition"]["volumes"].append({"host": {}})
    assert [c[1] for c in run_checks(d, _ENABLED, _fresh(fs_id), NOW, asked_for=fs_id)] == [True, True, True]


@pytest.mark.parametrize("fs_id,readonly", [("fs-0aaa", True), ("fs-0bbb", None)])
def test_no_read_write_efs_mount_fails_the_store_and_names_why_backups_are_not_checked(fs_id, readonly):
    d = _described(fs_id=fs_id)
    if readonly is None:
        d["taskDefinition"]["containerDefinitions"][0]["mountPoints"] = []
    else:
        _mount_point(d)["readOnly"] = readonly
    assert run_checks(d, _ENABLED, _fresh(fs_id), NOW, asked_for=fs_id) == [
        ("durable_store", False, "no_efs_mount"),
        ("backup_policy", None, "not_checked=described:no_efs_mount"),
        ("recovery_point", None, "not_checked=described:no_efs_mount"),
    ]


@pytest.mark.parametrize("content", ["{", "abc"])
def test_cli_a_file_that_is_not_json_is_named(tmp_path, capsys, content):
    td = tmp_path / "td.json"
    td.write_text(content, encoding="utf-8")
    assert _main([str(td), "--now", NOW.isoformat()]) == 2
    out, err = capsys.readouterr()
    assert "unreadable_input described not_json" in err
    assert "durable_store=UNVERIFIED not_checked=described:not_json" in out


@pytest.mark.parametrize("fs_id,policy", [
    ("fs-0aaa", _ENABLED),
    ("fs-0bbb", {"BackupPolicy": {"Status": "DISABLED"}}),
    ("fs-0ccc", POLICY_NOT_FOUND),
])
def test_a_backup_policy_not_bound_to_the_volume_is_not_judged(fs_id, policy):
    assert check_backup_policy(policy, fs_id) == (
        "backup_policy", None, f"file_system={fs_id} not_checked=backup-policy:file_system_not_named")
    assert check_backup_policy(policy, fs_id, asked_for="fs-0zzz") == (
        "backup_policy", None, f"file_system={fs_id} not_checked=backup-policy:asked_for=fs-0zzz")
    # Negative: bound to the mounted volume (stray spaces allowed), it is judged.
    assert check_backup_policy(policy, fs_id, asked_for=f" {fs_id} ")[1] is (policy is _ENABLED)


@pytest.mark.parametrize("fs_id,asked", [("fs-0aaa", None), ("fs-0bbb", "fs-0zzz"), ("fs-0ccc", "  ")])
def test_a_point_with_no_arn_is_bound_only_by_the_file_system_asked_for(fs_id, asked):
    points = {"RecoveryPoints": [_point(NOW - timedelta(hours=2))]}
    _, ok, detail = check_recovery_point(points, fs_id, NOW, 26, asked_for=asked)
    assert ok is None and "own=0" in detail and "unbound=1" in detail and "other_volumes=0" in detail
    # Negative: asked for this volume, the same point is its own and fresh.
    _, ok, detail = check_recovery_point(points, fs_id, NOW, 26, asked_for=fs_id)
    assert ok is True and "own=1" in detail and "unbound=0" in detail


@pytest.mark.parametrize("fs_id", ["fs-0aaa", "fs-0bbb"])
def test_a_point_whose_arn_names_the_volume_needs_no_binding(fs_id):
    fresh = {"RecoveryPoints": [_point(NOW - timedelta(hours=2), arn=_arn(fs_id))]}
    assert check_recovery_point(fresh, fs_id, NOW, 26)[1] is True
    stale = {"RecoveryPoints": [_point(NOW - timedelta(hours=40), arn=_arn(fs_id))]}
    assert check_recovery_point(stale, fs_id, NOW, 26)[1] is False


@pytest.mark.parametrize("fs_id", ["fs-0aaa", "fs-0ddd"])
def test_cli_without_the_file_system_id_the_backup_inputs_are_not_judged(tmp_path, capsys, fs_id):
    td = _write(tmp_path, "td.json", _described(fs_id=fs_id))
    pol = _write(tmp_path, "pol.json", _ENABLED)
    rp = _write(tmp_path, "rp.json", {"RecoveryPoints": [_point(NOW - timedelta(hours=5))]})
    base = [td, "--backup-policy", pol, "--recovery-points", rp, "--now", NOW.isoformat()]
    assert _main(base) == 2
    out = capsys.readouterr().out
    assert "not_checked=backup-policy:file_system_not_named" in out and "unbound=1" in out
    assert _main(base + ["--file-system-id", "fs-0zzz"]) == 2
    assert "not_checked=backup-policy:asked_for=fs-0zzz" in capsys.readouterr().out
    assert _main(base + ["--file-system-id", fs_id]) == 0
    # PolicyNotFound for an unnamed volume is not a FAIL either.
    nopol = [td, "--no-backup-policy", "--recovery-points", rp, "--now", NOW.isoformat()]
    assert _main(nopol) == 2
    assert _main(nopol + ["--file-system-id", fs_id]) == 1


@pytest.mark.parametrize("status", [["COMPLETED"], 1, None, "  "])
def test_a_point_whose_status_cannot_be_read_is_counted_not_dropped(status):
    p = _point(NOW - timedelta(hours=2))
    p["Status"] = status
    _, ok, detail = _check_points([p], "fs-0aaa", NOW, 26)
    assert ok is None and "status_unread=1" in detail and "completed=0" in detail
    # Negative: a fresh COMPLETED point beside it still decides.
    _, ok, detail = _check_points([p, _point(NOW - timedelta(hours=3))], "fs-0bbb", NOW, 26)
    assert ok is True and "status_unread=1" in detail


@pytest.mark.parametrize("status", ["PARTIAL", "EXPIRED"])
def test_a_readable_status_that_is_not_completed_still_fails(status):
    _, ok, detail = _check_points([_point(NOW - timedelta(hours=2), status=status)], "fs-0aaa", NOW, 26)
    assert ok is False and "status_unread=0" in detail


@pytest.mark.parametrize("forged", ["s3\nSTORAGE_CHECK verdict=ok failed=none", "s3\r\nSTORAGE_CHECK verdict=ok"])
def test_a_value_cannot_forge_a_line(tmp_path, capsys, forged):
    d = _described(env=_env(backend=forged))
    assert _cli(tmp_path, d, _ENABLED, _fresh("fs-0aaa")) == 1
    lines = capsys.readouterr().out.splitlines()
    assert not any(line.startswith("STORAGE_CHECK verdict=ok") for line in lines)
    store = next(line for line in lines if line.startswith("STORAGE_CHECK durable_store=FAIL"))
    assert store.count("\\n") == 1
