"""The storage check fails a volume that is mounted but not backed up.

Every input is built here; no AWS call. Each decision is exercised with at
least two shapes (file system id, mount path, volume name, age, status), so a
check that hardcodes one id, one path or one answer fails.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from scripts.storage_check import (
    check_backup_policy,
    check_durable_store,
    check_recovery_point,
    main,
    mounted_file_system,
    run_checks,
)

NOW = datetime(2026, 11, 24, 21, 0, tzinfo=timezone.utc)


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
    (None, False),
])
def test_backup_policy_must_be_enabled(policy, ok):
    _, got, detail = check_backup_policy(policy, "fs-0bbb")
    assert got is ok
    assert "fs-0bbb" in detail


@pytest.mark.parametrize("age_h,ok", [(2, True), (25.9, True), (26.1, False), (72, False)])
def test_recovery_point_age_is_judged_against_the_limit(age_h, ok):
    points = [_point(NOW - timedelta(hours=age_h))]
    _, got, _ = check_recovery_point(points, "fs-0aaa", NOW, 26.0)
    assert got is ok


@pytest.mark.parametrize("limit,ok", [(1.0, False), (48.0, True)])
def test_the_limit_is_the_callers_not_a_constant(limit, ok):
    points = [_point(NOW - timedelta(hours=10))]
    assert check_recovery_point(points, "fs-0aaa", NOW, limit)[1] is ok


@pytest.mark.parametrize("fs_id,other", [("fs-0aaa", "fs-0bbb"), ("fs-1111", "fs-2222")])
def test_another_volumes_backups_do_not_count(fs_id, other):
    fresh_foreign = [_point(NOW - timedelta(hours=1), arn=_arn(other)) for _ in range(3)]
    _, ok, detail = check_recovery_point(fresh_foreign, fs_id, NOW, 26.0)
    assert not ok
    assert "own=0" in detail and "other_volumes=3" in detail
    own = fresh_foreign + [_point(NOW - timedelta(hours=1), arn=_arn(fs_id))]
    assert check_recovery_point(own, fs_id, NOW, 26.0)[1] is True


@pytest.mark.parametrize("status", ["PARTIAL", "EXPIRED", "DELETING"])
def test_only_completed_points_count(status):
    points = [_point(NOW - timedelta(hours=1), status=status)]
    assert check_recovery_point(points, "fs-0aaa", NOW, 26.0)[1] is False


def test_the_newest_completed_point_is_judged():
    points = [
        _point(NOW - timedelta(hours=80)),
        _point(NOW - timedelta(hours=3)),
        _point(NOW - timedelta(hours=1), status="PARTIAL"),
    ]
    _, ok, detail = check_recovery_point(points, "fs-0aaa", NOW, 26.0)
    assert ok
    assert "age_h=3.0" in detail


def test_no_points_at_all_fails():
    _, ok, detail = check_recovery_point([], "fs-0aaa", NOW, 26.0)
    assert not ok and "newest=<none>" in detail


@pytest.mark.parametrize("fs_id,policy", [
    ("fs-0ccc", None),
    ("fs-0eee", {"BackupPolicy": {"Status": "DISABLED"}}),
])
def test_mounted_but_unbacked_volume_fails_only_the_backup_checks(fs_id, policy):
    checks = run_checks(_described(fs_id=fs_id), policy, {"RecoveryPoints": []}, NOW)
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

    empty = _write(tmp_path, "empty.json", None)
    assert main([td, "--backup-policy", empty, "--recovery-points", rp, "--now", NOW.isoformat()]) == 1
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
    _, ok, detail = check_recovery_point(points, "fs-0aaa", NOW, 26.0)
    assert not ok
    assert "undated=1" in detail


@pytest.mark.parametrize("now", ["not-a-time", "2026-13-40"])
def test_cli_unreadable_now_is_exit_two(tmp_path, now):
    td = _write(tmp_path, "td.json", _described())
    assert main([td, "--now", now]) == 2


@pytest.mark.parametrize("fs_id", ["", "   "])
def test_an_efs_volume_without_a_file_system_id_fails_the_backup_checks(fs_id):
    checks = run_checks(_described(fs_id=fs_id), {"BackupPolicy": {"Status": "ENABLED"}},
                        {"RecoveryPoints": [_point(NOW - timedelta(hours=1))]}, NOW)
    assert [(n, ok, d) for n, ok, d in checks][1:] == [
        ("backup_policy", False, "no_file_system_id"),
        ("recovery_point", False, "no_file_system_id"),
    ]


@pytest.mark.parametrize("limit", [26.0, 10.0])
def test_a_point_exactly_at_the_limit_still_counts(limit):
    at_limit = [_point(NOW - timedelta(hours=limit))]
    assert check_recovery_point(at_limit, "fs-0aaa", NOW, limit)[1] is True
    just_over = [_point(NOW - timedelta(hours=limit, seconds=36))]
    assert check_recovery_point(just_over, "fs-0aaa", NOW, limit)[1] is False


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
    assert check_recovery_point(points, "fs-0aaa", NOW, 4.0)[1] is ok


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
    ("/app/data", "/app/data/sub", False),
    ("/app/data", "/app", False),
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


# --- A bad input file is the checker's input fault: exit 2, never 1 (fault 2) ---

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
    assert "unreadable_input" in captured.err and "verdict=" not in captured.out


@pytest.mark.parametrize("blank", ["", "   \n"])
def test_cli_empty_task_definition_is_exit_two_not_a_failed_check(tmp_path, capsys, blank):
    td = tmp_path / "td.json"
    td.write_text(blank, encoding="utf-8")
    assert main([str(td)]) == 2
    assert "verdict=" not in capsys.readouterr().out


@pytest.mark.parametrize("flag", ["--backup-policy", "--recovery-points"])
def test_cli_an_empty_optional_file_is_still_a_missing_answer(tmp_path, capsys, flag):
    td = _write(tmp_path, "td.json", _described())
    empty = _write(tmp_path, "empty.json", None)
    assert main([td, flag, empty, "--now", NOW.isoformat()]) == 1
    assert "verdict=FAIL" in capsys.readouterr().out


# --- Notes: a future-dated point and the limit flag ---

@pytest.mark.parametrize("ahead_h", [1, 8760])
def test_a_recovery_point_dated_in_the_future_fails(ahead_h):
    _, ok, detail = check_recovery_point([_point(NOW + timedelta(hours=ahead_h))], "fs-0aaa", NOW, 26)
    assert ok is False and "age_h=-" in detail


def test_a_point_dated_exactly_now_counts():
    assert check_recovery_point([_point(NOW)], "fs-0aaa", NOW, 26)[1] is True
    assert check_recovery_point([_point(NOW)], "fs-0bbb", NOW, 1)[1] is True


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
    _, ok, detail = check_recovery_point([_point(NOW - timedelta(hours=1), arn=other)], fs_id, NOW, 26)
    assert ok is False
    assert "own=0" in detail and "other_volumes=1" in detail
