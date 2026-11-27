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


def test_durable_store_fails_without_an_efs_volume():
    _, ok, detail = check_durable_store(_described(efs=False))
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


def test_mounted_but_unbacked_volume_fails_only_the_backup_checks():
    checks = run_checks(_described(fs_id="fs-0ccc"), None, {"RecoveryPoints": []}, NOW)
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


def test_cli_unreadable_input_is_exit_two(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
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
