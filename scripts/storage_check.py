"""Storage check: is the volume that holds document bytes durable AND backed up.

Reads three JSON documents the AWS CLI already prints and makes no AWS call
itself (the same split as ``scripts/durable_store_rollout.py``):

  described           ``aws ecs describe-task-definition`` output for the
                      live task definition.
  --backup-policy     ``aws efs describe-backup-policy`` output for the EFS
                      file system that task mounts.
  --no-backup-policy  instead of the file: the CLI answered
                      ``PolicyNotFound`` (no backup policy exists).
  --recovery-points   ``aws backup list-recovery-points-by-resource`` (or
                      ``...-by-backup-vault``) output.

Checks, one ``STORAGE_CHECK`` line each, then a verdict line:

  * ``durable_store``  the container that mounts the EFS volume read-write
    keeps the document store on it with the startup guard on
    (``DOCUMENT_BYTE_STORE=efs``, ``DOCUMENT_STORE_ROOT`` = the mount,
    ``REQUIRE_DURABLE_DOCUMENT_STORE`` on). Each setting is read the way the
    app reads it: the backend stripped and lowercased, the root compared as a
    normalised path, the guard on only for a value the app takes as on.
  * ``backup_policy``  the EFS file system the task mounts has automatic
    backups ``ENABLED``.
  * ``recovery_point`` that same file system has a ``COMPLETED`` recovery
    point no older than ``--max-age-hours``. A recovery point that names a
    different file system is not counted: a backup vault can hold another
    volume's backups, and seeing a full vault is not evidence that this
    volume is backed up.

Every input is either read and understood, or the checks that need it say
``UNVERIFIED`` with ``not_checked=<input>:<reason>``. An input the checker
cannot read (missing, empty, not JSON, the wrong shape, the wrong command's
output) is never a pass and never a fail. The same holds inside an input: a
setting held as a secret (its running value is not in the task definition),
a task where more than one container mounts EFS read-write, an EFS volume
naming no file system, and a recovery point whose date cannot be read or
lies ahead of ``--now`` are each reported, not judged.

Exit 0 only when every check is ``ok``; 1 when any check fails; 2 when no
check failed and one is ``UNVERIFIED``, or ``--now`` cannot be read. It
never prints a secret: only setting names, the file system id, counts and
timestamps.
"""
from __future__ import annotations

import argparse
import json
import math
import posixpath
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.byte_store import guard_value_on
from scripts.durable_store_rollout import BACKEND, GUARD, ROOT

_PREFIX = "STORAGE_CHECK"

#: One day of automatic-backup cadence plus slack for the backup window.
DEFAULT_MAX_AGE_HOURS = 26.0

#: (name, ok, detail); ok is None when the input cannot show the answer.
Check = tuple[str, bool | None, str]


@dataclass(frozen=True)
class NotRead:
    """An input, or a part of one, the checker could not read."""

    reason: str


class _PolicyNotFound:
    """The CLI answered PolicyNotFound: the file system has no backup policy."""


POLICY_NOT_FOUND = _PolicyNotFound()

Input = dict | NotRead | None


class _Unreadable(ValueError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _object(value: Any, where: str) -> dict:
    if not isinstance(value, dict):
        raise _Unreadable(f"{where}_not_an_object")
    return value


def _objects(value: Any, where: str) -> list[dict]:
    """A list of objects; an absent or null list is empty."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise _Unreadable(f"{where}_not_a_list")
    return [_object(item, where) for item in value]


def _text(value: Any, where: str) -> str:
    """A string field; an absent or null field is empty."""
    if value is None:
        return ""
    if not isinstance(value, str):
        raise _Unreadable(f"{where}_not_text")
    return value


@dataclass(frozen=True)
class _Mount:
    container: dict
    path: str
    fs_id: str


def _taskdef(described: dict) -> dict:
    if "taskDefinition" in described:
        return _object(described["taskDefinition"], "taskDefinition")
    if "containerDefinitions" not in described:
        raise _Unreadable("no_task_definition")
    return described


def _efs_mounts(taskdef: dict) -> list[_Mount]:
    """Every read-write mount of an EFS volume, in every container."""
    containers = _objects(taskdef.get("containerDefinitions"), "containerDefinitions")
    if not containers:
        raise _Unreadable("no_containers")
    efs: dict[str, str] = {}
    for volume in _objects(taskdef.get("volumes"), "volumes"):
        if "efsVolumeConfiguration" in volume:
            cfg = _object(volume["efsVolumeConfiguration"], "efsVolumeConfiguration")
            efs[_text(volume.get("name"), "volume_name")] = _text(
                cfg.get("fileSystemId"), "fileSystemId").strip()
    mounts: list[_Mount] = []
    for container in containers:
        for m in _objects(container.get("mountPoints"), "mountPoints"):
            source = _text(m.get("sourceVolume"), "sourceVolume")
            read_only = m.get("readOnly")
            if read_only is not None and not isinstance(read_only, bool):
                raise _Unreadable("readOnly_not_a_flag")
            if source in efs and not read_only:
                mounts.append(_Mount(container, _text(m.get("containerPath"), "containerPath"), efs[source]))
    return mounts


def mounted_file_system(described: dict) -> str:
    """The EFS file system id behind the task's read-write EFS mount.

    Raises ``_Unreadable`` when the task does not show one file system.
    """
    ids = {m.fs_id for m in _efs_mounts(_taskdef(described))}
    if not ids:
        raise _Unreadable("no_efs_mount")
    if len(ids) > 1:
        raise _Unreadable(f"efs_file_systems={len(ids)}")
    fs_id = ids.pop()
    if not fs_id:
        raise _Unreadable("no_file_system_id")
    return fs_id


def _settings(container: dict) -> tuple[dict[str, str], set]:
    env = {}
    for e in _objects(container.get("environment"), "environment"):
        env[_text(e.get("name"), "environment_name")] = _text(e.get("value"), "environment_value")
    secrets = {_text(s.get("name"), "secret_name") for s in _objects(container.get("secrets"), "secrets")}
    return env, secrets


def _same_path(a: str, b: str) -> bool:
    a, b = (a or "").strip(), (b or "").strip()
    return bool(a) and bool(b) and posixpath.normpath(a) == posixpath.normpath(b)


def _not_checked(name: str, reason: str, fs_id: str = "") -> Check:
    shown = f"file_system={fs_id} " if fs_id else ""
    return (name, None, f"{shown}not_checked={reason}")


def check_durable_store(described: Input) -> Check:
    name = "durable_store"
    if not isinstance(described, dict):
        return _not_checked(name, f"described:{_reason(described)}")
    try:
        mounts = _efs_mounts(_taskdef(described))
    except _Unreadable as exc:
        return _not_checked(name, f"described:{exc.reason}")
    if not mounts:
        return (name, False, "no_efs_mount")
    if len(mounts) > 1:
        # Which mount holds documents is not in the task definition.
        return _not_checked(name, f"described:efs_mounts={len(mounts)}")
    mount = mounts[0]
    if not mount.path.startswith("/"):
        return (name, False, "efs_mount_not_absolute")
    try:
        env, secrets = _settings(mount.container)
    except _Unreadable as exc:
        return _not_checked(name, f"described:{exc.reason}")
    wrong: list[str] = []
    # A setting held as a secret runs with a value this file does not show,
    # even when a plain value of the same name is also there.
    unread = [f"{n}=<secret>" for n in (BACKEND, ROOT, GUARD) if n in secrets]
    if BACKEND not in secrets and env.get(BACKEND, "").strip().lower() != "efs":
        wrong.append(f"{BACKEND}={env.get(BACKEND) or '<unset>'}")
    if ROOT not in secrets and not _same_path(env.get(ROOT, ""), mount.path):
        wrong.append(f"{ROOT}={env.get(ROOT) or '<unset>'} mount={mount.path}")
    if GUARD not in secrets and not guard_value_on(env.get(GUARD, "")):
        shown = env.get(GUARD, "").strip()
        wrong.append(f"{GUARD}={shown + ' (off)' if shown else '<unset>'}")
    if wrong:
        return (name, False, " ".join(wrong + unread))
    if unread:
        return (name, None, " ".join(unread) + f" mount={mount.path}")
    return (name, True, f"mount={mount.path}")


def _reason(value: Any) -> str:
    return value.reason if isinstance(value, NotRead) else "not_given"


def check_backup_policy(policy: Input | _PolicyNotFound, fs_id: str) -> Check:
    name = "backup_policy"
    if policy is POLICY_NOT_FOUND:
        return (name, False, f"file_system={fs_id} status=PolicyNotFound")
    if not isinstance(policy, dict):
        return _not_checked(name, f"backup-policy:{_reason(policy)}", fs_id)
    try:
        if "BackupPolicy" not in policy:
            raise _Unreadable("no_BackupPolicy")
        status = _text(_object(policy["BackupPolicy"], "BackupPolicy").get("Status"), "Status")
        if not status:
            raise _Unreadable("no_Status")
    except _Unreadable as exc:
        return _not_checked(name, f"backup-policy:{exc.reason}", fs_id)
    return (name, status == "ENABLED", f"file_system={fs_id} status={status}")


def _parse_when(raw: Any) -> datetime:
    """An ISO time (UTC when it carries no zone) or epoch seconds.

    The AWS CLI prints epoch seconds unless told otherwise (v1's default).
    Raises TypeError or ValueError for anything else.
    """
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return datetime.fromtimestamp(raw, tz=timezone.utc)
    if not isinstance(raw, str):
        raise TypeError("not a time")
    when = datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when


def _names_file_system(point: dict, fs_id: str) -> bool:
    arn = _text(point.get("ResourceArn"), "ResourceArn")
    if not arn:
        # The by-resource listing omits the ARN; it was asked for this volume.
        return True
    return arn.rstrip("/").endswith(f"file-system/{fs_id}")


def check_recovery_point(
    points: Input, fs_id: str, now: datetime, max_age_hours: float
) -> Check:
    name = "recovery_point"
    if not isinstance(points, dict):
        return _not_checked(name, f"recovery-points:{_reason(points)}", fs_id)
    try:
        if "RecoveryPoints" not in points:
            raise _Unreadable("no_RecoveryPoints")
        listed = _objects(points["RecoveryPoints"], "RecoveryPoints")
        own = [p for p in listed if _names_file_system(p, fs_id)]
    except _Unreadable as exc:
        return _not_checked(name, f"recovery-points:{exc.reason}", fs_id)
    completed: list[datetime] = []
    undated = ahead = 0
    for p in own:
        if p.get("Status") != "COMPLETED":
            continue
        try:
            when = _parse_when(p.get("CreationDate"))
        except (TypeError, ValueError, OverflowError, OSError):
            undated += 1
            continue
        if when > now:
            # A date after --now is a clock or input fault, not a backup age.
            ahead += 1
        else:
            completed.append(when)
    detail = (
        f"file_system={fs_id} own={len(own)} completed={len(completed)} "
        f"undated={undated} ahead={ahead} other_volumes={len(listed) - len(own)}"
    )
    if completed:
        newest = max(completed)
        age_h = (now - newest) / timedelta(hours=1)
        detail += f" newest={newest.astimezone(timezone.utc).isoformat()} age_h={age_h:.1f}"
        if age_h <= max_age_hours:
            return (name, True, detail)
    else:
        detail += " newest=<none>"
    if undated or ahead:
        # A point this check could not date may be the fresh one.
        return (name, None, detail)
    return (name, False, detail)


def run_checks(
    described: Input,
    policy: Input | _PolicyNotFound,
    recovery_points: Input,
    now: datetime,
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS,
) -> list[Check]:
    checks = [check_durable_store(described)]
    if not isinstance(described, dict):
        reason = f"described:{_reason(described)}"
        return checks + [_not_checked("backup_policy", reason), _not_checked("recovery_point", reason)]
    try:
        fs_id = mounted_file_system(described)
    except _Unreadable as exc:
        reason = f"described:{exc.reason}"
        return checks + [_not_checked("backup_policy", reason), _not_checked("recovery_point", reason)]
    checks.append(check_backup_policy(policy, fs_id))
    checks.append(check_recovery_point(recovery_points, fs_id, now, max_age_hours))
    return checks


def _load(path: Path | None) -> Input:
    """The JSON object in ``path``; None when no path was given; else NotRead."""
    if path is None:
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        return NotRead(f"unreadable_file_{type(exc).__name__}")
    if not text:
        return NotRead("empty_file")
    try:
        data = json.loads(text)
    except ValueError:
        return NotRead("not_json")
    if not isinstance(data, dict):
        return NotRead("not_an_object")
    return data


def _max_age(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be a finite number of hours above 0")
    return value


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("described", type=Path, help="describe-task-definition JSON")
    policy_arg = ap.add_mutually_exclusive_group()
    policy_arg.add_argument("--backup-policy", type=Path, default=None)
    policy_arg.add_argument("--no-backup-policy", action="store_true",
                            help="describe-backup-policy answered PolicyNotFound")
    ap.add_argument("--recovery-points", type=Path, default=None)
    ap.add_argument("--max-age-hours", type=_max_age, default=DEFAULT_MAX_AGE_HOURS)
    ap.add_argument("--now", default=None, help="ISO time to judge age against (default: now)")
    args = ap.parse_args(argv)
    try:
        now = _parse_when(args.now) if args.now else datetime.now(timezone.utc)
    except ValueError:
        print(f"{_PREFIX} unreadable_input now", file=sys.stderr)
        return 2
    inputs = {
        "described": _load(args.described),
        "backup-policy": POLICY_NOT_FOUND if args.no_backup_policy else _load(args.backup_policy),
        "recovery-points": _load(args.recovery_points),
    }
    for label, value in inputs.items():
        if isinstance(value, NotRead):
            print(f"{_PREFIX} unreadable_input {label} {value.reason}", file=sys.stderr)
    checks = run_checks(inputs["described"], inputs["backup-policy"],
                        inputs["recovery-points"], now, args.max_age_hours)
    for name, ok, detail in checks:
        word = "UNVERIFIED" if ok is None else ("ok" if ok else "FAIL")
        print(f"{_PREFIX} {name}={word} {detail}")
    failed = [name for name, ok, _ in checks if ok is False]
    unverified = [name for name, ok, _ in checks if ok is None]
    if failed:
        print(f"{_PREFIX} verdict=FAIL failed={','.join(failed)}")
        return 1
    if unverified:
        print(f"{_PREFIX} verdict=UNVERIFIED unverified={','.join(unverified)}")
        return 2
    print(f"{_PREFIX} verdict=ok failed=none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
