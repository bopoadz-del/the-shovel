"""Storage check: is the volume that holds document bytes durable AND backed up.

Reads three JSON documents the AWS CLI already prints and makes no AWS call
itself (the same split as ``scripts/durable_store_rollout.py``):

  described         ``aws ecs describe-task-definition`` output for the live
                    task definition.
  --backup-policy   ``aws efs describe-backup-policy`` output for the EFS file
                    system that task mounts. Omit it (or pass an empty file)
                    when the CLI answered ``PolicyNotFound``.
  --recovery-points ``aws backup list-recovery-points-by-resource`` (or
                    ``...-by-backup-vault``) output.

Checks, one ``STORAGE_CHECK`` line each, then a verdict line:

  * ``durable_store``  the container has one read-write EFS mount and the
    document store is on it with the startup guard on
    (``DOCUMENT_BYTE_STORE=efs``, ``DOCUMENT_STORE_ROOT`` = the mount,
    ``REQUIRE_DURABLE_DOCUMENT_STORE`` set).
  * ``backup_policy``  the EFS file system the task mounts has automatic
    backups ``ENABLED``.
  * ``recovery_point`` that same file system has a ``COMPLETED`` recovery
    point no older than ``--max-age-hours``. A recovery point that names a
    different file system is not counted: a backup vault can hold another
    volume's backups, and seeing a full vault is not evidence that this
    volume is backed up.

Exit 0 only when every check is ``ok``; 1 when any check fails; 2 when an
input cannot be read. It never prints a secret: only setting names, the file
system id, counts and timestamps.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.durable_store_rollout import (  # noqa: E402
    BACKEND,
    GUARD,
    ROOT,
    RolloutRefused,
    efs_mount_path,
    environment,
)

_PREFIX = "STORAGE_CHECK"

#: One day of automatic-backup cadence plus slack for the backup window.
DEFAULT_MAX_AGE_HOURS = 26.0

Check = Tuple[str, bool, str]


def _taskdef(described: dict) -> dict:
    return described.get("taskDefinition", described)


def mounted_file_system(described: dict) -> str:
    """The EFS file system id behind the task's one read-write EFS mount."""
    taskdef = _taskdef(described)
    mount = efs_mount_path(taskdef)
    container = (taskdef.get("containerDefinitions") or [{}])[0]
    volume_name = next(
        m.get("sourceVolume") for m in container.get("mountPoints") or []
        if m.get("containerPath") == mount and not m.get("readOnly")
    )
    for volume in taskdef.get("volumes") or []:
        cfg = volume.get("efsVolumeConfiguration")
        if volume.get("name") == volume_name and cfg:
            fs_id = (cfg.get("fileSystemId") or "").strip()
            if fs_id:
                return fs_id
    raise RolloutRefused("no_file_system_id", "the EFS volume names no fileSystemId")


def check_durable_store(described: dict) -> Check:
    try:
        mount = efs_mount_path(_taskdef(described))
    except RolloutRefused as exc:
        return ("durable_store", False, exc.code)
    env = environment(_taskdef(described))
    wrong: List[str] = []
    if env.get(BACKEND) != "efs":
        wrong.append(f"{BACKEND}={env.get(BACKEND) or '<unset>'}")
    if env.get(ROOT) != mount:
        wrong.append(f"{ROOT}={env.get(ROOT) or '<unset>'} mount={mount}")
    if not (env.get(GUARD) or "").strip():
        wrong.append(f"{GUARD}=<unset>")
    if wrong:
        return ("durable_store", False, " ".join(wrong))
    return ("durable_store", True, f"mount={mount}")


def check_backup_policy(policy: Optional[dict], fs_id: str) -> Check:
    status = ((policy or {}).get("BackupPolicy") or {}).get("Status") or "<none>"
    ok = status == "ENABLED"
    return ("backup_policy", ok, f"file_system={fs_id} status={status}")


def _parse_when(raw: str) -> datetime:
    """An ISO time, read as UTC when it carries no zone. Raises ValueError."""
    when = datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when


def _names_file_system(point: dict, fs_id: str) -> bool:
    arn = point.get("ResourceArn")
    if not arn:
        # The by-resource listing omits the ARN; it was asked for this volume.
        return True
    return str(arn).rstrip("/").endswith(f"file-system/{fs_id}")


def check_recovery_point(
    points: Iterable[dict], fs_id: str, now: datetime, max_age_hours: float
) -> Check:
    points = list(points)
    own = [p for p in points if _names_file_system(p, fs_id)]
    foreign = len(points) - len(own)
    completed: List[datetime] = []
    undated = 0
    for p in own:
        if p.get("Status") != "COMPLETED":
            continue
        try:
            completed.append(_parse_when(str(p.get("CreationDate") or "")))
        except ValueError:
            # Counted and printed: a point whose date cannot be read is not
            # evidence of a recent backup.
            undated += 1
    detail = (
        f"file_system={fs_id} own={len(own)} completed={len(completed)} "
        f"undated={undated} other_volumes={foreign}"
    )
    if not completed:
        return ("recovery_point", False, detail + " newest=<none>")
    newest = max(completed)
    age_h = (now - newest) / timedelta(hours=1)
    detail += f" newest={newest.astimezone(timezone.utc).isoformat()} age_h={age_h:.1f}"
    return ("recovery_point", age_h <= max_age_hours, detail)


def run_checks(
    described: dict,
    policy: Optional[dict],
    recovery_points: Optional[dict],
    now: datetime,
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS,
) -> List[Check]:
    checks = [check_durable_store(described)]
    try:
        fs_id = mounted_file_system(described)
    except RolloutRefused as exc:
        checks.append(("backup_policy", False, exc.code))
        checks.append(("recovery_point", False, exc.code))
        return checks
    checks.append(check_backup_policy(policy, fs_id))
    points = (recovery_points or {}).get("RecoveryPoints") or []
    checks.append(check_recovery_point(points, fs_id, now, max_age_hours))
    return checks


def _load(path: Optional[Path]) -> Optional[dict]:
    if path is None:
        return None
    text = path.read_text(encoding="utf-8").strip()
    return json.loads(text) if text else None


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("described", type=Path, help="describe-task-definition JSON")
    ap.add_argument("--backup-policy", type=Path, default=None)
    ap.add_argument("--recovery-points", type=Path, default=None)
    ap.add_argument("--max-age-hours", type=float, default=DEFAULT_MAX_AGE_HOURS)
    ap.add_argument("--now", default=None, help="ISO time to judge age against (default: now)")
    args = ap.parse_args(argv)
    try:
        described = _load(args.described) or {}
        policy = _load(args.backup_policy)
        points = _load(args.recovery_points)
        now = _parse_when(args.now) if args.now else datetime.now(timezone.utc)
    except (OSError, ValueError) as exc:
        print(f"{_PREFIX} unreadable_input {type(exc).__name__}", file=sys.stderr)
        return 2
    checks = run_checks(described, policy, points, now, args.max_age_hours)
    for name, ok, detail in checks:
        print(f"{_PREFIX} {name}={'ok' if ok else 'FAIL'} {detail}")
    failed = [name for name, ok, _ in checks if not ok]
    print(f"{_PREFIX} verdict={'ok' if not failed else 'FAIL'} failed={','.join(failed) or 'none'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
