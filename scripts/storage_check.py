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
    ``REQUIRE_DURABLE_DOCUMENT_STORE`` on). Each setting is read the way the
    app reads it: the backend stripped and lowercased, the root compared as a
    normalised path, the guard on only for a value the app takes as on. A
    setting supplied as a ``secret`` has no value in the task definition, so
    the check cannot judge it and says ``UNVERIFIED``.
  * ``backup_policy``  the EFS file system the task mounts has automatic
    backups ``ENABLED``.
  * ``recovery_point`` that same file system has a ``COMPLETED`` recovery
    point no older than ``--max-age-hours`` and not dated in the future. A recovery point that names a
    different file system is not counted: a backup vault can hold another
    volume's backups, and seeing a full vault is not evidence that this
    volume is backed up.

Exit 0 only when every check is ``ok``; 1 when any check fails; 2 when an
input cannot be read (missing, empty task definition, not JSON, or JSON
that is not an object) or, with no check failed, a check is ``UNVERIFIED``. It never prints a secret: only setting names, the file
system id, counts and timestamps.
"""
from __future__ import annotations

import argparse
import json
import math
import posixpath
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
from app.core.byte_store import guard_value_on  # noqa: E402

_PREFIX = "STORAGE_CHECK"

#: One day of automatic-backup cadence plus slack for the backup window.
DEFAULT_MAX_AGE_HOURS = 26.0

#: (name, ok, detail); ok is None when the input cannot show the answer.
Check = Tuple[str, Optional[bool], str]


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


def _secret_names(taskdef: dict) -> set:
    container = (taskdef.get("containerDefinitions") or [{}])[0]
    return {s.get("name") for s in container.get("secrets") or []}


def _same_path(a: str, b: str) -> bool:
    a, b = (a or "").strip(), (b or "").strip()
    return bool(a) and bool(b) and posixpath.normpath(a) == posixpath.normpath(b)


def check_durable_store(described: dict) -> Check:
    taskdef = _taskdef(described)
    try:
        mount = efs_mount_path(taskdef)
    except RolloutRefused as exc:
        return ("durable_store", False, exc.code)
    env = environment(taskdef)
    secrets = _secret_names(taskdef)
    wrong: List[str] = []
    unread: List[str] = []
    for name in (BACKEND, ROOT, GUARD):
        if name in secrets and name not in env:
            unread.append(f"{name}=<secret>")
    backend = (env.get(BACKEND) or "").strip().lower()
    if BACKEND in env or BACKEND not in secrets:
        if backend != "efs":
            wrong.append(f"{BACKEND}={env.get(BACKEND) or '<unset>'}")
    if ROOT in env or ROOT not in secrets:
        if not _same_path(env.get(ROOT, ""), mount):
            wrong.append(f"{ROOT}={env.get(ROOT) or '<unset>'} mount={mount}")
    if GUARD in env or GUARD not in secrets:
        if not guard_value_on(env.get(GUARD, "")):
            shown = (env.get(GUARD) or "").strip()
            wrong.append(f"{GUARD}={shown + ' (off)' if shown else '<unset>'}")
    if wrong:
        return ("durable_store", False, " ".join(wrong + unread))
    if unread:
        return ("durable_store", None, " ".join(unread) + f" mount={mount}")
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
    return ("recovery_point", 0 <= age_h <= max_age_hours, detail)


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


def _load(path: Optional[Path], *, required: bool = False) -> Optional[dict]:
    """The JSON object in ``path``; None for an empty optional file.

    Raises ValueError for JSON that is not an object, or an empty required
    file, so a bad input is reported as unreadable (exit 2), not as a failed
    check (exit 1).
    """
    if path is None:
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        if required:
            raise ValueError(f"{path.name} is empty")
        return None
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} is not a JSON object")
    return data


def _max_age(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be a finite number of hours above 0")
    return value


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("described", type=Path, help="describe-task-definition JSON")
    ap.add_argument("--backup-policy", type=Path, default=None)
    ap.add_argument("--recovery-points", type=Path, default=None)
    ap.add_argument("--max-age-hours", type=_max_age, default=DEFAULT_MAX_AGE_HOURS)
    ap.add_argument("--now", default=None, help="ISO time to judge age against (default: now)")
    args = ap.parse_args(argv)
    try:
        described = _load(args.described, required=True)
        policy = _load(args.backup_policy)
        points = _load(args.recovery_points)
        now = _parse_when(args.now) if args.now else datetime.now(timezone.utc)
    except (OSError, ValueError) as exc:
        print(f"{_PREFIX} unreadable_input {type(exc).__name__}", file=sys.stderr)
        return 2
    checks = run_checks(described, policy, points, now, args.max_age_hours)
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
