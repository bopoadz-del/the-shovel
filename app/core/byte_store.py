"""Where uploaded document bytes live.

Two backends, selected by ``DOCUMENT_BYTE_STORE``:

* ``local`` — a directory on the process filesystem (``DOCUMENT_STORE_ROOT``,
  or ``DATA_DIR`` when that is unset). Development, and the current
  production task until a durable volume is attached.
* ``efs`` — the same directory layout on a filesystem that is a real mount,
  distinct from the container root. Production uses this once EFS (or
  another durable volume) is attached at ``DOCUMENT_STORE_ROOT``.

S3 is not a backend here. Adding an object store is a separate change with
its own IAM; this module does not pretend to implement it.

Bytes on disk stay in the existing encrypted format (``file_crypto``).
Uploads are still streamed in bounded blocks. A row whose ``file_path``
points at an older location is still read from that path when the file is
there.

The production startup guard (``enforce_startup_guard``) refuses to boot
into ephemeral container storage only when asked. The live ECS task has no
durable volume yet, so the guard stays off until
``REQUIRE_DURABLE_DOCUMENT_STORE`` is set. See ``deploy/aws/README.md``.
"""

from __future__ import annotations

import errno
import logging
import os
import stat
import uuid
from dataclasses import dataclass

from app.core import file_crypto

logger = logging.getLogger(__name__)

LOCAL = "local"
EFS = "efs"
BACKENDS = frozenset({LOCAL, EFS})

PRESENT = "present"
MISSING = "missing"
UNAVAILABLE = "unavailable"
NOT_STORED = "not_stored"

BYTES_MISSING = "bytes_missing"
MISSING_LABEL = "File missing, re-upload"
MISSING_DETAIL = (
    "The stored copy of this file is gone. Re-upload it to restore the "
    "document. Text already indexed from it is kept."
)

# Filesystem types that are the container's own disk or a ram disk. A
# directory on one of these is not a durable mount, even when its device
# id differs from `/` (a tmpfs does).
_EPHEMERAL_FSTYPES = frozenset({
    "tmpfs", "overlay", "overlayfs", "ramfs", "devtmpfs", "squashfs",
})

_NETWORK_ERRNO_NAMES = (
    "EIO", "ESTALE", "ETIMEDOUT", "ENETUNREACH", "EHOSTUNREACH", "ENETDOWN",
    "ECONNRESET", "EREMOTEIO", "ENOTCONN", "EROFS", "EBUSY", "ENOLINK",
    "EHOSTDOWN", "ETIMEDOUT",
)


class StoreError(Exception):
    """The document byte store could not complete an operation."""


class StoreConfigError(StoreError):
    """``DOCUMENT_BYTE_STORE`` or a path is not a configuration this build runs."""


class StoreUnavailable(StoreError):
    """The store root cannot be created or written. Not a missing document."""


@dataclass(frozen=True)
class MountIdentity:
    """One filesystem mount, as the kernel reports it."""

    device: int
    mountpoint: str
    fstype: str
    source: str


@dataclass(frozen=True)
class ByteInspection:
    """Whether a path's bytes can be confirmed.

    ``missing`` is only a confirmed absence (the directory can be stated
    and the file is not there). Permission, network, decryption and mount
    failures are ``unavailable`` so they cannot be shown as a lost file.
    """

    state: str
    reason: str
    detail: str = ""


def backend_name() -> str:
    """``local`` or ``efs``. Any other value, including an object store, is refused."""
    raw = os.getenv("DOCUMENT_BYTE_STORE", LOCAL).strip().lower() or LOCAL
    if raw not in BACKENDS:
        raise StoreConfigError(
            f"DOCUMENT_BYTE_STORE={raw!r} is not supported. "
            f"This build stores document bytes on a filesystem "
            f"({', '.join(sorted(BACKENDS))})."
        )
    return raw


def store_root() -> str:
    """Directory that new document bytes are written under."""
    configured = os.getenv("DOCUMENT_STORE_ROOT", "").strip()
    if configured:
        return configured
    return os.getenv("DATA_DIR", "./data")


def _is_production() -> bool:
    env = os.getenv("ENV", os.getenv("ENVIRONMENT", "")).strip().lower()
    return env in ("prod", "production")


def durable_guard_requested() -> bool:
    """True when the operator has turned the rollout gate on.

    The live task does not set this. Leaving it unset is what keeps a
    deploy from refusing to boot before a durable volume exists.
    """
    return os.getenv("REQUIRE_DURABLE_DOCUMENT_STORE", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _unescape_mount(value: str) -> str:
    return (
        value.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
    )


def read_mount_table() -> list[tuple[str, str, str]]:
    """``(mountpoint, fstype, source)`` from ``/proc/self/mountinfo``.

    An unreadable table is an empty list. Callers then have only the
    device id from ``stat``, which still distinguishes two filesystems.
    """
    try:
        with open("/proc/self/mountinfo", "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        logger.warning("could not read /proc/self/mountinfo", exc_info=True)
        return []
    rows: list[tuple[str, str, str]] = []
    for line in text.splitlines():
        if " - " not in line:
            continue
        left, right = line.split(" - ", 1)
        left_fields = left.split()
        right_fields = right.split()
        if len(left_fields) < 5 or len(right_fields) < 2:
            continue
        rows.append((
            _unescape_mount(left_fields[4]),
            right_fields[0],
            _unescape_mount(right_fields[1]),
        ))
    return rows


def _covers(mountpoint: str, real: str) -> bool:
    mp = mountpoint.rstrip("/") or "/"
    if mp == "/":
        return real.startswith("/")
    return real == mp or real.startswith(mp + os.sep)


def mount_identity(path: str) -> MountIdentity:
    """Identity of the mount that holds ``path`` (the path must exist)."""
    st = os.stat(path)
    real = os.path.realpath(path)
    best: tuple[str, str, str] | None = None
    best_len = -1
    for mountpoint, fstype, source in read_mount_table():
        if _covers(mountpoint, real) and len(mountpoint) > best_len:
            best = (mountpoint, fstype, source)
            best_len = len(mountpoint)
    if best is None:
        return MountIdentity(device=st.st_dev, mountpoint=real, fstype="", source="")
    return MountIdentity(
        device=st.st_dev, mountpoint=best[0], fstype=best[1], source=best[2],
    )


def durable_decision(store: MountIdentity, root: MountIdentity) -> bool:
    """True when ``store`` is a different, non-ephemeral mount from ``root``.

    A directory that exists is not enough: the container root is a
    directory too. The decision is the mount's device, its mount point,
    and its filesystem type.
    """
    if store.fstype.lower() in _EPHEMERAL_FSTYPES:
        return False
    if store.device == root.device:
        return False
    if (
        store.mountpoint
        and root.mountpoint
        and store.mountpoint.rstrip("/") == root.mountpoint.rstrip("/")
    ):
        return False
    return True


def _existing_ancestor(path: str) -> str:
    cur = os.path.abspath(path)
    while not os.path.exists(cur):
        parent = os.path.dirname(cur)
        if parent == cur:
            return cur
        cur = parent
    return cur


def is_durable_mount(path: str) -> bool:
    """True when ``path`` (or the closest existing parent) is a durable mount."""
    ancestor = _existing_ancestor(path)
    if not os.path.isdir(ancestor):
        return False
    try:
        store = mount_identity(ancestor)
        root = mount_identity("/")
    except OSError:
        logger.warning("mount identity unreadable for %s", path, exc_info=True)
        return False
    return durable_decision(store, root)


def probe_store(root: str) -> None:
    """Write, read back, and delete a probe file. Leaves nothing behind."""
    try:
        os.makedirs(root, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            f"document store probe cannot create {root}: {exc.strerror or exc}"
        ) from exc
    name = f".fork-store-probe-{uuid.uuid4().hex}"
    path = os.path.join(root, name)
    payload = b"probe-" + uuid.uuid4().hex.encode("ascii")
    try:
        with open(path, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        with open(path, "rb") as fh:
            got = fh.read()
        if got != payload:
            raise RuntimeError("document store probe read back different bytes")
    except OSError as exc:
        raise RuntimeError(
            f"document store probe failed under {root}: {exc.strerror or exc}"
        ) from exc
    finally:
        try:
            os.remove(path)
        except OSError:
            logger.warning("could not remove store probe %s", path, exc_info=True)
    if os.path.exists(path):
        raise RuntimeError("document store probe could not delete its file")


def _not_enforcing_message() -> str:
    return (
        "REQUIRE_DURABLE_DOCUMENT_STORE is unset, so production is booting "
        f"with document bytes on {store_root()!r} (DOCUMENT_BYTE_STORE=local). "
        "This process will accept uploads there even when that directory is "
        "the container's own disk. The live ECS task has no EFS volume yet; "
        "leave this flag unset until the volume is mounted and "
        "DOCUMENT_BYTE_STORE=efs, or the service will refuse to start. "
        "Until then a task restart can drop uploaded document bytes."
    )


def _flag_without_efs_message() -> str:
    return (
        "REQUIRE_DURABLE_DOCUMENT_STORE is set, but DOCUMENT_BYTE_STORE is "
        "local. Refusing to serve uploads from container storage. Mount a "
        "durable volume, set DOCUMENT_BYTE_STORE=efs, and point "
        "DOCUMENT_STORE_ROOT at that mount. If this deploy is the current "
        "live task and the volume is not attached yet, unset "
        "REQUIRE_DURABLE_DOCUMENT_STORE so the service can boot, then turn "
        "the flag on after the mount is in place."
    )


def _efs_mount_message(root: str) -> str:
    return (
        f"DOCUMENT_BYTE_STORE=efs but {root!r} is not a durable mount "
        "distinct from the container root filesystem. The check compares "
        "mount identity (device, mount point, and filesystem type) and then "
        "writes, reads, and deletes a probe file. A directory that merely "
        "exists on the container disk is not enough. Attach the EFS volume "
        "at this path before starting."
    )


def enforce_startup_guard() -> str:
    """Refuse ephemeral document storage when this process is not allowed to use it.

    Returns a short status for logs: ``local``, ``not-enforcing``, or
    ``enforced-efs``. Raises ``RuntimeError`` when the process must not serve.

    ``efs`` is checked in every environment: selecting it is a claim that
    the mount is already there. ``local`` is refused in production only
    when ``REQUIRE_DURABLE_DOCUMENT_STORE`` is set, so the current live
    task (no volume, flag unset) still boots.
    """
    backend = backend_name()
    if backend == EFS:
        root = store_root()
        try:
            durable = is_durable_mount(root)
        except OSError as exc:
            raise RuntimeError(
                _efs_mount_message(root) + f" ({exc.strerror or exc})"
            ) from exc
        if not durable:
            raise RuntimeError(_efs_mount_message(root))
        probe_store(root)
        logger.info("document byte store: efs mount verified at %s", root)
        return "enforced-efs"
    if _is_production() and durable_guard_requested():
        raise RuntimeError(_flag_without_efs_message())
    if _is_production():
        logger.error(_not_enforcing_message())
        return "not-enforcing"
    return "local"


def ensure_root() -> str:
    """Create the store root. A failure is an error, never a switch to the temp dir."""
    root = store_root()
    try:
        os.makedirs(root, exist_ok=True)
    except OSError as exc:
        raise StoreUnavailable(
            f"document store root {root} is not usable ({exc.strerror or exc})"
        ) from exc
    if not os.path.isdir(root):
        raise StoreUnavailable(f"document store root {root} is not a directory")
    return root


def allocate_path(stored_as: str) -> str:
    """A new path inside the store for ``stored_as`` (a file name, not a path)."""
    root = ensure_root()
    name = os.path.basename((stored_as or "").replace("\\", "/"))
    if not name or name in {".", ".."}:
        raise StoreConfigError("invalid document name")
    return os.path.join(root, name)


def assert_inside_store(path: str) -> None:
    """Refuse to write document bytes anywhere but the store root."""
    root = os.path.realpath(ensure_root())
    parent = os.path.realpath(os.path.dirname(path) or ".")
    candidate = os.path.join(parent, os.path.basename(path))
    if os.path.islink(path):
        raise StoreConfigError("refusing to write document bytes through a symlink")
    if candidate != root and not candidate.startswith(root + os.sep):
        raise StoreConfigError("refusing to write document bytes outside the store")


def write_stream(path: str, fileobj, **kwargs) -> int:
    """Stream ``fileobj`` to ``path`` in the existing encrypted format."""
    assert_inside_store(path)
    return file_crypto.write_document_stream(path, fileobj, **kwargs)


def write_bytes(path: str, data: bytes) -> None:
    """Write an already-buffered payload in the existing encrypted format."""
    assert_inside_store(path)
    file_crypto.write_document(path, data)


def _errno_reason(exc: OSError) -> str:
    err = exc.errno
    if err in (errno.EACCES, errno.EPERM):
        return "permission"
    names = {getattr(errno, name, None) for name in _NETWORK_ERRNO_NAMES}
    names.discard(None)
    if err in names:
        return "io"
    return "os"


def _store_blocks_absence_checks() -> ByteInspection | None:
    """Return an inspection when an ENOENT would not prove the file is gone.

    An EFS backend whose mount is not the durable volume makes every path
    look missing. A local root that cannot be stated is the same class of
    failure: the store itself is unavailable, not the document.
    """
    root = store_root()
    if backend_name() == EFS:
        try:
            healthy = is_durable_mount(root)
        except OSError as exc:
            return ByteInspection(UNAVAILABLE, "mount", exc.strerror or "mount")
        if not healthy:
            return ByteInspection(
                UNAVAILABLE, "mount", "efs store is not a distinct mount",
            )
        return None
    try:
        st = os.stat(root)
    except FileNotFoundError:
        return ByteInspection(UNAVAILABLE, "mount", "document store root does not exist")
    except PermissionError as exc:
        return ByteInspection(UNAVAILABLE, "permission", exc.strerror or "permission")
    except OSError as exc:
        return ByteInspection(UNAVAILABLE, _errno_reason(exc), exc.strerror or "os")
    if not stat.S_ISDIR(st.st_mode):
        return ByteInspection(UNAVAILABLE, "mount", "document store root is not a directory")
    return None


def inspect_path(path: str) -> ByteInspection:
    """Classify ``path`` without reading or decrypting it.

    Decryption is deliberately not attempted. A file that is present but
    cannot be decrypted is ``present``; the reader raises ``DecryptionError``
    later, and that stays an extraction failure.
    """
    if not path:
        return ByteInspection(NOT_STORED, "no_path", "")
    blocked = _store_blocks_absence_checks()
    if blocked is not None:
        return blocked
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return ByteInspection(MISSING, "enoent", "")
    except PermissionError as exc:
        return ByteInspection(UNAVAILABLE, "permission", exc.strerror or "permission")
    except OSError as exc:
        if exc.errno in (errno.ENOENT,):
            return ByteInspection(MISSING, "enoent", "")
        return ByteInspection(UNAVAILABLE, _errno_reason(exc), exc.strerror or "os")
    if not stat.S_ISREG(st.st_mode):
        return ByteInspection(UNAVAILABLE, "not_a_file", "")
    return ByteInspection(PRESENT, "ok", "")


def inspect_local_claim(
    *,
    file_path: str | None,
    drive_file_id: str | None,
    provenance: str | None,
) -> ByteInspection:
    """Classify the bytes a document row claims.

    A Drive-sourced row keeps no archive. A missing local path on that row
    is the no-archive rule, not a lost upload. A user upload (no Drive id,
    or provenance ``user_upload``) is a local claim: confirmed absence is
    ``missing``.
    """
    drive_id = (drive_file_id or "").strip()
    origin = (provenance or "").strip()
    path = file_path or ""
    if drive_id and origin != "user_upload":
        return ByteInspection(NOT_STORED, "drive_source", "")
    if not path:
        return ByteInspection(NOT_STORED, "no_path", "")
    return inspect_path(path)


def is_terminal_bytes_missing(result: object) -> bool:
    """True when an index result is a confirmed, terminal lost-file outcome."""
    return (
        isinstance(result, dict)
        and result.get("terminal") is True
        and result.get("error") == BYTES_MISSING
    )
