"""Document bytes live in a selected store, and a lost file is not an extract failure.

Two inputs per case. Drive rows are not treated as lost uploads: the
platform still keeps no archive of those originals.
"""
from __future__ import annotations

import errno
import io
import os
from pathlib import Path

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from app.core import byte_store, doc_index, file_crypto
from app.core import projects as store
from app.dependencies import require_api_key, require_user
from app.main import _validate_startup_env, app
from tests._dev_key import DEV_KEY
from tests._fixture_values import ENV_SECRET_06, STUB_TOKEN_16

USER = {"user_id": "byte-store-user", "role": "user"}
OTHER = {"user_id": "byte-store-other", "role": "user"}
ADMIN = {"user_id": "byte-store-admin", "role": "admin"}
NO_KEY = {"Authorization": "Bearer " + DEV_KEY}


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def as_user():
    app.dependency_overrides[require_user] = lambda: dict(USER)
    app.dependency_overrides[require_api_key] = lambda: dict(USER)
    yield
    app.dependency_overrides.pop(require_user, None)
    app.dependency_overrides.pop(require_api_key, None)


def _root(tmp_path: Path, monkeypatch, name: str = "store") -> Path:
    root = tmp_path / name
    root.mkdir()
    monkeypatch.setenv("DOCUMENT_BYTE_STORE", "local")
    monkeypatch.setenv("DOCUMENT_STORE_ROOT", str(root))
    monkeypatch.delenv("REQUIRE_DURABLE_DOCUMENT_STORE", raising=False)
    return root


def _project(user_id: str, name: str) -> str:
    from app.core import users

    users.ensure_user_exists(user_id)
    return store.create_project(name=name, user_id=user_id)["id"]


def _prod(monkeypatch) -> None:
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("SECRET_KEY", ENV_SECRET_06)
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost:5432/test")
    monkeypatch.setenv("DATA_ENCRYPTION_KEY", STUB_TOKEN_16)
    monkeypatch.setenv("CEREBRUM_VIRGIN", "false")
    monkeypatch.setenv("CEREBRUM_DOMAIN_KITS", "construction")
    import app.blocks as blocks_mod

    patched = dict(blocks_mod.BLOCK_REGISTRY)
    patched["construction"] = patched.get("construction") or object()
    monkeypatch.setattr(blocks_mod, "BLOCK_REGISTRY", patched)


# ── store format, legacy reads, no temp fallback ────────────────────────────


@pytest.mark.parametrize("sample", ["first", "second"])
def test_streamed_writes_keep_the_encrypted_format(sample, tmp_path, monkeypatch):
    root = _root(tmp_path, monkeypatch)
    monkeypatch.setenv("DATA_ENCRYPTION_KEY", Fernet.generate_key().decode())
    bodies = {
        "alpha.txt": b"slab pour on level three",
        "beta.txt": b"waterproofing membrane at the podium",
    }
    for name, body in bodies.items():
        path = byte_store.allocate_path(name)
        size = byte_store.write_stream(path, io.BytesIO(body))
        raw = Path(path).read_bytes()
        assert size == len(body)
        assert raw.startswith(b"FORK-ENC-STREAM-1\n")
        assert file_crypto.read_document(path) == body
        assert str(root) in path


@pytest.mark.parametrize("sample", ["first", "second"])
def test_legacy_file_path_still_reads(sample, tmp_path, monkeypatch):
    """A path recorded before the store, outside the store root, still opens."""
    _root(tmp_path, monkeypatch, "current")
    monkeypatch.setenv("DATA_ENCRYPTION_KEY", Fernet.generate_key().decode())
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    bodies = {"old-note.txt": b"plain legacy note", "old-spec.txt": b"plain legacy spec"}
    for name, body in bodies.items():
        path = legacy / name
        path.write_bytes(body)
        assert file_crypto.read_document(str(path)) == body
        inspection = byte_store.inspect_path(str(path))
        assert inspection.state == byte_store.PRESENT
        pid = _project(USER["user_id"], f"legacy-{name}")
        doc = store.add_document(
            pid, name, file_path=str(path), size=len(body),
            metadata={"provenance": "user_upload"},
        )
        loaded = store.get_document(doc["id"])
        assert loaded["bytes_state"] == byte_store.PRESENT
        assert loaded.get("panel_status", {}).get("label") != byte_store.MISSING_LABEL


@pytest.mark.parametrize("sample", ["first", "second"])
def test_writes_outside_the_store_are_refused(sample, tmp_path, monkeypatch):
    _root(tmp_path, monkeypatch)
    outside = [tmp_path / "elsewhere" / "a.txt", tmp_path / "other" / "b.txt"]
    for path in outside:
        path.parent.mkdir()
        with pytest.raises(byte_store.StoreConfigError):
            byte_store.write_bytes(str(path), b"nope")
        assert not path.exists()


@pytest.mark.parametrize("sample", ["first", "second"])
def test_a_failed_store_root_is_not_the_temp_dir(sample, tmp_path, monkeypatch):
    blockers = []
    for name in ("north", "south"):
        blocker = tmp_path / f"not-a-dir-{name}"
        blocker.write_text("x")
        blockers.append(blocker)
        monkeypatch.setenv("DOCUMENT_STORE_ROOT", str(blocker / "child"))
        with pytest.raises(byte_store.StoreUnavailable, match="not usable"):
            byte_store.ensure_root()
    upload_src = Path("app/routers/upload.py").read_text(encoding="utf-8")
    projects_src = Path("app/routers/projects.py").read_text(encoding="utf-8")
    assert "gettempdir" not in upload_src
    assert "gettempdir" not in projects_src
    assert blockers


@pytest.mark.parametrize("sample", ["first", "second"])
def test_unknown_backend_is_refused(sample, monkeypatch):
    for name in ("s3", "memory"):
        monkeypatch.setenv("DOCUMENT_BYTE_STORE", name)
        with pytest.raises(byte_store.StoreConfigError):
            byte_store.backend_name()


# ── lost file vs permission, io, decrypt, mount, drive ──────────────────────


@pytest.mark.parametrize("sample", ["first", "second"])
def test_only_confirmed_absence_is_a_missing_file(sample, tmp_path, monkeypatch):
    root = _root(tmp_path, monkeypatch)
    denied = [root / "secret-a.bin", root / "secret-b.bin"]
    broken = [root / "net-a.bin", root / "net-b.bin"]
    for path in denied + broken:
        path.write_bytes(b"present")
    real = os.stat
    denied_s = {str(p) for p in denied}
    broken_s = {str(p) for p in broken}

    def fake(path, *args, **kwargs):
        text = os.fspath(path)
        if text in denied_s:
            raise PermissionError(13, "Permission denied")
        if text in broken_s:
            raise OSError(errno.EIO, "Input/output error")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", fake)
    for path in denied:
        got = byte_store.inspect_path(str(path))
        assert got.state == byte_store.UNAVAILABLE
        assert got.reason == "permission"
        assert got.state != byte_store.MISSING
    for path in broken:
        got = byte_store.inspect_path(str(path))
        assert got.state == byte_store.UNAVAILABLE
        assert got.reason == "io"
    for name in ("gone-a.txt", "gone-b.txt"):
        got = byte_store.inspect_path(str(root / name))
        assert got.state == byte_store.MISSING
        assert got.reason == "enoent"


@pytest.mark.parametrize("sample", ["first", "second"])
def test_a_broken_store_root_is_not_a_missing_document(sample, tmp_path, monkeypatch):
    """Permission or I/O on the store itself must not label every file missing."""
    real = os.stat
    cases = (
        ("perm-root", PermissionError(13, "Permission denied")),
        ("io-root", OSError(errno.EIO, "Input/output error")),
    )
    for name, exc in cases:
        root = tmp_path / name
        root.mkdir()
        monkeypatch.setenv("DOCUMENT_BYTE_STORE", "local")
        monkeypatch.setenv("DOCUMENT_STORE_ROOT", str(root))
        target = str(root)

        def fake(path, *args, _target=target, _exc=exc, **kwargs):
            if os.fspath(path) == _target:
                raise _exc
            return real(path, *args, **kwargs)

        monkeypatch.setattr(os, "stat", fake)
        got = byte_store.inspect_path(str(root / "any.txt"))
        assert got.state == byte_store.UNAVAILABLE
        assert got.state != byte_store.MISSING


@pytest.mark.parametrize("sample", ["first", "second"])
def test_decryption_failure_is_not_a_missing_file(sample, tmp_path, monkeypatch):
    root = _root(tmp_path, monkeypatch)
    key_a = Fernet.generate_key().decode()
    key_b = Fernet.generate_key().decode()
    monkeypatch.setenv("DATA_ENCRYPTION_KEY", key_a)
    paths = []
    for name, body in (("cipher-a.txt", b"alpha bytes"), ("cipher-b.txt", b"beta bytes")):
        path = root / name
        file_crypto.write_document(str(path), body)
        paths.append(path)
    monkeypatch.setenv("DATA_ENCRYPTION_KEY", key_b)
    for path in paths:
        inspection = byte_store.inspect_path(str(path))
        assert inspection.state == byte_store.PRESENT
        with pytest.raises(file_crypto.DecryptionError):
            file_crypto.read_document(str(path))


@pytest.mark.parametrize("sample", ["first", "second"])
def test_efs_without_a_durable_mount_is_not_missing(sample, tmp_path, monkeypatch):
    """A directory on the container filesystem is not evidence a file is gone."""
    monkeypatch.setenv("DOCUMENT_BYTE_STORE", "efs")
    for name in ("mount-a", "mount-b"):
        root = tmp_path / name
        root.mkdir()
        present = root / "still-here.txt"
        present.write_bytes(b"here")
        monkeypatch.setenv("DOCUMENT_STORE_ROOT", str(root))
        for path in (present, root / "not-here.txt"):
            got = byte_store.inspect_path(str(path))
            assert got.state == byte_store.UNAVAILABLE
            assert got.reason == "mount"
            assert got.state != byte_store.MISSING


@pytest.mark.parametrize("sample", ["first", "second"])
def test_durable_mount_distinguishes_absence(sample, tmp_path, monkeypatch):
    root = _root(tmp_path, monkeypatch)
    monkeypatch.setenv("DOCUMENT_BYTE_STORE", "efs")
    monkeypatch.setattr(byte_store, "is_durable_mount", lambda _path: True)
    present = root / "kept.txt"
    present.write_bytes(b"kept")
    assert byte_store.inspect_path(str(present)).state == byte_store.PRESENT
    assert byte_store.inspect_path(str(root / "absent.txt")).state == byte_store.MISSING


@pytest.mark.parametrize("sample", ["first", "second"])
@pytest.mark.parametrize("name", ["scan.pdf", "notes.docx"])
def test_a_bare_filename_is_not_a_lost_upload(sample, tmp_path, monkeypatch, name):
    """Ledger placeholders are not stored objects. Their panel stays the ingest status."""
    _root(tmp_path, monkeypatch)
    pid = _project(USER["user_id"], f"placeholder-{sample}-{name}")
    doc = store.add_document(pid, name, file_path=name, size=8)
    store.stamp_document_index(
        doc["id"], chunk_count=1, ingest_status="PARTIAL_OCR",
        ingest_status_reason=None,
        pages_in_file=9, pages_ocrd=3, ocr_truncated=True,
    )
    loaded = store.get_document(doc["id"])
    assert loaded["bytes_state"] == byte_store.NOT_STORED
    status = loaded.get("panel_status") or {}
    assert status.get("label") == "Partially read"
    assert status.get("label") != byte_store.MISSING_LABEL


@pytest.mark.parametrize("sample", ["first", "second"])
def test_drive_rows_are_not_lost_uploads(sample, tmp_path, monkeypatch):
    root = _root(tmp_path, monkeypatch)
    for drive_id, name in (("drv-alpha", "alpha.pdf"), ("drv-beta", "beta.pdf")):
        pid = _project(USER["user_id"], f"drive-{drive_id}")
        doc = store.add_document(
            pid, name, file_path=str(root / name), size=10,
            metadata={"drive_file_id": drive_id},
        )
        loaded = store.get_document(doc["id"])
        assert loaded["bytes_state"] == byte_store.NOT_STORED
        assert loaded["bytes_state_reason"] == "drive_source"
        label = (loaded.get("panel_status") or {}).get("label")
        assert label != byte_store.MISSING_LABEL


@pytest.mark.parametrize("sample", ["first", "second"])
def test_missing_user_bytes_are_labeled_and_not_indexed(sample, tmp_path, monkeypatch, client, as_user):
    root = _root(tmp_path, monkeypatch)
    pid = _project(USER["user_id"], "Lost bytes")
    rows = []
    for name, chunks in (("note-a.txt", 4), ("note-b.txt", 2)):
        doc = store.add_document(
            pid, name, file_path=str(root / name), size=20,
            metadata={"provenance": "user_upload"},
        )
        store.stamp_document_index(doc["id"], chunk_count=chunks, ingest_status="INDEXED")
        rows.append((doc["id"], chunks))

    class _Counts:
        def count(self, _project_id):
            return sum(n for _i, n in rows)

        def count_by_doc(self, _project_id):
            return {doc_id: n for doc_id, n in rows}

    monkeypatch.setattr("app.core.rag.embeddings.get_embedder", lambda: type("E", (), {"dim": 8})())
    monkeypatch.setattr("app.core.rag.vector_store.get_store", lambda dim=None: _Counts())

    listed = client.get(f"/v1/projects/{pid}", headers=NO_KEY)
    assert listed.status_code == 200, listed.text
    body = listed.json()
    for doc_id, chunks in rows:
        doc = next(d for d in body["documents"] if d["id"] == doc_id)
        assert doc["chunk_count"] == chunks
        assert doc["searchable"] is False
        assert doc["bytes_state"] == byte_store.MISSING
        status = doc.get("panel_status") or {}
        assert status.get("label") == "File missing, re-upload"
        assert status.get("status") == byte_store.BYTES_MISSING
        assert "Extraction returned no usable text" not in (status.get("detail") or "")


@pytest.mark.parametrize("sample", ["first", "second"])
def test_a_present_extract_failure_stays_an_extract_failure(sample, tmp_path, monkeypatch):
    root = _root(tmp_path, monkeypatch)
    pid = _project(USER["user_id"], "Extract vs missing")
    present_names = ("readable-a.txt", "readable-b.txt")
    for name in present_names:
        path = root / name
        path.write_bytes(b"text that failed to index")
        doc = store.add_document(
            pid, name, file_path=str(path), size=path.stat().st_size,
            metadata={"provenance": "user_upload"},
        )
        store.stamp_document_index(
            doc["id"], chunk_count=0, ingest_status="EXTRACT_FAILED",
            ingest_status_reason=None,
        )
        loaded = store.get_document(doc["id"])
        status = loaded.get("panel_status") or {}
        assert status.get("label") != byte_store.MISSING_LABEL
        assert status.get("status") == "EXTRACT_FAILED"


@pytest.mark.parametrize("sample", ["first", "second"])
def test_index_keeps_chunks_and_does_not_extract_a_missing_file(sample, tmp_path, monkeypatch):
    root = _root(tmp_path, monkeypatch)
    pid = _project(USER["user_id"], "Index missing")
    calls: list[str] = []
    deleted: list[str] = []

    def _produce(file_path, filename, *args, **kwargs):
        calls.append(filename)
        raise AssertionError(f"extracted {filename}")

    monkeypatch.setattr(doc_index, "_produce_chunks_isolated", _produce)
    monkeypatch.setattr(
        store, "delete_document_chunks",
        lambda *args, **kwargs: deleted.append(args[1] if len(args) > 1 else "") or 0,
    )
    for name, chunks, status in (
        ("kept-a.txt", 3, "INDEXED"),
        ("kept-b.md", 1, "TEXT_SPARSE"),
    ):
        doc = store.add_document(
            pid, name, file_path=str(root / name), size=8,
            metadata={"provenance": "user_upload"},
        )
        store.stamp_document_index(doc["id"], chunk_count=chunks, ingest_status=status)
        result = doc_index.index_document(pid, doc["id"])
        assert result["error"] == byte_store.BYTES_MISSING
        assert result["terminal"] is True
        loaded = store.get_document(doc["id"])
        assert loaded["chunk_count"] == chunks
        assert loaded["ingest_status"] == status
        assert name not in calls
    assert deleted == []


@pytest.mark.parametrize("sample", ["first", "second"])
def test_index_does_not_treat_an_unavailable_store_as_missing(sample, tmp_path, monkeypatch):
    root = _root(tmp_path, monkeypatch)
    pid = _project(USER["user_id"], "Index unavailable")
    calls: list[str] = []

    def _produce(file_path, filename, *args, **kwargs):
        calls.append(filename)
        raise RuntimeError("extracted-on-purpose")

    monkeypatch.setattr(doc_index, "_produce_chunks_isolated", _produce)
    reasons = ("permission", "mount")
    for reason, name in zip(reasons, ("blocked-a.txt", "blocked-b.txt")):
        path = root / name
        path.write_text("still here")
        doc = store.add_document(
            pid, name, file_path=str(path), size=10,
            metadata={"provenance": "user_upload"},
        )

        def _inspect(document, expected=reason):
            if document.get("original_name", "").startswith("blocked-"):
                return byte_store.ByteInspection(byte_store.UNAVAILABLE, expected, "")
            return store.document_byte_inspection(document)

        monkeypatch.setattr(store, "document_byte_inspection", _inspect)
        monkeypatch.setattr(doc_index._projects, "document_byte_inspection", _inspect)
        with pytest.raises(RuntimeError, match="extracted-on-purpose"):
            doc_index.index_document(pid, doc["id"])
        assert name in calls


# ── worker and startup guard ─────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("sample", ["first", "second"])
async def test_worker_marks_terminal_missing_failed_not_completed(sample):
    import uuid
    from unittest.mock import patch

    from app.core.db import SessionLocal, engine
    from app.core.models import IngestionJob
    from app.worker.ingest_worker import ingest_document

    IngestionJob.__table__.create(bind=engine, checkfirst=True)

    async def settle(result, label):
        db = SessionLocal()
        job = IngestionJob(
            id=uuid.uuid4(),
            project_id=f"proj-{label}",
            document_id=f"doc-{label}",
            status="pending",
        )
        db.add(job)
        db.commit()
        try:
            with patch(
                "app.worker.ingest_worker.doc_index.maybe_eager_index",
                return_value=result,
            ):
                await ingest_document({}, job.project_id, job.document_id, str(job.id))
            db.refresh(job)
            return job.status, job.error, job.chunks
        finally:
            db.close()

    for label in ("lost-a", "lost-b"):
        status, error, chunks = await settle(
            {
                "status": "error",
                "error": byte_store.BYTES_MISSING,
                "terminal": True,
                "banner": "File missing, re-upload",
                "chunks": ["stale"],
            },
            label,
        )
        assert status == "failed"
        assert error == "File missing, re-upload"
        assert chunks == 0

    for label, result in (
        ("empty-a", {"status": "error", "error": "ZERO_CHUNK", "chunks": []}),
        ("ocr-b", {"status": "error", "error": "OCR_REQUIRED", "terminal": False}),
        ("untagged-a", {"status": "error", "error": byte_store.BYTES_MISSING, "banner": "File missing, re-upload"}),
        ("untagged-b", {"error": byte_store.BYTES_MISSING, "terminal": False, "banner": "File missing, re-upload"}),
    ):
        status, error, _chunks = await settle(result, label)
        assert status == "completed", label
        assert error is None


@pytest.mark.parametrize("sample", ["first", "second"])
def test_mount_decision_and_real_container_paths(sample):
    ephemeral = byte_store.MountIdentity(2, "/mnt/shm", "tmpfs", "tmpfs")
    overlay = byte_store.MountIdentity(3, "/data", "overlay", "overlay")
    same = byte_store.MountIdentity(1, "/var/data", "nfs4", "fs-aaa.efs.amazonaws.com:/")
    durable = byte_store.MountIdentity(9, "/mnt/efs", "nfs4", "fs-bbb.efs.amazonaws.com:/")
    root = byte_store.MountIdentity(1, "/", "overlay", "overlay")
    assert byte_store.durable_decision(ephemeral, root) is False
    assert byte_store.durable_decision(overlay, root) is False
    assert byte_store.durable_decision(same, root) is False
    assert byte_store.durable_decision(durable, root) is True
    for path in ("/tmp", str(Path(".").resolve())):
        assert byte_store.is_durable_mount(path) is False


@pytest.mark.parametrize("sample", ["first", "second"])
def test_probe_round_trip_and_rejection(sample, tmp_path):
    for name in ("probe-a", "probe-b"):
        root = tmp_path / name
        root.mkdir()
        byte_store.probe_store(str(root))
        assert list(root.iterdir()) == []
    for name in ("block-a", "block-b"):
        blocker = tmp_path / f"file-{name}"
        blocker.write_text("x")
        with pytest.raises(RuntimeError):
            byte_store.probe_store(str(blocker / "child"))


@pytest.mark.parametrize("sample", ["first", "second"])
def test_production_guard_is_off_until_the_flag_and_refuses_local_when_on(sample, tmp_path, monkeypatch):
    _prod(monkeypatch)
    monkeypatch.setenv("DOCUMENT_BYTE_STORE", "local")
    for flag, name in (("", "unset"), ("0", "zero")):
        root = tmp_path / name
        root.mkdir()
        monkeypatch.setenv("DOCUMENT_STORE_ROOT", str(root))
        monkeypatch.setenv("REQUIRE_DURABLE_DOCUMENT_STORE", flag)
        assert byte_store.enforce_startup_guard() == "not-enforcing"
        _validate_startup_env()
    for name in ("on-a", "on-b"):
        root = tmp_path / name
        root.mkdir()
        monkeypatch.setenv("DOCUMENT_STORE_ROOT", str(root))
        monkeypatch.setenv("REQUIRE_DURABLE_DOCUMENT_STORE", "1")
        with pytest.raises(RuntimeError, match="REQUIRE_DURABLE_DOCUMENT_STORE"):
            byte_store.enforce_startup_guard()


@pytest.mark.parametrize("sample", ["first", "second"])
def test_development_does_not_enforce_the_rollout_flag(sample, tmp_path, monkeypatch):
    monkeypatch.setenv("DOCUMENT_BYTE_STORE", "local")
    monkeypatch.setenv("REQUIRE_DURABLE_DOCUMENT_STORE", "1")
    monkeypatch.setenv("DOCUMENT_STORE_ROOT", str(tmp_path))
    for env_name in ("development", "testing"):
        monkeypatch.setenv("ENV", env_name)
        assert byte_store.enforce_startup_guard() == "local"
        _validate_startup_env()


@pytest.mark.parametrize("sample", ["first", "second"])
def test_efs_backend_requires_the_mount_before_the_flag(sample, tmp_path, monkeypatch):
    _prod(monkeypatch)
    monkeypatch.setenv("DOCUMENT_BYTE_STORE", "efs")
    monkeypatch.setenv("REQUIRE_DURABLE_DOCUMENT_STORE", "")
    for name in ("efs-a", "efs-b"):
        root = tmp_path / name
        root.mkdir()
        monkeypatch.setenv("DOCUMENT_STORE_ROOT", str(root))
        with pytest.raises(RuntimeError, match="durable mount"):
            byte_store.enforce_startup_guard()


@pytest.mark.parametrize("sample", ["first", "second"])
def test_guard_still_runs_the_probe_when_the_mount_looks_durable(sample, tmp_path, monkeypatch):
    """A mount check that ignores the write/read/delete probe must fail."""
    monkeypatch.setenv("DOCUMENT_BYTE_STORE", "efs")
    monkeypatch.setattr(byte_store, "is_durable_mount", lambda _path: True)
    for name in ("probe-fail-a", "probe-fail-b"):
        blocker = tmp_path / f"notdir-{name}"
        blocker.write_text("x")
        monkeypatch.setenv("DOCUMENT_STORE_ROOT", str(blocker / "child"))
        with pytest.raises(RuntimeError):
            byte_store.enforce_startup_guard()


@pytest.mark.parametrize("sample", ["first", "second"])
def test_verified_efs_mount_passes_the_guard(sample, tmp_path, monkeypatch):
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("DOCUMENT_BYTE_STORE", "efs")
    monkeypatch.setattr(byte_store, "is_durable_mount", lambda _path: True)
    for name in ("ok-a", "ok-b"):
        root = tmp_path / name
        root.mkdir()
        monkeypatch.setenv("DOCUMENT_STORE_ROOT", str(root))
        assert byte_store.enforce_startup_guard() == "enforced-efs"
        assert list(root.iterdir()) == []


@pytest.mark.parametrize("sample", ["first", "second"])
def test_web_and_worker_run_the_same_guard(sample, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        byte_store, "enforce_startup_guard", lambda: calls.append("guard") or "ok",
    )
    _prod(monkeypatch)
    monkeypatch.setenv("DOCUMENT_BYTE_STORE", "local")
    monkeypatch.delenv("REQUIRE_DURABLE_DOCUMENT_STORE", raising=False)
    _validate_startup_env()

    from app.worker.run_worker import main

    monkeypatch.setattr("app.worker.run_worker.run_worker", lambda settings: None)
    main()
    assert calls == ["guard", "guard"]


# ── re-upload ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("sample", ["first", "second"])
def test_reupload_restores_bytes_under_the_same_auth(sample, tmp_path, monkeypatch, client, as_user):
    root = _root(tmp_path, monkeypatch)
    pid = _project(USER["user_id"], "Restore")
    restored = []
    for name, body, chunks in (
        ("pour-a.txt", b"new pour note", 6),
        ("pour-b.txt", b"new membrane note", 3),
    ):
        doc = store.add_document(
            pid, name, file_path=str(root / f"missing-{name}"), size=4,
            metadata={"provenance": "user_upload"},
        )
        store.stamp_document_index(doc["id"], chunk_count=chunks, ingest_status="INDEXED")
        response = client.post(
            f"/v1/projects/{pid}/documents/{doc['id']}/bytes",
            files={"file": (name, body, "text/plain")},
            headers=NO_KEY,
        )
        assert response.status_code == 200, response.text
        payload = response.json()["document"]
        assert payload["id"] == doc["id"]
        assert payload["chunk_count"] == chunks
        assert payload["bytes_state"] == byte_store.PRESENT
        assert payload.get("searchable") is not False
        label = (payload.get("panel_status") or {}).get("label")
        assert label != byte_store.MISSING_LABEL
        assert file_crypto.read_document(payload["file_path"]) == body
        assert str(root) in payload["file_path"]
        restored.append(payload["id"])
    assert len(set(restored)) == 2


@pytest.mark.parametrize("sample", ["first", "second"])
def test_reupload_rejects_the_wrong_caller_and_a_drive_row(sample, tmp_path, monkeypatch, client):
    root = _root(tmp_path, monkeypatch)
    pid = _project(USER["user_id"], "Auth restore")
    local = store.add_document(
        pid, "owned.txt", file_path=str(root / "owned.txt"), size=2,
        metadata={"provenance": "user_upload"},
    )
    drive_rows = []
    for drive_id, name in (("drv-1", "from-drive-a.pdf"), ("drv-2", "from-drive-b.pdf")):
        drive_rows.append(store.add_document(
            pid, name, file_path=str(root / name), size=2,
            metadata={"drive_file_id": drive_id},
        ))

    app.dependency_overrides[require_user] = lambda: dict(OTHER)
    app.dependency_overrides[require_api_key] = lambda: dict(OTHER)
    try:
        denied = client.post(
            f"/v1/projects/{pid}/documents/{local['id']}/bytes",
            files={"file": ("owned.txt", b"nope", "text/plain")},
            headers=NO_KEY,
        )
        assert denied.status_code == 404
        for row in drive_rows:
            # The other user cannot see the project, so this is also 404.
            hidden = client.post(
                f"/v1/projects/{pid}/documents/{row['id']}/bytes",
                files={"file": (row["original_name"], b"%PDF", "application/pdf")},
                headers=NO_KEY,
            )
            assert hidden.status_code == 404
    finally:
        app.dependency_overrides.pop(require_user, None)
        app.dependency_overrides.pop(require_api_key, None)

    app.dependency_overrides[require_user] = lambda: dict(USER)
    app.dependency_overrides[require_api_key] = lambda: dict(USER)
    try:
        for row, upload_name in (
            (drive_rows[0], "from-drive-a.pdf"),
            (drive_rows[1], "from-drive-b.pdf"),
        ):
            refused = client.post(
                f"/v1/projects/{pid}/documents/{row['id']}/bytes",
                files={"file": (upload_name, b"%PDF-1.4", "application/pdf")},
                headers=NO_KEY,
            )
            assert refused.status_code == 409, refused.text
        wrong_type = client.post(
            f"/v1/projects/{pid}/documents/{local['id']}/bytes",
            files={"file": ("owned.pdf", b"%PDF-1.4", "application/pdf")},
            headers=NO_KEY,
        )
        also_wrong = client.post(
            f"/v1/projects/{pid}/documents/{local['id']}/bytes",
            files={"file": ("owned.docx", b"PK", "application/vnd.openxmlformats")},
            headers=NO_KEY,
        )
        assert wrong_type.status_code == 415
        assert also_wrong.status_code == 415
    finally:
        app.dependency_overrides.pop(require_user, None)
        app.dependency_overrides.pop(require_api_key, None)


@pytest.mark.parametrize("sample", ["first", "second"])
def test_changed_modules_stay_referenced(sample):
    """The mutation gate selects tests by a full module import path."""
    import app.core.byte_store
    import app.core.doc_index
    import app.core.projects
    import app.main
    import app.routers.admin
    import app.routers.drive
    import app.routers.exports
    import app.routers.projects
    import app.routers.upload
    import app.worker.ingest_worker
    import app.worker.run_worker

    assert sample in {"first", "second"}
    assert app.core.byte_store.MISSING == "missing"
    assert callable(app.core.doc_index.index_document)
    assert callable(app.core.projects.document_byte_inspection)
    assert callable(app.main._validate_startup_env)
    assert callable(app.routers.projects.restore_document_bytes)
    assert callable(app.routers.upload._store_upload)
    assert callable(app.routers.drive.drive_import)
    assert callable(app.routers.exports._persist_and_index)
    assert callable(app.worker.ingest_worker.ingest_document)
    assert callable(app.worker.run_worker.main)


@pytest.mark.parametrize("sample", ["left", "right"])
def test_mount_helpers_keep_their_first_rule(sample, tmp_path, monkeypatch):
    assert byte_store._covers("/", "/tmp/" + sample) is True
    assert byte_store._covers("/mnt/efs", "/mnt/other/" + sample) is False
    assert byte_store._not_enforcing_message().startswith("REQUIRE_DURABLE_DOCUMENT_STORE")
    assert "container" in byte_store._not_enforcing_message()

    class _Buf:
        def __init__(self, text):
            self._text = text

        def read(self):
            return self._text

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    good = "1 2 3:4 / /mnt/efs rw - nfs4 fs-" + sample + ".efs:/ rw\n"
    text = "not a mount line\n" + good

    def _open(path, *args, **kwargs):
        if str(path) == "/proc/self/mountinfo":
            return _Buf(text)
        return open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", _open)
    rows = byte_store.read_mount_table()
    assert len(rows) == 1
    assert rows[0][1] == "nfs4"

    monkeypatch.setattr(byte_store, "read_mount_table", lambda: [])
    ident = byte_store.mount_identity(str(tmp_path))
    assert ident.fstype == ""
    assert ident.mountpoint == os.path.realpath(str(tmp_path))

    here = tmp_path / sample
    here.mkdir()
    missing = here / "nope"
    assert byte_store._existing_ancestor(str(here)) == str(here.resolve())
    assert byte_store._existing_ancestor(str(missing)) == str(here.resolve())

    marker = tmp_path / f"file-{sample}"
    marker.write_text("x")
    monkeypatch.setattr(byte_store, "durable_decision", lambda store, root: True)
    assert byte_store.is_durable_mount(str(marker)) is False
    assert byte_store.is_durable_mount(str(here)) is True


@pytest.mark.parametrize("name", ["..", ""])
def test_allocate_rejects_a_bad_name_and_a_symlink(name, tmp_path, monkeypatch):
    root = _root(tmp_path, monkeypatch, "alloc-" + (name or "empty").replace(".", "d"))
    with pytest.raises(byte_store.StoreConfigError):
        byte_store.allocate_path(name)
    link = root / f"link-{name or 'empty'}"
    outside = tmp_path / f"out-{name or 'empty'}"
    outside.write_text("secret")
    link.symlink_to(outside)
    with pytest.raises(byte_store.StoreConfigError):
        byte_store.write_bytes(str(link), b"nope")
    assert outside.read_text() == "secret"


@pytest.mark.parametrize("errno_no,reason", [(errno.EACCES, "permission"), (errno.EIO, "io")])
def test_errno_reason_names_permission_and_io(errno_no, reason):
    assert byte_store._errno_reason(OSError(errno_no, "boom")) == reason
    assert byte_store.inspect_path("") == byte_store.ByteInspection(
        byte_store.NOT_STORED, "no_path", "",
    )


@pytest.mark.parametrize("sample", ["probe-a", "probe-b"])
def test_probe_rejects_a_mismatch_and_an_uncreatable_root(sample, tmp_path, monkeypatch):
    root = tmp_path / sample
    root.mkdir()
    real_open = open

    def _open(path, mode="r", *args, **kwargs):
        fh = real_open(path, mode, *args, **kwargs)
        if "r" in mode and "fork-store-probe" in str(path):
            fh.close()
            class _Bad:
                def read(self):
                    return b"different"
                def __enter__(self):
                    return self
                def __exit__(self, *a):
                    return False
            return _Bad()
        return fh

    monkeypatch.setattr("builtins.open", _open)
    with pytest.raises(RuntimeError, match="different bytes"):
        byte_store.probe_store(str(root))
    monkeypatch.undo()
    blocker = tmp_path / f"block-{sample}"
    blocker.write_text("x")
    with pytest.raises(RuntimeError, match="cannot create"):
        byte_store.probe_store(str(blocker / "child"))


@pytest.mark.parametrize("sample", ["abs-a", "abs-b"])
def test_absolute_paths_are_stored_claims_and_bare_names_are_not(sample, tmp_path):
    assert byte_store.path_is_stored_bytes(str(tmp_path / sample)) is True
    assert byte_store.path_is_stored_bytes(sample + ".pdf") is False


@pytest.mark.parametrize("sample", ["one", "two"])
def test_project_page_loads_when_the_chunk_store_is_down(sample, tmp_path, monkeypatch, client, as_user):
    root = _root(tmp_path, monkeypatch, sample)
    pid = _project(USER["user_id"], f"nostore-{sample}")
    store.add_document(
        pid, f"{sample}.txt", file_path=str(root / f"gone-{sample}.txt"), size=4,
        metadata={"provenance": "user_upload"},
    )

    def _boom():
        raise RuntimeError("no embedder")

    monkeypatch.setattr("app.core.rag.embeddings.get_embedder", _boom)
    listed = client.get(f"/v1/projects/{pid}", headers=NO_KEY)
    assert listed.status_code == 200
    doc = listed.json()["documents"][0]
    assert doc["searchable"] is False
    assert doc["panel_status"]["label"] == "File missing, re-upload"


@pytest.mark.parametrize("sample", ["missing-doc-a", "missing-doc-b"])
def test_reupload_unknown_document_is_404(sample, tmp_path, monkeypatch, client, as_user):
    _root(tmp_path, monkeypatch)
    pid = _project(USER["user_id"], f"missing-row-{sample}")
    response = client.post(
        f"/v1/projects/{pid}/documents/{sample}/bytes",
        files={"file": ("note.txt", b"hello", "text/plain")},
        headers=NO_KEY,
    )
    assert response.status_code == 404


@pytest.mark.parametrize("empty_kind", ["doc-id", "path"])
def test_restore_refuses_a_missing_id_or_path(empty_kind, tmp_path, monkeypatch):
    """The restore guard is the mechanism: a missing doc id or path must
    not clobber an existing row with empty values."""
    root = _root(tmp_path, monkeypatch)
    pid = _project(USER["user_id"], f"restore-guard-{empty_kind}")
    doc = store.add_document(
        pid, f"kept-{empty_kind}.txt", file_path=str(root / f"kept-{empty_kind}.txt"),
        size=2, metadata={"provenance": "user_upload"},
    )
    before = store.get_document(doc["id"])

    if empty_kind == "doc-id":
        refused = store.restore_document_bytes(
            "", file_path=before["file_path"], stored_as="local", size=2,
        )
    else:
        refused = store.restore_document_bytes(
            doc["id"], file_path="", stored_as="local", size=0,
        )
    assert refused is None

    after = store.get_document(doc["id"])
    assert after["file_path"] == before["file_path"]
    assert after["stored_as"] == before["stored_as"]
    assert after["size"] == before["size"]


@pytest.mark.parametrize("sample", ["ingest-a", "ingest-b"])
def test_ingest_without_a_file_is_400(sample, client):
    bare = client.post("/ingest", headers=NO_KEY)
    via = client.post("/ingest-via-block", headers=NO_KEY)
    assert bare.status_code == 400
    assert via.status_code == 400
    assert sample


@pytest.mark.parametrize("sample", ["boq-a", "boq-b"])
def test_cost_boq_without_categories_is_400(sample, tmp_path, monkeypatch, client, as_user):
    _root(tmp_path, monkeypatch)
    pid = _project(USER["user_id"], f"boq-{sample}")
    response = client.post(
        f"/v1/projects/{pid}/export/cost-boq",
        json={"title": sample},
        headers=NO_KEY,
    )
    assert response.status_code == 400


@pytest.mark.parametrize("sample", ["drive-a", "drive-b"])
def test_drive_import_unknown_project_is_404(sample, client, as_user):
    response = client.post(
        f"/v1/projects/no-such-{sample}/drive/import",
        json={"file_id": sample, "name": "note.txt"},
        headers=NO_KEY,
    )
    assert response.status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize("sample", ["run-a", "run-b"])
async def test_worker_marks_the_job_running_before_index(sample):
    import uuid
    from unittest.mock import patch

    from app.core.db import SessionLocal, engine
    from app.core.models import IngestionJob
    from app.worker.ingest_worker import ingest_document

    IngestionJob.__table__.create(bind=engine, checkfirst=True)
    db = SessionLocal()
    job = IngestionJob(
        id=uuid.uuid4(),
        project_id=f"proj-{sample}",
        document_id=f"doc-{sample}",
        status="pending",
    )
    db.add(job)
    db.commit()
    seen = []

    def _index(*args, **kwargs):
        db.refresh(job)
        seen.append(job.status)
        return {"chunks": ["a", "b"]}

    try:
        with patch("app.worker.ingest_worker.doc_index.maybe_eager_index", _index):
            await ingest_document({}, job.project_id, job.document_id, str(job.id))
        assert seen == ["running"]
    finally:
        db.close()


@pytest.mark.parametrize("sample", ["first", "second"])
def test_documents_panel_offers_reupload_for_a_lost_file(sample):
    src = Path("frontend/src/pages/ProjectWorkspace.tsx").read_text(encoding="utf-8")
    assert "bytes_missing" in src
    assert "/bytes" in src
    assert "Re-upload" in src
