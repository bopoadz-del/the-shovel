"""File resolvers produce readable bytes for Drive-ingested documents.

A Drive-ingested row keeps no local copy by rule (docs/INGEST_EXCLUSION_RULE.md):
its ``file_path`` names a file that is not on this disk and the bytes live in
Drive under ``drive_file_id``. The agent's file resolver and boq_processor's
resolver accepted a name match only when ``file_path`` existed, so a tool
reported "File not found" for a document the project holds. They now go
through ``projects.materialize_document_path``.

Every test builds its own project, rows and fake Drive downloader.
"""
from __future__ import annotations

import os
import uuid

import pytest


@pytest.fixture()
def store(tmp_path, monkeypatch):
    from app.core import projects

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    downloads: list[str] = []
    drive: dict[str, bytes] = {}

    def _download(fid):
        downloads.append(fid)
        if fid in drive:
            return drive[fid], None
        return None, f"no Drive file {fid}"

    monkeypatch.setattr("app.core.gdrive_service.download_file_bytes", _download)
    monkeypatch.setattr(
        "app.core.gdrive_service.get_file_metadata", lambda fid: ({}, None),
    )
    monkeypatch.setattr(
        "app.core.gdrive_service.download_public_file_bytes",
        lambda fid: (None, f"no public Drive file {fid}"),
    )
    monkeypatch.setattr(
        "app.core.gdrive_service.find_file_id_by_exact_name",
        lambda name: (None, f"no Drive file named {name}"),
    )
    projects._test_drive = drive  # type: ignore[attr-defined]
    projects._test_downloads = downloads  # type: ignore[attr-defined]
    yield projects
    for attr in ("_test_drive", "_test_downloads"):
        if hasattr(projects, attr):
            delattr(projects, attr)


def _project(store) -> str:
    return store.create_project(f"resolver-{uuid.uuid4().hex[:8]}")["id"]


def _names(tag: str) -> dict[str, str]:
    """Three distinct file names built per test, so no name repeats across runs."""
    return {
        "local": f"{tag}-local-bill.csv",
        "drive": f"{tag}-drive-bill.csv",
        "none": f"{tag}-orphan-bill.csv",
    }


def _seed(store, tmp_path, pid: str, names: dict[str, str]) -> dict[str, bytes]:
    from app.core import file_crypto

    local_bytes = f"item,qty\n{uuid.uuid4().hex},3\n".encode()
    drive_bytes = f"item,qty\n{uuid.uuid4().hex},7\n".encode()
    drive_id = f"drive-{uuid.uuid4().hex[:10]}"

    local_path = str(tmp_path / f"stored-{uuid.uuid4().hex[:6]}.csv")
    file_crypto.write_document(local_path, local_bytes)
    store.add_document(pid, names["local"], file_path=local_path, size=len(local_bytes))
    # Drive-ingested: stale local path, bytes only on Drive.
    store.add_document(
        pid, names["drive"], file_path=str(tmp_path / "gone" / "transient.csv"), size=0,
        metadata={"source": "p1b_server_drive_reingestion", "drive_file_id": drive_id},
    )
    # Neither local bytes nor a Drive pointer.
    store.add_document(pid, names["none"], file_path=str(tmp_path / "gone" / "lost.csv"), size=0)
    store._test_drive[drive_id] = drive_bytes
    return {"local": local_bytes, "drive": drive_bytes}


def _read_plain(path: str) -> bytes:
    from app.core import file_crypto

    return file_crypto.read_document(path)


RESOLVERS = ["runtime", "boq"]


def _resolver(kind):
    if kind == "runtime":
        from app.agents.runtime import _resolve_file_path

        return _resolve_file_path
    from app.blocks.boq_processor import _resolve_via_project

    return _resolve_via_project


@pytest.mark.parametrize("kind", RESOLVERS)
def test_local_and_drive_only_documents_both_resolve(store, tmp_path, kind):
    pid = _project(store)
    names = _names(uuid.uuid4().hex[:6])
    expected = _seed(store, tmp_path, pid, names)
    resolve = _resolver(kind)

    local = resolve(pid, names["local"])
    drive = resolve(pid, names["drive"])

    assert local != names["local"] and os.path.isabs(local)
    assert _read_plain(local) == expected["local"]
    # The resolved path is the materialised copy, not the bare name or the
    # stale ledger path.
    assert drive != names["drive"] and os.path.isabs(drive)
    assert os.path.exists(drive) and "gone" not in drive
    assert _read_plain(drive) == expected["drive"]


@pytest.mark.parametrize("kind", RESOLVERS)
def test_a_document_with_neither_bytes_nor_drive_id_is_still_not_found(store, tmp_path, kind):
    pid = _project(store)
    names = _names(uuid.uuid4().hex[:6])
    _seed(store, tmp_path, pid, names)

    assert _resolver(kind)(pid, names["none"]) == names["none"]


def test_boq_processor_reports_file_not_found_for_an_unmaterialisable_row(store, tmp_path):
    """Nothing is hidden: the block's own not-found error still reaches the user."""
    import asyncio

    from app.blocks.boq_processor import BOQProcessorBlock

    pid = _project(store)
    names = _names(uuid.uuid4().hex[:6])
    _seed(store, tmp_path, pid, names)

    out = asyncio.run(BOQProcessorBlock().execute(
        {"file_path": names["none"]}, {"project_id": pid},
    ))
    text = str(out)
    assert "not found" in text.lower(), text


def test_boq_processor_reads_a_drive_only_bill(store, tmp_path):
    import asyncio

    from app.blocks.boq_processor import BOQProcessorBlock

    pid = _project(store)
    names = _names(uuid.uuid4().hex[:6])
    _seed(store, tmp_path, pid, names)

    out = asyncio.run(BOQProcessorBlock().execute(
        {"file_path": names["drive"]}, {"project_id": pid},
    ))
    text = str(out).lower()
    assert "file not found" not in text, out


def test_drive_download_happens_once_per_resolve(store, tmp_path):
    """An exact match that cannot be materialised is not retried by the substring pass."""
    from app.agents.runtime import _resolve_file_path

    pid = _project(store)
    tag = uuid.uuid4().hex[:6]
    fid = f"drive-missing-{tag}"
    name = f"{tag}-absent-on-drive.csv"
    store.add_document(
        pid, name, file_path=str(tmp_path / "gone" / "x.csv"), size=0,
        metadata={"drive_file_id": fid},
    )

    assert _resolve_file_path(pid, name) == name
    assert store._test_downloads.count(fid) == 1
