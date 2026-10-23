"""The platform keeps no copy of an original (owner ruling R1).

Google Drive is the source of truth for admin-added project documents and the
extracted chunks are what the RAG needs (docs/INGEST_EXCLUSION_RULE.md). The
Drive ingest therefore makes no archive call, leaves no file behind, and a
cited source opens from the ledger row's Drive file id.

Replaces ``test_r2_archive_ingest_survives.py``: the R2 archive it guarded no
longer exists. Its two generic ``future_result_or_error`` guards (a worker
exception must not abort the main thread) are kept below.
"""
from __future__ import annotations

import importlib.util
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[1]
ARCHIVE_IMPORT = re.compile(r"\br2_storage\b|\bboto3\b|\bbotocore\b")


class _FakeDrive:
    def __init__(self, payload: bytes = b"%PDF-1.4 body") -> None:
        self.payload = payload
        self.calls: List[str] = []

    def download_file_bytes(self, fid):
        self.calls.append(fid)
        return self.payload, None


def _install_ingest_fakes(monkeypatch, added: List[Dict[str, Any]], seen: List[Dict[str, Any]]):
    rows: Dict[str, Dict[str, Any]] = {}

    def _add_document(**kw):
        doc = {"id": f"doc-{len(added) + 1}", **kw}
        added.append(kw)
        rows[doc["id"]] = doc
        return doc

    def _index_document(pid, did, **_k):
        path = rows[did]["file_path"]
        seen.append({"doc_id": did, "path": path, "bytes": Path(path).read_bytes()})
        return {"status": "ok", "rag_indexed": 2}

    def _set_file_path(did, path):
        rows.setdefault(did, {"id": did})["file_path"] = path
        return rows[did]

    monkeypatch.setattr("app.core.projects.add_document", _add_document)
    monkeypatch.setattr("app.core.projects.update_document_metadata", lambda *a, **k: None)
    monkeypatch.setattr("app.core.projects.set_document_drive_md5", lambda *a, **k: None)
    monkeypatch.setattr("app.core.projects.set_document_file_path", _set_file_path)
    monkeypatch.setattr("app.core.projects.find_document_by_sha", lambda *a, **k: None)
    monkeypatch.setattr("app.core.doc_index.index_document", _index_document)
    return rows


def _files_under(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.is_file()] if root.exists() else []


def test_no_module_provides_or_imports_an_r2_archive_client():
    assert importlib.util.find_spec("app.core.r2_storage") is None
    offenders = []
    for top in ("app", "scripts"):
        for path in (REPO / top).rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="replace")
            for n, line in enumerate(text.splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith(("import ", "from ")) and ARCHIVE_IMPORT.search(stripped):
                    offenders.append(f"{path.relative_to(REPO)}:{n}: {stripped}")
    assert offenders == [], offenders


def test_ingest_makes_no_archive_call_and_keeps_no_copy(tmp_path, monkeypatch):
    from scripts.p1b_ingest_drive_server import _ingest_file

    added: List[Dict[str, Any]] = []
    seen: List[Dict[str, Any]] = []
    _install_ingest_fakes(monkeypatch, added, seen)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    drive = _FakeDrive(b"%PDF-1.4 the original")

    _rel, result = _ingest_file(
        {"id": "drive-1", "name": "spec.pdf", "_drive_path": "Tier1/spec.pdf",
         "mimeType": "application/pdf", "size": 21},
        "proj", data_dir, "run-1", drive,
    )

    assert result.get("rag_indexed") == 2
    assert drive.calls == ["drive-1"]
    # The extractors read the downloaded bytes ...
    assert seen[0]["bytes"] == b"%PDF-1.4 the original"
    # ... from a transient copy that is gone once the file is indexed,
    assert not Path(seen[0]["path"]).exists()
    # never from the shared data volume, which stays empty.
    assert _files_under(data_dir) == []
    # The ledger row points back at Drive and carries no archive pointer.
    meta = added[0]["metadata"]
    assert meta["drive_file_id"] == "drive-1"
    assert not [k for k in meta if k.startswith("r2_")]
    assert not [k for k in result if k.startswith("r2_")]


def test_duplicate_content_discards_the_transient_copy(tmp_path, monkeypatch):
    from app.core import projects
    from scripts.p1b_ingest_drive_server import _ingest_file

    added: List[Dict[str, Any]] = []
    seen: List[Dict[str, Any]] = []
    _install_ingest_fakes(monkeypatch, added, seen)
    written: List[str] = []

    def _dup(**kw):
        written.append(kw["file_path"])
        raise projects.DuplicateContentError("doc-old", kw["content_sha256"])

    monkeypatch.setattr("app.core.projects.add_document", _dup)
    monkeypatch.setattr("app.core.projects.record_source_alias", lambda *a, **k: None)

    _, result = _ingest_file(
        {"id": "drive-2", "name": "dup.pdf", "_drive_path": "Tier1/dup.pdf",
         "mimeType": "application/pdf", "size": 13},
        "proj", tmp_path, "run-dup", _FakeDrive(),
    )
    assert result["error"] == "DUPLICATE_SHA"
    assert written and not Path(written[0]).exists()


def test_in_place_retry_indexes_the_fresh_download_not_an_old_copy(tmp_path, monkeypatch):
    """A retried row re-indexes what Drive returned now; an old local copy is never read."""
    from scripts.p1b_ingest_drive_server import _ingest_file

    added: List[Dict[str, Any]] = []
    seen: List[Dict[str, Any]] = []
    rows = _install_ingest_fakes(monkeypatch, added, seen)
    stale = tmp_path / "old_copy.pdf"
    stale.write_bytes(b"%PDF-1.4 stale bytes")
    rows["doc-9"] = {"id": "doc-9", "file_path": str(stale)}

    _ingest_file(
        {"id": "drive-9", "name": "retry.pdf", "_drive_path": "Tier1/retry.pdf",
         "mimeType": "application/pdf", "size": 14},
        "proj", tmp_path, "run-retry", _FakeDrive(b"%PDF-1.4 fresh"),
        existing_doc={"id": "doc-9"},
    )
    assert added == []
    assert seen[0]["doc_id"] == "doc-9"
    assert seen[0]["bytes"] == b"%PDF-1.4 fresh"
    assert not Path(seen[0]["path"]).exists()


def test_memory_admission_counts_one_plaintext_copy(tmp_path, monkeypatch):
    """Nothing is encrypted or uploaded on the ingest path any more, so a file
    is charged one copy of its size on each side, whatever the encryption
    setting -- not the Fernet + archive round-trip it no longer makes."""
    from app.core import extract_isolated, file_crypto
    from scripts.p1b_ingest_drive_server import _ingest_file

    added: List[Dict[str, Any]] = []
    seen: List[Dict[str, Any]] = []
    _install_ingest_fakes(monkeypatch, added, seen)
    monkeypatch.setattr(file_crypto, "encryption_enabled", lambda: True)
    demands: List[tuple] = []
    real_demand = extract_isolated.memory_demand
    monkeypatch.setattr(
        extract_isolated, "memory_demand",
        lambda size: demands.append(real_demand(size)) or demands[-1],
    )
    _ingest_file(
        {"id": "drive-3", "name": "a.pdf", "_drive_path": "Tier1/a.pdf",
         "mimeType": "application/pdf", "size": 13},
        "proj", tmp_path, "run-adm", _FakeDrive(),
    )
    assert demands == [(13, 13)]


def test_cited_source_opens_from_the_rows_drive_file_id(tmp_path, monkeypatch):
    """A row whose local path is gone resolves to Drive via its drive_file_id."""
    from app.core import projects

    calls: List[str] = []
    monkeypatch.setattr(
        "app.core.gdrive_service.download_file_bytes",
        lambda fid: calls.append(fid) or (b"%PDF-1.4 from drive", None),
    )
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    doc = {
        "id": "doc-c",
        "project_id": "proj",
        "original_name": "cited.pdf",
        "file_path": str(tmp_path / "gone.pdf"),
        "metadata": {"drive_file_id": "drive-cited"},
    }

    path, status = projects.materialize_document_file(doc)

    assert calls == ["drive-cited"]
    assert status == "ok"
    assert Path(path).read_bytes() == b"%PDF-1.4 from drive"


def test_future_result_or_error_contains_a_worker_exception():
    """Bare fut.result() on the main thread was the #477/#480 process abort."""
    from scripts.p1b_ingest_drive_server import future_result_or_error

    def _boom():
        raise PermissionError("Access Denied")

    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(_boom)
        rel, result = future_result_or_error(
            fut, {"name": "x.pdf", "_drive_path": "Tier1/x.pdf"},
        )
    assert rel == "Tier1/x.pdf"
    assert result["status"] == "error"
    assert "Access Denied" in result["error"]


def test_future_result_or_error_returns_ok_tuple():
    from scripts.p1b_ingest_drive_server import future_result_or_error

    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(lambda: ("Tier1/ok.pdf", {"status": "ok", "rag_indexed": 2}))
        rel, result = future_result_or_error(fut, {"name": "ok.pdf"})
    assert rel == "Tier1/ok.pdf"
    assert result["rag_indexed"] == 2
