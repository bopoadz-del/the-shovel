"""P-B2: the admin reindex job and the Drive ingest proof index on the
``corpus_build`` profile, and report what ``index_document`` returned."""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

from app.core import byte_store, doc_index, gdrive_service, projects
from app.routers import admin

ADMIN = {"role": "admin"}


def _recorder(result: Dict[str, Any], seen: List[tuple]):
    def fake_index(*args: Any, **kw: Any) -> Dict[str, Any]:
        seen.append((args, kw))
        return dict(result)
    return fake_index


@pytest.mark.parametrize("force_ocr, result, want_status", [
    (True, {"status": "ok", "total_chunks": 12}, "ok"),
    (False, {"status": "error", "error": "ZERO_CHUNK"}, "error"),
])
async def test_reindex_job_runs_corpus_build_and_records_result(
    monkeypatch, force_ocr, result, want_status,
):
    seen: List[tuple] = []
    monkeypatch.setattr(doc_index, "index_document", _recorder(result, seen))
    job_id = f"job-{force_ocr}"
    monkeypatch.setitem(admin._REINDEX_JOBS, job_id, {"status": "running"})

    await admin._doc_reindex_job(job_id, "p1", "d1", "default", force_ocr)

    (args, kw), = seen
    assert args == ("p1", "d1", "default")
    assert kw == {
        "force_ocr": force_ocr, "stamp_as_indexed": True,
        "profile": doc_index.PROFILE_CORPUS_BUILD,
    }
    job = admin._REINDEX_JOBS[job_id]
    assert job["status"] == want_status
    assert "finished_at" in job


@pytest.mark.parametrize("project_id", ["missing-a", "missing-b"])
async def test_ingest_proof_refuses_unknown_project_before_any_io(monkeypatch, project_id):
    seen: List[tuple] = []
    monkeypatch.setattr(projects, "get_project", lambda pid: None)
    monkeypatch.setattr(doc_index, "index_document", _recorder({"status": "ok"}, seen))

    async def no_resolve(*a: Any, **k: Any) -> Dict[str, Any]:
        raise AssertionError("resolved a Drive file for an unknown project")

    monkeypatch.setattr(admin, "_resolve_drive_proof_file", no_resolve)
    out = await admin.admin_drive_ingest_proof(
        admin._DriveProofRequest(project_id=project_id), auth=ADMIN,
    )
    assert out == {"ok": False, "error": f"project {project_id} not found"}
    assert seen == []


@pytest.mark.parametrize("result, want_ok", [
    ({"status": "ok", "total_chunks": 9}, True),
    ({"status": "error", "error": "ZERO_CHUNK"}, False),
])
async def test_ingest_proof_indexes_on_corpus_build(tmp_path, monkeypatch, result, want_ok):
    seen: List[tuple] = []
    blob = b"%PDF-1.4 proof bytes"
    monkeypatch.setattr(projects, "get_project", lambda pid: {"id": pid})

    async def resolve(project_id: str, file_id: Any) -> Dict[str, Any]:
        return {"file_id": "drv9", "name": "vol.pdf"}

    monkeypatch.setattr(admin, "_resolve_drive_proof_file", resolve)
    monkeypatch.setattr(gdrive_service, "download_file_bytes", lambda fid: (blob, None))
    monkeypatch.setattr(byte_store, "allocate_path", lambda name: str(tmp_path / name))
    monkeypatch.setattr(byte_store, "write_bytes", lambda path, raw: None)
    monkeypatch.setattr(projects, "add_document", lambda *a, **k: {"id": "doc7"})
    monkeypatch.setattr(doc_index, "index_document", _recorder(result, seen))

    out = await admin.admin_drive_ingest_proof(
        admin._DriveProofRequest(project_id="p1"), auth=ADMIN,
    )
    (args, kw), = seen
    assert args == ("p1", "doc7")
    assert kw == {"profile": doc_index.PROFILE_CORPUS_BUILD}
    assert out["ok"] is want_ok
    assert out["document_id"] == "doc7"
    if want_ok:
        assert out["chunk_count"] == 9
    else:
        assert out["error"] == "ZERO_CHUNK"
