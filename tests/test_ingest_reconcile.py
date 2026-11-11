"""S10 ingest reconcile: source-token resume, tombstones, coverage, OCR stamp.

Planner tests are in-memory. DB tests use the projects store. Nothing
downloads from Drive and nothing runs p1b --tier 1.
"""
from __future__ import annotations

import importlib

import pytest

from app.core import ingest_status as ist
from app.core.ingest_reconcile import (
    _tally_p3_counts,
    apply_tombstones,
    coverage_truth,
    plan_as_dict,
    plan_reconcile,
    resume_source_changed,
    should_skip_resume,
    source_content_token,
    stored_source_token,
)
from app.core.ingest_status import OCR_DEGRADED, TOMBSTONED


def test_source_token_prefers_md5_over_etag():
    assert source_content_token({
        "md5Checksum": "abc",
        "etag": "\"zzz\"",
    }) == "abc"
    assert source_content_token({"etag": "\"e1\""}) == "\"e1\""
    assert source_content_token({"id": "fid"}) is None
    assert source_content_token(None) is None


def test_resume_source_changed_requires_both_tokens():
    doc = {"drive_md5": "aaa"}
    assert resume_source_changed(doc, {"md5Checksum": "bbb"}) is True
    assert resume_source_changed(doc, {"md5Checksum": "aaa"}) is False
    assert resume_source_changed({"drive_md5": None}, {"md5Checksum": "bbb"}) is False
    assert resume_source_changed(doc, {"id": "fid"}) is False


def test_stored_token_falls_back_to_metadata():
    assert stored_source_token({"metadata": {"drive_md5": "m1"}}) == "m1"


def test_skip_resume_id_only_when_tokens_match():
    pdf = {
        "id": "d1",
        "original_name": "sheet.pdf",
        "drive_md5": "oldmd5",
        "ingest_status": ist.INDEXED,
        "extractor_version": ist.EXTRACTOR_VERSION,
    }
    assert should_skip_resume(pdf, 4, {"id": "f1", "md5Checksum": "oldmd5"})
    assert not should_skip_resume(pdf, 4, {"id": "f1", "md5Checksum": "newmd5"})
    # Historical row with no stored token: do not thrash the corpus.
    bare = {**pdf, "drive_md5": None}
    assert should_skip_resume(bare, 4, {"id": "f1", "md5Checksum": "newmd5"})


def test_skip_resume_still_retries_zero_chunk():
    doc = {"original_name": "a.pdf", "drive_md5": "aaa"}
    assert not should_skip_resume(doc, 0, {"md5Checksum": "aaa"})


def test_skip_resume_settled_zero_chunk_unless_source_changed():
    doc = {
        "original_name": "photo.jpg", "drive_md5": "aaa",
        "ingest_status": "UNSUPPORTED_TYPE", "ingest_status_reason": "jpg:terminal",
    }
    assert should_skip_resume(doc, 0, {"md5Checksum": "aaa"})
    assert not should_skip_resume(doc, 0, {"md5Checksum": "bbb"})
    dwg = {
        "original_name": "sheet.dwg", "drive_md5": "aaa",
        "ingest_status": "UNSUPPORTED_TYPE", "ingest_status_reason": "dwg:recoverable",
    }
    assert should_skip_resume(dwg, 0, {"md5Checksum": "aaa"}, parseable_exts={".pdf"})
    assert not should_skip_resume(dwg, 0, {"md5Checksum": "aaa"}, parseable_exts={".dwg"})


def test_plan_source_changed_and_tombstone():
    local = [
        {
            "id": "keep",
            "original_name": "a.pdf",
            "drive_md5": "aaa",
            "ingest_status": ist.INDEXED,
            "extractor_version": ist.EXTRACTOR_VERSION,
            "chunk_count": 4,
            "metadata": {"drive_file_id": "f1"},
        },
        {
            "id": "edited",
            "original_name": "b.pdf",
            "drive_md5": "old",
            "ingest_status": ist.INDEXED,
            "extractor_version": ist.EXTRACTOR_VERSION,
            "chunk_count": 4,
            "metadata": {"drive_file_id": "f2"},
        },
        {
            "id": "gone",
            "original_name": "c.pdf",
            "drive_md5": "ccc",
            "ingest_status": ist.INDEXED,
            "chunk_count": 2,
            "metadata": {"drive_file_id": "f3"},
        },
        {
            "id": "already_dead",
            "original_name": "d.pdf",
            "ingest_status": TOMBSTONED,
            "metadata": {"drive_file_id": "f4"},
        },
    ]
    drive = [
        {"id": "f1", "md5Checksum": "aaa"},
        {"id": "f2", "md5Checksum": "NEW"},
    ]
    plan = plan_reconcile(
        local_docs=local,
        drive_files=drive,
        walk_complete=True,
    )
    assert [a.doc_id for a in plan.to_index] == ["edited"]
    assert plan.to_index[0].reason == "source_changed"
    assert [a.doc_id for a in plan.to_tombstone] == ["gone"]
    assert plan.work_count == 2


def test_plan_does_not_tombstone_on_incomplete_walk():
    local = [{
        "id": "gone",
        "ingest_status": ist.INDEXED,
        "metadata": {"drive_file_id": "f3"},
    }]
    plan = plan_reconcile(
        local_docs=local, drive_files=[], walk_complete=False,
    )
    assert plan.to_tombstone == []
    assert plan.skipped_tombstones_reason == "walk_incomplete"


def test_plan_second_pass_indexes_zero_when_healthy():
    local = [{
        "id": "keep",
        "original_name": "a.pdf",
        "drive_md5": "aaa",
        "ingest_status": ist.INDEXED,
        "extractor_version": ist.EXTRACTOR_VERSION,
        "chunk_count": 4,
        "metadata": {"drive_file_id": "f1"},
    }]
    drive = [{"id": "f1", "md5Checksum": "aaa"}]
    first = plan_reconcile(local_docs=local, drive_files=drive, walk_complete=True)
    assert first.work_count == 0
    second = plan_reconcile(local_docs=local, drive_files=drive, walk_complete=True)
    assert second.index_count == 0
    assert second.work_count == 0
    assert plan_as_dict(second)["orphans_action"] == "report_only"


def test_plan_lists_reembed_but_does_not_imply_purge():
    plan = plan_reconcile(
        local_docs=[],
        drive_files=[],
        mismatch_doc_ids=["doc-old-model"],
        orphans=[{"chunk_id": "c1", "doc_id": "missing"}],
    )
    assert plan.to_reembed[0].reason == "embedding_mismatch"
    assert plan.orphans[0]["chunk_id"] == "c1"
    dumped = plan_as_dict(plan)
    assert dumped["orphans_action"] == "report_only"


def test_with_ocr_degraded_reason_token():
    assert OCR_DEGRADED in ist.with_ocr_degraded(None)
    assert ist.with_ocr_degraded("single_window:terminal") == (
        f"single_window:terminal+{OCR_DEGRADED}"
    )
    already = f"{OCR_DEGRADED}:{ist.RECOVERABLE}"
    assert ist.with_ocr_degraded(already) == already


def _reload_projects(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    import app.core.db as db_mod
    import app.core.users as users_mod
    from app.core import projects

    importlib.reload(db_mod)
    importlib.reload(users_mod)
    users_mod._initialized = False
    pm = importlib.reload(projects)
    pm._initialized = False
    pm.init_db()
    return pm


def test_tombstone_hides_row_without_deleting(monkeypatch, tmp_path):
    pm = _reload_projects(monkeypatch, tmp_path)
    pm.create_project("Tomb", user_id="system")
    proj = pm.list_projects("system")[0]
    doc = pm.add_document(
        proj["id"], "gone.pdf", size=10,
        metadata={"drive_file_id": "fid-gone"},
        drive_md5="abc",
    )
    got = pm.tombstone_document(doc["id"])
    assert got["ingest_status"] == TOMBSTONED
    assert got["retrieval_visible"] is False
    assert got["ingest_status_reason"] == "drive_deleted"
    # Idempotent.
    again = pm.tombstone_document(doc["id"])
    assert again["ingest_status"] == TOMBSTONED
    still = pm.get_document(doc["id"])
    assert still is not None
    assert still["id"] == doc["id"]


def test_apply_tombstones_dry_run_writes_nothing(monkeypatch, tmp_path):
    pm = _reload_projects(monkeypatch, tmp_path)
    pm.create_project("Dry", user_id="system")
    proj = pm.list_projects("system")[0]
    doc = pm.add_document(proj["id"], "x.pdf", size=1)
    plan = plan_reconcile(
        local_docs=[{
            "id": doc["id"],
            "ingest_status": ist.INDEXED,
            "metadata": {"drive_file_id": "missing"},
        }],
        drive_files=[],
        walk_complete=True,
    )
    assert apply_tombstones(plan.to_tombstone, execute=False) == 0
    assert pm.get_document(doc["id"])["ingest_status"] != TOMBSTONED
    assert apply_tombstones(plan.to_tombstone, execute=True) == 1
    assert pm.get_document(doc["id"])["ingest_status"] == TOMBSTONED


def test_add_document_persists_drive_md5(monkeypatch, tmp_path):
    pm = _reload_projects(monkeypatch, tmp_path)
    pm.create_project("Md5", user_id="system")
    proj = pm.list_projects("system")[0]
    doc = pm.add_document(
        proj["id"], "a.pdf", size=1, drive_md5="deadbeef",
    )
    assert doc["drive_md5"] == "deadbeef"
    assert pm.get_document(doc["id"])["drive_md5"] == "deadbeef"
    pm.set_document_drive_md5(doc["id"], "cafebabe")
    assert pm.get_document(doc["id"])["drive_md5"] == "cafebabe"


def test_coverage_truth_queryable(monkeypatch, tmp_path):
    pm = _reload_projects(monkeypatch, tmp_path)
    pm.create_project("Cov", user_id="system")
    proj = pm.list_projects("system")[0]
    live = pm.add_document(
        proj["id"], "live.pdf", size=1,
        metadata={"drive_file_id": "f-live"},
        drive_md5="aaa",
    )
    pm.stamp_document_index(live["id"], chunk_count=4, ingest_status=ist.INDEXED)
    gone = pm.add_document(
        proj["id"], "gone.pdf", size=1,
        metadata={"drive_file_id": "f-gone"},
        drive_md5="bbb",
    )
    pm.tombstone_document(gone["id"])
    ocr = pm.add_document(proj["id"], "scan.pdf", size=1)
    pm.stamp_document_index(
        ocr["id"], chunk_count=0, ingest_status=ist.ZERO_CHUNK,
        ingest_status_reason=ist.with_ocr_degraded(None),
        stamp_extractor_version=False,
    )
    report = coverage_truth(project_id=proj["id"], expected_embedding_model="fake")
    assert report["documents_total"] == 3
    assert report["tombstoned"] == 1
    assert report["ocr_degraded"] == 1
    assert report["orphans_action"] == "report_only"
    assert report["by_status"][ist.INDEXED] == 1
    assert report["by_status"][TOMBSTONED] == 1
    assert report["office"]["by_kind"][".pdf"]["documents"] == 3
    assert report["office"]["by_kind"][".pptx"]["documents"] == 0


def test_ocr_required_stamps_ocr_degraded(monkeypatch, tmp_path):
    """OCR skip above the ceiling must land OCR_DEGRADED on the ledger."""
    from app.core import doc_index, file_crypto

    pm = _reload_projects(monkeypatch, tmp_path)
    pm.create_project("OCR", user_id="system")
    proj = pm.list_projects("system")[0]
    path = str(tmp_path / "scan.pdf")
    file_crypto.write_document(path, b"%PDF-1.4 empty")
    doc = pm.add_document(proj["id"], "scan.pdf", file_path=path, size=20)

    monkeypatch.setattr(
        doc_index, "_extract_with_meta",
        lambda *a, **k: ("cover page only", {
            "ocr_skipped_too_large": True,
            "ocr_required": True,
            "ocr_pages": 0,
            "ocr_attempts": 0,
            "empty_text_pages": 2,
        }),
    )
    monkeypatch.setattr(doc_index, "chunk_extracted_document", lambda text, **k: ["c1", "c2"])
    monkeypatch.setattr(doc_index, "_boq_chunks_for_document", lambda *a, **k: [])
    monkeypatch.setattr(doc_index, "_drawing_chunks_for_document", lambda *a, **k: [])
    monkeypatch.setattr(doc_index, "_normalize_cesmm_in_chunks", lambda chunks: chunks)

    result = doc_index.index_document(proj["id"], doc["id"])
    assert result["error"] == "OCR_REQUIRED"
    stamped = pm.get_document(doc["id"])
    assert stamped["ingest_status"] == ist.ZERO_CHUNK
    assert OCR_DEGRADED in (stamped["ingest_status_reason"] or "")


def test_gdrive_list_fields_request_md5_not_etag():
    from pathlib import Path

    src = Path("app/core/gdrive_service.py").read_text(encoding="utf-8")
    assert "md5Checksum" in src
    assert "etag" not in src.lower()

def _p3_doc(
    doc_id: str,
    *,
    name: str = "doc.pdf",
    status: str = ist.INDEXED,
    chunks: int = 0,
    has_file: bool = True,
    bytes_state: str = "present",
    superseded_by: str | None = None,
    retrieval_visible: bool = True,
    content_sha256: str | None = None,
    drive_md5: str | None = None,
) -> dict:
    return {
        "id": doc_id,
        "original_name": name,
        "ingest_status": status,
        "chunk_count": chunks,
        "has_file": has_file,
        "bytes_state": bytes_state,
        "superseded_by": superseded_by,
        "retrieval_visible": retrieval_visible,
        "content_sha256": content_sha256,
        "drive_md5": drive_md5,
    }


def test_p3_spreadsheet_docs_count_chunks_from_the_store():
    # Rule (a): two spreadsheet shapes, chunk rows come from the store map.
    docs = [
        _p3_doc("s1", name="boq.xlsx", chunks=0),
        _p3_doc("s2", name="tracker.csv", chunks=0),
        _p3_doc("p1", name="spec.pdf", chunks=9),
    ]
    store = {"s1": 40, "s2": 3, "p1": 9}
    out = _tally_p3_counts(docs, store)
    ss = out["spreadsheets"]
    assert ss["documents"] == 2
    assert ss["with_chunks"] == 2
    assert ss["chunks"] == 43
    assert ss["by_extension"][".xlsx"]["chunks"] == 40
    assert ss["by_extension"][".csv"]["chunks"] == 3


def test_p3_spreadsheet_zero_chunk_is_counted_but_not_with_chunks():
    # Rule (a): a spreadsheet row with zero chunks is still a removal candidate.
    docs = [
        _p3_doc("s1", name="empty.xls", status=ist.ZERO_CHUNK, chunks=0),
        _p3_doc("s2", name="notes.txt", chunks=4),
    ]
    out = _tally_p3_counts(docs, {"s2": 4})
    ss = out["spreadsheets"]
    assert ss["documents"] == 1
    assert ss["with_chunks"] == 0
    assert ss["chunks"] == 0
    assert ss["by_status"][ist.ZERO_CHUNK] == 1


def test_p3_superseded_and_invisible_are_counted():
    # Rule (b) part 1: superseded_by and retrieval_visible are independent.
    docs = [
        _p3_doc("d1", superseded_by="d2"),
        _p3_doc("d2", retrieval_visible=False),
        _p3_doc("d3", superseded_by="d4", retrieval_visible=False),
    ]
    out = _tally_p3_counts(docs, {})
    assert out["superseded"] == {"superseded_by_set": 2, "retrieval_invisible": 2}


def test_p3_exact_duplicate_groups_by_sha_and_md5():
    # Rule (b) part 2: two group keys, duplicate_docs = sum(count-1).
    docs = [
        _p3_doc("a1", content_sha256="shaA", drive_md5="md5A"),
        _p3_doc("a2", content_sha256="shaA", drive_md5="md5A"),
        _p3_doc("a3", content_sha256="shaA"),
        _p3_doc("b1", content_sha256="shaB"),
        _p3_doc("b2", content_sha256="shaB"),
        _p3_doc("solo", content_sha256="shaC", drive_md5="md5C"),
    ]
    out = _tally_p3_counts(docs, {})
    assert out["duplicates"]["content_sha256"] == {"groups": 2, "duplicate_docs": 3}
    assert out["duplicates"]["drive_md5"] == {"groups": 1, "duplicate_docs": 1}


def test_p3_ghosts_need_no_source_and_no_chunks():
    # Rule (c): a row is a ghost only when bytes are gone AND chunks are zero.
    docs = [
        _p3_doc("g1", has_file=False, bytes_state="not_stored", chunks=0, status=ist.EXTRACT_FAILED),
        _p3_doc("g2", has_file=False, bytes_state="missing", chunks=0, status=ist.ZERO_CHUNK),
        _p3_doc("keep1", has_file=False, bytes_state="not_stored", chunks=5),
        _p3_doc("keep2", has_file=True, bytes_state="present", chunks=0),
    ]
    out = _tally_p3_counts(docs, {"keep1": 5})
    assert out["ghosts"]["total"] == 2
    assert out["ghosts"]["by_status"] == {ist.EXTRACT_FAILED: 1, ist.ZERO_CHUNK: 1}
    assert out["ghosts"]["by_bytes_state"] == {"not_stored": 1, "missing": 1}
