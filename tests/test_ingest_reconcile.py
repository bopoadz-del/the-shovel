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
    apply_ghost_tombstones,
    apply_tombstones,
    coverage_truth,
    is_ghost,
    p3_reconcile_dry_run,
    plan_as_dict,
    plan_ghost_tombstones,
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


@pytest.mark.parametrize(
    "name,chunks",
    [("boq.xlsx", 40), ("tracker.csv", 3)],
)
def test_p3_spreadsheet_doc_counts_chunks_from_store(name, chunks):
    # Rule (a): chunk rows come from the store map, not the cached column.
    out = _tally_p3_counts([_p3_doc("s1", name=name, chunks=0)], {"s1": chunks})
    ss = out["spreadsheets"]
    assert ss["documents"] == 1
    assert ss["with_chunks"] == 1
    assert ss["chunks"] == chunks
    ext = "." + name.rsplit(".", 1)[1]
    assert ss["by_extension"][ext]["chunks"] == chunks


@pytest.mark.parametrize("name", ["empty.xls", "blank.xlsm"])
def test_p3_spreadsheet_zero_chunk_is_counted_not_with_chunks(name):
    # Rule (a): a zero-chunk spreadsheet is still a removal candidate.
    out = _tally_p3_counts(
        [_p3_doc("s1", name=name, status=ist.ZERO_CHUNK, chunks=0)], {}
    )
    ss = out["spreadsheets"]
    assert ss["documents"] == 1
    assert ss["with_chunks"] == 0
    assert ss["chunks"] == 0
    assert ss["by_status"][ist.ZERO_CHUNK] == 1


@pytest.mark.parametrize(
    "doc,expected",
    [
        (_p3_doc("d1", superseded_by="d2"),
         {"superseded_by_set": 1, "retrieval_invisible": 0}),
        (_p3_doc("d2", retrieval_visible=False),
         {"superseded_by_set": 0, "retrieval_invisible": 1}),
        (_p3_doc("d3", superseded_by="d4", retrieval_visible=False),
         {"superseded_by_set": 1, "retrieval_invisible": 1}),
    ],
)
def test_p3_superseded_and_invisible_counted(doc, expected):
    # Rule (b) part 1: superseded_by and retrieval_visible are independent.
    assert _tally_p3_counts([doc], {})["superseded"] == expected


@pytest.mark.parametrize(
    "docs,sha_expected,md5_expected",
    [
        (
            [_p3_doc("a1", content_sha256="shaA", drive_md5="md5A"),
             _p3_doc("a2", content_sha256="shaA", drive_md5="md5A")],
            {"groups": 1, "duplicate_docs": 1},
            {"groups": 1, "duplicate_docs": 1},
        ),
        (
            [_p3_doc("b1", content_sha256="shaB"),
             _p3_doc("b2", content_sha256="shaB"),
             _p3_doc("b3", content_sha256="shaB")],
            {"groups": 1, "duplicate_docs": 2},
            {"groups": 0, "duplicate_docs": 0},
        ),
    ],
)
def test_p3_exact_duplicate_groups_by_sha_and_md5(docs, sha_expected, md5_expected):
    # Rule (b) part 2: duplicate_docs = sum(count-1) per group key.
    out = _tally_p3_counts(docs, {})
    assert out["duplicates"]["content_sha256"] == sha_expected
    assert out["duplicates"]["drive_md5"] == md5_expected


@pytest.mark.parametrize(
    "doc,is_ghost",
    [
        (_p3_doc("g1", has_file=False, bytes_state="not_stored", chunks=0,
                 status=ist.EXTRACT_FAILED), True),
        (_p3_doc("g2", has_file=False, bytes_state="missing", chunks=0,
                 status=ist.ZERO_CHUNK), True),
        (_p3_doc("keep1", has_file=False, bytes_state="not_stored", chunks=5), False),
        (_p3_doc("keep2", has_file=True, bytes_state="present", chunks=0), False),
    ],
)
def test_p3_ghost_needs_no_source_and_no_chunks(doc, is_ghost):
    # Rule (c): a row is a ghost only when bytes are gone AND chunks are zero.
    out = _tally_p3_counts([doc], {})
    assert out["ghosts"]["total"] == (1 if is_ghost else 0)


@pytest.mark.parametrize(
    "has_file,chunks,expected",
    [
        (False, 0, True),
        (False, 3, False),
        (True, 0, False),
        (True, 2, False),
    ],
)
def test_is_ghost_shared_predicate(has_file, chunks, expected):
    # The dry-run tally and the tombstone planner must share one rule.
    assert is_ghost(_p3_doc("d", has_file=has_file, chunks=chunks), chunks) is expected


@pytest.mark.parametrize(
    "doc,planned",
    [
        (_p3_doc("g1", has_file=False, bytes_state="not_stored", chunks=0,
                 status=ist.EXTRACT_FAILED), True),
        (_p3_doc("g2", has_file=False, bytes_state="missing", chunks=0,
                 status=ist.ZERO_CHUNK), True),
        (_p3_doc("k1", has_file=False, bytes_state="not_stored", chunks=5), False),
        (_p3_doc("k2", has_file=True, bytes_state="present", chunks=0), False),
        (_p3_doc("k3", has_file=False, bytes_state="not_stored", chunks=0,
                 status=TOMBSTONED), False),
        ({"id": "", "ingest_status": ist.EXTRACT_FAILED, "chunk_count": 0,
          "has_file": False}, False),
    ],
)
def test_plan_ghost_tombstones_selects_only_ghosts(doc, planned):
    actions = plan_ghost_tombstones([doc], {})
    assert [a.doc_id for a in actions] == ([doc["id"]] if planned else [])


def test_apply_ghost_tombstones_dry_run_then_execute(monkeypatch, tmp_path):
    pm = _reload_projects(monkeypatch, tmp_path)
    pm.create_project("Ghost", user_id="system")
    proj = pm.list_projects("system")[0]
    doc = pm.add_document(proj["id"], "ghost.pdf", size=1)
    pm.stamp_document_index(
        doc["id"], chunk_count=0, ingest_status=ist.EXTRACT_FAILED,
    )
    view = pm.get_document(doc["id"])
    assert view["has_file"] is False  # no bytes, no Drive pointer

    actions = plan_ghost_tombstones([view], {})
    assert [a.doc_id for a in actions] == [doc["id"]]

    assert apply_ghost_tombstones(actions, execute=False) == 0
    assert pm.get_document(doc["id"])["ingest_status"] != TOMBSTONED

    assert apply_ghost_tombstones(actions, execute=True) == 1
    got = pm.get_document(doc["id"])
    assert got["ingest_status"] == TOMBSTONED
    assert got["retrieval_visible"] is False
    assert got["ingest_status_reason"] == "ghost"


def test_ghost_tombstone_never_deletes_chunks(monkeypatch, tmp_path):
    import os

    from sqlalchemy import text
    from app.core.db import SessionLocal
    from app.core.models import rag_chunk_table_name
    from app.core.rag.vector_store import _rag_vector_namespace

    pm = _reload_projects(monkeypatch, tmp_path)
    pm.create_project("Keep", user_id="system")
    proj = pm.list_projects("system")[0]
    doc = pm.add_document(proj["id"], "keep.pdf", size=1)
    pm.stamp_document_index(doc["id"], chunk_count=2, ingest_status=ist.INDEXED)
    _seed_chunk_rows(proj["id"], doc["id"], 2)

    pm.tombstone_document_ghost(doc["id"])
    got = pm.get_document(doc["id"])
    assert got["ingest_status"] == TOMBSTONED
    assert got["retrieval_visible"] is False

    # NO-delete write: the chunk rows survive.
    table = rag_chunk_table_name(_rag_vector_namespace())
    with SessionLocal() as session:
        remaining = int(session.execute(
            text(f"SELECT COUNT(*) FROM {table} WHERE doc_id = :d"),
            {"d": doc["id"]},
        ).scalar() or 0)
    assert remaining == 2


def test_p3_ghost_tombstone_endpoint_plan_and_execute(monkeypatch, tmp_path):
    from app.routers.admin import admin_corpus_ghost_tombstone

    pm = _reload_projects(monkeypatch, tmp_path)
    pm.create_project("EG", user_id="system", project_id="egproj")
    doc = pm.add_document("egproj", "ghost.pdf", size=1)
    pm.stamp_document_index(
        doc["id"], chunk_count=0, ingest_status=ist.EXTRACT_FAILED,
    )

    plan = admin_corpus_ghost_tombstone(
        project_id="egproj", execute=False,
        auth={"user_id": "admin", "role": "admin"},
    )
    assert plan["project_id"] == "egproj"
    assert plan["dry_run"] is True
    assert plan["ghost_count"] == 1
    assert plan["applied"] == 0
    assert pm.get_document(doc["id"])["ingest_status"] != TOMBSTONED

    done = admin_corpus_ghost_tombstone(
        project_id="egproj", execute=True,
        auth={"user_id": "admin", "role": "admin"},
    )
    assert done["ghost_count"] == 1
    assert done["applied"] == 1
    assert done["dry_run"] is False
    got = pm.get_document(doc["id"])
    assert got["ingest_status"] == TOMBSTONED
    assert got["retrieval_visible"] is False


def _seed_chunk_rows(project_id: str, doc_id: str, count: int) -> None:
    """Insert ``count`` rows into the ACTIVE namespaced chunk table.

    The dry-run reads live chunk-row counts, not the cached ``chunk_count``
    column; a test must seed the store so a bypassed chunk-count query is
    observable. Reuses whatever width the namespace was already opened at.
    """
    import os
    from datetime import datetime, timezone

    import numpy as np
    from app.core.db import SessionLocal, engine
    from app.core.models import (
        EMBEDDING_DIM,
        make_rag_chunk_class,
        rag_chunk_class_for,
    )
    from app.core.rag.vector_store import ensure_table

    ns = (os.getenv("RAG_VECTOR_NAMESPACE", "v2") or "").strip()
    cls = rag_chunk_class_for(ns) or make_rag_chunk_class(ns, EMBEDDING_DIM, "test")
    ensure_table(cls.__table__, bind=engine)
    dim = int(cls.embedding_identity["dim"])
    vec = np.zeros(dim, dtype=np.float32)
    now = datetime.now(timezone.utc).isoformat()
    with SessionLocal() as session:
        for i in range(count):
            session.add(cls(
                chunk_id=f"{doc_id}-{i}",
                project_id=project_id,
                doc_id=doc_id,
                chunk_index=i,
                text=f"row {i}",
                embedding=vec,
                created_at=now,
            ))
        session.commit()


@pytest.mark.parametrize(
    "name,chunks",
    [("boq.xlsx", 4), ("tracker.csv", 2)],
)
def test_p3_reconcile_dry_run_queryable(monkeypatch, tmp_path, name, chunks):
    # The endpoint's loader is covered end-to-end on SQLite, and the project
    # filter is exercised against a second project's spreadsheet so a dropped
    # filter changes the count (kills the project-filter mutant).
    pm = _reload_projects(monkeypatch, tmp_path)
    pm.create_project("P3", user_id="system", project_id="p3proj")
    doc = pm.add_document("p3proj", name, size=10)
    # Cached column is stale (0); the store holds ``chunks`` rows. The dry-run
    # must report the store, so a bypassed chunk-count query (falling back to
    # the cached column) changes the assertion.
    pm.stamp_document_index(doc["id"], chunk_count=0, ingest_status=ist.INDEXED)
    _seed_chunk_rows("p3proj", doc["id"], chunks)
    pm.create_project("Other", user_id="system", project_id="otherproj")
    pm.add_document("otherproj", "other_boq.xlsx", size=10)

    report = p3_reconcile_dry_run(project_id="p3proj")
    assert report["project_id"] == "p3proj"
    assert report["spreadsheets"]["documents"] == 1
    assert report["spreadsheets"]["chunks"] == chunks


@pytest.mark.parametrize("name", ["boq.xlsx", "tracker.csv"])
def test_p3_dry_run_endpoint_returns_counts(monkeypatch, tmp_path, name):
    # The admin route hands back the tally dict (constant-return -> None must
    # fail), and the project filter is honoured.
    from app.routers.admin import admin_corpus_p3_dry_run

    pm = _reload_projects(monkeypatch, tmp_path)
    pm.create_project("EP", user_id="system", project_id="eproj")
    pm.add_document("eproj", name, size=10)

    result = admin_corpus_p3_dry_run(
        project_id="eproj", auth={"user_id": "admin", "role": "admin"},
    )
    assert isinstance(result, dict)
    assert result["project_id"] == "eproj"
    assert result["spreadsheets"]["documents"] == 1
