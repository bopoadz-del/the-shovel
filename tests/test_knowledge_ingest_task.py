"""Admin reference works: uploaded whole, stored by the web, indexed by the ingest task.

* The admin route has its own upload cap (``ADMIN_KNOWLEDGE_MAX_UPLOAD_MB``),
  for that route only; user caps are unchanged.
* The route streams the file to disk and hashes it block by block -- memory
  does not grow with the file -- records the row as pending and returns. It
  never extracts or indexes.
* The ingest task's knowledge phase indexes each pending document once with
  the batched whole-document pipeline (own size ceiling,
  ``ADMIN_KNOWLEDGE_PDF_MAX_MB``), stamps the final status, and does nothing
  at all when nothing is pending.

Synthetic file names, text and PDFs only.
"""
from __future__ import annotations

import hashlib
import sys
import tracemalloc
import uuid

import pytest
from fastapi.testclient import TestClient

from app.main import app

ROUTE = "/v1/admin/knowledge/documents"


@pytest.fixture
def gk(monkeypatch, tmp_path):
    """A fresh general-knowledge project id per test (shared DBs in CI)."""
    pid = f"gk_task_{uuid.uuid4().hex[:8]}"
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", pid)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from app.core.projects import init_db
    init_db()
    return pid


@pytest.fixture
def client(gk, monkeypatch):
    calls = []
    monkeypatch.setattr("app.core.doc_index.maybe_eager_index",
                        lambda *a, **k: calls.append(("eager", a)))
    monkeypatch.setattr("app.core.doc_index.index_document",
                        lambda *a, **k: calls.append(("index", a)))
    from app.dependencies import require_api_key
    app.dependency_overrides[require_api_key] = lambda: {"user_id": "admin-x", "role": "admin"}
    with TestClient(app) as c:
        c.index_calls = calls
        yield c
    app.dependency_overrides.clear()


def _post(client, name, data, ctype="text/plain"):
    return client.post(ROUTE, files={"file": (name, data, ctype)})


# ── step 3: the web route stores, it does not index ─────────────────────────


def test_upload_returns_pending_and_never_calls_the_indexer(client, gk):
    from app.core import projects as store

    body = b"Synthetic Reference Manual, Part 2: bearing pressure 150 kPa.\n" * 50
    r = _post(client, "synthetic_reference_manual.txt", body)
    assert r.status_code == 201, r.text
    assert r.json()["status"] == "pending"
    row = store.get_document(r.json()["document"]["id"])
    assert row["project_id"] == gk
    assert row["ingest_status"] == "UNVERIFIED"
    assert row["metadata"]["indexing"]["status"] == "pending"
    assert row["content_sha256"] == hashlib.sha256(body).hexdigest()
    # (The boot seed may index its own notes during app startup; the
    # uploaded document is never indexed by this request.)
    uploaded = r.json()["document"]["id"]
    assert not [c for c in client.index_calls if uploaded in c[1]]


def test_upload_hashes_while_streaming_not_by_reading_the_file_back(client, monkeypatch):
    from app.core import file_crypto

    def _whole_file_read(*_a, **_k):
        raise AssertionError("the upload read the whole stored file back into memory")

    monkeypatch.setattr(file_crypto, "read_document", _whole_file_read)
    body = b"Synthetic Standard, clause text. " * 4000
    r = _post(client, "synthetic_standard.txt", body)
    assert r.status_code == 201, r.text
    assert r.json()["document"]["content_sha256"] == hashlib.sha256(body).hexdigest()


class _SyntheticStream:
    """A file-like body of ``total`` bytes produced on demand (never held whole)."""

    def __init__(self, total: int):
        self.total = total
        self.sent = 0
        self.expected = hashlib.sha256()

    def read(self, n: int = -1) -> bytes:
        n = self.total - self.sent if n is None or n < 0 else n
        n = min(n, self.total - self.sent)
        if n <= 0:
            return b""
        block = bytes((self.sent + i) % 251 for i in range(min(n, 4096))) * (n // 4096 or 1)
        block = block[:n]
        self.sent += len(block)
        self.expected.update(block)
        return block


def test_storing_a_large_body_keeps_memory_flat(tmp_path, monkeypatch):
    from app.core import file_crypto

    monkeypatch.delenv("FILE_ENCRYPTION_KEY", raising=False)
    monkeypatch.setattr(file_crypto, "encryption_enabled", lambda: False)
    total = 48 * 1024 * 1024
    stream = _SyntheticStream(total)
    hasher = hashlib.sha256()
    tracemalloc.start()
    try:
        size = file_crypto.write_document_stream(
            str(tmp_path / "big.bin"), stream, max_bytes=total, hasher=hasher,
        )
        _cur, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert size == total
    assert hasher.hexdigest() == stream.expected.hexdigest()
    assert peak < total // 8, f"peak {peak} bytes grew with a {total}-byte body"


# ── step 2: the admin route has its own cap, for that route only ─────────────


def test_admin_knowledge_cap_applies_to_its_route_only(monkeypatch):
    from app.core import upload_limits

    monkeypatch.setenv("MAX_UPLOAD_SIZE", "4096")
    monkeypatch.setenv("MAX_DOC_UPLOAD_SIZE", "4096")
    monkeypatch.setenv("ADMIN_KNOWLEDGE_MAX_UPLOAD_MB", "3")
    assert upload_limits.request_body_limit(ROUTE) == 3 * 1024 * 1024
    assert upload_limits.request_body_limit("/v1/projects/p1/documents") == 4096
    assert upload_limits.request_body_limit() == 4096
    monkeypatch.delenv("ADMIN_KNOWLEDGE_MAX_UPLOAD_MB")
    assert upload_limits.admin_knowledge_max_bytes() == 200 * 1024 * 1024


def test_a_reference_work_above_the_user_cap_is_accepted_whole(client, monkeypatch):
    monkeypatch.setenv("MAX_DOC_UPLOAD_SIZE", "4096")
    monkeypatch.setenv("MAX_UPLOAD_SIZE", "4096")
    monkeypatch.setenv("ADMIN_KNOWLEDGE_MAX_UPLOAD_MB", "1")
    body = b"Synthetic Design Code chapter text. " * 1000  # ~36 KB
    assert _post(client, "synthetic_design_code.txt", body).status_code == 201
    monkeypatch.setenv("ADMIN_KNOWLEDGE_MAX_UPLOAD_MB", "0.01")  # ~10 KB
    assert _post(client, "synthetic_design_code_2.txt", body + b"x").status_code == 413


def test_the_early_guard_lets_the_knowledge_route_past_the_user_cap(monkeypatch):
    monkeypatch.setenv("MAX_DOC_UPLOAD_SIZE", "4096")
    monkeypatch.setenv("MAX_UPLOAD_SIZE", "4096")
    monkeypatch.setenv("ADMIN_KNOWLEDGE_MAX_UPLOAD_MB", "8")
    big = b"x" * (3 * 1024 * 1024)
    with TestClient(app) as c:
        user = c.post("/v1/projects/p-none/documents",
                      files={"file": ("synthetic.txt", big, "text/plain")})
        admin = c.post(ROUTE, files={"file": ("synthetic.txt", big, "text/plain")})
    assert user.status_code == 413
    assert admin.status_code != 413  # reaches the route (auth decides from here)


# ── step 3: the ingest task's knowledge phase ────────────────────────────────


def _pending_doc(gk, name, body: bytes, tmp_path):
    from app.core import projects as store
    from app.core.knowledge_ingest import pending_indexing_metadata
    from app.core.users import SYSTEM_USER_ID, ensure_user_exists

    ensure_user_exists(SYSTEM_USER_ID, role="admin")
    if not store.get_project(gk):
        store.create_project("General Knowledge", user_id=SYSTEM_USER_ID,
                             project_id=gk, origin="admin_drive_approved")
    path = tmp_path / f"{uuid.uuid4().hex[:6]}_{name}"
    path.write_bytes(body)
    return store.add_document(
        gk, name, path.name, str(path), len(body),
        content_sha256=hashlib.sha256(body).hexdigest(),
        metadata={"provenance": "admin_knowledge", **pending_indexing_metadata()},
    )


def test_knowledge_phase_indexes_pending_once_and_then_does_nothing(gk, tmp_path, monkeypatch):
    from app.core import doc_index, knowledge_ingest, projects as store

    doc = _pending_doc(gk, "synthetic_handbook.txt", b"Synthetic handbook text.", tmp_path)
    other = store.add_document(gk, "synthetic_seed_note.md", None, None, 0)  # not pending
    calls = []

    def _index(pid, did, *a, **k):
        calls.append((pid, did))
        return {"status": "ok", "indexed": 1, "total_chunks": 4}

    monkeypatch.setattr(doc_index, "index_document", _index)
    first = knowledge_ingest.process_pending_knowledge_documents()
    # The phase takes every pending admin-knowledge row in the database (rows
    # other tests left behind included), so assert on this test's own rows.
    assert first["pending"] >= 1 and first["indexed"] == first["pending"]
    assert (gk, doc["id"]) in calls
    calls[:] = [(gk, doc["id"])]
    indexing = store.get_document(doc["id"])["metadata"]["indexing"]
    assert indexing["status"] == "ok" and indexing["chunks"] == 4

    writes = []
    monkeypatch.setattr(store, "update_document_metadata",
                        lambda *a, **k: writes.append(a))
    second = knowledge_ingest.process_pending_knowledge_documents()
    assert second == {"pending": 0, "indexed": 0, "failed": 0}
    assert calls == [(gk, doc["id"])]  # nothing re-processed
    assert writes == []                 # nothing written
    assert other["id"] not in {c[1] for c in calls}


def test_a_failed_book_is_stamped_final_not_retried_forever(gk, tmp_path, monkeypatch):
    from app.core import doc_index, knowledge_ingest, projects as store

    doc = _pending_doc(gk, "synthetic_broken.txt", b"Synthetic text.", tmp_path)

    def _boom(*_a, **_k):
        raise RuntimeError("synthetic extractor failure")

    monkeypatch.setattr(doc_index, "index_document", _boom)
    assert knowledge_ingest.process_pending_knowledge_documents()["failed"] == 1
    assert store.get_document(doc["id"])["metadata"]["indexing"]["status"] == "error"
    assert knowledge_ingest.process_pending_knowledge_documents()["pending"] == 0


def _pdf(path, n_pages, figure_pages=()):
    import fitz

    doc = fitz.open()
    for n in range(1, n_pages + 1):
        page = doc.new_page()
        if n not in figure_pages:
            body = " ".join(f"pg{n}word{i}" for i in range(400))
            page.insert_textbox(fitz.Rect(40, 40, 560, 800), body, fontsize=9)
    doc.save(str(path))
    doc.close()


def test_a_whole_book_is_indexed_in_page_batches_with_pages_and_its_own_ceiling(
    gk, tmp_path, monkeypatch,
):
    """PDF_MAX_SIZE_MB would skip it; the knowledge ceiling does not."""
    from app.core import doc_index, knowledge_ingest, projects as store
    from app.core.rag import retriever

    pdf = tmp_path / "synthetic_code_book.pdf"
    _pdf(pdf, 7, figure_pages={3})
    doc = _pending_doc(gk, "synthetic_code_book.pdf", pdf.read_bytes(), tmp_path)
    monkeypatch.setenv("PDF_MAX_SIZE_MB", "0.001")         # live-upload ceiling: skip
    monkeypatch.setenv("ADMIN_KNOWLEDGE_PDF_MAX_MB", "50")  # whole-document ceiling
    monkeypatch.setenv("PDF_OCR_BATCH_PAGES", "2")
    from app.core import extract_isolated

    batches = []
    real_run = extract_isolated.run_isolated

    def _spy(fn, args, **kw):
        if fn is doc_index._extract_pdf_range_job:
            batches.append((args[2], args[3]))
        return real_run(fn, args, **kw)

    monkeypatch.setattr(extract_isolated, "run_isolated", _spy)
    captured = {}

    def _index_chunks(pid, did, chunks, *, pages=None):
        captured.update(chunks=list(chunks), pages=pages)
        return len(chunks)

    monkeypatch.setattr(retriever, "available", lambda: True)
    monkeypatch.setattr(retriever, "index_chunks", _index_chunks)

    tally = knowledge_ingest.process_pending_knowledge_documents()

    assert tally == {"pending": 1, "indexed": 1, "failed": 0}
    assert batches == [(0, 2), (2, 4), (4, 6), (6, 7)]  # one isolated child per batch
    assert captured["pages"] and captured["pages"][0] == 1
    assert set(p for p in captured["pages"] if p) <= {1, 2, 4, 5, 6, 7}
    assert max(p for p in captured["pages"] if p) >= 6
    row = store.get_document(doc["id"])
    assert row["metadata"]["indexing"]["status"] == "ok"
    assert row["metadata"]["indexing"]["pages_without_text"] == 1


def test_whole_document_ceiling_is_its_own_config(tmp_path, monkeypatch):
    from app.core import doc_index

    pdf = tmp_path / "synthetic_ceiling.pdf"
    _pdf(pdf, 2)
    monkeypatch.setenv("PDF_MAX_SIZE_MB", "0.001")
    monkeypatch.setenv("PDF_BATCH_MIN_MB", "0")
    _text, meta = doc_index._extract_with_meta(str(pdf), pdf.name)
    assert meta.get("skipped_too_large")  # an ordinary upload is skipped
    text, meta = doc_index._extract_with_meta(str(pdf), pdf.name, whole_document=True)
    assert "pg2word0" in text and not meta.get("skipped_too_large")
    monkeypatch.setenv("ADMIN_KNOWLEDGE_PDF_MAX_MB", "0.001")
    _text, meta = doc_index._extract_with_meta(str(pdf), pdf.name, whole_document=True)
    assert meta.get("skipped_too_large")


def test_only_admin_knowledge_rows_are_whole_documents():
    from app.core import doc_index

    assert doc_index.is_whole_document({"metadata": {"provenance": "admin_knowledge"}})
    assert not doc_index.is_whole_document({"metadata": {"provenance": "user_upload"}})
    assert not doc_index.is_whole_document({})


# ── the ingest task runs the phase every run, Drive or not ──────────────────


def test_ingest_task_runs_the_knowledge_phase_before_the_drive_check(monkeypatch, tmp_path, capsys):
    import scripts.p1b_ingest_drive_server as p1b
    from app.core import gdrive_service, knowledge_ingest
    from app.core.rag import embeddings, vector_store

    seen = []
    monkeypatch.setattr(knowledge_ingest, "process_pending_knowledge_documents",
                        lambda **k: seen.append(k) or {"pending": 1, "indexed": 1, "failed": 0})
    monkeypatch.setattr(gdrive_service, "is_configured", lambda: False)
    monkeypatch.setattr(embeddings, "reset_embedder_cache", lambda: None)
    monkeypatch.setattr(vector_store, "reset_store_cache", lambda: None)
    monkeypatch.setattr(sys, "argv", ["p1b_ingest_drive_server.py", "--tier", "1", "--resume"])
    monkeypatch.chdir(tmp_path)

    assert p1b.main() == 1  # Drive is still required for the Drive phase
    assert len(seen) == 1
    assert 'KNOWLEDGESUMMARY {"failed":0,"indexed":1,"pending":1}' in capsys.readouterr().err


def test_knowledge_counts_are_in_the_run_summary(monkeypatch, tmp_path):
    import scripts.p1b_ingest_drive_server as p1b
    from app.core import knowledge_ingest
    from tests.test_p1b_silent_exit import _run_main

    monkeypatch.setattr(knowledge_ingest, "process_pending_knowledge_documents",
                        lambda **k: {"pending": 2, "indexed": 2, "failed": 0})
    monkeypatch.setattr(
        p1b, "_ingest_file",
        lambda fm, *a, **k: (fm["_drive_path"], {"status": "ok", "rag_indexed": 3}),
    )
    out = _run_main(monkeypatch, tmp_path)
    assert out["code"] == 0
    assert out["report"]["accounting"]["knowledge_indexed"] == 2
    assert out["report"]["accounting"]["knowledge_pending"] == 2


def test_only_shard_zero_runs_the_knowledge_phase(monkeypatch):
    import scripts.p1b_ingest_drive_server as p1b
    from app.core import knowledge_ingest

    seen = []
    monkeypatch.setattr(knowledge_ingest, "process_pending_knowledge_documents",
                        lambda **k: seen.append(k) or {"pending": 0, "indexed": 0, "failed": 0})
    assert p1b.run_knowledge_phase(dry_run=False, shard_index=1)["indexed"] == 0
    assert seen == []
    p1b.run_knowledge_phase(dry_run=True, shard_index=0)
    assert seen == [{"dry_run": True, "log": p1b.log}]


def test_knowledge_phase_finds_rows_outside_its_own_configured_projects(gk, tmp_path, monkeypatch):
    """The ingest task may not carry the web task's general-knowledge config:
    a pending admin-knowledge row is processed wherever it was stored."""
    from app.core import doc_index, knowledge_ingest

    doc = _pending_doc(gk, "synthetic_code_book.txt", b"Synthetic code text.", tmp_path)
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "some_other_layer_id")
    calls = []
    monkeypatch.setattr(doc_index, "index_document",
                        lambda pid, did, *a, **k: calls.append((pid, did)) or
                        {"status": "ok", "indexed": 1, "total_chunks": 2})
    knowledge_ingest.process_pending_knowledge_documents()
    assert (gk, doc["id"]) in calls

