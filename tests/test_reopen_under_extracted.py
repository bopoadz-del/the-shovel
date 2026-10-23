"""Re-open large TEXT_SPARSE .docx that look under-extracted, not genuinely thin.

~57 retrieval_visible .docx sit at EXTRACTOR_VERSION + TEXT_SPARSE +
rag_indexed>0 with no rag_error. Most are honest single-window sheets.
A handful (live ~5, e.g. d0c0000d 2.3MB → 224 chars) are under-extracts:
big files, one tiny chunk. Those get the reopen sentinel so
``docx_stale_extractor_open`` selects them again.

Never null extractor_version. Never paint ingest_status. INDEXED stays
untouched. Small genuine-thin rows stay closed.
"""
from __future__ import annotations

import hashlib
import importlib
import io
from pathlib import Path

import docx

from app.core.ingest_status import EXTRACTOR_VERSION, INDEXED, TEXT_SPARSE
from app.core import ingest_status as ist


EMBED_FAILED_SENTINEL = f"{EXTRACTOR_VERSION}/embed-failed"
UNDER_EXTRACT_MIN_SIZE = 200_000
THIN_PREVIEW = "x" * 224


def _reload(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    import app.core.db as db_mod
    import app.core.users as users_mod
    from app.core import projects
    from app.core.rag.embeddings import reset_embedder_cache
    from app.core.rag.vector_store import reset_store_cache

    importlib.reload(db_mod)
    importlib.reload(users_mod)
    users_mod._initialized = False
    projects._initialized = False
    reset_store_cache()
    reset_embedder_cache()
    import app.core.doc_index as doc_index_mod
    importlib.reload(doc_index_mod)
    return importlib.reload(projects), users_mod


def _thin_docx_bytes() -> bytes:
    document = docx.Document()
    document.add_paragraph("one window only")
    buf = io.BytesIO()
    document.save(buf)
    return buf.getvalue()


def _seed_project(projects, users):
    projects.init_db()
    users.ensure_user_exists("u1")
    projects.create_project(name="P", client="C", user_id="u1")
    return projects.list_projects("u1")[0]


def _set_size(doc_id: str, size: int) -> None:
    from app.core.db import SessionLocal
    from app.core.models import Document

    with SessionLocal() as session:
        row = session.get(Document, doc_id)
        row.size = int(size)
        session.commit()


def _add_docx(
    projects,
    project_id: str,
    tmp_path: Path,
    *,
    name: str,
    ingest_status: str,
    extractor_version: str | None,
    chunk_count: int,
    size: int | None = None,
    reason: str = "single_window:terminal",
):
    dest = tmp_path / name
    raw = _thin_docx_bytes()
    dest.write_bytes(raw)
    doc = projects.add_document(
        project_id=project_id,
        original_name=name,
        stored_as=name,
        file_path=str(dest),
        size=len(raw),
        content_sha256=hashlib.sha256(name.encode()).hexdigest(),
        metadata={"drive_file_id": f"drive-{name}"},
    )
    projects.stamp_document_index(
        doc["id"],
        chunk_count=chunk_count,
        ingest_status=ingest_status,
        ingest_status_reason=reason,
        extractor_version=extractor_version,
    )
    if size is not None:
        _set_size(doc["id"], size)
    return projects.get_document(doc["id"])


def _put_ledger(project_id: str, document_id: str, filename: str, **fields):
    """Ledger entry (counts, flags) + the chunk text in the vector store --
    its only home since 0018; the selector measures text length there."""
    from app.core import doc_index
    from app.core.rag.embeddings import get_embedder
    from app.core.rag.vector_store import get_store

    chunks = list(fields.pop("chunks", [THIN_PREVIEW]))
    entry = {
        "document_id": document_id,
        "filename": filename,
        "chunk_count": len(chunks),
    }
    entry.update(fields)
    doc_index._upsert_index_entry(project_id, document_id, "document", entry)
    embedder = get_embedder()
    get_store(dim=embedder.dim).upsert_chunks(
        project_id, document_id, chunks, embedder.encode(chunks),
    )


def test_small_genuine_thin_excluded(monkeypatch, tmp_path):
    """Small TEXT_SPARSE + rag_indexed>0 stays closed — not under-extract."""
    projects, users = _reload(monkeypatch, tmp_path)
    proj = _seed_project(projects, users)
    thin = _add_docx(
        projects, proj["id"], tmp_path,
        name="genuinely_thin.docx",
        ingest_status=TEXT_SPARSE,
        extractor_version=EXTRACTOR_VERSION,
        chunk_count=1,
    )
    assert thin["size"] < UNDER_EXTRACT_MIN_SIZE
    _put_ledger(
        proj["id"], thin["id"], "genuinely_thin.docx",
        rag_indexed=1,
    )

    from scripts.reopen_embed_failed_docx import (
        main,
        select_under_extracted_docx_rows,
    )

    selected = select_under_extracted_docx_rows()
    assert selected == []
    assert main(["--under-extract"]) == 0
    after = projects.get_document(thin["id"])
    assert after["ingest_status"] == TEXT_SPARSE
    assert after["extractor_version"] == EXTRACTOR_VERSION
    assert not ist.docx_stale_extractor_open(
        after["ingest_status"],
        extension=".docx",
        extractor_version=after["extractor_version"],
    )


def test_large_under_extract_included(monkeypatch, tmp_path, capsys):
    """Large file + tiny extracted text gets the reopen sentinel."""
    projects, users = _reload(monkeypatch, tmp_path)
    proj = _seed_project(projects, users)
    under = _add_docx(
        projects, proj["id"], tmp_path,
        name="under_extracted.docx",
        ingest_status=TEXT_SPARSE,
        extractor_version=EXTRACTOR_VERSION,
        chunk_count=1,
        size=2_300_000,
    )
    _put_ledger(
        proj["id"], under["id"], "under_extracted.docx",
        rag_indexed=1,
        chunks=[THIN_PREVIEW],
    )
    assert not ist.docx_stale_extractor_open(
        under["ingest_status"],
        extension=".docx",
        extractor_version=under["extractor_version"],
    )

    from scripts.reopen_embed_failed_docx import (
        main,
        select_under_extracted_docx_rows,
    )

    selected = select_under_extracted_docx_rows()
    assert [row["id"] for row in selected] == [under["id"]]
    assert selected[0]["_text_len"] == len(THIN_PREVIEW)
    assert int(selected[0]["size"]) >= UNDER_EXTRACT_MIN_SIZE

    assert main(["--under-extract", "--dry-run"]) == 0
    dry = capsys.readouterr().out
    assert f"doc_id={under['id']}" in dry
    assert "size=2300000" in dry
    assert f"text_len={len(THIN_PREVIEW)}" in dry
    assert "count=1" in dry
    still = projects.get_document(under["id"])
    assert still["extractor_version"] == EXTRACTOR_VERSION
    assert still["ingest_status"] == TEXT_SPARSE

    assert main(["--under-extract"]) == 0
    out = capsys.readouterr().out
    after = projects.get_document(under["id"])
    assert after["id"] == under["id"]
    assert after["ingest_status"] == TEXT_SPARSE
    assert after["extractor_version"] is not None
    assert after["extractor_version"] == EMBED_FAILED_SENTINEL
    assert after["extractor_version"] != EXTRACTOR_VERSION
    assert ist.docx_stale_extractor_open(
        after["ingest_status"],
        extension=".docx",
        extractor_version=after["extractor_version"],
    )
    assert f"doc_id={under['id']}" in out
    assert "count=1" in out


def test_indexed_row_untouched_by_under_extract(monkeypatch, tmp_path):
    """INDEXED stays put even when the file is large and the preview is short."""
    projects, users = _reload(monkeypatch, tmp_path)
    proj = _seed_project(projects, users)
    indexed = _add_docx(
        projects, proj["id"], tmp_path,
        name="already_indexed.docx",
        ingest_status=INDEXED,
        extractor_version=EXTRACTOR_VERSION,
        chunk_count=12,
        size=2_300_000,
    )
    _put_ledger(
        proj["id"], indexed["id"], "already_indexed.docx",
        rag_indexed=12,
        chunks=[THIN_PREVIEW],
    )
    under = _add_docx(
        projects, proj["id"], tmp_path,
        name="under_extracted.docx",
        ingest_status=TEXT_SPARSE,
        extractor_version=EXTRACTOR_VERSION,
        chunk_count=1,
        size=200_000,
    )
    _put_ledger(
        proj["id"], under["id"], "under_extracted.docx",
        rag_indexed=1,
        chunks=[THIN_PREVIEW],
    )

    from scripts.reopen_embed_failed_docx import (
        main,
        select_under_extracted_docx_rows,
    )

    selected = select_under_extracted_docx_rows()
    assert [row["id"] for row in selected] == [under["id"]]

    assert main(["--under-extract"]) == 0
    still = projects.get_document(indexed["id"])
    assert still["ingest_status"] == INDEXED
    assert still["extractor_version"] == EXTRACTOR_VERSION
    assert still["chunk_count"] == 12
    reopened = projects.get_document(under["id"])
    assert reopened["extractor_version"] == EMBED_FAILED_SENTINEL
    assert reopened["ingest_status"] == TEXT_SPARSE
