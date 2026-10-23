"""Re-open TEXT_SPARSE .docx that were false-closed after an embed failure.

The current extractor stamp (``EXTRACTOR_VERSION``) plus TEXT_SPARSE used to
mean "the extractor already tried; this sheet is genuinely thin." That is
true only when the index ledger has no ``rag_error`` (and is not
``rag_indexed == 0``). Embed-failure rows were stamped current anyway and
dropped out of ``docx_stale_extractor_open``.

This script does not crawl Drive, does not touch p1b, does not null
``extractor_version``, and does not paint ingest_status green. It writes the
sentinel ``{EXTRACTOR_VERSION}/embed-failed`` so the stale-extractor selector
re-qualifies the row.
"""
from __future__ import annotations

import hashlib
import importlib
import io
from pathlib import Path

import docx
from sqlalchemy import func, select

from app.core.ingest_status import EXTRACTOR_VERSION, INDEXED, TEXT_SPARSE
from app.core import ingest_status as ist


EMBED_FAILED_SENTINEL = f"{EXTRACTOR_VERSION}/embed-failed"
SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "reopen_embed_failed_docx.py"


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


def _add_docx(
    projects,
    project_id: str,
    tmp_path: Path,
    *,
    name: str,
    ingest_status: str,
    extractor_version: str | None,
    chunk_count: int,
    retrieval_visible: bool = True,
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
        ingest_status_reason="single_window:terminal",
        extractor_version=extractor_version,
    )
    if not retrieval_visible:
        from app.core.db import SessionLocal
        from app.core.models import Document

        with SessionLocal() as session:
            row = session.get(Document, doc["id"])
            row.retrieval_visible = False
            session.commit()
    return projects.get_document(doc["id"])


def _put_ledger(project_id: str, document_id: str, filename: str, **fields):
    from app.core import doc_index

    def mutate(current):
        current = current or {
            "project_id": project_id,
            "built_at": "",
            "documents": [],
            "skipped": [],
        }
        entry = {
            "document_id": document_id,
            "filename": filename,
            "chunks": ["one window"],
        }
        entry.update(fields)
        current["documents"] = [
            d for d in current.get("documents", [])
            if d.get("document_id") != document_id
        ]
        current["documents"].append(entry)
        return current

    doc_index._update_index(project_id, mutate)


def _document_count() -> int:
    from app.core.db import SessionLocal
    from app.core.models import Document

    with SessionLocal() as session:
        return int(session.scalar(select(func.count()).select_from(Document)) or 0)


def test_sentinel_is_not_current_extractor_version():
    """Nulling the column is forbidden; the sentinel must still re-qualify."""
    assert EMBED_FAILED_SENTINEL != EXTRACTOR_VERSION
    assert EMBED_FAILED_SENTINEL
    assert ist.docx_stale_extractor_open(
        TEXT_SPARSE,
        extension=".docx",
        extractor_version=EMBED_FAILED_SENTINEL,
    )
    assert not ist.docx_stale_extractor_open(
        TEXT_SPARSE,
        extension=".docx",
        extractor_version=EXTRACTOR_VERSION,
    )


def test_current_sparse_with_rag_error_requalifies_after_reopen(
    monkeypatch, tmp_path,
):
    """(1) current version + TEXT_SPARSE + rag_error → re-qualifies."""
    projects, users = _reload(monkeypatch, tmp_path)
    proj = _seed_project(projects, users)
    failed = _add_docx(
        projects, proj["id"], tmp_path,
        name="embed_failed.docx",
        ingest_status=TEXT_SPARSE,
        extractor_version=EXTRACTOR_VERSION,
        chunk_count=1,
    )
    _put_ledger(
        proj["id"], failed["id"], "embed_failed.docx",
        rag_error="RuntimeError: 403 Forbidden",
        rag_indexed=0,
    )
    before_count = _document_count()
    assert not ist.docx_stale_extractor_open(
        failed["ingest_status"],
        extension=".docx",
        extractor_version=failed["extractor_version"],
    )

    from scripts.reopen_embed_failed_docx import main

    assert main([]) == 0
    after = projects.get_document(failed["id"])
    assert after["id"] == failed["id"]
    assert after["ingest_status"] == TEXT_SPARSE
    assert after["extractor_version"] is not None
    assert after["extractor_version"] != EXTRACTOR_VERSION
    assert after["extractor_version"] == EMBED_FAILED_SENTINEL
    assert _document_count() == before_count
    assert ist.docx_stale_extractor_open(
        after["ingest_status"],
        extension=ist.document_extension(after),
        extractor_version=after["extractor_version"],
    )

    from scripts.reextract_stale_docx import select_stale_docx_rows

    assert [row["id"] for row in select_stale_docx_rows()] == [failed["id"]]


def test_current_sparse_without_rag_error_stays_closed(monkeypatch, tmp_path):
    """(2) current version + TEXT_SPARSE + NO rag_error → stays closed."""
    projects, users = _reload(monkeypatch, tmp_path)
    proj = _seed_project(projects, users)
    thin = _add_docx(
        projects, proj["id"], tmp_path,
        name="genuinely_thin.docx",
        ingest_status=TEXT_SPARSE,
        extractor_version=EXTRACTOR_VERSION,
        chunk_count=1,
    )
    _put_ledger(
        proj["id"], thin["id"], "genuinely_thin.docx",
        rag_indexed=1,
    )

    from scripts.reopen_embed_failed_docx import main

    assert main([]) == 0
    after = projects.get_document(thin["id"])
    assert after["ingest_status"] == TEXT_SPARSE
    assert after["extractor_version"] == EXTRACTOR_VERSION
    assert not ist.docx_stale_extractor_open(
        after["ingest_status"],
        extension=".docx",
        extractor_version=after["extractor_version"],
    )

    from scripts.reextract_stale_docx import select_stale_docx_rows

    assert select_stale_docx_rows() == []


def test_indexed_row_untouched_even_with_rag_error(monkeypatch, tmp_path):
    """(3) INDEXED untouched — status gate, not just the ledger error."""
    projects, users = _reload(monkeypatch, tmp_path)
    proj = _seed_project(projects, users)
    indexed = _add_docx(
        projects, proj["id"], tmp_path,
        name="already_indexed.docx",
        ingest_status=INDEXED,
        extractor_version=EXTRACTOR_VERSION,
        chunk_count=12,
    )
    _put_ledger(
        proj["id"], indexed["id"], "already_indexed.docx",
        rag_error="RuntimeError: 403 Forbidden",
        rag_indexed=0,
    )
    failed = _add_docx(
        projects, proj["id"], tmp_path,
        name="embed_failed.docx",
        ingest_status=TEXT_SPARSE,
        extractor_version=EXTRACTOR_VERSION,
        chunk_count=1,
    )
    _put_ledger(
        proj["id"], failed["id"], "embed_failed.docx",
        rag_error="embedded 0 of 1 chunks",
        rag_indexed=0,
    )

    from scripts.reopen_embed_failed_docx import main, select_embed_failed_docx_rows

    selected = select_embed_failed_docx_rows()
    assert [row["id"] for row in selected] == [failed["id"]]

    assert main([]) == 0
    still = projects.get_document(indexed["id"])
    assert still["ingest_status"] == INDEXED
    assert still["extractor_version"] == EXTRACTOR_VERSION
    assert still["chunk_count"] == 12
    reopened = projects.get_document(failed["id"])
    assert reopened["extractor_version"] == EMBED_FAILED_SENTINEL


def test_dry_run_prints_doc_id_and_rag_error_and_writes_nothing(
    monkeypatch, tmp_path, capsys,
):
    projects, users = _reload(monkeypatch, tmp_path)
    proj = _seed_project(projects, users)
    failed = _add_docx(
        projects, proj["id"], tmp_path,
        name="embed_failed.docx",
        ingest_status=TEXT_SPARSE,
        extractor_version=EXTRACTOR_VERSION,
        chunk_count=1,
    )
    rag_error = "embedding stack unavailable"
    _put_ledger(
        proj["id"], failed["id"], "embed_failed.docx",
        rag_error=rag_error,
        rag_indexed=0,
    )
    thin = _add_docx(
        projects, proj["id"], tmp_path,
        name="genuinely_thin.docx",
        ingest_status=TEXT_SPARSE,
        extractor_version=EXTRACTOR_VERSION,
        chunk_count=1,
    )
    _put_ledger(proj["id"], thin["id"], "genuinely_thin.docx", rag_indexed=1)

    from scripts.reopen_embed_failed_docx import main

    assert main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    after_failed = projects.get_document(failed["id"])
    after_thin = projects.get_document(thin["id"])
    assert after_failed["extractor_version"] == EXTRACTOR_VERSION
    assert after_thin["extractor_version"] == EXTRACTOR_VERSION
    assert f"doc_id={failed['id']}" in out
    assert rag_error in out
    assert f"doc_id={thin['id']}" not in out
    assert "count=1" in out


def test_rag_indexed_zero_without_rag_error_is_embed_failure(
    monkeypatch, tmp_path,
):
    """Ledger ``rag_indexed == 0`` is the other discriminator."""
    projects, users = _reload(monkeypatch, tmp_path)
    proj = _seed_project(projects, users)
    failed = _add_docx(
        projects, proj["id"], tmp_path,
        name="zero_embed.docx",
        ingest_status=TEXT_SPARSE,
        extractor_version=EXTRACTOR_VERSION,
        chunk_count=1,
    )
    _put_ledger(proj["id"], failed["id"], "zero_embed.docx", rag_indexed=0)

    from scripts.reopen_embed_failed_docx import main

    assert main([]) == 0
    after = projects.get_document(failed["id"])
    assert after["extractor_version"] == EMBED_FAILED_SENTINEL
    assert ist.docx_stale_extractor_open(
        after["ingest_status"],
        extension=".docx",
        extractor_version=after["extractor_version"],
    )


def test_script_does_not_null_extractor_version_or_paint_status():
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "extractor_version = None" not in source
    assert "extractor_version=None" not in source
    assert "ingest_status" in source
    assert "INDEXED" not in source or "ingest_status" in source
    assert "p1b_ingest" not in source
    assert "walk_folder" not in source
