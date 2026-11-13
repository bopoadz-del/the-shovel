"""P-B2: ingest profiles chosen by how a document arrives, plus the status rule
that the table stages' output alone never earns INDEXED.

``user_upload`` keeps the platform-user size gates. ``corpus_build`` (the Drive
walk, the admin re-index) has no whole-file ceiling and no size-based OCR skip;
memory is bounded by isolated page batches and OCR by the page budget.
"""
from __future__ import annotations

import pytest

from app.core import doc_index
from app.core import ingest_status as ist


def _fake_range_job(calls: list):
    def _run_isolated(fn, args, fallback, **kw):
        _path, _name, start, end, remaining = args
        calls.append({"start": start, "end": end, "remaining": remaining})
        n = end - start
        return (f"text {start}-{end}", {"pages_read": n}), {}
    return _run_isolated


@pytest.mark.parametrize("size_mb", [45.0, 186.0])
def test_user_upload_skips_ocr_over_size_gate(tmp_path, monkeypatch, size_mb):
    """``ocr_size_gate=True`` (user_upload): an over-gate PDF gets OCR budget 0
    and the existing ``ocr_skipped_over_size_ceiling`` reason."""
    import app.core.extract_isolated as extract_isolated

    calls: list = []
    monkeypatch.setattr(doc_index, "_document_size_mb", lambda p: size_mb)
    monkeypatch.setattr(doc_index, "_quarantine_meta", lambda p, n: None)
    monkeypatch.setattr(doc_index, "_pdf_page_count", lambda p: 30)
    monkeypatch.setenv("PDF_OCR_MAX_SIZE_MB", "32")
    monkeypatch.setenv("PDF_OCR_PAGE_CAP", "80")
    monkeypatch.setenv("PDF_OCR_BATCH_PAGES", "20")
    monkeypatch.setattr(extract_isolated, "run_isolated", _fake_range_job(calls))

    _text, meta = doc_index._extract_pdf_batched(
        str(tmp_path / "set.pdf"), "set.pdf", max_size_mb=0.0, ocr_size_gate=True,
    )
    assert [c["remaining"] for c in calls] == [0, 0]
    assert meta["ocr_skipped_too_large"] is True
    assert meta["ocr_skip_reason"] == "ocr_skipped_over_size_ceiling"


@pytest.mark.parametrize("size_mb", [45.0, 186.0])
def test_corpus_build_never_skips_ocr_for_size(tmp_path, monkeypatch, size_mb):
    """``ocr_size_gate=False`` (corpus_build): the same PDF keeps the page
    budget, reads every batch, and carries no size-skip flag."""
    import app.core.extract_isolated as extract_isolated

    calls: list = []
    monkeypatch.setattr(doc_index, "_document_size_mb", lambda p: size_mb)
    monkeypatch.setattr(doc_index, "_quarantine_meta", lambda p, n: None)
    monkeypatch.setattr(doc_index, "_pdf_page_count", lambda p: 30)
    monkeypatch.setenv("PDF_OCR_MAX_SIZE_MB", "32")
    monkeypatch.setenv("PDF_OCR_PAGE_CAP", "80")
    monkeypatch.setenv("PDF_OCR_BATCH_PAGES", "20")
    monkeypatch.setattr(extract_isolated, "run_isolated", _fake_range_job(calls))

    _text, meta = doc_index._extract_pdf_batched(
        str(tmp_path / "set.pdf"), "set.pdf", max_size_mb=0.0, ocr_size_gate=False,
    )
    assert [(c["start"], c["end"]) for c in calls] == [(0, 20), (20, 30)]
    assert calls[0]["remaining"] == 80
    assert "ocr_skipped_too_large" not in meta
    assert meta["pages_in_file"] == 30
    assert meta["pages_read"] == 30


def _route(monkeypatch, tmp_path, name, profile):
    """Run ``_produce_chunks_isolated`` with both paths stubbed; return the path taken."""
    import app.core.extract_isolated as extract_isolated

    taken: dict = {}

    def _whole(*args, **kwargs):
        taken["whole"] = kwargs
        return ["batched"], {}

    def _single(*args):
        taken["single"] = True
        return ["per-file"], {}

    monkeypatch.setattr(doc_index, "_produce_whole_document_chunks", _whole)
    monkeypatch.setattr(doc_index, "_produce_chunks", _single)
    monkeypatch.setattr(doc_index, "_file_budget", lambda: (64, 30.0))
    monkeypatch.setattr(extract_isolated, "worth_isolating", lambda p: True)
    monkeypatch.setattr(
        extract_isolated, "run_isolated",
        lambda fn, args, fallback, **kw: (fn(*args), {}),
    )
    chunks, _meta = doc_index._produce_chunks_isolated(
        str(tmp_path / name), name, ".pdf", "p", "default", False, profile=profile,
    )
    return chunks, taken


@pytest.mark.parametrize("name", ["drawings.pdf", "volume.pdf"])
def test_whole_file_gate_refuses_upload_but_corpus_build_batches(tmp_path, monkeypatch, name):
    """A PDF above the whole-file gate but below the batch threshold: an upload
    stays on the per-file path (where the gate refuses it); the corpus build
    reads it in page batches with no ceiling and no OCR size skip."""
    monkeypatch.setattr(doc_index, "_document_size_mb", lambda p: 12.0)
    monkeypatch.setenv("PDF_BATCH_MIN_MB", "20")
    monkeypatch.setenv("PDF_MAX_SIZE_MB", "10")
    monkeypatch.setenv("PDF_OCR_MAX_SIZE_MB", "32")

    upload_chunks, upload_taken = _route(
        monkeypatch, tmp_path, name, doc_index.PROFILE_USER_UPLOAD,
    )
    assert upload_chunks == ["per-file"]
    assert "whole" not in upload_taken

    corpus_chunks, corpus_taken = _route(
        monkeypatch, tmp_path, name, doc_index.PROFILE_CORPUS_BUILD,
    )
    assert corpus_chunks == ["batched"]
    assert corpus_taken["whole"] == {"max_size_mb": 0.0, "ocr_size_gate": False}


@pytest.mark.parametrize("size_mb", [3.0, 9.5])
def test_corpus_build_leaves_ungated_pdf_on_per_file_path(tmp_path, monkeypatch, size_mb):
    """No size gate touches the PDF: the corpus build changes nothing for it."""
    monkeypatch.setattr(doc_index, "_document_size_mb", lambda p: size_mb)
    monkeypatch.setenv("PDF_BATCH_MIN_MB", "20")
    monkeypatch.setenv("PDF_MAX_SIZE_MB", "100")
    monkeypatch.setenv("PDF_OCR_MAX_SIZE_MB", "32")

    chunks, taken = _route(
        monkeypatch, tmp_path, "small.pdf", doc_index.PROFILE_CORPUS_BUILD,
    )
    assert chunks == ["per-file"]
    assert "whole" not in taken


@pytest.mark.parametrize("profile", ["corpus", "USER_UPLOAD"])
def test_unknown_profile_is_refused(profile):
    with pytest.raises(ValueError):
        doc_index._checked_profile(profile)


@pytest.mark.parametrize(
    ("meta", "expected"),
    [
        ({"skipped_too_large": True, "size_mb": 186.0}, "text_stage_skipped_over_size_ceiling"),
        ({"extract_failed": "timeout"}, "text_stage_failed"),
        ({"extract_partial": True, "pages_read": 40}, "text_stage_partial"),
        ({"pages_read": 297}, None),
        ({}, None),
    ],
)
def test_text_stage_shortfall_names_the_reason(meta, expected):
    assert doc_index._text_stage_shortfall(meta) == expected


def _stamp(monkeypatch, **kwargs):
    from app.core import projects as projects_mod

    written: dict = {}
    monkeypatch.setattr(projects_mod, "get_document", lambda doc_id: {})
    monkeypatch.setattr(projects_mod, "ingest_attempts", lambda existing, key: 0)
    monkeypatch.setattr(projects_mod, "extractor_family_proven", lambda: True)
    monkeypatch.setattr(
        projects_mod, "stamp_document_index",
        lambda doc_id, **kw: written.update(kw),
    )
    doc_index._stamp_index_ledger("doc-1", "set.pdf", 1640, **kwargs)
    return written


@pytest.mark.parametrize(
    "shortfall",
    ["text_stage_skipped_over_size_ceiling", "text_stage_partial"],
)
@pytest.mark.parametrize("stamp_as_indexed", [True, False])
def test_table_chunks_alone_never_stamp_indexed(monkeypatch, shortfall, stamp_as_indexed):
    """1,640 table chunks with a text-stage shortfall: TEXT_SPARSE with the
    reason, even on the re-index path that otherwise forces INDEXED."""
    written = _stamp(
        monkeypatch, stamp_as_indexed=stamp_as_indexed,
        rag_indexed=1640, text_stage_shortfall=shortfall,
    )
    assert written["ingest_status"] == ist.TEXT_SPARSE
    assert written["ingest_status_reason"] == shortfall
    assert written["chunk_count"] == 1640


@pytest.mark.parametrize("shortfall", [None, ""])
def test_read_document_still_stamps_indexed(monkeypatch, shortfall):
    """No shortfall: the re-index path stamps INDEXED as before."""
    written = _stamp(
        monkeypatch, stamp_as_indexed=True,
        rag_indexed=1640, text_stage_shortfall=shortfall,
    )
    assert written["ingest_status"] == ist.INDEXED


@pytest.mark.parametrize(("text_n", "table_n"), [(0, 1640), (412, 3)])
def test_stage_split_is_recorded(text_n, table_n):
    meta: dict = {}
    doc_index._record_stage_split(meta, text_n, table_n)
    assert meta == {"text_chunks": text_n, "table_chunks": table_n}


def _index_project_run(monkeypatch, tmp_path, name, profile, extract_meta, table_chunks):
    """Run ``index_project`` over one stored document; capture what it chose."""
    from app.core import projects as projects_mod
    from app.core.rag import retriever as rag_retriever

    doc = {
        "id": "doc-1", "original_name": name, "file_path": str(tmp_path / name),
        "uploaded_at": "2026-11-13T00:00:00Z", "size": 1, "bytes_state": "present",
    }
    seen: dict = {"extract": None, "stamps": [], "written": None}

    def _extract(file_path, filename, force_ocr=False, **kw):
        seen["extract"] = kw
        return "body text " * 50, dict(extract_meta)

    monkeypatch.setattr(projects_mod, "list_documents", lambda pid: [doc])
    monkeypatch.setattr(doc_index, "_load_index", lambda pid: None)
    monkeypatch.setattr(doc_index, "_write_index", lambda pid, data: seen.update(written=data))
    monkeypatch.setattr(doc_index, "_extract_with_meta", _extract)
    monkeypatch.setattr(
        doc_index, "_boq_chunks_for_document", lambda *a: [f"row {i}" for i in range(table_chunks)],
    )
    monkeypatch.setattr(doc_index, "_drawing_chunks_for_document", lambda *a: [])
    monkeypatch.setattr(doc_index, "_record_document_attributes", lambda *a: None)
    monkeypatch.setattr(rag_retriever, "available", lambda: False)
    monkeypatch.setattr(
        doc_index, "_stamp_index_ledger",
        lambda doc_id, filename, n, **kw: seen["stamps"].append((n, kw)),
    )
    result = doc_index.index_project("p", profile=profile)
    return result, seen


@pytest.mark.parametrize("size_mb", [45.0, 186.0])
def test_index_project_corpus_build_batches_a_gated_pdf(tmp_path, monkeypatch, size_mb):
    """The admin rebuild-all under corpus_build reads an over-gate PDF in page
    batches with no ceiling and no OCR size skip; as a user upload it keeps
    today's gates (plain per-file extraction)."""
    monkeypatch.setattr(doc_index, "_document_size_mb", lambda p: size_mb)
    monkeypatch.setenv("PDF_BATCH_MIN_MB", "20")
    monkeypatch.setenv("PDF_OCR_MAX_SIZE_MB", "32")

    _r, corpus = _index_project_run(
        monkeypatch, tmp_path, "set.pdf", doc_index.PROFILE_CORPUS_BUILD, {}, 0,
    )
    assert corpus["extract"] == {
        "whole_document": True, "max_size_mb": 0.0, "ocr_size_gate": False,
    }

    _r, upload = _index_project_run(
        monkeypatch, tmp_path, "set.pdf", doc_index.PROFILE_USER_UPLOAD, {}, 0,
    )
    assert upload["extract"] == {}


@pytest.mark.parametrize(("name", "size_mb"), [("small.pdf", 3.0), ("schedule.xlsx", 186.0)])
def test_index_project_corpus_build_leaves_ungated_documents_alone(
    tmp_path, monkeypatch, name, size_mb,
):
    """An ungated PDF, or any non-PDF however large, extracts as before."""
    monkeypatch.setattr(doc_index, "_document_size_mb", lambda p: size_mb)
    monkeypatch.setenv("PDF_BATCH_MIN_MB", "20")
    monkeypatch.setenv("PDF_MAX_SIZE_MB", "100")
    monkeypatch.setenv("PDF_OCR_MAX_SIZE_MB", "32")

    _r, seen = _index_project_run(
        monkeypatch, tmp_path, name, doc_index.PROFILE_CORPUS_BUILD, {}, 0,
    )
    assert seen["extract"] == {}


@pytest.mark.parametrize(
    ("extract_meta", "reason"),
    [
        ({"extract_partial": True, "extract_failed": None, "pages_read": 40},
         "text_stage_partial"),
        ({"skipped_too_large": True, "size_mb": 186.0},
         "text_stage_skipped_over_size_ceiling"),
    ],
)
def test_index_project_shortfall_keeps_chunks_and_stamps_text_sparse(
    tmp_path, monkeypatch, extract_meta, reason,
):
    """A batch that failed after earlier batches were read (or a skipped text
    stage): the document keeps its chunks, its entry names the reason, and
    the ledger gets the shortfall, never INDEXED and never ZERO_CHUNK."""
    monkeypatch.setattr(doc_index, "_document_size_mb", lambda p: 3.0)
    result, seen = _index_project_run(
        monkeypatch, tmp_path, "volume.pdf", doc_index.PROFILE_CORPUS_BUILD,
        extract_meta, 7,
    )
    assert result["indexed"] == 1
    entry = seen["written"]["documents"][0]
    assert entry["text_stage_shortfall"] == reason
    assert entry["chunk_count"] > 7
    [(n, kw)] = seen["stamps"]
    assert n == entry["chunk_count"]
    assert kw["text_stage_shortfall"] == reason
    assert not kw.get("stamp_as_indexed")


@pytest.mark.parametrize("extract_meta", [{}, {"pages_read": 297}])
def test_index_project_full_read_is_not_restamped(tmp_path, monkeypatch, extract_meta):
    """No shortfall: index_project leaves the ledger alone, as before."""
    monkeypatch.setattr(doc_index, "_document_size_mb", lambda p: 3.0)
    result, seen = _index_project_run(
        monkeypatch, tmp_path, "volume.pdf", doc_index.PROFILE_CORPUS_BUILD,
        extract_meta, 2,
    )
    assert result["indexed"] == 1
    assert "text_stage_shortfall" not in seen["written"]["documents"][0]
    assert seen["stamps"] == []


@pytest.mark.parametrize("profile", ["corpus", ""])
def test_index_project_refuses_unknown_profile(monkeypatch, profile):
    from app.core import projects as projects_mod

    listed: list = []
    monkeypatch.setattr(projects_mod, "list_documents", lambda pid: listed.append(pid) or [])
    with pytest.raises(ValueError):
        doc_index.index_project("p", profile=profile)
    assert listed == []
