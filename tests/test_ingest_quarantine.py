"""Non-PDF and non-Office bytes are quarantined before a parser opens them."""

from __future__ import annotations

import logging

import pytest

from app.core import file_crypto


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    from app.core import projects as projects_mod

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("DOC_EXTRACT_ISOLATE", "0")
    monkeypatch.delenv("DATA_ENCRYPTION_KEY", raising=False)
    monkeypatch.setattr(projects_mod, "_initialized", False)
    projects_mod.init_db()
    return tmp_path


def _index(tmp_path, name: str, payload: bytes):
    from app.core import doc_index, projects as projects_mod

    path = tmp_path / name
    file_crypto.write_document(str(path), payload)
    proj = projects_mod.create_project("quarantine")
    doc = projects_mod.add_document(
        proj["id"], name, file_path=str(path), size=path.stat().st_size,
    )
    return doc_index.index_document(proj["id"], doc["id"]), doc["id"]


def _minimal_pdf_bytes() -> bytes:
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "kept-page")
    blob = doc.tobytes()
    doc.close()
    return blob


def _fitz_calls(monkeypatch):
    import fitz

    calls: list = []
    real = fitz.open

    def wrapped(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(fitz, "open", wrapped)
    return calls


def _assert_quarantine(result, doc_id, caplog, reason, bucket, kind, banned):
    assert result.get("error") == reason
    lines = [
        r.getMessage()
        for r in caplog.records
        if reason in r.getMessage() and f"doc_id={doc_id}" in r.getMessage()
    ]
    assert len(lines) == 1
    line = lines[0]
    assert f"doc_id={doc_id}" in line
    assert f"size_bucket={bucket}" in line
    assert f"sniffed={kind}" in line
    for token in banned:
        assert token not in line


def test_zero_byte_pdf_is_quarantined_not_parsed(fresh_db, tmp_path, monkeypatch, caplog):
    calls = _fitz_calls(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="app.core.doc_index"):
        result, doc_id = _index(tmp_path, "empty.pdf", b"")
    assert calls == []
    _assert_quarantine(
        result, doc_id, caplog, "QUARANTINED_NOT_PDF", "0 B", "empty", ("empty.pdf",),
    )


@pytest.mark.parametrize("payload", [
    b"<!DOCTYPE html><html><body>placeholder</body></html>",
    b"<html><body>placeholder</body></html>",
    b"<HTML><BODY>placeholder</BODY></HTML>",
])
def test_html_placeholder_pdf_is_quarantined(fresh_db, tmp_path, monkeypatch, caplog, payload):
    calls = _fitz_calls(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="app.core.doc_index"):
        result, doc_id = _index(tmp_path, "placeholder.pdf", payload)
    assert calls == []
    _assert_quarantine(
        result, doc_id, caplog,
        "QUARANTINED_NOT_PDF", "1 B-1 KB", "html",
        ("placeholder.pdf", "placeholder"),
    )


def test_random_bytes_pdf_is_quarantined(fresh_db, tmp_path, monkeypatch, caplog):
    calls = _fitz_calls(monkeypatch)
    payload = b"\x00\x01\x02not-a-header" + (b"x" * 2000)
    with caplog.at_level(logging.WARNING, logger="app.core.doc_index"):
        result, doc_id = _index(tmp_path, "random.pdf", payload)
    assert calls == []
    _assert_quarantine(
        result, doc_id, caplog,
        "QUARANTINED_NOT_PDF", "1-100 KB", "other",
        ("random.pdf",),
    )


def test_minimal_pdf_still_parses(fresh_db, tmp_path, monkeypatch):
    calls = _fitz_calls(monkeypatch)
    result, _doc_id = _index(tmp_path, "real.pdf", _minimal_pdf_bytes())
    assert result.get("error") != "QUARANTINED_NOT_PDF"
    assert result.get("total_chunks", 0) > 0
    assert calls


def test_pdf_with_leading_junk_before_header_still_parses(fresh_db, tmp_path, monkeypatch):
    calls = _fitz_calls(monkeypatch)
    payload = b"\xff\xfejunk-prefix\n" + _minimal_pdf_bytes()
    assert b"%PDF-" in payload[:1024]
    result, _doc_id = _index(tmp_path, "prefixed.pdf", payload)
    assert result.get("error") != "QUARANTINED_NOT_PDF"
    assert result.get("total_chunks", 0) > 0
    assert calls


def test_zero_byte_docx_is_quarantined(fresh_db, tmp_path, monkeypatch, caplog):
    import docx

    calls: list = []
    real = docx.Document

    def wrapped(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(docx, "Document", wrapped)
    with caplog.at_level(logging.WARNING, logger="app.core.doc_index"):
        result, doc_id = _index(tmp_path, "empty.docx", b"")
    assert calls == []
    _assert_quarantine(
        result, doc_id, caplog,
        "QUARANTINED_NOT_OFFICE", "0 B", "empty", ("empty.docx",),
    )


def test_html_docx_is_quarantined(fresh_db, tmp_path, monkeypatch, caplog):
    with caplog.at_level(logging.WARNING, logger="app.core.doc_index"):
        result, doc_id = _index(
            tmp_path, "page.docx", b"<!DOCTYPE html><html></html>",
        )
    _assert_quarantine(
        result, doc_id, caplog,
        "QUARANTINED_NOT_OFFICE", "1 B-1 KB", "html", ("page.docx",),
    )


def test_random_xlsx_and_pptx_are_quarantined(fresh_db, tmp_path, monkeypatch, caplog):
    with caplog.at_level(logging.WARNING, logger="app.core.doc_index"):
        xlsx, xlsx_id = _index(tmp_path, "sheet.xlsx", b"not a zip")
        pptx, pptx_id = _index(tmp_path, "deck.pptx", b"\x00\x01random")
    _assert_quarantine(
        xlsx, xlsx_id, caplog,
        "QUARANTINED_NOT_OFFICE", "1 B-1 KB", "other", ("sheet.xlsx",),
    )
    _assert_quarantine(
        pptx, pptx_id, caplog,
        "QUARANTINED_NOT_OFFICE", "1 B-1 KB", "other", ("deck.pptx",),
    )


def test_real_docx_still_parses(fresh_db, tmp_path):
    import docx

    path = tmp_path / "real.docx"
    document = docx.Document()
    document.add_paragraph("kept-paragraph")
    document.save(path)
    payload = path.read_bytes()
    assert payload.startswith(b"PK\x03\x04")
    result, _doc_id = _index(tmp_path, "real.docx", payload)
    assert result.get("error") != "QUARANTINED_NOT_OFFICE"
    assert result.get("total_chunks", 0) > 0
