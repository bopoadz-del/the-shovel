"""Guard: an oversized BOQ PDF must be refused, never OOM the worker.

``boq_processor._parse_pdf`` runs ``pdfplumber.extract_tables()`` across every
page, which loads the whole PDF into memory. On the 2 GB box a large priced
BOQ PDF OOM-killed the worker and 502'd every concurrent user. The size guard
refuses above the OCR plaintext gate (32 MB / ``PDF_OCR_BOQ_MAX_SIZE_MB`` for
BOQ-named files) with a clear pointer to the xlsx/csv path instead of
crashing. These tests pin that the guard fires above the cap, stays out of
the way below it, and does not re-refuse the live 26.9 MB priced bill at the
old 20 MB cliff.
"""
import asyncio
import os
import tempfile

from cryptography.fernet import Fernet

from app.blocks.boq_processor import BOQProcessorBlock, _boq_pdf_max_mb


def _make_pdf(size_mb: float) -> str:
    """Write a throwaway .pdf of approximately ``size_mb`` megabytes.

    Content is junk — the size guard checks bytes BEFORE pdfplumber ever
    opens the file, so the bytes never need to be a valid PDF.
    """
    fd, path = tempfile.mkstemp(suffix=".pdf")
    with os.fdopen(fd, "wb") as fh:
        fh.write(b"%PDF-1.4\n")
        fh.write(b"0" * int(size_mb * 1024 * 1024))
    return path


def test_oversize_pdf_refused_without_parsing(monkeypatch):
    """Above the 32 MB OCR plaintext gate the refuse still fires."""
    monkeypatch.delenv("BOQ_PDF_MAX_MB", raising=False)
    monkeypatch.delenv("PDF_OCR_MAX_SIZE_MB", raising=False)
    path = _make_pdf(33)
    try:
        result = asyncio.run(BOQProcessorBlock().process({"file_path": path}))
    finally:
        os.remove(path)
    assert result["status"] == "error"
    assert result.get("boq_pdf_too_large") is True
    assert "xlsx" in result["error"].lower()


def test_small_pdf_not_blocked_by_size_guard(monkeypatch):
    monkeypatch.delenv("BOQ_PDF_MAX_MB", raising=False)
    path = _make_pdf(0.001)  # ~1 KB — well under the cap
    try:
        result = asyncio.run(BOQProcessorBlock().process({"file_path": path}))
    finally:
        os.remove(path)
    # The guard must NOT trip. (Parsing junk bytes still fails downstream — a
    # "no tables" / parse error — but that must NOT be the size refusal.)
    assert not result.get("boq_pdf_too_large")


def test_priced_boq_pdf_at_27mb_is_not_skipped_at_20mb(tmp_path, monkeypatch):
    """Live priced BOQ (doc d0c00001) is ~26.9 MB plaintext / ~28 MB recorded.

    The old hardcoded 20 MB parse cap refused it and asked for xlsx/csv, even
    after the same file OCR'd under the 32 MB gate. A leftover dashboard
    ``BOQ_PDF_MAX_MB=20`` must not re-refuse it.
    """
    monkeypatch.setenv("BOQ_PDF_MAX_MB", "20")
    monkeypatch.setenv("PDF_OCR_MAX_SIZE_MB", "25")
    path = str(
        tmp_path
        / "QZ-ENG-310-0000-EXC-BOQ-CA-000007-B_Bill of Quantities (Priced).pdf"
    )
    with open(path, "wb") as fh:
        fh.write(b"%PDF-1.4\n")
        fh.write(b"0" * int(27 * 1024 * 1024))

    result = asyncio.run(BOQProcessorBlock().process({"file_path": path}))
    assert not result.get("boq_pdf_too_large"), result
    assert "too large" not in (result.get("error") or "").lower()


def test_boq_pdf_size_gate_uses_plaintext_not_ciphertext(tmp_path, monkeypatch):
    """Fernet on-disk size must not refuse a 27 MB plaintext BOQ (#449 class).

    Live d0c00001 is ~26.9 MB plaintext; encrypted at rest that is ~36 MB on
    disk — above the 32 MB OCR gate — so measuring getsize would 413 a file
    the OCR path already accepted.
    """
    monkeypatch.setenv("DATA_ENCRYPTION_KEY", Fernet.generate_key().decode())
    monkeypatch.delenv("BOQ_PDF_MAX_MB", raising=False)
    monkeypatch.setenv("PDF_OCR_MAX_SIZE_MB", "32")

    from app.core import file_crypto
    from app.core import doc_index

    path = str(
        tmp_path
        / "QZ-ENG-310-0000-EXC-BOQ-CA-000007-B_Bill of Quantities (Priced).pdf"
    )
    file_crypto.write_document(path, b"%PDF-1.4 dummy")

    monkeypatch.setattr(doc_index, "_document_size_mb", lambda p: 26.9)
    real_getsize = os.path.getsize

    def _getsize(p):
        if os.path.abspath(p) == os.path.abspath(path):
            return 38 * 1024 * 1024
        return real_getsize(p)

    monkeypatch.setattr(os.path, "getsize", _getsize)

    result = asyncio.run(BOQProcessorBlock().process({"file_path": path}))
    assert not result.get("boq_pdf_too_large"), result


def test_boq_pdf_max_mb_floors_at_ocr_gate(monkeypatch):
    """Leftover 20 / 25 env vars cannot lower the refuse below 32 MB."""
    monkeypatch.setenv("BOQ_PDF_MAX_MB", "20")
    monkeypatch.setenv("PDF_OCR_MAX_SIZE_MB", "25")
    assert _boq_pdf_max_mb("notes.pdf") >= 32
    assert _boq_pdf_max_mb(
        "QZ-ENG-310-0000-EXC-BOQ-CA-000007-B_Bill of Quantities (Priced).pdf"
    ) >= 32


def test_boq_pdf_max_mb_can_raise(monkeypatch):
    monkeypatch.setenv("BOQ_PDF_MAX_MB", "80")
    monkeypatch.delenv("PDF_OCR_MAX_SIZE_MB", raising=False)
    assert _boq_pdf_max_mb("notes.pdf") == 80.0
