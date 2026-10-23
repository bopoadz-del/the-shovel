"""An out-of-memory extraction must never be reported as an empty document.

Regression test for the second half of the isolation outage.

``RLIMIT_AS`` on the extraction child worked exactly as designed -- it stopped
the box being SIGKILLed (measured 2026-08-23: peak 2554 MB against the old
3892 MB-then-death). What did not work was the REPORTING. Only allocations
made through CPython's allocator raise ``MemoryError``; PyMuPDF calls MuPDF's
own ``calloc`` and raises ``RuntimeError`` instead. The live traceback::

    RuntimeError: code=2: calloc (4104 x 1 bytes) failed   <- PyMuPDF
    MemoryError                                            <- pdfminer, in close()
    pdfplumber.utils.exceptions.PdfminerException          <- the wrapper

``except MemoryError: raise`` never matched the RuntimeError that reached the
handler, so ``except Exception: return "", {}`` absorbed it and a 738-page
specification -- with hundreds of pages already extracted -- was reported as a
*successful empty file* and indexed as ZERO_CHUNK.
"""
from __future__ import annotations

import pytest

from app.core import doc_index


# The exact string MuPDF produced in production.
MUPDF_OOM = "code=2: calloc (4104 x 1 bytes) failed"


class TestClassifier:
    def test_the_mupdf_runtime_error_that_shipped(self):
        """No MemoryError anywhere in this chain -- only the text says so."""
        assert doc_index._is_memory_exhaustion(RuntimeError(MUPDF_OOM))

    def test_memory_error_wrapped_by_a_library_exception(self):
        """pdfplumber wraps pdfminer's MemoryError in its own type."""
        try:
            try:
                raise MemoryError()
            except MemoryError as inner:
                raise ValueError("PdfminerException") from inner
        except ValueError as exc:
            assert doc_index._is_memory_exhaustion(exc)

    def test_implicit_context_is_followed_too(self):
        """Raised DURING handling, without `from` -- __context__, not __cause__."""
        try:
            try:
                raise MemoryError()
            except MemoryError:
                raise RuntimeError("cleanup failed")
        except RuntimeError as exc:
            assert doc_index._is_memory_exhaustion(exc)

    @pytest.mark.parametrize("msg", [
        "out of memory",
        "Cannot allocate memory",
        "std::bad_alloc",
        "malloc of 8192 bytes failed",
    ])
    def test_other_allocator_dialects(self, msg):
        assert doc_index._is_memory_exhaustion(RuntimeError(msg))

    @pytest.mark.parametrize("exc", [
        ValueError("cannot open broken document"),
        RuntimeError("code=2: syntax error in object stream"),
        OSError("no such file"),
        KeyError("fonts"),
    ])
    def test_ordinary_failures_are_not_mistaken_for_oom(self, exc):
        """A corrupt PDF must still degrade to empty. If this over-matches,
        every broken file starts raising MemoryError instead of being skipped."""
        assert not doc_index._is_memory_exhaustion(exc)

    def test_a_cycle_cannot_hang_the_walk(self):
        a = RuntimeError("a")
        b = RuntimeError("b")
        a.__cause__ = b
        b.__cause__ = a
        assert doc_index._is_memory_exhaustion(a) is False


def _install_fake_pdf(monkeypatch, tmp_path, pages_before_oom: int, total: int = 50):
    """A PDF whose pages extract fine until memory runs out mid-document."""
    pdf = tmp_path / "spec.pdf"
    pdf.write_bytes(b"%PDF-1.4 " + b"0" * 4096)

    class _Page:
        def __init__(self, i): self.i = i
        def get_text(self):
            if self.i >= pages_before_oom:
                raise RuntimeError(MUPDF_OOM)
            return f"page {self.i} body text long enough to skip OCR entirely"

    class _Doc:
        def __iter__(self): return iter([_Page(i) for i in range(total)])
        def close(self): pass

    import fitz
    monkeypatch.setattr(fitz, "open", lambda *a, **k: _Doc())
    # pdfplumber is the other OOM hazard; keep it out of this test's way.
    monkeypatch.setattr(doc_index, "_pdf_tables_enabled", lambda *a, **k: False)
    return pdf


def test_partial_text_survives_an_oom(monkeypatch, tmp_path):
    """THE FIX: pages already extracted must not be thrown away.

    Before this, 40 good pages plus one failed allocation returned ("", {}).
    """
    pdf = _install_fake_pdf(monkeypatch, tmp_path, pages_before_oom=40)

    text, meta = doc_index._extract_pdf(str(pdf))

    assert meta.get("extract_oom_partial") is True, "must be flagged as partial"
    assert "page 0 body" in text and "page 39 body" in text
    assert "page 40" not in text, "extraction stopped where memory ran out"
    assert meta["extract_parts"] == 40
    assert meta["extract_chars"] == len(text) - text.count("\n")


def test_nothing_salvageable_raises_rather_than_returning_empty(monkeypatch, tmp_path):
    """With no usable pages there is nothing to keep, so the caller must be
    told it was a MEMORY failure -- not handed a clean empty document."""
    pdf = _install_fake_pdf(monkeypatch, tmp_path, pages_before_oom=0)

    with pytest.raises(MemoryError):
        doc_index._extract_pdf(str(pdf))


def test_a_corrupt_pdf_still_degrades_to_empty(monkeypatch, tmp_path):
    """Only memory failures change behaviour; ordinary breakage must not start
    raising, or one bad file aborts a whole ingest run."""
    pdf = tmp_path / "corrupt.pdf"
    pdf.write_bytes(b"%PDF-1.4\nnot a pdf")

    import fitz
    monkeypatch.setattr(doc_index, "_pdf_tables_enabled", lambda *a, **k: False)
    monkeypatch.setattr(fitz, "open", lambda *a, **k: (_ for _ in ()).throw(
        ValueError("cannot open broken document")))

    text, meta = doc_index._extract_pdf(str(pdf))
    assert text == ""
    assert meta.get("extract_failed") == "ValueError"
    assert "cannot open" in (meta.get("extract_failed_detail") or "")


def test_partial_text_records_that_ocr_was_involved(monkeypatch, tmp_path):
    """A partial document that leaned on OCR is still low-confidence text, and
    must keep saying so -- the flag drives downstream answer confidence."""
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF-1.4 " + b"0" * 4096)

    class _Page:
        def __init__(self, i): self.i = i
        def get_text(self):
            if self.i >= 3:
                raise RuntimeError(MUPDF_OOM)
            return ""          # image-only page -> forces the OCR branch

    class _Doc:
        def __iter__(self): return iter([_Page(i) for i in range(20)])
        def close(self): pass

    import fitz
    monkeypatch.setattr(fitz, "open", lambda *a, **k: _Doc())
    monkeypatch.setattr(doc_index, "_pdf_tables_enabled", lambda *a, **k: False)
    monkeypatch.setattr(doc_index, "_ocr_pdf_page", lambda page: f"ocr text {page.i}")

    text, meta = doc_index._extract_pdf(str(pdf))

    assert meta.get("extract_oom_partial") is True
    assert meta.get("ocr_low_quality") is True, "OCR'd partial text is not high-confidence"
    assert "ocr text 0" in text


def test_impl_promotes_a_native_oom_so_isolation_can_report_it(monkeypatch, tmp_path):
    """_extract_with_meta_impl is the layer the forked child actually calls.
    If it absorbs the failure, run_isolated sees a clean ("", {}) result and
    reports success -- exactly what production did.

    Deliberately NOT a PDF: _extract_pdf already raises MemoryError itself, so
    routing through it would exercise the `except MemoryError: raise` clause
    and prove nothing about this handler. Coverage caught that -- the first
    version of this test passed while these lines never ran. A non-PDF
    extractor reaches the generic `except Exception` here, which is the clause
    under test.
    """
    docx_file = tmp_path / "spec.docx"
    docx_file.write_bytes(b"PK not really a docx")

    import docx as _docx

    def _oom(*a, **k):
        raise RuntimeError("std::bad_alloc")   # native allocator, not MemoryError

    monkeypatch.setattr(_docx, "Document", _oom)

    with pytest.raises(MemoryError):
        doc_index._extract_with_meta_impl(str(docx_file), "spec.docx")


def test_impl_still_absorbs_ordinary_extractor_failures(monkeypatch, tmp_path):
    """The control for the test above: a normal extractor error must NOT be
    promoted to MemoryError, or every corrupt .docx starts aborting ingest
    runs. The failure is still named in meta so ZERO_CHUNK is not silent.
    """
    docx_file = tmp_path / "broken.docx"
    docx_file.write_bytes(b"PK not really a docx")

    import docx as _docx
    monkeypatch.setattr(_docx, "Document", lambda *a, **k: (_ for _ in ()).throw(
        ValueError("file is not a zip file")))

    text, meta = doc_index._extract_with_meta_impl(str(docx_file), "broken.docx")
    assert text == ""
    assert meta["extract_failed"] == "ValueError"
    assert "file is not a zip file" in meta["extract_failed_detail"]
