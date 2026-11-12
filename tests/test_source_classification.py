"""Source-file classification (SPEC_CORPUS_FINAL_BUILD_10NOV P2).

Two pure mechanisms: ``classify_source_document`` (extension + PDF text-layer
=> exactly one manifest class) and ``pdf_has_text_layer`` (sampled page spread).
Every decision is exercised with at least two inputs, per the mutation gate.
"""
from __future__ import annotations

import pytest

from app.core import ingest_status as ist


def _write_text_pdf(path, pages: list[str | None]) -> None:
    """Create a real PDF at ``path``. A page entry of ``None`` is image-only
    (no text layer); a non-empty string is inserted as that page's text layer."""
    import fitz

    doc = fitz.open()
    try:
        for text in pages:
            page = doc.new_page(width=595, height=842)
            if text:
                page.insert_text(fitz.Point(72, 72), text, fontsize=12)
        doc.save(path)
    finally:
        doc.close()


@pytest.mark.parametrize("ext", [".xls", ".xlsx", ".csv"])
def test_extractable_spreadsheets_are_text(ext):
    """2026-11-06 spreadsheet class reversed 2026-11-11: a spreadsheet with a
    text extractor is a TEXT candidate, not a separate excluded class."""
    assert ist.classify_source_document(ext) == ist.SOURCE_CLASS_TEXT


@pytest.mark.parametrize("ext", [".xlsm", ".xlsb"])
def test_spreadsheets_without_an_extractor_are_non_text(ext):
    """The same single ingestibility declaration decides: no extractor, no TEXT."""
    assert ext not in ist.TEXT_BEARING_EXTS
    assert ist.classify_source_document(ext) == ist.SOURCE_CLASS_NON_TEXT


@pytest.mark.parametrize("ext", [".kmz", ".kml"])
def test_geospatial_containers_are_non_text(ext):
    assert ist.classify_source_document(ext) == ist.SOURCE_CLASS_NON_TEXT


@pytest.mark.parametrize("ext", [".docx", ".pptx", ".txt"])
def test_text_bearing_non_pdf_formats_are_text(ext):
    assert ist.classify_source_document(ext) == ist.SOURCE_CLASS_TEXT


@pytest.mark.parametrize("ext", [".dwg", ".zip", ".jpg", ".png"])
def test_never_text_or_spreadsheet_formats_are_non_text(ext):
    assert ist.classify_source_document(ext) == ist.SOURCE_CLASS_NON_TEXT


@pytest.mark.parametrize(
    "ext,flag",
    [(".pdf", False), (".PDF", False)],
)
def test_pdf_without_a_text_layer_is_scanned_image(ext, flag):
    assert ist.classify_source_document(ext, has_text_layer=flag) == ist.SOURCE_CLASS_SCANNED_IMAGE


@pytest.mark.parametrize(
    "ext,flag",
    [(".pdf", True), (".pdf", None)],
)
def test_pdf_with_or_unverified_text_layer_is_a_text_candidate(ext, flag):
    assert ist.classify_source_document(ext, has_text_layer=flag) == ist.SOURCE_CLASS_TEXT


@pytest.mark.parametrize(
    "ext,expected",
    [("PDF", ist.SOURCE_CLASS_TEXT), (".Pdf", ist.SOURCE_CLASS_TEXT), ("", ist.SOURCE_CLASS_NON_TEXT)],
)
def test_extension_normalisation(ext, expected):
    assert ist.classify_source_document(ext) == expected


def test_text_pdf_has_a_text_layer(tmp_path, monkeypatch):
    monkeypatch.delenv("DATA_ENCRYPTION_KEY", raising=False)
    from app.core import doc_index

    path = str(tmp_path / "spec.pdf")
    _write_text_pdf(path, ["Specification text layer"])
    assert doc_index.pdf_has_text_layer(path) is True


def test_image_only_pdf_has_no_text_layer(tmp_path, monkeypatch):
    monkeypatch.delenv("DATA_ENCRYPTION_KEY", raising=False)
    from app.core import doc_index

    path = str(tmp_path / "scan.pdf")
    _write_text_pdf(path, [None, None])
    assert doc_index.pdf_has_text_layer(path) is False


def test_sampling_spread_finds_text_buried_in_the_tail(tmp_path, monkeypatch):
    monkeypatch.delenv("DATA_ENCRYPTION_KEY", raising=False)
    from app.core import doc_index

    # A text layer only on the last page: sampling only the head would miss it.
    path = str(tmp_path / "cover_then_scan.pdf")
    _write_text_pdf(path, [None, None, None, None, "real text on the last page"])
    assert doc_index.pdf_has_text_layer(path, sample_pages=5) is True


def test_unreadable_pdf_reports_no_text_layer(tmp_path, monkeypatch):
    monkeypatch.delenv("DATA_ENCRYPTION_KEY", raising=False)
    from app.core import doc_index

    path = tmp_path / "broken.pdf"
    path.write_bytes(b"%PDF-1.4 not really a pdf")
    assert doc_index.pdf_has_text_layer(str(path)) is False


@pytest.mark.parametrize(
    "total,count,expected",
    [
        (0, 5, []),
        (5, 0, []),
        (0, 1, []),
        (5, 1, [0]),
        (3, 5, [0, 1, 2]),
        (1, 5, [0]),
        (1, 1, [0]),
        (10, 3, [0, 4, 9]),
        (10, 2, [0, 9]),
    ],
)
def test_sample_page_indices_spread(total, count, expected):
    from app.core import doc_index

    assert doc_index._sample_page_indices(total, count) == expected
