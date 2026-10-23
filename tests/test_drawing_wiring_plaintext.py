"""Drawing-table wiring must open the plaintext PDF, not the stored path.

The extractor decrypts via ``open_plaintext`` and deletes that temp before
wiring runs. Wiring used to call ``fitz.open`` on the stored path, which in
production is Fernet ciphertext under ``DATA_DIR``. PyMuPDF then raises
FileDataError and the schedule is skipped, while the text index still lands.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest

fitz = pytest.importorskip("fitz", reason="PyMuPDF is the drawing reader")

from app.core import doc_index, file_crypto

NAME = "sheet-DWG.pdf"


def _drawing_pdf(path) -> str:
    doc = fitz.open()
    page = doc.new_page(width=842, height=595)
    page.insert_text(fitz.Point(40, 40), "DRAWING NO: S-1", fontsize=8)
    xs, ys = [40, 170, 300, 430], [140, 175, 210]
    for y in ys:
        page.draw_line(fitz.Point(xs[0], y), fitz.Point(xs[-1], y))
    for x in xs:
        page.draw_line(fitz.Point(x, ys[0]), fitz.Point(x, ys[-1]))
    cells = [["BAR MARK", "DIAMETER", "QTY"], ["B1", "16", "48"]]
    for r, row in enumerate(cells):
        for c, value in enumerate(row):
            page.insert_text(fitz.Point(xs[c] + 6, ys[r] + 22), value, fontsize=8)
    doc.save(str(path))
    doc.close()
    return str(path)


def _wired(path: str) -> list[str]:
    return doc_index._drawing_chunks_for_document(path, NAME, ".pdf", "p1")


def _assert_wired(chunks: list[str]) -> None:
    blob = "\n".join(chunks)
    assert "B1" in blob and "48" in blob
    assert "DRAWING TABLE" in blob


def test_plaintext_pdf_still_wires(monkeypatch, tmp_path):
    monkeypatch.delenv("DATA_ENCRYPTION_KEY", raising=False)
    path = _drawing_pdf(tmp_path / "plain.pdf")
    _assert_wired(_wired(path))


def test_encrypted_store_still_wires(monkeypatch, tmp_path):
    from cryptography.fernet import Fernet

    monkeypatch.setenv("DATA_ENCRYPTION_KEY", Fernet.generate_key().decode())
    plain = tmp_path / "plain.pdf"
    _drawing_pdf(plain)
    stored = tmp_path / "stored.pdf"
    file_crypto.write_document(str(stored), plain.read_bytes())
    on_disk = stored.read_bytes()
    assert file_crypto.looks_encrypted(on_disk)
    assert b"%PDF-" not in on_disk
    _assert_wired(_wired(str(stored)))


def test_truncated_store_uses_plaintext_copy(monkeypatch, tmp_path):
    """Stored bytes are a truncated PDF. The plaintext copy is the real file."""
    monkeypatch.delenv("DATA_ENCRYPTION_KEY", raising=False)
    good = tmp_path / "good.pdf"
    _drawing_pdf(good)
    stored = tmp_path / "stored.pdf"
    stored.write_bytes(good.read_bytes()[:64])

    @contextmanager
    def _plain(_path):
        yield str(good)

    monkeypatch.setattr(doc_index.file_crypto, "open_plaintext", _plain)
    _assert_wired(_wired(str(stored)))


def test_non_pdf_artifact_uses_plaintext_copy(monkeypatch, tmp_path):
    """Stored path is a non-PDF artifact that still carries a .pdf suffix."""
    monkeypatch.delenv("DATA_ENCRYPTION_KEY", raising=False)
    good = tmp_path / "good.pdf"
    _drawing_pdf(good)
    stored = tmp_path / "stored.pdf"
    stored.write_bytes(b"ocr-text-artifact")

    @contextmanager
    def _plain(_path):
        yield str(good)

    monkeypatch.setattr(doc_index.file_crypto, "open_plaintext", _plain)
    _assert_wired(_wired(str(stored)))
