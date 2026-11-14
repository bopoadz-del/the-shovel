"""P-B3: an image-only PDF is a bill only by its own content.

Every input is built here: synthetic word boxes for page content and PDFs
drawn with PyMuPDF (one with a text layer, one image-only). No document, no
bill name, no fixture file. Each decision has at least two inputs.
"""

from __future__ import annotations

import fitz
import pytest

from app.core import bill_classify as bc
from app.core import bill_table_geometry as g

H = 20.0
CW = 9.0
LINE = 34.0


def _cell(text: str, x: float, y: float, page: int) -> list[g.Word]:
    out, cur = [], x
    for tok in text.split():
        w = CW * len(tok)
        out.append(g.Word(tok, page, cur, y, w, H))
        cur += w + CW
    return out


def _page(columns, lines, page: int, header: bool = True) -> list[g.Word]:
    words: list[g.Word] = []
    y = 100.0
    if header:
        for label, x in columns:
            words += _cell(label, x, y, page)
        y += LINE
    for line in lines:
        if isinstance(line, str):
            words += _cell(line, columns[0][1], y, page)
        else:
            for (_label, x), text in zip(columns, line):
                if text:
                    words += _cell(text, x, y, page)
        y += LINE
    return words


COLS_A = [("Item", 40), ("Description", 120), ("Unit", 600), ("Qty", 700), ("Rate", 820), ("Amount", 960)]
COLS_B = [("Ref", 40), ("Amount SAR", 160), ("Unit Rate SAR", 330), ("Quantity", 520), ("Unit", 640), ("Description", 740)]


def _priced_a(n: int, wrong: int = 0) -> list[list[str]]:
    """n priced rows in COLS_A order; the first ``wrong`` have amount off by 7%."""
    out = []
    for i in range(n):
        q, r = 10 + i, 5.0 + i
        a = q * r * (1.07 if i < wrong else 1.0)
        out.append([f"A{i}", f"Work item {i}", "m", str(q), f"{r:.2f}", f"{a:,.2f}"])
    return out


def _priced_b(n: int, wrong: int = 0) -> list[list[str]]:
    out = []
    for i in range(n):
        q, r = 200 + i, 2.5 + i
        a = q * r * (1.07 if i < wrong else 1.0)
        out.append([f"E{i}", f"{a:,.2f}", f"{r:.2f}", str(q), "m3", f"Earth task {i}"])
    return out


def _unpriced_a(n: int) -> list[list[str]]:
    return [[f"U{i}", f"Measured item {i}", "m2", str(30 + i), "", ""] for i in range(n)]


def _unpriced_b(n: int) -> list[list[str]]:
    return [[f"V{i}", "", "", str(40 + i), "Nr", f"Fitting {i}"] for i in range(n)]


def _prose(page: int) -> list[g.Word]:
    words: list[g.Word] = []
    for k, sentence in enumerate(
        ["The parties agree to the terms below", "Payment is due within thirty days", "Signed on behalf of both"]
    ):
        words += _cell(sentence, 60, 100 + k * LINE, page)
    return words


# --- bill_verdict: one decision per case, two inputs each -------------------


@pytest.mark.parametrize(
    "words,reason",
    [
        (_page(COLS_A, ["Bill 1 Pipework"] + _priced_a(4), page=1), bc.BILL_PRICED),
        (_page(COLS_B, ["Section 2"] + _priced_b(5), page=3), bc.BILL_PRICED),
        (_page(COLS_A, _unpriced_a(3), page=1), bc.BILL_UNPRICED),
        (_page(COLS_B, _unpriced_b(4), page=2), bc.BILL_UNPRICED),
    ],
)
def test_bill_content_is_a_bill(words, reason):
    v = bc.bill_verdict(words)
    assert v.is_bill is True
    assert v.reason == reason
    assert v.header_pages == 1


@pytest.mark.parametrize(
    "words,reason",
    [
        (_prose(1), bc.NOT_BILL_NO_HEADER),
        (_prose(1) + _prose(2), bc.NOT_BILL_NO_HEADER),
        # A header that resolves, but a cover/summary with too few figure rows.
        (_page(COLS_A, _priced_a(2), page=1), bc.NOT_BILL_FEW_ROWS),
        (_page(COLS_B, [], page=1), bc.NOT_BILL_FEW_ROWS),
        # Bill-like headings whose figures do not reconcile.
        (_page(COLS_A, _priced_a(4, wrong=3), page=1), bc.NOT_BILL_UNRECONCILED),
        (_page(COLS_B, _priced_b(5, wrong=5), page=1), bc.NOT_BILL_UNRECONCILED),
    ],
)
def test_not_bill_content(words, reason):
    v = bc.bill_verdict(words)
    assert v.is_bill is False
    assert v.reason == reason


@pytest.mark.parametrize(
    "n,wrong,is_bill",
    [(4, 2, True), (4, 3, False), (6, 3, True), (6, 4, False)],
)
def test_reconciled_share_boundary_is_half(n, wrong, is_bill):
    v = bc.bill_verdict(_page(COLS_A, _priced_a(n, wrong=wrong), page=1))
    assert v.priced_rows == n
    assert v.reconciled_rows == n - wrong
    assert v.is_bill is is_bill


@pytest.mark.parametrize("n,is_bill", [(2, False), (3, True)])
def test_measured_row_minimum(n, is_bill):
    v = bc.bill_verdict(_page(COLS_B, _priced_b(n), page=1))
    assert v.measured_rows == n
    assert v.is_bill is is_bill


@pytest.mark.parametrize(
    "first,second",
    [
        (_page(COLS_A, _priced_a(2), page=1), _page(COLS_A, _priced_a(2), page=9, header=False)),
        (_page(COLS_B, _priced_b(1), page=4), _page(COLS_B, _priced_b(4), page=5, header=False)),
    ],
)
def test_sampled_pages_do_not_carry_columns(first, second):
    """Sampled pages need not be adjacent: a page without its own header adds
    no rows, so the second page's rows are not counted."""
    v = bc.bill_verdict(first + second)
    assert v.pages_sampled == 2
    assert v.header_pages == 1
    assert v.is_bill is False
    assert v.reason == bc.NOT_BILL_FEW_ROWS


# --- classify_image_only_bill on real PDFs -----------------------------------


def _text_pdf(path, pages: int) -> str:
    doc = fitz.open()
    for i in range(pages):
        doc.new_page().insert_text((72, 72), f"Typed page {i}")
    doc.save(path)
    doc.close()
    return str(path)


def _image_pdf(path, pages: int) -> str:
    doc = fitz.open()
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 40, 40), False)
    pix.set_rect(pix.irect, (200, 200, 200))
    for _ in range(pages):
        doc.new_page().insert_image(fitz.Rect(50, 50, 300, 300), pixmap=pix)
    doc.save(path)
    doc.close()
    return str(path)


@pytest.mark.parametrize("pages", [1, 6])
def test_text_layer_pdf_is_never_a_candidate(tmp_path, pages):
    called = []
    path = _text_pdf(tmp_path / "t.pdf", pages)
    v = bc.classify_image_only_bill(path, page_words=lambda p, idx: called.append(idx) or [])
    assert v.is_bill is False
    assert v.reason == bc.NOT_BILL_TEXT_LAYER
    assert called == []


@pytest.mark.parametrize(
    "pages,sample,expected_idx,content,reason",
    [
        (7, 5, [0, 2, 3, 4, 6], lambda: _page(COLS_A, _priced_a(4), page=1), bc.BILL_PRICED),
        (3, 5, [0, 1, 2], lambda: _page(COLS_B, _priced_b(3), page=2), bc.BILL_PRICED),
        (7, 5, [0, 2, 3, 4, 6], lambda: _prose(1), bc.NOT_BILL_NO_HEADER),
        (2, 1, [0], lambda: _page(COLS_A, _priced_a(4, wrong=4), page=1), bc.NOT_BILL_UNRECONCILED),
    ],
)
def test_image_only_pdf_is_classified_by_sampled_content(tmp_path, pages, sample, expected_idx, content, reason):
    seen = []

    def page_words(path, idx):
        seen.append(list(idx))
        return content()

    path = _image_pdf(tmp_path / "s.pdf", pages)
    v = bc.classify_image_only_bill(path, sample_pages=sample, page_words=page_words)
    assert seen == [expected_idx]
    assert v.reason == reason


@pytest.mark.parametrize("pages,expected", [(1, [1]), (4, [1, 3, 4])])
def test_ocr_sampled_pages_renders_each_sampled_page(tmp_path, monkeypatch, pages, expected):
    """Rendering is real (PyMuPDF); the OCR engine is stood in for so the
    test checks which pages are rendered and that their words are joined."""
    rendered = []

    def fake_ocr(image, page, lang="eng"):
        rendered.append((page, image.size[0] > 0 and image.size[1] > 0))
        return [g.Word(f"w{page}", page, 0, 0, 10, 10)]

    monkeypatch.setattr(g, "ocr_page_words", fake_ocr)
    path = _image_pdf(tmp_path / "o.pdf", pages)
    indices = [p - 1 for p in expected]
    words = bc._ocr_sampled_pages(path, indices)
    assert [w.text for w in words] == [f"w{p}" for p in expected]
    assert rendered == [(p, True) for p in expected]
