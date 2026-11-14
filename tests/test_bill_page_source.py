"""P-B3: word boxes come per PAGE from the best source that page has.

A page with a text layer gives its words with their boxes from that layer and
is never OCR'd; a page without one is rendered and OCR'd. Candidacy for the
derived sheet is "the index holds no row chunks for it" plus the content
verdict, never "some page has text".

Every input is built here: PDFs drawn with PyMuPDF (text pages carry a bill
table typed at known positions, scanned pages carry only an image) and the
OCR engine stood in for, so a scanned page's boxes are exactly the boxes the
test chooses. No document, bill, page or file is named. Each decision has at
least two inputs.
"""

from __future__ import annotations

import fitz
import pytest

from app.core import bill_classify as bc
from app.core import bill_derived as bd
from app.core import bill_table_geometry as g
from tests.test_bill_derived import (
    COLS_A,
    COLS_B,
    _Docs,
    _page,
    _row_a,
    _row_b,
    _source,
)

SCALE = bc.RENDER_DPI / 72.0
FONT_PT = 6.5


def _bill_page_words(cols, row, label: str, n: int, page: int, start: int) -> list[g.Word]:
    rows = [row(start + i, 10 + i, 4 + i) for i in range(n)]
    return _page(cols, [label, *rows], page=page)


def _type_words(page: fitz.Page, words: list[g.Word]) -> None:
    """Type each word at its box (pixel space -> PDF points)."""
    for w in words:
        page.insert_text((w.left / SCALE, (w.top + w.height) / SCALE), w.text,
                         fontsize=FONT_PT)


def _scan_image(page: fitz.Page) -> None:
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 40, 40), False)
    pix.set_rect(pix.irect, (220, 220, 220))
    page.insert_image(fitz.Rect(30, 30, 400, 500), pixmap=pix)


def _pdf(path, kinds: list[str], content: dict[int, list[g.Word]]) -> str:
    """One page per entry of ``kinds`` ("text" or "scan"); a text page types
    its content, a scan page carries only an image."""
    doc = fitz.open()
    for i, kind in enumerate(kinds):
        page = doc.new_page(width=842, height=595)
        if kind == "text":
            _type_words(page, content.get(i + 1, []))
        else:
            _scan_image(page)
    doc.save(str(path))
    doc.close()
    return str(path)


def _fake_ocr(boxes: dict[int, list[g.Word]], calls: list[int]):
    def ocr(image, page, lang="eng"):
        assert image.size[0] > 0 and image.size[1] > 0
        calls.append(page)
        return list(boxes.get(page, []))
    return ocr


CONTENT = {
    "a": lambda page, start: _bill_page_words(COLS_A, _row_a, "Section Works", 4, page, start),
    "b": lambda page, start: _bill_page_words(COLS_B, _row_b, "Earthworks Bill", 5, page, start),
}


# -- per-page source -----------------------------------------------------------


@pytest.mark.parametrize("kinds", [["text", "scan"], ["scan", "text", "scan"]])
def test_each_page_is_read_from_its_own_source(tmp_path, monkeypatch, kinds):
    content = {i + 1: CONTENT["a"](i + 1, 10 * i) for i in range(len(kinds))}
    path = _pdf(tmp_path / "m.pdf", kinds, content)
    calls: list[int] = []
    monkeypatch.setattr(g, "ocr_page_words", _fake_ocr(content, calls))

    reads = bc.read_pages(path, list(range(len(kinds))))

    want = [bc.SOURCE_TEXT_LAYER if k == "text" else bc.SOURCE_OCR for k in kinds]
    assert [r.source for r in reads] == want
    assert calls == [i + 1 for i, k in enumerate(kinds) if k == "scan"]
    for r in reads:
        assert r.words and {w.page for w in r.words} == {r.index + 1}


@pytest.mark.parametrize("cols", ["a", "b"])
def test_mixed_document_yields_rows_from_both_kinds_of_page(tmp_path, monkeypatch, cols):
    content = {1: CONTENT[cols](1, 0), 2: CONTENT[cols](2, 50)}
    path = _pdf(tmp_path / "mixed.pdf", ["text", "scan"], content)
    calls: list[int] = []
    monkeypatch.setattr(g, "ocr_page_words", _fake_ocr(content, calls))
    docs = _Docs({**_source(), "file_path": path}, pages=2)
    docs.install(monkeypatch, tmp_path)

    plan = bd.plan_bill_sheet("p1", "src1")  # real classifier, real page source

    assert plan["reason"] in (bc.BILL_PRICED, bc.BILL_UNPRICED)
    assert plan["pages_text_layer"] == 1 and plan["pages_without_text_layer"] == 1
    assert plan["pages_read_text_layer"] == 1 and plan["pages_read_ocr"] == 1
    assert calls.count(1) == 0  # the text page is never OCR'd
    rows_per_page = len([w for w in content[1] if w.text[0] in "AE" and w.text[1:].isdigit()])
    assert plan["rows"] == 2 * rows_per_page
    assert {s["pages"][0] for s in plan["sections"]} | {s["pages"][1] for s in plan["sections"]} == {1, 2}


@pytest.mark.parametrize("cols", ["a", "b"])
def test_all_text_and_all_scanned_give_the_same_rows_for_the_same_boxes(tmp_path, monkeypatch, cols):
    content = {1: CONTENT[cols](1, 0), 2: CONTENT[cols](2, 50)}
    text_pdf = _pdf(tmp_path / "t.pdf", ["text", "text"], content)
    # The scanned twin's OCR returns exactly the boxes the text layer gives.
    boxes = {r.index + 1: list(r.words) for r in bc.read_pages(text_pdf, [0, 1])}
    scan_pdf = _pdf(tmp_path / "s.pdf", ["scan", "scan"], {})
    calls: list[int] = []
    monkeypatch.setattr(g, "ocr_page_words", _fake_ocr(boxes, calls))

    plans = {}
    for name, path in (("text", text_pdf), ("scan", scan_pdf)):
        docs = _Docs({**_source(), "file_path": path}, pages=2)
        docs.install(monkeypatch, tmp_path)
        plans[name] = bd.plan_bill_sheet("p1", "src1")

    assert plans["text"]["rows"] > 0
    assert plans["text"]["plan_hash"] == plans["scan"]["plan_hash"]
    assert plans["text"]["sections"] == plans["scan"]["sections"]
    assert (plans["text"]["pages_read_text_layer"], plans["text"]["pages_read_ocr"]) == (2, 0)
    assert (plans["scan"]["pages_read_text_layer"], plans["scan"]["pages_read_ocr"]) == (0, 2)


@pytest.mark.parametrize("kinds", [["text", "scan"], ["scan", "text"]])
def test_work_dir_keeps_each_pages_source(tmp_path, monkeypatch, kinds):
    content = {1: CONTENT["a"](1, 0), 2: CONTENT["a"](2, 50)}
    path = _pdf(tmp_path / "w.pdf", kinds, content)
    monkeypatch.setattr(g, "ocr_page_words", _fake_ocr(content, []))
    docs = _Docs({**_source(), "file_path": path}, pages=2)
    docs.install(monkeypatch, tmp_path)
    work = str(tmp_path / "work")

    first = bd.plan_bill_sheet("p1", "src1", work_dir=work)

    def no_read(path, indices):
        raise AssertionError("re-read a page whose boxes were kept")
    again = bd.plan_bill_sheet("p1", "src1", work_dir=work, page_source=no_read)
    assert again["plan_hash"] == first["plan_hash"]
    assert again["pages_read_cache"] == 2
    assert (again["pages_text_layer"], again["pages_without_text_layer"]) == (1, 1)
    assert (again["pages_read_text_layer"], again["pages_read_ocr"]) == (0, 0)


@pytest.mark.parametrize("n_pages", [1, 3])
def test_word_boxes_saved_before_sources_were_recorded_count_as_ocr(tmp_path, monkeypatch, n_pages):
    import dataclasses
    import json
    import os

    work = tmp_path / "work"
    work.mkdir()
    content = {p: CONTENT["b"](p, 10 * p) for p in range(1, n_pages + 1)}
    for p, words in content.items():
        with open(os.path.join(work, f"page_{p - 1:05d}.json"), "w", encoding="utf-8") as fh:
            json.dump([dataclasses.asdict(w) for w in words], fh)
    docs = _Docs(_source(), pages=n_pages)
    docs.install(monkeypatch, tmp_path)

    def no_read(path, indices):
        raise AssertionError("re-read a page whose boxes were kept")
    plan = bd.plan_bill_sheet("p1", "src1", classify=lambda p: bc.BillVerdict(True, bc.BILL_PRICED),
                              work_dir=str(work), page_source=no_read)
    assert plan["pages_without_text_layer"] == n_pages and plan["pages_text_layer"] == 0
    assert plan["rows"] == 5 * n_pages


# -- candidacy -----------------------------------------------------------------


@pytest.mark.parametrize("attr, indexed", [
    ({"verified": False, "reason": "not_cross_checked", "priced_rows": 12}, True),
    ({"verified": False, "reason": "pages_skipped", "priced_rows": 3, "pages_skipped": 2}, True),
    ({"verified": False, "reason": "parse_failed", "detail": "x"}, False),
    ({"verified": False, "reason": "parse_error", "detail": "x"}, False),
    ({"verified": False, "reason": "no_priced_rows", "priced_rows": 0}, False),
    (None, False),
])
def test_rows_indexed_as_rows_reads_the_recorded_parse(attr, indexed):
    meta = {} if attr is None else {"boq_total": attr}
    assert bc.rows_indexed_as_rows({"metadata": meta}) is indexed


@pytest.mark.parametrize("priced_rows", [1, 40])
def test_document_with_rows_already_indexed_is_not_a_candidate(tmp_path, monkeypatch, priced_rows):
    meta = {"boq_total": {"verified": False, "reason": "not_cross_checked",
                          "priced_rows": priced_rows}}
    docs = _Docs({**_source(), "metadata": meta})
    docs.install(monkeypatch, tmp_path)

    def no_classify(path):
        raise AssertionError("classified a bill whose rows are already indexed")

    for step in (bd.plan_bill_sheet, lambda *a, **k: bd.derive_bill_sheet(*a, plan_hash="x", **k)):
        out = step("p1", "src1", classify=no_classify)
        assert out == {"derived": False, "reason": bc.NOT_CANDIDATE_ROWS_INDEXED}
    assert docs.added == [] and docs.indexed == []


@pytest.mark.parametrize("meta", [
    {"boq_total": {"verified": False, "reason": "parse_failed"}},
    {},
])
def test_document_whose_row_parse_did_not_succeed_is_classified(tmp_path, monkeypatch, meta):
    docs = _Docs({**_source(), "metadata": meta})
    docs.install(monkeypatch, tmp_path)
    seen: list[str] = []

    def classify(path):
        seen.append(path)
        return bc.BillVerdict(False, bc.NOT_BILL_NO_HEADER)

    out = bd.plan_bill_sheet("p1", "src1", classify=classify)
    assert seen == ["/store/src1.pdf"]
    assert out == {"derived": False, "reason": bc.NOT_BILL_NO_HEADER}


@pytest.mark.parametrize("kinds", [["text"] * 3, ["text", "scan", "text"]])
def test_a_text_layer_no_longer_excludes_a_bill(tmp_path, monkeypatch, kinds):
    content = {i + 1: CONTENT["a"](i + 1, 10 * i) for i in range(len(kinds))}
    path = _pdf(tmp_path / "x.pdf", kinds, content)
    monkeypatch.setattr(g, "ocr_page_words", _fake_ocr(content, []))
    verdict = bc.classify_bill_pdf(path)
    assert verdict.is_bill is True
    assert verdict.pages_sampled == len(kinds) and verdict.header_pages == len(kinds)


class _StubPage:
    """Just what ``text_layer_words`` reads from a page: its words and the
    rotation matrix (identity: an unrotated page)."""

    def __init__(self, words):
        self._words = words
        self.rotation_matrix = fitz.Matrix(1, 1)

    def get_text(self, kind):
        assert kind == "words"
        return self._words


@pytest.mark.parametrize("raw, kept", [
    ([(10, 20, 40, 30, "Qty", 0, 0, 0), (50, 20, 60, 30, " ", 0, 0, 1)], ["Qty"]),
    ([(5, 5, 9, 9, " ", 0, 0, 0), (72, 36, 108, 45, "Rate", 0, 0, 1),
      (120, 36, 150, 45, "12.50", 0, 0, 2)], ["Rate", "12.50"]),
])
def test_text_layer_words_drop_blank_words_and_scale_boxes(raw, kept):
    words = bc.text_layer_words(_StubPage(raw), number=4)
    assert [w.text for w in words] == kept
    by_text = {r[4]: r for r in raw}
    for w in words:
        x0, y0, x1, y1 = by_text[w.text][:4]
        assert w.page == 4
        assert (w.left, w.top) == pytest.approx((x0 * SCALE, y0 * SCALE))
        assert (w.width, w.height) == pytest.approx(((x1 - x0) * SCALE, (y1 - y0) * SCALE))
