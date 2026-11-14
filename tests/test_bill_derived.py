"""P-B3: an image-only bill gets its own derived spreadsheet row, linked to the
source and indexed through the spreadsheet row-chunk path.

Every input is built here: synthetic OCR word boxes drawn in two column
orders (one with a section split across pages), and in-memory stand-ins for
the documents table. The derived workbook is real and is read back through
``doc_index._spreadsheet_row_chunks``. Each decision has at least two inputs.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from app.core import bill_classify as bc
from app.core import bill_derived as bd
from app.core import bill_table_geometry as g
from app.core import byte_store, doc_index, projects

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


def _page(columns, lines, page: int) -> list[g.Word]:
    """A page: header line, then section labels (str) and rows (list)."""
    words: list[g.Word] = []
    y = 100.0
    for label, x in columns:
        words += _cell(label, x, y, page)
    y += LINE
    for line in lines:
        if isinstance(line, str):
            words += _cell(line, columns[1][1], y, page)
        else:
            for (_label, x), text in zip(columns, line):
                if text:
                    words += _cell(text, x, y, page)
        y += LINE
    return words


COLS_A = [("Item", 40), ("Description", 120), ("Unit", 600), ("Qty", 700), ("Rate", 820), ("Amount", 960)]
COLS_B = [("Ref", 40), ("Description", 160), ("Amount SAR", 520), ("Unit Rate SAR", 700), ("Quantity", 880), ("Unit", 1000)]


def _row_a(i: int, q: float, r: float, off: float = 1.0) -> list[str]:
    return [f"A{i}", f"Work item {i}", "m", f"{q:g}", f"{r:.2f}", f"{q * r * off:,.2f}"]


def _row_b(i: int, q: float, r: float, off: float = 1.0) -> list[str]:
    return [f"E{i}", f"Earth task {i}", f"{q * r * off:,.2f}", f"{r:.2f}", f"{q:g}", "m3"]


def _bill_a() -> tuple[list[g.Word], dict[str, tuple[int, float]], int]:
    """Two sections in COLS_A order; the second runs across pages 2 and 3.
    One row's amount is wrong (kept and flagged)."""
    s1 = [_row_a(i, 10 + i, 5 + i) for i in range(4)]
    s2p2 = [_row_a(10 + i, 3 + i, 12 + i) for i in range(3)]
    s2p3 = [_row_a(20 + i, 7 + i, 9 + i, off=1.10 if i == 0 else 1.0) for i in range(2)]
    words = (
        _page(COLS_A, ["Section One Preliminaries", *s1], page=1)
        + _page(COLS_A, ["Section Two Concrete", *s2p2], page=2)
        + _page(COLS_A, s2p3, page=3)
    )
    def total(rows):
        return sum(g.parse_number(r[5]) for r in rows)
    want = {
        "Section One Preliminaries": (4, total(s1)),
        "Section Two Concrete": (5, total(s2p2) + total(s2p3)),
    }
    return words, want, 1


def _bill_b() -> tuple[list[g.Word], dict[str, tuple[int, float]], int]:
    """One section in COLS_B order (amount before rate before quantity)."""
    s = [_row_b(i, 200 + i, 2.5 + i) for i in range(6)]
    words = _page(COLS_B, ["Earthworks Bill", *s], page=1)
    return words, {"Earthworks Bill": (6, sum(g.parse_number(r[2]) for r in s))}, 0


class _Docs:
    """In-memory stand-in for the documents table (only what the module uses)."""

    def __init__(self, source: dict[str, Any], others: list[dict[str, Any]] | None = None):
        self.rows = {source["id"]: source, **{o["id"]: o for o in others or []}}
        self.added: list[dict[str, Any]] = []
        self.meta_updates: list[tuple[str, dict]] = []
        self.indexed: list[tuple[tuple, dict]] = []

    def install(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(projects, "get_document", lambda i: self.rows.get(i))
        monkeypatch.setattr(projects, "list_documents", lambda pid, **k: list(self.rows.values()))
        monkeypatch.setattr(projects, "add_document", self._add)
        monkeypatch.setattr(
            projects, "update_document_metadata",
            lambda i, m: self.meta_updates.append((i, m)),
        )
        monkeypatch.setattr(byte_store, "allocate_path", lambda name: str(tmp_path / name))
        def write(path, data):
            with open(path, "wb") as fh:
                fh.write(data)
        monkeypatch.setattr(byte_store, "write_bytes", write)
        monkeypatch.setattr(bd, "_page_count", lambda path: 3)

        def fake_index(*a, **k):
            self.indexed.append((a, k))
            return {"status": "ok", "total_chunks": 0}
        monkeypatch.setattr(doc_index, "index_document", fake_index)

    def _add(self, project_id, original_name, **kw):
        row = {"id": f"d{len(self.rows)}", "project_id": project_id,
               "original_name": original_name, **kw}
        self.rows[row["id"]] = row
        self.added.append(row)
        return row


def _source(i: str = "src1", name: str = "scan.pdf") -> dict[str, Any]:
    return {"id": i, "original_name": name, "file_path": f"/store/{i}.pdf", "metadata": {}}


def _BILL(path: str) -> bc.BillVerdict:
    return bc.BillVerdict(True, bc.BILL_PRICED)


def _amount_sum(chunks: list[str]) -> float:
    total = 0.0
    for c in chunks:
        for part in c.split("; "):
            if part.startswith("Amount:"):
                total += g.parse_number(part.split(":", 1)[1].strip()) or 0.0
    return total


@pytest.mark.parametrize("build", [_bill_a, _bill_b])
def test_bill_gets_derived_row_whose_sheets_match_the_drawn_tables(tmp_path, monkeypatch, build):
    import openpyxl

    words, want, want_flagged = build()
    docs = _Docs(_source())
    docs.install(monkeypatch, tmp_path)

    out = bd.derive_bill_sheet("p1", "src1", classify=_BILL, page_words=lambda p, idx: words)

    assert out["derived"] is True
    assert out["rows"] == sum(n for n, _ in want.values())
    assert out["rows_flagged"] == want_flagged
    assert out["sheets"] == len(want)
    (row,) = docs.added
    assert row["metadata"]["derived_from"] == "src1"
    assert row["metadata"]["derivation"] == bd.DERIVATION
    assert row["reingest_of"] is None
    assert row["original_name"].endswith(".xlsx")
    # The source PDF row is linked to the derived row, not hidden.
    assert docs.meta_updates == [("src1", {"derived_document": row["id"]})]
    assert docs.indexed == [(("p1", row["id"]), {"profile": doc_index.PROFILE_CORPUS_BUILD})]

    # The stored workbook: one sheet per section, row counts and sums as drawn.
    wb = openpyxl.load_workbook(row["file_path"], data_only=True)
    assert len(wb.sheetnames) == len(want)
    for ws in wb.worksheets:
        values = [list(r) for r in ws.iter_rows(values_only=True)]
        section = values[0][0]
        header = values[1]
        amount_col = header.index("Amount")
        body = values[2:]
        n, total = want[section]
        assert len(body) == n
        assert sum(g.parse_number(str(r[amount_col])) for r in body) == pytest.approx(total)
        assert "Source Page" in header and "Check" in header

    # Read back through the existing spreadsheet row-chunk path.
    chunks = doc_index._spreadsheet_row_chunks(row["file_path"], row["original_name"], ".xlsx")
    assert len(chunks) == out["rows"]
    assert _amount_sum(chunks) == pytest.approx(sum(t for _, t in want.values()))
    for section in want:
        assert sum(f"Bill: {section}" in c for c in chunks) == want[section][0]


@pytest.mark.parametrize("verdict", [
    bc.BillVerdict(False, bc.NOT_BILL_TEXT_LAYER),
    bc.BillVerdict(False, bc.NOT_BILL_NO_HEADER),
])
def test_not_a_bill_gets_no_derived_row(tmp_path, monkeypatch, verdict):
    docs = _Docs(_source())
    docs.install(monkeypatch, tmp_path)

    def no_ocr(path, idx):
        raise AssertionError("OCR'd every page of a document that is not a bill")

    out = bd.derive_bill_sheet("p1", "src1", classify=lambda p: verdict, page_words=no_ocr)

    assert out == {"derived": False, "reason": verdict.reason}
    assert docs.added == [] and docs.indexed == [] and docs.meta_updates == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("source, reason", [
    (_source(name="sheet.xlsx"), bd.SKIP_NOT_PDF),
    ({**_source(), "metadata": {"derived_from": "other"}}, bd.SKIP_IS_DERIVED),
    ({**_source(), "file_path": None}, bd.SKIP_NO_FILE),
])
def test_skips_before_classifying(tmp_path, monkeypatch, source, reason):
    docs = _Docs(source)
    docs.install(monkeypatch, tmp_path)

    def no_classify(path):
        raise AssertionError("classified a document that cannot have a derived sheet")

    out = bd.derive_bill_sheet("p1", "src1", classify=no_classify)
    assert out == {"derived": False, "reason": reason}
    assert docs.added == []


@pytest.mark.parametrize("words", [
    [],
    _page(COLS_A, ["Only a section label"], page=1),
])
def test_bill_with_no_extractable_rows_gets_no_row(tmp_path, monkeypatch, words):
    docs = _Docs(_source())
    docs.install(monkeypatch, tmp_path)
    out = bd.derive_bill_sheet("p1", "src1", classify=_BILL, page_words=lambda p, i: words)
    assert out["derived"] is False and out["reason"] == bd.SKIP_NO_ROWS
    assert docs.added == []


@pytest.mark.parametrize("prior_visible, want_reingest", [(True, "old9"), (False, None)])
def test_rerun_supersedes_the_visible_previous_derived_row(
    tmp_path, monkeypatch, prior_visible, want_reingest,
):
    words, _, _ = _bill_b()
    prior = {"id": "old9", "original_name": "scan.bill.xlsx", "file_path": "x",
             "metadata": {"derived_from": "src1"}, "retrieval_visible": prior_visible}
    unrelated = {"id": "u1", "original_name": "o.bill.xlsx", "file_path": "y",
                 "metadata": {"derived_from": "another"}, "retrieval_visible": True}
    docs = _Docs(_source(), [prior, unrelated])
    docs.install(monkeypatch, tmp_path)

    out = bd.derive_bill_sheet("p1", "src1", classify=_BILL, page_words=lambda p, i: words)

    assert docs.added[0]["reingest_of"] == want_reingest
    assert out["superseded"] == want_reingest


# ── the ingest hook in doc_index.index_document ──────────────────────────


def _hooked(monkeypatch, filename: str, derive):
    calls: list[tuple] = []
    monkeypatch.setattr(
        doc_index, "_index_document",
        lambda *a, **k: {"status": "ok", "filename": filename, "total_chunks": 4},
    )
    monkeypatch.setattr(doc_index, "_record_pages_without_text", lambda *a: None)

    def fake(project_id, document_id):
        calls.append((project_id, document_id))
        return derive()
    monkeypatch.setattr(bd, "derive_bill_sheet", fake)
    return calls


@pytest.mark.parametrize("profile, filename, want_calls", [
    (doc_index.PROFILE_CORPUS_BUILD, "scan.pdf", 1),
    (doc_index.PROFILE_CORPUS_BUILD, "SCAN.PDF", 1),
    (doc_index.PROFILE_USER_UPLOAD, "scan.pdf", 0),
    (doc_index.PROFILE_CORPUS_BUILD, "bill.xlsx", 0),
])
def test_hook_runs_on_corpus_build_pdfs_only(monkeypatch, profile, filename, want_calls):
    calls = _hooked(monkeypatch, filename, lambda: {"derived": True, "rows": 3})
    result = doc_index.index_document("p1", "d1", profile=profile)
    assert len(calls) == want_calls
    assert result["total_chunks"] == 4
    assert ("derived_bill" in result) is bool(want_calls)
    if want_calls:
        assert calls == [("p1", "d1")]
        assert result["derived_bill"] == {"derived": True, "rows": 3}


@pytest.mark.parametrize("exc", [RuntimeError("ocr died"), OSError("disk full")])
def test_hook_failure_never_changes_the_source_result(monkeypatch, exc):
    def boom():
        raise exc
    _hooked(monkeypatch, "scan.pdf", boom)
    result = doc_index.index_document("p1", "d1", profile=doc_index.PROFILE_CORPUS_BUILD)
    assert result["status"] == "ok" and result["total_chunks"] == 4
    assert result["derived_bill"] == {
        "derived": False, "reason": "error", "error": type(exc).__name__,
    }


def test_derived_name_stays_inside_the_store(tmp_path, monkeypatch):
    words, _, _ = _bill_b()
    docs = _Docs(_source(name="nested/dir/scan.pdf"))
    docs.install(monkeypatch, tmp_path)
    bd.derive_bill_sheet("p1", "src1", classify=_BILL, page_words=lambda p, i: words)
    (row,) = docs.added
    assert os.path.dirname(row["file_path"]) == str(tmp_path)
    assert row["original_name"] == "scan.bill.xlsx"
