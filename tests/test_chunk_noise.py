"""RQ-1: the index refuses noise chunks. Two inputs per decision."""
from __future__ import annotations

import pytest

from app.core import chunk_noise, doc_index


@pytest.mark.parametrize(
    "row, empty",
    [
        ("0 | 0 | 0 | 0", True),
        ("- | 0.00 | (0) | 0%", True),
        ("0 | 0 | 0 0 | 0 | 0", True),  # rows re-flowed onto one line
        ("12 | 4.5 | 54", False),
        ("Excavation | 0 | 0", False),
        ("0 | 0 | 1,250.00", False),
    ],
)
def test_empty_row(row, empty):
    assert chunk_noise.is_empty_row(row) is empty


@pytest.mark.parametrize(
    "text, reason",
    [
        ("VERIFIED-TOTAL GUARD for 'x.pdf': the total could not be read.", "guard_sentence"),
        ("The contractor shall verify the total before the guard rail is fixed.", None),
        ("0 | 0 | 0\n0 | 0 | 0", "empty_row"),
        ("Concrete C40 | m3 | 120 | 450 | 54000", None),
    ],
)
def test_refusal_reason(text, reason):
    assert chunk_noise.refusal_reason(text) == reason


H = "Item | Description | Unit | Qty"
R1 = "1.1 | Excavation | m3 | 120"
R2 = "2.1 | Backfill | m3 | 80"


@pytest.mark.parametrize(
    "lines, kept, counts",
    [
        # a header re-printed above the next bill, and a zero row: both dropped
        ([H, R1, "0 | 0 | 0 | 0", H, R2], [H, R1, R2],
         {"empty_row": 1, "header_repeat": 1}),
        # two DIFFERENT label rows are both headers, both kept
        ([H, R1, "Ref | Note", R2], [H, R1, "Ref | Note", R2],
         {"empty_row": 0, "header_repeat": 0}),
        # a repeated row that states a figure is data, kept
        (["Item | Qty", "Bolt M12 | 40", "Bolt M12 | 40"],
         ["Item | Qty", "Bolt M12 | 40", "Bolt M12 | 40"],
         {"empty_row": 0, "header_repeat": 0}),
        # a blank line is layout, neither kept nor counted as a row
        ([H, "", "   ", R1], [H, R1], {"empty_row": 0, "header_repeat": 0}),
    ],
)
def test_clean_table_text(lines, kept, counts):
    out, got = chunk_noise.clean_table_text(chr(10).join(lines))
    assert out.splitlines() == kept
    assert got == counts


@pytest.mark.parametrize(
    "chunks, pages, kept, kept_pages, counts",
    [
        (["Clause one.", "0 | 0 | 0", "Clause three."], [1, 2, 3],
         ["Clause one.", "Clause three."], [1, 3], {"empty_row": 1}),
        (["Clause one.", "Clause two."], [4, 5],
         ["Clause one.", "Clause two."], [4, 5], {}),
        # pages of another length are not aligned: passed through untouched
        (["0 | 0", "Clause two."], [7], ["Clause two."], [7], {"empty_row": 1}),
    ],
)
def test_refuse_chunks_keeps_pages_aligned(chunks, pages, kept, kept_pages, counts):
    got, got_pages, got_counts = chunk_noise.refuse_chunks(chunks, pages)
    assert got == kept
    assert got_pages == kept_pages
    assert got_counts == counts


@pytest.mark.parametrize(
    "filename, expect_refused",
    [("sheet.xlsx", {"empty_row": 2, "header_repeat": 1}), ("notes.txt", None)],
)
def test_text_chunks_clean_only_table_documents(filename, expect_refused):
    text = "A | B\n0 | 0\nA | B\n0 | 0\nWall | 12"
    meta: dict = {}
    chunks, pages = doc_index._text_chunks_with_pages(text, meta, "default", filename)
    assert len(chunks) == len(pages)
    assert meta.get("noise_refused") == expect_refused
    joined = " ".join(chunks)
    assert "Wall | 12" in joined
    if expect_refused is None:
        assert joined.count("A | B") == 2


def _write_xlsx(path, rows):
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Bill 1"
    for r in rows:
        ws.append(r)
    wb.save(path)


@pytest.mark.parametrize(
    "row, ext, emitted",
    [
        (["0", 0, 0, 0], ".xlsx", False),
        (["Blockwork 200mm", 50, 30, 1500], ".xlsx", True),
        # the row chunker reads spreadsheets only
        (["Blockwork 200mm", 50, 30, 1500], ".pdf", False),
    ],
)
def test_row_chunker_refuses_all_zero_row(tmp_path, row, ext, emitted):
    path = tmp_path / "w.xlsx"
    _write_xlsx(path, [["Description", "Quantity", "Rate", "Amount"], row])
    chunks = doc_index._spreadsheet_row_chunks(str(path), "w" + ext, ext)
    assert bool(chunks) is emitted


@pytest.mark.parametrize("ext, opened", [(".pdf", False), (".xlsx", True)])
def test_row_chunker_opens_only_spreadsheets(tmp_path, caplog, ext, opened):
    # Bytes no workbook reader accepts: a spreadsheet extension is attempted
    # (and its failure logged); any other extension is never opened.
    path = tmp_path / ("x" + ext)
    path.write_bytes(b"%PDF-1.4 not a workbook")
    with caplog.at_level("WARNING"):
        assert doc_index._spreadsheet_row_chunks(str(path), path.name, ext) == []
    assert ("spreadsheet row wiring skipped" in caplog.text) is opened
