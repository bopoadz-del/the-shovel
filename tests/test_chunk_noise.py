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


def test_clean_table_text_drops_header_repeat_keeps_first_and_data():
    text = "\n".join([
        "Item | Description | Unit | Qty",
        "1.1 | Excavation | m3 | 120",
        "0 | 0 | 0 | 0",
        "Item | Description | Unit | Qty",
        "2.1 | Backfill | m3 | 80",
    ])
    out, counts = chunk_noise.clean_table_text(text)
    lines = out.splitlines()
    assert lines == [
        "Item | Description | Unit | Qty",
        "1.1 | Excavation | m3 | 120",
        "2.1 | Backfill | m3 | 80",
    ]
    assert counts == {"empty_row": 1, "header_repeat": 1}


def test_clean_table_text_keeps_repeated_data_row_with_figures():
    # A repeated row that states a figure is data, not a header repeat.
    text = "Item | Qty\nBolt M12 | 40\nBolt M12 | 40"
    out, counts = chunk_noise.clean_table_text(text)
    assert out.splitlines() == ["Item | Qty", "Bolt M12 | 40", "Bolt M12 | 40"]
    assert counts == {"empty_row": 0, "header_repeat": 0}


def test_refuse_chunks_keeps_pages_aligned():
    chunks = ["Real clause text on page one.", "0 | 0 | 0", "Clause on page three."]
    kept, pages, counts = chunk_noise.refuse_chunks(chunks, [1, 2, 3])
    assert kept == ["Real clause text on page one.", "Clause on page three."]
    assert pages == [1, 3]
    assert counts == {"empty_row": 1}


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
    "row, emitted",
    [(["0", 0, 0, 0], False), (["Blockwork 200mm", 50, 30, 1500], True)],
)
def test_row_chunker_refuses_all_zero_row(tmp_path, row, emitted):
    path = tmp_path / "w.xlsx"
    _write_xlsx(path, [["Description", "Quantity", "Rate", "Amount"], row])
    chunks = doc_index._spreadsheet_row_chunks(str(path), "w.xlsx", ".xlsx")
    assert bool(chunks) is emitted
