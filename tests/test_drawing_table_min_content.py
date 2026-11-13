"""P-D1 (2): a recovered drawing table earns a chunk only when it has content.

Table detection on a dense drawing sheet also boxes isolated labels. Those
one-line "tables" were being indexed as `DRAWING TABLE` chunks. The rule is
general: at least two non-empty lines (a header line counts) and at least two
non-empty cells between them. Every input below is built here.
"""
from __future__ import annotations

import pytest

from app.core.doc_index import _drawing_table_chunks, _drawing_table_has_content


def _table(header, rows, kind="unclassified", page=3):
    return {
        "header": header,
        "rows": rows,
        "row_count": len(rows),
        "kind": kind,
        "page": page,
    }


# Shapes that carry no structure: one non-empty line, or one non-empty cell.
NOISE = [
    pytest.param(_table([], [["", "", "LVL +12.400", "", ""]]), id="one-row-one-cell"),
    pytest.param(_table([], [["GRID", "-07"]]), id="one-row-two-cells"),
    pytest.param(_table(["", "", ""], [["", "NOTE", ""]]), id="blank-header-one-cell-row"),
    pytest.param(_table(["KEY"], [[""], [" "]]), id="header-one-cell-blank-rows"),
]

# Shapes that do: a header with a body row, or two body rows without a header.
CONTENT = [
    pytest.param(
        _table(["BAR MARK", "DIA", "QTY"], [["B1", "16", "24"]], kind="rebar_schedule"),
        id="header-plus-one-row",
    ),
    pytest.param(_table([], [["D01", "900x2100"], ["D02", "1000x2100"]]), id="two-rows-no-header"),
    pytest.param(_table(["REV"], [["C"]]), id="header-cell-plus-row-cell"),
]


@pytest.mark.parametrize("table", NOISE)
def test_noise_table_has_no_content(table):
    assert _drawing_table_has_content(table) is False


@pytest.mark.parametrize("table", CONTENT)
def test_structured_table_has_content(table):
    assert _drawing_table_has_content(table) is True


@pytest.mark.parametrize("table", NOISE)
def test_noise_table_yields_no_chunk(table):
    assert _drawing_table_chunks([table], "S-101") == []


@pytest.mark.parametrize("table", CONTENT)
def test_structured_table_yields_one_chunk(table):
    chunks = _drawing_table_chunks([table], "S-101")
    assert len(chunks) == 1
    assert chunks[0].startswith("DRAWING TABLE [S-101]")


def test_mixed_sheet_keeps_only_structured_tables():
    keep_a = _table(["DOOR", "SIZE"], [["D1", "900"]], kind="door_schedule")
    keep_b = _table([], [["R1", "WALL"], ["R2", "FLOOR"]])
    drop_a = _table([], [["", "SL +3.000", ""]])
    drop_b = _table([], [["AX", "-02"]])
    chunks = _drawing_table_chunks([drop_a, keep_a, drop_b, keep_b], "A-200")
    assert len(chunks) == 2
    assert "D1 | 900" in chunks[0]
    assert "R2 | FLOOR" in chunks[1]
    assert not any("SL +3.000" in c or "AX | -02" in c for c in chunks)


def test_missing_header_and_rows_keys_are_noise():
    assert _drawing_table_has_content({"kind": "legend", "page": 1}) is False
    assert _drawing_table_has_content({"header": None, "rows": None}) is False
