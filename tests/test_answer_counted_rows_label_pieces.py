"""A table renders a computed row set when it prints each row's label split
across cells (app/agents/answer_counted_rows._piece_match).

A bill whose reference sits inside the description cell gives each row no
``ref``, so the row's label is its description. A table then prints that
label in pieces (the band in one cell, the reference in another) and no cell
is a whole label. Those pieces name the one row whose label holds all of
them; a dropped row is then named under the table like any other.

Every row set, description, reference and figure below is constructed here;
none is copied from any document or live answer.
"""
from __future__ import annotations

import pytest

from app.agents.answer_counted_rows import name_dropped_rows
from app.agents.base.tools import bill_rows_statement

# (description with the reference inside it, figure, unit). No ref, no page.
SET_PITS = (
    ("A |Pit type P, depth 1.0-1.5m R 20", 4.0, "No"),
    ("B |Pit type P, depth 1.5-2.0m R 21", 11.0, "No"),
    ("C |Pit type P, depth 2.0-2.5m R22.1", 7.0, "No"),
    ("D |Pit type P, depth 2.5-3.0m R22.2", 3.0, "No"),
)
SET_DUCTS = (
    ("Duct 0.6 x 0.6m, run 10-20m Y 5", 120.0, "m"),
    ("Duct 0.6 x 0.6m, run 20-30m Y 6", 85.5, "m"),
    ("Duct 0.6 x 0.6m, run 30-40m Y 7", 40.0, "m"),
)

# How a table prints each row in pieces: (band cell, reference cell).
PIECES_PITS = (("1.0–1.5 m", "R 20"), ("1.5–2.0 m", "R 21"),
               ("2.0–2.5 m", "R 22.1"), ("2.5–3.0 m", "R22.2"))
PIECES_DUCTS = (("0.6 × 0.6 m", "10–20 m"), ("0.6 × 0.6 m", "20–30 m"),
                ("0.6 × 0.6 m", "30–40 m"))


def _statements(rows):
    unit = rows[0][2]
    total = sum(r[1] for r in rows)
    return bill_rows_statement({"status": "success", "bills": [{
        "document_id": "doc-y", "filename": "invented bill.xlsx", "status": "success",
        "measure": "quantity", "terms": ["band"], "section_terms": [],
        "count": len(rows), "sum": total, "unit": unit, "sum_by_unit": {unit: total},
        "rows": [{"ref": "", "pages": [], "section": ["S"], "description": d,
                  "unit": u, "quantity": q} for d, q, u in rows],
    }]})


def _fmt(value: float) -> str:
    return f"{value:g}"


def _answer(rows, pieces, printed, *, extra=None) -> str:
    unit = rows[0][2]
    total = sum(r[1] for r in rows)
    lines = [f"Band\tWhere\tQty ({unit})"]
    for i in printed:
        lines.append(f"{pieces[i][0]}\t{pieces[i][1]}\t{_fmt(rows[i][1])}")
    if extra is not None:
        lines.append(extra)
    lines.append(f"Total\t\t{_fmt(total)}")
    return (f"{_fmt(total)} {unit} across the matching rows.\n\n" + "\n".join(lines)
            + "\n\nThe bill prints no subtotal for this item.")


CASES = [
    pytest.param(SET_PITS, PIECES_PITS, [0, 1, 2], [3], id="pits-last-dropped"),
    pytest.param(SET_DUCTS, PIECES_DUCTS, [0, 2], [1], id="ducts-middle-dropped"),
]


@pytest.mark.parametrize("rows,pieces,printed,missing", CASES)
def test_rows_printed_in_pieces_name_the_dropped_row(rows, pieces, printed, missing):
    text = _answer(rows, pieces, printed)
    out = name_dropped_rows(text, _statements(rows))
    note = next(line for line in out.split("\n") if "Not printed" in line)
    unit = rows[0][2]
    assert out != text
    assert f"prints {len(printed)} of them" in out
    for i in missing:
        assert f"{rows[i][0]}: {_fmt(rows[i][1])} {unit}" in out
    for i in printed:
        assert rows[i][0] not in note


@pytest.mark.parametrize("rows,pieces", [
    pytest.param(SET_PITS, PIECES_PITS, id="pits"),
    pytest.param(SET_DUCTS, PIECES_DUCTS, id="ducts"),
])
def test_every_row_printed_in_pieces_is_left_alone(rows, pieces):
    text = _answer(rows, pieces, range(len(rows)))
    assert name_dropped_rows(text, _statements(rows)) == text


@pytest.mark.parametrize("rows,pieces,printed,extra", [
    # "R22" is a piece of two labels (R22.1 and R22.2): it names neither.
    pytest.param(SET_PITS, PIECES_PITS, [0, 1], "\tR22\t10", id="pits-two-labels"),
    # The size alone is a piece of every label.
    pytest.param(SET_DUCTS, PIECES_DUCTS, [0], "0.6 × 0.6 m\t\t125.5", id="ducts-every-label"),
])
def test_a_row_whose_pieces_fit_more_than_one_label_is_not_guessed(rows, pieces, printed, extra):
    text = _answer(rows, pieces, printed, extra=extra)
    assert name_dropped_rows(text, _statements(rows)) == text


@pytest.mark.parametrize("rows,pieces,printed,extra", [
    pytest.param(SET_PITS, PIECES_PITS, [0, 1, 2], "3.0–3.5 m\tR 23\t3", id="pits-band-not-held"),
    pytest.param(SET_DUCTS, PIECES_DUCTS, [0, 1], "0.6 × 0.6 m\t40–50 m\t40", id="ducts-run-not-held"),
])
def test_a_row_whose_pieces_fit_no_label_leaves_the_table_alone(rows, pieces, printed, extra):
    text = _answer(rows, pieces, printed, extra=extra)
    assert name_dropped_rows(text, _statements(rows)) == text


@pytest.mark.parametrize("rows,pieces,printed,wrong", [
    pytest.param(SET_PITS, PIECES_PITS, [0, 1, 2], 1, id="pits"),
    pytest.param(SET_DUCTS, PIECES_DUCTS, [0, 2], 0, id="ducts"),
])
def test_a_figure_that_is_not_the_sets_leaves_the_table_alone(rows, pieces, printed, wrong):
    text = _answer(rows, pieces, printed)
    bad = f"{pieces[wrong][0]}\t{pieces[wrong][1]}\t{_fmt(rows[wrong][1] + 1)}"
    good = f"{pieces[wrong][0]}\t{pieces[wrong][1]}\t{_fmt(rows[wrong][1])}"
    text = text.replace(good, bad)
    assert name_dropped_rows(text, _statements(rows)) == text
