"""#411 grader 1420Z point 1: a size is matched as a size.

A size printed with a unit, in the ask and in the bill's own words, is read to
metres, so every spelling of it with a unit selects the same rows. A size with
no unit (grader 1515Z point 2) is the plain number it prints: it is never read
at another magnitude. A size the bill prints nowhere selects nothing and is
named as ignored. Every bill here is invented in the test.
"""

from __future__ import annotations

import pytest

from app.core import bill_aggregate as agg
from app.core import bill_table_geometry as g


def _row(path, page, ref, desc, unit="nr", qty="1"):
    return g.BillRow(section=path[-1], source_page=page,
                     fields={"description": desc, "unit": unit, "quantity": qty},
                     extra={"Item": ref}, path=tuple(path))


#: Two invented bills: one prints its sizes in metres in the heading, the
#: other in millimetres in the description. Each has rows of a size the asks
#: name and rows of other sizes.
BILL_HEADING_METRES = [
    _row(("Chambers", "Chamber 1.8 x 1.8 m"), 1, "A1", "Precast chamber depth n.e. 2 m", qty="4"),
    _row(("Chambers", "Chamber 1.8 x 1.8 m"), 1, "A2", "Precast chamber depth 2 to 3 m", qty="5"),
    _row(("Chambers", "Chamber 1.2 x 1.2 m"), 2, "A3", "Precast chamber depth n.e. 2 m", qty="7"),
    _row(("Chambers", "Chamber 2.4 x 1.8 m"), 2, "A4", "Precast chamber depth n.e. 2 m", qty="2"),
]
BILL_DESC_MILLIMETRES = [
    _row(("Pits",), 1, "P1", "Inspection pit 1800 x 1800 mm internal", qty="3"),
    _row(("Pits",), 1, "P2", "Inspection pit 1,800 x 1,800 internal", qty="6"),
    _row(("Pits",), 2, "P3", "Inspection pit 900 x 900 mm internal", qty="8"),
    _row(("Pits",), 2, "P4", "Inspection pit 180 x 180 cm internal", qty="1"),
]


def _rows(tmp_path, bill, name):
    path = tmp_path / f"{name}.xlsx"
    g.write_bill_workbook(bill, str(path))
    return agg.join_page_continuations(agg.read_sheet_rows(str(path)))


def _refs(rows, ask):
    picked = agg.selection_from_ask(rows, ask)
    out = agg.aggregate(rows, terms=picked["terms"], section=picked["section_terms"],
                        fold=True, sizes=picked["sizes"])
    return sorted(r["ref"] for r in out["rows"]), out["sum"], picked


SPELLINGS = ["How many chambers 1.8 x 1.8 m?", "How many chambers 1.8m x 1.8m?",
             "How many chambers 180 x 180 cm?",
             "كم عدد chambers ١٫٨ × ١٫٨ م"]


@pytest.mark.parametrize("bill, name, want, total", [
    (BILL_HEADING_METRES, "metres", ["A1", "A2"], 9.0),
    # P2 prints 1,800 x 1,800 with no unit: a plain number, not a metric size
    (BILL_DESC_MILLIMETRES, "millimetres", ["P1", "P4"], 4.0),
], ids=["size-in-heading-metres", "size-in-description-millimetres"])
def test_every_spelling_of_one_size_with_a_unit_selects_the_same_rows(tmp_path, bill, name, want, total):
    rows = _rows(tmp_path, bill, name)
    for ask in SPELLINGS:
        ask = ask if name == "metres" else ask.replace("chambers", "pits")
        refs, figure, picked = _refs(rows, ask)
        assert refs == want, ask
        assert figure == total, ask
        assert len(picked["sizes"]) == 1 and picked["ignored"] == [], ask


#: An invented bill whose rows print one plain number at four magnitudes.
BILL_PLAIN_NUMBERS = [
    _row(("Posts",), 1, "N1", "Post type 1.8 galvanised", qty="2"),
    _row(("Posts",), 1, "N2", "Post type 18 galvanised", qty="3"),
    _row(("Posts",), 1, "N3", "Post type 180 galvanised", qty="5"),
    _row(("Posts",), 1, "N4", "Post type 1800 galvanised", qty="7"),
    _row(("Posts",), 2, "N5", "Post type 400 galvanised", qty="11"),
]


@pytest.mark.parametrize("ask, want", [
    ("How many posts 1.8?", ["N1"]),
    ("How many posts 180?", ["N3"]),
    ("How many posts 1800?", ["N4"]),
], ids=["1.8", "180", "1800"])
def test_a_bare_number_selects_only_rows_printing_that_number(tmp_path, ask, want):
    rows = _rows(tmp_path, BILL_PLAIN_NUMBERS, "plain")
    refs, _, picked = _refs(rows, ask)
    assert refs == want
    assert picked["ignored"] == []


@pytest.mark.parametrize("ask, ignored", [
    # a bare 4 is not 400 (or 0.04); it is named, not matched at a magnitude
    ("How many posts 4?", ["4"]),
    ("How many posts 0.18?", ["0.18"]),
], ids=["4-not-400", "0.18-not-18"])
def test_a_bare_number_in_no_row_is_named_ignored_not_read_at_a_magnitude(tmp_path, ask, ignored):
    rows = _rows(tmp_path, BILL_PLAIN_NUMBERS, "plain")
    picked = agg.selection_from_ask(rows, ask)
    assert picked["sizes"] == [] and picked["ignored"] == ignored


@pytest.mark.parametrize("ask, want", [
    ("How many chambers 1800 x 1800?", []),
    ("How many pits 1800 x 1800?", ["P1", "P2"]),
], ids=["metres-bill", "millimetres-bill"])
def test_a_bare_dimension_matches_only_the_numbers_it_prints(tmp_path, ask, want):
    bill, name = ((BILL_HEADING_METRES, "metres") if "chambers" in ask
                  else (BILL_DESC_MILLIMETRES, "millimetres"))
    rows = _rows(tmp_path, bill, name)
    picked = agg.selection_from_ask(rows, ask)
    assert (picked["ignored"] == ["1800 x 1800"]) is (not want)
    if want:
        assert _refs(rows, ask)[0] == want


@pytest.mark.parametrize("ask, ignored", [
    ("How many chambers 3 x 3 m?", ["3 x 3 m"]),
    ("How many chambers 750 x 750 mm?", ["750 x 750 mm"]),
], ids=["metres", "millimetres"])
def test_a_size_in_no_row_selects_nothing_and_is_named_ignored(tmp_path, ask, ignored):
    rows = _rows(tmp_path, BILL_HEADING_METRES, "metres")
    picked = agg.selection_from_ask(rows, ask)
    assert picked["sizes"] == [] and picked["ignored"] == ignored
    # the word still selects; the ignored size is said, not silently dropped
    out = agg._aggregate_bill(rows, terms=None, section=None, measure="quantity", ask=ask)
    assert out["count"] == 4
    assert ignored[0] in out["statement"]


@pytest.mark.parametrize("ask, want", [
    ("How many chambers 2.4 x 1.8 m?", ["A4"]),
    ("How many chambers 1.2 x 1.2 m?", ["A3"]),
], ids=["2.4x1.8", "1.2x1.2"])
def test_a_size_is_not_split_into_its_digits(tmp_path, ask, want):
    # '1.8' must not become the words '1' and '8' (which every 1.8 row prints)
    rows = _rows(tmp_path, BILL_HEADING_METRES, "metres")
    refs, _, picked = _refs(rows, ask)
    assert refs == want
    assert not any(w.isdigit() for w in picked["terms"] + picked["section_terms"])


@pytest.mark.parametrize("text, metres, printed", [
    ("1.8 x 1.8 m", (1.8, 1.8), (1.8, 1.8)),
    ("1,800 x 1,800 mm", (1.8, 1.8), (1800.0, 1800.0)),
    ("1800 x 1800", None, (1800.0, 1800.0)),
    ("DN150 pipe", None, None),
    ("1.800,5 m", (1800.5,), (1800.5,)),
], ids=["metres", "mm-thousands", "unitless", "code-not-size", "decimal-comma"])
def test_sizes_in_reads_to_metres_only_with_a_unit(text, metres, printed):
    found = agg.sizes_in(text)
    if printed is None:
        assert found == []
    else:
        assert [(s.metres, s.printed) for s in found] == [(metres, printed)]
