"""The converter's index -> body bill-number recovery (bill_table_geometry).

Some bills print their section heading in the body without a number; the
number lives only in the document's own index ("BILL NO. 6 - ELECTRICAL
NETWORK"). The section pass builds a number<->title map from heading-shaped
lines (labels or measured row descriptions) and prefixes an unnumbered
heading with its index number. Every input is invented here from word boxes
or labels; nothing names a real bill, project or document.
"""

from __future__ import annotations

import pytest

from app.core import bill_table_geometry as g

H = 20.0


def _lab(text, page=1, top=0.0, above=False):
    return g._Label(text, 40.0, H, page=page, top=top, bottom=top + H, above_header=above)


def _row(page, description="Item", labels=()):
    return g.BillRow(
        section="",
        source_page=page,
        fields={"description": description},
        labels=tuple(_lab(t, page=page) for t in labels),
    )


@pytest.mark.parametrize("rows, printed", [
    ([_row(1, "BILL NO. 6 - ELECTRICAL NETWORK"), _row(1, "Bill Number 8 Street Lighting Network")],
     []),
    ([], [_lab("General Summary of Quantities"),
          _lab("BILL NO. 6 - ELECTRICAL NETWORK"),
          _lab("Bill # 8 STREET LIGHTING NETWORK"),
          _lab("BILL NO. 9 - 123")]),  # figure-only title and non-heading label map nothing
], ids=["summary-as-rows", "summary-as-labels"])
def test_index_number_map_from_rows_and_labels(rows, printed):
    assert g.index_number_map(rows, printed) == {
        "electrical network": "6",
        "street lighting network": "8",
    }


def _section_scenario(with_index):
    """Four pages: a running header, a page-number-only label, and the body
    heading printed in the title band. ``with_index`` adds the document's own
    index entry carrying the number on page 1."""
    pages = 4
    printed = []
    for p in range(1, pages + 1):
        printed.append(_lab("Contractor Limited", page=p, top=0.0))
        printed.append(_lab(f"Page {p}", page=p, top=34.0))
    if with_index:
        printed.append(_lab("BILL NO. 6 - ELECTRICAL NETWORK", page=1, top=68.0, above=True))
    for p in range(2, pages + 1):
        printed.append(_lab("ELECTRICAL NETWORK", page=p, top=68.0, above=True))
    rows = [_row(p, description=f"Item {p}") for p in range(2, pages + 1)]
    return rows, printed, pages


@pytest.mark.parametrize("with_index, expected", [
    (True, ("Bill No. 6 - ELECTRICAL NETWORK",)),
    (False, ("ELECTRICAL NETWORK",)),
], ids=["indexed-heading-adopts-number", "unindexed-heading-unchanged"])
def test_unnumbered_body_heading_adopts_index_number(with_index, expected):
    rows, printed, pages = _section_scenario(with_index)
    g.section_paths(rows, printed, pages)
    assert [r.path for r in rows] == [expected, expected, expected]


def _map(*headings):
    return g.index_number_map([], [_lab(h) for h in headings])


@pytest.mark.parametrize("headings, expected", [
    (("BILL NO. 1 - GENERAL ITEMS", "BILL NO. 6 - ELECTRICAL NETWORK"),
     {"general items": "1", "electrical network": "6"}),
    (("BILL NO. 1 - GENERAL ITEMS", "BILL NO. 2 - GENERAL ITEMS", "BILL NO. 6 - ELECTRICAL NETWORK"),
     {"electrical network": "6"}),  # GENERAL ITEMS is ambiguous -> dropped
], ids=["unique-titles", "duplicate-title-ambiguous"])
def test_duplicate_index_title_is_ambiguous(headings, expected):
    assert _map(*headings) == expected


@pytest.mark.parametrize("text, index, expected", [
    ("BILL NO. 5 - ROADS AND PAVINGS", {"roads and pavings": "5"}, "BILL NO. 5 - ROADS AND PAVINGS"),
    ("ROADS AND PAVINGS", {"roads and pavings": "5"}, "Bill No. 5 - ROADS AND PAVINGS"),
], ids=["already-numbered", "unnumbered-adopts-number"])
def test_already_numbered_heading_is_untouched(text, index, expected):
    assert g._numbered(text, index) == expected
