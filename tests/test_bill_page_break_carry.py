"""#411 grader 1420Z point 2: a section that runs over a page break keeps the
heading that was open when the page broke (the sheet READER; nothing stored
changes). Every bill here is invented in the test.

The extractor's heading path carries over a page; a label re-printed at the
head of the next page (the bill's running heading, which cuts the path short,
or the section's 'continued' line, which replaces its innermost label) moves
the rest of the section into another sheet. The reader puts those rows back.
"""

from __future__ import annotations

import pytest

from app.core import bill_aggregate as agg
from app.core import bill_table_geometry as g


def _row(path, page, ref, desc, qty, unit="nr"):
    return g.BillRow(section=path[-1], source_page=page,
                     fields={"description": desc, "unit": unit, "quantity": qty},
                     extra={"Item": ref}, path=tuple(path))


OPEN = ("Bill 4 Drainage", "Chambers 1.8 x 1.8 m")
RUNNING = ("Bill 4 Drainage",)                                  # running heading cut the path
CONTINUED = ("Bill 4 Drainage", "Chambers 1.8 x 1.8 m continued")  # a 'continued' line
NEXT = ("Bill 4 Drainage", "Gullies")


def _bill(head_of_page_3, first_ref_on_3):
    """Chambers on page 2 (refs D1-D3), the rest of the section on page 3
    under ``head_of_page_3``, then a new section on page 3 and page 4."""
    return [
        _row(OPEN, 2, "D1", "Chamber depth n.e. 1.5 m", "4"),
        _row(OPEN, 2, "D2", "Chamber depth 1.5 to 2 m", "6"),
        _row(OPEN, 2, "D3", "Chamber depth 2 to 3 m", "5"),
        _row(head_of_page_3, 3, first_ref_on_3, "Chamber depth 3 to 4 m", "2"),
        _row(head_of_page_3, 3, "", "Chamber depth over 4 m", "1"),
        _row(NEXT, 3, "D6", "Road gully", "9"),
        _row(NEXT, 4, "D7", "Yard gully", "3"),
    ]


def _read(tmp_path, bill):
    path = tmp_path / "bill.xlsx"
    g.write_bill_workbook(bill, str(path))
    return agg.read_sheet_rows(str(path))


@pytest.mark.parametrize("head", [RUNNING, CONTINUED], ids=["running-heading", "continued-line"])
def test_a_section_over_a_page_break_is_counted_and_summed_whole(tmp_path, head):
    rows = _read(tmp_path, _bill(head, "D4"))
    out = agg._aggregate_bill(rows, terms=None, section=None, measure="quantity",
                              ask="How many chambers 1.8 x 1.8 m? total quantity")
    assert out["count"] == 5
    assert out["sum"] == 18.0
    assert [p["section"] for p in out["by_section"].values()] == [list(OPEN)]
    # the next section is untouched
    gullies = agg._aggregate_bill(rows, terms=None, section=None, measure="quantity",
                                  ask="How many gullies?")
    assert gullies["count"] == 2 and gullies["sum"] == 12.0


@pytest.mark.parametrize("head, first_ref, carried", [
    # the item numbers go on (D3 -> D4): the evidence; the open heading is kept
    (RUNNING, "D4", True),
    (CONTINUED, "D4", True),
    # they restart (D3 -> D1) or jump (D3 -> D9): a new section
    (RUNNING, "D1", False),
    (CONTINUED, "D9", False),
    # no reference, or one with no number: no evidence, NOT carried, even
    # under a re-printed label (grader 1515Z point 3: a wrong carry inflates a sum)
    (CONTINUED, "", False),
    (RUNNING, "", False),
    (RUNNING, "item", False),
    # the next number under another prefix (D3 -> E4) is another sequence
    (RUNNING, "E4", False),
], ids=["running-sequence-continues", "continued-sequence-continues", "sequence-restarts",
        "sequence-jumps", "continued-no-ref", "running-heading-alone-no-ref", "ref-without-number",
        "other-prefix"])
def test_only_the_item_number_sequence_carries_a_block(tmp_path, head, first_ref, carried):
    rows = _read(tmp_path, _bill(head, first_ref))
    on_3 = [r.path for r in rows if r.pages == [3] and "Chamber" in r.description]
    assert on_3 == [OPEN if carried else head] * 2


@pytest.mark.parametrize("bill, paths", [
    # a new heading at the head of the page: not a continuation
    ([_row(OPEN, 2, "D1", "Chamber", "1"), _row(NEXT, 3, "D2", "Gully", "1")], [OPEN, NEXT]),
    # the same running heading, but not at the next page (a page between): kept
    ([_row(OPEN, 2, "D1", "Chamber", "1"), _row(NEXT, 3, "D2", "Gully", "1"),
      _row(RUNNING, 5, "D3", "Chamber", "1")], [OPEN, NEXT, RUNNING]),
], ids=["new-heading", "not-the-next-page"])
def test_only_the_head_of_the_next_page_continues_the_open_heading(tmp_path, bill, paths):
    assert [r.path for r in _read(tmp_path, bill)] == paths


@pytest.mark.parametrize("path, open_path, expected", [
    (("A",), ("A", "B"), True),
    (("A", "B more"), ("A", "B"), True),
    (("A", "C"), ("A", "B"), False),
    (("X",), ("A", "B"), False),
    (("A", "B"), ("A", "B"), False),
    ((), ("A",), False),
], ids=["cut-short", "continued", "sibling", "other-bill", "same", "empty"])
def test_continues(path, open_path, expected):
    assert agg._continues(path, open_path) is expected
