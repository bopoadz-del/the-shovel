"""The scope an ask states, read as numbers and as the row's own cells.

A count or a total over bill rows must not depend on the words the model
happened to pass on the turn: the same ask has to select the same rows every
time. These tests build their own workbooks (the real derived-sheet layout,
written through ``write_bill_workbook``) and drive every decision by at least
two distinct inputs.

Nothing here names a project, a document, a bill, a size or a figure from a
real tender: the numbers and the labels are invented for the case.
"""

from __future__ import annotations

import pytest

from app.core import bill_aggregate as agg
from app.core import bill_table_geometry as g

#: Two heading levels that name a network and a section, and a page label that
#: names a bill the way a document's own page numbering does (part/bill/page).
NET = "STORM WATER NETWORK"
OTHER = "ROADS AND PAVEMENTS"
BILL_SEVEN = "DGII - Procurement of Contractor, Infrastructure Package 1 Page S/7/2"
BILL_NINE = "DGII - Procurement of Contractor, Infrastructure Package 1 Page S/9/1"
BILL_SEVEN_LABEL = "DGII - Procurement of Contractor, Infrastructure Package 1 Page S/7/3"


def _row(path, ref, desc, unit="", qty="", rate="", amount="", page=1):
    fields = {"description": desc}
    for k, v in (("unit", unit), ("quantity", qty), ("rate", rate), ("total", amount)):
        if v:
            fields[k] = v
    return g.BillRow(section=path[-1], source_page=page, fields=fields,
                     extra={"Item": ref} if ref else {}, path=tuple(path))


def _sheet(tmp_path, rows, name="bill.xlsx"):
    path = tmp_path / name
    g.write_bill_workbook(rows, str(path))
    return agg.read_sheet_rows(str(path))


#: One sheet, both manhole families, and the row that collides with a 1.5 ask
#: on words alone: "depth 5.0-5.5m" splits into "5" and "5m", and the item code
#: carries the bare digit the ask's size also prints.
MANHOLES = [
    _row((BILL_NINE, NET, "Box Structure Manholes"), "", "A |Size 1.5 x 1.5m, depth 1.5-2.0m K 2", "No", "4", "10", "40"),
    _row((BILL_NINE, NET, "Box Structure Manholes"), "", "B |Size 1.5 x 1.5m, depth 4.0-4.5m K137.1", "No", "7", "10", "70"),
    _row((BILL_NINE, NET, "Box Structure Manholes"), "", "A |Size 1.8 x 1.8m, depth 2.0-2.5m K 3", "No", "2", "10", "20"),
    _row((BILL_NINE, NET, "Box Structure Manholes"), "", "B |Size 1.8 x 1.8m, depth 5.0-5.5m K137.3", "No", "6", "10", "60"),
    _row((BILL_NINE, NET, "Box Structure Manholes"), "", "C |Size 1.8 x 1.8m, depth 4.0-4.5m K137.5", "No", "9", "10", "90"),
]


@pytest.mark.parametrize("ask, size, count, total", [
    ("How many 1.5 x 1.5 m box structure manholes are billed, all depths together?", [1.5, 1.5], 2, 11.0),
    ("How many 1.8 x 1.8 m box structure manholes are billed, all depths together?", [1.8, 1.8], 3, 17.0),
], ids=["smaller-family", "larger-family"])
def test_a_size_is_compared_as_numbers_not_as_words(tmp_path, ask, size, count, total):
    out = agg.aggregate(_sheet(tmp_path, MANHOLES), terms="1, 5, 5m, x, 8", ask=ask)
    assert (out["count"], out["sum"]) == (count, total)
    assert out["scope"]["size"] == size


@pytest.mark.parametrize("ask, count, total", [
    ("How many 1.2 x 1.2 m chambers are billed?", 2, 5.0),
    ("How many 1300 x 1300 mm chambers are billed?", 1, 3.0),
], ids=["metres-ask", "millimetres-ask"])
def test_a_dimension_is_read_in_the_unit_it_states(tmp_path, ask, count, total):
    rows = [
        _row((BILL_NINE, NET, "Chambers"), "", "A |Size 1200 x 1200 mm, depth 1.5-2.0m", "No", "2", "1", "2"),
        _row((BILL_NINE, NET, "Chambers"), "", "B |Size 1.2 x 1.2m, depth 2.0-2.5m", "No", "3", "1", "3"),
        _row((BILL_NINE, NET, "Chambers"), "", "C |Size 1300 x 1300 mm, depth 2.5-3.0m", "No", "3", "1", "3"),
        _row((BILL_NINE, NET, "Chambers"), "", "D |Size 1.5 x 1.5m, depth 3.0-3.5m", "No", "4", "1", "4"),
    ]
    out = agg.aggregate(_sheet(tmp_path, rows), terms="", ask=ask)
    assert (out["count"], out["sum"]) == (count, total)


@pytest.mark.parametrize("depth, qty", [("1.5-2.0m", 4), ("4.0-4.5m", 7)], ids=["upper", "lower"])
def test_a_band_is_the_band_cell_of_the_rows_own_line(tmp_path, depth, qty):
    # The sheet head says 4.0-4.5m; each row's own description says its band.
    # A row whose depth label disagrees with its sheet head is read from its
    # OWN cell, never from the head or from the row above it.
    path = (BILL_NINE, NET, "Box Structure Manholes at 4.0-4.5m")
    rows = [
        _row(path, "", "A |Size 1.5 x 1.5m, depth 1.5-2.0m K 2", "No", "4", "10", "40"),
        _row(path, "", "B |Size 1.5 x 1.5m, depth 4.0-4.5m K137.1", "No", "7", "10", "70"),
    ]
    out = agg.aggregate(_sheet(tmp_path, rows), terms="", ask=f"How many at {depth} depth are billed?")
    assert (out["count"], out["sum"]) == (1, float(qty))


@pytest.mark.parametrize("ask, count, total", [
    ("In Bill No. 9, what is the total length at 2.5 to 3 m depth?", 2, 800.0),
    ("In Bill No. 7, what is the total length at 2.5 to 3 m depth?", 3, 300.0),
], ids=["asking-bill", "other-bill"])
def test_a_band_that_names_a_bill_admits_only_that_bills_rows(tmp_path, ask, count, total):
    # The same band label on two bills' sheets, and one bill's row on a sheet
    # in another network: the ask's bill decides, and nothing unions the two.
    rows = [
        _row((BILL_NINE, NET, "Depth 2.5 to 3m"), "", "A |Nominal bore: 400mm", "m", "500", "1", "500"),
        _row((BILL_NINE, NET, "Depth 2.5 to 3m"), "", "B |Nominal bore: 600mm", "m", "300", "1", "300"),
        _row((BILL_SEVEN, OTHER, "Depth 2.5 to 3m"), "", "A |Nominal bore: 400mm", "m", "100", "1", "100"),
        _row((BILL_SEVEN, OTHER, "Depth 2.5 to 3m"), "", "B |Nominal bore: 600mm", "m", "150", "1", "150"),
        _row((BILL_SEVEN_LABEL, OTHER, "Depth 2.5 to 3m"), "", "C |Nominal bore: 900mm", "m", "50", "1", "50"),
    ]
    out = agg.aggregate(_sheet(tmp_path, rows), terms="", ask=ask)
    assert (out["count"], out["sum"]) == (count, total)


@pytest.mark.parametrize("ask, count, total", [
    ("What is the total length of storm water pipe at 2.5 to 3 m depth?", 2, 800.0),
    ("What is the total length of roads pipe at 2.5 to 3 m depth?", 3, 300.0),
], ids=["network-in-the-ask", "other-network"])
def test_the_band_alone_never_unions_two_bills(tmp_path, ask, count, total):
    # No bill is named: the network the ask names is printed in the heading
    # path of exactly one bill's rows, so those are the rows admitted.
    rows = [
        _row((BILL_NINE, NET, "Depth 2.5 to 3m"), "", "A |Nominal bore: 400mm", "m", "500", "1", "500"),
        _row((BILL_NINE, NET, "Depth 2.5 to 3m"), "", "B |Nominal bore: 600mm", "m", "300", "1", "300"),
        _row((BILL_SEVEN, OTHER, "Depth 2.5 to 3m"), "", "A |Nominal bore: 400mm", "m", "100", "1", "100"),
        _row((BILL_SEVEN, OTHER, "Depth 2.5 to 3m"), "", "B |Nominal bore: 600mm", "m", "150", "1", "150"),
        _row((BILL_SEVEN_LABEL, OTHER, "Depth 2.5 to 3m"), "", "C |Nominal bore: 900mm", "m", "50", "1", "50"),
    ]
    out = agg.aggregate(_sheet(tmp_path, rows), terms="", ask=ask)
    assert (out["count"], out["sum"]) == (count, total)


@pytest.mark.parametrize("ask", [
    "How many 1.5 x 1.5 m box structure manholes are billed?",
    "What is the total length of storm water pipe at 2.5 to 3 m depth?",
], ids=["manhole-ask", "pipe-ask"])
def test_the_same_ask_selects_the_same_rows_whatever_terms_arrive(tmp_path, ask):
    rows = MANHOLES + [
        _row((BILL_NINE, NET, "Depth 2.5 to 3m"), "", "A |Nominal bore: 400mm", "m", "500", "1", "500"),
    ]
    sheet = _sheet(tmp_path, rows)
    trials = [
        agg.aggregate(sheet, terms=terms, ask=ask)["rows"]
        for terms in ("", "1, 5, 5m, x", "bore, nominal, depth", "size, manhole, 1.8")
    ]
    first = [(r["description"], r["quantity"], r["unit"]) for r in trials[0]]
    assert [[(r["description"], r["quantity"], r["unit"]) for r in t] for t in trials] == [first] * len(trials)
    assert agg.aggregate(sheet, terms="", ask=ask)["rows"] == trials[0]


@pytest.mark.parametrize("terms, count", [("chamber", 2), ("manhole", 1)], ids=["chambers", "manholes"])
def test_an_ask_that_states_no_scope_still_uses_the_terms(tmp_path, terms, count):
    path = (BILL_NINE, NET, "Chambers")
    rows = [
        _row(path, "", "A |Inspection chamber", "No", "2", "1", "2"),
        _row(path, "", "B |Inspection chamber, deeper", "No", "3", "1", "3"),
        _row(path, "", "C |Manhole ring", "No", "4", "1", "4"),
    ]
    sheet = _sheet(tmp_path, rows)
    with_ask = agg.aggregate(sheet, terms=terms, ask="What does the bill allow for this?")
    without = agg.aggregate(sheet, terms=terms)
    assert with_ask["count"] == count == without["count"]
    assert with_ask["scope"] is None


@pytest.mark.parametrize("noise", ["escalation", "wayleave"], ids=["one-word", "other-word"])
def test_a_word_the_document_never_prints_does_not_empty_the_answer(tmp_path, noise):
    rows = [
        _row((BILL_NINE, NET, "Chambers"), "", "A |Size 1.5 x 1.5m, depth 1.5-2.0m", "No", "2", "1", "2"),
        _row((BILL_NINE, NET, "Chambers"), "", "B |Size 1.5 x 1.5m, depth 2.0-2.5m", "No", "3", "1", "3"),
    ]
    plain = agg.aggregate(_sheet(tmp_path, rows), terms="", ask="How many 1.5 x 1.5 m chambers?")
    noisy = agg.aggregate(_sheet(tmp_path, rows, name="b2.xlsx"), terms="",
                          ask=f"How many 1.5 x 1.5 m chambers, and what {noise} applies?")
    assert noisy["count"] == plain["count"] == 2
    assert noisy["sum"] == plain["sum"] == 5.0


@pytest.mark.parametrize("ask, count, total", [
    ("How many pipe bores at 2.5 to 3 m depth are billed?", 2, 800.0),
    ("How many pipe bores at 2.5 to 3 m depth, in metres?", 2, 800.0),
], ids=["plural-in-the-ask", "plural-again"])
def test_a_plural_in_the_ask_binds_the_singular_the_row_prints(tmp_path, ask, count, total):
    # "bores" in the ask, "bore" on two rows and a row that prints neither:
    # the ask's word is folded, so it binds the rows that print it and leaves
    # the row that does not.
    path = (BILL_NINE, NET, "Depth 2.5 to 3m")
    rows = [
        _row(path, "", "A |Nominal bore: 400mm", "m", "500", "1", "500"),
        _row(path, "", "B |Nominal bore: 600mm", "m", "300", "1", "300"),
        _row(path, "", "C |Cable culvert, 3.6m wide", "m", "700", "1", "700"),
    ]
    out = agg.aggregate(_sheet(tmp_path, rows), terms="", ask=ask)
    assert (out["count"], out["sum"]) == (count, total)


@pytest.mark.parametrize("ask", [
    "In Bill No. 7, how many chambers are billed?",
    "For bill 7, how many chambers are billed?",
], ids=["heading-form", "lower-case"])
def test_the_word_bill_in_the_ask_names_no_row(tmp_path, ask):
    # One sheet's rows name their bill in a heading, the other's in a page
    # label. Both are bill 7, so both are admitted: "bill" is how the ask asks,
    # not a word a row has to print.
    rows = [
        _row(("Bill No. 7 - Drainage Network", "Chambers"), "", "A |Inspection chamber", "No", "2", "1", "2"),
        _row((BILL_SEVEN_LABEL, OTHER, "Chambers"), "", "B |Inspection chamber, deeper", "No", "3", "1", "3"),
    ]
    out = agg.aggregate(_sheet(tmp_path, rows), terms="", ask=ask)
    assert (out["count"], out["sum"]) == (2, 5.0)


@pytest.mark.parametrize("ask", ["How many 1.5 x 1.5 m chambers?", "How many at 2.5 to 3 m depth?"],
                         ids=["size", "band"])
def test_an_empty_output_set_states_no_scope(tmp_path, ask):
    assert agg.ask_scope(ask, []) is None
    row = _row((BILL_NINE, NET, "Chambers"), "", "A |Size 1.5 x 1.5m, depth 2.5-3.0m", "No", "2", "1", "2")
    assert agg.ask_scope(ask, _sheet(tmp_path, [row])) is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("ask, count", [
    ("How many 1.5 x 1.5 m box structure manholes are billed?", 2),
    ("In Bill No. 7, how many 1.5 x 1.5 m box structure manholes are billed?", 1),
], ids=["in-scope-bill", "named-bill"])
async def test_the_tool_scopes_by_the_ask_the_user_wrote(tmp_path, monkeypatch, ask, count):
    # The model's terms are the drifting token set; the ask is the same on every
    # trial. The handler hands the ask to the module, so the row set follows it.
    from app.agents.base.tools import handle_bill_rows_aggregate
    from app.agents.core.tool_registry import ToolCall
    from app.agents.runtime import Agent
    from app.core import projects

    rows = [
        _row((BILL_NINE, NET, "Box Structure Manholes"), "", "A |Size 1.5 x 1.5m, depth 1.5-2.0m", "No", "2", "1", "2"),
        _row((BILL_SEVEN, OTHER, "Box Structure Manholes"), "", "A |Size 1.5 x 1.5m, depth 2.0-2.5m", "No", "3", "1", "3"),
    ]
    path = tmp_path / "tool.xlsx"
    g.write_bill_workbook(rows, str(path))
    monkeypatch.setattr(agg, "derived_bill_documents",
                        lambda pid, did=None: [{"id": "d1", "original_name": "tool.xlsx"}])
    monkeypatch.setattr(projects, "materialize_document_path", lambda doc: str(path))
    agent = Agent(name="test-agent", description="t", system_prompt="t", allowed_blocks=[])
    out = await handle_bill_rows_aggregate(ToolCall(
        agent=agent, name="bill_rows_aggregate", args={"terms": "1, 5, 5m, x"},
        project_id="p", user_message=ask,
    ))
    assert out["ok"] is True
    assert out["result"]["bills"][0]["count"] == count
