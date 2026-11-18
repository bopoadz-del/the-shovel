"""#411 (b): when the user's words match rows in two sections of a bill, the
tool states each section's own total (named by its heading path) and the
total of all, so the user sees what was added and can say which was meant.
An ask whose rows sit in one section states one total, as before.

Every bill here is invented in the test; nothing names a real bill, row or
question. Each decision is driven by at least two inputs.
"""

from __future__ import annotations

import pytest

from app.agents.base.tools import bill_rows_statement
from app.core import bill_aggregate as agg
from app.core import bill_table_geometry as g


def _row(path, page, ref, desc, unit="", qty="", amount=""):
    fields = {"description": desc}
    for k, v in (("unit", unit), ("quantity", qty), ("total", amount)):
        if v:
            fields[k] = v
    return g.BillRow(section=path[-1], source_page=page, fields=fields,
                     extra={"Item": ref} if ref else {}, path=tuple(path))


UPPER = ("Works", "Upper runs")
LOWER = ("Works", "Lower runs")
OTHER = ("Works", "Fencing")

#: The same item (a duct) is billed in two sections; a third section has none.
BILL = [
    _row(UPPER, 1, "U1", "Steel duct 100 wide", "m", "40", "400"),
    _row(UPPER, 1, "U2", "Steel duct 200 wide", "m", "25", "500"),
    _row(LOWER, 2, "L1", "Steel duct 100 wide", "m", "7", "70"),
    _row(LOWER, 2, "L2", "Copper duct 300 wide", "m", "3", "90"),
    _row(OTHER, 3, "F1", "Timber rail", "m", "120", "3000"),
]


@pytest.fixture
def rows(tmp_path):
    path = tmp_path / "bill.xlsx"
    g.write_bill_workbook(BILL, str(path))
    return agg.join_page_continuations(agg.read_sheet_rows(str(path)))


def _result(rows, ask):
    picked = agg.selection_from_ask(rows, ask)
    out = agg.aggregate(rows, terms=picked["terms"], section=picked["section_terms"],
                        measure=agg.measure_from_ask(ask, agg.MEASURE_QUANTITY), fold=True,
                        sizes=picked["sizes"])
    out.update(status="success", filename="bill.xlsx")
    return out


@pytest.mark.parametrize("ask, sections, total", [
    # words that match rows of two sections: one entry per section, bill order
    ("What is the total length of steel ducts?",
     [(["Works", "Upper runs"], 2, 65.0), (["Works", "Lower runs"], 1, 7.0)], 72.0),
    ("What is the total cost of ducts 100 wide?",
     [(["Works", "Upper runs"], 1, 400.0), (["Works", "Lower runs"], 1, 70.0)], 470.0),
    # words that match rows of one section: one entry
    ("What is the total length of copper ducts?", [(["Works", "Lower runs"], 1, 3.0)], 3.0),
    ("What is the total cost of the timber rail?", [(["Works", "Fencing"], 1, 3000.0)], 3000.0),
], ids=["two-steel", "two-100", "one-copper", "one-rail"])
def test_each_section_the_rows_sit_in_carries_its_own_total(rows, ask, sections, total):
    res = _result(rows, ask)
    got = [(p["section"], p["count"], p["sum"]) for p in res["by_section"].values()]
    assert got == sections
    assert res["sum"] == total
    assert sum(p["sum"] for p in res["by_section"].values()) == total


@pytest.mark.parametrize("ask, named, absent", [
    ("What is the total length of steel ducts?",
     ["Works > Upper runs: 2 row(s), total quantity 65 m",
      "Works > Lower runs: 1 row(s), total quantity 7 m",
      "all 2 sections together: 72 m"], []),
    ("What is the total cost of ducts 100 wide?",
     ["Works > Upper runs: 1 row(s), total amount 400",
      "Works > Lower runs: 1 row(s), total amount 70",
      "all 2 sections together: 470"], []),
    ("What is the total length of copper ducts?", ["Total quantity 3 m"], ["sections"]),
    ("What is the total cost of the timber rail?", ["Total amount 3,000"], ["sections"]),
], ids=["two-steel", "two-100", "one-copper", "one-rail"])
def test_the_sentence_names_both_sections_and_the_total_of_all(rows, ask, named, absent):
    text = agg.statement(_result(rows, ask))
    for part in named:
        assert part in text
    for part in absent:
        assert part not in text


@pytest.mark.parametrize("ask, sums", [
    # two sections: each section's total, then the total of all
    ("What is the total length of steel ducts?",
     [(65.0, ("Works > Upper runs",), ("U1 (p.1)", "U2 (p.1)")),
      (7.0, ("Works > Lower runs",), ("L1 (p.2)",)),
      (72.0, (), ("U1 (p.1)", "U2 (p.1)", "L1 (p.2)"))]),
    ("What is the total cost of ducts 100 wide?",
     [(400.0, ("Works > Upper runs",), ("U1 (p.1)",)),
      (70.0, ("Works > Lower runs",), ("L1 (p.2)",)),
      (470.0, (), ("U1 (p.1)", "L1 (p.2)"))]),
    # one section: the one total only
    ("What is the total length of copper ducts?", [(3.0, (), ("L2 (p.2)",))]),
    ("What is the total cost of the timber rail?", [(3000.0, (), ("F1 (p.3)",))]),
], ids=["two-steel", "two-100", "one-copper", "one-rail"])
def test_the_tools_statements_give_a_total_per_section_then_all(rows, ask, sums):
    stated = bill_rows_statement({"bills": [_result(rows, ask)]})
    assert stated[0].basis == "count"
    got = [(s.figure, s.within, s.rows) for s in stated if s.basis == "sum"]
    assert got == sums


@pytest.mark.parametrize("ask, units", [
    # mixed units inside one section stay per unit in that section's entry
    ("How long are the ducts and rails in total?", None),
    ("What is the total length of steel ducts?", {"m": 72.0}),
], ids=["mixed-free", "one-unit"])
def test_section_entries_keep_the_unit_rule(rows, ask, units):
    res = _result(rows, ask)
    if units is None:
        # nothing in this bill prints both words in one description
        assert res["count"] == 0 and res["by_section"] == {}
    else:
        assert res["sum_by_unit"] == units
        assert all(p["unit"] == "m" for p in res["by_section"].values())


@pytest.mark.parametrize("ask, row_lists", [
    # two sections and a row without a figure: the rows are the only lists of rows
    ("What is the total length of steel ducts?", {"rows": 3}),
    ("What is the total length of the timber rail?", {"rows": 1}),
], ids=["two-sections", "one-section"])
def test_a_bill_result_holds_no_list_of_objects_but_its_rows(rows, ask, row_lists):
    """Section totals and stated figures are figures, not rows: the result's
    size cap counts every list of objects as rows and drops it from the
    never-cut outline (#428), so only the rows may be such lists."""
    from app.agents import runtime

    res = agg._aggregate_bill(rows, ask=ask, terms="", section="", measure=agg.MEASURE_QUANTITY)
    found = {path: len(items) for path, items in runtime._row_lists(res)}
    assert found == row_lists
    assert isinstance(res["by_section"], dict) and isinstance(res["stated_figures"], dict)


@pytest.mark.parametrize("paths, keys", [
    # two heading paths printed alike stay two sections
    ([("Works", "Upper runs"), ("Works", "Upper runs", "")],
     ["Works > Upper runs", "Works > Upper runs (2)"]),
    ([("Works", "Upper runs"), ("Works", "Lower runs")],
     ["Works > Upper runs", "Works > Lower runs"]),
], ids=["printed-alike", "distinct"])
def test_section_totals_are_keyed_one_per_path(paths, keys):
    parts = [{"section": list(p), "count": 1, "sums": {"m": 2.0}} for p in paths]
    out = agg._by_section(parts, agg.MEASURE_QUANTITY)
    assert list(out) == keys
    assert [v["section"] for v in out.values()] == [list(p) for p in paths]
