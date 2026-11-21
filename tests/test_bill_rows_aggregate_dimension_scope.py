"""The dimension an ask totals, read from the row's own Unit cell.

A total of a stated dimension ("the total billed length of ...") must not add
rows that measure something else: a row counted in units (``No``, ``Nr``), a
section's carry total (``sum``) and a row with no figure are not lengths. The
same vocabulary reads the ask's noun and the row's unit cell, so the rule is
one predicate for every document, every bill and every dimension -- nothing
here names a project, a document, a bill, a size or a figure from a real
tender, and every test drives the decision by at least two distinct inputs.
"""

from __future__ import annotations

import pytest

from app.core import bill_aggregate as agg
from app.core import bill_table_geometry as g

NET = "SURFACE WATER DRAINAGE"
BILL = "Page S/7/2"


def _row(path, desc, unit="", qty="", rate="", amount="", page=1):
    fields = {"description": desc}
    for k, v in (("unit", unit), ("quantity", qty), ("rate", rate), ("total", amount)):
        if v:
            fields[k] = v
    return g.BillRow(section=path[-1], source_page=page, fields=fields, path=tuple(path))


def _sheet(tmp_path, rows, name="bill.xlsx"):
    path = tmp_path / name
    g.write_bill_workbook(rows, str(path))
    return agg.read_sheet_rows(str(path))


def _mixed_sheet(depth: str, first: str, second: str) -> list:
    """One band's rows, every one of them printing the ask's item words and
    differing only in what its own Unit cell measures and whether it carries a
    figure. The two length rows are the only lengths."""
    path = (BILL, NET, f"Depth {depth}")
    return [
        _row(path, "A |Storm water pipe, first run", "m", first, "1", first),
        _row(path, "B |Storm water pipe, second run", "m", second, "1", second),
        _row(path, "C |Storm water pipe, riser to chamber", "No", "4", "1", "4"),
        _row(path, "D |Storm water pipe, catch pit", "Nr", "2", "1", "2"),
        _row(path, "E |Storm water pipe, carried to collection", "sum", "", "", "10"),
        _row(path, "F |Storm water pipe, unmeasured run", "", "22", "1", "22"),
    ]


@pytest.mark.parametrize("depth, first, second", [
    ("2.5 to 3m", "500", "300"),
    ("4.0 to 4.5m", "700", "150"),
], ids=["upper-band", "lower-band"])
def test_a_length_total_admits_only_the_rows_that_measure_a_length(tmp_path, depth, first, second):
    # The band admits six rows, four of which do not measure a length: a count
    # in units, a count in another spelling, the section's carry total and a
    # row whose unit cell is blank. Only the two metre rows are the length.
    sheet = _sheet(tmp_path, _mixed_sheet(depth, first, second))
    out = agg.aggregate(sheet, terms="storm water pipe",
                        ask=f"What is the total billed length of storm water pipe at {depth} depth?")
    assert (out["count"], out["sum"]) == (2, float(first) + float(second))
    assert {row["unit"] for row in out["rows"]} == {"m"}
    assert out["scope"]["dimension"] == "length"


@pytest.mark.parametrize("depth", ["2.5 to 3m", "4.0 to 4.5m"], ids=["upper-band", "lower-band"])
def test_a_count_ask_is_not_narrowed_to_the_unit_a_row_measures_in(tmp_path, depth):
    # No dimension is stated: "how many" asks for a count of the rows the band
    # and the words admit, whatever each row's own unit measures. The set is
    # the whole band (the mixed units leave the single sum unsettled), so the
    # mechanism must not silently drop the counted rows.
    sheet = _sheet(tmp_path, _mixed_sheet(depth, "500", "300"))
    out = agg.aggregate(sheet, terms="storm water pipe",
                        ask=f"How many storm water pipe items at {depth} depth are billed?")
    assert out["count"] == 6
    assert out["sum"] is None
    assert out["scope"]["dimension"] is None


@pytest.mark.parametrize("depth", ["2.5 to 3m", "4.0 to 4.5m"], ids=["upper-band", "lower-band"])
def test_a_total_that_states_no_dimension_is_not_narrowed(tmp_path, depth):
    # "The total" of a band says nothing about which measurement is totalled:
    # the band's own unit is a length, but counting what lies in the band is as
    # legitimate as measuring it, so no unit family may be inferred.
    sheet = _sheet(tmp_path, _mixed_sheet(depth, "500", "300"))
    out = agg.aggregate(sheet, terms="storm water pipe",
                        ask=f"What is the total of storm water pipe at {depth} depth?")
    assert out["count"] == 6
    assert set(out["sum_by_unit"]) == {"m", "No", "Nr", ""}


@pytest.mark.parametrize("place", ["north area", "site area"], ids=["one-place", "other-place"])
def test_a_dimension_word_out_of_measuring_position_scopes_nothing(tmp_path, place):
    # "Area" in "the north area" names a place, not a measurement. Read as a
    # dimension it would admit only rows whose unit is square metres and empty
    # this band, so a noun only counts behind a total word or before "of".
    sheet = _sheet(tmp_path, _mixed_sheet("2.5 to 3m", "500", "300"))
    out = agg.aggregate(
        sheet, terms="storm water pipe",
        ask=f"How many storm water pipe segments in the {place} at 2.5 to 3m depth are billed?",
    )
    assert out["count"] == 6
    assert out["scope"]["dimension"] is None


@pytest.mark.parametrize("depth", ["2.5 to 3m", "4.0 to 4.5m"], ids=["upper-band", "lower-band"])
def test_the_unit_cell_is_read_the_way_the_document_prints_it(tmp_path, depth):
    # A Unit cell carries the print's case, spacing and punctuation: "m?",
    # "m>" and " M " are one unit, and the section's "| sum" is not a length.
    # The four length rows are all admitted and all carry their figure; the
    # carry line is in neither set. (The sum is reported per printed unit, so
    # four spellings are four keys -- the binding, not the reporting, is what
    # this test decides.)
    path = (BILL, NET, f"Depth {depth}")
    rows = [
        _row(path, f"A |Storm water pipe {i}", unit, "100", "1", "100")
        for i, unit in enumerate(("m?", "m>", " M ", "m"))
    ] + [_row(path, "E |Storm water pipe total", "| sum", "", "", "400")]
    out = agg.aggregate(_sheet(tmp_path, rows), terms="storm water pipe",
                        ask=f"What is the total billed length of storm water pipe at {depth} depth?")
    assert out["count"] == 4
    assert set(out["sum_by_unit"]) == {"m?", "m>", "M", "m"}
    assert sum(out["sum_by_unit"].values()) == 400.0
    assert out["without_figure"] == []


@pytest.mark.parametrize("depth", ["2.5 to 3m", "4.0 to 4.5m"], ids=["upper-band", "lower-band"])
def test_a_length_row_without_the_figure_is_not_in_the_total(tmp_path, depth):
    # A row that measures a length but prints no quantity states no length to
    # add: its amount alone cannot enter a length total.
    path = (BILL, NET, f"Depth {depth}")
    rows = [
        _row(path, "A |Storm water pipe, measured", "m", "100", "1", "100"),
        _row(path, "B |Storm water pipe, amount only", "m", "", "", "999"),
    ]
    out = agg.aggregate(_sheet(tmp_path, rows), terms="storm water pipe",
                        ask=f"What is the total billed length of storm water pipe at {depth} depth?")
    assert (out["count"], out["sum"]) == (1, 100.0)


@pytest.mark.parametrize("ask, unit, count", [
    ("What is the total billed volume of storm water pipe at 2.5 to 3m depth?", "m3", 2),
    ("What is the total billed weight of storm water pipe at 2.5 to 3m depth?", "kg", 2),
], ids=["volume", "weight"])
def test_another_dimension_is_read_by_the_same_vocabulary(tmp_path, ask, unit, count):
    # The mechanism is the vocabulary, not one unit: cubic metres answer a
    # volume total and kilograms a weight total, and the counted row in the
    # same band answers neither.
    path = (BILL, NET, "Depth 2.5 to 3m")
    rows = [
        _row(path, "A |Storm water pipe, first run", unit, "5", "1", "5"),
        _row(path, "B |Storm water pipe, second run", unit, "7", "1", "7"),
        _row(path, "C |Storm water pipe, riser to chamber", "No", "4", "1", "4"),
    ]
    out = agg.aggregate(_sheet(tmp_path, rows), terms="storm water pipe", ask=ask)
    assert (out["count"], out["sum"]) == (count, 12.0)


@pytest.mark.parametrize("ask, unit", [
    ("What is the total billed length of storm water pipe at 2.5 to 3m depth?", "m"),
    ("What is the total billed volume of storm water pipe at 2.5 to 3m depth?", "m3"),
], ids=["length", "volume"])
def test_the_same_dimension_ask_selects_the_same_rows_whatever_terms_arrive(tmp_path, ask, unit):
    # The model's terms drift from trial to trial; the ask does not. The row
    # set follows the ask, so the same ask selects the same rows every time.
    path = (BILL, NET, "Depth 2.5 to 3m")
    rows = [
        _row(path, "A |Storm water pipe, first run", unit, "5", "1", "5"),
        _row(path, "B |Storm water pipe, second run", unit, "7", "1", "7"),
        _row(path, "C |Storm water pipe, riser to chamber", "No", "4", "1", "4"),
    ]
    sheet = _sheet(tmp_path, rows)
    trials = [
        agg.aggregate(sheet, terms=terms, ask=ask)["rows"]
        for terms in ("", "storm, water, pipe", "riser, chamber, depth", "5, 7, 4")
    ]
    first = [(row["description"], row["unit"]) for row in trials[0]]
    assert [[(row["description"], row["unit"]) for row in t] for t in trials] == [first] * len(trials)
    assert len(first) == 2
