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

@pytest.mark.parametrize("ask", [
    "What is the total billed length of storm water pipe?",
    "What is the billed length of storm water pipe?",
], ids=["total-cue", "of-cue"])
def test_a_heading_that_prints_the_noun_does_not_narrow_a_dimension_total(tmp_path, ask):
    # The noun "length" is printed in ONE section heading; every row prints
    # "pipe". Read also as a content word, the noun would confine the answer to
    # that heading -- and to nothing at all on a document that prints no such
    # heading -- so a noun read as a measurement is not a word to look for.
    rows = [
        _row((BILL, NET, "PIPE LENGTH"), "A |Storm water pipe, first run", "m", "40", "1", "40"),
        _row((BILL, NET, "PIPE LENGTH"), "B |Storm water pipe, chamber", "No", "4", "1", "4"),
        _row((BILL, NET, "PIPE RUNS"), "C |Storm water pipe, second run", "m", "60", "1", "60"),
        _row((BILL, NET, "PIPE RUNS"), "D |Storm water pipe, carried", "sum", "", "", "100"),
    ]
    out = agg.aggregate(_sheet(tmp_path, rows), terms="storm water pipe", ask=ask)
    assert (out["count"], out["sum"]) == (2, 100.0)
    assert {row["section"][-1] for row in out["rows"]} == {"PIPE LENGTH", "PIPE RUNS"}


@pytest.mark.parametrize("unit, noun", [("m", "length"), ("m3", "volume")], ids=["length", "volume"])
def test_a_row_that_prints_the_noun_does_not_scope_the_other_rows_of_the_total(tmp_path, unit, noun):
    # One row's own description happens to print the noun; the other row of the
    # same measurement does not. A dimension total is bound by what each row's
    # Unit cell measures, so the second row is not excluded by the first row's
    # wording.
    path = (BILL, NET, "PIPE RUNS")
    rows = [
        _row(path, f"A |Storm water pipe, billed {noun} 40", unit, "40", "1", "40"),
        _row(path, "B |Storm water pipe, second run", unit, "60", "1", "60"),
    ]
    out = agg.aggregate(_sheet(tmp_path, rows), terms="storm water pipe",
                        ask=f"What is the total billed {noun} of storm water pipe?")
    assert (out["count"], out["sum"]) == (2, 100.0)


@pytest.mark.parametrize("unit, ask", [
    ("Mtrs", "What is the total billed length of storm water pipe?"),
    ("Kgs", "What is the total billed weight of storm water pipe?"),
], ids=["length-plural", "weight-plural"])
def test_a_unit_printed_in_the_plural_measures_the_singular_dimension(tmp_path, unit, ask):
    # A bill writes its units in either number. A plural the vocabulary lists
    # only in the singular must still measure the dimension and carry the
    # figure; the counted and the figureless rows stay out of the total.
    path = (BILL, NET, "PIPE RUNS")
    rows = [
        _row(path, f"A |Storm water pipe, first {unit}", unit, "5", "1", "5"),
        _row(path, f"B |Storm water pipe, second {unit}", unit, "7", "1", "7"),
        _row(path, "C |Storm water pipe, chamber", "No", "4", "1", "4"),
        _row(path, "D |Storm water pipe, unmeasured run", "", "22", "1", "22"),
    ]
    out = agg.aggregate(_sheet(tmp_path, rows), terms="storm water pipe", ask=ask)
    assert (out["count"], out["sum"]) == (2, 12.0)
    assert {row["unit"] for row in out["rows"]} == {unit}


@pytest.mark.parametrize("place", ["rising main", "washout run"], ids=["one-place", "other-place"])
def test_the_asks_other_words_still_scope_a_dimension_total(tmp_path, place):
    # Only the noun read as a measurement leaves the word scopes. The ask's
    # other words still bind: the row that does not print them is not in this
    # total even though its own unit measures the dimension.
    path = (BILL, NET, "PIPE RUNS")
    rows = [
        _row(path, f"A |Storm water pipe, {place}", "m", "40", "1", "40"),
        _row(path, "B |Storm water pipe, second run", "m", "60", "1", "60"),
    ]
    out = agg.aggregate(_sheet(tmp_path, rows), terms="storm water pipe",
                        ask=f"What is the total billed length of storm water pipe in the {place}?")
    assert (out["count"], out["sum"]) == (1, 40.0)


@pytest.mark.parametrize("ask, unit", [
    ("How many metres of storm water pipe are billed?", "m"),
    ("What is the total tonnes of storm water pipe?", "t"),
], ids=["length-by-unit", "mass-by-unit"])
def test_an_ask_that_names_the_measurement_by_its_unit_binds_the_dimension(tmp_path, ask, unit):
    # The ask can name the dimension with a unit instead of with the noun. It is
    # the same vocabulary that reads the row's Unit cell, so a row counted in
    # units and a section's carry total are still not members of the total, and
    # the total is the two rows that measure the unit the ask named.
    path = (BILL, NET, "PIPE RUNS")
    rows = [
        _row(path, "A |Storm water pipe, first run", unit, "5", "1", "5"),
        _row(path, "B |Storm water pipe, second run", unit, "7", "1", "7"),
        _row(path, "C |Storm water pipe, chamber", "No", "4", "1", "4"),
        _row(path, "D |Storm water pipe, carried", "sum", "", "", "12"),
    ]
    out = agg.aggregate(_sheet(tmp_path, rows), terms="storm water pipe", ask=ask)
    assert (out["count"], out["sum"]) == (2, 12.0)
    assert {row["unit"] for row in out["rows"]} == {unit}


@pytest.mark.parametrize("depth, phrase", [
    ("2.5 to 3m", "2.5 to 3 m of depth"),
    ("4.0 to 4.5m", "4.0 to 4.5 m of depth"),
], ids=["upper-band", "lower-band"])
def test_a_unit_attached_to_a_number_states_a_size_not_a_total(tmp_path, depth, phrase):
    # The same unit spelling that binds a total when the ask totals a
    # measurement binds nothing when it is attached to a number: the number is
    # the size of the thing, so counting what lies in the band is still open.
    sheet = _sheet(tmp_path, _mixed_sheet(depth, "500", "300"))
    out = agg.aggregate(sheet, terms="storm water pipe",
                        ask=f"What is the total of storm water pipe at {phrase}?")
    assert out["scope"]["dimension"] is None
    assert out["count"] == 6


@pytest.mark.parametrize("ask, unit, other", [
    ("What is the total square metres of formwork?", "m2", "m"),
    ("What is the total cubic metres of concrete?", "m3", "m"),
], ids=["area-by-unit", "volume-by-unit"])
def test_the_qualifier_in_front_of_the_unit_names_the_dimension(tmp_path, ask, unit, other):
    # A metre is a length, but "square metres" is an area and "cubic metres" a
    # volume: the qualifier decides, so the rows of the other measurement are
    # not in this total even though every row prints the ask's item word.
    path = (BILL, NET, "MEASURED WORKS")
    item = ask.rsplit(" ", 1)[-1].rstrip("?")
    rows = [
        _row(path, f"A |{item.capitalize()} to soffit, first", unit, "5", "1", "5"),
        _row(path, f"B |{item.capitalize()} to soffit, second", unit, "7", "1", "7"),
        _row(path, f"C |{item.capitalize()} to soffit, run", other, "9", "1", "9"),
    ]
    out = agg.aggregate(_sheet(tmp_path, rows), terms=item, ask=ask)
    assert (out["count"], out["sum"]) == (2, 12.0)
    assert {row["unit"] for row in out["rows"]} == {unit}


@pytest.mark.parametrize("ask, terms, count", [
    ("How many storm water pipe segments in each of the bills are billed?", "", 6),
    ("How many storm water pipe segments are billed in metres?", "storm water pipe", 6),
], ids=["count-word", "unit-not-in-measuring-position"])
def test_a_word_that_could_be_a_unit_binds_no_dimension_out_of_position(tmp_path, ask, terms, count):
    # "each" and "metres" are spellings the unit table knows, but neither totals
    # a measurement here: one is a count, the other is not in a measuring
    # position. Inferring a dimension from either would drop counted rows from
    # a count ask, so the question is counted, not measured.
    sheet = _sheet(tmp_path, _mixed_sheet("2.5 to 3m", "500", "300"))
    out = agg.aggregate(sheet, terms=terms, ask=ask)
    assert (out["scope"] or {}).get("dimension") is None
    assert out["count"] == count
    assert set(out["sum_by_unit"]) == {"m", "No", "Nr", ""}


@pytest.mark.parametrize("unit", ["mtr", "Mtrs"], ids=["singular", "plural"])
def test_the_unit_an_ask_names_is_not_a_word_to_find(tmp_path, unit):
    # The unit the ask printed is the measurement, not an item word: requiring
    # "metres" in the rows would empty a total of rows that print the unit only
    # in their own Unit cell.
    path = (BILL, NET, "PIPE RUNS")
    rows = [
        _row(path, "A |Storm water pipe, first run", unit, "40", "1", "40"),
        _row(path, "B |Storm water pipe, second run", unit, "60", "1", "60"),
    ]
    out = agg.aggregate(_sheet(tmp_path, rows), terms="storm water pipe",
                        ask="How many metres of storm water pipe are billed?")
    assert (out["count"], out["sum"]) == (2, 100.0)
