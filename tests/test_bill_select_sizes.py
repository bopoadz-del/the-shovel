"""A number in a bill call's terms is a SIZE, and a bare number is itself.

Before this rule a number in ``terms`` was split into loose digit words:
``1.8`` became the words ``1`` and ``8`` and selected any row printing both
anywhere, and ``1800mm`` never selected a row printing ``1.8 m``. Now a size
with a unit is read to metres on the call and on the bill alike, and a size
with no unit keeps only its printed numbers (``1.8`` never selects 18, 180 or
1800; ``4`` never selects 400). Every bill here is invented in the test and
written through the derivation's own ``write_bill_workbook``; each decision has
at least two inputs with different outcomes.
"""

from __future__ import annotations

import json

import pytest

from app.core import bill_aggregate as agg
from app.core import bill_table_geometry as g


def _row(path, page, ref, desc, unit="", qty=""):
    fields = {"description": desc}
    for k, v in (("unit", unit), ("quantity", qty)):
        if v:
            fields[k] = v
    return g.BillRow(section=path[-1], source_page=page, fields=fields,
                     extra={"Item": ref} if ref else {}, path=tuple(path))


CHAMBERS = ("Drainage", "Chambers")
#: Box chambers at four magnitudes of the same digits, a dimension in two
#: spellings, a code glued to digits, and a row that prints the digits 1 and 8
#: apart (the loose-word false match).
BILL = [
    _row(CHAMBERS, 2, "C1", "Box chamber 1.8 m deep", "nr", "3"),
    _row(CHAMBERS, 2, "C2", "Box chamber 18 m long run", "nr", "5"),
    _row(CHAMBERS, 2, "C3", "Box chamber 180 deep", "nr", "7"),
    _row(CHAMBERS, 3, "C4", "Box chamber 1800 mm deep", "nr", "11"),
    _row(CHAMBERS, 3, "C5", "Box chamber type 1 with 8 rungs", "nr", "13"),
    _row(CHAMBERS, 3, "C6", "Box chamber 1.2 x 0.9 m on plan", "nr", "17"),
    _row(CHAMBERS, 4, "C7", "Box chamber 1200 x 900 mm on plan", "nr", "19"),
    _row(CHAMBERS, 4, "C8", "Box chamber for DN150 outlet", "nr", "23"),
    _row(CHAMBERS, 4, "C9", "Box chamber 4 m deep", "nr", "29"),
    _row(CHAMBERS, 4, "C10", "Box chamber 400 deep", "nr", "31"),
]


@pytest.fixture
def rows(tmp_path):
    path = tmp_path / "bill.xlsx"
    g.write_bill_workbook(BILL, str(path))
    return agg.read_sheet_rows(str(path))


def _refs(result):
    return [r["ref"] for r in result["rows"]]


# ---- the bare number is itself ---------------------------------------------

@pytest.mark.parametrize("terms, refs", [
    # A bare 1.8 selects the row printing 1.8 (here with a unit) and the row
    # printing 1800 mm (1.8 m) is NOT a bare 1.8: no magnitude is borrowed.
    ("chamber 1.8", ["C1"]),
    ("chamber 18", ["C2"]),
    ("chamber 180", ["C3"]),
    ("chamber 1800", ["C4"]),
    ("chamber 4", ["C9"]),
    ("chamber 400", ["C10"]),
], ids=["1.8", "18", "180", "1800", "4", "400"])
def test_a_bare_number_selects_only_rows_printing_that_number(rows, terms, refs):
    assert _refs(agg.aggregate(rows, terms=terms)) == refs


@pytest.mark.parametrize("terms, refs, total", [
    ("chamber 1.8", ["C1"], 3.0),
    ("chamber 1 8", ["C5"], 13.0),
], ids=["decimal-is-one-number", "two-numbers-are-two"])
def test_a_decimal_is_not_two_loose_digit_words(rows, terms, refs, total):
    # On the word rule both asks were the words {1, 8}: each selected C5.
    result = agg.aggregate(rows, terms=terms)
    assert _refs(result) == refs
    assert result["sum"] == total


# ---- a size with a unit is one length in any spelling ----------------------

@pytest.mark.parametrize("terms, refs", [
    ("chamber 1.8 m", ["C1", "C4"]),
    ("chamber 1800mm", ["C1", "C4"]),
    # C3 prints a bare 180 (no unit): it is matched by the number it prints.
    ("chamber 180 cm", ["C1", "C3", "C4"]),
    ("chamber 1,8 m", ["C1", "C4"]),
    ("chamber 18 m", ["C2"]),
    ("chamber 0.4 cm", []),
], ids=["m", "mm", "cm", "decimal-comma", "other-length", "no-such-length"])
def test_a_size_with_a_unit_selects_every_spelling_of_that_length(rows, terms, refs):
    assert _refs(agg.aggregate(rows, terms=terms)) == refs


@pytest.mark.parametrize("terms, refs", [
    ("chamber 1.2 x 0.9 m", ["C6", "C7"]),
    ("chamber 1200 x 900 mm", ["C6", "C7"]),
    ("chamber 1200 by 900 mm", ["C6", "C7"]),
    ("chamber 0.9 x 1.2 m", []),
    ("chamber 1200 x 900", ["C7"]),
], ids=["m", "mm", "by", "members-in-order", "bare-dimension"])
def test_a_dimension_is_one_size(rows, terms, refs):
    assert _refs(agg.aggregate(rows, terms=terms)) == refs


@pytest.mark.parametrize("terms, refs", [
    ("chamber ١٫٨ م", ["C1", "C4"]),   # Arabic-Indic digits, decimal mark, metres
    ("chamber ۱۸۰۰ مم", ["C1", "C4"]),  # Eastern digits, mm
], ids=["arabic-indic-m", "eastern-mm"])
def test_arabic_digits_and_units_read_as_the_latin_size(rows, terms, refs):
    assert _refs(agg.aggregate(rows, terms=terms)) == refs


# ---- what is not a size ----------------------------------------------------

@pytest.mark.parametrize("terms, refs", [
    ("chamber dn150", ["C8"]),
    ("chamber 150", []),
], ids=["code-is-a-word", "digits-of-a-code-are-not-a-size"])
def test_a_number_glued_to_a_letter_is_a_code(rows, terms, refs):
    assert _refs(agg.aggregate(rows, terms=terms)) == refs


@pytest.mark.parametrize("text, printed, metres", [
    ("1.8", (1.8,), None),
    ("1,800", (1800.0,), None),
    ("1,800.5", (1800.5,), None),
    ("1.800,5", (1800.5,), None),
    ("1.250.000", (1250000.0,), None),
    ("1,8 m", (1.8,), (1.8,)),
    ("1.8 x 1.8 m", (1.8, 1.8), (1.8, 1.8)),
    ("1.8 m x 900", (1.8, 900.0), (1.8, 900.0)),
    ("900 x 1.2 m", (900.0, 1.2), (900.0, 1.2)),
    ("2 km", (2.0,), (2000.0,)),
], ids=["decimal", "thousands-comma", "thousands-and-decimal", "european",
        "thousands-repeated", "decimal-comma-unit", "unit-after", "unit-before-only",
        "unit-from-later", "km"])
def test_sizes_in_reads_numbers_and_units(text, printed, metres):
    (size,) = agg.sizes_in(text)
    assert size.printed == printed
    assert size.metres == metres


@pytest.mark.parametrize("a, b, same", [
    ("1.8 m", "1800 mm", True),
    ("1.8 m", "1.8", True),      # one side bare: compared by printed number
    ("1800", "1.8 m", False),
    ("1.8", "18", False),
    ("1.8 x 1.8 m", "1.8", False),
    ("2 km", "2000 m", True),
], ids=["units", "bare-vs-unit", "bare-other-magnitude", "bare-vs-bare",
        "dimension-vs-single", "km"])
def test_size_same(a, b, same):
    (x,), (y,) = agg.sizes_in(a), agg.sizes_in(b)
    assert x.same(y) is same
    assert y.same(x) is same


# ---- where the size is printed, and what the result says --------------------

@pytest.mark.parametrize("terms, refs, total", [
    ("pipe 0.15 m", ["P1"], 40.0),
    ("pipe 225mm", ["P2"], 60.0),
    ("pipe 300 mm", [], 0.0),
], ids=["first-section", "second-section", "no-section"])
def test_a_size_in_the_heading_path_selects_that_sections_rows(tmp_path, terms, refs, total):
    bill = [
        _row(("Pipes", "150 mm diameter"), 2, "P1", "Clay pipe in trench", "m", "40"),
        _row(("Pipes", "225 mm diameter"), 2, "P2", "Clay pipe in trench", "m", "60"),
    ]
    path = tmp_path / "pipes.xlsx"
    g.write_bill_workbook(bill, str(path))
    result = agg.aggregate(agg.read_sheet_rows(str(path)), terms=terms)
    assert _refs(result) == refs
    assert result["sum"] == total


@pytest.mark.parametrize("terms, words, sizes", [
    ("chamber 1800mm", ["chamber"], ["1800mm"]),
    ("box chamber", ["box", "chamber"], []),
], ids=["with-size", "words-only"])
def test_the_result_names_the_sizes_with_the_words(rows, terms, words, sizes):
    result = agg.aggregate(rows, terms=terms)
    assert result["sizes"] == sizes
    assert result["terms"] == words + sizes


@pytest.mark.parametrize("terms, count", [("1.8 m", 2), ("400", 1)], ids=["unit", "bare"])
def test_a_size_alone_is_a_selection(rows, terms, count):
    assert agg.aggregate(rows, terms=terms)["count"] == count


@pytest.mark.parametrize("terms", ["", "  "], ids=["empty", "blank"])
def test_no_words_and_no_size_is_still_refused(rows, terms):
    with pytest.raises(ValueError):
        agg.aggregate(rows, terms=terms)


@pytest.mark.parametrize("terms, refs", [
    (["chamber", "1.8 m"], ["C1", "C4"]),
    (["chamber", "400"], ["C10"]),
], ids=["list-with-unit", "list-bare"])
def test_a_term_list_reads_sizes_too(rows, terms, refs):
    assert _refs(agg.aggregate(rows, terms=terms)) == refs


# ---- through the tool, as the agent calls it ---------------------------------

@pytest.fixture
def stored_bill(tmp_path, monkeypatch):
    from app.core import bill_derived, projects

    path = str(tmp_path / "chambers.xlsx")
    g.write_bill_workbook(BILL, path)
    docs = [{"id": "d1", "original_name": "chambers.bill.xlsx", "path": path,
             "metadata": {"derivation": bill_derived.DERIVATION, "derived_from": "s1"}}]
    monkeypatch.setattr(projects, "list_documents", lambda pid, **_kw: docs)
    monkeypatch.setattr(projects, "materialize_document_path", lambda d: d["path"])


@pytest.mark.asyncio
@pytest.mark.parametrize("terms, count, total", [
    ("box chamber 1.8", 1, 3.0),
    ("box chamber 1800 mm", 2, 14.0),
    ("box chamber 1.8 x 1.8", 0, 0.0),
], ids=["bare", "unit", "absent-dimension"])
async def test_the_tool_reads_sizes_the_same_way(stored_bill, terms, count, total):
    from app.agents.base.tools import bill_rows_statement
    from app.agents.runtime import Agent

    runtime = Agent(name="test-agent", description="t", system_prompt="t", allowed_blocks=[])
    call = {"function": {"name": "bill_rows_aggregate", "arguments": json.dumps({"terms": terms})}}
    out = await runtime._run_tool_call(call, project_id="p")
    (bill,) = out["result"]["bills"]
    assert (bill["count"], bill["sum"]) == (count, total)
    stated = bill_rows_statement(out["result"])
    assert stated[0].figure == count
    # The statement names the size it selected by, as printed in the call.
    assert agg.sizes_in(terms)[0].text in stated[0].terms
