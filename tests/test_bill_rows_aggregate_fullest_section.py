"""The rows an ask selects do not depend on which other rows share a word.

The ask's content words are graded over the whole document, so one
description anywhere that prints a heading word makes that word optional,
and a sibling section that shares only a generic word used to be admitted
beside the section the ask names. These tests build their own workbooks and
drive every decision by two distinct inputs. Nothing here names a project,
a document, a bill or a figure from a real tender.
"""

from __future__ import annotations

import pytest

from app.core import bill_aggregate as agg
from app.core import bill_table_geometry as g

BAND = "Depth 1.5 to 2m"


def _row(path, desc, unit, qty, page=1):
    fields = {"description": desc, "unit": unit, "quantity": qty, "rate": "1", "total": qty}
    return g.BillRow(section=path[-1], source_page=page, fields=fields, extra={}, path=tuple(path))


def _sheet(tmp_path, rows, name):
    path = tmp_path / name
    g.write_bill_workbook(rows, str(path))
    return agg.read_sheet_rows(str(path))


def _two_networks(named, other, stray):
    """Two bills with a band of bored rows each; the networks share one word.

    ``stray`` is the description of a counted row elsewhere in the named
    network; it may or may not print the word that tells the networks apart.
    """
    rows = []
    for page, net, sizes in ((4, other, (150, 225, 300)), (5, named, (375, 450))):
        for size in sizes:
            rows.append(_row((f"Bill No. {page} - {net}", "Ducts", BAND),
                             f"A |Nominal bore: {size}mm", "m", str(size), page=page))
    rows.append(_row((f"Bill No. 5 - {named}", "Chambers", BAND), f"Z |{stray}", "Nr", "4", page=5))
    return rows


@pytest.mark.parametrize("named, other, word, ask", [
    ("RAIN RUNOFF NETWORK", "FOUL RUNOFF NETWORK", "Rain runoff",
     "What is the total length of rain runoff duct at 1.5 to 2 m depth, all bores?"),
    ("HILL CULVERT NETWORK", "ROAD CULVERT NETWORK", "Hill culvert",
     "What is the total length of hill culvert duct at 1.5 to 2 m depth, all bores?"),
], ids=["one-pair", "other-pair"])
def test_a_stray_description_does_not_admit_a_sibling_section(tmp_path, named, other, word, ask):
    # The same sheet twice: once the chamber row prints the network's
    # distinguishing word, once it does not. The ask names that word, so only
    # the named network's bored rows are in, whichever description is printed.
    plain = _sheet(tmp_path, _two_networks(named, other, "Access chamber cover"), "plain.xlsx")
    stray = _sheet(tmp_path, _two_networks(named, other, f"{word} access chamber cover"), "stray.xlsx")
    a = agg.aggregate(plain, terms="", ask=ask)
    b = agg.aggregate(stray, terms="", ask=ask)
    assert [(r["description"], r["quantity"]) for r in a["rows"]] == \
        [(r["description"], r["quantity"]) for r in b["rows"]]
    assert (b["count"], b["sum"]) == (2, 825.0)
    assert {r["section"][0] for r in b["rows"]} == {f"Bill No. 5 - {named}"}


@pytest.mark.parametrize("ask", [
    "How many widgets and gadgets are billed?",
    "How many gadgets and widgets are billed in total?",
], ids=["one-order", "other-order"])
def test_two_sections_that_each_print_a_word_the_other_does_not_both_stay(tmp_path, ask):
    # An ask that lists two items: neither section's words contain the
    # other's, so neither is a sibling of the other and both are counted.
    rows = [
        _row(("Bill No. 2 - WIDGET WORKS", "Section A"), "A |Widget, type one", "Nr", "2"),
        _row(("Bill No. 2 - WIDGET WORKS", "Section A"), "B |Widget, type two", "Nr", "3"),
        _row(("Bill No. 3 - GADGET WORKS", "Section B"), "A |Gadget, type one", "Nr", "7"),
    ]
    out = agg.aggregate(_sheet(tmp_path, rows, "two.xlsx"), terms="", ask=ask)
    assert (out["count"], out["sum"]) == (3, 12.0)


@pytest.mark.parametrize("extra", ["concrete", "steel"], ids=["one-word", "other-word"])
def test_a_word_one_row_prints_never_splits_its_section(tmp_path, extra):
    # The ask word is printed in an unrelated heading and in one row's
    # description, so it is an optional word, and so is the network word an
    # unrelated description also prints. The named section is read as a
    # whole: the row that also prints the word does not push its sibling out.
    rows = [
        _row(("Bill No. 5 - RAIN RUNOFF NETWORK", "Ducts", BAND), "A |Nominal bore: 375mm", "m", "375", page=5),
        _row(("Bill No. 5 - RAIN RUNOFF NETWORK", "Ducts", BAND), f"B |Nominal bore: 450mm, {extra}", "m", "450", page=5),
        _row(("Bill No. 4 - FOUL RUNOFF NETWORK", "Ducts", BAND), "A |Nominal bore: 150mm", "m", "150", page=4),
        _row((f"Bill No. 6 - {extra.upper()} WORKS", "Slabs"), "A |Slab over runoff outfall", "m3", "40", page=6),
    ]
    ask = f"What is the total length of {extra} rain runoff duct at 1.5 to 2 m depth, all bores?"
    out = agg.aggregate(_sheet(tmp_path, rows, "split.xlsx"), terms="", ask=ask)
    assert out["scope"]["either"] == sorted([extra, "runoff"])
    assert (out["count"], out["sum"]) == (2, 825.0)


@pytest.mark.parametrize("ask, count", [
    ("What is the total length of rain runoff duct at 1.5 to 2 m depth?", 2),
    ("What is the total length of foul runoff duct at 1.5 to 2 m depth?", 3),
], ids=["one-network", "other-network"])
def test_the_section_the_ask_names_is_the_one_kept(tmp_path, ask, count):
    rows = _two_networks("RAIN RUNOFF NETWORK", "FOUL RUNOFF NETWORK", "Rain runoff access chamber cover")
    out = agg.aggregate(_sheet(tmp_path, rows, "named.xlsx"), terms="", ask=ask)
    assert out["count"] == count
