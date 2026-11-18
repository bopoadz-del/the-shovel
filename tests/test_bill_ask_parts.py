"""#411 after #428: an ask of two parts ('kerbs and ducts') selects no row
with all its words together; each call then reads the part of the USER's
words it names. The call never adds a word, and never narrows an ask whose
words together select rows. Every bill here is invented in the test.
"""

from __future__ import annotations

import pytest

from app.core import bill_aggregate as agg


def _row(ref, desc, qty, path=("Bill 1",)):
    return agg.SheetRow(path=path, ref=ref, description=desc, unit="m", quantity=qty,
                        rate="", amount="", pages=[1])


BILL = [_row("A1", "Precast kerb type B", "10"), _row("A2", "Precast kerb type C", "5"),
        _row("B1", "Cable duct 100", "7"), _row("C1", "Kerb with duct crossing", "2"),
        _row("D1", "Timber rail", "4")]


def _refs(ask, terms):
    out = agg._aggregate_bill(BILL, terms=terms, section="", measure="quantity", ask=ask)
    return sorted(r["ref"] for r in out["rows"]), out["selected_by"]


@pytest.mark.parametrize("ask, terms, refs, by", [
    # all the ask's words select row C1: the call cannot narrow it to 'kerb'
    ("how many kerb duct crossing items?", "kerb", ["C1"], "ask"),
    # two parts that never meet: each call reads its own part
    ("how many kerb and rail items?", "kerb", ["A1", "A2", "C1"], "ask_part"),
    ("how many kerb and rail items?", "rail", ["D1"], "ask_part"),
    # a call naming none of the ask's words: the ask decides (nothing)
    ("how many kerb and rail items?", "duct", [], "ask"),
    # a call naming every word: nothing narrower
    ("how many kerb and rail items?", "kerb rail", [], "ask"),
], ids=["whole-ask-selects", "part-kerb", "part-rail", "call-word-not-asked", "call-names-all"])
def test_a_call_reads_only_a_part_of_the_users_words(ask, terms, refs, by):
    assert _refs(ask, terms) == (refs, by)


@pytest.mark.parametrize("picked, terms, expected", [
    ({"terms": ["kerb", "rail"], "section_terms": [], "sizes": []}, "kerbs",
     {"terms": ["kerb"], "section_terms": [], "sizes": []}),
    ({"terms": ["pit"], "section_terms": [], "sizes": ["1.8 x 1.8 m"]}, "1800 x 1800 mm",
     {"terms": [], "section_terms": [], "sizes": ["1.8 x 1.8 m"]}),
    # a bare size is its plain numbers: 1800 x 1800 is not 1.8 x 1.8 m
    ({"terms": ["pit"], "section_terms": [], "sizes": ["1.8 x 1.8 m"]}, "1800 x 1800", None),
    ({"terms": ["kerb"], "section_terms": [], "sizes": []}, "kerb", None),
    ({"terms": ["kerb", "rail"], "section_terms": [], "sizes": []}, "", None),
], ids=["plural", "size-with-a-unit", "bare-size-not-a-magnitude", "all-named", "no-call-words"])
def test_part_of_ask(picked, terms, expected):
    assert agg._part_of_ask(picked, terms, None) == expected
