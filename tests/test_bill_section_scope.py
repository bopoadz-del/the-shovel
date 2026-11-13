"""S-A7: the bill-rows action reads a bill's sections from the bill itself.

Every test builds its own synthetic bill. The bill prints the same item words
(a pipe, steel bars) in two or more of its sections at different rates, so a
document-wide lexical ranking cannot tell which section's row answers an ask.
An ask that names a section must get only that section's rows; an ask that
names none must see which section each row belongs to; a bill that prints no
heading must rank exactly as before.
"""
from __future__ import annotations

import re
from functools import partial
from types import SimpleNamespace

import pytest

from app.core.rag.bill_sections import (
    BillSection,
    in_named_span,
    named_sections,
    section_label,
    section_map,
    words,
)
from app.core.rag.evidence_actions import (
    _bill_rows_retrieve,
    bill_rows_action,
    bill_section_maps,
)
from app.core.rag.evidence_loop import build_actions
from app.core.rag.retriever import priced_row_count

BILL = "bill-doc"
SPEC = "spec-doc"

#: Two spellings of a section heading; the map must read both the same way.
HEADING_STYLES = ("BILL NO. {n}", "Bill Number {n}")


def _row(desc, unit, qty, rate):
    return f"{desc}\n{unit}\n{qty:,}\n{rate:,.2f}\n{qty * rate:,.2f}"


def _chunk(doc_id, index, text):
    return SimpleNamespace(
        doc_id=doc_id, chunk_id=f"{doc_id}:{index}", chunk_index=index,
        text=text, score=None,
    )


def _sectioned_bill(*, headings=True, style=HEADING_STYLES[0]):
    """Four bills plus a general summary page; pipe and steel recur."""
    def b(n):
        return style.format(n=n)

    def h(text):
        return text if headings else ""

    texts = [
        h(f"GENERAL SUMMARY\n{b(1)} Preliminaries\n{b(2)} Drainage\n"
          f"{b(3)} Pumping Station\n{b(4)} Roads\n"),
        h(f"{b(1)}\nPRELIMINARIES\n")
        + _row("Site office establishment and upkeep", "sum", 1, 10000),
        h(f"{b(1)} - Carried to General Summary\n") + "Page total 10,000.00",
        h(f"{b(2)}\nDRAINAGE\n")
        + _row("Ductile iron pipe 300mm diameter laid in trench", "m", 120, 85),
        h(f"{b(2)} DRAINAGE (continued)\n")
        + _row("Reinforcement steel bars to manholes", "t", 2, 3100),
        h(f"{b(2)} - Carried to General Summary\n{b(3)}\nPUMPING STATION\n")
        + "Preamble: rates include testing.",
        _row("Ductile iron pipe 300mm diameter laid in trench", "m", 40, 92)
        + "\n" + _row("Reinforcement steel bars to wet well", "t", 18, 3050),
        h(f"{b(4)}\nROADS\n")
        + _row("Reinforcement steel bars to culvert", "t", 6, 2990),
    ]
    return [_chunk(BILL, i, t) for i, t in enumerate(texts)]


def _spec():
    return [
        _chunk(SPEC, 0, "Ductile iron pipe shall comply with BS EN 545. Pipe "
               "joints shall be tested."),
        _chunk(SPEC, 1, "Reinforcement steel bars shall be grade B500B."),
        _chunk(SPEC, 2, "The Time for Completion is stated in the Contract Data."),
    ]


def _bm25_over(chunks, calls=None):
    def bm25(project_id, query, k=50, doc_ids=None):
        if calls is not None:
            calls.append((query, None if doc_ids is None else tuple(doc_ids)))
        q = {w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 2}
        scored = []
        for c in chunks:
            if doc_ids is not None and c.doc_id not in doc_ids:
                continue
            hits = len(q & set(re.findall(r"[a-z0-9]+", c.text.lower())))
            if hits:
                scored.append((hits, c))
        scored.sort(key=lambda p: -p[0])
        return [c for _, c in scored[:k]]
    return bm25


def _fetch_over(chunks):
    def fetch(project_id, doc_ids):
        return [c for c in chunks if c.doc_id in doc_ids]
    return fetch


def _action(bill, *, ask, sections=True, calls=None):
    corpus = bill + _spec()
    return bill_rows_action(
        _bm25_over(corpus, calls), "p1", row_count=priced_row_count,
        typed_bills=lambda _pid: [BILL],
        bill_chunks=_fetch_over(corpus) if sections else None,
        ask=ask,
    )


def _map(bill):
    return section_map((c.chunk_index, c.text) for c in bill)


@pytest.mark.parametrize(("index", "rows"), [(3, 1), (6, 2)])
def test_fixture_rows_are_priced_rows(index, rows):
    assert priced_row_count(_sectioned_bill()[index].text) == rows


@pytest.mark.parametrize("style", HEADING_STYLES)
def test_section_map_reads_spans_from_headings_and_carried_rows(style):
    spans = {s.number: (s.title, s.start, s.end) for s in _map(_sectioned_bill(style=style))}
    # The general summary page (chunk 0) lists every bill: not a start.
    assert spans == {
        "1": ("PRELIMINARIES", 1, 2),
        "2": ("DRAINAGE", 3, 5),         # running header at 4 does not split
        "3": ("PUMPING STATION", 5, 6),  # opened in the chunk that closes 2
        "4": ("ROADS", 7, 7),            # runs to the document's last chunk
    }


@pytest.mark.parametrize("chunks", [
    [],
    [(0, "Specification clause 3.1: pipes shall be tested.")],
    [(c.chunk_index, c.text) for c in _sectioned_bill(headings=False)],
])
def test_document_without_headings_has_no_map(chunks):
    assert section_map(chunks) == []


@pytest.mark.parametrize(("text", "expected"), [
    ("the bill of pipes", {"pipe"}),
    ("Reinforcement steel to the wet well", {"reinforcement", "steel", "wet", "well"}),
])
def test_words_drop_stopwords_and_fold_plurals(text, expected):
    assert words(text) == expected


@pytest.mark.parametrize(("text", "expected"), [
    ("pipe rate in Bill No. 2", {"2"}),
    ("pipe rate in bill 3", {"3"}),
    ("pipe rate in the pumping station bill", {"3"}),
    ("pipe rate for the roads", {"4"}),
    ("pipe rate in Bill No. 9", set()),   # a number the map does not hold
    ("ductile iron pipe rate", set()),
    (None, set()),
])
def test_named_sections_by_number_and_by_title(text, expected):
    assert named_sections(text, _map(_sectioned_bill())) == expected


@pytest.mark.parametrize(("title", "ask", "expected"), [
    ("EARTHWORKS", "excavation rate in the earthworks", {"7"}),
    ("SITE CLEARANCE", "rate for site clearance", {"7"}),
])
def test_a_one_section_bill_is_named_by_its_title(title, ask, expected):
    sections = [BillSection("7", title, 0, 3)]
    assert named_sections(ask, sections) == expected


@pytest.mark.parametrize(("index", "label"), [
    (3, "[Bill No. 2 - DRAINAGE]"),
    (5, "[Bill No. 2 - DRAINAGE | Bill No. 3 - PUMPING STATION]"),
    (0, ""),       # the summary page belongs to no section
    (None, ""),    # a chunk without an index
])
def test_section_label_names_every_owning_section(index, label):
    assert section_label(index, _map(_sectioned_bill())) == label


@pytest.mark.parametrize(("index", "numbers", "inside"), [
    (4, {"2"}, True),
    (6, {"2"}, False),
    (6, {"2", "3"}, True),
    (None, {"2"}, False),
])
def test_in_named_span(index, numbers, inside):
    assert in_named_span(index, _map(_sectioned_bill()), numbers) is inside


@pytest.mark.parametrize(("ask", "kept", "excluded", "label"), [
    ("Rate for ductile iron pipe in Bill No. 2?", "85.00", "92.00",
     "[Bill No. 2 - DRAINAGE]"),
    ("What is the ductile iron pipe rate in the pumping station bill?", "92.00",
     "85.00", "[Bill No. 3 - PUMPING STATION]"),
])
def test_ask_naming_a_section_gets_only_that_sections_rows(ask, kept, excluded, label):
    text = _action(_sectioned_bill(), ask=ask)("ductile iron pipe rate")
    assert kept in text
    assert excluded not in text
    assert label in text


@pytest.mark.parametrize(("item", "kept"), [
    ("reinforcement steel bars bill no. 2", "3,100.00"),
    ("reinforcement steel bars roads", "2,990.00"),
])
def test_plan_item_naming_a_section_wins_over_the_ask(item, kept):
    text = _action(_sectioned_bill(), ask="Compare steel rates across the bills")(item)
    assert kept in text
    others = {"3,100.00", "3,050.00", "2,990.00"} - {kept}
    assert not any(other in text for other in others)


@pytest.mark.parametrize("ask", [
    "What are the steel reinforcement rates?",
    "List every reinforcement rate in the bill of quantities",
])
def test_ask_without_section_tags_every_row_with_its_section(ask):
    text = _action(_sectioned_bill(), ask=ask)("reinforcement steel bars")
    for amount, label in (
        ("3,100.00", "[Bill No. 2 - DRAINAGE]"),
        ("3,050.00", "[Bill No. 3 - PUMPING STATION]"),
        ("2,990.00", "[Bill No. 4 - ROADS]"),
    ):
        assert amount in text
        assert label in text
        # The label leads the chunk that prints the row.
        assert text.index(label) < text.index(amount)


@pytest.mark.parametrize("ask", [
    "Rate for ductile iron pipe in Bill No. 2?",
    "ductile iron pipe rate in the pumping station bill",
])
def test_bill_without_headings_ranks_as_before(ask):
    bill = _sectioned_bill(headings=False)
    with_sections = _action(bill, ask=ask)("ductile iron pipe rate")
    without = _action(bill, ask=ask, sections=False)("ductile iron pipe rate")
    assert with_sections == without
    assert "85.00" in without and "92.00" in without
    assert "[Bill No." not in without


@pytest.mark.parametrize("item", ["ductile iron pipe rate", "reinforcement steel bars"])
def test_failed_section_fetch_ranks_as_before(item):
    corpus = _sectioned_bill() + _spec()

    def broken(_pid, _ids):
        raise RuntimeError("store down")

    assert bill_section_maps(broken, "p1", [BILL]) == {}
    action = bill_rows_action(
        _bm25_over(corpus), "p1", row_count=priced_row_count,
        typed_bills=lambda _pid: [BILL], bill_chunks=broken,
        ask="Rate in Bill No. 2?",
    )
    plain = _action(_sectioned_bill(), ask="Rate in Bill No. 2?", sections=False)
    assert action(item) == plain(item)


@pytest.mark.parametrize(("doc_ids", "mapped"), [
    ([BILL], {BILL}),
    ([SPEC], set()),          # no heading: no map
    ([], set()),              # nothing asked for
])
def test_bill_section_maps_keeps_documents_that_print_headings(doc_ids, mapped):
    corpus = _sectioned_bill() + _spec()
    maps = bill_section_maps(_fetch_over(corpus), "p1", doc_ids)
    assert set(maps) == mapped
    for sections, chunks in maps.values():
        assert [s.number for s in sections] == ["1", "2", "3", "4"]
        assert len(chunks) == 8


@pytest.mark.parametrize("item", [
    "time for completion contract data",
    "joints shall be tested to BS EN 545",
])
def test_item_with_no_priced_row_is_not_a_bill_item(item):
    assert _action(_sectioned_bill(), ask="When is the Time for Completion?")(item) == ""


@pytest.mark.parametrize("item", ["ductile iron pipe rate", "reinforcement steel bars"])
def test_each_item_is_fetched_once_per_ask(item):
    calls: list = []
    action = _action(_sectioned_bill(), ask="rates in Bill No. 2", calls=calls)
    first = action(item)
    second = action(item)
    assert first == second
    assert len(calls) == 2  # one global + one bill-scoped query, not four


@pytest.mark.parametrize(("ask", "kept", "excluded"), [
    ("Rate for ductile iron pipe in Bill No. 2?", "85.00", "92.00"),
    ("ductile iron pipe rate in the pumping station bill", "92.00", "85.00"),
])
def test_retrieve_scopes_to_the_named_section(ask, kept, excluded):
    corpus = _sectioned_bill() + _spec()
    text = _bill_rows_retrieve(
        _bm25_over(corpus), "p1", 200, priced_row_count, "ductile iron pipe rate",
        typed_bills=lambda _pid: [BILL],
        section_maps=partial(bill_section_maps, _fetch_over(corpus), "p1"),
        ask=ask,
    )
    assert kept in text
    assert excluded not in text


@pytest.mark.parametrize(("question", "kept", "excluded"), [
    ("Rate of ductile iron pipe in the pumping station bill", "92.00", "85.00"),
    ("Rate of ductile iron pipe in Bill No. 2", "85.00", "92.00"),
])
def test_build_actions_carries_the_question_to_the_bill_action(question, kept, excluded):
    corpus = _sectioned_bill() + _spec()
    actions = build_actions(
        retrieve_fn=lambda q, pid, k=5: ([], 0),
        bm25_fn=_bm25_over(corpus),
        project_id="p1",
        typed_bills=lambda _pid: [BILL],
        bill_chunks=_fetch_over(corpus),
        question=question,
    )
    bill_text = actions[2]("ductile iron pipe rate")
    assert kept in bill_text and excluded not in bill_text


@pytest.mark.parametrize("doc_ids", [[], [None, ""]])
def test_bill_section_maps_does_not_read_the_store_for_no_document(doc_ids):
    calls: list = []

    def fetch(project_id, ids):
        calls.append(ids)
        return []

    assert bill_section_maps(fetch, "p1", doc_ids) == {}
    assert calls == []


@pytest.mark.parametrize(("ask", "kept", "excluded"), [
    ("Rate for ductile iron pipe in Bill No. 2?", "85.00", "92.00"),
    ("ductile iron pipe rate in the pumping station bill", "92.00", "85.00"),
])
def test_named_section_keeps_bills_that_print_no_heading(ask, kept, excluded):
    """A second priced document with no section map is kept as it was:
    only a mapped bill is cut to the named section."""
    other = [_chunk("rate-sheet", 0, _row(
        "Ductile iron pipe 300mm diameter laid in trench", "m", 10, 77))]
    corpus = _sectioned_bill() + other + _spec()
    action = bill_rows_action(
        _bm25_over(corpus), "p1", row_count=priced_row_count,
        typed_bills=lambda _pid: [BILL], bill_chunks=_fetch_over(corpus), ask=ask,
    )
    text = action("ductile iron pipe rate")
    assert kept in text and "77.00" in text
    assert excluded not in text
