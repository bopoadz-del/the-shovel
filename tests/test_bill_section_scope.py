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
from types import SimpleNamespace

from app.core.rag.bill_sections import named_sections, section_map
from app.core.rag.evidence_actions import bill_rows_action
from app.core.rag.evidence_loop import build_actions
from app.core.rag.retriever import priced_row_count

BILL = "bill-doc"
SPEC = "spec-doc"


def _row(desc, unit, qty, rate):
    return f"{desc}\n{unit}\n{qty:,}\n{rate:,.2f}\n{qty * rate:,.2f}"


def _chunk(doc_id, index, text):
    return SimpleNamespace(
        doc_id=doc_id, chunk_id=f"{doc_id}:{index}", chunk_index=index,
        text=text, score=None,
    )


def _sectioned_bill(*, headings=True):
    """Four bills plus a general summary page; pipe and steel recur."""
    def h(text):
        return text if headings else ""

    texts = [
        h("GENERAL SUMMARY\nBill No. 1 Preliminaries\nBill No. 2 Drainage\n"
          "Bill No. 3 Pumping Station\nBill No. 4 Roads\n"),
        h("BILL NO. 1\nPRELIMINARIES\n")
        + _row("Site office establishment and upkeep", "sum", 1, 10000),
        h("Bill No. 1 - Carried to General Summary\n") + "Page total 10,000.00",
        h("BILL NO. 2\nDRAINAGE\n")
        + _row("Ductile iron pipe 300mm diameter laid in trench", "m", 120, 85),
        h("BILL NO. 2 DRAINAGE (continued)\n")
        + _row("Reinforcement steel bars to manholes", "t", 2, 3100),
        h("Bill No. 2 - Carried to General Summary\nBILL NO. 3\nPUMPING STATION\n")
        + "Preamble: rates include testing.",
        _row("Ductile iron pipe 300mm diameter laid in trench", "m", 40, 92)
        + "\n" + _row("Reinforcement steel bars to wet well", "t", 18, 3050),
        h("BILL NO. 4\nROADS\n")
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
        words = {w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 2}
        scored = []
        for c in chunks:
            if doc_ids is not None and c.doc_id not in doc_ids:
                continue
            text_words = set(re.findall(r"[a-z0-9]+", c.text.lower()))
            hits = len(words & text_words)
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


def test_fixture_rows_are_priced_rows():
    bill = _sectioned_bill()
    assert priced_row_count(bill[3].text) == 1
    assert priced_row_count(bill[6].text) == 2


def test_section_map_reads_spans_from_headings_and_carried_rows():
    sections = section_map((c.chunk_index, c.text) for c in _sectioned_bill())
    spans = {s.number: (s.title, s.start, s.end) for s in sections}
    # The general summary page (chunk 0) lists every bill: not a start.
    assert spans == {
        "1": ("PRELIMINARIES", 1, 2),
        "2": ("DRAINAGE", 3, 5),        # running header at 4 does not split
        "3": ("PUMPING STATION", 5, 6),  # opened in the chunk that closes 2
        "4": ("ROADS", 7, 7),           # runs to the document's last chunk
    }


def test_no_heading_document_has_no_map():
    assert section_map((c.chunk_index, c.text) for c in _sectioned_bill(headings=False)) == []


def test_named_sections_by_number_and_by_title():
    sections = section_map((c.chunk_index, c.text) for c in _sectioned_bill())
    assert named_sections("pipe rate in Bill No. 2", sections) == {"2"}
    assert named_sections("pipe rate in bill 3", sections) == {"3"}
    assert named_sections("pipe rate in the pumping station bill", sections) == {"3"}
    assert named_sections("pipe rate for the roads", sections) == {"4"}
    # A number the map does not hold, and no title, names nothing.
    assert named_sections("pipe rate in Bill No. 9", sections) == set()
    assert named_sections("ductile iron pipe rate", sections) == set()


def test_ask_naming_section_two_gets_only_section_two_rows():
    action = _action(_sectioned_bill(), ask="Rate for ductile iron pipe in Bill No. 2?")
    text = action("ductile iron pipe rate")
    assert "85.00" in text
    assert "92.00" not in text           # the same pipe in bill 3
    assert "[Bill No. 2 - DRAINAGE]" in text


def test_ask_naming_section_three_by_title_gets_only_its_rows():
    action = _action(
        _sectioned_bill(), ask="What is the ductile iron pipe rate in the pumping station bill?",
    )
    text = action("ductile iron pipe rate")
    assert "92.00" in text
    assert "85.00" not in text
    assert "[Bill No. 3 - PUMPING STATION]" in text


def test_plan_item_naming_a_section_wins_over_the_ask():
    action = _action(_sectioned_bill(), ask="Compare steel rates across the bills")
    two = action("reinforcement steel bars bill no. 2")
    four = action("reinforcement steel bars roads")
    assert "3,100.00" in two and "2,990.00" not in two and "3,050.00" not in two
    assert "2,990.00" in four and "3,100.00" not in four and "3,050.00" not in four


def test_ask_without_section_tags_every_row_with_its_section():
    action = _action(_sectioned_bill(), ask="What are the steel reinforcement rates?")
    text = action("reinforcement steel bars")
    for amount, label in (
        ("3,100.00", "[Bill No. 2 - DRAINAGE]"),
        ("3,050.00", "[Bill No. 3 - PUMPING STATION]"),
        ("2,990.00", "[Bill No. 4 - ROADS]"),
    ):
        assert amount in text
        assert label in text
        # The label leads the chunk that prints the row.
        assert text.index(label) < text.index(amount)


def test_bill_without_headings_ranks_as_before():
    bill = _sectioned_bill(headings=False)
    ask = "Rate for ductile iron pipe in Bill No. 2?"
    with_sections = _action(bill, ask=ask)("ductile iron pipe rate")
    without = _action(bill, ask=ask, sections=False)("ductile iron pipe rate")
    assert with_sections == without
    assert "85.00" in without and "92.00" in without
    assert "[Bill No." not in without


def test_failed_section_fetch_ranks_as_before():
    bill = _sectioned_bill()
    corpus = bill + _spec()

    def broken(_pid, _ids):
        raise RuntimeError("store down")

    action = bill_rows_action(
        _bm25_over(corpus), "p1", row_count=priced_row_count,
        typed_bills=lambda _pid: [BILL], bill_chunks=broken,
        ask="Rate for ductile iron pipe in Bill No. 2?",
    )
    text = action("ductile iron pipe rate")
    assert "85.00" in text and "92.00" in text
    assert "[Bill No." not in text


def test_item_with_no_priced_row_is_not_a_bill_item():
    action = _action(_sectioned_bill(), ask="When is the Time for Completion?")
    assert action("time for completion contract data") == ""


def test_each_item_is_fetched_once_per_ask():
    calls: list = []
    action = _action(_sectioned_bill(), ask="pipe rate in Bill No. 2", calls=calls)
    first = action("ductile iron pipe rate")
    second = action("ductile iron pipe rate")
    assert first == second
    assert len(calls) == 2  # one global + one bill-scoped query, not four


def test_build_actions_carries_the_question_to_the_bill_action():
    corpus = _sectioned_bill() + _spec()
    actions = build_actions(
        retrieve_fn=lambda q, pid, k=5: ([], 0),
        bm25_fn=_bm25_over(corpus),
        project_id="p1",
        typed_bills=lambda _pid: [BILL],
        bill_chunks=_fetch_over(corpus),
        question="Rate of ductile iron pipe in the pumping station bill",
    )
    bill_text = actions[2]("ductile iron pipe rate")
    assert "92.00" in bill_text and "85.00" not in bill_text
