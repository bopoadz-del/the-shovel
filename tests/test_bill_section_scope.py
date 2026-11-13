"""S-A7: the bill-rows action reads a bill's sections from the bill itself.

Every test builds its own synthetic bill. The bill prints the same item words
(a pipe, steel bars) in two or more of its sections at different rates, so a
document-wide lexical ranking cannot tell which section's row answers an ask.
An ask that names a section must get only that section's rows; an ask that
names none must get EVERY row of the item across the whole bill, each
labelled with its section, so the model can sum them (owner ruling 03:48Z);
a bill that prints no heading gives its rows labelled "section unknown".
"""
from __future__ import annotations

import re
from functools import partial
from types import SimpleNamespace

import pytest

from app.core.rag.bill_sections import (
    UNKNOWN_SECTION,
    BillSection,
    in_named_span,
    item_rows,
    named_sections,
    section_label,
    section_map,
    words,
)
from app.core.rag.evidence_actions import (
    ITEM_ROWS_NOTE,
    _bill_rows_retrieve,
    bill_rows_action,
    bill_section_maps,
)
from app.core.rag.evidence_loop import build_actions
from app.core.rag.retriever import priced_row_count, priced_row_segments

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


def _action(bill, *, ask, sections=True, calls=None, rows=True):
    corpus = bill + _spec()
    return bill_rows_action(
        _bm25_over(corpus, calls), "p1", row_count=priced_row_count,
        typed_bills=lambda _pid: [BILL],
        bill_chunks=_fetch_over(corpus) if sections else None,
        ask=ask,
        row_segments=priced_row_segments if rows else None,
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
def test_bill_without_headings_ranks_its_chunks_as_before(ask):
    bill = _sectioned_bill(headings=False)
    with_sections = _action(bill, ask=ask, rows=False)("ductile iron pipe rate")
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


@pytest.mark.parametrize(("doc_ids", "numbers"), [
    ([BILL], {BILL: ["1", "2", "3", "4"]}),
    ([SPEC], {SPEC: []}),     # no heading: read, with an empty map
    ([], {}),                 # nothing asked for
])
def test_bill_section_maps_reads_every_fetched_bill(doc_ids, numbers):
    corpus = _sectioned_bill() + _spec()
    maps = bill_section_maps(_fetch_over(corpus), "p1", doc_ids)
    assert {d: [s.number for s in m[0]] for d, m in maps.items()} == numbers
    assert {d: len(m[1]) for d, m in maps.items()} == {
        d: len([c for c in corpus if c.doc_id == d]) for d in numbers
    }


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


# --- Owner ruling 03:48Z: an item recurs across sections with no billed sum.
# With no section named the action hands EVERY row of the item, labelled;
# with a section named, only that section's rows.

RECUR = "recur-doc"


def _recurring_bill(*, item, other, headings=True, style=HEADING_STYLES[0]):
    """Three sections each billing ``item`` once (different qty / rate), no
    summary and no total for the item; ``other`` is billed in section 2 only.
    Chunk 1 prints section 1's last row, then section 2's heading and row."""
    def h(n, title):
        return f"{style.format(n=n)}\n{title}\n" if headings else ""

    texts = [
        h(1, "SITEWORKS") + _row(f"A1 {item} in trench", "m", 300, 45),
        _row("A2 Excavation in soft material", "m3", 900, 12) + "\n"
        + h(2, "BUILDINGS") + _row(f"B1 {item} in ducts", "m", 75, 52),
        _row(f"B2 {other}", "m", 40, 180),
        h(3, "EXTERNAL WORKS") + _row(f"C1 {item} under roads", "m", 120, 61),
    ]
    return [_chunk(RECUR, i, t) for i, t in enumerate(texts)]


def _recur_action(bill, ask):
    return bill_rows_action(
        _bm25_over(bill), "p1", row_count=priced_row_count,
        typed_bills=lambda _pid: [RECUR], bill_chunks=_fetch_over(bill),
        ask=ask, row_segments=priced_row_segments,
    )


def _row_lines(text):
    """The labelled row lines of the action's complete row list."""
    head, _, rest = text.partition("listed here.]\n")
    assert head and rest, "no item row list in the evidence"
    return [
        ln for ln in rest.splitlines()
        if ln.startswith("[") and "] " in ln and ln[-1].isdigit()
    ]


def _amount(line):
    return float(line.rsplit(" ", 1)[1].replace(",", ""))


def _label(line):
    return line.split("] ", 1)[0] + "]"


ITEMS = (
    ("uPVC potable water pipe 110mm", "Galvanised steel handrail"),
    ("HDPE irrigation main 63mm", "Precast concrete kerb"),
)


@pytest.mark.parametrize(("item", "other"), ITEMS)
def test_total_ask_gets_every_section_row_and_their_sum(item, other):
    bill = _recurring_bill(item=item, other=other)
    text = _recur_action(bill, f"What is the total length of {item}?")(
        f"total quantity of {item}")
    lines = _row_lines(text)
    assert [_label(ln) for ln in lines] == [
        "[Bill No. 1 - SITEWORKS]", "[Bill No. 2 - BUILDINGS]",
        "[Bill No. 3 - EXTERNAL WORKS]",
    ]
    assert sum(_amount(ln) for ln in lines) == 300 * 45 + 75 * 52 + 120 * 61
    assert ITEM_ROWS_NOTE.split("{item}")[0] in text
    assert "(3 rows, not sampled)" in text
    assert "overall contract or BOQ total" in text


@pytest.mark.parametrize(("item", "other"), ITEMS)
def test_section_ask_gets_only_that_sections_row(item, other):
    bill = _recurring_bill(item=item, other=other)
    text = _recur_action(bill, f"How much {item} is in Bill No. 2?")(item)
    lines = _row_lines(text)
    assert len(lines) == 1
    assert _label(lines[0]) == "[Bill No. 2 - BUILDINGS]"
    assert _amount(lines[0]) == 75 * 52


@pytest.mark.parametrize(("item", "other"), ITEMS)
def test_item_in_one_section_gets_one_row(item, other):
    bill = _recurring_bill(item=item, other=other)
    lines = _row_lines(_recur_action(bill, f"total {other}?")(other))
    assert len(lines) == 1
    assert _label(lines[0]) == "[Bill No. 2 - BUILDINGS]"
    assert _amount(lines[0]) == 40 * 180


@pytest.mark.parametrize(("item", "other"), ITEMS)
def test_bill_without_headings_labels_every_row_unknown(item, other):
    bill = _recurring_bill(item=item, other=other, headings=False)
    lines = _row_lines(_recur_action(bill, f"total {item}")(item))
    assert [_label(ln) for ln in lines] == [UNKNOWN_SECTION] * 3
    assert sorted(_amount(ln) for ln in lines) == sorted(
        [300 * 45, 75 * 52, 120 * 61])


@pytest.mark.parametrize("style", HEADING_STYLES)
def test_row_above_a_new_heading_stays_in_the_old_section(style):
    bill = _recurring_bill(item="potable water pipe", other="kerb", style=style)
    sections = section_map((c.chunk_index, c.text) for c in bill)
    rows = item_rows(
        [(c.chunk_index, c.text) for c in bill], sections, "excavation soft",
        row_segments=priced_row_segments,
    )
    assert len(rows) == 1
    assert _label(rows[0]) == "[Bill No. 1 - SITEWORKS]"


@pytest.mark.parametrize(("item", "expected"), [
    ("potable water pipework", 3),              # "pipework" covers "pipe"
    ("total amount of potable water pipe", 3),  # column words never match
])
def test_item_words_fold_prefixes_and_ignore_column_words(item, expected):
    bill = _recurring_bill(item="uPVC potable water pipe", other="kerb")
    rows = item_rows(
        [(c.chunk_index, c.text) for c in bill],
        section_map((c.chunk_index, c.text) for c in bill), item,
        row_segments=priced_row_segments,
    )
    assert len(rows) == expected


@pytest.mark.parametrize("item", ["bar", "zinc chromate primer"])
def test_item_the_bill_does_not_print_gets_no_row(item):
    bill = _recurring_bill(item="barrier cable", other="kerb")
    rows = item_rows(
        [(c.chunk_index, c.text) for c in bill],
        section_map((c.chunk_index, c.text) for c in bill), item,
        row_segments=priced_row_segments,
    )
    assert rows == []


@pytest.mark.parametrize("copies", [2, 3])
def test_a_row_printed_by_overlapping_chunks_is_listed_once(copies):
    bill = _recurring_bill(item="potable water pipe", other="kerb")
    last = bill[-1]
    bill += [
        _chunk(RECUR, last.chunk_index + i, last.text.split("\n", 2)[2])
        for i in range(1, copies)
    ]
    lines = _row_lines(_recur_action(bill, "total potable water pipe")(
        "potable water pipe"))
    assert len(lines) == 3


@pytest.mark.parametrize("rows", [60, 150])
def test_item_rows_are_not_cut_by_the_chunk_budget(rows, monkeypatch):
    monkeypatch.setenv("AGENTIC_EVIDENCE_MAX_TOKENS", "200")
    monkeypatch.setenv("AGENTIC_EVIDENCE_MAX_TOKENS_PER_DOC", "200")
    bill = [_chunk(RECUR, 0, "BILL NO. 1\nPIPEWORK\n")] + [
        _chunk(RECUR, i, _row(f"P{i} potable water pipe run", "m", i, 10))
        for i in range(1, rows + 1)
    ]
    text = _recur_action(bill, "total potable water pipe")("potable water pipe")
    lines = _row_lines(text)
    assert len(lines) == rows
    assert sum(_amount(ln) for ln in lines) == 10 * rows * (rows + 1) // 2
    assert all(_label(ln) == "[Bill No. 1 - PIPEWORK]" for ln in lines)


@pytest.mark.parametrize(("ask", "label"), [
    ("ductile iron pipe in the drainage bill", "[Bill No. 2 - DRAINAGE]"),
    ("ductile iron pipe in the pumping station bill",
     "[Bill No. 3 - PUMPING STATION]"),
])
def test_a_bill_whose_section_is_not_named_is_kept_whole(ask, label):
    """Two typed bills; the ask names a section by a title only the first
    prints. The
    second bill is not restricted, so all its rows still reach the model."""
    other = _recurring_bill(item="ductile iron pipe", other="kerb")
    corpus = _sectioned_bill() + other
    action = bill_rows_action(
        _bm25_over(corpus), "p1", row_count=priced_row_count,
        typed_bills=lambda _pid: [BILL, RECUR], bill_chunks=_fetch_over(corpus),
        ask=ask, row_segments=priced_row_segments,
    )
    labels = [_label(ln) for ln in _row_lines(action("ductile iron pipe"))]
    assert labels == [label, "[Bill No. 1 - SITEWORKS]",
                      "[Bill No. 2 - BUILDINGS]", "[Bill No. 3 - EXTERNAL WORKS]"]


@pytest.mark.parametrize(("text", "count"), [
    ("Pipe 110mm\nm\n300\n45.00\n13,500.00\nKerb\nm\n40\n180.00\n7,200.00", 2),
    ("Preamble only, no priced row: 12 m of pipe", 0),
])
def test_priced_row_segments_match_the_row_count(text, count):
    assert len(priced_row_segments(text)) == count == priced_row_count(text)


@pytest.mark.parametrize(("text", "first", "second"), [
    ("A1 | Pipe | m | 300 | 45.00 | 13,500.00\nA2 | Kerb | m | 40 | 180.00 | 7,200.00",
     "A1 Pipe m 300 45.00 13,500.00", "\nA2 Kerb m 40 180.00 7,200.00"),
    ("B1 Duct m 10 5.00 50.00 B2 Valve nr 2 300.00 600.00",
     "B1 Duct m 10 5.00 50.00", " B2 Valve nr 2 300.00 600.00"),
])
def test_priced_row_segments_carry_the_description_above_each_row(text, first, second):
    assert priced_row_segments(text) == [first, second]
