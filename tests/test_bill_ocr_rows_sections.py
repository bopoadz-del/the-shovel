"""S-A10: an OCR'd priced bill gives its sections and its whole rows.

Every test builds its own synthetic OCR-shaped bill text. The shapes are the
ones a scanned bill prints, not any one document's words:

* the heading printed as a page header (number line, title on the next line),
  in the body (number and title on one line), or inside a table cell after a
  stray mark, with the "O" of "NO" read as a zero;
* a row whose description wraps over several lines with the figures on the
  last line in pipe-separated columns, against a row printed on one line;
* a chunk boundary falling inside a wrapped row;
* a weight column printing "tonne" / "ton" / "tn" against "t".
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from app.core.rag.bill_sections import item_rows, section_label, section_map
from app.core.rag.evidence_actions import bill_rows_action
from app.core.rag.retriever import priced_row_count, priced_row_segments

DOC = "ocr-bill"


def _pairs(texts):
    return list(enumerate(texts))


def _rows(texts, item):
    pairs = _pairs(texts)
    return item_rows(pairs, section_map(pairs), item, row_segments=priced_row_segments)


def _figures(qty, rate):
    return f"| {qty:,} | {rate:,.2f} | {qty * rate:,.2f}"


# --- (a) headings ---------------------------------------------------------

@pytest.mark.parametrize(("heading", "number", "title"), [
    ("BILL NO. 3\nEARTHWORKS", "3", "EARTHWORKS"),               # page header
    ("BILL NO. 7 - CONCRETE WORKS [cont'd]", "7", "CONCRETE WORKS [cont'd]"),
    ("|BILL N0. 5\nMASONRY", "5", "MASONRY"),                    # table cell, 0 for O
    ("' BILL NO, 12 | DRAINAGE", "12", "DRAINAGE"),              # stray mark, comma
])
def test_ocr_heading_forms_open_a_section(heading, number, title):
    sections = section_map([(0, heading + "\nA. |Item text\nm3 | 10 | 5.00 | 50.00")])
    assert [(s.number, s.title) for s in sections] == [(number, title)]


@pytest.mark.parametrize("sentence", [
    "Rates in Bill No. 3 include overheads.",
    "Refer to item 4 of bill no. 2 for testing.",
])
def test_a_sentence_mentioning_a_bill_is_not_a_heading(sentence):
    assert section_map([(0, sentence)]) == []


@pytest.mark.parametrize(("first", "second"), [("2", "3"), ("4", "9")])
def test_every_chunk_of_a_bill_page_carries_its_bill(first, second):
    """Page-header bills: each chunk is labelled with the nearest heading
    above it in page order, including chunks that print no heading."""
    texts = [
        f"|BILL NO. {first}\nGENERAL ITEMS\nA. |Site office\nsum | 1 | 900.00 | 900.00",
        "B. |Survey of the site\nnr | 2 | 40.00 | 80.00",
        f"BILL NO. {first}\nGENERAL ITEMS\nC. |Temporary fencing\nm | 10 | 3.00 | 30.00",
        f"BILL N0. {second}\nCONCRETE\nA. |Blinding\nm3 | 4 | 60.00 | 240.00",
        "B. |Mass concrete\nm3 | 5 | 70.00 | 350.00",
    ]
    sections = section_map(_pairs(texts))
    labels = [section_label(i, sections) for i in range(len(texts))]
    assert labels[:2] == [f"[Bill No. {first} - GENERAL ITEMS]"] * 2
    assert labels[4] == f"[Bill No. {second} - CONCRETE]"


# --- (b) row assembly -----------------------------------------------------

def _wrapped_row(ref, noun, qty, rate):
    return (
        f"{ref}. |Supply, cut, bend and fix high yield\n"
        f"{noun} to approved shapes including\n"
        f"tying wire, spacers and chairs as\n"
        f"specified. | t {_figures(qty, rate)}"
    )


@pytest.mark.parametrize(("qty", "rate"), [(12, 3100), (48, 2950)])
def test_a_wrapped_row_is_one_row_with_its_figures(qty, rate):
    texts = ["BILL NO. 4\nCONCRETE\n" + _wrapped_row("A", "steel reinforcement", qty, rate)]
    rows = _rows(texts, "reinforcement")
    assert len(rows) == 1
    assert "reinforcement" in rows[0] and f"{qty * rate:,.2f}" in rows[0]


@pytest.mark.parametrize("cut", [1, 2, 3])
def test_a_row_cut_by_a_chunk_boundary_keeps_its_description(cut):
    """The chunk boundary falls after line ``cut`` of a wrapped row: the
    item word sits in the chunk above, the figures in the chunk below."""
    lines = _wrapped_row("A", "steel reinforcement", 12, 3100).split("\n")
    texts = [
        "BILL NO. 4\nCONCRETE\nB. |Formwork to soffits\nm2 | 30 | 25.00 | 750.00\n"
        + "\n".join(lines[:cut]),
        "\n".join(lines[cut:]) + "\nC. |Formwork to sides\nm2 | 20 | 22.00 | 440.00",
    ]
    rows = _rows(texts, "steel reinforcement")
    assert len(rows) == 1
    assert "37,200.00" in rows[0] and "[Bill No. 4 - CONCRETE]" in rows[0]


@pytest.mark.parametrize("gap", [2, 5])
def test_chunks_that_are_not_adjacent_are_not_joined(gap):
    """Only the chunk printed directly before continues into a chunk."""
    texts = {
        0: "BILL NO. 4\nCONCRETE\nA. |Supply high yield steel reinforcement",
        gap: "to shapes | t | 12 | 3,100.00 | 37,200.00",
    }
    pairs = sorted(texts.items())
    rows = item_rows(pairs, section_map(pairs), "reinforcement",
                     row_segments=priced_row_segments)
    assert rows == []


@pytest.mark.parametrize("count", [3, 6])
def test_wrapped_rows_across_many_chunks_are_each_listed_once(count):
    """Each chunk ends inside the next row: every row is still listed once."""
    body = "\n".join(
        _wrapped_row(chr(65 + i), "steel reinforcement", 10 + i, 3000)
        for i in range(count)
    ).split("\n")
    # Cut the lines into chunks of three lines: rows are four lines long, so
    # every boundary but the first falls inside a row.
    chunks = ["BILL NO. 4\nCONCRETE"] + [
        "\n".join(body[i:i + 3]) for i in range(0, len(body), 3)
    ]
    rows = _rows(chunks, "reinforcement")
    assert len(rows) == count
    assert sorted(r.rsplit(" ", 1)[-1] for r in rows) == sorted(
        f"{(10 + i) * 3000:,.2f}" for i in range(count))


@pytest.mark.parametrize("unit", ["tonne", "Tonnes", "ton", "Tons", "tn", "t"])
def test_a_weight_column_unit_is_a_priced_row(unit):
    text = f"A. |High yield bars\n| {unit} | 12 | 3,100.00 | 37,200.00"
    assert priced_row_count(text) == 1 == len(priced_row_segments(text))


@pytest.mark.parametrize("unit", ["nos", "Nos"])
def test_a_count_column_unit_is_a_priced_row(unit):
    text = f"Gully grating\n{unit} 6 150.00 900.00"
    assert priced_row_count(text) == 1


@pytest.mark.parametrize("text", [
    "Lap length 40 t 3,100.00 37,200.00",            # 40 x 3,100 != 37,200
    "Grade 500 tonnes per year 20 50.00",            # not a row shape
])
def test_unit_words_alone_do_not_make_a_row(text):
    assert priced_row_count(text) == 0


# --- the action and its trace ---------------------------------------------

def _chunk(index, text):
    return SimpleNamespace(
        doc_id=DOC, chunk_id=f"{DOC}:{index}", chunk_index=index, text=text,
        score=None,
    )


@pytest.mark.parametrize(("qty", "rate"), [(12, 3100), (30, 2875)])
def test_action_lists_the_wrapped_row_and_traces_the_map(qty, rate, caplog):
    lines = _wrapped_row("A", "steel reinforcement", qty, rate).split("\n")
    corpus = [
        _chunk(0, "BILL NO. 4\nCONCRETE\nB. |Formwork\nm2 | 30 | 25.00 | 750.00\n"
               + "\n".join(lines[:2])),
        _chunk(1, "\n".join(lines[2:])),
    ]

    def bm25(project_id, query, k=50, doc_ids=None):
        return list(corpus)

    action = bill_rows_action(
        bm25, "p1", row_count=priced_row_count, typed_bills=lambda _p: [DOC],
        bill_chunks=lambda _p, ids: [c for c in corpus if c.doc_id in ids],
        ask="total reinforcement", row_segments=priced_row_segments,
    )
    with caplog.at_level(logging.INFO, logger="app.core.rag.evidence_actions"):
        text = action("steel reinforcement")
    listed = [ln for ln in text.splitlines() if ln.startswith("[Bill No. 4 - CONCRETE] ")]
    assert len(listed) == 1 and f"{qty * rate:,.2f}" in listed[0]
    trace = [r.getMessage() for r in caplog.records
             if "rag_bill_section_trace" in r.getMessage()]
    assert trace and f"mapped={{'{DOC}': 1}}" in trace[-1]
