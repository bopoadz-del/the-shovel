"""L1: a bill is segmented into priced rows once per evidence action.

The evidence loop asks the bill-rows action once per plan item. Splitting a
bill into labelled priced rows does not depend on the item, so the action
builds each bill's row index once and only scores each item against it.
Nothing the model reads may change: every test here compares the evidence
text byte for byte with the per-item path, on bills built in the test.
"""
from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from app.core.rag import bill_sections
from app.core.rag.bill_sections import row_index, rows_for_item, section_map
from app.core.rag.evidence_actions import bill_rows_action
from app.core.rag.retriever import priced_row_count, priced_row_segments


def _row(desc, unit, qty, rate):
    return f"{desc}\n{unit}\n{qty:,}\n{rate:,.2f}\n{qty * rate:,.2f}"


def _chunk(doc_id, index, text):
    return SimpleNamespace(
        doc_id=doc_id, chunk_id=f"{doc_id}:{index}", chunk_index=index,
        text=text, score=None,
    )


def _headed_bill(doc_id):
    """Three headed sections; culvert and gabion rows recur across them."""
    texts = [
        "BILL NO. 1\nEARTHWORKS\n"
        + _row("Excavate trench for culvert in rock", "m3", 310, 42)
        + "\n" + _row("Gabion basket filled with stone", "m3", 55, 180),
        "BILL NO. 2\nCULVERTS\n"
        + _row("Precast box culvert 2x1.5m laid", "m", 64, 1450)
        + "\n" + _row("Gabion mattress to culvert outlet", "m2", 90, 75),
        "BILL NO. 3\nROADWORKS\n"
        + _row("Asphalt wearing course 50mm", "m2", 2400, 31)
        + "\n" + _row("Culvert headwall in situ concrete", "nr", 6, 5200),
    ]
    return [_chunk(doc_id, i, t) for i, t in enumerate(texts)]


def _plain_bill(doc_id):
    """No headings at all: every row is "section unknown"."""
    texts = [
        _row("Timber hoarding to site perimeter", "m", 420, 38)
        + "\n" + _row("Galvanised palisade fence 2.4m", "m", 180, 96),
        _row("Timber formwork to pile caps", "m2", 260, 44)
        + "\n" + _row("Palisade gate double leaf", "nr", 2, 3800),
    ]
    return [_chunk(doc_id, i, t) for i, t in enumerate(texts)]


def _bm25_over(chunks):
    def bm25(project_id, query, k=50, doc_ids=None):
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


def _action(corpus, bills, ask, segments=priced_row_segments):
    return bill_rows_action(
        _bm25_over(corpus), "p1", row_count=priced_row_count,
        typed_bills=lambda _pid: list(bills),
        bill_chunks=lambda _pid, ids: [c for c in corpus if c.doc_id in ids],
        ask=ask, row_segments=segments,
    )


CASES = [
    pytest.param(
        _headed_bill("bq-a"), ["bq-a"], "culvert rates",
        ["culvert", "gabion stone", "asphalt course", "culvert in Bill No. 2"],
        id="headed-bill",
    ),
    pytest.param(
        _headed_bill("bq-b") + _plain_bill("bq-c"), ["bq-b", "bq-c"], "fencing",
        ["palisade", "timber", "gabion mattress", "formwork pile caps"],
        id="two-bills-one-unheaded",
    ),
]


@pytest.mark.parametrize(("corpus", "bills", "ask", "items"), CASES)
def test_evidence_is_byte_equal_to_a_fresh_action_per_item(corpus, bills, ask, items):
    shared = _action(corpus, bills, ask)
    for item in items:
        assert shared(item) == _action(corpus, bills, ask)(item), item


@pytest.mark.parametrize(("corpus", "bills", "ask", "items"), CASES)
def test_each_bill_is_segmented_once_per_action(corpus, bills, ask, items):
    def counted(n):
        calls = []

        def segments(text):
            calls.append(text)
            return priced_row_segments(text)

        action = _action(corpus, bills, ask, segments)
        for item in items[:n]:
            action(item)
        return len(calls)

    once = counted(1)
    assert once > 0
    # More items over the same bills segment no more chunks.
    assert counted(len(items)) == once


def _old_rule(rows, item):
    """The per-row reading of the asked words that the index replaces:
    a word is asked when SOME row covers it, read row by row."""
    row_words = [bill_sections.words(t) for _l, t in rows]
    asked = {
        w for w in bill_sections.words(item) - bill_sections._COLUMN_WORDS
        if any(bill_sections._covers(w, ws) for ws in row_words)
    }
    if not asked:
        return []
    need = (len(asked) + 1) // 2
    out, seen = [], set()
    for (label, text), ws in zip(rows, row_words):
        if sum(bill_sections._covers(w, ws) for w in asked) < need:
            continue
        key = (label, " ".join(text.lower().split()[-bill_sections._ROW_KEY_TOKENS:]))
        if key not in seen:
            seen.add(key)
            out.append(f"{label} {text}")
    return out


@pytest.mark.parametrize(("chunks", "items"), [
    (_headed_bill("x"), ["culverts", "gabion basket stone", "nothing billed"]),
    (_plain_bill("y"), ["palisade fencing", "timberwork", "concrete"]),
])
def test_vocabulary_reading_selects_the_rows_the_per_row_reading_does(chunks, items):
    pairs = [(c.chunk_index, c.text) for c in chunks]
    index = row_index(pairs, section_map(pairs), row_segments=priced_row_segments)
    assert index.rows
    hits = 0
    for item in items:
        got = rows_for_item(index, item)
        assert got == _old_rule(list(index.rows), item), item
        hits += bool(got)
    # Each bill has an item that selects rows and one that selects none.
    assert 0 < hits < len(items)
