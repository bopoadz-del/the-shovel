"""The bill-rows action: a figure item that is only in a priced bill reaches
the evidence under the #210 budget, whatever the bill's file shape.

Every test builds its own corpus. Two corpus shapes are covered: a PDF-shaped
bill (page text, rows split across lines by the extractor) and an XLSX-shaped
bill (tab-separated rows). In both, drawings and specifications that merely
mention the item out-rank every bill chunk in the global lexical ranking, so
the plain lexical action's top-k never reaches the bill.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from app.core.rag.evidence_actions import (
    _bill_rows_retrieve,
    bill_document_ids,
    bill_rows_action,
    lexical_action,
)
from app.core.rag.evidence_loop import run_project_figure_retrieval
from app.core.rag.retriever import priced_row_count


def _c(doc_id, text, score):
    return SimpleNamespace(doc_id=doc_id, text=text, score=score)


def _noise(n, item_words):
    """``n`` drawing/spec chunks that mention the item but price nothing."""
    out = []
    for i in range(n):
        doc = "drawing-vol" if i % 2 else "spec-vol"
        out.append(_c(
            doc,
            f"Section 3.{i} {item_words} shall comply with clause 2.10. "
            f"Grid 6000 x 7200, level +3.450, see sheet {100 + i}.",
            10.0 - i * 0.01,
        ))
    return out


def _pdf_bill(rows):
    """A PDF-shaped bill: one chunk per row block, cells on separate lines."""
    out = []
    for i, (desc, qty, unit, rate) in enumerate(rows):
        amount = qty * rate
        out.append(_c(
            "bill-pdf",
            f"Page {i + 7}\nItem {i + 1}\n{desc}\n{unit}\n{qty:,}\n"
            f"{rate:,.2f}\n{amount:,.2f}",
            1.0 - i * 0.01,
        ))
    return out


def _xlsx_bill(rows):
    """An XLSX-shaped bill: tab-separated rows, unit before quantity."""
    out = []
    for i, (desc, qty, unit, rate) in enumerate(rows):
        amount = qty * rate
        out.append(_c(
            "bill-xlsx",
            f"Ref\tDescription\tUnit\tQty\tRate\tAmount\n"
            f"B{i + 1}\t{desc}\t{unit}\t{qty}\t{rate:.2f}\t{amount:.2f}",
            1.0 - i * 0.01,
        ))
    return out


_ROWS = [
    ("Precast kerb type B laid to line", 840, "m", 42.5),
    ("Geotextile separator membrane", 3120, "m2", 6.75),
    ("Granular sub-base 150 thick", 1460, "m3", 31.0),
]


def _bm25(corpus):
    """Fake ``store.bm25_search``: chunks sharing a word with the query, by score."""
    def _search(project_id, query, k=50):
        words = {w.lower() for w in query.split() if len(w) > 3}
        hits = [c for c in corpus if words & {w.lower().strip(".,") for w in c.text.split()}]
        return sorted(hits, key=lambda c: c.score, reverse=True)[:k]
    return _search


def _no_semantic(query, project_id, k=5):
    return [], 0


@pytest.mark.parametrize("bill_shape", [_pdf_bill, _xlsx_bill])
def test_priced_rows_mark_the_bill_and_not_drawings_or_specs(bill_shape):
    bill = bill_shape(_ROWS)
    noise = _noise(10, "kerb geotextile sub-base")
    assert all(priced_row_count(c.text) == 1 for c in bill)
    assert all(priced_row_count(c.text) == 0 for c in noise)
    assert bill_document_ids(bill + noise, row_count=priced_row_count) == {
        bill[0].doc_id,
    }


@pytest.mark.parametrize("text", [
    "Clause 4.10 cover 50 mm, 3 m 12.00 40.00 lap length",
    "Provisional item 0 m 5.00 0.00",
    "Forecast 2026 m3 1200 1300 1400",
    "",
])
def test_numbers_that_do_not_multiply_out_are_not_priced_rows(text):
    assert priced_row_count(text) == 0


@pytest.mark.parametrize("bill_shape,item,figure", [
    (_pdf_bill, "geotextile separator membrane", "21,060.00"),
    (_xlsx_bill, "granular sub-base", "45260.00"),
])
def test_bill_only_item_crowded_out_of_lexical_reaches_bill_rows(bill_shape, item, figure):
    corpus = _noise(80, item) + bill_shape(_ROWS)
    bm25 = _bm25(corpus)
    lexical = lexical_action(bm25, "p")(item)
    assert figure not in lexical  # the global top-50 is all drawings/specs
    handed = bill_rows_action(bm25, "p", row_count=priced_row_count)(item)
    assert figure in handed
    assert "Section 3." not in handed  # nothing from the non-bill documents


@pytest.mark.parametrize("bill_shape,item,figure", [
    (_pdf_bill, "precast kerb", "35,700.00"),
    (_xlsx_bill, "geotextile separator membrane", "21060.00"),
])
async def test_loop_evidence_contains_bill_only_figure_under_budget(
    monkeypatch, bill_shape, item, figure,
):
    monkeypatch.setenv("AGENTIC_EVIDENCE_MAX_TOKENS", "2000")
    monkeypatch.setenv("AGENTIC_EVIDENCE_MAX_TOKENS_PER_DOC", "400")
    corpus = _noise(80, item) + bill_shape(_ROWS)

    async def _plan(question):
        return {"choice": {"message": {"content": f"{item} rate amount"}}}

    outcome = await run_project_figure_retrieval(
        question=f"what is the amount for {item}?",
        plan_caller=_plan,
        retrieve_fn=_no_semantic,
        bm25_fn=_bm25(corpus),
        project_id="p",
        first_pass_evidence="unrelated first pass",
    )
    assert figure in outcome.evidence
    assert len(outcome.evidence) // 4 <= 2000 + 200  # budget + truncation note


@pytest.mark.parametrize("bill_shape", [_pdf_bill, _xlsx_bill])
async def test_item_in_neither_bill_nor_documents_stays_missing(bill_shape):
    corpus = _noise(20, "kerb") + bill_shape(_ROWS)

    async def _plan(question):
        return {"choice": {"message": {"content": "waterproofing tanking"}}}

    outcome = await run_project_figure_retrieval(
        question="what is the rate for waterproofing tanking?",
        plan_caller=_plan,
        retrieve_fn=_no_semantic,
        bm25_fn=_bm25(corpus),
        project_id="p",
        first_pass_evidence="unrelated first pass",
    )
    assert outcome.missing == ("waterproofing tanking",)


@pytest.mark.parametrize("bill_shape", [_pdf_bill, _xlsx_bill])
def test_bill_rows_respects_the_per_document_cap(monkeypatch, bill_shape):
    rows = [(f"Item line {i} kerb", 10 + i, "m", 5.0) for i in range(60)]
    bill = bill_shape(rows)
    per_doc = 120
    out = bill_rows_action(_bm25(bill), "p", row_count=priced_row_count)
    monkeypatch.setenv("AGENTIC_EVIDENCE_MAX_TOKENS_PER_DOC", str(per_doc))
    text = out("kerb")
    body = text.split("\n[context_truncated")[0]
    assert len(body) // 4 <= per_doc + 60  # whole chunks; joins add newlines
    assert "context_truncated" in text


@pytest.mark.parametrize("item", ["kerb", "geotextile"])
def test_bill_rows_logs_the_handed_documents(caplog, item):
    corpus = _noise(5, item) + _pdf_bill(_ROWS)
    with caplog.at_level(logging.INFO, logger="app.core.rag.evidence_actions"):
        _bill_rows_retrieve(_bm25(corpus), "p", 200, priced_row_count, item)
    lines = [r.getMessage() for r in caplog.records
             if "rag_evidence_action_trace action=bill_rows" in r.getMessage()]
    assert lines and "bill-pdf" in lines[0]
    assert "spec-vol" not in lines[0]


@pytest.mark.parametrize("raw,want", [
    ("", 200), ("abc", 200), ("0", 200), ("-5", 200), ("75", 75), ("400", 400),
])
def test_bill_pool_depth_reads_env_with_a_safe_default(monkeypatch, raw, want):
    from app.core.rag.evidence_actions import bill_pool_k
    monkeypatch.setenv("AGENTIC_BILL_POOL_K", raw)
    assert bill_pool_k() == want


@pytest.mark.parametrize("depth", [3, 7])
def test_bill_rows_pool_depth_is_what_the_lexical_call_gets(monkeypatch, depth):
    seen = []

    def _search(project_id, query, k=50):
        seen.append(k)
        return []

    monkeypatch.setenv("AGENTIC_BILL_POOL_K", str(depth))
    assert bill_rows_action(_search, "p", row_count=priced_row_count)("kerb") == ""
    assert seen == [depth]


@pytest.mark.parametrize("docs,want", [
    (["a", "a", "b", "", "a"], ["a", "b"]),
    (["x", "", "x"], ["x"]),
])
def test_handed_docs_are_distinct_in_order(docs, want):
    from app.core.rag.evidence_actions import _handed_docs
    assert _handed_docs([_c(d, "t", 1.0) for d in docs]) == want


@pytest.mark.parametrize("doc_id", ["", None])
def test_a_chunk_without_a_document_never_marks_a_bill(doc_id):
    priced = _pdf_bill(_ROWS)[0].text
    chunks = [SimpleNamespace(doc_id=doc_id, text=priced, score=1.0)]
    assert bill_document_ids(chunks, row_count=priced_row_count) == set()


@pytest.mark.parametrize("bill_shape,item,figure", [
    (_pdf_bill, "precast kerb", "35,700.00"),
    (_xlsx_bill, "granular sub-base", "45260.00"),
])
def test_bill_rows_retrieve_hands_only_bill_chunks(bill_shape, item, figure):
    corpus = _noise(60, item) + bill_shape(_ROWS)
    handed = _bill_rows_retrieve(_bm25(corpus), "p", 200, priced_row_count, item)
    assert figure in handed
    assert "Section 3." not in handed
