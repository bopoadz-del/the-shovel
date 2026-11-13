"""S-A13: for a billed-quantity or rate ask, a typed bill's rows lead.

Every test builds its own first-pass context through the real excerpt
formatter (``inject.format_chunks_as_system_message``), so the marker this
module splits on cannot drift from the one the context carries. Documents
are typed by the ``doc_type`` attribute alone; the names here are invented
and nothing in the product code reads them.
"""
from __future__ import annotations

import pytest

from app.agents import runtime
from app.agents.runtime import _agentic_figure_retrieval_augment, _bill_led_rag_msg
from app.core.rag.evidence_actions import (
    BILL_LEAD_NOTE,
    NO_BILL_ROW_NOTE,
    bill_led_evidence,
)
from app.core.rag.inject import format_chunks_as_system_message
from app.core.rag.retriever import priced_row_count
from app.core.rag.vector_store import Chunk

#: (item description, bill row, workbook row). A bill row and a workbook
#: row for the same item, each a priced row (qty x rate = amount).
ITEMS = [
    (
        "chamber type 4, 900 mm diameter, depth 1.5-2.0 m",
        "C Chamber type 4 diameter 900 depth 1.5-2.0 m Nr 12 4,100.00 49,200.00",
        "Chamber type 4 diameter 900 depth 1.5-2.0 m Nr 3 5,200.00 15,600.00",
    ),
    (
        "gully grating class D400",
        "F Gully grating class D400 Nr 40 250.00 10,000.00",
        "Gully grating class D400 Nr 55 240.00 13,200.00",
    ),
]

BILL_ID, SHEET_ID, SPEC_ID = "doc-bill-1", "doc-sheet-1", "doc-spec-1"
NAMES = {
    BILL_ID: "volume_3_priced_bill.pdf",
    SHEET_ID: "site_material_tracker.xlsx",
    SPEC_ID: "drainage_specification.pdf",
}


def _chunk(doc_id: str, idx: int, text: str, score: float) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}-{idx}", project_id="p-test", doc_id=doc_id,
        chunk_index=idx, text=text, score=score, source_name=NAMES[doc_id],
    )


def _context(*chunks: Chunk) -> str:
    return format_chunks_as_system_message(list(chunks), len(chunks))["content"]


def _excerpt_order(content: str) -> list[str]:
    import re

    return re.findall(r"\[doc_id=([^\]\s]+) chunk=", content)


def _lead(content: str, typed) -> str:
    return bill_led_evidence(
        content, typed_bills=typed, row_count=priced_row_count, doc_name=NAMES.get,
    )


@pytest.mark.parametrize("item,bill_row,sheet_row", ITEMS)
def test_bill_row_leads_a_workbook_row_for_the_same_item(item, bill_row, sheet_row):
    # The workbook outranks the bill in the first pass.
    content = _context(
        _chunk(SHEET_ID, 0, sheet_row, 0.9),
        _chunk(SPEC_ID, 4, f"The {item} shall be installed per clause 5.", 0.8),
        _chunk(BILL_ID, 7, bill_row, 0.7),
    )
    out = _lead(content, [BILL_ID])
    assert _excerpt_order(out) == [BILL_ID, SHEET_ID, SPEC_ID]
    note = BILL_LEAD_NOTE.format(
        bills=NAMES[BILL_ID], bill_verb="is", others=NAMES[SHEET_ID], other_verb="is",
    )
    assert note in out
    # The note sits before every excerpt, and every excerpt survives whole.
    assert out.index(note) < out.index("[doc_id=")
    assert out.count("[doc_id=") == 3
    assert bill_row in out and sheet_row in out
    # The header the formatter wrote is kept ahead of the note.
    header = content.split("[doc_id=")[0].rstrip()
    assert out.startswith(header)


@pytest.mark.parametrize("item,bill_row,sheet_row", ITEMS)
def test_workbook_only_keeps_its_row_leading_and_says_not_a_bill(item, bill_row, sheet_row):
    content = _context(
        _chunk(SHEET_ID, 0, sheet_row, 0.9),
        _chunk(BILL_ID, 2, f"Bill 4 Drainage. Item descriptions for {item} follow.", 0.6),
    )
    out = _lead(content, [BILL_ID])
    assert _excerpt_order(out) == [SHEET_ID, BILL_ID]
    note = NO_BILL_ROW_NOTE.format(others=NAMES[SHEET_ID], other_pronoun="it is")
    assert note in out
    assert out.index(note) < out.index("[doc_id=")
    assert "the lead figure is the bill's own row" not in out


@pytest.mark.parametrize("typed", [[], ["doc-not-in-this-context"]])
def test_untyped_or_absent_bills_leave_the_context_as_it_was(typed):
    _item, bill_row, sheet_row = ITEMS[0]
    content = _context(
        _chunk(SHEET_ID, 0, sheet_row, 0.9),
        _chunk(BILL_ID, 7, bill_row, 0.7),
    )
    out = _lead(content, typed)
    if typed:
        # A typed bill exists but none is in the context: the workbook is
        # named as not a bill, the order is the first pass's.
        assert _excerpt_order(out) == [SHEET_ID, BILL_ID]
        assert "no excerpt here that prints a priced row" in out
        assert f"taken from {NAMES[SHEET_ID]}, {NAMES[BILL_ID]}" in out
        assert "they are not a bill" in out
    else:
        # A project with no typed bill behaves as today.
        assert out == content


@pytest.mark.parametrize("bill_row", [ITEMS[0][1], ITEMS[1][1]])
def test_bill_alone_or_no_priced_row_is_unchanged(bill_row):
    only_bill = _context(
        _chunk(BILL_ID, 7, bill_row, 0.9),
        _chunk(SPEC_ID, 4, "Chambers shall be precast per clause 5.", 0.8),
    )
    assert _lead(only_bill, [BILL_ID]) == only_bill
    no_rows = _context(
        _chunk(SHEET_ID, 1, "Delivery schedule: chambers week 12.", 0.9),
        _chunk(SPEC_ID, 4, "Chambers shall be precast per clause 5.", 0.8),
    )
    assert _lead(no_rows, [BILL_ID]) == no_rows


@pytest.mark.parametrize("n_sheets", [1, 2])
def test_two_bills_and_sheets_are_all_named_each_group_in_order(n_sheets):
    _item, bill_row, sheet_row = ITEMS[0]
    second_bill, second_sheet = "doc-bill-2", "doc-sheet-2"
    names = {**NAMES, second_bill: "volume_4_priced_bill.pdf",
             second_sheet: "quantity_comparison.xlsx"}
    chunks = [Chunk(chunk_id="s1", project_id="p", doc_id=SHEET_ID, chunk_index=0,
                    text=sheet_row, score=0.95, source_name=names[SHEET_ID])]
    if n_sheets == 2:
        chunks.append(Chunk(chunk_id="s2", project_id="p", doc_id=second_sheet,
                            chunk_index=0, text=sheet_row, score=0.9,
                            source_name=names[second_sheet]))
    chunks += [
        Chunk(chunk_id="b2", project_id="p", doc_id=second_bill, chunk_index=3,
              text=bill_row, score=0.8, source_name=names[second_bill]),
        Chunk(chunk_id="b1", project_id="p", doc_id=BILL_ID, chunk_index=9,
              text=bill_row, score=0.7, source_name=names[BILL_ID]),
        Chunk(chunk_id="b1b", project_id="p", doc_id=BILL_ID, chunk_index=10,
              text="continued from previous page", score=0.6,
              source_name=names[BILL_ID]),
    ]
    out = bill_led_evidence(
        _context(*chunks), typed_bills=[BILL_ID, second_bill],
        row_count=priced_row_count, doc_name=names.get,
    )
    sheets = [SHEET_ID] + ([second_sheet] if n_sheets == 2 else [])
    assert _excerpt_order(out) == [second_bill, BILL_ID, BILL_ID] + sheets
    others = ", ".join(names[s] for s in sheets)
    note = BILL_LEAD_NOTE.format(
        bills=f"{names[second_bill]}, {names[BILL_ID]}", bill_verb="are",
        others=others, other_verb="is" if n_sheets == 1 else "are",
    )
    assert note in out


@pytest.mark.parametrize("name", [None, ""])
def test_a_document_without_a_display_name_is_named_by_its_id(name):
    _item, bill_row, sheet_row = ITEMS[1]
    out = bill_led_evidence(
        _context(_chunk(SHEET_ID, 0, sheet_row, 0.9), _chunk(BILL_ID, 1, bill_row, 0.8)),
        typed_bills=[BILL_ID], row_count=priced_row_count, doc_name=lambda _d: name,
    )
    assert f"[bill source rule: {BILL_ID} is typed as a bill; {SHEET_ID} is not." in out


# ── runtime seam ──────────────────────────────────────────────────────────

FIGURE_ASKS = [
    "What is the billed quantity of chamber type 4, 900 mm diameter?",
    "what is the rate for gully grating class D400",
]


def _docs(doc_type_of_sheet: str = "document"):
    return lambda _pid: [
        {"id": BILL_ID, "original_name": NAMES[BILL_ID], "doc_type": "BOQ "},
        {"id": SHEET_ID, "original_name": NAMES[SHEET_ID], "doc_type": doc_type_of_sheet},
        {"id": None, "original_name": "orphan", "doc_type": "boq"},
    ]


@pytest.mark.parametrize("ask", FIGURE_ASKS)
def test_runtime_leads_with_the_typed_bill_for_a_figure_ask(ask):
    _item, bill_row, sheet_row = ITEMS[0]
    msg = {"role": "system", "content": _context(
        _chunk(SHEET_ID, 0, sheet_row, 0.9), _chunk(BILL_ID, 7, bill_row, 0.7),
    )}
    out = _bill_led_rag_msg(ask, "p-test", msg, documents=_docs())
    assert out["role"] == "system"
    assert _excerpt_order(out["content"]) == [BILL_ID, SHEET_ID]
    assert f"{NAMES[BILL_ID]} is typed as a bill; {NAMES[SHEET_ID]} is not" in out["content"]
    assert msg["content"] != out["content"]  # the caller's message is not mutated


@pytest.mark.parametrize("ask", [
    "Summarise the drainage method statement",
    "export the bill to excel",
])
def test_runtime_leaves_a_non_figure_ask_alone(ask):
    _item, bill_row, sheet_row = ITEMS[0]
    msg = {"role": "system", "content": _context(
        _chunk(SHEET_ID, 0, sheet_row, 0.9), _chunk(BILL_ID, 7, bill_row, 0.7),
    )}

    reads: list[str] = []

    def _recording(pid):
        reads.append(pid)
        return _docs()(pid)

    # The documents would reorder this context; a non-figure ask never reads them.
    assert _bill_led_rag_msg(ask, "p-test", msg, documents=_recording) is msg
    assert reads == []


@pytest.mark.parametrize("failure", [RuntimeError("db down"), ValueError("bad row")])
def test_runtime_document_read_failure_keeps_the_message(failure):
    _item, bill_row, sheet_row = ITEMS[0]
    msg = {"role": "system", "content": _context(
        _chunk(SHEET_ID, 0, sheet_row, 0.9), _chunk(BILL_ID, 7, bill_row, 0.7),
    )}

    def _fail(_pid):
        raise failure

    assert _bill_led_rag_msg(FIGURE_ASKS[0], "p-test", msg, documents=_fail) is msg


@pytest.mark.parametrize("sheet_type", ["boq", "document"])
def test_runtime_types_documents_by_doc_type_only(sheet_type):
    _item, bill_row, sheet_row = ITEMS[1]
    msg = {"role": "system", "content": _context(
        _chunk(SHEET_ID, 0, sheet_row, 0.9), _chunk(BILL_ID, 7, bill_row, 0.7),
    )}
    out = _bill_led_rag_msg(FIGURE_ASKS[1], "p-test", msg, documents=_docs(sheet_type))
    if sheet_type == "boq":
        # Both are typed bills: nothing is secondary, the message is kept.
        assert out is msg
    else:
        assert _excerpt_order(out["content"]) == [BILL_ID, SHEET_ID]


@pytest.mark.parametrize("ask", FIGURE_ASKS)
async def test_augment_hands_the_composer_and_model_the_bill_first(monkeypatch, ask):
    """The reorder happens before the no-LLM composer reads the first pass."""
    import app.core.projects as projects
    import app.core.rag.retriever as retriever

    seen: list[str] = []

    def _compose(_q, evidence):
        seen.append(evidence)
        return {"qty": "1", "rate": "1", "amount": "1"}

    monkeypatch.setattr(projects, "list_document_names", _docs())
    monkeypatch.setattr(retriever, "compose_priced_boq_row", _compose)
    _item, bill_row, sheet_row = ITEMS[0]
    rag = {"role": "system", "content": _context(
        _chunk(SHEET_ID, 0, sheet_row, 0.9), _chunk(BILL_ID, 7, bill_row, 0.7),
    )}

    class _NoLLM:
        async def _call_llm(self, *a, **k):
            raise AssertionError("no LLM call when the composer answers")

    out = await _agentic_figure_retrieval_augment(_NoLLM(), ask, "p-test", "k", "u", rag)
    assert _excerpt_order(out["content"]) == [BILL_ID, SHEET_ID]
    assert seen and _excerpt_order(seen[0]) == [BILL_ID, SHEET_ID]
    assert runtime._bill_led_rag_msg is _bill_led_rag_msg
