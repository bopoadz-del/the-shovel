"""The conflict line at the answer's real exit (``answer_exit.check_end_event``)
for the three shapes seen live, and the positives that must stay.

1. A "conflict" made of citation markers, list or row numbers and document
   codes: never a value, so never a line.
2. A plain question with one right answer, where another handed passage
   carries another number for ANOTHER thing: no line.
3. A headline that says the sources agree and states one value, while two
   handed documents give two values: the conflict statement LEADS, names
   both values and both documents in words, and no sentence asserting only
   one of the values for that thing is left.

Negatives: documents that agree are untouched; a reply that already states
both values is untouched.

Every document, value and question here is invented for this file; the
expected values are written by hand. Each decision is shown on two inputs.
"""

from __future__ import annotations

import re

import pytest

from app.agents import answer_exit
from app.agents.evidence_conflicts import extract, same
from app.core.rag.inject import format_chunks_as_system_message
from app.core.rag.vector_store import Chunk


def _chunk(doc_id: str, idx: int, name: str, text: str, page: int) -> Chunk:
    c = Chunk(chunk_id=f"{doc_id}:{idx}", project_id="proj-invented", doc_id=doc_id,
              chunk_index=idx, text=text, score=0.8)
    c.source_name = name
    c.page = page
    return c


def _run(question: str, docs: list[tuple[str, str, int, str]], answer: str, cited: int = 0) -> dict:
    """One turn through the real exit: the docs are handed as pre-injected
    excerpts; the Sources panel names only the cited one (no passage)."""
    token = answer_exit._TURN.set(None)
    try:
        turn = answer_exit.open_turn()
        turn.question = question
        handed = [_chunk(d, i, n, t, p) for i, (d, n, p, t) in enumerate(docs)]
        rag = format_chunks_as_system_message(handed, total_candidates=len(handed), query=question)
        answer_exit.note_evidence(rag, [rag, {"role": "user", "content": question}], turn)
        d, n, p, _t = docs[cited]
        row = {"doc_id": d, "doc_name": n, "page_or_section": f"p. {p}", "page": p, "score": 0.8,
               "confidence": "High", "layer": "own", "layer_label": "Project"}
        return answer_exit.check_end_event({"type": "end", "content": answer, "sources": [row]}, turn=turn)
    finally:
        answer_exit._TURN.reset(token)


# ── 1. markers, list numbers and codes are never values ─────────────────────

_MARKER_SHAPE = [
    pytest.param(
        "Which bill item covers bulk excavation?",
        [("bq-1", "Harlow Yard Bill No 2.pdf", 3, "Item No: 1, bulk excavation to reduced level."),
         ("bq-2", "Harlow Yard Bill No 3.pdf", 5, "Item No: 6, bulk excavation in rock."),
         ("bq-3", "Harlow Yard Register.pdf", 1, "Reference: HY-INF-204-BOQ, bulk excavation items priced.")],
        "Bulk excavation is measured in Bill No 2, item 1 [source: Harlow Yard Bill No 2.pdf].",
        id="item-numbers-and-codes"),
    pytest.param(
        "Which sheet shows the outfall chamber?",
        [("dw-1", "Brook Lane Drawing Index.pdf", 2, "Sheet: 4, outfall chamber general arrangement."),
         ("dw-2", "Brook Lane Drawing Register.pdf", 7, "Sheet: 9, inlet chamber details."),
         ("dw-3", "Brook Lane Transmittal.pdf", 1, "Drawing: BL-DR-C-0412, outfall chamber sections.")],
        "The outfall chamber is on sheet 4 [source: Brook Lane Drawing Index.pdf] and in BL-DR-C-0412.",
        id="sheet-numbers-and-drawing-codes"),
]


@pytest.mark.parametrize("question, docs, answer", _MARKER_SHAPE)
def test_markers_list_numbers_and_codes_are_no_conflict(question, docs, answer):
    out = _run(question, docs, answer)
    assert not out.get("conflicts"), out.get("content")
    assert "differ" not in out["content"]


@pytest.mark.parametrize("sentence", [
    "The cover is given in the schedule [3][4].",
    "Item No: 1, see [source: Spec 75 mm Note.pdf] chunk=12 doc_id=ab-90.",
    "Reference: QX-INF-053-BOQ, page 6.",
])
def test_no_value_is_read_from_a_citation_marker_or_a_code(sentence):
    assert extract(sentence) == []


# ── 2. a plain question with one right answer ───────────────────────────────

_PLAIN_ROW = [
    pytest.param(
        "How many days of moist curing does the pump station concrete need?",
        [("sp-1", "Pump Station Concrete Specification.pdf", 14,
          "Pump station concrete shall be moist cured for not less than 7 days."),
         ("ms-1", "Concreting Method Statement.pdf", 4,
          "Concrete curing: 7 days for pump station walls."),
         ("ms-2", "Curing Compound Data Sheet.pdf", 6,
          "Pump station concrete treated with an approved curing compound: 3 days.")],
        "Moist curing is not less than 7 days [source: Pump Station Concrete Specification.pdf].",
        id="curing-days"),
    pytest.param(
        "What is the minimum cover to the wet well base slab reinforcement?",
        [("sp-2", "Wet Well Structural Notes.pdf", 3,
          "Minimum cover to wet well base slab reinforcement is 75 mm."),
         ("ms-3", "Reinforcement Method Statement.pdf", 8,
          "Cover to wet well base slab bars: 75 mm."),
         ("ms-4", "Waterproofing Method Statement.pdf", 2,
          "Wet well base slab: cover to the waterproofing membrane 50 mm.")],
        "The minimum cover is 75 mm [source: Wet Well Structural Notes.pdf].",
        id="slab-cover"),
]


@pytest.mark.parametrize("question, docs, answer", _PLAIN_ROW)
def test_another_number_for_another_thing_is_no_conflict(question, docs, answer):
    out = _run(question, docs, answer)
    assert not out.get("conflicts"), out.get("content")
    assert "differ" not in out["content"]


# ── 3. an "agree" headline over two values ──────────────────────────────────

_AGREE_HEADLINE = [
    pytest.param(
        "What strength is the reinforcement for the pump station?",
        [("dg-1", "Pump Station General Notes.pdf", 2,
          "Reinforcement for the pump station shall be high yield bars of 500 N/mm2."),
         ("ms-5", "Pump Station Method Statement.pdf", 9,
          "Reinforcement for the pump station: high yield bars of 420 N/mm2 delivered in bundles.")],
        "The documents agree: the pump station reinforcement is 420 N/mm2 "
        "[source: Pump Station Method Statement.pdf]. Bars are delivered in bundles.",
        ("500 N/mm2", "420 N/mm2"),
        ("Pump Station General Notes", "Pump Station Method Statement"),
        id="reinforcement-strength"),
    pytest.param(
        "What cover is required to the pile cap reinforcement?",
        [("dg-2", "PC-220 Pile Cap Details.pdf", 7,
          "Note 6. Minimum concrete cover to pile cap reinforcement is 75 mm."),
         ("ms-6", "Method Statement Pile Caps.pdf", 5,
          "Fixing. Spacers give a cover to pile cap reinforcement of 50 mm throughout.")],
        "Both sources agree that the pile cap reinforcement cover is 50 mm "
        "[source: Method Statement Pile Caps.pdf]. Spacers are used throughout.",
        ("75 mm", "50 mm"),
        ("PC-220 Pile Cap Details", "Method Statement Pile Caps"),
        id="pile-cap-cover"),
]


def _single_value_sentences(text: str, values: tuple[str, str]) -> list[str]:
    """Sentences (outside the conflict line) that state exactly one of the values."""
    vals = [extract(v)[0] for v in values]
    out = []
    for line in text.split("\n")[1:]:
        for sent in re.split(r"(?<=[.!?])\s+", line):
            got = extract(sent)
            stated = {i for i, v in enumerate(vals) for g in got if same(g, v)}
            if len(stated) == 1:
                out.append(sent)
    return out


@pytest.mark.parametrize("question, docs, answer, values, names", _AGREE_HEADLINE)
def test_conflict_statement_leads_and_replaces_the_one_value_headline(question, docs, answer, values, names):
    out = _run(question, docs, answer, cited=1)
    conflicts = out.get("conflicts") or []
    assert len(conflicts) == 1, out["content"]
    first = out["content"].split("\n")[0]
    assert first.startswith("The documents differ on this:")
    for v in values:
        assert v in first
    for n in names:
        assert n in first
    assert "agree" not in out["content"]
    assert _single_value_sentences(out["content"], values) == []
    # A sentence about something else stays.
    assert re.search(r"(delivered in bundles|used throughout)", out["content"])


@pytest.mark.parametrize("question, docs, answer, values, names", _AGREE_HEADLINE)
def test_documents_that_agree_are_untouched(question, docs, answer, values, names):
    agreeing = [docs[0][:3] + (docs[0][3].replace(values[0], values[1]),), docs[1]]
    out = _run(question, agreeing, answer, cited=1)
    assert not out.get("conflicts")
    assert "differ" not in out["content"]


@pytest.mark.parametrize("question, docs, answer, values, names", _AGREE_HEADLINE)
def test_a_reply_that_already_states_both_is_untouched(question, docs, answer, values, names):
    both = (f"{names[0]} gives {values[0]} and {names[1]} gives {values[1]} "
            f"[source: {docs[1][1]}]; no document says which governs.")
    out = _run(question, docs, both, cited=1)
    assert not out.get("conflicts")
    assert "The documents differ on this" not in out["content"]
    for v in values:
        assert v in out["content"]
