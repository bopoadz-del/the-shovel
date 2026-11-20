"""The thing a conflict is about comes from the QUESTION, and is read from
each value's own clause (grader 14:02Z, points 2 and 3).

1. A generic question over two named elements: the question names only the
   kind of value ("the concrete grade", "the completion date"); one passage
   gives a value for one element, another for a different element. No
   conflict. The same passages under a question that names the element do
   conflict when two documents differ for it.
2. A value's thing is read from its own clause: words a long passage holds
   far from the value, or in another step of a heading path, do not make the
   value about the asked thing.
3. One word shared with the question is not the question's thing.
4. A reply that states both values under a sentence saying the documents
   agree: the conflict statement LEADS, placed by the exit; the reply follows.

Every document, value and question here is invented for this file; the
expected values are written by hand. Each decision is shown on two inputs.
"""

from __future__ import annotations

import pytest

from app.agents import answer_contract, answer_exit
from app.agents.evidence_conflicts import _question_thing, detect, extract
from app.core.rag.inject import format_chunks_as_system_message
from app.core.rag.vector_store import Chunk

_LINE = "The documents differ on this:"


def _chunk(doc_id: str, idx: int, name: str, text: str, page: int) -> Chunk:
    c = Chunk(chunk_id=f"{doc_id}:{idx}", project_id="proj-invented", doc_id=doc_id,
              chunk_index=idx, text=text, score=0.8)
    c.source_name = name
    c.page = page
    return c


def _run(question: str, docs: list[tuple[str, str, int, str]], answer: str) -> dict:
    """One turn through the real exit, every doc handed as a pre-injected excerpt."""
    token = answer_exit._TURN.set(None)
    try:
        turn = answer_exit.open_turn()
        turn.question = question
        handed = [_chunk(d, i, n, t, p) for i, (d, n, p, t) in enumerate(docs)]
        rag = format_chunks_as_system_message(handed, total_candidates=len(handed), query=question)
        answer_exit.note_evidence(rag, [rag, {"role": "user", "content": question}], turn)
        d, n, p, _t = docs[0]
        row = {"doc_id": d, "doc_name": n, "page_or_section": f"p. {p}", "page": p, "score": 0.8,
               "confidence": "High", "layer": "own", "layer_label": "Project"}
        return answer_exit.check_end_event({"type": "end", "content": answer, "sources": [row]}, turn=turn)
    finally:
        answer_exit._TURN.reset(token)


# ── 1. a generic question over two named elements ───────────────────────────

_TWO_ELEMENTS = [
    pytest.param(
        "What is the concrete grade?",
        [("sl-1", "Brackenford Slab Schedule.pdf", 3, "Suspended slabs are concrete grade C40 throughout."),
         ("bm-2", "Brackenford Beam Schedule.pdf", 4, "Transfer beams are concrete grade C50 throughout.")],
        "The slabs are concrete grade C40 [source: Brackenford Slab Schedule.pdf].",
        "What is the concrete grade for the suspended slabs?",
        [("sl-1", "Brackenford Slab Schedule.pdf", 3, "Suspended slabs are concrete grade C40 throughout."),
         ("sp-3", "Brackenford Concrete Specification.pdf", 9, "Suspended slabs are concrete grade C35 throughout.")],
        id="grade-slab-and-beam"),
    pytest.param(
        "What is the completion date?",
        [("cd-1", "Ottermere Contract Data.pdf", 2, "Section 1 (car park) completion date: 30 June 2027."),
         ("cd-2", "Ottermere Phasing Plan.pdf", 6, "Section 2 (retail block) completion date: 14 July 2028.")],
        "The completion date is 30 June 2027 [source: Ottermere Contract Data.pdf].",
        "What is the completion date for the car park section?",
        [("cd-1", "Ottermere Contract Data.pdf", 2, "Section 1 (car park) completion date: 30 June 2027."),
         ("cd-3", "Ottermere Programme Narrative.pdf", 5, "Section 1 (car park) completion date: 22 September 2027.")],
        id="date-two-sections"),
]


@pytest.mark.parametrize("question, docs, answer, named_q, named_docs", _TWO_ELEMENTS)
def test_a_generic_question_over_two_elements_is_no_conflict(question, docs, answer, named_q, named_docs):
    assert _question_thing(question) == set()
    assert extract(answer), "the answer states a value: the test is not vacuous"
    out = _run(question, docs, answer)
    assert not out.get("conflicts")
    assert _LINE not in out["content"]


@pytest.mark.parametrize("question, docs, answer, named_q, named_docs", _TWO_ELEMENTS)
def test_the_same_values_under_a_question_naming_the_element_do_conflict(question, docs, answer, named_q, named_docs):
    assert _question_thing(named_q)
    out = _run(named_q, named_docs, answer)
    assert len(out.get("conflicts") or []) == 1, out["content"]
    assert out["content"].startswith(_LINE)


# ── 2. a value's thing is read from its own clause ──────────────────────────

_FILLER = ("The contractor shall keep the works tidy, protect finished surfaces, keep access routes "
           "clear for deliveries, record weather each day, and keep a copy of the approved drawings "
           "at the site office for inspection by the engineer at any reasonable time.")

_FAR = [
    pytest.param(
        "What cover is required to the pile cap reinforcement?",
        ("Minimum concrete cover to pile cap reinforcement is 75 mm.",
         f"Pile cap reinforcement is fixed on site. {_FILLER} " .replace(". The", ", and the")
         + "Blinding below the pile caps is laid 50 mm thick"),
        "The cover to the pile cap reinforcement is 75 mm.",
        id="far-in-a-long-passage"),
    pytest.param(
        "What cover is required to the pile cap reinforcement?",
        ("Minimum concrete cover to pile cap reinforcement is 75 mm.",
         "Pile caps > Reinforcement and cover > Blinding layer laid 50 mm thick"),
        "The cover to the pile cap reinforcement is 75 mm.",
        id="another-step-of-a-heading-path"),
]


@pytest.mark.parametrize("question, texts, answer", _FAR)
def test_words_far_from_a_value_do_not_make_it_about_the_thing(question, texts, answer):
    rows = [{"doc_name": "Fallowfield Pile Cap Details.pdf", "snippet": texts[0]},
            {"doc_name": "Fallowfield Method Statement.pdf", "snippet": texts[1]}]
    assert detect(answer, rows, question) == []


@pytest.mark.parametrize("question, texts, answer", _FAR)
def test_the_same_value_in_its_own_clause_does_conflict(question, texts, answer):
    rows = [{"doc_name": "Fallowfield Pile Cap Details.pdf", "snippet": texts[0]},
            {"doc_name": "Fallowfield Method Statement.pdf",
             "snippet": f"{_FILLER} Spacers give a cover to pile cap reinforcement of 50 mm."}]
    found = detect(answer, rows, question)
    assert len(found) == 1
    assert {v["value"] for v in found[0]["values"]} == {"75 mm", "50 mm"}


# ── 3. one word shared with the question is not its thing ───────────────────

@pytest.mark.parametrize("question, rows, answer", [
    pytest.param(
        "What cover is required to the pile cap reinforcement?",
        [{"doc_name": "Kestrel Pile Cap Details.pdf",
          "snippet": "Minimum concrete cover to pile cap reinforcement is 75 mm."},
         {"doc_name": "Kestrel Roofing Specification.pdf",
          "snippet": "Roof insulation cover boards are 50 mm."}],
        "The cover to the pile cap reinforcement is 75 mm.", id="cover"),
    pytest.param(
        "How long must the basement slab be cured after the pour?",
        [{"doc_name": "Kestrel Concrete Specification.pdf",
          "snippet": "The basement slab shall be cured for 7 days after the pour."},
         {"doc_name": "Kestrel Screed Method Statement.pdf",
          "snippet": "Screeds are cured under polythene for 3 days."}],
        "The basement slab must be cured for 7 days after the pour.", id="cured"),
])
def test_one_shared_word_is_not_the_thing_the_question_names(question, rows, answer):
    assert detect(answer, rows, question) == []


# ── 4. both values under an "agree" headline: the conflict statement leads ──

_AGREE_BOTH = [
    pytest.param(
        "What cover is required to the pile cap reinforcement?",
        [("dg-2", "PC-220 Pile Cap Details.pdf", 7,
          "Note 6. Minimum concrete cover to pile cap reinforcement is 75 mm."),
         ("ms-6", "Method Statement Pile Caps.pdf", 5,
          "Fixing. Spacers give a cover to pile cap reinforcement of 50 mm throughout.")],
        "The documents agree on the pile cap reinforcement cover: the details give 75 mm and the "
        "method statement gives 50 mm.",
        id="cover-75-50"),
    pytest.param(
        "What is the yield strength of the pump station reinforcement?",
        [("dw-1", "Pump Station Reinforcement Drawing.pdf", 2,
          "Pump station reinforcement: high yield bars, yield strength 420 N/mm2."),
         ("bq-4", "Pump Station Bill of Quantities.pdf", 11,
          "Pump station reinforcement high yield bars, yield strength 500 N/mm2, supplied cut and bent.")],
        "Both sources are consistent: the drawing gives 420 N/mm2 for the pump station reinforcement "
        "and the bill gives 500 N/mm2.",
        id="strength-420-500"),
]


@pytest.mark.parametrize("question, docs, answer", _AGREE_BOTH)
def test_both_values_under_an_agree_headline_get_the_leading_conflict_statement(question, docs, answer):
    out = _run(question, docs, answer)
    assert len(out.get("conflicts") or []) == 1, out["content"]
    first, _, rest = out["content"].partition("\n")
    assert first.startswith(_LINE)
    # The statement names both values; the model's sentence asserting that
    # the documents agree about them is replaced by it (advisor 18:31Z).
    for _d, name, _p, text in docs:
        assert name.rsplit(".", 1)[0] in first
    assert "agree" not in out["content"] and "consistent" not in out["content"]


@pytest.mark.parametrize("question, docs, answer", _AGREE_BOTH)
def test_both_values_stated_as_differing_are_left_as_written(question, docs, answer):
    told = answer.split(":", 1)[1].strip()
    told = told[0].upper() + told[1:] + " No document says which governs."
    out = _run(question, docs, told)
    assert not out.get("conflicts")
    assert _LINE not in out["content"]


@pytest.mark.parametrize("question, docs, answer", _AGREE_BOTH)
def test_contract_alone_places_the_statement_first(question, docs, answer):
    rows = [{"doc_name": n, "snippet": t, "doc_id": d} for d, n, _p, t in docs]
    out = answer_contract.apply(answer, sources=rows, question=question)
    assert out.text.startswith(_LINE)


# ── 5. a loosely worded second anchor never shrinks the thing ───────────────

@pytest.mark.parametrize("question, rows, answer", [
    pytest.param(
        "What cover is required to the pile cap reinforcement?",
        [{"doc_name": "Wrenfield Pile Cap Details.pdf",
          "snippet": "Minimum concrete cover to pile cap reinforcement is 75 mm."},
         {"doc_name": "Wrenfield Pile Notes.pdf", "snippet": "Pile cover: 75 mm nominal."},
         {"doc_name": "Wrenfield Pile Shaft Schedule.pdf", "snippet": "Pile shaft cover 50 mm."}],
        "The cover to the pile cap reinforcement is 75 mm.", id="pile-cap-cover"),
    pytest.param(
        "How long must the basement slab concrete be cured?",
        [{"doc_name": "Wrenfield Concrete Specification.pdf",
          "snippet": "Basement slab concrete shall be cured for 7 days."},
         {"doc_name": "Wrenfield Site Notes.pdf", "snippet": "Concrete cured 7 days."},
         {"doc_name": "Wrenfield Precast Schedule.pdf", "snippet": "Precast concrete cured 3 days."}],
        "The basement slab concrete must be cured for 7 days.", id="basement-slab-curing"),
])
def test_a_loosely_worded_anchor_does_not_shrink_the_thing(question, rows, answer):
    """The thing is named by the anchor naming the most asked words; a second
    anchor that names fewer never lets a value for another thing in."""
    assert detect(answer, rows, question) == []
    # Without the loose anchor's narrower words the other thing is out as well,
    # and a real second value for the asked thing is still a conflict.
    real = rows[:1] + [dict(rows[0], doc_name="Wrenfield Method Statement.pdf",
                            snippet=rows[0]["snippet"].replace(" 75 mm", " 60 mm").replace(" 7 days", " 5 days"))]
    assert len(detect(answer, real, question)) == 1


# ── 6. when even the best anchor shares one word with the question ──────────

@pytest.mark.parametrize("question, rows, answer", [
    pytest.param(
        "What cover is required to the pile cap reinforcement?",
        [{"doc_name": "Thornbury Wall Details.pdf", "snippet": "Wall cover is 75 mm."},
         {"doc_name": "Thornbury Slab Details.pdf", "snippet": "Slab cover is 50 mm."}],
        "The cover is 75 mm.", id="cover"),
    pytest.param(
        "How long must the basement slab concrete be cured after the pour?",
        [{"doc_name": "Thornbury Screed Notes.pdf", "snippet": "Screed is cured for 7 days."},
         {"doc_name": "Thornbury Render Notes.pdf", "snippet": "Render is cured for 3 days."}],
        "It is cured for 7 days.", id="cured"),
])
def test_a_thing_sharing_one_word_with_the_question_is_no_conflict(question, rows, answer):
    """Two values that share only one of the question's thing words with it
    are not shown to be about the asked thing."""
    assert len(_question_thing(question)) >= 2
    assert detect(answer, rows, question) == []
    # The same two passages naming the asked thing do conflict.
    named = [dict(r, snippet=r["snippet"].replace("Wall", "Pile cap reinforcement").replace(
        "Slab", "Pile cap reinforcement").replace("Screed", "Basement slab concrete").replace(
        "Render", "Basement slab concrete")) for r in rows]
    assert len(detect(answer, named, question)) == 1
