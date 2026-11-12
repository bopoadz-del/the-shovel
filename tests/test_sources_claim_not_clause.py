"""S-R2: the sources builder classifies the answer's claim, not any clause in it.

A grounded answer that states a figure from an injected project chunk and then
adds a limitation ("... so I cannot convert the percentage into a money
figure") used to lose every source row, because the refusal test is a
substring check over the whole answer. The turn's own figure provenance now
decides: when it credits a figure to an injected project document, the answer
is not a decline. A pure decline states no project figure and still records
no sources.

Each test builds its own audit record and reference context and runs the real
provenance recorder; synthetic documents and figures only.
"""
from __future__ import annotations

from app.agents.runtime import _build_sources_from_audit, _record_figure_provenance

_PROJECT_ID = "7c1d0b2e"
_DOC_ID = "d_cond"
_DOC_NAME = "particular_conditions_synthetic.pdf"
_CLAUSE = (
    "Sub-Clause 8.8 Delay Damages shall be 0.07% of the Accepted Contract "
    "Amount per day, subject to a maximum of 8% of the Accepted Contract Amount."
)
_ASK = "What is the delay damages rate and cap, in money?"


def _audit() -> dict:
    return {
        "project_id": _PROJECT_ID,
        "user_message_preview": _ASK,
        "chunks": [{
            "doc_id": _DOC_ID,
            "layer": "own",
            "source_name": _DOC_NAME,
            "text": _CLAUSE,
            "score": 0.81,
            "page": 12,
            "chunk_index": 3,
            "chunk_id": "c3",
        }],
    }


def _context() -> dict:
    return {
        "role": "system",
        "content": (
            "REFERENCE CONTEXT\n"
            f"[doc_id={_DOC_ID} chunk=3 score=0.810 class=project_corpus "
            f"layer=own page=12 src={_DOC_NAME}] {_CLAUSE}"
        ),
    }


def _finish_turn(answer: str) -> list[dict]:
    """Run the turn's provenance recorder, then the sources builder, as the
    streaming path does (postprocess first, sources on the end event)."""
    audit = _audit()
    _record_figure_provenance(answer, _context(), [{"role": "user", "content": _ASK}], audit)
    return _build_sources_from_audit(audit, answer, [])


def test_grounded_answer_with_a_limitation_keeps_its_sources():
    answer = (
        "Delay damages are 0.07% of the Accepted Contract Amount per day, capped "
        "at 8% of the Accepted Contract Amount (Sub-Clause 8.8). The excerpt does "
        "not state the Accepted Contract Amount itself, so I cannot convert the "
        "percentage into a money figure."
    )
    sources = _finish_turn(answer)
    assert sources, "a grounded answer with a caveat clause must keep its sources"
    assert {s["doc_id"] for s in sources} == {_DOC_ID}
    assert {s.get("source_class") for s in sources} == {"project_corpus"}


def test_pure_decline_still_records_no_sources():
    answer = (
        "I cannot find the delay damages rate in the project documents, so I "
        "cannot state it in money."
    )
    assert _finish_turn(answer) == []


def test_project_credit_for_a_document_not_injected_does_not_lift_a_decline():
    """The credit must point at this turn's injected set: a provenance entry
    for some other document does not make a caveat answer grounded here."""
    audit = _audit()
    audit["provenance"] = [{
        "figure": "0.07%", "source": "project_document", "doc_id": "d_elsewhere",
    }]
    answer = "I cannot confirm the delay damages rate from these excerpts."
    assert _build_sources_from_audit(audit, answer, []) == []
