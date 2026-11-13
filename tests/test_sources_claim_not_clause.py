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

import pytest

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


_GROUNDED_WITH_CAVEAT = (
    "Delay damages are 0.07% of the Accepted Contract Amount per day, capped "
    "at 8% of the Accepted Contract Amount (Sub-Clause 8.8). The excerpt does "
    "not state the Accepted Contract Amount itself, so I cannot convert the "
    "percentage into a money figure."
)
_PURE_DECLINE = (
    "I cannot find the delay damages rate in the project documents, so I "
    "cannot state it in money."
)


@pytest.mark.parametrize(
    "answer,expected_docs",
    [
        (_GROUNDED_WITH_CAVEAT, {_DOC_ID}),
        (_PURE_DECLINE, set()),
    ],
    ids=["grounded-plus-caveat", "pure-decline"],
)
def test_sources_follow_the_answers_claim_not_a_caveat_clause(answer, expected_docs):
    """Both answers carry a caveat phrase ("i cannot"). Only the one whose
    provenance credits a figure to the injected project chunk keeps sources."""
    sources = _finish_turn(answer)
    assert {s["doc_id"] for s in sources} == expected_docs
    if expected_docs:
        assert {s.get("source_class") for s in sources} == {"project_corpus"}


@pytest.mark.parametrize(
    "credited_doc,expected_docs",
    [
        (_DOC_ID, {_DOC_ID}),
        ("d_elsewhere", set()),
    ],
    ids=["credit-in-injected-set", "credit-outside-injected-set"],
)
def test_only_a_credit_to_this_turns_injected_set_lifts_a_caveat(credited_doc, expected_docs):
    """A project_document credit counts only when it points at a document
    injected this turn; a credit for some other document does not make a
    caveat answer grounded here."""
    audit = _audit()
    audit["provenance"] = [{
        "figure": "0.07%", "source": "project_document", "doc_id": credited_doc,
    }]
    answer = "Delay damages are 0.07% per day; I cannot confirm the money amount."
    sources = _build_sources_from_audit(audit, answer, [])
    assert {s["doc_id"] for s in sources} == expected_docs
