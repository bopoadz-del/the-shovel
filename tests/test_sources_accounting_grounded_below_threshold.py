"""S1 sources accounting: a turn answered out of a project document records
project_corpus, even when retrieval fired the confidence/identifier trust gate.

Battery Q5 ("What performance bond or retention does the contract require?")
answered the Harbourline contract verbatim -- "Clause 4.2 ... 10% ... Clause 14.3
... 5%" -- yet the Sources panel was empty, because the chunk came back below the
retrieval threshold and _build_sources_from_audit dropped every chunk when the
trust gate fired. The chunk the answer QUOTES is the evidence it stands on; it is
recorded. A paraphrase or a decline (no verbatim quote) still records nothing, so
a genuinely ungrounded low-confidence turn cannot fabricate a sources panel.

Synthetic audit records only.
"""
from __future__ import annotations

import pytest

from app.agents.runtime import _build_sources_from_audit

# A real (non-master) project id is a hex slug.
_PROJECT_ID = "4e9480af"
_CLAUSE = (
    "Clause 4.2 Performance Security shall be ten percent of the Accepted "
    "Contract Amount and Clause 14.3 Retention shall be five percent of each "
    "Interim Payment Certificate released on the Taking-Over Certificate"
)


def _audit(threshold_fired: bool) -> dict:
    return {
        "project_id": _PROJECT_ID,
        "user_message_preview": (
            "What performance bond or retention does the contract require?"
        ),
        "threshold_fired": threshold_fired,
        "chunks": [{
            "doc_id": "d_con",
            "layer": "own",
            "source_name": "harbourline_contract.pdf",
            "text": _CLAUSE,
            "score": 0.33,
            "page": 1,
            "chunk_index": 0,
            "chunk_id": "c0",
        }],
    }


@pytest.mark.parametrize("threshold_fired", [True, False])
def test_a_grounded_project_answer_records_project_corpus(threshold_fired):
    """The answer quotes the contract clause verbatim, so the own-layer chunk is
    the evidence -- project_corpus is recorded whether or not the trust gate
    fired (the below-threshold case is the Q5 regression this fixes)."""
    answer = f"The contract requires the following. {_CLAUSE}."
    sources = _build_sources_from_audit(_audit(threshold_fired), answer, [])
    classes = {s.get("source_class") for s in sources}
    assert "project_corpus" in classes, sources


@pytest.mark.parametrize(
    "answer",
    [
        "I don't have that in the project document sections.",
        "The bond and retention percentages are not something I can state here.",
    ],
)
def test_an_ungrounded_below_threshold_turn_records_no_source(answer):
    """Trust gate fired and the answer does NOT quote the chunk (a decline or a
    paraphrase) -> no fabricated sources panel (the pre-existing guard is
    preserved)."""
    sources = _build_sources_from_audit(_audit(True), answer, [])
    assert sources == [], sources


_OTHER_CLAUSE = (
    "Clause 9.1 the Contractor shall carry out the Works in accordance with the "
    "Programme and the method statements approved by the Engineer before start"
)


def test_only_the_quoted_chunk_is_recorded_when_the_gate_fired():
    """Two below-threshold own-layer chunks reach the audit; the answer quotes
    only the first. The Sources panel records ONLY the quoted chunk -- the
    unquoted one is not fabricated into the panel. (This also kills the
    ``chunks = grounded`` filter: dropping it would wrongly list d_other.)"""
    audit = _audit(True)
    audit["chunks"].append({
        "doc_id": "d_other",
        "layer": "own",
        "source_name": "harbourline_programme.pdf",
        "text": _OTHER_CLAUSE,
        "score": 0.31,
        "page": 2,
        "chunk_index": 0,
        "chunk_id": "c9",
    })
    answer = f"The contract requires the following. {_CLAUSE}."
    sources = _build_sources_from_audit(audit, answer, [])
    doc_ids = {s.get("doc_id") for s in sources}
    assert doc_ids == {"d_con"}, sources
