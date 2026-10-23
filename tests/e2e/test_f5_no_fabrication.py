"""F5 — a figure that is not in the corpus BLOCKS instead of being invented.

The Fork's core safety contract. On a construction project a fabricated rate
goes into a payment certificate and a fabricated cover figure goes into a
slab. "I don't have that" is always a better answer than a plausible one.

This drives the real gate over the real retrieval layer: documents are
ingested, a question is asked, and an answer carrying a figure that does NOT
trace to what came back is refused — with a stated reason and a stated
remedy, not a bare refusal.

The companion assertion matters just as much: a figure that IS grounded must
pass untouched. A gate that blocks everything is not safe, it is broken, and
it would be abandoned within a week of a pilot.
"""
from __future__ import annotations

import pytest

from app.core.rag import vector_store as vs
from app.core.rag.inject import format_chunks_as_system_message
from app.agents.runtime import _cost_grounding_gate, _CG_REFUSAL


PRICED_BOQ = (
    "BILL OF QUANTITIES - SECTION 3. Item 3.14 Supply and place grade C40 "
    "concrete to suspended slabs, measured net. Unit m3. Rate 385.00 SAR/m3. "
    "Quantity 1,240 m3."
)
UNPRICED_SPEC = (
    "SECTION 03300 CAST-IN-PLACE CONCRETE. Concrete for suspended slabs "
    "shall be grade C40/50. Nominal cover to reinforcement shall be 25 mm. "
    "No rates or prices are stated in this specification."
)


@pytest.fixture
def corpus(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")
    monkeypatch.setenv("COST_GROUNDING_GATE", "1")
    vs.reset_store_cache()

    from app.core.rag.embeddings import get_embedder, reset_embedder_cache
    reset_embedder_cache()
    emb = get_embedder()
    store = vs.get_store(dim=emb.dim)

    def ingest(project_id, doc_id, text):
        store.upsert_chunks(project_id=project_id, doc_id=doc_id,
                            chunks=[text], embeddings=emb.encode([text]))

    # One project holds a PRICED BOQ, the other only an unpriced spec.
    ingest("priced_project", "boq-section-3.xlsx", PRICED_BOQ)
    ingest("unpriced_project", "spec-03300.pdf", UNPRICED_SPEC)
    return store


def _context_for(project_id, question):
    from app.core.rag.retriever import retrieve_with_filter
    chunks, _ = retrieve_with_filter(question, project_id, k=5)
    return format_chunks_as_system_message(chunks, total_candidates=len(chunks))


def test_a_rate_that_is_not_in_the_corpus_is_refused(corpus):
    """The load-bearing one. The project has a specification and no prices;
    an answer that names a rate anyway must not reach the client."""
    rag = _context_for("unpriced_project", "what is the rate for C40 concrete")
    answer = "The rate for grade C40 concrete to suspended slabs is 420.00 SAR/m3."

    result = _cost_grounding_gate(answer, rag, messages=[])

    assert result == _CG_REFUSAL, (
        "an ungrounded rate passed the gate — this is the fabrication the "
        f"platform exists to prevent.\ngate returned: {result[:200]}"
    )


def test_the_refusal_says_why_and_what_to_do(corpus):
    """A blocked answer that does not say what would unblock it just reads
    as the tool being broken."""
    rag = _context_for("unpriced_project", "what is the rate for C40 concrete")
    result = _cost_grounding_gate("Use 420.00 SAR/m3.", rag, messages=[])

    lowered = result.lower()
    assert "don't have a rate" in lowered or "not have a rate" in lowered, \
        "the refusal does not state the reason"
    assert "boq" in lowered or "quote" in lowered, \
        "the refusal does not state the remedy"


def test_a_rate_that_IS_in_the_corpus_passes_untouched(corpus):
    """The other half of the contract. A gate that refuses everything is
    not safe, it is useless — and it gets switched off."""
    rag = _context_for("priced_project", "what is the rate for C40 concrete")
    answer = "Per BOQ item 3.14 the rate for grade C40 concrete is 385.00 SAR/m3."

    result = _cost_grounding_gate(answer, rag, messages=[])

    assert result == answer, (
        "a rate that IS in the priced BOQ was refused — the gate is blocking "
        f"grounded answers.\nreturned: {result[:200]}"
    )


def test_an_answer_with_no_figures_is_never_touched(corpus):
    """Only cost answers are gated. Prose must pass through unchanged."""
    rag = _context_for("unpriced_project", "what grade of concrete is specified")
    answer = "The specification calls for grade C40/50 concrete to suspended slabs."

    assert _cost_grounding_gate(answer, rag, messages=[]) == answer


def test_a_rate_invented_with_no_context_at_all_is_refused(corpus):
    """The worst case: nothing was retrieved, and the model answered anyway.
    Missing context must read as DENY, not as 'nothing to contradict me'."""
    result = _cost_grounding_gate(
        "The rate is 420.00 SAR/m3.", rag_sys_msg=None, messages=[]
    )
    assert result == _CG_REFUSAL, "no context at all still let a rate through"


def test_an_empty_context_message_is_also_denied(corpus):
    """The empty-retrieval shape `format_chunks_as_system_message` really
    returns, not just None."""
    empty = format_chunks_as_system_message([], total_candidates=0)
    assert empty["content"] == ""

    result = _cost_grounding_gate("The rate is 420.00 SAR/m3.", empty, messages=[])
    assert result == _CG_REFUSAL


def test_a_rate_from_a_DIFFERENT_project_does_not_ground_this_one(corpus):
    """Isolation and grounding meet here: the priced project's 385.00 SAR/m3
    must not ground an answer given in the unpriced project."""
    rag = _context_for("unpriced_project", "rate for C40 concrete")
    result = _cost_grounding_gate(
        "The rate is 385.00 SAR/m3.", rag, messages=[]
    )
    assert result == _CG_REFUSAL, (
        "another project's rate grounded this project's answer"
    )
