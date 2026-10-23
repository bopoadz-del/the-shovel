"""Natural questions must find the same chunks their terse phrasings find.

Live-fire campaign phase 3, F18. Measured against a real 203-page contract
and a 129-page tender indexed on the live service: the needle chunk ranks
FIRST (0.933) for a query overlapping the answer's wording, and falls out
of the top-12 when the same fact is asked as a natural question. The
interrogative scaffolding drags the query vector away from the document's
declarative prose.

Two probe campaigns pinned the mechanism:

  * stopword distillation FAILS -- "Conditions Contract days Time
    Completion whole Works" breaks the noun phrases and ranks nowhere;
  * wrapper stripping that keeps phrases contiguous WORKS -- "days is the
    Time for Completion for the whole of the Works" took the 852-day
    needle from outside the top-12 to rank 1 (0.933), and the tender
    variant took its needle to rank 4 (0.588).

The fix is dual-query retrieval: search with BOTH phrasings, merge by max
score per chunk. This fence pins the transform to the exact strings the
live probes validated, and the merge machinery around it.
"""
from __future__ import annotations

import pytest

ACTIVE = "p_contract"

NATURAL = (
    "In the Conditions of Contract, how many days is the Time for "
    "Completion for the whole of the Works?"
)
STRIPPED = "days is the Time for Completion for the whole of the Works"

TENDER_NATURAL = (
    "Which three consultancy firms merged to form the tenderer, and in "
    "which year was each founded?"
)
TENDER_STRIPPED = (
    "three consultancy firms merged to form the tenderer, and in year "
    "was each founded"
)

NEEDLE_TEXT = (
    "The Time for Completion for the whole of the Works shall be 852 days "
    "calculated from the Commencement Date as defined in Sub-Clause 8.1."
)
BOILER_TEXT = (
    "The Contractor shall comply with all applicable laws and shall give "
    "all notices required by the Employer's Requirements."
)


def _chunk(chunk_id, project_id, score, text):
    from app.core.rag.vector_store import Chunk
    return Chunk(chunk_id=chunk_id, project_id=project_id,
                 doc_id=f"doc-{chunk_id}", chunk_index=0,
                 text=text, score=score)


def _install(monkeypatch, results_by_query_text, calls=None):
    """Fake store whose candidates depend on the QUERY TEXT it was asked
    with -- the exact discriminator dual-query retrieval introduces."""
    from app.core.rag import retriever as ret

    def fake_search(self, project_id, qvec, k, query_text=None):
        if calls is not None:
            calls.append(query_text)
        return [
            _chunk(cid, project_id, score, text)
            for cid, score, text in results_by_query_text.get(query_text, [])
        ]

    monkeypatch.setattr("app.core.rag.vector_store.VectorStore.search", fake_search)
    monkeypatch.setattr(ret, "_doc_name_for_id", lambda _id: "contract.pdf",
                        raising=False)
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")
    monkeypatch.delenv("MASTER_CORPUS_SOURCE_PROJECT_ID", raising=False)
    monkeypatch.delenv("RAG_DUAL_QUERY", raising=False)
    # This file measures dual-query merge, not the A3 TfC value boost.
    monkeypatch.setattr(
        ret, "_apply_asked_particular_value_boost", lambda *_a, **_k: None,
    )
    return ret


# -- the transform, pinned to the strings the live probes validated ------

def test_contract_question_strips_to_the_probe_winning_string():
    from app.core.rag.retriever import _strip_question_wrapper
    assert _strip_question_wrapper(NATURAL) == STRIPPED


def test_tender_question_strips_with_phrases_intact():
    from app.core.rag.retriever import _strip_question_wrapper
    assert _strip_question_wrapper(TENDER_NATURAL) == TENDER_STRIPPED


def test_a_terse_query_yields_no_variant():
    """No wrapper to strip -> None -> no second search is ever attempted."""
    from app.core.rag.retriever import _strip_question_wrapper
    assert _strip_question_wrapper("Time for Completion days") is None
    assert _strip_question_wrapper("concrete grade C40 raft") is None


def test_a_stripped_stub_too_short_to_search_yields_none():
    from app.core.rag.retriever import _strip_question_wrapper
    assert _strip_question_wrapper("What is it?") is None
    assert _strip_question_wrapper("") is None


def test_content_phrases_are_never_broken():
    """The negative result that killed the stopword route: 'the', 'is',
    'for', 'of' are phrase glue and must survive the strip."""
    from app.core.rag.retriever import _strip_question_wrapper
    out = _strip_question_wrapper(NATURAL)
    assert "the Time for Completion for the whole of the Works" in out


# -- the retrieval machinery ---------------------------------------------

def test_needle_reachable_only_via_the_variant_enters_topk(monkeypatch):
    """RED before the fix: only NATURAL is searched, the needle never
    enters the pool, and the answer grounds on boilerplate."""
    ret = _install(monkeypatch, {
        NATURAL: [("boiler1", 0.55, BOILER_TEXT), ("boiler2", 0.52, BOILER_TEXT)],
        STRIPPED: [("needle", 0.933, NEEDLE_TEXT)],
    })
    chunks, _ = ret.retrieve_with_filter(NATURAL, ACTIVE, k=5)
    ids = [c.chunk_id for c in chunks]
    assert "needle" in ids, f"variant leg never ran: {ids}"
    assert ids[0] == "needle", f"needle must outrank boilerplate: {ids}"


def test_a_chunk_found_by_both_legs_keeps_its_best_score(monkeypatch):
    ret = _install(monkeypatch, {
        NATURAL: [("shared", 0.40, NEEDLE_TEXT)],
        STRIPPED: [("shared", 0.90, NEEDLE_TEXT)],
    })
    chunks, _ = ret.retrieve_with_filter(NATURAL, ACTIVE, k=5)
    assert len(chunks) == 1
    assert chunks[0].score == pytest.approx(0.90)


def test_kill_switch_disables_the_second_leg(monkeypatch):
    calls = []
    ret = _install(monkeypatch, {
        NATURAL: [("boiler1", 0.55, BOILER_TEXT)],
        STRIPPED: [("needle", 0.933, NEEDLE_TEXT)],
    }, calls=calls)
    monkeypatch.setenv("RAG_DUAL_QUERY", "0")
    chunks, _ = ret.retrieve_with_filter(NATURAL, ACTIVE, k=5)
    assert NATURAL in calls
    assert STRIPPED not in calls
    assert [c.chunk_id for c in chunks] == ["boiler1"]


def test_terse_queries_cost_no_second_search(monkeypatch):
    calls = []
    terse = "concrete grade C40 raft"
    ret = _install(monkeypatch, {
        terse: [("needle", 0.933, NEEDLE_TEXT)],
    }, calls=calls)
    chunks, _ = ret.retrieve_with_filter(terse, ACTIVE, k=5)
    assert calls == [terse]
    assert [c.chunk_id for c in chunks] == ["needle"]


def test_alt_leg_failure_never_breaks_the_primary(monkeypatch):
    from app.core.rag import retriever as ret

    def fake_search(self, project_id, qvec, k, query_text=None):
        if query_text == STRIPPED:
            raise RuntimeError("alt leg store error")
        return [_chunk("boiler1", project_id, 0.55, BOILER_TEXT)]

    monkeypatch.setattr("app.core.rag.vector_store.VectorStore.search", fake_search)
    monkeypatch.setattr(ret, "_doc_name_for_id", lambda _id: "contract.pdf",
                        raising=False)
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")
    monkeypatch.delenv("MASTER_CORPUS_SOURCE_PROJECT_ID", raising=False)
    chunks, _ = ret.retrieve_with_filter(NATURAL, ACTIVE, k=5)
    assert [c.chunk_id for c in chunks] == ["boiler1"]
