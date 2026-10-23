"""A3 cross-contract contamination fence.

Wave-1 FAIL: a AB-2023-101 (Infrastructure Package 1) contract question
cited AB-2022 in Sources / prose. Retrieval scores the whole project;
identifier matching is token-soup on chunk text ('ab' + '2023' + '101'
can appear in a AB-2022 Conditions of Contract as a prefix, a date, and
a clause number); src= truncation kept the filename tail and dropped
the PREFIX-YEAR-SEQ.

This fence is fail-closed: a named contract/doc id keeps only that id's
files. Wrong-contract is its own defect class — a AB-2023 fixture must
not return AB-2022 chunks or cites.
"""
from __future__ import annotations

import pytest

from app.core.rag.retriever import (
    extract_contract_doc_ids,
    filename_matches_named_contracts,
)
from app.core.rag.vector_store import Chunk


DD23_NAME = (
    "AB-2023-101_the client project II Infrastructure Package 1_"
    "Vol 1 - Conditions of Contract.pdf"
)
DD22_NAME = "AB-2022-202 - Volume 1 - Conditions of Contract.pdf"

# Shared FIDIC-shaped prose so semantic + token-soup would mix years
# without the filename fence.
_COC_PROSE = (
    "Conditions of Contract. Time for Completion for the whole of the "
    "Works is 365 calendar days. Delay Damages are 0.1 percent of the "
    "Contract Price per day. Clause 101 records the notice period. "
    "Programme dated 15 March 2023. The AB prefix appears on every "
    "tender drawing title block."
)

DD23_QUERY = (
    "Per the AB-2023-101 Infrastructure Package 1 executed contract, "
    "what is the Time for Completion for the whole of the Works?"
)


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")
    from app.core.rag import embeddings as _emb, vector_store as _vs
    _emb.reset_embedder_cache()
    _vs.reset_store_cache()
    from app.core.rag.embeddings import Embedder
    from app.core.rag.vector_store import get_store
    from sqlalchemy import delete as _sa_delete
    e = Embedder(model_name="fake")
    store = get_store(dim=e.dim)
    with store._lock:
        with store._session_factory()() as session:
            session.execute(_sa_delete(store._rag_chunk_cls))
            session.commit()
    yield store, e
    _emb.reset_embedder_cache()
    _vs.reset_store_cache()


def test_extract_contract_doc_ids_from_query_and_underscore_filename():
    assert extract_contract_doc_ids(DD23_QUERY) == ["ab-2023-101"]
    assert extract_contract_doc_ids(DD23_NAME) == ["ab-2023-101"]
    assert extract_contract_doc_ids(DD22_NAME) == ["ab-2022-202"]
    assert extract_contract_doc_ids("What are the Delay Damages?") == []
    # Drawing codes are not PREFIX-YEAR-SEQ contract ids.
    assert extract_contract_doc_ids(
        "QZ-ENG-320-0000-EXC-DWG-LI-200-0001056-04"
    ) == []


def test_filename_match_rejects_other_year_and_token_soup():
    named = ["ab-2023-101"]
    assert filename_matches_named_contracts(DD23_NAME, named) is True
    assert filename_matches_named_contracts(DD22_NAME, named) is False
    # Token soup in a AB-2022 chunk is not a match when the filename is known.
    assert filename_matches_named_contracts(
        DD22_NAME, named, chunk_text=_COC_PROSE,
    ) is False
    # Unresolved filename: contiguous id in text only, never soup.
    assert filename_matches_named_contracts(
        "", named, chunk_text="see AB-2023-101 Vol 1 clause 1.1.75",
    ) is True
    assert filename_matches_named_contracts(
        "", named, chunk_text=_COC_PROSE,
    ) is False


def _index_both_years(store, embedder):
    store.upsert_chunks(
        "proj_a", "doc_dd23", [_COC_PROSE], embedder.encode([_COC_PROSE]),
    )
    store.upsert_chunks(
        "proj_a", "doc_dd22", [_COC_PROSE + " demolition package"],
        embedder.encode([_COC_PROSE + " demolition package"]),
    )


def _name_map(doc_id: str) -> str:
    return {"doc_dd23": DD23_NAME, "doc_dd22": DD22_NAME}.get(doc_id, "")


def test_ab2023_query_does_not_return_ab2022_chunks(isolated_store, monkeypatch):
    """Fail-closed: named AB-2023-101 must not surface AB-2022 files."""
    from app.core.rag import retriever as ret

    store, e = isolated_store
    _index_both_years(store, e)
    monkeypatch.setattr(ret, "_doc_name_for_id", _name_map)

    # Force both years into the semantic + identifier pool at equal score
    # so the fence, not ranking, is what excludes AB-2022.
    real_search = store.search

    def both_hot(project_id, query_vec, k=20, query_text=None):
        out = real_search(project_id, query_vec, k=k, query_text=query_text)
        for c in out:
            c.score = 0.82
        return out

    monkeypatch.setattr(store, "search", both_hot)

    chunks, _ = ret.retrieve_with_filter(DD23_QUERY, "proj_a", k=5)
    names = [_name_map(c.doc_id) for c in chunks]
    assert chunks, "AB-2023-101 is in the fixture — empty is a miss, not mix"
    assert all("AB-2023-101" in n for n in names), names
    assert not any("AB-2022" in n for n in names), names
    blob = " ".join((c.text or "") + " " + n for c, n in zip(chunks, names))
    assert "AB-2022" not in blob
    assert "AB-2022-202" not in blob


def test_unknown_contract_id_is_empty_not_other_year(isolated_store, monkeypatch):
    """Fail closed: a named id that is not in the store must not fall back."""
    from app.core.rag import retriever as ret

    store, e = isolated_store
    _index_both_years(store, e)
    monkeypatch.setattr(ret, "_doc_name_for_id", _name_map)

    chunks, _ = ret.retrieve_with_filter(
        "What does AB-2024-999 say about Delay Damages?",
        "proj_a",
        k=5,
    )
    names = [_name_map(c.doc_id) for c in chunks]
    assert chunks == []
    assert not any("AB-2022" in n or "AB-2023" in n for n in names)


def test_unnamed_query_does_not_mix_contract_years(isolated_store, monkeypatch):
    from app.core.rag import retriever as ret

    store, e = isolated_store
    _index_both_years(store, e)
    monkeypatch.setattr(ret, "_doc_name_for_id", _name_map)

    real_search = store.search

    def both_hot(project_id, query_vec, k=20, query_text=None):
        out = real_search(project_id, query_vec, k=k, query_text=query_text)
        for c in out:
            c.score = 0.9 if c.doc_id == "doc_dd23" else 0.89
        return out

    monkeypatch.setattr(store, "search", both_hot)

    chunks, _ = ret.retrieve_with_filter(
        "What is the Time for Completion for the whole of the Works?",
        "proj_a",
        k=5,
    )
    years = {extract_contract_doc_ids(_name_map(c.doc_id))[0][:7] for c in chunks
             if extract_contract_doc_ids(_name_map(c.doc_id))}
    # PREFIX-YEAR only — one year in the result set.
    assert len(years) <= 1, years


def test_rag_inject_drops_wrong_year_and_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")
    from app.core.rag.inject import rag_inject

    wrong = Chunk(
        chunk_id="c22", project_id="proj_a", doc_id="doc_dd22",
        chunk_index=0, text=_COC_PROSE, score=0.91,
    )
    wrong.source_name = DD22_NAME

    def only_wrong(query, project_id, k=5, intent=None, **kwargs):
        return [wrong], 0

    monkeypatch.setattr("app.core.rag.inject.retrieve_with_filter", only_wrong)

    msg, audit = rag_inject(
        user_message=DD23_QUERY,
        project_id="proj_a",
        conversation_id="ws-proj_a-1",
        user_id="u1",
        agent_name="project-assistant",
    )
    assert msg is None
    assert audit["identifier_miss"] is True
    assert audit["threshold_fired"] is True
    assert "ab-2023-101" in audit["extracted_contract_ids"]


def test_rag_inject_keeps_named_year_and_names_it(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")
    from app.core.rag.inject import rag_inject

    right = Chunk(
        chunk_id="c23", project_id="proj_a", doc_id="doc_dd23",
        chunk_index=12, text=_COC_PROSE, score=0.88,
    )
    right.source_name = DD23_NAME
    wrong = Chunk(
        chunk_id="c22", project_id="proj_a", doc_id="doc_dd22",
        chunk_index=4, text=_COC_PROSE, score=0.90,
    )
    wrong.source_name = DD22_NAME

    def mixed(query, project_id, k=5, intent=None, **kwargs):
        return [wrong, right], 0

    monkeypatch.setattr("app.core.rag.inject.retrieve_with_filter", mixed)

    msg, audit = rag_inject(
        user_message=DD23_QUERY,
        project_id="proj_a",
        conversation_id="ws-proj_a-1",
        user_id="u1",
        agent_name="project-assistant",
    )
    assert msg is not None
    assert "AB-2023-101" in msg["content"]
    assert "AB-2022" not in msg["content"]
    assert "CONTRACT ATTRIBUTION" in msg["content"]
    assert audit.get("identifier_miss") is not True
    assert [c["doc_id"] for c in audit["chunks"]] == ["doc_dd23"]


def test_sources_panel_drops_wrong_year(monkeypatch):
    from app.agents.runtime import _build_sources_from_audit

    def fake_doc(doc_id):
        return {"original_name": DD23_NAME if doc_id == "d23" else DD22_NAME}

    monkeypatch.setattr(
        "app.core.projects.get_document", fake_doc, raising=False,
    )
    # runtime imports projects inside the function; patch the module used there.
    import app.core.projects as projects_mod
    monkeypatch.setattr(projects_mod, "get_document", fake_doc)

    audit = {
        "project_id": "proj_a",
        "user_message_preview": DD23_QUERY,
        "extracted_contract_ids": ["ab-2023-101"],
        "identifier_miss": False,
        "threshold_fired": False,
        "chunks": [
            {"doc_id": "d22", "chunk_index": 4, "chunk_id": "c22",
             "score": 0.9, "layer": "own"},
            {"doc_id": "d23", "chunk_index": 12, "chunk_id": "c23",
             "score": 0.8, "layer": "own"},
        ],
    }
    out = _build_sources_from_audit(audit, "Time for Completion is 365 days.")
    names = [s["doc_name"] for s in out]
    assert names, "named contract is in the audit — empty here is a miss"
    assert all("AB-2023-101" in n for n in names), names
    assert not any("AB-2022" in n for n in names), names


def test_sources_fail_closed_when_only_wrong_year(monkeypatch):
    from app.agents.runtime import _build_sources_from_audit
    import app.core.projects as projects_mod

    monkeypatch.setattr(
        projects_mod, "get_document",
        lambda did: {"original_name": DD22_NAME},
    )
    audit = {
        "project_id": "proj_a",
        "user_message_preview": DD23_QUERY,
        "extracted_contract_ids": ["ab-2023-101"],
        "identifier_miss": False,
        "threshold_fired": False,
        "chunks": [
            {"doc_id": "d22", "chunk_index": 4, "chunk_id": "c22",
             "score": 0.9, "layer": "own"},
        ],
    }
    assert _build_sources_from_audit(audit, "cited the demolition contract") == []
