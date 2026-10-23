"""Tests for the source-backed answer contract in rag_inject.

When a user asks for an exact reference and the retriever cannot find a
chunk containing that reference, the injector must decline to provide
RAG context so the model does not hallucinate an answer.
"""
from __future__ import annotations

import json
import os

import pytest


@pytest.fixture
def isolated_data_dir(tmp_path, monkeypatch):
    """Fresh DATA_DIR and RAG caches."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")
    from app.core.rag import embeddings as _emb, vector_store as _vs
    _emb.reset_embedder_cache()
    _vs.reset_store_cache()
    yield tmp_path
    _emb.reset_embedder_cache()
    _vs.reset_store_cache()


def test_rag_inject_declines_when_identifier_missing(isolated_data_dir, monkeypatch):
    """If the user asks for VO Ref 99 and no chunk contains it, injection
    should be skipped and the audit record should flag identifier_miss."""
    from app.core.rag import retriever as ret
    from app.core.rag.inject import rag_inject
    from app.core.rag.vector_store import Chunk

    # Fake retriever returns a semantically-similar chunk that does NOT
    # contain the requested identifier.
    def fake_retrieve(query, project_id, k=5, **kwargs):
        return [
            Chunk(
                chunk_id="c1",
                project_id=project_id,
                doc_id="doc_x",
                chunk_index=0,
                text="Variation orders are tracked in a register.",
                score=0.85,
            )
        ], 0

    monkeypatch.setattr("app.core.rag.inject.retrieve_with_filter", fake_retrieve)

    msg, audit = rag_inject(
        user_message="What is the status of VO Ref 99?",
        project_id="proj_a",
        conversation_id="ws-proj_a",
        user_id="u1",
        agent_name="project-assistant",
    )

    assert msg is None
    assert audit["identifier_miss"] is True
    assert any("vo" in i.lower() for i in audit["extracted_identifiers"])
    assert audit["threshold_fired"] is True


def test_rag_inject_includes_context_when_identifier_present(isolated_data_dir, monkeypatch):
    """If a retrieved chunk contains the identifier, context should be injected."""
    from app.core.rag import retriever as ret
    from app.core.rag.inject import rag_inject
    from app.core.rag.vector_store import Chunk

    def fake_retrieve(query, project_id, k=5, **kwargs):
        return [
            Chunk(
                chunk_id="c1",
                project_id=project_id,
                doc_id="doc_x",
                chunk_index=0,
                text="VO Ref: 99 | Status: Closed | Closed date: 2024-02-12",
                score=0.85,
            )
        ], 0

    monkeypatch.setattr("app.core.rag.inject.retrieve_with_filter", fake_retrieve)

    msg, audit = rag_inject(
        user_message="What is the status of VO Ref 99?",
        project_id="proj_a",
        conversation_id="ws-proj_a",
        user_id="u1",
        agent_name="project-assistant",
    )

    assert msg is not None
    assert "VO Ref: 99" in msg["content"]
    assert audit.get("identifier_miss") is not True
