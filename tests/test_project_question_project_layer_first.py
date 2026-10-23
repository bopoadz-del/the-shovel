"""A project question cites the project's own clause ahead of general knowledge.

Layer rule, not a document rule: for a lookup question framed in the asker's
own project or contract ("this contract", "the project specification"), no
general-knowledge chunk outranks the active project's best chunk -- however
dense the code or handbook clause on the same topic is. General knowledge still
follows (it is context, not removed). A question that names an outside standard
and does not speak of its own project keeps the ordinary merged ranking.

Synthetic data only: a made-up project clause and a made-up code clause.
"""
from __future__ import annotations

GK = "gk_layer"
ACTIVE = "p_active"
KNOB_VARS = ("RAG_GK_SCORE_MARGIN", "RAG_OWN_DOC_BOOST", "RAG_GK_TOPK_CAP", "RAG_GK_LEXICAL_FOLD")

PROJECT_CLAUSE = "Clause 7.4 Concrete cover to foundations cast against soil shall be 75 mm."
CODE_CLAUSE = "Section 20.5 Minimum specified concrete cover for members cast against earth is 75 mm."


def _chunk(chunk_id, project_id, score, text):
    from app.core.rag.vector_store import Chunk
    return Chunk(chunk_id=chunk_id, project_id=project_id, doc_id=f"doc-{chunk_id}",
                 chunk_index=0, text=text, score=score)


def _install(monkeypatch, active_chunks, gk_chunks):
    from app.core.rag import retriever as ret

    def fake_search(self, project_id, qvec, k, query_text=None):
        if project_id == ACTIVE:
            return list(active_chunks)
        if project_id == GK:
            return list(gk_chunks)
        return []

    monkeypatch.setattr("app.core.rag.vector_store.VectorStore.search", fake_search)
    monkeypatch.setattr(ret, "_doc_name_for_id", lambda _id: "source.pdf", raising=False)
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", GK)
    for var in KNOB_VARS:
        monkeypatch.delenv(var, raising=False)
    # The live configuration: GK must beat the project by a margin, raw score.
    monkeypatch.setenv("RAG_GK_SCORE_MARGIN", "0.10")
    monkeypatch.setenv("RAG_GK_LEXICAL_FOLD", "1")
    return ret


def _order(ret, query):
    chunks, _ = ret.retrieve_with_filter(query, ACTIVE, k=3)
    return [c.chunk_id for c in chunks]


def test_project_question_cites_the_project_clause_first(monkeypatch):
    # The code clause clears the margin on its own score (0.95 >= 0.80 + 0.10).
    ret = _install(monkeypatch,
                   active_chunks=[_chunk("proj", ACTIVE, 0.80, PROJECT_CLAUSE)],
                   gk_chunks=[_chunk("code", GK, 0.95, CODE_CLAUSE)])
    order = _order(ret, "What concrete cover does this contract require for foundations cast against soil?")
    assert order[0] == "proj"
    assert "code" in order  # general knowledge stays as context, behind the project


def test_project_specification_framing_is_a_project_question(monkeypatch):
    ret = _install(monkeypatch,
                   active_chunks=[_chunk("proj", ACTIVE, 0.80, PROJECT_CLAUSE)],
                   gk_chunks=[_chunk("code", GK, 0.95, CODE_CLAUSE)])
    assert _order(ret, "Per the project specification, what cover is required against soil?")[0] == "proj"


def test_a_standards_question_keeps_the_merged_ranking(monkeypatch):
    ret = _install(monkeypatch,
                   active_chunks=[_chunk("proj", ACTIVE, 0.80, PROJECT_CLAUSE)],
                   gk_chunks=[_chunk("code", GK, 0.95, CODE_CLAUSE)])
    assert _order(ret, "What minimum concrete cover does the code give for members cast against earth?")[0] == "code"

