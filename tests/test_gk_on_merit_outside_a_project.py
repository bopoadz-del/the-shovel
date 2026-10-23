"""General knowledge competes on merit unless the question is framed in a project.

Layer rule, read from the question and from which project a chunk belongs to:
  * a question framed inside the project -- "this contract", "the project
    specification", or a reference code that names one of the project's own
    documents -- keeps the general-knowledge margin (project first);
  * any other question lets a general-knowledge chunk in whenever it is the
    best match on its raw score (no margin), so a loaded code or handbook can
    answer a question about what that code says.
Synthetic chunks, names and codes only.
"""
from __future__ import annotations

GK = "gk_layer_t"
ACTIVE = "p_active_t"
KNOBS = ("RAG_GK_SCORE_MARGIN", "RAG_OWN_DOC_BOOST", "RAG_GK_TOPK_CAP", "RAG_GK_LEXICAL_FOLD")

PROJECT_TEXT = "Section 9 Ventilation: corridor extract fans run continuously during occupancy."
CODE_TEXT = "Chapter 12 Smoke control: corridor pressurisation keeps escape routes clear of smoke."


def _chunk(cid, pid, score, text, doc):
    from app.core.rag.vector_store import Chunk
    return Chunk(chunk_id=cid, project_id=pid, doc_id=doc, chunk_index=0, text=text, score=score)


def _install(monkeypatch, active, gk, names):
    from app.core.rag import retriever as ret

    def fake_search(self, project_id, qvec, k, query_text=None):
        return list(active) if project_id == ACTIVE else list(gk) if project_id == GK else []

    monkeypatch.setattr("app.core.rag.vector_store.VectorStore.search", fake_search)
    monkeypatch.setattr(ret, "_doc_name_for_id", lambda d: names.get(d, "x.pdf"), raising=False)
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", GK)
    for v in KNOBS:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("RAG_GK_SCORE_MARGIN", "0.10")
    monkeypatch.setenv("RAG_GK_LEXICAL_FOLD", "1")
    return ret


NAMES = {"d_proj": "QZ-4471-090 Building Services Specification.pdf", "d_code": "Smoke Control Handbook.pdf"}


def _order(ret, q):
    chunks, _ = ret.retrieve_with_filter(q, ACTIVE, k=3)
    return [c.chunk_id for c in chunks]


def test_outside_a_project_the_better_general_knowledge_chunk_answers(monkeypatch):
    ret = _install(monkeypatch, [_chunk("proj", ACTIVE, 0.80, PROJECT_TEXT, "d_proj")],
                   [_chunk("code", GK, 0.85, CODE_TEXT, "d_code")], NAMES)
    assert _order(ret, "How does corridor pressurisation keep escape routes clear of smoke?")[0] == "code"


def test_outside_a_project_a_weaker_general_knowledge_chunk_stays_out(monkeypatch):
    ret = _install(monkeypatch, [_chunk("proj", ACTIVE, 0.80, PROJECT_TEXT, "d_proj")],
                   [_chunk("code", GK, 0.70, CODE_TEXT, "d_code")], NAMES)
    assert _order(ret, "How do corridor extract fans run?") == ["proj"]


def test_a_code_naming_a_project_document_keeps_the_margin(monkeypatch):
    ret = _install(monkeypatch, [_chunk("proj", ACTIVE, 0.80, PROJECT_TEXT, "d_proj")],
                   [_chunk("code", GK, 0.85, CODE_TEXT, "d_code")], NAMES)
    assert _order(ret, "Under QZ-4471-090, how is smoke kept out of the escape corridors?")[0] == "proj"


def test_a_code_found_only_in_project_text_does_not_frame_the_question(monkeypatch):
    # Both chunks carry the code (both earn the identifier bonus); only a
    # project DOCUMENT NAME carrying it would frame the question.
    ret = _install(monkeypatch,
                   [_chunk("proj", ACTIVE, 0.80, PROJECT_TEXT + " Design to RT-5582.", "d_proj")],
                   [_chunk("code", GK, 0.85, CODE_TEXT + " RT-5582 governs.", "d_code")], NAMES)
    assert _order(ret, "What does RT-5582 say about corridor pressurisation?")[0] == "code"
