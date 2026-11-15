"""At the injection cap, an exact score tie is broken by document kind.

The cap walks chunks in score order and drops what does not fit, so inside an
exact tie the retrieval order alone decided which document was handed. A
chunk from a progress-record kind (the type table's ``records_progress``:
programme, report, minutes) now yields a tie to a chunk of another kind,
unless the question names that kind. Different scores keep their order.

Every test builds its own project, documents and chunks through the real
documents table (the stored doc_type comes from the type table at
add_document). No real document, project, question or value is named.
"""
from __future__ import annotations

import pytest

from app.core.rag.inject import apply_token_cap, break_score_ties_by_kind
from app.core.rag.vector_store import Chunk

_BODY = "invented clause body " * 40  # ~200 tokens, so the cap fits one


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MAX_RAG_TOKENS", "300")
    from app.core import projects, users
    users.init_db()
    projects.init_db()
    users.ensure_user_exists("u-k1")
    return projects.create_project("Synthetic Tie Works", user_id="u-k1")["id"]


def _stored(project_id: str, name: str) -> str:
    from app.core import projects
    return projects.add_document(project_id, original_name=name, stored_as=name,
                                 file_path=f"/tmp/{name}", size=1)["id"]


def _chunk(doc_id: str, idx: int, score: float) -> Chunk:
    return Chunk(chunk_id=f"p:{doc_id}:{idx}", project_id="p", doc_id=doc_id,
                 chunk_index=idx, text=_BODY, score=score)


# progress-record kind first in retrieval order, the other kind second
_PAIRS = [
    ("pour sequence programme rev 2.xer", "kerb detail drawing sheet 4.pdf"),
    ("site weekly report 12.pdf", "finishes specification section 9.pdf"),
]


@pytest.mark.parametrize("progress_name,other_name", _PAIRS)
def test_a_tie_hands_the_other_kind_when_the_question_names_neither(
        project, progress_name, other_name):
    p = _chunk(_stored(project, progress_name), 0, 0.61)
    o = _chunk(_stored(project, other_name), 3, 0.61)
    kept, _ = apply_token_cap([p, o], query="what grade is the kerb backing")
    assert [c.chunk_id for c in kept] == [o.chunk_id]


@pytest.mark.parametrize("progress_name,other_name,question", [
    (_PAIRS[0][0], _PAIRS[0][1], "when does the programme show the pour"),
    (_PAIRS[1][0], _PAIRS[1][1], "what does the weekly report say about it"),
])
def test_a_tie_keeps_its_order_when_the_question_names_the_progress_kind(
        project, progress_name, other_name, question):
    p = _chunk(_stored(project, progress_name), 0, 0.61)
    o = _chunk(_stored(project, other_name), 3, 0.61)
    kept, _ = apply_token_cap([p, o], query=question)
    assert [c.chunk_id for c in kept] == [p.chunk_id]


@pytest.mark.parametrize("progress_name,other_name", _PAIRS)
def test_a_higher_score_is_never_overridden_by_kind(
        project, progress_name, other_name):
    p = _chunk(_stored(project, progress_name), 0, 0.62)
    o = _chunk(_stored(project, other_name), 3, 0.61)
    kept, _ = apply_token_cap([p, o], query="what grade is the kerb backing")
    assert [c.chunk_id for c in kept] == [p.chunk_id]


@pytest.mark.parametrize("first_id,second_id", [("no-row-a", "no-row-b"),
                                                ("gone-1", "gone-2")])
def test_chunks_with_no_stored_kind_keep_their_order(project, first_id, second_id):
    a, b = _chunk(first_id, 0, 0.5), _chunk(second_id, 1, 0.5)
    got = break_score_ties_by_kind([a, b], "what grade is the kerb backing")
    assert [c.chunk_id for c in got] == [a.chunk_id, b.chunk_id]


@pytest.mark.parametrize("progress_name,other_name", _PAIRS)
def test_the_kind_comes_from_the_table_flag_not_the_type_name(
        project, monkeypatch, progress_name, other_name):
    """With the flag removed from the table, the same stored kinds tie in
    retrieval order: the rule reads ``records_progress``, nothing else."""
    from app.core import doc_types
    real = doc_types.list_types()
    monkeypatch.setattr(doc_types, "list_types", lambda: [
        {k: v for k, v in t.items() if k != "records_progress"} for t in real])
    p = _chunk(_stored(project, progress_name), 0, 0.61)
    o = _chunk(_stored(project, other_name), 3, 0.61)
    kept, _ = apply_token_cap([p, o], query="what grade is the kerb backing")
    assert [c.chunk_id for c in kept] == [p.chunk_id]
