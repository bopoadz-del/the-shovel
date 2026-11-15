"""The retrieval trace says what kind of document each chunk came from.

Every line of ``rag_retrieval_trace`` names the chunks handed to the model
and the ones left out. Before this, it did not say what KIND of document each
chunk came from, so a trace could not show that, for example, programme or
report chunks held the handed slots on a question that asks for a specified
value. The marker already told the model the kind (S-R11, doc_typing); the
trace now carries the same two stored columns, for handed and not-handed
chunks alike.

Every test builds its own project, documents and chunks. No real document,
project, question or value is named.
"""
from __future__ import annotations

import json
import logging

import pytest

from app.core.rag import inject
from app.core.rag.inject import _emit_retrieval_trace, _trace_chunk
from app.core.rag.vector_store import Chunk


def _chunk(doc: str, idx: int, *, doc_type: str = "", doc_role: str = "") -> Chunk:
    c = Chunk(chunk_id=f"p:{doc}:{idx}", project_id="p", doc_id=doc,
              chunk_index=idx, text="invented body", score=0.5)
    c.source_name = f"invented-{doc}.pdf"
    c.doc_type, c.doc_role = doc_type, doc_role
    return c


def _trace_line(caplog, *, retrieved, handed) -> dict:
    with caplog.at_level(logging.INFO, logger="app.rag.retrieval_trace"):
        _emit_retrieval_trace(
            query_label="user_message", project_id=None,
            conversation_id=None, agent_name="t", retrieved=retrieved,
            handed=handed, threshold_fired=False, identifier_miss=False,
            top_score=0.5,
        )
    line = next(r.getMessage() for r in caplog.records
                if "rag_retrieval_trace" in r.getMessage())
    return json.loads(line.split(" ", 1)[1])


@pytest.fixture
def one_project(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_RETRIEVAL_TRACE", "1")
    from app.core import projects, users
    users.init_db()
    projects.init_db()
    users.ensure_user_exists("u-trace-kind")
    return projects.create_project("Synthetic Trace Kind Works",
                                   user_id="u-trace-kind")["id"]


# --- through the real stored columns -----------------------------------------

@pytest.mark.parametrize("handed_name,handed_role,left_name,left_role", [
    ("pour sequence programme.xer", "baseline_schedule",
     "lifting method statement rev B.docx", "weekly_report"),
    ("site diary week 9.pdf", "weekly_report",
     "finishes specification section 4.pdf", "baseline_schedule"),
])
def test_trace_carries_the_stored_kind_of_handed_and_left_out_chunks(
        one_project, caplog, handed_name, handed_role, left_name, left_role):
    """Neither chunk carries a kind when the trace is written: both get the
    stored doc_type / doc_role of their own documents row."""
    from app.core import projects
    h = projects.add_document(one_project, original_name=handed_name,
                              stored_as=handed_name, file_path=f"/tmp/{handed_name}",
                              size=1, role=handed_role)
    o = projects.add_document(one_project, original_name=left_name,
                              stored_as=left_name, file_path=f"/tmp/{left_name}",
                              size=1, role=left_role)
    handed_chunk, left_chunk = _chunk(h["id"], 0), _chunk(o["id"], 3)

    payload = _trace_line(caplog, retrieved=[handed_chunk, left_chunk],
                          handed=[handed_chunk])

    (got_h,) = payload["chunks"]
    (got_o,) = payload["not_handed"]
    assert (got_h["doc_type"], got_h["doc_role"]) == (h["doc_type"], h["doc_role"])
    assert (got_o["doc_type"], got_o["doc_role"]) == (o["doc_type"], o["doc_role"])
    assert got_h["doc_role"] == handed_role and got_o["doc_role"] == left_role
    assert "text" not in got_h and "text" not in got_o


def test_a_chunk_with_no_documents_row_traces_an_empty_kind(one_project, caplog):
    """An id the store does not know is traced with no kind, not a guess
    from its file name."""
    orphan = _chunk("no-such-doc", 1)
    orphan.source_name = "pour sequence programme.xer"
    payload = _trace_line(caplog, retrieved=[orphan], handed=[])
    (got,) = payload["not_handed"]
    assert (got["doc_type"], got["doc_role"]) == ("", "")


# --- a kind already on the chunk is kept and not read again ------------------

@pytest.mark.parametrize("carried", [True, False])
def test_kind_on_the_chunk_is_kept_and_only_missing_kinds_are_read(
        monkeypatch, caplog, carried):
    monkeypatch.setenv("RAG_RETRIEVAL_TRACE", "1")
    from app.core import projects
    asked: list[list[str]] = []

    def _kinds(ids):
        asked.append(list(ids))
        return {i: ("read_type", "read_role") for i in ids}

    monkeypatch.setattr(projects, "document_kinds", _kinds)
    c = (_chunk("d-carried", 0, doc_type="own_type", doc_role="own_role")
         if carried else _chunk("d-bare", 0))

    payload = _trace_line(caplog, retrieved=[c], handed=[c])

    (got,) = payload["chunks"]
    if carried:
        assert (got["doc_type"], got["doc_role"]) == ("own_type", "own_role")
        assert asked == []
    else:
        assert (got["doc_type"], got["doc_role"]) == ("read_type", "read_role")
        assert asked == [["d-bare"]]


@pytest.mark.parametrize("doc_type,doc_role", [("", ""), (" drawing ", "design")])
def test_trace_chunk_reports_the_kind_it_holds(doc_type, doc_role):
    got = _trace_chunk(1, _chunk("d", 0, doc_type=doc_type, doc_role=doc_role))
    assert (got["doc_type"], got["doc_role"]) == (doc_type.strip(), doc_role.strip())


def test_a_failed_kind_read_still_writes_the_line(monkeypatch, caplog):
    """Instrumentation never costs a turn: the line is written, kind empty."""
    monkeypatch.setenv("RAG_RETRIEVAL_TRACE", "1")
    from app.core import projects

    def _boom(ids):
        raise RuntimeError("store unavailable")

    monkeypatch.setattr(projects, "document_kinds", _boom)
    c = _chunk("d-x", 2)
    payload = _trace_line(caplog, retrieved=[c], handed=[c])
    (got,) = payload["chunks"]
    assert got["chunk_id"] == "p:d-x:2"
    assert (got["doc_type"], got["doc_role"]) == ("", "")
    assert inject.retrieval_trace_enabled()
