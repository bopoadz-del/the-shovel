"""A search hit carries the origin its document declares for its values (D3).

A knowledge copy generated from an encoded table says so on an opening line
of its own (``doc_typing.declared_origin``). Before this change nothing read
that line back: a hit on the copy looked like any other document, so a reader
comparing two sources' rates could not tell a copy of a table from a document.

1. ``opening_samples`` reads the opening chunks of the named documents, in
   the named projects only.
2. ``_declared_origins`` keeps only documents whose own text declares an
   encoded table, read in the searched project and the general-knowledge
   libraries.
3. A search row carries ``declared_origin`` only for such a document; a failed
   read leaves every row as it was.

Every test builds its own projects, documents and chunks from invented words
and runs on two inputs. No corpus, project, document or value is named.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.core import doc_index
from app.core import doc_typing_backfill as bf
from tests.test_ingest_reconcile import _reload_projects

_DECLARED = "Generated from `qz/tables/qzrates.json` by a qz script."


def _store(monkeypatch, tmp_path, docs):
    """``docs``: [(project_id, doc_id, [chunk texts])]. Returns SessionLocal."""
    from app.core.db import SessionLocal, engine
    from app.core.models import RagChunk

    _reload_projects(monkeypatch, tmp_path)
    RagChunk.__table__.create(bind=engine, checkfirst=True)
    with SessionLocal() as s:
        for pid, did, texts in docs:
            for i, text in enumerate(texts):
                s.add(RagChunk(chunk_id=f"{did}-{i}", project_id=pid, doc_id=did,
                               chunk_index=i, text=text,
                               embedding=np.zeros(8, dtype=np.float32),
                               created_at="2026-01-01T00:00:00Z"))
        s.commit()
    return SessionLocal


# ── 1. the opening of the named documents, in the named projects ─────────


@pytest.mark.parametrize("projects, expected", [
    (["qzp"], {"d1": "qz first\nqz second"}),
    (["qzother"], {}),
])
def test_opening_samples_read_only_the_named_projects(monkeypatch, tmp_path, projects, expected):
    session_local = _store(monkeypatch, tmp_path, [
        ("qzp", "d1", ["qz first", "qz second"]),
        ("qzp", "d2", ["qz not asked"]),
    ])
    with session_local() as s:
        assert bf.opening_samples(s, projects, ["d1"]) == expected


@pytest.mark.parametrize("doc_ids, expected", [
    ([], {}),
    (["d2", "d2"], {"d2": "qz not asked"}),
])
def test_opening_samples_read_only_the_named_documents(monkeypatch, tmp_path, doc_ids, expected):
    session_local = _store(monkeypatch, tmp_path, [
        ("qzp", "d1", ["qz first"]),
        ("qzp", "d2", ["qz not asked"]),
    ])
    with session_local() as s:
        assert bf.opening_samples(s, ["qzp"], doc_ids) == expected


# ── 2. only a declared encoded table is kept ─────────────────────────────


@pytest.mark.parametrize("opening, expected", [
    ([f"# QZ rate copy\n\n{_DECLARED}", "| qz item | 12 |"],
     {"kind": "encoded_table", "source": "qz/tables/qzrates.json"}),
    (["# QZ rate survey\n\nRates from a published qz survey.", "| qz item | 12 |"], None),
])
def test_only_a_document_declaring_its_table_has_an_origin(monkeypatch, tmp_path, opening, expected):
    _store(monkeypatch, tmp_path, [("qzp", "dx", opening)])
    monkeypatch.setattr("app.core.rag.retriever._general_knowledge_project_ids", lambda: [])
    got = doc_index._declared_origins("qzp", ["dx"])
    assert got.get("dx") == expected


@pytest.mark.parametrize("libraries, found", [(["qzlib"], True), ([], False)])
def test_a_copy_in_a_knowledge_library_is_read(monkeypatch, tmp_path, libraries, found):
    """The copy lives in a general-knowledge library, not the searched
    project: it is read only when that library is configured."""
    _store(monkeypatch, tmp_path, [("qzlib", "dk", [_DECLARED])])
    monkeypatch.setattr("app.core.rag.retriever._general_knowledge_project_ids",
                        lambda: list(libraries))
    assert ("dk" in doc_index._declared_origins("qzuser", ["dk"])) is found


# ── 3. the search row ─────────────────────────────────────────────────────


class _Chunk:
    def __init__(self, chunk_id, doc_id, score, text):
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.score = score
        self.text = text
        self.page = None
        self.layer = "own"


@pytest.fixture
def corpus(monkeypatch):
    import app.core.rag.retriever as retr

    box = {"chunks": [], "openings": {}, "fail": False}
    monkeypatch.setattr(
        retr, "retrieve_with_filter",
        lambda query, project_id, k=5, **_kw: (list(box["chunks"]), 0),
    )
    monkeypatch.setattr(retr, "_doc_name_for_id", lambda d: f"{d}.md")
    monkeypatch.setattr(retr, "_general_knowledge_project_ids", lambda: [])
    monkeypatch.setattr(doc_index, "_load_index", lambda _pid: {"documents": []})
    monkeypatch.setattr(doc_index._projects, "get_document", lambda d: {})
    monkeypatch.setattr("app.core.rag.retrieval_worker.enabled", lambda: False)

    def _openings(_session, _pids, doc_ids, limit=None):
        if box["fail"]:
            raise RuntimeError("store down")
        return {d: box["openings"][d] for d in doc_ids if d in box["openings"]}

    monkeypatch.setattr(bf, "opening_samples", _openings)
    return box


@pytest.mark.parametrize("declares", [True, False])
def test_a_row_carries_the_origin_its_document_declares(corpus, declares):
    corpus["chunks"] = [
        _Chunk("x1", "dx", 0.9, "qzslab rate 12"),
        _Chunk("y1", "dy", 0.8, "qzslab rate 15"),
    ]
    corpus["openings"] = {
        "dx": _DECLARED if declares else "Rates from a qz survey.",
        "dy": "Rates from a qz survey.",
    }
    rows, _trace = doc_index._search_project_documents_traced("qzp", "qzslab rate", 5)
    by_doc = {r["document_id"]: r for r in rows}
    if declares:
        assert by_doc["dx"]["declared_origin"] == {
            "kind": "encoded_table", "source": "qz/tables/qzrates.json"}
    else:
        assert "declared_origin" not in by_doc["dx"]
    assert "declared_origin" not in by_doc["dy"]


@pytest.mark.parametrize("fail", [False, True])
def test_a_failed_origin_read_leaves_the_rows(corpus, fail):
    corpus["chunks"] = [_Chunk("x1", "dx", 0.9, "qzslab rate 12")]
    corpus["openings"] = {"dx": _DECLARED}
    corpus["fail"] = fail
    rows, _trace = doc_index._search_project_documents_traced("qzp", "qzslab rate", 5)
    assert [r["document_id"] for r in rows] == ["dx"]
    assert ("declared_origin" in rows[0]) is (not fail)
