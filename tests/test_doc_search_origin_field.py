"""Search results must disclose WHICH corpus each hit came from.

Live 2026-08-02: a project owning ZERO documents returned five
Master-Corpus documents from /v1/projects/{id}/documents/search with
nothing in the payload marking them as such. That is the labeled
Master-Corpus fallback (retrieve_with_filter STEP 0b) working as designed
— but a caller of this endpoint could not tell an own-document hit from a
fallback hit, so it would silently attribute another project's document
to this one.

WHY THIS FILE EXISTS AT ALL: the first attempt at the fix added `origin`
to the intermediate `best` dict inside search_project_documents — but the
returned rows are REBUILT from scratch further down, so the field never
reached the wire. It shipped green because every test asserted on
internals or on unrelated behaviour. These tests assert on the value the
CALLER actually receives.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.core import doc_index

# pytest.ini sets `asyncio_mode = auto`, so an `async def` test is awaited by
# pytest-asyncio on a loop it owns. An earlier revision drove the coroutine
# with `asyncio.get_event_loop().run_until_complete(...)`, which passed
# locally (a loop happened to exist) and failed on a clean CI runner with
# "RuntimeError: There is no current event loop in thread 'MainThread'".
# Await directly — never hand-roll a loop in this suite.


def _chunk(doc_id, text, score, layer):
    return SimpleNamespace(
        chunk_id=f"c_{doc_id}", project_id="p1", doc_id=doc_id,
        chunk_index=0, text=text, score=score, layer=layer,
    )


@pytest.fixture
def planted(monkeypatch):
    """Three hits, one per corpus layer."""
    chunks = [
        _chunk("d_own", "own project spec text", 0.90, "own"),
        _chunk("d_gk", "curated knowledge text", 0.80, "general_knowledge"),
        _chunk("d_master", "master corpus text", 0.70, "master_corpus"),
    ]
    import app.core.rag.retriever as retr
    monkeypatch.setattr(retr, "retrieve_with_filter", lambda q, pid, k=5, **kw: (chunks, 0))
    monkeypatch.setattr(retr, "_doc_name_for_id", lambda d: f"{d}.pdf")
    monkeypatch.setattr(doc_index, "_load_index", lambda pid: None)
    return chunks


async def _search(pid="p1", q="concrete"):
    return await doc_index.search_project_documents(pid, q, top_k=5)


async def test_every_returned_row_carries_origin(planted):
    """The load-bearing assertion — on the RETURNED rows, not internals."""
    rows = await _search()

    assert rows, "planted chunks should produce results"
    for r in rows:
        assert "origin" in r, f"row is missing 'origin': {r}"


async def test_origin_reports_the_real_layer_per_document(planted):
    rows = {r["document_id"]: r["origin"] for r in await _search()}

    assert rows["d_own"] == "own"
    assert rows["d_gk"] == "general_knowledge"
    assert rows["d_master"] == "master_corpus"


async def test_master_corpus_fallback_is_distinguishable_from_own(planted):
    """The actual user-facing question: is this my document, or not mine?"""
    rows = await _search()
    mine = [r["filename"] for r in rows if r["origin"] == "own"]
    not_mine = [r["filename"] for r in rows if r["origin"] != "own"]

    assert mine == ["d_own.pdf"]
    assert sorted(not_mine) == ["d_gk.pdf", "d_master.pdf"]


async def test_origin_defaults_to_own_when_the_chunk_has_no_layer(monkeypatch):
    """A chunk from a path that never set `layer` must not read as a fallback.

    Defaulting the other way would mark a project's own documents as
    foreign — a worse failure than the gap this field closes.
    """
    bare = SimpleNamespace(
        chunk_id="c1", project_id="p1", doc_id="d1", chunk_index=0,
        text="some text", score=0.5,
    )  # deliberately NO `layer` attribute
    import app.core.rag.retriever as retr
    monkeypatch.setattr(retr, "retrieve_with_filter", lambda q, pid, k=5, **kw: ([bare], 0))
    monkeypatch.setattr(retr, "_doc_name_for_id", lambda d: "d1.pdf")
    monkeypatch.setattr(doc_index, "_load_index", lambda pid: None)

    rows = await _search()

    assert rows[0]["origin"] == "own"
