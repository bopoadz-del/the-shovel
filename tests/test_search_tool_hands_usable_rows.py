"""The search tool hands usable rows (S-R9, slice 1).

1. A row shows the passage of each chunk that matches the query. It is not
   the chunk's first words cut by position.
2. A row says where its passages sit: their pages, and the stored document
   type and role. A field the store does not hold is left out.
3. The tool message is cut by whole rows to fit the per-message cap. A row is
   never cut part-way through, and the count of rows left out is stated.

Every test builds its own chunks and rows from invented words and runs on
two inputs. No corpus, project, document or value is named.
"""
from __future__ import annotations

import json

import pytest

from app.core import doc_index


class _Chunk:
    def __init__(self, chunk_id, doc_id, score, text, page=None):
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.score = score
        self.text = text
        self.page = page
        self.layer = "own"


def _filler(n, stem="lorem"):
    return [f"{stem}{i}" for i in range(n)]


def _long_text_with_fact_at(position, total, fact_words):
    words = _filler(total)
    words[position:position + len(fact_words)] = fact_words
    return " ".join(words)


# ── 1. the passage that matches the query ────────────────────────────────


@pytest.mark.parametrize("position", [5, 230])
def test_passage_holds_the_query_words_wherever_they_sit(monkeypatch, position):
    """Wherever the matching sentence sits, it is in the passage. Before
    this change, the input at word 230 lost it to the head cut."""
    monkeypatch.delenv("SEARCH_SNIPPET_WORDS", raising=False)
    fact = ["qzwidget", "rating", "is", "seventeen", "qzunits"]
    text = _long_text_with_fact_at(position, 400, fact)
    passage = doc_index._matching_passage(text, "qzwidget rating qzunits", 120)
    assert " ".join(fact) in passage
    assert len(passage.replace("...", "").split()) == 120


@pytest.mark.parametrize("n_words", [1, 120])
def test_a_chunk_within_the_width_comes_back_unchanged(n_words):
    text = " ".join(_filler(n_words, "ipsum"))
    assert doc_index._matching_passage(text, "ipsum3 anything", 120) == text


@pytest.mark.parametrize("has_term", [False, True])
def test_a_chunk_with_no_query_term_keeps_its_head(has_term):
    words = _filler(300)
    if has_term:
        words[250] = "qzmarker"
    passage = doc_index._matching_passage(" ".join(words), "qzmarker", 120)
    assert passage.startswith("lorem0 ") is (not has_term)
    assert ("qzmarker" in passage) is has_term


def test_a_term_the_chunk_repeats_weighs_less_than_a_rare_one():
    """Two query terms: one repeated across the opening words, one rare
    term much later. The window holding the rare term wins, because a
    term the chunk repeats on every line says little about where the
    asked-for passage is."""
    words = _filler(400)
    for i in range(0, 100, 4):
        words[i] = "qzcommon"
    words[300] = "qzrare"
    passage = doc_index._matching_passage(" ".join(words), "qzcommon qzrare", 120)
    assert "qzrare" in passage
    # and the twin: with only the repeated term asked, the head stays
    passage2 = doc_index._matching_passage(" ".join(words), "qzcommon", 120)
    assert passage2.startswith("qzcommon ")


# ── 2. the row says where its passages sit ───────────────────────────────


@pytest.fixture
def corpus(monkeypatch):
    import app.core.rag.retriever as retr

    box = {"chunks": [], "docs": {}}
    monkeypatch.setattr(
        retr, "retrieve_with_filter",
        lambda query, project_id, k=5, **_kw: (list(box["chunks"]), 0),
    )
    monkeypatch.setattr(retr, "_doc_name_for_id", lambda d: f"{d}.pdf")
    monkeypatch.setattr(doc_index, "_load_index", lambda _pid: {"documents": []})
    monkeypatch.setattr(doc_index._projects, "get_document",
                        lambda d: box["docs"].get(d))
    monkeypatch.setattr("app.core.rag.retrieval_worker.enabled", lambda: False)
    monkeypatch.delenv("SEARCH_SNIPPET_WORDS", raising=False)
    monkeypatch.delenv("SEARCH_CHUNKS_PER_DOC", raising=False)
    return box


@pytest.mark.parametrize("position", [10, 260])
def test_rows_carry_the_matching_passage_of_a_long_chunk(corpus, position):
    """The query names the thing; the value follows it. The passage is
    centred on the matched words, so the value after them is kept too."""
    fact = ["qzgasket", "pressure", "is", "nine", "qzbar"]
    corpus["chunks"] = [
        _Chunk("x1", "dx", 0.9, _long_text_with_fact_at(position, 500, fact)),
        _Chunk("y1", "dy", 0.8, " ".join(_filler(40, "short"))),
    ]
    rows, _trace = doc_index._search_project_documents_traced("p", "qzgasket pressure", 5)
    by_doc = {r["document_id"]: r for r in rows}
    assert " ".join(fact) in by_doc["dx"]["snippet"]
    assert by_doc["dy"]["snippet"] == " ".join(_filler(40, "short"))


@pytest.mark.parametrize("pages,stored", [
    ((7, 3), {"doc_type": "qztype", "doc_role": "qzrole"}),
    ((None, 4), {"doc_type": "qzother", "doc_role": "qzsecond"}),
])
def test_rows_name_pages_type_and_role_only_when_stored(corpus, pages, stored):
    corpus["chunks"] = [
        _Chunk("x1", "dx", 0.9, "qzalpha one", page=pages[0]),
        _Chunk("x2", "dx", 0.85, "qzalpha two", page=pages[1]),
        _Chunk("y1", "dy", 0.8, "qzalpha three", page=None),
    ]
    corpus["docs"] = {"dx": stored, "dy": {"doc_type": "", "doc_role": None}}
    rows, _trace = doc_index._search_project_documents_traced("p", "qzalpha", 5)
    by_doc = {r["document_id"]: r for r in rows}
    assert by_doc["dx"]["pages"] == [p for p in pages if p]
    assert by_doc["dx"]["doc_type"] == stored["doc_type"]
    assert by_doc["dx"]["doc_role"] == stored["doc_role"]
    for key in ("pages", "doc_type", "doc_role"):
        assert key not in by_doc["dy"]


@pytest.mark.parametrize("fails", [False, True])
def test_a_failed_document_lookup_still_returns_the_row(corpus, monkeypatch, fails):
    def _lookup(_d):
        if fails:
            raise RuntimeError("store down")
        return {"doc_type": "qztype", "doc_role": "qzrole"}

    monkeypatch.setattr(doc_index._projects, "get_document", _lookup)
    corpus["chunks"] = [_Chunk("x1", "dx", 0.9, "qzbeta text", page=2),
                        _Chunk("y1", "dy", 0.8, "qzbeta more")]
    rows, _trace = doc_index._search_project_documents_traced("p", "qzbeta", 5)
    assert [r["document_id"] for r in rows] == ["dx", "dy"]
    assert rows[0]["pages"] == [2]
    assert ("doc_type" in rows[0]) is (not fails)
    assert ("doc_role" in rows[0]) is (not fails)


# ── 3. the tool message is cut by whole rows ─────────────────────────────


def _row(i, words):
    return {"document_id": f"d{i}", "filename": f"f{i}.pdf",
            "snippet": " ".join(_filler(words, f"w{i}x")), "score": 0.5}


@pytest.mark.parametrize("n_rows,words,cap", [(3, 20, 8000), (10, 200, 8000)])
def test_the_payload_fits_the_cap_with_every_row_whole(n_rows, words, cap):
    from app.agents.base.tools import search_rows_within_cap

    rows = [_row(i, words) for i in range(n_rows)]
    payload = search_rows_within_cap(rows, cap)
    assert len(json.dumps(payload, default=str)) <= cap
    shown = payload["results"]
    assert shown == rows[: len(shown)]  # each shown row is the row, unchanged
    withheld = n_rows - len(shown)
    assert payload.get("rows_withheld", 0) == withheld
    if withheld:
        assert str(withheld) in payload["note"]
    else:
        assert payload == {"results": rows}  # a result that fits is unchanged


@pytest.mark.parametrize("words", [30, 3000])
def test_only_a_first_row_too_big_alone_is_shortened(words):
    from app.agents.base.tools import search_rows_within_cap

    rows = [_row(0, words), _row(1, 10)]
    cap = 2000
    payload = search_rows_within_cap(rows, cap)
    assert len(json.dumps(payload, default=str)) <= cap
    first = payload["results"][0]
    if words == 30:
        assert payload == {"results": rows}
    else:
        assert first["document_id"] == "d0"
        assert first["snippet"].startswith("w0x0 w0x1")
        assert first["snippet"].endswith(" ...")
        assert payload["rows_withheld"] == 1


@pytest.mark.parametrize("words", [20, 400])
async def test_the_tool_hands_a_payload_within_the_runtime_cap(corpus, monkeypatch, words):
    from app.agents.base.tools import handle_search_project_documents

    monkeypatch.setattr("app.agents.runtime._effective_tool_result_max_chars", lambda: 3000)
    monkeypatch.setenv("SEARCH_SNIPPET_WORDS", "400")
    corpus["chunks"] = [
        _Chunk(f"c{i}", f"d{i}", 0.9 - i / 100, " ".join(_filler(words, f"t{i}x")))
        for i in range(6)
    ]

    class _Call:
        name = "search_project_documents"
        project_id = "p-test"
        args = {"query": "anything", "top_k": 6}

    out = await handle_search_project_documents(_Call())
    payload = out["result"]
    assert len(json.dumps(payload, default=str)) <= 3000
    assert len(payload["results"]) + payload.get("rows_withheld", 0) == 6
    if words == 20:
        assert "rows_withheld" not in payload
    else:
        assert payload["rows_withheld"] > 0
