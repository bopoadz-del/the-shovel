"""Part 2 of the re-scoring CPU cut: the parse cache is scoped to one search.

Part 1 (test_contract_data_parse_cache.py) pins the mechanism: the memoised
:func:`filled_particulars_rows` is identity-identical to the raw parse and
never persists across scopes. This file pins the ACTIVATION: one
``retrieve_with_filter`` call runs inside a single
:func:`per_search_parse_cache` scope, so a candidate chunk re-read by several
re-scoring predicates is parsed once per search, and the ranked result is
unchanged from the unscoped path.
"""
from __future__ import annotations

import contextlib
from unittest import mock

import pytest

from app.core import contract_data_chunks as cdc
from app.core.rag import retriever as ret
from app.core.rag.vector_store import Chunk

PARTICULARS_A = (
    "CONTRACT DATA particulars — filled-in amount / duration / percentage\n"
    "Time for Completion: 730 calendar days\n"
    "Delay damages: 0.05% of the Contract Price per day\n"
)
PARTICULARS_B = (
    "CONTRACT DATA particulars — filled-in amount / duration / percentage\n"
    "Accepted Contract Amount: AED 12,345,678.90\n"
    "Performance Security: 10% of the Accepted Contract Amount\n"
)


def _chunk(cid: str, project_id: str, score: float, text: str) -> Chunk:
    return Chunk(
        chunk_id=cid, project_id=project_id, doc_id=f"doc-{cid}",
        chunk_index=0, text=text, score=score,
    )


def _install(monkeypatch, corpus):
    def fake_search(self, project_id, qvec, k, query_text=None):
        return [
            _chunk(cid, pid, score, text)
            for cid, pid, score, text in corpus
        ]

    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.search", fake_search,
    )
    monkeypatch.setattr(ret, "_doc_name_for_id", lambda _id: "contract.pdf",
                        raising=False)
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")
    monkeypatch.delenv("MASTER_CORPUS_SOURCE_PROJECT_ID", raising=False)
    monkeypatch.delenv("RAG_DUAL_QUERY", raising=False)
    return ret


@contextlib.contextmanager
def _noop_cache():
    yield


QUERIES = [
    "What is the Time for Completion?",
    "What is the Accepted Contract Amount?",
]

CORPORA = [
    [("a", "p", 0.9, PARTICULARS_A), ("b", "p", 0.8, PARTICULARS_B)],
    [("c", "p", 0.9, PARTICULARS_B), ("d", "p", 0.7, PARTICULARS_A)],
]


@pytest.mark.parametrize("query", QUERIES)
def test_retrieve_with_filter_scopes_the_parse_cache(monkeypatch, query):
    """The decorator enters the scope on every call, even an early return."""
    entered = []

    @contextlib.contextmanager
    def spy_cache():
        entered.append(True)
        yield

    monkeypatch.setattr(ret, "per_search_parse_cache", spy_cache)
    # Empty query takes the early-return path, still inside the decorator.
    assert ret.retrieve_with_filter("", "p", 5) == ([], 0)
    assert entered == [True], "retrieve_with_filter must open the parse scope"


@pytest.mark.parametrize("query", QUERIES)
@pytest.mark.parametrize("corpus", CORPORA)
def test_cache_on_off_rank_is_identical(monkeypatch, query, corpus):
    """ids, order and scores are the same scoped vs unscoped."""
    ret_mod = _install(monkeypatch, corpus)
    on = ret_mod.retrieve_with_filter(query, "p", 5)

    monkeypatch.setattr(ret_mod, "per_search_parse_cache", _noop_cache)
    off = ret_mod.retrieve_with_filter(query, "p", 5)

    def fingerprint(result):
        chunks, filtered = result
        return ([c.chunk_id for c in chunks],
                [round(c.score or 0.0, 6) for c in chunks], filtered)

    assert fingerprint(on) == fingerprint(off)


@pytest.mark.parametrize("query", QUERIES)
@pytest.mark.parametrize("corpus", CORPORA)
def test_search_parses_each_chunk_once(monkeypatch, query, corpus):
    """Inside one search each chunk parses once; scoping saves the re-reads."""
    ret_mod = _install(monkeypatch, corpus)

    with mock.patch.object(cdc, "parse_contract_data_rows",
                           wraps=cdc.parse_contract_data_rows) as parse:
        ret_mod.retrieve_with_filter(query, "p", 5)
        on_count = parse.call_count
        unique_texts = {text for _, _, _, text in corpus}

        # Cache off: the same candidate is re-parsed by each re-scoring
        # predicate that reads it, so strictly more parses than unique chunks.
        monkeypatch.setattr(ret_mod, "per_search_parse_cache", _noop_cache)
        parse.reset_mock()
        ret_mod.retrieve_with_filter(query, "p", 5)
        off_count = parse.call_count

    assert on_count <= len(unique_texts), (
        f"cache on parsed {on_count} for {len(unique_texts)} chunks"
    )
    assert off_count > on_count, (
        f"scoping should save parses; on={on_count} off={off_count}"
    )


@pytest.mark.parametrize("result,args", [
    ("result", ("a",)),
    (42, (1, 2)),
])
def test_scoped_per_search_parse_cache_opens_scope_and_returns(
    monkeypatch, result, args,
):
    """The decorator opens one scope per call and returns the wrapped call's
    result; ``constant_return`` must fail it."""
    entered = []

    @contextlib.contextmanager
    def spy_scope():
        entered.append(True)
        yield

    monkeypatch.setattr(ret, "per_search_parse_cache", spy_scope)
    calls = []

    @ret._scoped_per_search_parse_cache
    def target(*a, **k):
        calls.append((a, k))
        return result

    assert target(*args) == result
    assert calls == [(args, {})]
    assert entered == [True]
