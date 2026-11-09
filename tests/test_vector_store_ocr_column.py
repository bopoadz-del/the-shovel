"""Direct, fast kill tests for the private vector_store methods the precise
CI-M selector (scripts/mutation_gate.py ``tests_for``) would otherwise mark
uncovered: they are named by no existing test because they are exercised only
indirectly (or, for the pgvector legs, never in the SQLite CI environment).

Each test names the method it covers so CI-M selects it, and is written to
fail when the ``constant_return`` / ``drop_rule`` mutants are applied. No
corpus fixtures; SQLite paths run against an isolated in-memory-ish store.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import numpy as np
import pytest

from app.core.rag import vector_store as vs


# ────────────────────────────────────────────────────────────────────────────
# _ensure_ocr_column — add when missing, idempotent when present
# ────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("ocr_present", [True, False])
def test_ensure_ocr_column_is_idempotent(monkeypatch, ocr_present):
    calls = []

    class _Conn:
        def execute(self, sql):
            calls.append(str(sql))

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class _Eng:
        def begin(self):
            return _Conn()

    class _Inspector:
        def __init__(self, eng):
            self.eng = eng

        def get_columns(self, table):
            cols = [{"name": "chunk_id"}]
            if ocr_present:
                cols.append({"name": "ocr"})
            return cols

    monkeypatch.setattr("sqlalchemy.inspect", _Inspector)
    vs._ensure_ocr_column(_Eng(), "chunks_test")
    assert (len(calls) == 0) == ocr_present


# ────────────────────────────────────────────────────────────────────────────
# _chunk_row_columns — the shared read column list includes ocr
# ────────────────────────────────────────────────────────────────────────────

def test_chunk_row_columns_includes_ocr():
    class _Cls:
        chunk_id = "chunk_id"
        project_id = "project_id"
        doc_id = "doc_id"
        chunk_index = "chunk_index"
        text = "text"
        knowledge_layer = "knowledge_layer"
        authority = "authority"
        page = "page"
        ocr = "ocr"

    st = object.__new__(vs.VectorStore)
    st._rag_chunk_cls = _Cls
    assert "ocr" in st._chunk_row_columns()


# ────────────────────────────────────────────────────────────────────────────
# _bm25_select — the shared BM25 statement selects the ocr column
# ────────────────────────────────────────────────────────────────────────────

def test_bm25_select_selects_ocr_column():
    st = object.__new__(vs.VectorStore)
    st._table_name = "chunks_test"
    assert "c.ocr" in st._bm25_select("c.text_search @@ q", "")


# ────────────────────────────────────────────────────────────────────────────
# _search_numpy — SQLite semantic leg hydrates the ocr flag
# ────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("ocr_flag", [True, False])
def test_search_numpy_hydrates_ocr(tmp_path, monkeypatch, ocr_flag):
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    store = vs.VectorStore(db_path=str(tmp_path / "rag.db"))
    store.upsert_chunks(
        "p", "d1", ["scanned steel beam"], np.ones((1, store.dim), dtype=np.float32),
        ocr_flags=[ocr_flag],
    )
    got = store._search_numpy("p", np.ones(store.dim, dtype=np.float32), k=1)
    assert got
    assert got[0].ocr == ocr_flag


def test_search_numpy_returns_empty_for_no_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    store = vs.VectorStore(db_path=str(tmp_path / "rag.db"))
    got = store._search_numpy("p", np.ones(store.dim, dtype=np.float32), k=1)
    assert not got


# ────────────────────────────────────────────────────────────────────────────
# _bm25_sqlite — SQLite lexical leg hydrates the ocr flag
# ────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("ocr_flag", [True, False])
def test_bm25_sqlite_hydrates_ocr(tmp_path, monkeypatch, ocr_flag):
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    store = vs.VectorStore(db_path=str(tmp_path / "rag.db"))
    store.upsert_chunks(
        "p", "d1", ["scanned steel beam"], np.ones((1, store.dim), dtype=np.float32),
        ocr_flags=[ocr_flag],
    )
    got = store._bm25_sqlite("p", "steel", 5)
    assert got
    assert got[0].ocr == ocr_flag


def test_bm25_sqlite_empty_query_returns_empty(tmp_path, monkeypatch):
    # Fake session on purpose: with the empty-query guard intact the FTS5 leg
    # must return [] without ever executing MATCH. A drop_rule mutant that
    # removes the guard would reach the fake session and return the injected
    # row, failing ``not got`` -- which is exactly what must be observed.
    st, _ = _pg_store(tmp_path, monkeypatch, rows=[_row(bm25_rank=0.5)])
    got = st._bm25_sqlite("p", "!!!", 5)
    assert not got


# ────────────────────────────────────────────────────────────────────────────
# pgvector legs — never reachable in the SQLite CI environment, so they are
# exercised against a bare store with a fake session.
# ────────────────────────────────────────────────────────────────────────────

def _row(**kw):
    base = dict(
        chunk_id="c0", project_id="p", doc_id="d", chunk_index=0,
        text="scanned steel beam", score=0.9, knowledge_layer=None,
        authority=None, page=1, ocr=True, rank=0.1,
    )
    base.update(kw)
    return SimpleNamespace(**base)


class _Result:
    def __init__(self, rows=()):
        self._rows = list(rows)

    def all(self):
        return list(self._rows)

    def scalar(self):
        return " 'steel' "


class _Session:
    def __init__(self, rows=()):
        self.rows = list(rows)
        self.sent = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, stmt, params=None):
        self.sent.append(str(stmt))
        return _Result(self.rows)


def _pg_store(tmp_path, monkeypatch, rows=(), visibility=True):
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    real = vs.VectorStore(db_path=str(tmp_path / "rag.db"))
    st = object.__new__(vs.VectorStore)
    st._rag_chunk_cls = real._rag_chunk_cls
    st.dim = real.dim
    st._table_name = "chunks_test"
    st._use_pgvector = True
    st._lock = contextlib.nullcontext()
    st._docs_visibility_ready = lambda s=None: visibility
    st._visibility_ready = visibility
    st._project_is_small = lambda s, p: True
    session = _Session(rows)
    st._session_factory = lambda: lambda: session
    monkeypatch.setattr(vs, "_memo_get", lambda key: vs._MISS)
    monkeypatch.setattr(vs, "_memo_put", lambda key, val: val)
    monkeypatch.setattr(vs, "_enable_iterative_scan", lambda s: None)
    return st, session


def test_search_pgvector_hydrates_ocr(tmp_path, monkeypatch):
    st, _ = _pg_store(tmp_path, monkeypatch, rows=[_row()])
    got = st._search_pgvector("p", np.ones(st.dim, dtype=np.float32), 5)
    assert got
    assert got[0].ocr is True


def test_search_pgvector_applies_visibility_filter(tmp_path, monkeypatch):
    st, session = _pg_store(tmp_path, monkeypatch, rows=[_row()], visibility=True)
    st._search_pgvector("p", np.ones(st.dim, dtype=np.float32), 5)
    assert any("retrieval_visible" in s for s in session.sent)


def test_bm25_postgres_hydrates_ocr(tmp_path, monkeypatch):
    st, _ = _pg_store(tmp_path, monkeypatch, rows=[_row(rank=0.5)])
    got = st._bm25_postgres("p", "steel", 5)
    assert got
    assert got[0].ocr is True


def test_bm25_postgres_empty_query_returns_empty(tmp_path, monkeypatch):
    st, _ = _pg_store(tmp_path, monkeypatch, rows=[_row()])
    got = st._bm25_postgres("p", "!!!", 5)
    assert not got


# ─────────────────────────────────────────────────────────────────────────────
# Guard branches on the mutated methods: a drop_rule mutant removes the first
# ``if``, so each test below fails when that guard is gone (the guarded early
# return is the only thing standing between the caller and a downstream query
# or a length-mismatch write).
# ─────────────────────────────────────────────────────────────────────────────

def test_upsert_chunks_rejects_mismatched_lengths(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    store = vs.VectorStore(db_path=str(tmp_path / "rag.db"))
    with pytest.raises(ValueError):
        store.upsert_chunks(
            "p", "d", ["one", "two"], np.ones((1, store.dim), dtype=np.float32),
        )


@pytest.mark.parametrize("url", ["sqlite:///g1.db", "sqlite:///g2.db"])
def test_ensure_schema_memoizes_init(monkeypatch, url):
    parent_calls = []
    monkeypatch.setattr(vs, "_ensure_sqlite_parent_dir", lambda u: parent_calls.append(u))
    monkeypatch.setattr(
        vs, "_engine_for_url",
        lambda u: SimpleNamespace(dialect=SimpleNamespace(name="sqlite")),
    )
    monkeypatch.setattr(vs, "ensure_table", lambda *a, **k: None)
    monkeypatch.setattr(vs, "_ensure_page_column", lambda *a, **k: None)
    monkeypatch.setattr(vs, "_ensure_ocr_column", lambda *a, **k: None)

    class _T:
        __tablename__ = "chunks_guard"
        __table__ = SimpleNamespace(indexes=[])

    monkeypatch.setattr(vs, "_INITIALIZED_NAMESPACES", set())
    vs._ensure_schema(url, _T)
    vs._ensure_schema(url, _T)
    # Memoised: the second call returns at the guard and never reaches the
    # parent-dir step. drop_rule (guard removed) would re-initialise and
    # append a second entry.
    assert parent_calls == [url]


@pytest.mark.parametrize("project_id", ["", None])
def test_chunks_containing_all_skips_query_on_falsy_project(project_id):
    st = object.__new__(vs.VectorStore)
    st._table_name = "chunks_t"
    st._lock = contextlib.nullcontext()
    st._docs_visibility_ready = lambda s: False
    st._hidden_doc_sql = lambda t: ""
    session = _Session(rows=[])
    st._session_factory = lambda: (lambda: session)
    got = st.chunks_containing_all(project_id, ["steel"])
    assert got == []
    assert session.sent == []


@pytest.mark.parametrize(
    "project_id,identifiers", [("", ["D549.2"]), ("p", [])],
)
def test_identifier_search_skips_query_when_empty(project_id, identifiers):
    st = object.__new__(vs.VectorStore)
    st._table_name = "chunks_t"
    st._lock = contextlib.nullcontext()
    st._docs_visibility_ready = lambda s: False
    st._hidden_doc_sql = lambda t: ""
    session = _Session(rows=[])
    st._session_factory = lambda: (lambda: session)
    got = st.identifier_search(project_id, identifiers)
    assert got == []
    assert session.sent == []
