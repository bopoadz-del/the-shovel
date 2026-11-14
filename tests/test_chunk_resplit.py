"""RQ-4: stored chunks longer than the embedder window are re-split in place.

Every test builds its own text, project, documents and chunk rows. The token
counter is injected, so the bound under test is whatever window is passed.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

import pytest

from app.core import chunk_resplit as cr
from tests.test_ingest_reconcile import _reload_projects


def words(s: str) -> int:
    return len(re.findall(r"\S+", s))


PROSE = (
    "The contractor shall cure the concrete for seven days. Surfaces stay moist "
    "throughout the period.\n\nThe free fall of concrete shall not exceed 3.5 m "
    "during placement. Samples are taken per 50 m3 of each mix.\n\n"
    "Reinforcement is grade 420 deformed bar to the stated standard."
)
TABLE = "\n".join(
    f"| {i}.1 | Pipe DN{100 * i} | m | {12.5 * i} | {3.25 * i} |" for i in range(1, 9)
)


@pytest.mark.parametrize("text, window, boundary", [
    (PROSE, 12, "sentence"),
    (TABLE, 16, "line"),
])
def test_partition_is_exact_and_every_piece_fits(text, window, boundary):
    pieces = cr.partition(text, words, window)
    assert len(pieces) > 1
    assert "".join(pieces) == text  # pure partition: nothing lost, nothing added
    assert all(words(p) <= window for p in pieces)
    if boundary == "line":
        assert all(p.endswith("\n") for p in pieces[:-1])  # whole rows only
    else:
        assert all(re.search(r"[.!?;:]\s*$", p) for p in pieces[:-1])
    assert not any(p.startswith(".5") for p in pieces)  # never inside 3.5


@pytest.mark.parametrize("text, window, expected", [
    (PROSE, 10_000, [PROSE]),       # fits: one piece, untouched
    (TABLE, words(TABLE), [TABLE]),  # exactly at the window: untouched
])
def test_text_within_the_window_is_one_piece(text, window, expected):
    assert cr.partition(text, words, window) == expected


def nonspace(s: str) -> int:
    return len(s.replace(" ", ""))


@pytest.mark.parametrize("text", ["X" * 23, "AB" * 11 + "C"])
def test_single_run_longer_than_the_window_is_cut_only_then(text):
    pieces = cr.partition(text, nonspace, 5)
    assert "".join(pieces) == text
    assert all(nonspace(p) <= 5 for p in pieces)
    assert len(pieces) == 5


@pytest.mark.parametrize("text", ["alpha beta gamma delta", "12.50 300.75 4100.25 9.5"])
def test_word_boundary_wins_over_a_hard_cut(text):
    pieces = cr.partition(text, nonspace, 10)
    assert len(pieces) > 1 and "".join(pieces) == text
    whole = set(text.split())
    assert all(set(p.split()) <= whole for p in pieces)  # no word or number cut


# ── store-level plan / execute ───────────────────────────────────────────

def _project(monkeypatch, tmp_path, pid):
    pm = _reload_projects(monkeypatch, tmp_path)
    pm.create_project(pid, user_id="system", project_id=pid)
    return pm


def _project_on_job_db(monkeypatch, tmp_path, pid):
    """Like ``_project`` but on the job's own DATABASE_URL (PostgreSQL in test-postgres).

    ``_reload_projects`` drops DATABASE_URL, so an earlier test in the same
    worker may have left ``app.core.db`` bound to SQLite; reload it back.
    """
    import importlib

    import app.core.db as db_mod
    import app.core.users as users_mod
    from app.core import projects

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    importlib.reload(db_mod)
    importlib.reload(users_mod)
    users_mod._initialized = False
    pm = importlib.reload(projects)
    pm._initialized = False
    pm.init_db()
    assert db_mod.engine.dialect.name == "postgresql"
    pm.create_project(pid, user_id="system", project_id=pid)
    return pm


def _cls():
    import os

    from app.core.db import engine
    from app.core.models import EMBEDDING_DIM, make_rag_chunk_class, rag_chunk_class_for
    from app.core.rag.vector_store import ensure_table

    ns = (os.getenv("RAG_VECTOR_NAMESPACE", "v2") or "").strip()
    cls = rag_chunk_class_for(ns) or make_rag_chunk_class(ns, EMBEDDING_DIM, "test")
    ensure_table(cls.__table__, bind=engine)
    return cls


def _doc(pm, pid, name, texts, doc_type, page):
    import numpy as np
    from sqlalchemy import text as sql

    from app.core import ingest_status as ist
    from app.core.db import SessionLocal

    doc = pm.add_document(pid, name, size=10)
    pm.stamp_document_index(doc["id"], chunk_count=len(texts), ingest_status=ist.INDEXED)
    cls = _cls()
    vec = np.zeros(int(cls.embedding_identity["dim"]), dtype=np.float32)
    now = datetime.now(timezone.utc).isoformat()
    with SessionLocal() as s:
        s.execute(sql("UPDATE documents SET doc_type = :t WHERE id = :d"),
                  {"t": doc_type, "d": doc["id"]})
        for i, body in enumerate(texts):
            s.add(cls(chunk_id=f"{doc['id']}-{i}", project_id=pid, doc_id=doc["id"],
                      chunk_index=i, text=body, embedding=vec, created_at=now,
                      page=page + i, ocr=True, knowledge_layer="L2A"))
        s.commit()
    return doc["id"]


def _rows(did):
    from sqlalchemy import select

    from app.core.db import SessionLocal

    cls = _cls()
    with SessionLocal() as s:
        return [(r.chunk_id, r.chunk_index, r.text, r.page, r.ocr, r.knowledge_layer)
                for r in s.execute(select(cls).where(cls.doc_id == did)
                                   .order_by(cls.chunk_index)).scalars()]


@pytest.fixture
def window(monkeypatch):
    monkeypatch.setattr(cr, "embedder_window", lambda: (words, 12, "test-model"))


@pytest.mark.parametrize("texts, doc_type", [
    (["Short heading.", PROSE, "Closing line."], "method_statement"),
    ([TABLE, "| 9.1 | Manhole | nr | 4 | 900 |"], "boq"),
])
def test_plan_writes_nothing_and_execute_needs_the_live_hash(
    monkeypatch, tmp_path, window, texts, doc_type,
):
    pm = _project(monkeypatch, tmp_path, "rproj")
    did = _doc(pm, "rproj", "a.pdf", texts, doc_type, page=7)
    other = _doc(pm, "rproj", "b.pdf", [PROSE], doc_type, page=1)  # not listed
    before, before_other = _rows(did), _rows(other)

    plan = cr.chunk_resplit(project_id="rproj", doc_ids=[did])["plan"]
    active = plan["tables"][plan["active_table"]]
    expected = cr.partition(texts[0] if doc_type == "boq" else PROSE, words, 12)
    assert active["by_doc_type"][doc_type]["parents_over_window"] == 1
    assert active["pieces_to_write"] == len(expected)
    assert plan["doc_ids_found"] == 1 and plan["embedding_time"]["sample"] > 0
    assert _rows(did) == before  # the plan writes nothing

    wrong = cr.chunk_resplit(project_id="rproj", doc_ids=[did], execute=True,
                             expected_plan_hash="0" * 64)
    assert wrong["executed"] is False and wrong["refused"]
    assert _rows(did) == before

    done = cr.chunk_resplit(project_id="rproj", doc_ids=[did], execute=True,
                            expected_plan_hash=plan["plan_hash"])
    assert done["documents"] == {did: "replaced"}
    after = _rows(did)
    assert [r[1] for r in after] == list(range(len(after)))
    assert "".join(r[2] for r in after) == "".join(texts)  # no text lost
    parent = next(r for r in before if words(r[2]) > 12)
    pieces = [r for r in after if r[0].split("#")[0] == parent[0]]
    assert len(pieces) == len(expected)
    assert {(r[3], r[4], r[5]) for r in pieces} == {(parent[3], parent[4], parent[5])}
    assert all(words(r[2]) <= 12 for r in after)
    assert _rows(other) == before_other  # outside the list: untouched
    assert pm.get_document(did)["chunk_count"] == len(after)

    again = cr.chunk_resplit(project_id="rproj", doc_ids=[did])["plan"]
    assert again["tables"][again["active_table"]]["parents_over_window"] == 0


@pytest.mark.parametrize("only, replaced", [("boq", "bill"), ("spec", "spec")])
def test_doc_type_filter_slices_the_execute(monkeypatch, tmp_path, window, only, replaced):
    pm = _project(monkeypatch, tmp_path, "fproj")
    ids = {"bill": _doc(pm, "fproj", "bill.pdf", [TABLE], "boq", page=1),
           "spec": _doc(pm, "fproj", "spec.pdf", [PROSE], "spec", page=1)}
    plan = cr.chunk_resplit(project_id="fproj", doc_ids=list(ids.values()))["plan"]
    done = cr.chunk_resplit(project_id="fproj", doc_ids=list(ids.values()), execute=True,
                            expected_plan_hash=plan["plan_hash"], doc_type=only)
    assert done["documents"] == {ids[replaced]: "replaced"}
    untouched = next(k for k in ids if k != replaced)
    assert len(_rows(ids[untouched])) == 1


@pytest.mark.parametrize("stale, state", [(True, "changed"), (False, "replaced")])
def test_compare_and_set_skips_a_document_changed_after_the_plan(
    monkeypatch, tmp_path, stale, state,
):
    import numpy as np

    from app.core.db import SessionLocal

    pm = _project(monkeypatch, tmp_path, "cproj")
    did = _doc(pm, "cproj", "c.pdf", [PROSE], "spec", page=3)
    rows = [(r[0], r[1], r[2]) for r in _rows(did)]
    pieces = cr.partition(PROSE, words, 12)
    item = {"doc_type": "spec", "parents": {rows[0][0]: pieces},
            "rows_hash": cr.doc_rows_hash(rows if not stale else [(rows[0][0], 0, "edited")])}
    cls = _cls()
    dim = int(cls.embedding_identity["dim"])
    vectors = {rows[0][0]: [np.ones(dim, dtype=np.float32)] * len(pieces)}
    with SessionLocal() as s:
        assert cr._replace_document(s, cls, "cproj", did, item, vectors) == state
        s.commit()
    assert len(_rows(did)) == (1 if stale else len(pieces))


_ON_POSTGRES = __import__("os").getenv("PYTEST_USE_POSTGRES", "").strip().lower() in (
    "1", "true", "yes",
)


@pytest.mark.skipif(not _ON_POSTGRES, reason="text_search is a PostgreSQL column (test-postgres job)")
@pytest.mark.parametrize("texts", [[PROSE], [TABLE]])
def test_each_piece_text_search_is_derived_from_its_own_text(
    monkeypatch, tmp_path, window, texts,
):
    """A piece must not be found by a word that only its parent's other pieces hold.

    ``text_search`` is GENERATED ALWAYS from ``text`` and is not an ORM column,
    so the column copy in ``_replace_document`` cannot carry the parent's
    vector into a piece. This checks it on the live schema.
    """
    from sqlalchemy import text as sql

    from app.core.db import SessionLocal

    import os

    from app.core.rag import vector_store as vs

    pm = _project_on_job_db(monkeypatch, tmp_path, "tsproj")
    # Build the chunk table the way the app does at startup (the GENERATED
    # text_search column + GIN index come from _ensure_schema, not the ORM).
    monkeypatch.setattr(vs, "_INITIALIZED_NAMESPACES", set())
    vs._ensure_schema(os.environ["DATABASE_URL"], _cls())
    did = _doc(pm, "tsproj", "t.pdf", texts, "spec", page=1)
    plan = cr.chunk_resplit(project_id="tsproj", doc_ids=[did])["plan"]
    done = cr.chunk_resplit(project_id="tsproj", doc_ids=[did], execute=True,
                            expected_plan_hash=plan["plan_hash"])
    assert done["documents"] == {did: "replaced"}
    table = cr._active_table()
    cls = _cls()
    assert table == cls.__tablename__
    assert "text_search" not in {c.key for c in cls.__mapper__.column_attrs}
    parent = texts[0]
    with SessionLocal() as s:
        generated = s.execute(sql(
            "SELECT a.attgenerated FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid "
            "WHERE c.relname = :t AND a.attname = 'text_search'"), {"t": table}).scalar()
        assert generated == "s"  # STORED generated column: Postgres derives it
        rows = s.execute(sql(
            f"SELECT chunk_id, text, text_search::text = to_tsvector('english', text)::text, "
            f"text_search::text = to_tsvector('english', :p)::text "
            f"FROM {table} WHERE doc_id = :d ORDER BY chunk_index"),
            {"d": did, "p": parent}).all()
        assert len(rows) > 1
        for cid, body, own, as_parent in rows:
            assert own is True and as_parent is False
            # a lexeme of the parent that this piece's own text lacks (taken from
            # Postgres' parser, so numbers and codes count, not only words)
            foreign = [r[0] for r in s.execute(sql(
                "SELECT unnest(tsvector_to_array(to_tsvector('english', :p))) "
                "EXCEPT SELECT unnest(tsvector_to_array(to_tsvector('english', :b))) "
                "ORDER BY 1"), {"p": parent, "b": body}).all()]
            assert foreign
            for lex in foreign:
                hits = s.execute(sql(
                    f"SELECT count(*) FROM {table} WHERE chunk_id = :c "
                    "AND text_search @@ CAST(quote_literal(:w) AS tsquery)"),
                    {"c": cid, "w": lex}).scalar()
                assert hits == 0
                # the same lexeme does hit the parent's vector: the probe is live
                assert s.execute(sql(
                    "SELECT to_tsvector('english', :p) @@ CAST(quote_literal(:w) AS tsquery)"),
                    {"p": parent, "w": lex}).scalar() is True


@pytest.mark.parametrize("doc_ids", [[], ["  ", ""]])
def test_no_whole_project_run(monkeypatch, tmp_path, window, doc_ids):
    _project(monkeypatch, tmp_path, "nproj")
    with pytest.raises(ValueError):
        cr.chunk_resplit(project_id="nproj", doc_ids=doc_ids)


# ── the embedder's own window ────────────────────────────────────────────

class _Tok:
    def __init__(self, specials):
        self.specials = specials

    def num_special_tokens_to_add(self, pair=False):
        return self.specials

    def __call__(self, text, add_special_tokens=False, truncation=False):
        return {"input_ids": text.split()}


class _Model:
    def __init__(self, max_len, specials):
        self.max_seq_length = max_len
        self.tokenizer = _Tok(specials)


@pytest.mark.parametrize("max_len, specials, window", [(512, 2, 510), (256, 3, 253)])
def test_token_window_is_the_models_max_length_minus_special_tokens(max_len, specials, window):
    from app.core.rag.embeddings import Embedder

    emb = Embedder("fake")
    emb._backend = "sentence_transformers"
    emb._model = _Model(max_len, specials)
    count_tokens, got = emb.token_window()
    assert got == window
    assert count_tokens("a b c") == 3


class _SpecialTok(_Tok):
    """Word tokenizer that wraps a single text in ``specials`` extra tokens."""

    def __call__(self, text, add_special_tokens=True, truncation=False):
        ids = text.split()
        return {"input_ids": (["<s>"] * self.specials + ids) if add_special_tokens else ids}


@pytest.mark.parametrize("max_len, specials", [(16, 2), (11, 3)])
def test_a_piece_plus_the_special_tokens_never_exceeds_max_seq_length(max_len, specials):
    """What the model actually reads (piece + its special tokens) stays within bound."""
    from app.core.rag.embeddings import Embedder

    emb = Embedder("fake")
    emb._backend = "sentence_transformers"
    emb._model = _Model(max_len, specials)
    emb._model.tokenizer = _SpecialTok(specials)
    count_tokens, window = emb.token_window()
    assert window == max_len - specials
    pieces = cr.partition(PROSE + "\n\n" + TABLE, count_tokens, window)
    assert len(pieces) > 1
    model_reads = [len(emb._model.tokenizer(p)["input_ids"]) for p in pieces]
    assert max(model_reads) <= max_len
    assert all(r == count_tokens(p) + specials for r, p in zip(model_reads, pieces))


@pytest.mark.parametrize("backend", ["fake", "model2vec"])
def test_token_window_is_none_without_a_bounded_tokenizer(backend):
    from app.core.rag.embeddings import Embedder

    emb = Embedder("fake")
    emb._backend = backend
    assert emb.token_window() is None


# ── guards ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("window", [0, -3])
def test_partition_refuses_a_window_that_is_not_positive(window):
    with pytest.raises(ValueError):
        cr.partition(PROSE, words, window)


@pytest.mark.parametrize("project_id", ["", None])
def test_chunk_resplit_requires_a_project(monkeypatch, window, project_id):
    with pytest.raises(ValueError):
        cr.chunk_resplit(project_id=project_id, doc_ids=["d1"])


class _Emb:
    model_name = "stub-model"

    def __init__(self, tw):
        self.tw = tw

    def token_window(self):
        return self.tw


@pytest.mark.parametrize("tw, expected", [
    (None, None),
    ((words, 77), (words, 77, "stub-model")),
])
def test_embedder_window_reads_the_loaded_embedder(monkeypatch, tw, expected):
    import app.core.rag.embeddings as emb_mod

    monkeypatch.setattr(emb_mod, "get_embedder", lambda *a, **k: _Emb(tw))
    assert cr.embedder_window() == expected


@pytest.mark.parametrize("tw", [None])
def test_plan_refuses_without_a_tokenizer_window(monkeypatch, tw):
    monkeypatch.setattr(cr, "embedder_window", lambda: tw)
    out = cr.chunk_resplit(project_id="p", doc_ids=["d1"])
    assert out["executed"] is False and out["refused"]


# ── admin routes ─────────────────────────────────────────────────────────

ADMIN = {"role": "admin"}


@pytest.mark.parametrize("raw, ids", [("a, b ,c", ["a", "b", "c"]), ("only", ["only"])])
def test_doc_id_list_parses_the_comma_list(raw, ids):
    from app.routers import admin as admin_router

    assert admin_router._doc_id_list(raw) == ids


@pytest.mark.parametrize("raw", ["", " , ,"])
def test_doc_id_list_refuses_an_empty_list(raw):
    from fastapi import HTTPException

    from app.routers import admin as admin_router

    with pytest.raises(HTTPException) as err:
        admin_router._doc_id_list(raw)
    assert err.value.status_code == 400


@pytest.mark.parametrize("execute", [False, True])
def test_routes_turn_a_bad_request_into_400(monkeypatch, tmp_path, window, execute):
    from fastapi import HTTPException

    from app.routers import admin as admin_router

    _project(monkeypatch, tmp_path, "hproj")
    with pytest.raises(HTTPException) as err:
        if execute:
            admin_router.admin_corpus_chunk_resplit_execute(
                project_id="", doc_ids="d1", expected_plan_hash="0" * 64,
                doc_type=None, auth=ADMIN)
        else:
            admin_router.admin_corpus_chunk_resplit_plan(
                project_id="", doc_ids="d1", sample=0, auth=ADMIN)
    assert err.value.status_code == 400


@pytest.mark.parametrize("execute", [False, True])
def test_routes_return_the_plan_and_refuse_a_wrong_hash(monkeypatch, tmp_path, window, execute):
    from app.routers import admin as admin_router

    pm = _project(monkeypatch, tmp_path, "gproj")
    did = _doc(pm, "gproj", "g.pdf", [PROSE], "spec", page=1)
    before = _rows(did)
    if execute:
        out = admin_router.admin_corpus_chunk_resplit_execute(
            project_id="gproj", doc_ids=did, expected_plan_hash="0" * 64,
            doc_type=None, auth=ADMIN)
        assert out["refused"] and out["executed"] is False
    else:
        out = admin_router.admin_corpus_chunk_resplit_plan(
            project_id="gproj", doc_ids=did, sample=0, auth=ADMIN)
        assert out["plan"]["doc_ids_found"] == 1
    assert out["plan"]["tables"][out["plan"]["active_table"]]["parents_over_window"] == 1
    assert _rows(did) == before
