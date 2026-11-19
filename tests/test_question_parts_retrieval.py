"""A question that names several things is searched whole AND once per part.

Every test builds its own input: its own questions, its own store and its
own chunks. Nothing here names a corpus, a project or a probe.

* ``question_parts`` splits at generic joiners only; a question that does
  not split gives ``[]`` (and the retriever then calls the store exactly as
  before -- the byte-for-byte guard);
* ``merge_leg_lists`` gives each chunk its BEST rank over the whole question
  and its parts, so a rank can only stay or rise;
* ``VectorStore.search`` runs every part's two legs on the ``retrieval_parts``
  pool, fuses once, keeps ``k``, and records the parts in the trace summary
  as digests and leg sizes only; a failing part leaves the whole question's
  result standing.
"""

from __future__ import annotations

import os
import tempfile
import threading
import uuid

import pytest

from tests._isolated_store import give_documents_table

from app.core.rag import retriever as rt
from app.core.rag import vector_store as vs
from app.core.rag.embeddings import get_embedder, reset_embedder_cache
from app.core.rag.vector_store import Chunk, VectorStore, merge_leg_lists


def _c(cid: str, score: float = 0.5) -> Chunk:
    return Chunk(
        chunk_id=cid, project_id="p", doc_id=cid.split(":")[0], chunk_index=0,
        text=cid, score=score,
    )


# ── question_parts ────────────────────────────────────────────────────────

@pytest.mark.parametrize("question, want", [
    (
        "Compare the anchor bolt torque in the erection manual against the "
        "bolt schedule on the steel drawings",
        ["Compare the anchor bolt torque in the erection manual",
         "the bolt schedule on the steel drawings"],
    ),
    (
        "Which gravel grading does the paving specification require; which "
        "grading does the supplier submittal offer?",
        ["Which gravel grading does the paving specification require",
         "which grading does the supplier submittal offer"],
    ),
])
def test_a_question_naming_two_things_splits_into_its_parts(question, want):
    assert rt.question_parts(question) == want


@pytest.mark.parametrize("question", [
    "What is the minimum curing period for the podium slab?",
    "Salt and pepper",  # a joiner, but neither side has two content terms
])
def test_a_question_that_does_not_split_gives_no_parts(question):
    assert rt.question_parts(question) == []


@pytest.mark.parametrize("question, want", [
    (
        "roof membrane laps, gutter outlet sizes, parapet coping joints, "
        "facade sealant colours, louvre blade pitch and canopy fixing centres",
        ["roof membrane laps", "gutter outlet sizes",
         "parapet coping joints", "facade sealant colours"],
    ),
    (
        "pump duty points versus motor ratings compared with the valve "
        "schedule sizes; starter panel ratings and cable tray widths, "
        "earthing conductor sizes",
        ["pump duty points", "motor ratings", "the valve schedule sizes",
         "starter panel ratings"],
    ),
])
def test_parts_are_capped_and_kept_in_the_question_order(question, want):
    parts = rt.question_parts(question)
    assert len(parts) == vs.QUERY_PARTS_MAX == 4
    assert parts == want


@pytest.mark.parametrize("question, want", [
    ("lintel bearing lengths, lintel bearing lengths and sill drip details",
     ["lintel bearing lengths", "sill drip details"]),
    ("Kerb Joint Spacing; kerb joint spacing versus channel fall gradients",
     ["Kerb Joint Spacing", "channel fall gradients"]),
])
def test_a_repeated_part_is_searched_once(question, want):
    assert rt.question_parts(question) == want


# ── merge_leg_lists ───────────────────────────────────────────────────────

@pytest.mark.parametrize("lists, want_rank, want_order, kept", [
    (
        [["d:1", "d:2", "d:3", "d:4"], ["d:4", "e:9"], ["f:7", "d:2", "d:3"]],
        {"d:1": 1, "d:4": 1, "f:7": 1, "d:2": 2, "e:9": 2, "d:3": 3},
        ["d:1", "d:4", "f:7", "d:2", "e:9", "d:3"],
        ("d:4", 0),
    ),
    (
        [["m:5", "m:6"], ["n:1", "n:2", "m:6"]],
        {"m:5": 1, "n:1": 1, "m:6": 2, "n:2": 2},
        ["m:5", "n:1", "m:6", "n:2"],
        ("m:6", 0),
    ),
])
def test_merge_takes_each_chunks_best_rank_and_never_lowers_one(
    lists, want_rank, want_order, kept,
):
    built = [[_c(cid) for cid in lst] for lst in lists]
    merged, rank = merge_leg_lists(built)
    assert rank == want_rank
    # Every whole-question chunk keeps or improves its rank.
    for r, c in enumerate(built[0], 1):
        assert rank[c.chunk_id] <= r
    # Ties: the earlier list first (the whole question's before a part's).
    assert [c.chunk_id for c in merged] == want_order
    # The instance kept is the earliest list's (the whole question's).
    cid, li = kept
    inst = next(c for c in merged if c.chunk_id == cid)
    assert any(inst is c for c in built[li])


@pytest.mark.parametrize("ids", [["x:3", "x:1", "x:2"], ["y:9"]])
def test_merge_of_one_list_is_that_list(ids):
    whole = [_c(cid) for cid in ids]
    merged, rank = merge_leg_lists([whole])
    assert merged == whole
    assert rank == {cid: i for i, cid in enumerate(ids, 1)}


# ── the retriever passes parts only for a question that splits ───────────

class _SpyStore:
    """Records every ``search`` call's keyword names; returns nothing."""

    def __init__(self):
        self.calls = []

    def search(self, project_id, qvec, k=5, query_text=None, **kw):
        self.calls.append((project_id, query_text, sorted(kw), kw))
        return []

    def count(self, project_id):
        return 0

    def identifier_search(self, *a, **kw):
        return []

    def bm25_search(self, *a, **kw):
        return []


@pytest.fixture
def spy(monkeypatch):
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")
    reset_embedder_cache()
    store = _SpyStore()
    monkeypatch.setattr(rt, "get_store", lambda dim=None: store)
    monkeypatch.setattr(rt, "available", lambda: True)
    return store


@pytest.mark.parametrize("question", [
    "What is the minimum curing period for the podium slab?",
    "Who signs the hot works permit",
])
def test_a_question_that_does_not_split_calls_the_store_as_before(spy, question):
    rt.retrieve_with_filter(question, "proj-a", k=5)
    assert spy.calls, "the store was never searched"
    for _pid, _qt, kw_names, _kw in spy.calls:
        assert "query_parts" not in kw_names


@pytest.mark.parametrize("question, n", [
    ("Compare the anchor bolt torque in the erection manual against the "
     "bolt schedule on the steel drawings", 2),
    ("roof membrane laps, gutter outlet sizes and parapet coping joints", 3),
])
def test_a_question_that_splits_sends_its_parts_with_the_whole_search(spy, question, n):
    rt.retrieve_with_filter(question, "proj-b", k=5)
    with_parts = [kw for _p, _q, names, kw in spy.calls if "query_parts" in names]
    assert len(with_parts) == 1, "parts go with the raw question's search only"
    sent = with_parts[0]["query_parts"]
    assert [t for _v, t in sent] == rt.question_parts(question)
    assert len(sent) == n
    dim = get_embedder().dim
    assert all(len(v) == dim for v, _t in sent)


def test_the_switch_turns_parts_off(spy, monkeypatch):
    monkeypatch.setenv("RAG_QUESTION_PARTS", "off")
    rt.retrieve_with_filter(
        "roof membrane laps, gutter outlet sizes and parapet coping joints",
        "proj-c", k=5,
    )
    assert all("query_parts" not in names for _p, _q, names, _kw in spy.calls)


# ── VectorStore.search with parts (a real SQLite store) ──────────────────

_PID = "qparts_proj"
_DOCS = [
    ("manual", "anchor bolt torque erection manual tightening sequence"),
    ("schedule", "bolt schedule steel drawings grade quantities"),
    ("noise1", "canteen opening hours and parking arrangements"),
    ("noise2", "landscape planting palette irrigation drip lines"),
]


@pytest.fixture
def real_store(monkeypatch):
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_HYBRID_SEARCH", "true")
    reset_embedder_cache()
    emb = get_embedder()
    tmp = tempfile.mkdtemp(prefix="qparts_")
    path = os.path.join(tmp, f"v_{uuid.uuid4().hex}.db")
    vs.reset_store_cache()
    store = VectorStore(db_path=path, dim=emb.dim)
    give_documents_table(store)
    for doc_id, text in _DOCS:
        store.upsert_chunks(_PID, doc_id, [text], emb.encode([text]))
    yield store, emb
    store.close()


def _parts(emb, texts):
    return list(zip(emb.encode_queries(texts), texts))


@pytest.mark.parametrize("whole_text, parts", [
    ("canteen opening", ["bolt schedule steel", "anchor torque manual"]),
    ("landscape irrigation", ["steel drawings grade", "tightening sequence"]),
])
def test_parts_add_what_only_a_part_finds_and_keep_k(real_store, whole_text, parts):
    store, emb = real_store
    qv = emb.encode_queries([whole_text])[0]
    alone = store.search(_PID, qv, k=2, query_text=whole_text)
    both = store.search(_PID, qv, k=10, query_text=whole_text,
                        query_parts=_parts(emb, parts))
    got = {c.doc_id for c in both}
    assert {"manual", "schedule"} <= got, "a part's own lexical hit is in the list"
    assert {c.doc_id for c in alone} <= got, "nothing the whole question found is lost"
    capped = store.search(_PID, qv, k=2, query_text=whole_text,
                          query_parts=_parts(emb, parts))
    assert len(capped) == 2


@pytest.mark.parametrize("whole_text", ["canteen parking", "planting palette"])
def test_without_parts_the_search_is_unchanged(real_store, whole_text):
    store, emb = real_store
    qv = emb.encode_queries([whole_text])[0]
    a = store.search(_PID, qv, k=4, query_text=whole_text)
    b = store.search(_PID, qv, k=4, query_text=whole_text, query_parts=None)
    c = store.search(_PID, qv, k=4, query_text=whole_text, query_parts=[])
    key = lambda xs: [(x.chunk_id, round(x.score, 9)) for x in xs]  # noqa: E731
    assert key(a) == key(b) == key(c)


@pytest.mark.parametrize("parts", [
    ["bolt schedule steel", "anchor torque manual"],
    ["landscape planting palette", "canteen parking arrangements", "steel drawings grade"],
])
def test_the_trace_records_the_parts_as_digests_and_leg_sizes(real_store, monkeypatch, parts):
    store, emb = real_store
    monkeypatch.setattr(vs, "_retrieval_trace_on", lambda: True)
    token = vs.start_turn_memo()
    try:
        qv = emb.encode_queries(["opening hours"])[0]
        store.search(_PID, qv, k=5, query_text="opening hours",
                     query_parts=_parts(emb, parts))
        rec = vs.fusion_records(_PID)[-1]
    finally:
        vs.end_turn_memo(token)
    qp = rec["query_parts"]
    assert qp["count"] == len(parts) == len(qp["parts"])
    for row, text in zip(qp["parts"], parts):
        assert row["query"] == vs._query_ref(text)
        assert text not in repr(rec)
        assert row["vector"] >= 1 and row["lexical"] >= 1


@pytest.mark.parametrize("whole_text", ["canteen opening", "irrigation drip"])
def test_a_failing_part_leaves_the_whole_question_standing(real_store, monkeypatch, whole_text):
    store, emb = real_store
    qv = emb.encode_queries([whole_text])[0]
    key = lambda xs: [(x.chunk_id, round(x.score, 9)) for x in xs]  # noqa: E731
    before = key(store.search(_PID, qv, k=4, query_text=whole_text))
    main = threading.get_ident()
    for leg in ("_semantic_search", "bm25_search"):
        real = getattr(VectorStore, leg)

        def flaky(self, *a, _real=real, **kw):
            if threading.get_ident() != main:
                raise RuntimeError("part leg down")
            return _real(self, *a, **kw)

        monkeypatch.setattr(VectorStore, leg, flaky)
    monkeypatch.setattr(vs, "_retrieval_trace_on", lambda: True)
    token = vs.start_turn_memo()
    try:
        out = store.search(_PID, qv, k=4, query_text=whole_text,
                           query_parts=_parts(emb, ["bolt schedule steel", "anchor torque manual"]))
        rec = vs.fusion_records(_PID)[-1]
    finally:
        vs.end_turn_memo(token)
    assert key(out) == before
    for row in rec["query_parts"]["parts"]:
        assert row["vector_failed"] and row["lexical_failed"]
        assert row["vector"] == 0 and row["lexical"] == 0


@pytest.mark.parametrize("parts", [
    ["bolt schedule steel", "anchor torque manual"],
    ["landscape planting palette", "canteen parking", "steel drawings grade"],
])
def test_part_legs_run_on_their_own_pool_not_the_callers_thread(real_store, monkeypatch, parts):
    store, emb = real_store
    seen = []
    real = VectorStore.bm25_search

    def spy(self, *a, **kw):
        seen.append(threading.current_thread().name)
        return real(self, *a, **kw)

    monkeypatch.setattr(VectorStore, "bm25_search", spy)
    qv = emb.encode_queries(["canteen opening"])[0]
    store.search(_PID, qv, k=4, query_text="canteen opening",
                 query_parts=_parts(emb, parts))
    me = threading.current_thread().name
    assert seen.count(me) == 1, "the whole question's lexical leg runs in the caller"
    assert sum(n.startswith("retrieval_parts-work") for n in seen) == len(parts)
