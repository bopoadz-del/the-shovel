"""S-R7b: the pre_injection trace carries a summary of the fused candidates.

Reciprocal-rank fusion with k=60 ranks a chunk found by BOTH legs above a
rank-1 chunk found by one leg (2/(60+r) > 1/61 for every r <= 50 per leg),
so the best candidate only one leg found can sit below every agreeing
candidate. The trace now shows that: how many candidates each leg found
alone, and where the best lexical-only and best vector-only candidate ended
up (fused rank, scored position, pool position, handed position, or the step
that dropped it). Counts and ranks only; no ranking changes.

Every input is constructed here: invented texts, ids and scores. The real
``VectorStore.search`` and ``_rrf_combine`` run; only the two legs and the
side lookups are stubbed. The helpers are the S-R7 fixture's.
"""
from __future__ import annotations

import json
import logging

import pytest

from app.core.rag import fusion_trace as ft
from app.core.rag import retriever
from app.core.rag import vector_store as vs
from app.core.rag.inject import _emit_retrieval_trace
from app.core.rag.vector_store import Chunk
from tests.test_fused_rank_is_chunk_score import (
    PID, QUERY, _chunk, _ids, _install, _vector_leg,
)


@pytest.fixture
def turn():
    token = vs.start_turn_memo()
    try:
        yield
    finally:
        vs.end_turn_memo(token)


def _as_lexical(c: Chunk, score: float) -> Chunk:
    return Chunk(**{**c.__dict__, "score": score})


def _legs(n_both: int, *, lexical_only_first: bool = True):
    """50 vector hits; the lexical leg returns one lexical-only chunk ``x``
    at rank 1 followed by the first ``n_both`` vector hits."""
    semantic = _vector_leg(50)
    x = _chunk("x", 0.09, kind="lexical", n=101)
    agree = [_as_lexical(c, round(0.08 - i * 0.001, 6))
             for i, c in enumerate(semantic[:n_both])]
    lexical = [x] + agree if lexical_only_first else agree + [x]
    return semantic, lexical


# ── 1. counts, and the lexical-only rank-1 hit buried under agreement ──────

@pytest.mark.parametrize("n_both", [10, 15])
def test_lexical_only_rank_one_is_counted_and_followed(
        monkeypatch, tmp_path, turn, n_both):
    # S-R7b arithmetic before the S-R7c leg share: share 0 is the fused order.
    monkeypatch.setenv("RAG_FUSION_LEG_SHARE", "0")
    semantic, lexical = _legs(n_both)
    _install(monkeypatch, tmp_path, semantic, lexical)

    pooled, _ = retriever.retrieve_with_filter(QUERY, PID, k=10)
    rec = vs.fused_candidates_record(PID)

    assert rec["n_candidates"] == 51
    assert rec["n_both_legs"] == n_both
    assert rec["n_lexical_only"] == 1
    assert rec["n_vector_only"] == 50 - n_both
    lex = rec["best_lexical_only"]
    # Every agreeing candidate outranks the lexical leg's rank-1 hit.
    assert lex["leg_rank"] == 1
    assert lex["fused_rank"] == n_both + 1
    assert lex["in_fused_list"] is True
    vec = rec["best_vector_only"]
    assert vec["leg_rank"] == n_both + 1
    assert vec["fused_rank"] == n_both + 2
    # Its fate is stated, and agrees with the pool the retriever returned.
    assert "x" not in _ids(pooled)
    assert lex["pool_position"] is None
    assert lex["scored_position"] > len(pooled)
    assert lex["dropped_by"] == ft.BELOW_TARGET


# ── 2. legs that agree: no single-leg candidate, the pool is unchanged ─────

@pytest.mark.parametrize("count", [20, 7])
def test_agreeing_legs_trace_no_single_leg_and_same_pool(
        monkeypatch, tmp_path, count):
    semantic = _vector_leg(count)
    lexical = [_as_lexical(c, round(0.09 - i * 0.001, 6))
               for i, c in enumerate(semantic)]
    _install(monkeypatch, tmp_path, semantic, lexical)

    outside, _ = retriever.retrieve_with_filter(QUERY, PID, k=10)
    token = vs.start_turn_memo()
    try:
        inside, _ = retriever.retrieve_with_filter(QUERY, PID, k=10)
        rec = vs.fused_candidates_record(PID)
    finally:
        vs.end_turn_memo(token)

    assert _ids(inside) == _ids(outside)
    assert [c.score for c in inside] == [c.score for c in outside]
    assert rec["n_both_legs"] == count == rec["n_candidates"]
    assert rec["n_lexical_only"] == rec["n_vector_only"] == 0
    assert rec["best_lexical_only"] is None
    assert rec["best_vector_only"] is None


# ── 3. tracing changes no ranking when a single-leg hit exists ─────────────

@pytest.mark.parametrize("n_both", [0, 12])
def test_trace_on_and_off_give_the_same_pool(monkeypatch, tmp_path, n_both):
    # S-R7b arithmetic before the S-R7c leg share: share 0 is the fused order.
    monkeypatch.setenv("RAG_FUSION_LEG_SHARE", "0")
    semantic, lexical = _legs(n_both)
    _install(monkeypatch, tmp_path, semantic, lexical)

    outside, _ = retriever.retrieve_with_filter(QUERY, PID, k=10)
    token = vs.start_turn_memo()
    try:
        inside, _ = retriever.retrieve_with_filter(QUERY, PID, k=10)
        rec = vs.fused_candidates_record(PID)
    finally:
        vs.end_turn_memo(token)
    assert _ids(inside) == _ids(outside)
    assert rec["best_lexical_only"]["fused_rank"] == n_both + 1


# ── 3b. with the leg share on, the trace follows the order that is cut ─────

@pytest.mark.parametrize("n_both", [10, 15])
def test_trace_follows_the_order_after_the_leg_share(
        monkeypatch, tmp_path, turn, n_both):
    """S-R7c places the lexical leg's rank-1 hit at its own leg rank before
    the cut; the trace reports that position, not the pre-share one."""
    monkeypatch.setenv("RAG_FUSION_LEG_SHARE", "2")
    semantic, lexical = _legs(n_both)
    _install(monkeypatch, tmp_path, semantic, lexical)

    pooled, _ = retriever.retrieve_with_filter(QUERY, PID, k=10)
    rec = vs.fused_candidates_record(PID)

    lex = rec["best_lexical_only"]
    assert lex["leg_rank"] == 1
    assert lex["fused_rank"] == 1 < n_both + 1
    assert lex["in_fused_list"] is True
    assert "x" in _ids(pooled)
    assert lex["pool_position"] == _ids(pooled).index("x") + 1
    assert lex["dropped_by"] is None


# ── 4. a keep-loop drop is named ───────────────────────────────────────────

@pytest.mark.parametrize("n_both", [3, 8])
def test_keep_loop_drop_reason_is_recorded(monkeypatch, tmp_path, turn, n_both):
    semantic, lexical = _legs(n_both)
    # The default noise pattern matches an Office lock-file name.
    lexical[0] = Chunk(**{**lexical[0].__dict__, "doc_id": "~$lock"})
    _install(monkeypatch, tmp_path, semantic, lexical)

    pooled, _ = retriever.retrieve_with_filter(QUERY, PID, k=30)
    lex = vs.fused_candidates_record(PID)["best_lexical_only"]
    assert "x" not in _ids(pooled)
    assert lex["scored_position"] is not None
    assert lex["dropped_by"] == "noise_filename"


# ── 5. the fusion summary and the follow, on their own ─────────────────────

class _C:
    def __init__(self, cid):
        self.chunk_id = cid


@pytest.mark.parametrize("fused_k", [2, 3])
def test_best_single_leg_past_the_fused_cut_is_overfetch_cut(fused_k):
    order = [_C("b1"), _C("b2"), _C("b3"), _C("lx"), _C("vx")]
    sem = {"b1": 1, "b2": 2, "b3": 3, "vx": 4}
    bm = {"b1": 2, "b2": 3, "b3": 4, "lx": 1}
    summary = ft.leg_summary(order, sem, bm, fused_k)
    assert (summary["n_both_legs"], summary["n_lexical_only"],
            summary["n_vector_only"]) == (3, 1, 1)
    out = ft.Follow(summary).finish(order[:fused_k], order[:fused_k], order[:1])
    for leg, rank in (("lexical_only", 4), ("vector_only", 5)):
        entry = out[f"best_{leg}"]
        assert entry["fused_rank"] == rank
        assert entry["in_fused_list"] is False
        assert entry["dropped_by"] == ft.OVERFETCH_CUT


@pytest.mark.parametrize("handed_has_it", [True, False])
def test_public_view_has_no_chunk_id_and_states_handed(handed_has_it):
    order = [_C("b1"), _C("lx")]
    summary = ft.leg_summary(order, {"b1": 1}, {"b1": 1, "lx": 1}, 10)
    rec = ft.Follow(summary).finish(order, order, order)
    handed = order if handed_has_it else order[:1]
    view = ft.public_view(rec, handed)
    lex = view["best_lexical_only"]
    assert "chunk_id" not in lex
    assert lex["pool_position"] == 2
    if handed_has_it:
        assert lex["handed_position"] == 2 and lex["dropped_by"] is None
    else:
        assert lex["handed_position"] is None
        assert lex["dropped_by"] == ft.NOT_HANDED
    assert view["best_vector_only"] is None


@pytest.mark.parametrize("reason_order", [("rerank",), ("below",)])
def test_follow_tells_rerank_cut_from_below_target(reason_order):
    order = [_C("b1"), _C("lx")]
    summary = ft.leg_summary(order, {"b1": 1}, {"b1": 1, "lx": 1}, 10)
    loop = order if reason_order == ("rerank",) else order[:1]
    out = ft.Follow(summary).finish(order, loop, order[:1])
    expected = ft.RERANK_CUT if reason_order == ("rerank",) else ft.BELOW_TARGET
    assert out["best_lexical_only"]["dropped_by"] == expected


# ── 6. the pre_injection line carries it ───────────────────────────────────

@pytest.mark.parametrize("n_both", [10, 14])
def test_pre_injection_line_carries_fused_candidates(
        monkeypatch, tmp_path, turn, caplog, n_both):
    # S-R7b arithmetic before the S-R7c leg share: share 0 is the fused order.
    monkeypatch.setenv("RAG_FUSION_LEG_SHARE", "0")
    semantic, lexical = _legs(n_both)
    _install(monkeypatch, tmp_path, semantic, lexical)
    monkeypatch.setenv("RAG_RETRIEVAL_TRACE", "1")
    pooled, _ = retriever.retrieve_with_filter(QUERY, PID, k=10)

    with caplog.at_level(logging.INFO, logger="app.rag.retrieval_trace"):
        _emit_retrieval_trace(
            query_label="user_message", project_id=PID,
            conversation_id=None, agent_name="t", retrieved=pooled,
            handed=pooled[:5], threshold_fired=False, identifier_miss=False,
            top_score=0.8,
        )
    line = next(r.getMessage() for r in caplog.records
                if "rag_retrieval_trace" in r.getMessage())
    payload = json.loads(line.split(" ", 1)[1])
    fc = payload["fused_candidates"]
    assert fc["n_both_legs"] == n_both
    assert fc["n_lexical_only"] == 1
    assert fc["best_lexical_only"]["fused_rank"] == n_both + 1
    assert fc["best_lexical_only"]["handed_position"] is None
    assert "chunk_id" not in json.dumps(fc)


@pytest.mark.parametrize("handed", [[], [_C("b1")]])
def test_no_fusion_traces_none(handed):
    """Vector-only degrade or the lexical-only path: no fusion ran."""
    follow = ft.Follow(None)
    follow.drop(_C("b1"), "noise_filename")
    assert follow.finish(handed, handed, handed) is None
    assert ft.public_view(None, handed) is None


# ── 7. the turn-memo accessors ─────────────────────────────────────────────

@pytest.mark.parametrize("summaries", [[{"n_candidates": 3}],
                                       [{"n_candidates": 1}, {"n_candidates": 9}]])
def test_fusion_records_returns_the_turns_summaries_in_order(turn, summaries):
    assert vs.fusion_records(PID) == []
    for s in summaries:
        vs.record_fusion(PID, s)
    assert vs.fusion_records(PID) == summaries
    assert vs.fusion_records("another-project") == []


@pytest.mark.parametrize("record", [{"n_candidates": 4}, None])
def test_record_fused_candidates_is_read_back_and_overwritten(turn, record):
    vs.record_fused_candidates(PID, {"n_candidates": 99})
    vs.record_fused_candidates(PID, record)
    assert vs.fused_candidates_record(PID) == record


# ── leg_top: each leg's own top ranks, for an offline replay ───────────────


class _T:
    def __init__(self, cid, text):
        self.chunk_id = cid
        self.text = text


@pytest.mark.parametrize("sem, bm, depth, want_vec, want_lex", [
    # The vector leg's #1 is found by the lexical leg only at #9.
    ({"v1": 1, "b1": 2, "v3": 3}, {"x1": 1, "b1": 2, "v1": 9}, 2,
     [("v1", 1, 9), ("b1", 2, 2)], [("x1", 1, None), ("b1", 2, 2)]),
    # Depth 1, and a leg rank absent from the order is not traced.
    ({"gone": 1, "b1": 2}, {"b1": 1}, 1, [], [("b1", 1, 2)]),
])
def test_leg_top_lists_each_legs_own_ranks(sem, bm, depth, want_vec, want_lex):
    ids = list(dict.fromkeys([*bm, *sem]))
    order = [_T(c, "w" * (10 * (i + 1))) for i, c in enumerate(ids) if c != "gone"]
    top = ft.leg_top(order, sem, bm, depth)
    for name, want in (("vector", want_vec), ("lexical", want_lex)):
        got = [(e["chunk_id"], e["leg_rank"], e["other_leg_rank"]) for e in top[name]]
        assert got == want
        for e in top[name]:
            pos = [c.chunk_id for c in order].index(e["chunk_id"])
            assert e["fused_rank"] == pos + 1
            assert e["chars"] == len(order[pos].text)


@pytest.mark.parametrize("handed_n", [1, 2])
def test_leg_top_public_view_states_pool_and_handed_without_ids(handed_n):
    order = [_T("b1", "aaaa"), _T("lx", "bb"), _T("v9", "c")]
    summary = ft.leg_summary(order, {"b1": 1, "v9": 2}, {"b1": 2, "lx": 1}, 10)
    rec = ft.Follow(summary).finish(order, order, order[:2])
    view = ft.public_view(rec, order[:handed_n])
    lex = view["leg_top"]["lexical"]
    assert all("chunk_id" not in e for es in view["leg_top"].values() for e in es)
    assert [e["leg_rank"] for e in lex] == [1, 2]
    assert [e["pool_position"] for e in lex] == [2, 1]
    assert [e["handed_position"] for e in lex] == [2 if handed_n == 2 else None, 1]
    assert [e["pool_position"] for e in view["leg_top"]["vector"]] == [1, None]


@pytest.mark.parametrize("text", ["abc", "x" * 3001])
def test_pre_injection_chunk_carries_its_length_not_its_text(text):
    from app.core.rag.inject import _trace_chunk

    from types import SimpleNamespace

    c = SimpleNamespace(chunk_id="c1", doc_id="d1", text=text, score=0.5,
                        source_name="n.pdf")
    entry = _trace_chunk(1, c)
    assert entry["chars"] == len(text)
    assert text not in json.dumps(entry)


# ── leg_docs: each leg folded to documents (measurement only) ─────────────

class _D:
    def __init__(self, cid, doc):
        self.chunk_id = cid
        self.doc_id = doc
        self.text = "t"


@pytest.mark.parametrize("sem, bm, want_vec, want_lex", [
    # one document fills the lexical head; the other appears only in vector
    ({"a1": 1, "b1": 2}, {"a1": 3, "a2": 1, "a3": 2},
     [("A", 1, 1), ("B", 2, 1)], [("A", 1, 3)]),
    # the same chunks, ranks moved: best rank and order follow the leg
    ({"a1": 2, "b1": 1}, {"a1": 1, "a2": 4, "b1": 2},
     [("B", 1, 1), ("A", 2, 1)], [("A", 1, 2), ("B", 2, 1)]),
])
def test_leg_docs_folds_each_leg_to_best_rank_and_count(sem, bm, want_vec, want_lex):
    docs = {"a1": "A", "a2": "A", "a3": "A", "b1": "B"}
    order = [_D(cid, d) for cid, d in docs.items()]
    out = ft.leg_docs(order, sem, bm)
    for name, want in (("vector", want_vec), ("lexical", want_lex)):
        got = [(e["doc_id"], e["best_leg_rank"], e["n_chunks"]) for e in out[name]]
        assert got == want


@pytest.mark.parametrize("in_lexical", [True, False])
def test_leg_docs_places_a_document_or_states_it_absent(in_lexical):
    order = [_D("a1", "A"), _D("z9", "Z")]
    bm = {"a1": 1, **({"z9": 40} if in_lexical else {})}
    view = ft.public_view(
        ft.Follow(ft.leg_summary(order, {"a1": 1}, bm, 10)).finish(order, order, order),
        order,
    )
    lex = {e["doc_id"]: e["best_leg_rank"] for e in view["leg_docs"]["lexical"]}
    vec = {e["doc_id"] for e in view["leg_docs"]["vector"]}
    assert ("Z" in lex) is in_lexical
    if in_lexical:
        assert lex["Z"] == 40
    assert "Z" not in vec
    assert all("chunk_id" not in e for es in view["leg_docs"].values() for e in es)


@pytest.mark.parametrize("orphan_in_order", [False, True])
def test_leg_docs_never_counts_a_chunk_it_cannot_tie_to_a_document(orphan_in_order):
    # a leg rank for a chunk the candidate list cannot place (absent from
    # it, or present with no doc_id) is not a document: it is not listed
    # and adds to no document's count
    order = [_D("a1", "A")] + ([_D("x7", None)] if orphan_in_order else [])
    out = ft.leg_docs(order, {"x7": 1, "a1": 2}, {"a1": 1, "x7": 2})
    for name, rank in (("vector", 2), ("lexical", 1)):
        got = [(e["doc_id"], e["best_leg_rank"], e["n_chunks"]) for e in out[name]]
        assert got == [("A", rank, 1)]


# ── leg_docs cap and per-leg total ───────────────────────────────────────

def _many_docs(n_docs: int):
    """``n_docs`` documents, one chunk each, ranked 1..n in both legs in
    opposite orders."""
    order = [_D(f"c{i}", f"D{i:03d}") for i in range(1, n_docs + 1)]
    sem = {f"c{i}": i for i in range(1, n_docs + 1)}
    bm = {f"c{i}": n_docs + 1 - i for i in range(1, n_docs + 1)}
    return order, sem, bm


@pytest.mark.parametrize("n_docs", [ft.LEG_DOCS_CAP - 1, ft.LEG_DOCS_CAP,
                                    ft.LEG_DOCS_CAP + 1, 3 * ft.LEG_DOCS_CAP])
def test_leg_docs_is_capped_best_rank_first_and_states_the_total(n_docs):
    order, sem, bm = _many_docs(n_docs)
    summary = ft.leg_summary(order, sem, bm, 10)
    listed = min(n_docs, ft.LEG_DOCS_CAP)
    assert summary["leg_docs_total"] == {"vector": n_docs, "lexical": n_docs}
    for name, mine in (("vector", sem), ("lexical", bm)):
        got = [e["best_leg_rank"] for e in summary["leg_docs"][name]]
        assert got == list(range(1, listed + 1))
    # the vector leg keeps D001.. ; the lexical leg keeps the far end
    assert summary["leg_docs"]["vector"][0]["doc_id"] == "D001"
    assert summary["leg_docs"]["lexical"][0]["doc_id"] == f"D{n_docs:03d}"
    # the total survives the public view, so a shortened list is visible
    view = ft.public_view(summary, order)
    assert view["leg_docs_total"] == {"vector": n_docs, "lexical": n_docs}
    assert len(view["leg_docs"]["vector"]) == listed


@pytest.mark.parametrize("cap, want", [(None, 5), (2, 2)])
def test_leg_docs_cap_argument_bounds_each_leg(cap, want):
    order, sem, bm = _many_docs(5)
    out = ft.leg_docs(order, sem, bm, cap=cap)
    assert [len(out["vector"]), len(out["lexical"])] == [want, want]


# ── an id that is also its document's stored name is digested ────────────

@pytest.mark.parametrize("stored_name, digested", [
    ("A", True),           # parent-row fallback: original_name == id
    ("spec.pdf", False),   # an uploaded document: the id is not a name
])
def test_leg_docs_digest_an_id_that_is_its_own_name(stored_name, digested):
    order = [_D("a1", "A")]
    summary = ft.leg_summary(order, {"a1": 1}, {"a1": 1}, 10)
    view = ft.public_view(summary, order, {"A": stored_name})
    entry = view["leg_docs"]["vector"][0]
    if digested:
        assert "doc_id" not in entry
        assert entry["doc_digest"] == ft.doc_digest("A")
        assert "A" not in json.dumps(view["leg_docs"])
    else:
        assert entry["doc_id"] == "A" and "doc_digest" not in entry
    # the record itself is not changed by the view
    assert summary["leg_docs"]["vector"][0]["doc_id"] == "A"


@pytest.mark.parametrize("read_fails", [False, True])
def test_trace_name_read_failure_digests_every_listed_id(monkeypatch, read_fails):
    from app.core import projects
    from app.core.rag import inject

    def _names(ids):
        if read_fails:
            raise RuntimeError("db down")
        return {d: "upload.pdf" for d in ids}

    monkeypatch.setattr(projects, "document_names", _names)
    order = [_D("a1", "A")]
    summary = ft.leg_summary(order, {"a1": 1}, {"a1": 1}, 10)
    entry = inject._public_fused(summary, order)["leg_docs"]["lexical"][0]
    assert ("doc_digest" in entry) is read_fails
    assert ("doc_id" in entry) is (not read_fails)


@pytest.mark.parametrize("doc_id", ["A", "spec-rev-B.pdf"])
def test_doc_digest_is_the_ids_own_sha256_prefix(doc_id):
    import hashlib

    want = hashlib.sha256(doc_id.encode("utf-8")).hexdigest()[:ft.DOC_DIGEST_HEX]
    assert ft.doc_digest(doc_id) == want
    # two different ids never share a digest here: the digest reads the id
    other = doc_id + "x"
    assert ft.doc_digest(other) != ft.doc_digest(doc_id)
