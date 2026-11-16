"""Every retrieved chunk that was NOT handed carries the rule that dropped it,
written where that rule leaves it out; an exact score tie at the token cap is
broken neutrally, and only when the cut splits it.

Codes: ``duplicate``, ``family_share``, ``char_budget`` (the handed set,
``select_handed``), ``noise_filter``, ``score_tie_overlap``, ``token_budget``
(the token cap, ``apply_token_cap``), ``below_threshold`` (nothing handed),
else ``unexplained``.

The tie-break reads only the question and the passage text (shared words,
then chunk id). No document kind, name or type orders a tie.

Every test builds its own chunks. No real document, project, question or
value is named.
"""
from __future__ import annotations

import json
import logging

import pytest

from app.core.rag import handed_set
from app.core.rag.inject import _emit_retrieval_trace, apply_token_cap
from app.core.rag.vector_store import Chunk
from app.infra.monitoring import request_id_ctx
from tests._fixture_values import STUB_TOKEN_02

_Q = "what grade is the kerb backing"
_ON_TOPIC = "invented note kerb backing grade stated here "      # 4 question words
_OFF_TOPIC = "invented note about something else entirely now "  # 0 question words


def _chunk(doc: str, idx: int, score: float, text: str = "") -> Chunk:
    c = Chunk(chunk_id=f"p:{doc}:{idx}", project_id="p", doc_id=doc,
              chunk_index=idx, text=text or f"invented clause {doc} {idx} " * 6,
              score=score)
    c.source_name = f"invented-{doc}.pdf"
    return c


def _same_size(*texts: str) -> list[str]:
    """Pad each text to one length, so only the words differ."""
    n = max(len(t) for t in texts) + 8
    return [(t + " pad" * n)[:n] for t in texts]


def _cap_one(monkeypatch, c: Chunk) -> None:
    """MAX_RAG_TOKENS that holds exactly one chunk of ``c``'s size."""
    monkeypatch.setenv("MAX_RAG_TOKENS", str(len(c.text) // 4 + 2))


def _cut(chunks, query=_Q):
    why: dict = {}
    kept, _ = apply_token_cap(chunks, query=query, reasons=why)
    return [c.chunk_id for c in kept], why


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in ("RAG_HANDED_CHAR_BUDGET", "RAG_HANDED_POOL_FACTOR",
                 "RAG_HANDED_DOC_SHARE", "RAG_HANDED_DOC_MARGIN", "RAG_K"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MAX_RAG_TOKENS", "100000")


# --- the handed set: reasons written by select_handed ---------------------------

@pytest.mark.parametrize("doc,body", [("alpha", "invented lintel clause " * 5),
                                      ("beta", "invented sealant clause " * 7)])
def test_the_same_text_twice_in_one_document_is_a_duplicate(doc, body):
    first, again = _chunk(doc, 0, 0.8, body), _chunk(doc, 1, 0.7, body)
    why: dict = {}
    handed, _ = handed_set.select_handed([first, again], k=4, reasons=why)
    assert [c.chunk_id for c in handed] == [first.chunk_id]
    assert why == {again.chunk_id: "duplicate"}


@pytest.mark.parametrize("top,low", [(0.9, 0.5), (0.7, 0.3)])
def test_a_chunk_past_the_character_budget_is_char_budget(monkeypatch, top, low):
    a, b = _chunk("alpha", 0, top), _chunk("beta", 0, low)
    monkeypatch.setenv("RAG_HANDED_CHAR_BUDGET", str(len(a.text) + 5))
    why: dict = {}
    handed, _ = handed_set.select_handed([a, b], k=4, reasons=why)
    assert [c.chunk_id for c in handed] == [a.chunk_id]
    assert why == {b.chunk_id: "char_budget"}


@pytest.mark.parametrize("scores", [(0.90, 0.85, 0.80), (0.60, 0.58, 0.50)])
def test_a_chunk_deferred_behind_another_document_is_family_share(
        monkeypatch, scores):
    a0, a1, b0 = (_chunk("alpha", 0, scores[0]), _chunk("alpha", 1, scores[1]),
                  _chunk("beta", 0, scores[2]))
    monkeypatch.setenv("RAG_HANDED_CHAR_BUDGET", str(len(a0.text) * 2 + 5))
    why: dict = {}
    handed, _ = handed_set.select_handed([a0, a1, b0], k=2, reasons=why)
    assert [c.chunk_id for c in handed] == [a0.chunk_id, b0.chunk_id]
    assert why == {a1.chunk_id: "family_share"}


@pytest.mark.parametrize("top,low", [(0.9, 0.5), (0.7, 0.3)])
def test_select_handed_without_reasons_hands_the_same_set(monkeypatch, top, low):
    a, b = _chunk("alpha", 0, top), _chunk("beta", 0, low)
    monkeypatch.setenv("RAG_HANDED_CHAR_BUDGET", str(len(a.text) + 5))
    plain, _ = handed_set.select_handed([a, b], k=4)
    traced, _ = handed_set.select_handed([a, b], k=4, reasons={})
    assert [c.chunk_id for c in plain] == [c.chunk_id for c in traced] == [a.chunk_id]


# --- the token cap: different scores --------------------------------------------

@pytest.mark.parametrize("top,low", [(0.9, 0.5), (0.62, 0.61)])
def test_a_lower_score_past_the_token_cap_is_token_budget(monkeypatch, top, low):
    a, b = _chunk("alpha", 0, top), _chunk("beta", 0, low)
    _cap_one(monkeypatch, a)
    assert _cut([a, b]) == ([a.chunk_id], {b.chunk_id: "token_budget"})


@pytest.mark.parametrize("top_text,low_text", [
    (_OFF_TOPIC, _ON_TOPIC), (_OFF_TOPIC + "x", _ON_TOPIC + "y")])
def test_a_higher_score_is_never_overridden_by_word_overlap(
        monkeypatch, top_text, low_text):
    hi_t, lo_t = _same_size(top_text, low_text)
    hi, lo = _chunk("alpha", 0, 0.62, hi_t), _chunk("beta", 0, 0.61, lo_t)
    _cap_one(monkeypatch, hi)
    assert _cut([hi, lo]) == ([hi.chunk_id], {lo.chunk_id: "token_budget"})


# --- the token cap: an exact tie the cut splits ---------------------------------

@pytest.mark.parametrize("first_text,second_text,winner", [
    (_OFF_TOPIC, _ON_TOPIC, 1),   # the on-topic passage is second: moved up
    (_ON_TOPIC, _OFF_TOPIC, 0),   # the on-topic passage is first: stays
])
def test_a_split_tie_hands_the_passage_sharing_more_question_words(
        monkeypatch, first_text, second_text, winner):
    t1, t2 = _same_size(first_text, second_text)
    pair = [_chunk("alpha", 0, 0.61, t1), _chunk("beta", 0, 0.61, t2)]
    _cap_one(monkeypatch, pair[0])
    kept, why = _cut(pair)
    assert kept == [pair[winner].chunk_id]
    assert why == {pair[1 - winner].chunk_id: "score_tie_overlap"}


@pytest.mark.parametrize("ids", [("beta", "alpha"), ("delta", "charlie")])
def test_equal_overlap_falls_back_to_the_chunk_id(monkeypatch, ids):
    t1, t2 = _same_size(_ON_TOPIC, _ON_TOPIC + "z")
    first, second = _chunk(ids[0], 0, 0.61, t1), _chunk(ids[1], 0, 0.61, t2)
    _cap_one(monkeypatch, first)
    kept, why = _cut([first, second])
    assert kept == [second.chunk_id]  # the lower chunk id, not retrieval order
    assert why == {first.chunk_id: "score_tie_overlap"}


@pytest.mark.parametrize("first_kind,second_kind", [
    ("schedule", "drawing"), ("drawing", "schedule")])
def test_the_document_kind_never_orders_a_tie(monkeypatch, first_kind, second_kind):
    """The same two passages, with their kinds swapped: the passage sharing
    more question words is handed both times."""
    t1, t2 = _same_size(_OFF_TOPIC, _ON_TOPIC)
    a, b = _chunk("alpha", 0, 0.61, t1), _chunk("beta", 0, 0.61, t2)
    a.doc_type, b.doc_type = first_kind, second_kind
    _cap_one(monkeypatch, a)
    assert _cut([a, b]) == ([b.chunk_id], {a.chunk_id: "score_tie_overlap"})


@pytest.mark.parametrize("first_text,second_text", [
    (_OFF_TOPIC, _ON_TOPIC), (_ON_TOPIC, _OFF_TOPIC)])
def test_a_tie_that_all_fits_keeps_its_retrieval_order(first_text, second_text):
    """No cap pressure, no reorder: the order handed is the retrieval order
    even when the second passage shares more question words."""
    t1, t2 = _same_size(first_text, second_text)
    a, b = _chunk("alpha", 0, 0.61, t1), _chunk("beta", 0, 0.61, t2)
    assert _cut([a, b]) == ([a.chunk_id, b.chunk_id], {})


@pytest.mark.parametrize("extra", [0, 40])
def test_a_tie_dropped_whole_is_token_budget_and_keeps_order(monkeypatch, extra):
    """A tie where nothing fits is not decided by the tie: every member is
    token_budget, none is reordered."""
    top = _chunk("top", 0, 0.9)
    t1, t2 = _same_size(_OFF_TOPIC * 3 + "q" * extra, _ON_TOPIC * 3)
    a, b = _chunk("alpha", 0, 0.5, t1), _chunk("beta", 0, 0.5, t2)
    _cap_one(monkeypatch, top)
    assert _cut([top, a, b]) == (
        [top.chunk_id], {a.chunk_id: "token_budget", b.chunk_id: "token_budget"})


# --- size versus tie -------------------------------------------------------------

@pytest.mark.parametrize("big_first", [True, False])
def test_a_tied_passage_too_big_for_the_room_is_token_budget_not_a_tie(
        monkeypatch, big_first):
    """A large tied passage left out because it does not fit the room left,
    while a smaller tied one does, lost on size: ``token_budget``."""
    small = _chunk("small", 0, 0.61, _OFF_TOPIC)
    big = _chunk("big", 0, 0.61, _ON_TOPIC * 30)
    _cap_one(monkeypatch, small)
    pair = [big, small] if big_first else [small, big]
    assert _cut(pair) == ([small.chunk_id], {big.chunk_id: "token_budget"})


# --- the noise filter, at its decision point --------------------------------------

@pytest.mark.parametrize("noise_at", [0, 1])
def test_a_chunk_the_noise_filter_removed_is_noise_filter(monkeypatch, noise_at):
    from app.core.rag import retriever
    operand = _chunk("op", 0, 0.9, "invented operand clause")
    others = [_chunk("alpha", 0, 0.5, "invented one"), _chunk("beta", 0, 0.4, "invented two")]
    noise = others[noise_at]
    monkeypatch.setattr(retriever, "query_asks_delay_damages_daily_amount", lambda q: True)
    monkeypatch.setattr(retriever, "_daily_rate_preference",
                        lambda t: 2 if t == operand.text else 0)
    monkeypatch.setattr(retriever, "_has_standalone_excl_vat_aca", lambda t: False)
    monkeypatch.setattr(retriever, "_is_daily_damages_cap_noise",
                        lambda t: t == noise.text)
    kept, why = _cut([operand, *others], query="invented daily amount ask")
    assert noise.chunk_id not in kept and operand.chunk_id in kept
    assert why == {noise.chunk_id: "noise_filter"}


# --- the trace line ----------------------------------------------------------------

def _trace(caplog, *, retrieved, handed, threshold_fired, drop_reasons):
    with caplog.at_level(logging.INFO, logger="app.rag.retrieval_trace"):
        _emit_retrieval_trace(
            query_label="user_message", project_id=None, conversation_id=None,
            agent_name="t", retrieved=retrieved, handed=handed,
            threshold_fired=threshold_fired, identifier_miss=False,
            top_score=0.5, drop_reasons=drop_reasons,
        )
    line = next(r.getMessage() for r in caplog.records
                if "rag_retrieval_trace" in r.getMessage())
    return json.loads(line.split(" ", 1)[1])


@pytest.mark.parametrize("fired,expected", [(True, "below_threshold"),
                                            (False, "unexplained")])
def test_a_chunk_with_no_recorded_rule_is_never_left_blank(
        monkeypatch, caplog, fired, expected):
    monkeypatch.setenv("RAG_RETRIEVAL_TRACE", "1")
    a, b = _chunk("alpha", 0, 0.9), _chunk("beta", 0, 0.5)
    line = _trace(caplog, retrieved=[a, b], handed=[], threshold_fired=fired,
                  drop_reasons=None)
    assert [e["dropped_by"] for e in line["not_handed"]] == [expected, expected]


# --- end to end: rag_inject -> the emitted trace -----------------------------------

def _inject_trace(monkeypatch, caplog, pool, ask=_Q):
    from app.core.rag import inject
    monkeypatch.setenv("RAG_RETRIEVAL_TRACE", "1")
    monkeypatch.setenv("RAG_CONFIDENCE_THRESHOLD", "0.4")
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_02)
    monkeypatch.setattr(inject, "retrieve_with_filter",
                        lambda query, project_id, k=5, **kw: (list(pool)[:k], 0))
    monkeypatch.setattr(inject._budget, "snapshot", lambda day: {
        "day": day, "consumed": 9, "budget": 10, "remaining": 1, "degraded": False})
    monkeypatch.setattr(inject._budget, "consume", lambda **kw: None)
    token = request_id_ctx.set("rid-drop")
    try:
        with caplog.at_level(logging.INFO, logger="app.rag.retrieval_trace"):
            _msg, audit = inject.rag_inject(
                user_message=ask, project_id="p", conversation_id="ws-p-1",
                user_id="u", agent_name="project-assistant",
            )
    finally:
        request_id_ctx.reset(token)
    lines = [r.getMessage() for r in caplog.records
             if r.getMessage().startswith("rag_retrieval_trace")]
    assert len(lines) == 1
    line = json.loads(lines[0].split(" ", 1)[1])
    return audit, line


@pytest.mark.parametrize("on_topic_second", [True, False])
def test_end_to_end_a_split_tie_is_named_in_the_trace(
        monkeypatch, caplog, on_topic_second):
    texts = _same_size(_OFF_TOPIC, _ON_TOPIC)
    if not on_topic_second:
        texts.reverse()
    a, b = _chunk("alpha", 0, 0.61, texts[0]), _chunk("beta", 0, 0.61, texts[1])
    on_topic, off_topic = (b, a) if on_topic_second else (a, b)
    _cap_one(monkeypatch, a)
    audit, line = _inject_trace(monkeypatch, caplog, [a, b])
    assert [c["chunk_id"] for c in audit["chunks"]] == [on_topic.chunk_id]
    assert [c["chunk_id"] for c in line["chunks"]] == [on_topic.chunk_id]
    assert [(e["chunk_id"], e["dropped_by"]) for e in line["not_handed"]] == [
        (off_topic.chunk_id, "score_tie_overlap")]


@pytest.mark.parametrize("scores", [(0.90, 0.85, 0.80), (0.60, 0.58, 0.50)])
def test_end_to_end_handed_set_and_cap_reasons_reach_the_trace(
        monkeypatch, caplog, scores):
    a0, dup, b0 = (_chunk("alpha", 0, scores[0]),
                   _chunk("alpha", 1, scores[1]), _chunk("beta", 0, scores[2]))
    dup.text = a0.text
    monkeypatch.setenv("RAG_HANDED_CHAR_BUDGET", str(len(a0.text) + 5))
    _audit, line = _inject_trace(monkeypatch, caplog, [a0, dup, b0])
    assert [c["chunk_id"] for c in line["chunks"]] == [a0.chunk_id]
    assert {e["chunk_id"]: e["dropped_by"] for e in line["not_handed"]} == {
        dup.chunk_id: "duplicate", b0.chunk_id: "char_budget"}
