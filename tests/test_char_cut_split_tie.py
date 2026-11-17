"""An exact score tie split by the handed set's CHARACTER budget is broken by
the same neutral rule as at the token cap: the question's own words in the
passage text, then the chunk id. Only when the character walk keeps one
member of the tie and skips another; never across a family-share deferral;
never over a higher score. A tied passage that would have fit alone is named
``score_tie_overlap``; one too big for the room stays ``char_budget``.

Every test builds its own chunks. No real document, project, question or
value is named; no document kind is read.
"""
from __future__ import annotations

import pytest

from app.core.rag import handed_set
from app.core.rag.inject import tie_break_key, _terms
from app.core.rag.vector_store import Chunk
from tests.test_trace_drop_reasons import (
    _OFF_TOPIC, _ON_TOPIC, _Q, _chunk, _inject_trace, _same_size,
)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in ("RAG_HANDED_CHAR_BUDGET", "RAG_HANDED_POOL_FACTOR",
                 "RAG_HANDED_DOC_SHARE", "RAG_HANDED_DOC_MARGIN", "RAG_K"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MAX_RAG_TOKENS", "100000")


def _key(query: str = _Q):
    terms = _terms(query)
    return lambda c: tie_break_key(c, terms)


def _select(chunks, *, k=8, query=_Q):
    why: dict = {}
    handed, _ = handed_set.select_handed(chunks, k=k, reasons=why,
                                         tie_key=_key(query))
    return [c.chunk_id for c in handed], why


def _lead_and_tie(first_text: str, second_text: str):
    """A higher-scored lead passage, then two same-size passages tied below it."""
    t1, t2 = _same_size(first_text, second_text)
    lead = _chunk("lead", 0, 0.90, "invented lead clause " * 4)
    pair = [_chunk("alpha", 0, 0.73, t1), _chunk("beta", 0, 0.73, t2)]
    return lead, pair


def _room_for_lead_and_one(monkeypatch, lead: Chunk, one: Chunk) -> None:
    monkeypatch.setenv("RAG_HANDED_CHAR_BUDGET", str(len(lead.text) + len(one.text) + 3))


@pytest.mark.parametrize("first_text,second_text,winner", [
    (_OFF_TOPIC, _ON_TOPIC, 1),   # the on-topic passage is second: moved up
    (_ON_TOPIC, _OFF_TOPIC, 0),   # the on-topic passage is first: stays
])
def test_a_tie_split_by_the_character_budget_hands_the_on_topic_passage(
        monkeypatch, first_text, second_text, winner):
    lead, pair = _lead_and_tie(first_text, second_text)
    _room_for_lead_and_one(monkeypatch, lead, pair[0])
    handed, why = _select([lead, *pair])
    assert handed == [lead.chunk_id, pair[winner].chunk_id]
    assert why == {pair[1 - winner].chunk_id: "score_tie_overlap"}


@pytest.mark.parametrize("first_text,second_text", [
    (_OFF_TOPIC, _ON_TOPIC), (_ON_TOPIC, _OFF_TOPIC)])
def test_a_tie_that_all_fits_keeps_its_retrieval_order(first_text, second_text):
    lead, pair = _lead_and_tie(first_text, second_text)
    handed, why = _select([lead, *pair])
    assert handed == [lead.chunk_id, pair[0].chunk_id, pair[1].chunk_id]
    assert why == {}


@pytest.mark.parametrize("top,low", [(0.90, 0.50), (0.74, 0.73)])
def test_a_higher_score_is_never_overridden_at_the_character_cut(
        monkeypatch, top, low):
    t_top, t_low = _same_size(_OFF_TOPIC, _ON_TOPIC)
    a, b = _chunk("alpha", 0, top, t_top), _chunk("beta", 0, low, t_low)
    monkeypatch.setenv("RAG_HANDED_CHAR_BUDGET", str(len(a.text) + 3))
    handed, why = _select([a, b])
    assert handed == [a.chunk_id]
    assert why == {b.chunk_id: "char_budget"}


@pytest.mark.parametrize("big_first", [True, False])
def test_a_tied_passage_too_big_for_the_room_stays_char_budget(
        monkeypatch, big_first):
    lead = _chunk("lead", 0, 0.90, "invented lead clause " * 4)
    small = _chunk("alpha", 0, 0.73, _OFF_TOPIC)
    big = _chunk("beta", 0, 0.73, (_ON_TOPIC + " ") * 6)
    tie = [big, small] if big_first else [small, big]
    _room_for_lead_and_one(monkeypatch, lead, small)
    handed, why = _select([lead, *tie])
    assert handed == [lead.chunk_id, small.chunk_id]
    assert why == {big.chunk_id: "char_budget"}


@pytest.mark.parametrize("first_text,second_text", [
    (_OFF_TOPIC, _ON_TOPIC), (_ON_TOPIC, _OFF_TOPIC)])
def test_the_tie_break_never_moves_a_deferred_chunk_ahead(
        monkeypatch, first_text, second_text):
    """Two chunks of one document and one of another; the diversity pass
    defers the second chunk of the first document. It ties with the other
    document's chunk, but the tie-break must not undo the deferral."""
    t_lead, t_def, t_other = _same_size("invented lead clause", first_text, second_text)
    a0 = _chunk("alpha", 0, 0.90, t_lead)
    a1 = _chunk("alpha", 1, 0.80, t_def)
    b0 = _chunk("beta", 0, 0.80, t_other)
    monkeypatch.setenv("RAG_HANDED_CHAR_BUDGET", str(len(a0.text) * 2 + 3))
    handed, why = _select([a0, a1, b0], k=2)
    assert handed == [a0.chunk_id, b0.chunk_id]
    assert why == {a1.chunk_id: "family_share"}


@pytest.mark.parametrize("first_text,second_text", [
    (_OFF_TOPIC, _ON_TOPIC), (_ON_TOPIC, _OFF_TOPIC)])
def test_without_a_tie_key_the_walk_is_unchanged(monkeypatch, first_text, second_text):
    lead, pair = _lead_and_tie(first_text, second_text)
    _room_for_lead_and_one(monkeypatch, lead, pair[0])
    why: dict = {}
    handed, _ = handed_set.select_handed([lead, *pair], k=8, reasons=why)
    assert [c.chunk_id for c in handed] == [lead.chunk_id, pair[0].chunk_id]
    assert why == {pair[1].chunk_id: "char_budget"}


@pytest.mark.parametrize("on_topic_second", [True, False])
def test_end_to_end_a_character_cut_tie_is_named_in_the_trace(
        monkeypatch, caplog, on_topic_second):
    texts = [_OFF_TOPIC, _ON_TOPIC] if on_topic_second else [_ON_TOPIC, _OFF_TOPIC]
    lead, pair = _lead_and_tie(*texts)
    on_topic, off_topic = (pair[1], pair[0]) if on_topic_second else (pair[0], pair[1])
    _room_for_lead_and_one(monkeypatch, lead, pair[0])
    audit, line = _inject_trace(monkeypatch, caplog, [lead, *pair])
    assert [c["chunk_id"] for c in audit["chunks"]] == [lead.chunk_id, on_topic.chunk_id]
    assert [c["chunk_id"] for c in line["chunks"]] == [lead.chunk_id, on_topic.chunk_id]
    assert [(e["chunk_id"], e["dropped_by"]) for e in line["not_handed"]] == [
        (off_topic.chunk_id, "score_tie_overlap")]
