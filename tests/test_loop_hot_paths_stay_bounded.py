"""Per-turn work that runs on the event loop stays bounded.

CI 2026-10-10 (15 concurrent users): the event loop ran 377 ms late. The
stalls sampled were the emit leak guard re-reading the whole answer on every
token, and the calculator display-name index rebuilt for every message.
"""
from __future__ import annotations

from app.agents import runtime as rt
from app.lib import construction_formulas as cf


def test_a_marker_split_across_two_tokens_still_trips_the_guard():
    guard = rt._EmitLeakGuard()
    marker = rt._INTERNAL_CONTEXT_MARKERS[0]
    assert guard.check({"type": "token", "content": "ok " + marker[:10]}) is not None
    assert guard.check({"type": "token", "content": marker[10:] + " more"}) is None
    assert guard.tripped


def test_the_guard_reads_a_bounded_tail_not_the_whole_answer(monkeypatch):
    seen: list[int] = []
    real = rt._looks_like_internal_context_leak

    def counting(text):
        seen.append(len(text))
        return real(text)

    monkeypatch.setattr(rt, "_looks_like_internal_context_leak", counting)
    guard = rt._EmitLeakGuard()
    piece = "The defects notification period is 365 days. "
    for _ in range(3000):
        guard.check({"type": "token", "content": piece})
    assert not guard.tripped
    assert max(seen) <= guard.TAIL_CHARS + len(piece)


def test_a_retrieval_marker_late_in_a_long_answer_still_trips():
    guard = rt._EmitLeakGuard()
    for _ in range(500):
        guard.check({"type": "token", "content": "plain words "})
    guard.check({"type": "token", "content": "[doc_id=d1 chu"})
    guard.check({"type": "token", "content": "nk=4 score=0.9]"})
    assert guard.tripped


def test_the_display_name_index_is_built_once(monkeypatch):
    cf._DISPLAY_PHRASES.clear()
    cf.calculators_named_by_display("delay damages per day for this contract")
    calls = []
    real = cf._phrase_words
    monkeypatch.setattr(cf, "_phrase_words", lambda t: calls.append(t) or real(t))
    named = cf.calculators_named_by_display("what are the delay damages per day")
    assert calls == ["what are the delay damages per day"]
    assert "delay_damages_daily" in named
