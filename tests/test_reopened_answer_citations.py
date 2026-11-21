"""W3: a reopened answer shows what the live answer showed.

The live ``end`` frame passes the answer contract: citations leave the text
as numbered markers and ride beside it as ``citations`` for the Sources
panel. The stored answer is written before that, so a reopened conversation
used to show the bracketed source tags inline and no panel entries. These
tests build their own invented answers (seeded, no real document) and hold
the reopened message to the live frame.
"""

from __future__ import annotations

import random
import re
import uuid

import pytest

_SEED = 20261121
_MARKER_RE = re.compile(r"\[(\d+)\]")


def _names(rng: random.Random, n: int) -> list[str]:
    stems = ["slab", "pier", "duct", "kerb", "riser", "lintel", "plinth", "mullion", "sump", "parapet"]
    exts = [".pdf", ".docx", ".xlsx", ".md"]
    picked = rng.sample(stems, n)
    return [f"{s.title()}_{rng.randint(100, 999)}{rng.choice(exts)}" for s in picked]


def _answer(rng: random.Random, names: list[str]) -> tuple[str, list[str]]:
    sections = [rng.choice([f"p. {rng.randint(2, 90)}", f"Section {rng.randint(1, 9)}.{rng.randint(1, 9)}"])
                for _ in names]
    lines = [f"Item {i + 1} is {rng.randint(10, 900)} mm [source: {n}, {s}]."
             for i, (n, s) in enumerate(zip(names, sections))]
    return " ".join(lines), sections


def _store(cid: str, raw: str, citations: list[dict] | None):
    from app.core import agent_memory

    agent_memory.get_or_create_conversation(cid, "project-assistant", "proj_w3")
    agent_memory.append_message(cid, "user", "What sizes are given?")
    msg = agent_memory.append_message(cid, "assistant", raw)
    if citations:
        assert agent_memory.attach_citations(msg["id"], citations)
    return msg


def _live(raw: str) -> dict:
    from app.agents import answer_exit

    return answer_exit.check_end_event({"type": "end", "content": raw}, raw, answer_exit.Turn())


@pytest.mark.parametrize("trial", range(4))
def test_reopened_answer_is_the_live_answer(trial):
    from app.agents import answer_exit
    from app.routers import agents as agents_router

    rng = random.Random(_SEED + trial)
    names = _names(rng, rng.randint(1, 4))
    raw, _sections = _answer(rng, names)
    live = _live(raw)
    assert live.get("citations"), "the invented answer cites something"

    cid = f"w3-reopen-{uuid.uuid4().hex[:8]}"
    _store(cid, raw, answer_exit.citations_to_store(live))
    reopened = agents_router._reopened_messages(cid)
    user, answer = reopened
    assert user["content"] == "What sizes are given?", "a user's own words are shown as typed"
    assert answer["content"] == live["content"]
    for name in names:
        assert name not in answer["content"], "no file name inline: it lives in the Sources panel"
    got = [(c["marker"], c["document"], c["section"]) for c in answer["citations"]]
    want = [(c["marker"], c["document"], c["section"]) for c in live["citations"]]
    assert got == want
    assert sorted({int(m) for m in _MARKER_RE.findall(answer["content"])}) == [c["marker"] for c in answer["citations"]]


def test_stored_tier_is_read_back_onto_its_entry():
    from app.routers import agents as agents_router

    rng = random.Random(_SEED + 50)
    names = _names(rng, 2)
    raw, sections = _answer(rng, names)
    tiers = [f"tier-{rng.randint(1000, 9999)}", None]
    stored = [{"marker": i + 1, "document": n, "section": s, "tier": t, "kind": "document"}
              for i, (n, s, t) in enumerate(zip(names, sections, tiers))]
    cid = f"w3-tier-{uuid.uuid4().hex[:8]}"
    _store(cid, raw, stored)
    answer = agents_router._reopened_messages(cid)[1]
    by_doc = {c["document"]: c for c in answer["citations"]}
    assert by_doc[names[0]]["tier"] == tiers[0]
    assert by_doc[names[1]]["tier"] is None


def test_answer_stored_without_citations_is_still_numbered():
    """An answer stored before citations were kept (or whose check stored
    none) is numbered on the way out all the same."""
    from app.routers import agents as agents_router

    rng = random.Random(_SEED + 60)
    names = _names(rng, 2)
    raw, _ = _answer(rng, names)
    cid = f"w3-none-{uuid.uuid4().hex[:8]}"
    _store(cid, raw, None)
    answer = agents_router._reopened_messages(cid)[1]
    assert [c["document"] for c in answer["citations"]] == names
    assert "[source:" not in answer["content"]


def test_answer_without_citations_is_left_alone():
    from app.routers import agents as agents_router

    rng = random.Random(_SEED + 70)
    text = f"The pour takes {rng.randint(2, 9)} hours."
    cid = f"w3-plain-{uuid.uuid4().hex[:8]}"
    _store(cid, text, None)
    answer = agents_router._reopened_messages(cid)[1]
    assert answer["content"] == text
    assert not answer.get("citations")
    assert "coverage" not in answer


def test_contract_failure_leaves_the_message_loadable(monkeypatch):
    from app.agents import answer_contract
    from app.routers import agents as agents_router

    rng = random.Random(_SEED + 80)
    names = _names(rng, 1)
    raw, _ = _answer(rng, names)
    cid = f"w3-fail-{uuid.uuid4().hex[:8]}"
    _store(cid, raw, None)

    def boom(*_a, **_k):
        raise RuntimeError("invented failure")

    monkeypatch.setattr(answer_contract, "apply", boom)
    reopened = agents_router._reopened_messages(cid)
    assert len(reopened) == 2 and reopened[1]["role"] == "assistant"
    assert reopened[1]["content"].strip()
