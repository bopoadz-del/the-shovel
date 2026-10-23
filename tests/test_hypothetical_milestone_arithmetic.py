"""User-supplied milestone durations are arithmetic, not a Contract Data veto.

Live theshovel.ai ~27d6940, project-assistant / master_corpus:

    "With M1=397d, M3=487d and M5=731d all starting at right-of-access
     on the same date, which milestone drives completion and by how much
     over M1?"

The assistant rejected the premise on every run ("I can't answer that as
posed, because the premise doesn't hold against the Contract Data…").
Sometimes a conditional "if simultaneous, M5 finishes 334 days after M1"
was buried at the end. Probe: 0/5 correct substance.

Owner rule: this is a hypothetical. The numbers in the question are the
operands. Access-date particulars must not override them. Longest of the
supplied set drives completion; the delta over the named baseline is
longest − baseline (731 − 397 = 334).

Figures in this file are synthetic or the live duration triple only —
no client party names, emails, or proprietary prose.
"""
from __future__ import annotations

import pytest

from app.agents.runtime import (
    _apply_rag_context,
    _forced_specific_tool,
    _graft_hypothetical_milestone_arithmetic,
    _postprocess_answer,
    _strip_answer_routing_preamble,
)
from app.core.hypothetical_milestone_arithmetic import (
    HEADING,
    answer_leads_with_premise_rejection,
    compose_hypothetical_milestone_arithmetic,
    query_is_hypothetical_milestone_arithmetic,
)
from app.core.rag.inject import format_chunks_as_system_message
from app.core.rag.vector_store import Chunk

# Live duration triple (no party names). Same arithmetic the owner named.
LIVE_ASK = (
    "With M1=397d, M3=487d and M5=731d all starting at right-of-access "
    "on the same date, which milestone drives completion and by how much "
    "over M1?"
)

# Fully synthetic — different ids and days so the test is not a live dump.
SYNTH_ASK = (
    "With M2=120d, M4=200d and M8=355d all starting at right of access "
    "on the same date, which milestone drives completion and by how much "
    "over M2?"
)

CONFLICTING_CD = (
    "CONTRACT DATA particulars — filled-in amount / duration / percentage "
    "[XX-2099-001_Contract Data.pdf].\nCONTRACT DATA\n"
    "The Contractor is given right of access to Area North on 01 Jan 2099 "
    "and to Area East on 01 Jun 2099. Milestone 2 and Milestone 8 therefore "
    "do not share a start date.\n"
)

REJECTION = (
    "I can't answer that as posed, because the premise doesn't hold against "
    "the Contract Data: the milestones do not all start on the same date. "
    "If they were simultaneous, Milestone 8 would finish 235 days after "
    "Milestone 2."
)

REFUSAL_WORDING = "using ONLY the reference context"
AVAILABLE = {"construction_calc", "search_project_documents"}


def _chunk(text: str) -> Chunk:
    return Chunk(
        chunk_id="c0",
        project_id="p",
        doc_id="cd",
        chunk_index=0,
        text=text,
        score=1.0,
    )


def _context(question: str, *texts: str) -> str:
    chunks = [_chunk(t) for t in texts] or [_chunk("no particulars")]
    return format_chunks_as_system_message(
        chunks, 1, query=question,
    )["content"]


def _msgs(question: str) -> list[dict]:
    return [{"role": "user", "content": question}]


# ── detector ──────────────────────────────────────────────────────────────


def test_the_live_ask_is_hypothetical_arithmetic():
    assert query_is_hypothetical_milestone_arithmetic(LIVE_ASK)
    assert query_is_hypothetical_milestone_arithmetic(SYNTH_ASK)


@pytest.mark.parametrize(
    "question",
    [
        "What is the Time for Completion for Milestone 5?",
        "Which Milestones have a Time for Completion of 240 days?",
        "Among the northern milestones, which has the longest Time for Completion?",
        "How many milestones are there?",
        "Calculate the concrete volume for a raft 25m x 18m x 1.2m thick",
    ],
)
def test_document_lookups_and_unrelated_calcs_are_left_alone(question):
    assert not query_is_hypothetical_milestone_arithmetic(question)


# ── compose ───────────────────────────────────────────────────────────────


def test_synthetic_numbers_name_the_longest_and_the_delta():
    composed = compose_hypothetical_milestone_arithmetic(SYNTH_ASK)
    assert composed is not None
    assert composed["delta"] == 235
    assert 8 in composed["drivers"]
    assert composed["baseline"] == 2
    line = composed["line"]
    assert "235" in line
    assert "355" in line and "120" in line
    assert "Milestone 8" in line or "M8" in line
    assert not answer_leads_with_premise_rejection(line)


def test_live_duration_triple_is_334_over_m1():
    composed = compose_hypothetical_milestone_arithmetic(LIVE_ASK)
    assert composed is not None
    assert composed["delta"] == 334
    assert 5 in composed["drivers"]
    assert composed["baseline"] == 1
    line = composed["line"]
    assert "334" in line
    assert "731" in line and "397" in line
    assert "Milestone 5" in line or "M5" in line


def test_tied_longest_names_every_driver():
    ask = (
        "With M1=100d, M3=180d and M6=180d all starting on the same date, "
        "which milestone drives completion and by how much over M1?"
    )
    composed = compose_hypothetical_milestone_arithmetic(ask)
    assert composed is not None
    assert composed["delta"] == 80
    assert composed["drivers"] == (3, 6)
    line = composed["line"]
    assert "80" in line
    assert ("Milestone 3" in line or "M3" in line)
    assert ("Milestone 6" in line or "M6" in line)


# ── inject instruction ────────────────────────────────────────────────────


def test_conflicting_contract_data_does_not_silence_the_instruction():
    ctx = _context(SYNTH_ASK, CONFLICTING_CD)
    assert HEADING in ctx
    block = ctx[ctx.index(HEADING): ctx.index(HEADING) + 700]
    assert "do not reject" in block.lower() or "must not override" in block.lower()
    assert "235" in block
    assert "Milestone 8" in block or "M8" in block


def test_a_lookup_does_not_get_the_heading():
    ctx = _context(
        "What is the Time for Completion for Milestone 5?",
        CONFLICTING_CD,
    )
    assert HEADING not in ctx


# ── RAG directive: not the strict grounding that forbids user numbers ─────


def test_the_directive_is_not_the_strict_lookup_clamp():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": SYNTH_ASK},
    ]
    applied = _apply_rag_context(
        messages, {"role": "system", "content": "REFERENCE CONTEXT\n" + CONFLICTING_CD},
    )
    assert applied is True
    content = messages[-1]["content"]
    assert REFUSAL_WORDING not in content, content[-600:]
    assert "CALCULATION REQUEST:" in content


def test_the_calculator_is_not_forced():
    """No registered formula owns 'which milestone is longest'. Do not steal."""
    assert _forced_specific_tool(_msgs(SYNTH_ASK), AVAILABLE) is None
    assert _forced_specific_tool(_msgs(LIVE_ASK), AVAILABLE) is None


# ── graft / product path ──────────────────────────────────────────────────


def test_graft_does_not_lead_with_premise_rejection():
    out = _graft_hypothetical_milestone_arithmetic(REJECTION, _msgs(SYNTH_ASK))
    assert not answer_leads_with_premise_rejection(out)
    assert "235" in out
    assert "Milestone 8" in out.split("\n", 1)[0] or "M8" in out.split("\n", 1)[0]


def test_postprocess_answers_the_hypothetical_from_the_users_numbers():
    out = _postprocess_answer(
        REJECTION,
        {"role": "system", "content": CONFLICTING_CD},
        _msgs(SYNTH_ASK),
    )
    lead = out.split("\n", 1)[0]
    assert not answer_leads_with_premise_rejection(out)
    assert "235" in lead
    assert "Milestone 8" in lead or "M8" in lead


def test_an_already_correct_lead_is_left_alone():
    composed = compose_hypothetical_milestone_arithmetic(SYNTH_ASK)
    assert composed is not None
    already = composed["line"] + " Access dates in Contract Data are ignored."
    assert _graft_hypothetical_milestone_arithmetic(
        already, _msgs(SYNTH_ASK),
    ) == already


def test_a_lookup_answer_is_not_rewritten():
    text = "The Time for Completion for Milestone 5 is 400 days."
    assert _graft_hypothetical_milestone_arithmetic(
        text, _msgs("What is the Time for Completion for Milestone 5?"),
    ) == text


def test_the_instruction_heading_never_surfaces_in_an_answer():
    echoed = (
        HEADING + " — the user supplied milestone durations.\n"
        "Milestone 8 drives completion, by 235 days over Milestone 2."
    )
    out = _strip_answer_routing_preamble(echoed)
    assert HEADING not in out
    assert "235" in out


def test_kill_switch_restores_the_fail(monkeypatch):
    monkeypatch.setenv("HYPOTHETICAL_MILESTONE_ARITHMETIC", "0")
    assert not query_is_hypothetical_milestone_arithmetic(SYNTH_ASK)
    assert compose_hypothetical_milestone_arithmetic(SYNTH_ASK) is None
    assert HEADING not in _context(SYNTH_ASK, CONFLICTING_CD)
    assert _graft_hypothetical_milestone_arithmetic(
        REJECTION, _msgs(SYNTH_ASK),
    ) == REJECTION
