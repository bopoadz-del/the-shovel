"""A delay of N days, asked for damages, is a money answer.

Live 4b3f4b9, Set 1 E2, one run in three:

    "If Milestone 1 is 30 days late, what are the milestone delay damages?"

Retrieval was finally right -- the 0.015%-per-day Milestone row AND the
Accepted Contract Amount were both in front of the model -- and it still
stopped short:

    "... that is 0.45% of the Contract Price. If you want the figure in
     money rather than percent, I need the Contract Price to apply it to --
     the Contract Data gives the Accepted Contract Amount as SAR ..., but the
     percentage is expressed against the 'Contract Price', so confirm which
     base the contract intends before I extend it."

A careful hedge, and the wrong one here: the platform already settles it for
the whole-of-Works question (E1) -- the rate is applied to the Accepted
Contract Amount excluding VAT -- and the owner's expected answer for E2 does
the same: 0.015% x ACA x 30. E1 gets that instruction; E2's wording ("30
days late", no "calculate", no currency) never triggered it.
"""
from __future__ import annotations

import pytest

from app.core.rag.inject import format_chunks_as_system_message
from app.core.rag.vector_store import Chunk

LABEL = ("CONTRACT DATA particulars — filled-in amount / duration / percentage "
         "[XX-2099-001_Contract Data.pdf].\nCONTRACT DATA\n")
RATES = LABEL + (
    "8.8.1: | | Delay Damages (for the whole of the Works): 0.1% of the Contract "
    "Price per calendar day |\n"
    "8.8.1: | | Delay Damages (if applicable per Milestone): Milestone | Delay Damages\n"
    "|: | | Milestone 1 | 0.015% of the Contract Price per calendar day\n"
    "8.8.1: | | Maximum Amount of Delay Damages: 10% of the Contract Price |\n"
)
ACA = LABEL + "1.1.1: | | Accepted Contract Amount: SAR 1,000,000,000.00 excluding VAT |\n"

E2 = "If Milestone 1 is 30 days late, what are the milestone delay damages?"
HEADING = "DELAY DAMAGES OVER A PERIOD"


def _chunk(cid, text):
    return Chunk(chunk_id=cid, project_id="p", doc_id="cd", chunk_index=0, text=text, score=1.0)


def _context(question, *texts):
    chunks = [_chunk(f"c{i}", t) for i, t in enumerate(texts)]
    return format_chunks_as_system_message(chunks, 1, query=question)["content"]


def test_the_model_is_told_which_base_and_to_finish_the_sum():
    ctx = _context(E2, RATES, ACA)
    assert HEADING in ctx
    block = ctx[ctx.index(HEADING):ctx.index(HEADING) + 900]
    assert "Accepted Contract Amount" in block and "excluding" in block.lower()
    assert "number of days" in block.lower()
    # The rate for the row that was ASKED: a Milestone question must not be
    # answered with the whole-of-Works rate sitting one line above it.
    assert "milestone" in block.lower() and "whole of the works" in block.lower()
    assert "do not ask" in block.lower() or "do not stop" in block.lower()


@pytest.mark.parametrize(
    "question",
    [
        E2,
        "Milestone 2 finishes 6 weeks behind; what delay damages apply?",
        "What are the delay damages for 14 calendar days of delay to the whole of the Works?",
    ],
)
def test_every_phrasing_of_a_delay_period_gets_it(question):
    assert HEADING in _context(question, RATES, ACA)


def test_without_the_sum_in_front_of_it_the_model_is_not_told_to_use_one():
    """The instruction is only honest when both operands are in the excerpts."""
    assert HEADING not in _context(E2, RATES)
    assert HEADING not in _context(E2, ACA)


@pytest.mark.parametrize(
    "question",
    [
        "What are the Delay Damages for the whole of the Works?",          # A5: a rate lookup
        "What is the Maximum Amount of Delay Damages?",
        "What is the Time for Completion for Milestone 5?",
        "What is the Accepted Contract Amount, excluding VAT?",
    ],
)
def test_a_question_with_no_delay_period_is_left_alone(question):
    assert HEADING not in _context(question, RATES, ACA)


def test_the_whole_of_works_calculation_keeps_its_own_instruction():
    """E1 must not get both blocks: it has a composer of its own."""
    ctx = _context("Calculate the delay damages per calendar day in SAR for the "
                   "whole of the Works.", RATES, ACA)
    assert "DELAY DAMAGES PER CALENDAR DAY" in ctx
    assert HEADING not in ctx


def test_the_instruction_heading_never_surfaces_in_an_answer():
    """Models echo their instructions. Every other heading is stripped when
    that happens; this one has to be on the same list."""
    from app.agents.runtime import _strip_answer_routing_preamble

    echoed = (HEADING + " — excerpts below state the delay-damages rate.\n"
              "Milestone 1 delay damages for 30 days are SAR 4,500,000.00.")
    out = _strip_answer_routing_preamble(echoed)
    assert HEADING not in out
    assert "SAR 4,500,000.00" in out


def test_a_question_that_is_both_gets_one_instruction_not_two():
    """"Calculate ... in SAR for 14 calendar days of delay" is E1-shaped AND
    names a period. Two blocks giving different recipes is worse than one."""
    ctx = _context("Calculate the delay damages in SAR for 14 calendar days of delay "
                   "to the whole of the Works.", RATES, ACA)
    assert ("DELAY DAMAGES PER CALENDAR DAY" in ctx) != (HEADING in ctx)
