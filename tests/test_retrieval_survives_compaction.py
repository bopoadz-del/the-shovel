"""The turn's evidence must be the last thing the budget takes.

Measured on the real thing: a five-chunk Contract Data brief is 8072
characters -- 1704 of instructions and 6368 of evidence. The compactor cut
any system message over 2500 characters to ``content[:2500]``, which kept
every word of the instructions and 796 characters of evidence: ONE excerpt
of five, sliced mid-sentence, with nothing to say four were gone.

That is the worst possible trade. The header tells the model it is holding
authoritative context and must answer from it; the compactor then removes
the context and leaves the instruction. It is also why two live answers on
554e0b9 reported a fact "missing" that retrieval had actually returned:

    D1  "the specific Contract Data entry ... that names the individual
         Engineer's Representative is missing from what was returned"
    E1  "The retrieved excerpts do not contain the Contract Price expressed
         in SAR"

Both read as retrieval misses. Neither was.
"""
from __future__ import annotations

import json

import pytest

from app.agents.runtime import (
    _LAST_TOOL_PREVIEW,
    _RETRIEVAL_BRIEF_MARKER,
    _TOOL_PREVIEW_FLOOR,
    _approx_message_chars,
    _compact_messages_for_tpm,
    _compact_retrieval_message,
)
from app.core.rag.inject import format_chunks_as_system_message


class _Chunk:
    def __init__(self, i: int, text: str):
        self.doc_id = f"doc{i:04d}"
        self.chunk_index = i
        self.score = 2.5 - i * 0.1          # descending, as the retriever orders
        self.text = text
        self.source_name = "AB-2023-101_Contract_Data.pdf"
        self.revision = ""
        self.superseded = False
        self.layer = "own"


def _brief(n: int = 5, body: str | None = None) -> dict:
    body = body or ("Delay Damages 0.1% of the Contract Price per calendar "
                    "day. " * 20)
    return format_chunks_as_system_message(
        [_Chunk(i, f"EXCERPT-{i} " + body) for i in range(n)],
        total_candidates=n,
    )


def _turn(tool_chars: int = 30000, n_chunks: int = 5) -> list:
    return [
        {"role": "system", "content": "You are a construction assistant. " * 40},
        _brief(n_chunks),
        {"role": "user", "content": "What is the daily rate for Delay Damages?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "1", "type": "function",
             "function": {"name": "boq_processor", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "1", "name": "boq_processor",
         "content": "x" * tool_chars},
    ]


def _rag(out) -> str:
    for m in out:
        if (isinstance(m, dict) and m.get("role") == "system"
                and _RETRIEVAL_BRIEF_MARKER in str(m.get("content") or "")):
            return m["content"]
    return ""


def _excerpts(text: str) -> int:
    return text.count("[doc_id=")


# -- the headline: evidence is not what pays for the tool dump ------------


def test_a_huge_tool_dump_no_longer_costs_the_evidence():
    """The E1/D1 shape, stated as the comparison it is.

    Mutation killed: cutting the retrieval message to 2500 alongside
    boilerplate. The assertion is written against what THAT would have left,
    so the test says what changed rather than restating the new constant.
    """
    msgs = _turn(tool_chars=30000)
    before = _rag(msgs)
    assert _excerpts(before) == 5

    would_have_survived = _excerpts(before[:2500])
    assert would_have_survived == 1, "the old flat cut left one excerpt"

    out = _compact_messages_for_tpm(msgs, budget=16000)
    after = _rag(out)
    assert _excerpts(after) >= 4, "the tool dump must give up its characters first"
    assert _excerpts(after) > would_have_survived
    assert _approx_message_chars(out) <= 16000


def test_all_the_evidence_survives_when_there_is_room():
    """Priority, not merely a bigger allowance: with a tool result that can
    be compacted, the excerpts should not be touched at all."""
    msgs = _turn(tool_chars=12000)
    before = _rag(msgs)
    out = _compact_messages_for_tpm(msgs, budget=16000)
    assert _rag(out) == before
    assert _excerpts(_rag(out)) == 5
    assert "BUDGET NOTE" not in _rag(out)


def test_boilerplate_system_messages_are_still_cut_flat():
    """Only the brief is special. A long prompt preamble is prompt text."""
    msgs = _turn(tool_chars=30000)
    out = _compact_messages_for_tpm(msgs, budget=6000)
    plain = [
        m for m in out
        if isinstance(m, dict) and m.get("role") == "system"
        and _RETRIEVAL_BRIEF_MARKER not in str(m.get("content") or "")
    ]
    assert plain, "the boilerplate system message should still be present"


# -- when it must give, it gives whole excerpts and says so ---------------


def test_excerpts_are_dropped_whole_and_counted():
    out = _compact_messages_for_tpm(_turn(tool_chars=30000), budget=9000)
    text = _rag(out)
    kept = _excerpts(text)
    assert 0 < kept < 5
    assert "BUDGET NOTE" in text
    assert f"{5 - kept} of 5" in text
    # Whole, not sliced: every surviving excerpt still carries its own body.
    for i in range(kept):
        assert f"EXCERPT-{i} " in text


def test_the_weakest_evidence_goes_first():
    """The retriever orders by score, so the tail is the weakest. Dropping
    from the head would throw away the best match the search found.

    Mutation killed: slicing from the front, or reversing the keep order.
    """
    text = _rag(_compact_messages_for_tpm(_turn(tool_chars=30000), budget=9000))
    kept = _excerpts(text)
    assert "EXCERPT-0 " in text, "the top match must survive"
    assert f"EXCERPT-{kept - 1} " in text
    assert f"EXCERPT-{kept} " not in text


def test_nothing_is_dropped_silently():
    """A smaller sample makes the absence rule matter MORE. If the model is
    not told the sample shrank, it reasons about a set it no longer has."""
    text = _rag(_compact_messages_for_tpm(_turn(tool_chars=30000), budget=9000))
    note = text[text.index("BUDGET NOTE"):]
    assert "lowest relevance first" in note
    assert "Never report that" in note
    assert "search again" in note


# -- the rules outlive the evidence ---------------------------------------


@pytest.mark.parametrize("budget", [16000, 9000, 6000, 3000, 1500, 400])
def test_the_honesty_rules_always_survive(budget):
    """The one outcome worse than a short hop is instructions with the
    caveats stripped: SCOPE OF ABSENCE is exactly what stops a partial view
    becoming a claim about the customer's contract.

    Mutation killed: slicing the header when even it does not fit.
    """
    text = _rag(_compact_messages_for_tpm(_turn(tool_chars=30000), budget=budget))
    assert _RETRIEVAL_BRIEF_MARKER in text
    assert "SCOPE OF ABSENCE" in text
    assert "NEVER state" in text


def test_zero_surviving_excerpts_still_reports_the_loss():
    """Zero is a real outcome, not an edge case to fall out of the loop."""
    text = _compact_retrieval_message(_brief(5)["content"], budget=2100)
    assert _excerpts(text) == 0
    assert "BUDGET NOTE" in text
    assert "5 of 5" in text
    assert "SCOPE OF ABSENCE" in text


def test_a_brief_that_already_fits_is_returned_unchanged():
    content = _brief(5)["content"]
    assert _compact_retrieval_message(content, budget=len(content)) == content
    assert _compact_retrieval_message(content, budget=len(content) + 5000) == content


def test_a_system_message_of_an_unexpected_shape_falls_back_to_slicing():
    """Defensive: if inject.py's marker format changes, we must still fit
    the budget rather than returning something oversized."""
    odd = _RETRIEVAL_BRIEF_MARKER + " " + ("no excerpt markers here. " * 200)
    out = _compact_retrieval_message(odd, budget=500)
    assert len(out) <= 500 + len("\n[truncated for TPM]")
    assert "[truncated for TPM]" in out


# -- the hop still fits ---------------------------------------------------


@pytest.mark.parametrize("budget", [16000, 12000, 9000, 6000])
def test_the_hop_fits_at_realistic_budgets(budget):
    out = _compact_messages_for_tpm(_turn(tool_chars=30000), budget=budget)
    assert _approx_message_chars(out) <= budget


def test_a_hop_already_under_budget_is_untouched():
    msgs = _turn(tool_chars=200, n_chunks=2)
    assert _compact_messages_for_tpm(msgs, budget=1_000_000) is msgs


def test_roles_and_pairing_survive():
    msgs = _turn(tool_chars=30000)
    out = _compact_messages_for_tpm(msgs, budget=9000)
    assert [m.get("role") for m in out] == [m.get("role") for m in msgs]


def test_a_tool_preview_yields_before_any_excerpt_is_dropped():
    """Excerpts are dropped WHOLE, so the coarse unit must come out of the
    cheap budget. A 170-character overage must not cost a 1300-character
    excerpt while a 6000-character tool preview sits untouched.

    Mutation killed: running the evidence pass straight after the first
    tool pass, without letting the tool previews spend down to their floor.
    """
    for tool_chars in (60000, 30000, 12000):
        out = _compact_messages_for_tpm(_turn(tool_chars=tool_chars), budget=16000)
        assert _excerpts(_rag(out)) == 5, tool_chars
        assert "BUDGET NOTE" not in _rag(out), tool_chars
        tool = next(m for m in out if isinstance(m, dict) and m.get("role") == "tool")
        shown = json.loads(tool["content"])["chars_shown"]
        assert _TOOL_PREVIEW_FLOOR <= shown <= _LAST_TOOL_PREVIEW
        assert _approx_message_chars(out) <= 16000


def test_the_evidence_only_gives_once_the_tools_are_at_their_floor():
    """And when it does give, it is because there is genuinely nothing left
    to take -- not because the order was wrong."""
    out = _compact_messages_for_tpm(_turn(tool_chars=30000), budget=6000)
    text = _rag(out)
    assert _excerpts(text) < 5
    assert "BUDGET NOTE" in text
    tool = next(m for m in out if isinstance(m, dict) and m.get("role") == "tool")
    assert json.loads(tool["content"])["chars_shown"] == _TOOL_PREVIEW_FLOOR
