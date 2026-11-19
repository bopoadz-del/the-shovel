"""Tool recovery names the rule that fired when it replaces the final text.

Live count row (TESTER v2j_n1, 19 Nov): two turns ended as the recovery
block (every computed call of the turn, the 88 AND a 72) with no model
sentence, so ``_postprocess_answer`` replaced the model's synthesis. It has
two recovery sites, (a) the text itself needed recovery and (b) a gate
removed every clause, and no line told them apart. ``tool_recovery_reason``
is the one implementation behind the recovery decision and its trace;
``recover_answer_traced`` logs the rule and lengths only (never the text)
when the recovery replaces the text. No answer changes.

Every input here is invented by the test.
"""
from __future__ import annotations

import logging
import re

import pytest

from app.agents import runtime as rt
from app.agents.runtime import _EMPTY_RESPONSE_FALLBACK

_LINE = "postprocess: recovery replaced final_text"

# Two inputs per rule.
CASES = {
    "empty": ["", "   \n  "],
    "no_words": ["---", "* **"],
    "fallback_line": [_EMPTY_RESPONSE_FALLBACK, f"  {_EMPTY_RESPONSE_FALLBACK}\n"],
    "unable_to_generate": [
        "I was unable to generate the Brask manifold schedule.",
        "Unable to generate a reply for the Quillon crate count.",
    ],
    "provider_limit": [
        "Provider said HTTP 413 for the Brask manifold request.",
        "Rate limited: 9000 tokens per minute exceeded on the Quillon ask.",
    ],
    "search_promise": [
        "I'll search the drawings register for the Brask manifold now.",
        "No sheet names a Quillon valve crate. Let me search the delivery notes next.",
    ],
    "tool_error_json": [
        '{"status": "error", "error": "Brask manifold calc failed"}',
        '{"status": "failed", "error": "no Quillon crate rows"}',
    ],
    "formula_template": [
        "Duration = Quantity / Daily Production",
        "Crate Count = Pallets / Crates Per Pallet",
    ],
    "truncation_notice": [
        "Tool result exceeded 4000 characters for the Brask file.",
        "Next window at char_offset=12000 for the Quillon notes.",
    ],
    "unbound_delay_damages": [
        "Delay damages: 0% of SAR 0.00 per day.",
        "The cap is 0 % of AED 0 on the Quillon contract.",
    ],
}
# Usable answers: never handed to recovery.
USABLE = [
    "The Quillon valve crate is not in any project document, so there is no count to cite.",
    "88 Nr Brask manifolds are measured across the two pages of the section.",
]


def _predispatch(draft: str) -> list[dict]:
    return [
        {"role": "user", "content": f"PLATFORM PRE-DISPATCH: wir_form\nAuthoritative draft: {draft}\nDo not search again."},
        {"role": "user", "content": "the operator's ask"},
    ]


DRAFTS = [
    "WIR-77 Brask manifold inspection, level 3, Thursday.",
    "Quillon crate delivery check, gate B, 14 crates received.",
]


@pytest.mark.parametrize("reason,texts", sorted(CASES.items()))
def test_each_rule_is_named(reason, texts):
    for text in texts:
        assert rt.tool_recovery_reason(text) == reason, text
        assert rt._text_needs_tool_recovery(text) is True


@pytest.mark.parametrize("text", USABLE)
def test_a_usable_answer_has_no_reason(text):
    assert rt.tool_recovery_reason(text) is None
    assert rt._text_needs_tool_recovery(text) is False


@pytest.mark.parametrize("text,draft", [
    (CASES["search_promise"][0], DRAFTS[0]),
    (CASES["empty"][0], DRAFTS[1]),
])
def test_a_replacement_is_traced_by_rule_and_lengths_never_text(text, draft, caplog):
    msgs = _predispatch(draft)
    caplog.set_level(logging.INFO, logger=rt._LOG.name)
    out = rt.recover_answer_traced(text, msgs)
    # The answer is exactly the untraced call's.
    assert out == rt._recover_answer_from_tool_messages(text, msgs) == draft
    lines = [r.getMessage() for r in caplog.records if _LINE in r.getMessage()]
    assert lines == [
        f"{_LINE} reason={rt.tool_recovery_reason(text)} raw={len(text)}c recovered={len(draft)}c"
    ]
    assert draft not in lines[0]
    if text.strip():
        assert text not in lines[0]


@pytest.mark.parametrize("text,msgs", [
    (USABLE[0], _predispatch(DRAFTS[0])),  # usable: recovery keeps it
    ("", []),                               # nothing to recover from
])
def test_no_replacement_no_line(text, msgs, caplog):
    caplog.set_level(logging.INFO, logger=rt._LOG.name)
    assert rt.recover_answer_traced(text, msgs) == rt._recover_answer_from_tool_messages(text, msgs)
    assert not [r for r in caplog.records if _LINE in r.getMessage()]


@pytest.mark.parametrize("raw_len,draft", [(41, DRAFTS[0]), (2902, DRAFTS[1])])
def test_the_callers_reason_and_raw_length_win(raw_len, draft, caplog):
    caplog.set_level(logging.INFO, logger=rt._LOG.name)
    out = rt.recover_answer_traced("", _predispatch(draft), reason="gate_emptied", raw_len=raw_len)
    assert out == draft
    lines = [r.getMessage() for r in caplog.records if _LINE in r.getMessage()]
    assert lines == [f"{_LINE} reason=gate_emptied raw={raw_len}c recovered={len(draft)}c"]


def _postprocess(text, msgs):
    return rt._postprocess_answer(text, None, msgs)


@pytest.mark.parametrize("draft", DRAFTS)
def test_postprocess_site_a_names_the_rule(draft, caplog):
    caplog.set_level(logging.INFO, logger=rt._LOG.name)
    text = CASES["search_promise"][1]
    _postprocess(text, _predispatch(draft))
    lines = [r.getMessage() for r in caplog.records if _LINE in r.getMessage()]
    assert lines[0] == f"{_LINE} reason=search_promise raw={len(text)}c recovered={len(draft)}c"


@pytest.mark.parametrize("draft", DRAFTS)
def test_postprocess_site_b_names_the_emptying_gate(draft, monkeypatch, caplog):
    import app.agents.citation_provenance as cp

    seen: list[str] = []

    def _record(t, *a, **k):
        seen.append(t)
        return t

    monkeypatch.setattr(rt, "_cost_grounding_gate", _record)
    monkeypatch.setattr(cp, "gate", lambda t, *a, **k: "")
    caplog.set_level(logging.INFO, logger=rt._LOG.name)
    _postprocess(USABLE[1], _predispatch(draft))
    lines = [r.getMessage() for r in caplog.records if _LINE in r.getMessage()]
    assert len(seen) == 1 and seen[0].strip()
    assert lines == [f"{_LINE} reason=gate_emptied raw={len(seen[0])}c recovered={len(draft)}c"]
    assert not re.search(r"Brask|Quillon", lines[0])
