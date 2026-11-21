"""A sentence under a stated conflict that also carries a worked result stays
(exam 8 classes X1 / X7: a figure computed from the user's own inputs is
stated, never lost behind the conflict line).

The leading conflict line names the differing values and their documents.
It replaces a sentence that only restates ONE of those values about the
thing. It must not replace a sentence that also states a typed value the
conflict line does not carry (a result worked from the value, a second
quantity): only the competing value is superseded, not the rest of the
sentence.

Two decisions, two inputs each:
* kept: one competing value + a worked result of another value;
* dropped (unchanged): one competing value and nothing else.

Every document, value and question here is invented for this file; the
expected values are written by hand.
"""

from __future__ import annotations

import pytest

from app.agents import answer_contract

_LINE = "The documents differ on this:"


def _rows(*pairs: tuple[str, str]) -> list[dict]:
    return [{"doc_id": f"d{i}", "doc_name": n, "snippet": t} for i, (n, t) in enumerate(pairs)]


_CASES = [
    pytest.param(
        "Embankment fill field dry density 1.74 t/m3, laboratory maximum 1.88. Pass?",
        ("Corvale Earthworks Specification.pdf",
         "Embankment fill dry density shall be not less than 95% of the laboratory maximum."),
        ("Corvale Earthworks Bill.pdf",
         "Embankment fill dry density to be not less than 97% of the laboratory maximum."),
        "The embankment fill dry density of 1.74 t/m3 is 92.6% of the laboratory maximum, "
        "below the 95% required - FAIL.",
        "92.6%",
        "The embankment fill dry density must be at least 95% of the laboratory maximum.",
        id="percent-compaction",
    ),
    pytest.param(
        "Slab bottom cover to reinforcement for the podium slab, 300 mm slab with 16 mm bars. Effective depth?",
        ("Tarnby Structural Notes.pdf", "Podium slab bottom cover to reinforcement 40 mm."),
        ("Tarnby Concrete Specification.pdf", "Podium slab bottom cover to reinforcement 50 mm."),
        "With a podium slab bottom cover of 40 mm, the effective depth is 252 mm.",
        "252 mm",
        "The podium slab bottom cover to reinforcement is 40 mm.",
        id="length-cover",
    ),
]


@pytest.mark.parametrize("question, a, b, worked, result, restated", _CASES)
def test_a_sentence_with_a_worked_result_stays_under_the_conflict(question, a, b, worked, result, restated):
    text = answer_contract.apply(worked, sources=_rows(a, b), question=question).text
    assert text.startswith(_LINE), text
    assert result in text, text


@pytest.mark.parametrize("question, a, b, worked, result, restated", _CASES)
def test_a_sentence_restating_one_value_is_still_replaced(question, a, b, worked, result, restated):
    text = answer_contract.apply(restated, sources=_rows(a, b), question=question).text
    assert text.startswith(_LINE), text
    assert restated not in text, text
