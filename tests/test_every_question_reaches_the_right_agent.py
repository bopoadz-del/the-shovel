"""Routing: does the question reach something that can answer it?

Live, 2026-09-20: "Our BOQ has 1,250 m2 of blockwork at SAR 62/m2 but site
measured 1,318 m2. What is the cost variance ...?" was answered in 4 s with
"I could not confirm this reference in the indexed project sources" -- three
asks out of three, pinned to quantity-surveyor too. The identifier extractor
read the unit RATE "62/m2" as a reference code (it looks like page "d/3/3"),
retrieval found no document called 62/m2, and the missing-reference shortcut
answered before anything could calculate. Every number the question needs is
in the question.
"""
import pytest

from app.agents.runtime import _should_short_circuit_rag_miss

_MISS = {"identifier_miss": True, "threshold_fired": True}


@pytest.mark.parametrize("rate", ["62/m2", "410/m3", "1,900/day", "95/m", "3.5/t", "12.50/hr"])
def test_a_unit_rate_is_not_a_document_reference(rate):
    q = f"Site measured 1,318 m2 against the BOQ at SAR {rate}. What is the variance?"
    assert not _should_short_circuit_rag_miss({**_MISS, "extracted_identifiers": [rate]}, None, q)


@pytest.mark.parametrize("ref", ["d/3/3", "LTR-MN-000372", "D529.2", "clause 13.5"])
def test_a_real_reference_that_retrieval_missed_still_short_circuits(ref):
    q = f"What does {ref} say?"
    assert _should_short_circuit_rag_miss({**_MISS, "extracted_identifiers": [ref]}, None, q)


def test_the_live_question_is_not_short_circuited():
    q = ("Our BOQ has 1,250 m2 of blockwork at SAR 62/m2 but site measured "
         "1,318 m2. What is the cost variance and is it within a 5% tolerance?")
    assert not _should_short_circuit_rag_miss({**_MISS, "extracted_identifiers": ["62/m2"]}, None, q)


# ── a question in quotation marks is still a question ───────────────────────
# Owner, live 0a95d03: the beam question typed WITH quotation marks around it
# got "I could not confirm this reference in the indexed project sources";
# without them, 112.5 kN·m. The extractor treats a quoted span as an exact
# reference -- right for "LTR-MN-000372", wrong for a whole sentence.

_QUOTED_QUESTIONS = [
    "maximum bending moment in a 6 m simply supported beam carrying 25 kn/m?",
    "what concrete volume is a slab 12 m by 8 m and 0.2 m thick",
    "how many days of delay reach the 10% cap",
]


@pytest.mark.parametrize("sentence", _QUOTED_QUESTIONS)
def test_a_quoted_sentence_is_not_a_document_reference(sentence):
    q = f'"{sentence}"'
    assert not _should_short_circuit_rag_miss(
        {**_MISS, "extracted_identifiers": [sentence]}, None, q)


@pytest.mark.parametrize("ref", ["ltr-mn-000372", "d/3/3", "ab-2023-101", "clause 13.5",
                                 "vol 2 specification 6 of 11"])
def test_a_quoted_reference_still_short_circuits(ref):
    q = f'What does "{ref}" say?'
    assert _should_short_circuit_rag_miss({**_MISS, "extracted_identifiers": [ref]}, None, q)
