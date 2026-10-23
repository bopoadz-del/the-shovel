"""Advance Payment percentage must not be the delay-damages daily rate.

A synthetic contract states two different percentages: Advance Payment
is 10% of the Accepted Contract Amount, and Delay Damages are 0.1% of
the Contract Price per day. The clause that introduces Advance Payment
sits immediately before the delay-damages rate. The percentage graft
used to open with that neighbouring 0.1% and call it the Advance Payment.
"""
from __future__ import annotations

import re

from app.agents.runtime import _postprocess_answer
from app.lib.construction_formulas_commercial import (
    extract_named_percentage_particular,
    parse_delay_damages_rate_percent,
)

ASK = "What is the Advance Payment percentage?"

# Invented contract. No client name, document, or corpus figure.
# One paragraph, the way a scanned clause reads: Advance Payment is
# introduced, the delay-damages daily rate is the next percentage, and
# the filled Advance Payment percentage comes after that.
SYNTHETIC_CONTRACT = (
    "Sub-Clause 14.2 Advance Payment. The amount of the advance payment "
    "is the percentage of the Accepted Contract Amount stated in the "
    "Contract Data. Sub-Clause 8.8 Delay Damages are 0.1% of the Contract "
    "Price per day. Contract Data: Advance Payment 10% of the Accepted "
    "Contract Amount. Delay Damages for the whole of the Works 0.1% of "
    "the Contract Price per day. Accepted Contract Amount excluding VAT "
    "SAR 2,400,000.00."
)


def test_advance_payment_percentage_is_not_the_delay_damages_daily_rate():
    parsed = extract_named_percentage_particular(ASK, SYNTHETIC_CONTRACT)
    assert parsed is not None
    assert parsed["percent"] == 10.0

    # The daily rate is still the delay-damages figure on this contract.
    assert parse_delay_damages_rate_percent(SYNTHETIC_CONTRACT) == 0.1

    rag = {"role": "system", "content": SYNTHETIC_CONTRACT}
    msgs = [{"role": "user", "content": ASK}]
    out = _postprocess_answer("", rag, msgs)
    first = next((ln.strip() for ln in out.splitlines() if ln.strip()), "")
    assert re.search(r"\b10\s*%", first), first
    assert not re.search(r"0\.1\s*%", first), first
    assert re.search(r"(?i)advance payment", first)

    # A model that already called the Advance Payment 0.1% must not keep
    # that sentence. 0.10% is the same daily rate and is not "10%".
    wrong = "The Advance Payment is 0.1% of the Contract Price per day."
    corrected = _postprocess_answer(wrong, rag, msgs)
    assert re.search(r"(?i)advance payment is 10\s*%", corrected)
    assert not re.search(r"(?i)advance payment is 0\.1\s*%", corrected)
    dotted = _postprocess_answer(
        "The Advance Payment is 0.10% of the Contract Price per day.",
        rag, msgs,
    )
    assert re.search(r"(?i)advance payment is 10\s*%", dotted)
    assert not re.search(r"(?i)advance payment is 0\.10\s*%", dotted)
