"""A bill extension computed from the user's own figures must pass the gate.

Live-fire find 2026-08-15 (F25). "Extend 100 square metres at the
operator-instructed rate of 250 riyals" was answered inline with SAR 25,000
-- and the cost-grounding gate replaced it with the canned rate refusal,
twice, because the derivation closure grounded pairwise SUMS and
DIFFERENCES of user-supplied figures but not PRODUCTS. The extension
qty x rate is the most basic quantity-surveying operation; refusing to
multiply two numbers the user dictated is not grounding discipline, it is
a broken closure.

Products and quotients of grounded figures now ground. A figure that is
NOT derivable from the user's numbers, retrieval, or a tool result still
refuses -- the 450-from-a-drawing fabrication class stays dead.
"""
from __future__ import annotations

import app.agents.runtime as rt

USER_MSG = {
    "role": "user",
    "content": (
        "I am instructing the rate: use 250 riyals per square metre for "
        "tiling. Extend 100 square metres at that instructed rate and show "
        "the arithmetic."
    ),
}


def gate(answer, messages):
    return rt._cost_grounding_gate(answer, None, messages)


def test_the_live_extension_now_passes():
    """RED before the fix: replaced with the canned refusal."""
    answer = "Extension: 100 m2 x SAR 250/m2 = SAR 25,000 (operator-instructed rate)."
    assert gate(answer, [USER_MSG]) == answer


def test_unit_rate_backout_passes():
    """amount / qty -- the inverse derivation a QS legitimately performs."""
    msg = {"role": "user", "content":
           "The bill shows SAR 361,000 for 950 m3 of raft concrete. What unit rate is that?"}
    answer = "That is SAR 380 /m3 (361,000 / 950)."
    assert gate(answer, [msg]) == answer


def test_a_fabricated_rate_still_refuses():
    """The gate's purpose is intact: a figure not derivable from the user's
    numbers (nor retrieval, nor tools) is refused."""
    answer = "The market rate is about SAR 6,137 /m2 for tiling."
    assert gate(answer, [USER_MSG]) == rt._CG_REFUSAL


def test_sums_and_differences_still_ground():
    msg = {"role": "user", "content":
           "Planned SAR 4,200,000 versus actual SAR 5,100,000 -- what is the variance?"}
    answer = "The overrun is SAR 900,000."
    assert gate(answer, [msg]) == answer
