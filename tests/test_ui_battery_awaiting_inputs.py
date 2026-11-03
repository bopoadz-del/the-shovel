"""S1c: the asks-for-inputs route is read from STATE, not from the answer's
wording.

The product emits, on the end event, the operands a tool asked for and did not
get (``awaiting_inputs = {"missing": [...]}``), or an empty list when the turn
answered. The observer trusts that state over any phrase. A completed answer
that closes with an additive offer ("if you give me the Accepted Contract
Amount I can convert ...") is not an ask: the same question run twice must not
flip the route because the model rephrased its closing courtesy.

Only when a turn carries no such state does the observer fall back to the
answer's phrasing, and it records ``_asks_inferred`` so that fallback can be
watched to zero. Structural only -- no per-question strings.
"""
from __future__ import annotations

import pytest

from scripts.ui_battery_runner import (
    ROUTE_ASKS_INPUTS,
    _asks_for_inputs,
    classify_route,
    turn_from_events,
)

# The two real closing sentences Q2 produced across runs on one sha
# (ORDERS_03NOV_0530Z): an answered contract lookup that then OFFERS to compute.
# Both match the old prose regex; neither is an ask.
Q2_OFFER_PHRASINGS = [
    "Clause 8.7 sets delay damages at 0.1% per day, capped at 10%. If you give me "
    "the Accepted Contract Amount (or want me to search the contract for it), I can "
    "convert that into a daily money figure.",
    "Clause 8.7 sets delay damages at 0.1% per day, capped at 10%. To convert the "
    "rate into a money figure I would need the Accepted Contract Amount -- say the "
    "word and I'll search the contract for it.",
]


@pytest.mark.parametrize("answer", Q2_OFFER_PHRASINGS)
def test_empty_state_overrides_an_offer_phrase(answer):
    """awaiting_inputs == {"missing": []} means the turn answered: not an ask,
    even though the closing offer matches the input-request phrasing."""
    turn = {"answer": answer, "provenance": [], "sources": [],
            "awaiting_inputs": {"missing": []}}
    assert _asks_for_inputs(turn) is False


@pytest.mark.parametrize(
    "missing",
    [["platform_area"], ["contract_value", "gross_valuation"]],
)
def test_missing_state_is_an_ask_regardless_of_wording(missing):
    """A non-empty missing list is authoritative: the turn asked, whatever the
    answer text happens to say (here it carries no request phrase at all)."""
    turn = {"answer": "Here is the certificate layout.", "provenance": [],
            "sources": [], "awaiting_inputs": {"missing": missing}}
    assert _asks_for_inputs(turn) is True


@pytest.mark.parametrize(
    "answer,expected",
    [
        ("The calculator needs the platform area. What value should I use?", True),
        ("The membrane lap is 150 mm per the specification.", False),
    ],
)
def test_without_state_falls_back_to_prose_and_marks_inferred(answer, expected):
    """No awaiting_inputs on the turn -> the observer reads the phrasing (the
    legacy path) and records that it had to infer."""
    turn = {"answer": answer, "provenance": [], "sources": []}
    assert _asks_for_inputs(turn) is expected
    assert turn.get("_asks_inferred") is True


@pytest.mark.parametrize(
    "missing,question",
    [
        (["platform_area"], "Scaffold load capacity for the stated configuration?"),
        (["contract_value", "gross_valuation"], "Issue a payment certificate."),
    ],
)
def test_compute_turn_with_missing_state_routes_to_asks(missing, question):
    """End to end through the observer: a compute question whose tool asked for
    operands (state present) is routed asks_for_inputs, with no arithmetic run."""
    events = [
        {"type": "end", "content": "The calculator needs more inputs.",
         "sources": [], "provenance": [],
         "awaiting_inputs": {"missing": missing}},
    ]
    turn = turn_from_events(events)
    assert turn["awaiting_inputs"] == {"missing": missing}
    assert classify_route(turn, question) == ROUTE_ASKS_INPUTS


@pytest.mark.parametrize(
    "has_key,payload",
    [
        (True, {"type": "end", "content": "x", "sources": [], "provenance": [],
                "awaiting_inputs": {"missing": ["volume"]}}),
        (False, {"type": "end", "content": "x", "sources": [], "provenance": []}),
    ],
)
def test_turn_from_events_carries_state_only_when_present(has_key, payload):
    """The observable turn exposes awaiting_inputs exactly when the end event
    carried it; absent, the key is not invented (the prose fallback decides)."""
    turn = turn_from_events([payload])
    assert ("awaiting_inputs" in turn) is has_key
