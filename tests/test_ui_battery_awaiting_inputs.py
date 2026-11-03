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
    ROUTE_CALCULATOR,
    _CALC_TOOL,
    _arithmetic_was_performed,
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
    "missing,question",
    [
        (["platform_area"], "Scaffold load capacity for the stated configuration?"),
        (["volume"], "Carbon footprint of the concrete for the specified grade and volume?"),
    ],
)
def test_a_dispatched_calc_that_asked_is_not_arithmetic(missing, question):
    """The calculator ran but errored on a missing operand (state present, no
    calculator figure in provenance). That is an ask, not a computation, so the
    compute question routes asks_for_inputs -- not calculator."""
    turn = {
        "tools": [_CALC_TOOL], "provenance": [], "sources": [],
        "answer": "The calculator needs more inputs.",
        "awaiting_inputs": {"missing": missing},
    }
    assert _arithmetic_was_performed(turn) is False
    assert classify_route(turn, question) == ROUTE_ASKS_INPUTS


@pytest.mark.parametrize(
    "question",
    [
        "Concrete volume for a raft 30 x 20 x 1.5?",
        "Dig volume for a pit 3.4 x 2.6 x 1.15?",
    ],
)
def test_a_calc_that_produced_a_figure_is_arithmetic(question):
    """A credited calculator figure means arithmetic ran -> calculator route,
    whatever else the turn carries."""
    turn = {
        "tools": [_CALC_TOOL],
        "provenance": [{"source": "calculator", "figure": "945 m3", "inputs": {"x": 1}}],
        "sources": [], "answer": "The volume is 945 m3.",
        "awaiting_inputs": {"missing": []},
    }
    assert _arithmetic_was_performed(turn) is True
    assert classify_route(turn, question) == ROUTE_CALCULATOR


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
