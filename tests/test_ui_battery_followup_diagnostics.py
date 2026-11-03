"""S1c follow-up (ORDERS_03NOV_0715Z): the three additive observer items.

1. _arith_inferred -- the one calculator path with no evidence (the calc tool
   was dispatched but the turn carried neither a credited figure nor
   awaiting-inputs state) records a flag, like the prose fallback.
2. _grounded for asks_for_inputs requires that the product ACTUALLY asked, so an
   evidence-free turn that did not ask is not counted as a grounded ask.
3. The per-row diagnostic: classify_route records the intent and the branch
   taken on the turn.

Structural only; no per-question strings.
"""
from __future__ import annotations

import pytest

from scripts.ui_battery_runner import (
    ROUTE_ASKS_INPUTS,
    ROUTE_PROJECT_RAG,
    _CALC_TOOL,
    _PROV_USER,
    _arithmetic_was_performed,
    _grounded,
    classify_route,
)


@pytest.mark.parametrize(
    "turn,expect_result,expect_flag",
    [
        # Bare tool, no figure, no state -> arithmetic by last resort, flagged.
        ({"tools": [_CALC_TOOL], "provenance": [], "sources": []}, True, True),
        # A credited figure -> arithmetic by evidence, NOT the inferred path.
        ({"tools": [_CALC_TOOL],
          "provenance": [{"source": "calculator", "figure": "9 m3"}], "sources": []},
         True, False),
    ],
)
def test_arith_inferred_flag_only_on_the_evidence_free_path(turn, expect_result, expect_flag):
    assert _arithmetic_was_performed(turn) is expect_result
    assert bool(turn.get("_arith_inferred")) is expect_flag


@pytest.mark.parametrize(
    "turn,expect_grounded",
    [
        # Asked (awaiting state present), only the user's own kinds -> grounded.
        ({"answer": "What platform area should I use?", "provenance": [], "sources": [],
          "awaiting_inputs": {"missing": ["platform_area"]}}, True),
        # No state, no ask phrasing -> did not ask -> NOT a grounded ask, even
        # though it carries no disallowed evidence.
        ({"answer": "The membrane lap is 150 mm.", "provenance": [], "sources": []}, False),
    ],
)
def test_grounded_asks_requires_the_product_actually_asked(turn, expect_grounded):
    assert _grounded(ROUTE_ASKS_INPUTS, turn["provenance"], turn["sources"], turn) is expect_grounded


@pytest.mark.parametrize(
    "turn,question,expect_intent,expect_branch",
    [
        ({"tools": [], "sources": [{"source_class": "project_corpus"}], "provenance": [],
          "answer": "The DNP is 365 days."},
         "What is the Defects Notification Period?", "lookup", "lookup/project"),
        ({"tools": [_CALC_TOOL], "sources": [], "provenance": [],
          "answer": "The calculator needs the platform area.",
          "awaiting_inputs": {"missing": ["platform_area"]}},
         "Scaffold load capacity for the stated configuration?", "compute", "compute/ask"),
    ],
)
def test_classify_route_records_intent_and_branch(turn, question, expect_intent, expect_branch):
    route = classify_route(turn, question)
    assert turn["_intent"] == expect_intent
    assert turn["_route_branch"] == expect_branch
    # the branch is consistent with the route returned
    assert route in (ROUTE_PROJECT_RAG, ROUTE_ASKS_INPUTS)


@pytest.mark.parametrize(
    "provenance,sources,expected_true",
    [
        ([{"source": _PROV_USER}], [], True),   # asks test still backward-compatible w/o turn
        ([{"source": "project_document"}], [], False),  # disallowed kind -> not a grounded ask
    ],
)
def test_grounded_without_a_turn_keeps_the_kinds_only_check(provenance, sources, expected_true):
    """Called with no turn (unit use), the asks clause falls back to the
    kinds-only check it had before -- backward compatible."""
    assert _grounded(ROUTE_ASKS_INPUTS, provenance, sources) is expected_true
