"""S1c (product side): a turn records the operands a tool asked for and did not
get, and a completed answer carries that state on its way out.

``note_tool_result`` reads a tool's own ``missing`` / ``needs_input`` envelope
-- the same shape the calculator's missing-operand envelope and the payment
certificate's missing-input envelope both use -- and records the field names on
the turn. ``check_result`` then puts that state on the result as
``awaiting_inputs``: the names when a tool asked, an empty list when the turn
answered. The route observer reads this instead of the answer's wording.

Structural only: the envelopes here are synthetic and name no real question.
"""
from __future__ import annotations

import logging

import pytest

from app.agents import answer_exit as ax
from app.agents.answer_exit import _missing_input_names


@pytest.mark.parametrize(
    "payload,expected",
    [
        # Calculator missing-operand envelope (construction_formulas).
        ({"status": "needs_input", "needs_input": True,
          "missing": ["platform_area"], "question": "What platform area?"},
         ["platform_area"]),
        # Payment certificate missing-input envelope (boq.py), nested in result.
        ({"status": "error",
          "result": {"needs_input": True,
                     "missing": ["contract_value", "gross_valuation"],
                     "question": "What is the contract value?"}},
         ["contract_value", "gross_valuation"]),
    ],
)
def test_a_needs_input_envelope_records_its_missing_fields(payload, expected):
    turn = ax.Turn()
    ax.note_tool_result("construction_calc", payload, turn)
    assert turn.awaiting_inputs == expected


@pytest.mark.parametrize(
    "payload",
    [
        {"status": "success", "result": {"value": 42}},
        {"status": "error", "error": "the provider was unavailable"},
    ],
)
def test_an_answered_or_failed_tool_records_no_awaiting_inputs(payload):
    """No missing/needs_input in the envelope -> nothing recorded, so the turn
    reads as "answered" rather than "asked"."""
    turn = ax.Turn()
    ax.note_tool_result("construction_calc", payload, turn)
    assert turn.awaiting_inputs == []


@pytest.mark.parametrize(
    "awaiting,expected",
    [
        (["platform_area"], {"missing": ["platform_area"]}),
        ([], {"missing": []}),
    ],
)
def test_check_result_carries_state_on_a_completed_answer(awaiting, expected):
    turn = ax.Turn(awaiting_inputs=list(awaiting))
    out = ax.check_result({"answer": "Here is the result.", "sources": []}, turn)
    assert out["awaiting_inputs"] == expected


@pytest.mark.parametrize(
    "result",
    [
        {"status": "error", "error": "heavy reasoning failed"},
        {"sources": []},
    ],
)
def test_a_result_with_no_answer_is_left_without_state(result):
    """A hard error carries no answer, so no state is asserted -- the reader's
    own fallback still runs rather than being told "not asking"."""
    turn = ax.Turn(awaiting_inputs=["contract_value"])
    out = ax.check_result(dict(result), turn)
    assert "awaiting_inputs" not in out


@pytest.mark.parametrize("value", ["a plain string", ["a", "list"]])
def test_check_result_returns_a_non_dict_unchanged(value):
    """The isinstance guard: a non-dict result passes straight through. (Also
    pins the first branch so dropping it, which would make dict(value) raise, is
    caught.)"""
    assert ax.check_result(value, ax.Turn()) is value


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({"missing": ["volume"], "needs_input": True}, ["volume"]),
        ("not a dict at all", []),
    ],
)
def test_missing_input_names_reads_only_mapping_envelopes(payload, expected):
    """The payload guard: a non-mapping yields no names rather than raising."""
    assert _missing_input_names(payload) == expected


@pytest.mark.parametrize(
    "payload",
    [
        {"missing": ["volume"], "needs_input": True},
        {"status": "error", "result": {"missing": ["contract_value"], "needs_input": True}},
    ],
)
def test_note_tool_result_with_no_turn_is_a_silent_noop(payload, caplog):
    """The turn-is-None guard: with no turn to record on, note_tool_result does
    nothing -- it does not fall into its body and swallow an AttributeError
    (which would log a 'could not record' warning). Pins the guard so dropping
    it is caught."""
    ax._TURN.set(None)
    with caplog.at_level(logging.WARNING, logger="app.agents.answer_exit"):
        ax.note_tool_result("construction_calc", payload, turn=None)
    assert not any("could not record" in r.getMessage() for r in caplog.records)
