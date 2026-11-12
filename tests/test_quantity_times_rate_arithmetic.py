"""A total/cost ask that carries its own quantity and unit rate is arithmetic.

Live C25 (unseen v2, 12 Nov): "Total cost of storm water pipe (890 m at 310)?"
came back 278,900 — the model multiplied the two supplied figures itself and
got it wrong by 3,000 (890 × 310 = 275,900). Arithmetic must never be done by
the model: when the ask carries the operands, the calculator block multiplies
them deterministically and the answer cites the tool result as calculated.

Two inputs per decision (owner rule): the mechanism is proven on two operand
pairs, not a single pinned question.
"""
from __future__ import annotations

import ast

import pytest

from app.agents import runtime as rt
from app.lib import construction_formulas as cf
from app.lib.construction_formulas_commercial import (
    query_is_quantity_times_rate,
    quantity_times_rate_from_ask,
)


# ── quantity × unit rate (the C25 class) ───────────────────────────────────

@pytest.mark.parametrize(
    ("ask", "quantity", "unit_rate", "total"),
    [
        ("Total cost of storm water pipe (890 m at 310)?", 890.0, 310.0, 275900.0),
        ("Total cost of paving (120 m2 at 55)?", 120.0, 55.0, 6600.0),
        ("Total price of formwork (340 m2 at 12)?", 340.0, 12.0, 4080.0),
    ],
)
def test_quantity_times_rate_multiplication_is_deterministic(
    ask, quantity, unit_rate, total,
):
    assert query_is_quantity_times_rate(ask)
    assert quantity_times_rate_from_ask(ask) == (quantity, unit_rate)
    out = cf.run_calculation(None, {"text": ask})
    assert out["status"] == "success"
    assert out["calculation"] == "arithmetic"
    assert out["result"]["total_cost"] == total


# ── explicit operator arithmetic (890 × 310) ──────────────────────────────

@pytest.mark.parametrize(
    ("ask", "expected"),
    [
        ("calculate 890 × 310", ("890 * 310", 275900.0)),
        ("how much is 12 x 4", ("12*4", 48.0)),
        ("evaluate 100 / 4", ("100 / 4", 25.0)),
    ],
)
def test_explicit_multiplication_is_evaluated_not_left_to_the_model(ask, expected):
    assert cf.self_contained_arithmetic(ask) == expected


@pytest.mark.parametrize(
    ("ask", "expected"),
    [
        ("calculate 890 × 310", 275900.0),
        ("how much is 12 x 4", 48.0),
    ],
)
def test_explicit_multiplication_runs_through_the_calculator(ask, expected):
    out = cf.run_calculation(None, {"text": ask})
    assert out["status"] == "success"
    assert out["result"] == expected


# ── routing ────────────────────────────────────────────────────────────────

def test_quantity_times_rate_is_a_formula_style_ask():
    ask = "Total cost of storm water pipe (890 m at 310)?"
    assert rt._message_is_formula_style_ask(ask)
    assert rt._looks_like_self_contained_calculation(ask)


def test_arithmetic_predispatch_is_answered_from_the_result():
    # A deterministic run with no registered name (calculation="arithmetic")
    # must still be answered from the tool result, never recomputed.
    results = (
        {"ok": True, "result": {"calculation": "arithmetic", "result": 275900.0}},
        {
            "ok": True,
            "result": {
                "calculation": "arithmetic",
                "quantity": 890.0,
                "unit_rate": 310.0,
                "result": {"total_cost": 275900.0},
            },
        },
    )
    for result in results:
        instr = rt._formula_predispatch_instruction(None, result)
        assert "already been run" in instr
        assert "Do not answer from Master Corpus excerpts" in instr


# ── the vetted AST walk refuses non-numeric constants, never evaluates ─────

@pytest.mark.parametrize("expr", ["True", "None", "'seven'"])
def test_non_numeric_constant_is_refused_not_evaluated(expr):
    # bool / None / string constants are not arithmetic: the walk raises and
    # the safe evaluator reports None rather than a numeric value.
    assert cf._safe_arithmetic_value(expr) is None


@pytest.mark.parametrize(
    "expr",
    ["x", "open('passwd').read()", "[1, 2][0]"],
)
def test_disallowed_ast_nodes_raise_in_the_vetted_walk(expr):
    # Names, calls, subscripts (and every other non-arithmetic node) must be
    # rejected by the walk itself, never evaluated.
    tree = ast.parse(expr, mode="eval")
    with pytest.raises(ValueError):
        cf._eval_arithmetic_ast(tree)


# ── document lookups and dimensions are not hijacked ──────────────────────

def test_a_bare_rate_lookup_is_not_a_multiplication():
    assert not query_is_quantity_times_rate("what is the rate for 600x600 floor tiles")


def test_a_quantity_at_a_dimension_is_not_a_rate():
    # "310 mm" is a dimension, not a unit rate.
    assert not query_is_quantity_times_rate("the pipe is 890 m at 310 mm diameter")


@pytest.mark.parametrize(
    "text",
    ["the pipe runs 890 m at 310", "the slab is 120 m2 at 55"],
)
def test_a_quantity_at_a_rate_without_a_total_word_is_not_arithmetic(text):
    # A quantity joined to a rate by "at" is only a multiplication when the
    # ask also names a total/cost. Without that word it is not arithmetic.
    assert not query_is_quantity_times_rate(text)
    assert quantity_times_rate_from_ask(text) is None


def test_a_document_question_is_untouched():
    assert not query_is_quantity_times_rate("what is the specification of the 890 m pipe")
    assert not query_is_quantity_times_rate("list the documents in this project")


def test_a_registered_volume_calc_still_wins():
    ask = "calculate the concrete volume for a raft 25m x 18m x 1.2m thick"
    assert not query_is_quantity_times_rate(ask)
    assert rt._looks_like_self_contained_calculation(ask)
