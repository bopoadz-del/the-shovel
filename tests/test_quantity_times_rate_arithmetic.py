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

from app.agents import runtime as rt
from app.lib import construction_formulas as cf
from app.lib.construction_formulas_commercial import (
    query_is_quantity_times_rate,
    quantity_times_rate_from_ask,
)


# ── quantity × unit rate (the C25 class) ───────────────────────────────────

def test_c25_storm_water_pipe_multiplication_is_deterministic():
    ask = "Total cost of storm water pipe (890 m at 310)?"
    assert query_is_quantity_times_rate(ask)
    assert quantity_times_rate_from_ask(ask) == (890.0, 310.0)
    out = cf.run_calculation(None, {"text": ask})
    assert out["status"] == "success"
    assert out["calculation"] == "arithmetic"
    assert out["result"]["total_cost"] == 275900.0


def test_second_operand_pair_is_multiplied_too():
    ask = "Total cost of paving (120 m2 at 55)?"
    assert query_is_quantity_times_rate(ask)
    assert quantity_times_rate_from_ask(ask) == (120.0, 55.0)
    out = cf.run_calculation(None, {"text": ask})
    assert out["status"] == "success"
    assert out["result"]["total_cost"] == 6600.0


# ── explicit operator arithmetic (890 × 310) ──────────────────────────────

def test_explicit_multiplication_is_evaluated_not_left_to_the_model():
    assert cf.self_contained_arithmetic("calculate 890 × 310") == ("890 * 310", 275900.0)
    assert cf.self_contained_arithmetic("how much is 12 x 4") == ("12*4", 48.0)


def test_explicit_multiplication_runs_through_the_calculator():
    out = cf.run_calculation(None, {"text": "calculate 890 × 310"})
    assert out["status"] == "success"
    assert out["result"] == 275900.0


# ── routing ────────────────────────────────────────────────────────────────

def test_quantity_times_rate_is_a_formula_style_ask():
    ask = "Total cost of storm water pipe (890 m at 310)?"
    assert rt._message_is_formula_style_ask(ask)
    assert rt._looks_like_self_contained_calculation(ask)


# ── document lookups and dimensions are not hijacked ──────────────────────

def test_a_bare_rate_lookup_is_not_a_multiplication():
    assert not query_is_quantity_times_rate("what is the rate for 600x600 floor tiles")


def test_a_quantity_at_a_dimension_is_not_a_rate():
    # "310 mm" is a dimension, not a unit rate.
    assert not query_is_quantity_times_rate("the pipe is 890 m at 310 mm diameter")


def test_a_document_question_is_untouched():
    assert not query_is_quantity_times_rate("what is the specification of the 890 m pipe")
    assert not query_is_quantity_times_rate("list the documents in this project")


def test_a_registered_volume_calc_still_wins():
    ask = "calculate the concrete volume for a raft 25m x 18m x 1.2m thick"
    assert not query_is_quantity_times_rate(ask)
    assert rt._looks_like_self_contained_calculation(ask)
