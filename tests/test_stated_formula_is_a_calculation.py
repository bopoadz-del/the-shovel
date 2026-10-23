"""A question that states its own formula and inputs is a calculation.

The user wrote the expression and the value, so the turn goes to the
calculator. A unit written with a power ("kN/m^9", "mm^9") is not a formula,
and a formula with no input is a question about the formula. Synthetic asks
whose formulas and numbers appear in no battery case.
"""
from __future__ import annotations

import pytest

from app.agents.runtime import _message_is_formula_style_ask


@pytest.mark.parametrize("ask", [
    "What does c*T^9 give for c = 0.37 at a reach of 1.9 m?",
    "Using k*x**9 with k = 0.61, what is the force at 1.07 m?",
    "Evaluate a*d^9 for a = 0.29 kN/m and d = 1.45 m.",
    "What does q*u^9 give for u = 1.15 m and q = 9.7 kN/m?",
])
def test_a_stated_formula_with_an_input_is_a_calculation(ask):
    assert _message_is_formula_style_ask(ask)


@pytest.mark.parametrize("ask", [
    "Which drawing shows the 0.37 kN/m^9 load?",
    "What is the screed rate per mm^9 in the bill?",
    "What does T^9 mean in the radiation clause of the specification?",
])
def test_a_unit_power_or_a_formula_without_input_is_not(ask):
    assert not _message_is_formula_style_ask(ask)


def test_a_conflicting_excerpt_does_not_put_a_stated_formula_under_the_lookup_clamp():
    """The cause of the refusal: a lookup-shaped turn gets "answer using ONLY
    the reference context ... the context wins", so an excerpt carrying a
    different method outranks the formula the user wrote. A stated formula
    with inputs gets the calculation directive instead."""
    from app.agents.runtime import _apply_rag_context

    excerpt = {"role": "system", "content": "Note 4.1: pressure is taken as c*T^7 for this site."}
    messages = [{"role": "user", "content": "What does c*T^9 give for c = 0.37 at a reach of 1.9 m?"}]
    assert _apply_rag_context(messages, excerpt)
    folded = messages[-1]["content"]
    assert "using ONLY the reference context" not in folded
    assert "Do NOT refuse" in folded
