"""A series of decimals the user typed is not a pasted bill line.

A pasted bill line is code, description, quantity, and its description is
words. When the description field only had to be "not a comma", a monthly
series such as "0.5, 1.5, 2" read as a bill line, the turn was taken as an
inline-BOQ ask, and the calculators were hidden from the model.
"""
from __future__ import annotations

import pytest

from app.agents.runtime import _inline_boq_hard_excludes
from app.core.site_vocab import message_has_inline_boq_lines

_CALCULATORS = {"construction_calc", "formula_executor", "formula_executor_v2"}

# Inputs built here; none is taken from a product answer.
SERIES_ASKS = (
    "Monthly progress value: 0.5, 1.5, 2.5, 2.0, 0.5 M over five months. "
    "Retention 5 %. What is the cumulative curve?",
    "Daily output in tonnes 4.25, 6.75, 8 then 7.5; average it and give the "
    "peak day.",
)

BILL_LINE_ASKS = (
    "Price these lines:\n2.1,Formwork to slab soffit,340,m2,38,12920\n"
    "2.2,Reinforcement,18,t,3100,55800",
    "Lines from the bill:\n7.4, Gypsum ceiling board, 125, m2\n"
    "7.5, Paint two coats, 410, m2",
)


@pytest.mark.parametrize("text", SERIES_ASKS)
def test_decimal_series_is_not_a_bill_line(text):
    assert message_has_inline_boq_lines(text) is False


@pytest.mark.parametrize("text", BILL_LINE_ASKS)
def test_bill_line_with_worded_description_is_read(text):
    assert message_has_inline_boq_lines(text) is True


@pytest.mark.parametrize("text", SERIES_ASKS)
def test_series_ask_keeps_the_calculators(text):
    assert _inline_boq_hard_excludes(text) == set()


@pytest.mark.parametrize("text", BILL_LINE_ASKS)
def test_bill_line_ask_still_hides_the_calculators(text):
    assert _inline_boq_hard_excludes(text) == _CALCULATORS
