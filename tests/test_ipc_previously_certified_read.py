"""X1 (exam 8): the amount certified before this period is read in every phrasing.

The IPC figure reader knew one spelling of this label: "previous certif...".
An ask that said "Previously certified 1,600,000" (the adverb form), or put
the word after the verb ("certified previously"), lost the amount. The
certificate then ran as if nothing had been certified before, and its net
figure was wrong under any reading of the ask.

The mechanism tested here: the label is a word that places the amount before
this period, next to any inflection of "certify", in either order. A figure
whose label is not of that kind is never taken as the previous amount.

Every decision runs on two inputs, one each way. All figures are invented,
and no project, document or scenario is named.
"""
from __future__ import annotations

import pytest

from app.containers.construction import ConstructionContainer
from app.containers.construction.boq import (
    _payment_figures_from_message,
    ipc_args_from_ask,
)

# --- the label is read in each form; a near-miss label is not ----------------

@pytest.mark.parametrize("ask, expected", [
    # adverb before the verb: the form the old reader dropped
    ("Previously certified 730,000. Retention 5 %.", 730_000.0),
    ("and previously certified was 410,500", 410_500.0),
    # the noun form the old reader already knew still reads
    ("previous certificates total 2,250,000", 2_250_000.0),
    # other words for "before this period"
    ("Prior certification 96,000 to date", 96_000.0),
    ("already certified 1,150,000", 1_150_000.0),
    # the word after the verb
    ("the amount certified previously 384,000", 384_000.0),
    ("certified earlier: 52,000", 52_000.0),
])
def test_previous_amount_read_in_every_form(ask, expected):
    assert _payment_figures_from_message(ask).get("previous_certified") == expected


@pytest.mark.parametrize("ask", [
    # "certified" with no word for "before" is this period's work, not the
    # previous amount
    "certified work 640,000 this period",
    "certified to date 3,100,000",
    # a "before" word on another label never pulls in a figure
    "previously agreed retention 5 %, previous valuation 900,000",
    # a rate after the label is a rate, not an amount
    "previously certified retention 10 %",
])
def test_figure_without_a_previous_certified_label_is_not_read(ask):
    assert "previous_certified" not in _payment_figures_from_message(ask)


def test_previous_amount_word_order_both_ways_same_value():
    before = _payment_figures_from_message("previously certified 270,000")
    after = _payment_figures_from_message("certified previously 270,000")
    assert before.get("previous_certified") == after.get("previous_certified") == 270_000.0


# --- the read amount reaches the tool args; an explicit arg still wins --------

@pytest.mark.parametrize("existing, expected", [
    ({}, 620_000.0),                                # filled from the ask
    ({"previous_certified": 455_000}, 455_000),     # explicit arg wins
])
def test_ipc_args_from_ask_carries_previous_amount(existing, expected):
    ask = ("Contract 5,000,000, gross valuation 2,000,000. Previously certified "
           "620,000. Retention 10 %.")
    assert ipc_args_from_ask(ask, existing)["previous_certified"] == expected


# --- end to end: the certificate deducts what the ask says was certified ------

def _payload(ask: str) -> dict:
    params = ipc_args_from_ask(ask, {})
    payload = {"message": ask, "user_message": ask, "text": ask}
    payload.update({k: v for k, v in params.items()
                    if k not in ("message", "user_message", "text")})
    return payload


@pytest.mark.parametrize("ask, previous", [
    # invented: gross 1,500,000; retention 10 % = 150,000; no advance stated
    ("Gross valuation 1,500,000 on a 6,000,000 contract. Retention 10 %. "
     "Previously certified 900,000.", 900_000.0),
    ("Gross valuation 1,500,000 on a 6,000,000 contract. Retention 10 %.", 0.0),
])
async def test_certificate_deducts_the_previous_amount_the_ask_states(ask, previous):
    result = await ConstructionContainer().payment_certificate(_payload(ask), {})
    assert result["status"] == "success", result
    assert result["deductions"]["previous_payments"] == previous
    # gross less retention less what was certified before
    assert result["payment"]["net_due_this_period"] == pytest.approx(
        1_500_000 - 150_000 - previous)
