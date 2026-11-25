"""A sum worked from part of an ask is not the ask's answer.

When no calculator is named, ``run_calculation`` falls back to arithmetic
the ask carries itself: an operator expression ("calculate 890 x 310") or
one quantity at one rate ("total cost of 890 m at 310"). That is right when
those operands are the ask. When the ask writes other figures too, the
expression is a fragment of a longer working (a door's width x height
inside a painting programme, one crash day at its rate inside a least-cost
crash), and returning it as "success, calculated" presents the fragment as
the result.

The rule reads only the ask's own figures: the fallback answers when every
distinct figure the ask writes is one of its operands.

Every ask here is built in this file. Each decision has two inputs, and
every expected figure is worked by hand in the comment beside it.
"""
from __future__ import annotations

import pytest

from app.lib import construction_formulas as cf
from tests._chat_path_harness import calculator_calls, install_model_stub, run_turns


# --- Decision 1: the operator expression is the whole ask, or a fragment ----

@pytest.mark.parametrize("ask, expected", [
    # 37 * 41 = 1517; the ask writes 37 and 41 only.
    ("calculate 37 x 41", ("37*41", 1517.0)),
    # (12.5 - 2) * 8 = 84; the ask writes 12.5, 2 and 8 only.
    ("how much is (12.5 - 2) * 8", ("(12.5 - 2) * 8", 84.0)),
])
def test_an_expression_that_uses_every_written_figure_is_evaluated(ask, expected):
    assert cf.self_contained_arithmetic(ask) == expected


@pytest.mark.parametrize("ask", [
    # 1.2 x 2.4 is one opening; 60 m2 and 3 coats are also written.
    "Hall 60 m2, one 1.2 x 2.4 opening, 3 coats. How many litres of paint?",
    # 7 x 9 is one bay; 14 bays and 3.5 m are also written.
    "A 7 x 9 bay repeated 14 times at 3.5 m spacing: how much steel?",
])
def test_an_expression_that_leaves_written_figures_out_is_a_fragment(ask):
    assert cf.self_contained_arithmetic(ask) is None


# --- Decision 2: quantity at a rate is the whole ask, or a fragment ---------

@pytest.mark.parametrize("ask, total", [
    # 45 * 70 = 3150
    ("Total cost of kerbs (45 m at 70)?", 3150.0),
    # 260 * 18 = 4680
    ("Total price of blinding (260 m2 at 18)?", 4680.0),
])
def test_one_quantity_at_one_rate_is_still_multiplied(ask, total):
    out = cf.run_calculation(None, {"text": ask})
    assert out["status"] == "success", out
    assert out["calculation"] == "arithmetic"
    assert out["result"]["total_cost"] == total


@pytest.mark.parametrize("ask", [
    # One activity's crash day at its rate; four other figures are written.
    "Cut 4 days at least cost. Crash per day: P up to 3 d at 2,500, Q 2 d at 6,000.",
    # One trade line at its rate; the second line and the 6 days are written.
    "Total cost for 6 days: labour 3 d at 900, plant 5 d at 1,400?",
])
def test_one_rate_line_inside_a_longer_working_is_not_the_total(ask):
    out = cf.run_calculation(None, {"text": ask})
    assert out.get("calculation") != "arithmetic", out
    assert out["status"] == "error"
    assert "Unknown calculation" in out["error"]


# --- Decision 3: what counts as a written figure ----------------------------

@pytest.mark.parametrize("ask, operands", [
    # The "2" of m2 is a unit, not a figure.
    ("Total cost of screed 75 m2 at 16?", (75, 16)),
    # The "30" of C30 is a grade code, not a figure.
    ("Total cost of C30 concrete, 40 m3 at 95?", (40, 95)),
])
def test_unit_and_code_digits_are_not_written_figures(ask, operands):
    assert cf.operands_are_the_whole_ask(ask, operands)


@pytest.mark.parametrize("ask, operands", [
    # A figure written twice is covered once.
    ("890 m at 310 - total cost of the 890 m run?", (890, 310)),
    ("Total for 12 x 4; check 12 x 4 again", (12, 4)),
])
def test_a_repeated_figure_is_covered_once(ask, operands):
    assert cf.operands_are_the_whole_ask(ask, operands)


@pytest.mark.parametrize("ask, operands", [
    ("Total cost of 50 m at 20 plus 15 m at 30?", (50, 20)),
    ("calculate 9 x 11 for each of 6 floors", (9, 11)),
])
def test_an_unused_written_figure_breaks_the_cover(ask, operands):
    assert not cf.operands_are_the_whole_ask(ask, operands)


@pytest.mark.parametrize("ask", ["", "how much is it in total?"])
def test_an_ask_that_writes_no_figure_is_not_covered(ask):
    assert not cf.operands_are_the_whole_ask(ask, (3, 4))


# --- Decision 4: through the chat path (model stubbed) ----------------------

def _arithmetic_successes(out: dict) -> list[dict]:
    return [
        call.get("result") for call in calculator_calls(out)
        if isinstance(call.get("result"), dict)
        and call["result"].get("calculation") == "arithmetic"
        and call["result"].get("status") == "success"
    ]


@pytest.mark.parametrize("ask", [
    "Office floor 80 m2, 36 m of walls, 3 rooms, 2.7 m ceiling, one 0.8 x 2.0 "
    "door per room. One coat of paint, a painter does 4 m2 per hour, 2 painters, "
    "8 h days. How much time is that?",
    "Shop floor 150 m2, 52 m of walls, 5 rooms, 3.2 m ceiling, one 1.0 x 2.2 "
    "door per room. Two coats, a painter does 5 m2 per hour, 3 painters, "
    "7 h days. How many days?",
])
async def test_chat_path_does_not_answer_a_programme_with_one_door(monkeypatch, ask):
    install_model_stub(monkeypatch)
    (out,) = await run_turns([ask])
    assert _arithmetic_successes(out) == [], out.get("tool_calls")
