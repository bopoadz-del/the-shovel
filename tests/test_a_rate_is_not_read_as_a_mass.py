"""A figure the user wrote as a rate is not read as the input's own unit.

Lane 3 ring on 838e85a (3/3): "Each crane does 10 lifts/h" ran crane
planning with a 10 t crane (1 crane; hand 240 / (10 x 8) = 3). The input is
the crane's capacity in t; the model passed a bare 10, and nothing tied it
to the "lifts/h" the user wrote. When the ask writes a figure as a rate
("10 lifts/h", "12 m2 per day") and the input is not a rate, the rate goes
with the figure, so the unit check asks instead of computing. Invented asks.
"""
from app.lib import construction_formulas as cf


def test_a_lift_rate_is_asked_about_not_used_as_tonnes():
    out = cf.run_calculation("crane_planning", {
        "text": "We have 240 lifts to make. Each crane does 10 lifts/h over an 8 h day. How many cranes?",
        "total_lift_demand_tons": 240, "crane_capacity_tons": 10})
    assert out["status"] == "error" and "lifts/h" in out["error"]
    assert "?" in out["error"]


def test_a_real_capacity_still_runs():
    out = cf.run_calculation("crane_planning", {
        "text": "Lift 2,400 t with a 50 t crane, 8 h shifts.", "total_lift_demand_tons": 2400, "crane_capacity_tons": 50})
    assert out["status"] == "success"


def test_rate_words_are_recognised():
    assert cf._RATE_WORDS.match("lifts/h over an 8 h day").group(0) == "lifts/h"
    assert cf._RATE_WORDS.match("m2 per gang-day").group(0) == "m2 per gang-day"
    assert cf._RATE_WORDS.match("t crane") is None
