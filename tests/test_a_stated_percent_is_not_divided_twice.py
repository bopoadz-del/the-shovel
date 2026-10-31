"""A figure the user wrote as a percentage is not divided by 100 twice.

Lane 5 on 9c1d155: "bulking 150%" ran as 1.5% (240 m3 -> 243.6 m3). The model
passed 1.5 (150% already as a fraction); the input is declared as a fraction
from 0 to 1, so 1.5 was out of range and the percent rule read it as a bare
percentage and divided again. When the user's words state the figure as that
percentage (1.5 x 100 = 150%), the value is already a fraction: it is not
divided, and the range check answers (150% is outside 0% to 100%). Invented
pit dimensions.
"""
from app.lib import construction_formulas as cf

ASK = "Pit 10 x 8 x 3 m, bulking 150%. Loose volume?"
DIMS = {"length_m": 10, "width_m": 8, "depth_m": 3}


def test_a_fraction_of_a_stated_percent_is_refused_by_range_not_shrunk():
    out = cf.run_calculation("excavation_volume", {"text": ASK, **DIMS, "bulking_factor": 1.5})
    assert out["status"] == "error"
    assert "150%" in out["error"] and "0% to 100%" in out["error"]


def test_the_same_figure_written_any_way_gets_the_same_answer():
    answers = {str(v): cf.run_calculation("excavation_volume", {"text": ASK, **DIMS, "bulking_factor": v})["status"]
               for v in (1.5, 150, "150%")}
    assert set(answers.values()) == {"error"}, answers


def test_a_bare_percentage_with_no_percent_sign_is_still_read_as_one():
    out = cf.run_calculation("excavation_volume", {"text": "Pit 10 x 8 x 3 m, bulking 25.", **DIMS, "bulking_factor": 25})
    assert out["status"] == "success" and round(out["result"]["loose_volume_m3"], 1) == 300.0


def test_stated_percents_are_read_from_the_words():
    assert cf._stated_percents("bulking 150%, waste 12.5 %, 10 per cent, 5 percent") == frozenset({150.0, 12.5, 10.0, 5.0})
