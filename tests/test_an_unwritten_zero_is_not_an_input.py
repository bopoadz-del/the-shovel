"""A zero the user never wrote is not the user's input.

LANE2 (a6fbb9d, 0846786, f2de03b; 1/3): a crane hire cost ask gave no crane
count, the call carried num_cranes=0, and the answer refused "The number of
cranes given (0) is outside ..." -- a figure the user never gave, reported as
theirs. A zero the caller sends that the user's own words (this turn's ask)
never write is left out: the calculator asks for a required input, or uses
an optional input's declared default. A zero the user did write is kept and
judged as before; a switch (an input whose default is True/False) keeps 0 as
"off"; a call with no user words is unchanged.
"""
import pytest

from app.lib import construction_formulas as cf

HIRE = {"crane_capacity_tons": 50, "duration_months": 6}


def _run(ask, name, params):
    token = cf.set_current_ask(ask)
    try:
        return cf.run_calculation(name, params)
    finally:
        cf._CURRENT_ASK.reset(token)


def test_a_phantom_required_zero_is_asked_for_not_refused_as_given():
    out = _run("Crane hire for 6 months, 50 t crawler crane, how much will it cost?",
               "crane_cost_estimate", {"num_cranes": 0, **HIRE})
    assert out["status"] == "error"
    assert "num_cranes" in (out.get("missing") or []), out
    assert "given (0)" not in out["error"]


def test_a_phantom_optional_zero_takes_the_declared_default():
    out = _run("2 cranes, 50 t, for 6 months", "crane_cost_estimate",
               {"num_cranes": 2, "remote_area_factor": 0, **HIRE})
    assert out["status"] == "success", out
    assert out["result"]["remote_area_factor"] == 1.0


@pytest.mark.parametrize("ask", ["Crane hire, 0 cranes, 6 months, 50 t", "zero cranes, 50 t, 6 months"])
def test_a_zero_the_user_wrote_is_kept_and_judged(ask):
    out = _run(ask, "crane_cost_estimate", {"num_cranes": 0, **HIRE})
    assert out["status"] == "error" and "given (0)" in out["error"], out


def test_a_switch_keeps_zero_as_off():
    out = _run("2 cranes, 50 t, 6 months, operator only", "crane_cost_estimate",
               {"num_cranes": 2, "include_riggers": 0, **HIRE})
    assert out["status"] == "success", out
    assert out["result"]["riggers_sar_month"] == 0


def test_a_call_with_no_user_words_is_unchanged():
    out = cf.run_calculation("crane_cost_estimate", {"num_cranes": 0, **HIRE})
    assert out["status"] == "error" and "given (0)" in out["error"], out


# LANE4 (#994 parked): the check was per ask, so any "no", "none", "without"
# or "0" anywhere turned it off for every input. It is per input now, by the
# input's own declared label: a zero word vouches only for the input it sits
# beside.
@pytest.mark.parametrize("ask", [
    "Crane hire for 6 months, 50 t crane, without riggers -- what does it cost?",
    "No rush: crane hire for 6 months, 50 t, how much?",
    "Crane hire, 50 t, 6 months, 0% remote uplift, cost?",
])
def test_a_zero_word_for_another_input_does_not_vouch_for_the_crane_count(ask):
    out = _run(ask, "crane_cost_estimate", {"num_cranes": 0, **HIRE})
    assert "num_cranes" in (out.get("missing") or []), out
    assert "given (0)" not in out.get("error", "")


@pytest.mark.parametrize("ask", ["0 cranes, 50 t, 6 months", "no cranes needed, 50 t for 6 months",
                                 "number of cranes: zero, 50 t, 6 months"])
def test_a_zero_written_beside_the_input_is_kept(ask):
    out = _run(ask, "crane_cost_estimate", {"num_cranes": 0, **HIRE})
    assert out["status"] == "error" and "given (0)" in out["error"], out


def test_the_input_words_come_from_its_declared_label():
    assert cf._zero_written_for("num_cranes", "number of cranes", "we need 0 cranes")
    assert not cf._zero_written_for("num_cranes", "number of cranes", "no riggers please, two cranes")
