"""A per-day rate gives days, and a minus sign the user wrote is never dropped.

Order 0345Z (LANE2 on 1746046, 3/3): "Plastering 4800 m2, gang does -120 m2
per day. How many days?" -- the model passed productivity=120 (sign lost),
the unit words written on "-120" were never matched to 120, and the answer
read "manhours required 40 h". A negative rate is invalid input: it is
refused in the input's own words and asked for again, never normalised.
"""
from app.lib import construction_formulas as cf

NEGATIVE = "Plastering 4800 m2, gang does -120 m2 per day. How many days?"
POSITIVE = "Plastering 4800 m2, gang does 120 m2 per day. How many days?"


def _run(ask, **params):
    return cf.run_calculation("productivity_manpower_duration", {"text": ask, **params})


def test_quantity_and_a_rate_per_day_give_days_not_man_hours():
    out = _run(POSITIVE, quantity=4800, productivity=120)["result"]
    assert out.get("duration") == 40.0
    assert "manhours_required" not in out or out.get("manhours_required") is None


def test_a_dropped_minus_sign_is_restored_and_the_rate_refused():
    out = _run(NEGATIVE, quantity=4800, productivity=120)
    assert out["status"] == "error"
    assert "-120 was given" in out["error"] and "greater than zero" in out["error"]
    assert out.get("needs_input") and "daily_production" in out.get("missing", [])
    assert "manhours" not in str(out).lower() or "manhours_required" not in str(out.get("result") or "")


def test_a_negative_rate_passed_as_written_is_refused_too():
    out = _run(NEGATIVE, quantity=4800, productivity=-120)
    assert out["status"] == "error" and "greater than zero" in out["error"]


def test_a_range_dash_is_not_a_minus_sign():
    assert cf._written_only_negative(150, "between 120-150 m2 per day") is False
    assert cf._written_only_negative(120, "gang does -120 m2 per day") is True
    assert cf._written_only_negative(120, "120 m2 per day, not -120") is False


# LANE2 on f2de03b, 3/3: the call reaches the calculator with the user's
# words only in this turn's ask (not in ``text``), the rate under any key,
# as a number or as "120 m2/day", sometimes beside the model's own positive
# paraphrase. The user's minus survives every one of those shapes.
import pytest  # noqa: E402


@pytest.fixture
def negative_ask():
    token = cf.set_current_ask(NEGATIVE)
    yield
    cf._CURRENT_ASK.reset(token)


@pytest.mark.parametrize("params", [
    {"quantity": 4800, "productivity": 120},
    {"quantity": 4800, "productivity": 120, "unit": "m2/day"},
    {"quantity": 4800, "productivity": "120 m2/day"},
    {"quantity": 4800, "daily_production": 120},
    {"quantity": 4800, "productivity": 120, "text": "Plastering 4800 m2 at 120 m2/day"},
])
def test_the_users_minus_survives_every_call_shape(negative_ask, params):
    out = cf.run_calculation("productivity_manpower_duration", params)
    assert out["status"] == "error", out
    assert "-120" in out["error"], out["error"]
    assert not (out.get("result") or {}).get("duration")


def test_a_positive_rate_in_any_shape_still_gives_days():
    token = cf.set_current_ask(POSITIVE)
    try:
        for params in ({"quantity": 4800, "productivity": "120 m2/day"}, {"quantity": 4800, "productivity": 120}):
            assert cf.run_calculation("productivity_manpower_duration", params)["result"]["duration"] == 40.0
    finally:
        cf._CURRENT_ASK.reset(token)


def test_the_model_is_told_to_pass_figures_verbatim():
    """The contract: each figure as the user wrote it, as a string with its
    sign and unit; signs never changed, units never converted or normalised."""
    from app.agents.driver import tools as driver_tools

    driver_tools._SCHEMA_CACHE.clear()
    fn = driver_tools.calculator_schema(None)["function"]
    text = fn["description"]
    assert "exactly as the user wrote it, as a string with its sign and unit" in text
    assert "never change its sign, never convert or normalise its unit" in text
    assert "sign and unit included" in fn["parameters"]["properties"]["params"]["description"]


# LANE2/LANE3 (#993 parked on a split): the model stripped the minus for every
# alias of the rate. Every alias this calculator takes, in every shape a call
# can carry, refuses the user's -120 in its own words; never a generic
# "needs paired inputs", never a duration.
@pytest.mark.parametrize("key", ["productivity", "daily_production", "productivity_rate"])
@pytest.mark.parametrize("value", [120, -120, "120 m2 per day", "-120 m2 per day"])
def test_every_rate_alias_in_every_shape_refuses_the_users_minus(negative_ask, key, value):
    out = cf.run_calculation("productivity_manpower_duration", {"quantity": 4800, key: value})
    assert out["status"] == "error", out
    # Refused in the user's own figure and asked again: by the formula
    # ("must be greater than zero; -120 was given") or by the input's range
    # ("given (-120) is outside ... accepts (0 to ...)").
    assert "-120" in out["error"], out["error"]
    assert out.get("needs_input") and out.get("question"), out
    assert "paired inputs" not in out["error"]
    assert not (out.get("result") or {}).get("duration")


@pytest.mark.parametrize("key", ["productivity", "daily_production", "productivity_rate"])
@pytest.mark.parametrize("value", [120, "120 m2 per day"])
def test_every_rate_alias_with_a_positive_rate_gives_days(key, value):
    token = cf.set_current_ask(POSITIVE)
    try:
        out = cf.run_calculation("productivity_manpower_duration", {"quantity": 4800, key: value})
    finally:
        cf._CURRENT_ASK.reset(token)
    assert out["status"] == "success", out
    assert out["result"]["duration"] == 40.0
