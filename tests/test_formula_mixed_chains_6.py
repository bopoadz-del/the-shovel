"""Formula mixed scenarios, part 6: the plan's negative twins and M33 (CI part), no model (exam 6).

From the formula test plan, part 3 (mixed scenarios), section 4 "What a
coder lane writes from this". Parts 1 to 5 cover the CI chains. This file
adds what that section still lists:
- the M09 twin with a zero rate, which must be refused;
- the M07 twin with advance 0;
- M33 (live only; CI part "partial"). When the previously certified amount
  is not given, the payment step must not silently invent it. If it falls
  back to a value, it must name that input as a default it used, so the
  answer can ask for exactly that one thing. Once the amount is given,
  nothing is defaulted and the plan's turn-2 hand values hold.

Every expected value is a HAND calculation from the published definition,
written beside its row; none was copied from the product's output. Each
decision runs on two inputs, one each way. Plan tolerance: 0.5 %. All inputs
are invented numbers; no project, person or document is named.
"""
from __future__ import annotations

import pytest

from app.lib.construction_formulas import run_calculation

_REL = 5e-3  # plan tolerance for mixed scenarios: +-0.5 %


def _ok(name: str, params: dict) -> dict:
    env = run_calculation(name, params)
    assert env.get("status") == "success", (name, env)
    return env["result"]


# --------------------- M09 twin: a zero or negative rate is refused, not divided by
# crews = Q / (D x R): with R = 0 there is no answer, so no crew count may come back.
# The positive twin is a valid rate on the same volume: 640 / (10 x 28) = 2.286 -> 3.

@pytest.mark.parametrize("rate, refused, crews, crews_up", [
    (0, True, None, None),          # plan section 4: "M09 with zero rate (must refuse)"
    (-5, True, None, None),         # invented: a negative rate is no rate either
    (28, False, 2.2857, 3),         # 640 / 280 = 2.286 -> 3
    (32, False, 2.0, 2),            # invented: 640 / 320 = 2.0 -> 2 (exact, no round-up)
])
def test_m09_zero_rate_is_refused_valid_rate_sizes_crews(rate, refused, crews, crews_up):
    vol = _ok("concrete_volume", {"length_m": 40, "width_m": 20, "thickness_m": 0.8})
    assert vol["net_volume_m3"] == pytest.approx(640.0, rel=_REL)  # 40 x 20 x 0.8
    env = run_calculation("crew_sizing", {"quantity": vol["net_volume_m3"],
                                          "duration_days": 10, "rate_per_crew_day": rate})
    if refused:
        assert env.get("status") != "success", env
        assert "rate_per_crew_day" in repr(env), env
        assert "crews_required" not in repr(env.get("result") or {}), env
    else:
        assert env.get("status") == "success", env
        assert env["result"]["crews_required"] == pytest.approx(crews, rel=_REL)
        assert env["result"]["crews_required_rounded_up"] == crews_up


# ------------------------------- M07 twin: advance 0 recovers nothing
# this period = % x contract - previous; retention = rate x period;
# recovery = advance % x period; net = period - retention - recovery.

@pytest.mark.parametrize(
    "advance_pct, contract, done, total, previous, ret_rate, period, retention, recovery, net", [
        # plan section 4 twin of M07: 900 / 2,500 = 36 %; 0.36 x 8 M = 2,880,000;
        # - 1,600,000 = 1,280,000; 5 % = 64,000; advance 0 -> 0; net 1,216,000
        (0.0, 8_000_000, 900, 2_500, 1_600_000, 0.05, 1_280_000, 64_000, 0, 1_216_000),
        # the plan's M07 with its 10 % advance, the other way: recovery 128,000;
        # net 1,280,000 - 64,000 - 128,000 = 1,088,000
        (0.10, 8_000_000, 900, 2_500, 1_600_000, 0.05, 1_280_000, 64_000, 128_000, 1_088_000),
        # invented: 450 / 1,200 = 37.5 %; 0.375 x 4 M = 1,500,000; - 900,000 = 600,000;
        # 10 % = 60,000; advance 0 -> 0; net 540,000
        (0.0, 4_000_000, 450, 1_200, 900_000, 0.10, 600_000, 60_000, 0, 540_000),
    ])
def test_m07_twin_advance_zero_recovers_nothing(advance_pct, contract, done, total, previous,
                                                ret_rate, period, retention, recovery, net):
    prog = _ok("progress_quantity", {"total_qty": total, "planned_qty": done,
                                     "actual_qty": done})
    this_period = prog["actual_percent"] / 100 * contract - previous
    assert this_period == pytest.approx(period, rel=_REL)
    pay = _ok("calculate_payment", {"claimed_amount": this_period,
                                    "certified_amount": this_period,
                                    "retention_rate": ret_rate,
                                    "cumulative_previous_certified": previous,
                                    "contract_value": contract})
    assert pay["retention_held"] == pytest.approx(retention, rel=_REL)
    advance_recovery = advance_pct * this_period
    assert advance_recovery == pytest.approx(recovery, rel=_REL, abs=1e-9)
    assert pay["net_payment_due"] - advance_recovery == pytest.approx(net, rel=_REL)
    assert (advance_recovery == 0) is (advance_pct == 0)


# ---------------- M33 (CI part): the previously certified amount is not invented
# Turn 1 gives valuation to date and retention, not what was certified before.
# The payment step may not treat "not given" as a figure the user stated: if it
# falls back to a value, that input must be named as a default it used.

def _defaults_named(pay: dict) -> set[str]:
    return {d.get("parameter") for d in pay.get("defaults_used") or []}


@pytest.mark.parametrize("given_previous", [False, True])
def test_m33_missing_previous_certified_is_disclosed_not_silent(given_previous):
    params = {"claimed_amount": 1_280_000, "certified_amount": 1_280_000,
              "retention_rate": 0.05, "contract_value": 8_000_000}
    if given_previous:
        params["cumulative_previous_certified"] = 1_600_000
    env = run_calculation("calculate_payment", params)
    if env.get("status") != "success":
        # Refusing is the other acceptable answer: it must then name the input.
        assert not given_previous, env
        assert "cumulative_previous_certified" in repr(env), env
        return
    named = _defaults_named(env["result"])
    assert ("cumulative_previous_certified" in named) is (not given_previous), env


@pytest.mark.parametrize(
    "valuation, contract, previous, ret_rate, period, retention, net, cumulative", [
        # plan M33 turn 2: 2,880,000 - 1,600,000 = 1,280,000; 5 % = 64,000;
        # no advance mentioned, none applied: net 1,216,000; cumulative 2,880,000
        (2_880_000, 8_000_000, 1_600_000, 0.05, 1_280_000, 64_000, 1_216_000, 2_880_000),
        # invented: 3,300,000 - 2,500,000 = 800,000; 10 % = 80,000; net 720,000
        (3_300_000, 6_000_000, 2_500_000, 0.10, 800_000, 80_000, 720_000, 3_300_000),
    ])
def test_m33_turn_two_with_previous_given_recomputes_the_period(valuation, contract, previous,
                                                                ret_rate, period, retention,
                                                                net, cumulative):
    this_period = valuation - previous
    assert this_period == pytest.approx(period, rel=_REL)
    pay = _ok("calculate_payment", {"claimed_amount": this_period,
                                    "certified_amount": this_period,
                                    "retention_rate": ret_rate,
                                    "cumulative_previous_certified": previous,
                                    "contract_value": contract})
    assert pay["retention_held"] == pytest.approx(retention, rel=_REL)
    assert pay["net_payment_due"] == pytest.approx(net, rel=_REL)
    assert pay["cumulative_certified"] == pytest.approx(cumulative, rel=_REL)
    assert pay["percent_complete"] == pytest.approx(valuation / contract * 100, rel=_REL)
    assert "cumulative_previous_certified" not in _defaults_named(pay)
