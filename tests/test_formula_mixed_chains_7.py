"""Formula mixed scenarios, part 7: the other side of M11, M16, M21 (exam 6).

From the formula test plan, part 3 (mixed scenarios). Parts 1 to 6 already
run all of M01-M33 in CI; part 1 runs M11, M16 and M21 on their surplus,
area-given and days-given cases only. This file adds the other side of each
decision in those three chains: a deficit and a balanced site (M11), an area
derived from width x height and an area that cannot be derived (M16), damages
from a rate on the contract amount and no total without days (M21). Each chain calls the calculators in order
through ``run_calculation`` (the entry point ``construction_calc`` uses); the
named output of one step is typed into the named input of the next, and every
intermediate in the plan's "Hand" line is asserted, not only the last.

Every expected value is a HAND calculation from the published definition,
written beside its row; none was copied from the product's output. Each
chain runs on the plan's case and an invented case worked by hand the same
way. Where a step decides something (surplus or deficit, area given or
derived, a total only when days are given), one input goes each way. Plan
tolerance: 0.5 % on any number; counts rounded up are exact.

Chains here: M11, M16 and M21. All inputs are invented numbers; no project,
person or document is named.

Conventions, read from each docstring before writing the test (plan rule for
a flagged convention):
- M11 ``cut_fill_balance``: balance = cut - fill in bank m3, positive is a
  surplus to export, negative a deficit to import; the loose volume to haul
  is |balance| x (1 + bulking), as a fraction. Truck loads are the plan's
  arithmetic step (loose / truck size, rounded up), then ``unit_cost_total``.
- M16 ``wind_load_on_formwork``: q = 0.613 V^2 (the plan accepts it: 245.2 Pa
  against 0.5 x 1.225 x V^2 = 245.0 Pa, inside tolerance), F = q x A x shape
  factor, M = F x h / 2. With no area the area is width x height. The ballast
  is the plan's arithmetic step (no formula): W = FoS x M / lever.
- M21 ``delay_damages_daily`` gives ``total_amount`` only when days are
  given; ``calculate_payment`` takes retention as a fraction. Payable after
  damages is the plan's arithmetic step: net due - damages.
"""
from __future__ import annotations

import math

import pytest

from app.lib.construction_formulas import run_calculation

_REL = 5e-3  # plan tolerance for mixed scenarios: +-0.5 %
_G = 9.81  # m/s2, kN to tonnes for the ballast line


def _ok(name: str, params: dict) -> dict:
    env = run_calculation(name, params)
    assert env.get("status") == "success", (name, env)
    return env["result"]


# ------------------------------------------- M11 cut and fill -> loads -> haul cost
# balance = cut - fill; loose = |balance| x (1 + b); loads = ceil(loose / truck);
# cost = loads x rate per load.

@pytest.mark.parametrize(
    "cut, fill, bulking, truck, rate, balance, status, loose, loads, cost", [
        # plan: 4,200 - 3,000 = 1,200 surplus; 1,200 x 1.2 = 1,440 loose;
        # 1,440 / 12 = 120 loads (exact); 120 x 45 = 5,400
        (4_200, 3_000, 0.20, 12, 45, 1_200, "export_surplus", 1_440, 120, 5_400),
        # invented: 2,500 - 3,100 = -600 deficit; 600 x 1.25 = 750 loose;
        # 750 / 12 = 62.5 -> 63 loads (rounded up); 63 x 50 = 3,150
        (2_500, 3_100, 0.25, 12, 50, -600, "import_deficit", 750, 63, 3_150),
    ])
def test_m11_cut_fill_balance_then_loads_then_haul_cost(cut, fill, bulking, truck, rate,
                                                        balance, status, loose, loads, cost):
    bal = _ok("cut_fill_balance", {"cut_volume_m3": cut, "fill_volume_m3": fill,
                                   "bulking_factor": bulking})
    assert bal["balance_bank_m3"] == pytest.approx(balance, rel=_REL)
    assert bal["status"] == status
    assert bal["haul_loose_m3"] == pytest.approx(loose, rel=_REL)
    n_loads = math.ceil(bal["haul_loose_m3"] / truck - 1e-9)
    assert n_loads == loads
    haul = _ok("unit_cost_total", {"quantity": n_loads, "unit_rate": rate})
    assert haul["total_cost"] == pytest.approx(cost, rel=_REL)


@pytest.mark.parametrize("cut, fill, status, loose", [
    (3_000, 3_000, "balanced", 0),          # invented: nothing to move, nothing to haul
    (3_000, 2_999, "export_surplus", 1.2),  # invented: 1 m3 surplus x 1.2 = 1.2 loose
])
def test_m11_balanced_site_hauls_nothing(cut, fill, status, loose):
    bal = _ok("cut_fill_balance", {"cut_volume_m3": cut, "fill_volume_m3": fill,
                                   "bulking_factor": 0.20})
    assert bal["status"] == status
    assert bal["haul_loose_m3"] == pytest.approx(loose, abs=1e-6)


# ------------------------------------ M16 formwork wind -> overturning -> ballast
# q = 0.5 x 1.225 x V^2 (hand); F = q x A x shape; M = F x h / 2;
# W = FoS x M / lever; tonnes = W / 9.81.

@pytest.mark.parametrize(
    "params, fos, lever, q_kpa, force, moment, ballast_kn, ballast_t", [
        # plan: q = 0.5 x 1.225 x 400 = 245.0 Pa = 0.245 kPa; F = 0.245 x 1.3 x 80
        # = 25.48 kN; M = 25.48 x 4 / 2 = 50.96 kN.m; W = 1.5 x 50.96 / 1.5
        # = 50.96 kN = 5.20 t
        ({"wind_velocity_m_s": 20, "formwork_area_m2": 80, "formwork_height_m": 4,
          "shape_factor": 1.3}, 1.5, 1.5, 0.245, 25.48, 50.96, 50.96, 5.195),
        # invented, no area given: A = 5 x 3 = 15 m2; q = 0.5 x 1.225 x 625
        # = 382.8 Pa = 0.3828 kPa; F = 0.3828 x 1.2 x 15 = 6.891 kN;
        # M = 6.891 x 3 / 2 = 10.336 kN.m; base 2 m, lever 1.0 m:
        # W = 1.5 x 10.336 / 1.0 = 15.50 kN = 1.580 t
        ({"wind_velocity_m_s": 25, "formwork_width_m": 5, "formwork_height_m": 3,
          "shape_factor": 1.2}, 1.5, 1.0, 0.3828, 6.891, 10.336, 15.50, 1.580),
    ])
def test_m16_formwork_wind_then_overturning_then_ballast(params, fos, lever, q_kpa, force,
                                                         moment, ballast_kn, ballast_t):
    wind = _ok("wind_load_on_formwork", params)
    assert wind["wind_pressure_kpa"] == pytest.approx(q_kpa, rel=_REL)
    assert wind["wind_force_kn"] == pytest.approx(force, rel=_REL)
    assert wind["overturning_moment_kn_m"] == pytest.approx(moment, rel=_REL)
    w_kn = fos * wind["overturning_moment_kn_m"] / lever
    assert w_kn == pytest.approx(ballast_kn, rel=_REL)
    assert w_kn / _G == pytest.approx(ballast_t, rel=_REL)


@pytest.mark.parametrize("params", [
    {"wind_velocity_m_s": 20, "formwork_height_m": 4, "shape_factor": 1.3},  # no area, no width
    {"wind_velocity_m_s": 20, "formwork_width_m": 5, "shape_factor": 1.3},   # no area, no height
])
def test_m16_missing_area_is_refused_not_defaulted(params):
    env = run_calculation("wind_load_on_formwork", params)
    assert env.get("status") != "success", env
    assert "area" in repr(env), env
    assert "wind_force_kn" not in repr(env.get("result") or {}), env


# --------------------------- M21 per-day damages -> payment certificate -> deduction
# damages = per day x days; retention = rate x certified; net = certified - retention;
# cumulative = previous + certified; % = cumulative / contract; payable = net - damages.

@pytest.mark.parametrize(
    "damages_in, claimed, certified, ret, previous, contract, damages, retention, net,"
    " cumulative, pct, disputed, payable", [
        # plan: 15,000 x 12 = 180,000; 5 % of 2,250,000 = 112,500; net 2,137,500;
        # 3,000,000 + 2,250,000 = 5,250,000 = 43.75 % of 12 M; disputed 150,000;
        # payable 2,137,500 - 180,000 = 1,957,500
        ({"amount_per_day": 15_000, "days": 12}, 2_400_000, 2_250_000, 0.05, 3_000_000,
         12_000_000, 180_000, 112_500, 2_137_500, 5_250_000, 43.75, 150_000, 1_957_500),
        # invented, damages as a rate: 0.1 % x 5,000,000 = 5,000 a day x 9 = 45,000;
        # 10 % of 840,000 = 84,000; net 756,000; 1,500,000 + 840,000 = 2,340,000
        # = 46.8 % of 5 M; disputed 60,000; payable 756,000 - 45,000 = 711,000
        ({"rate_percent": 0.1, "contract_amount": 5_000_000, "days": 9}, 900_000, 840_000,
         0.10, 1_500_000, 5_000_000, 45_000, 84_000, 756_000, 2_340_000, 46.8, 60_000, 711_000),
    ])
def test_m21_damages_then_payment_then_deduction(damages_in, claimed, certified, ret, previous,
                                                 contract, damages, retention, net, cumulative,
                                                 pct, disputed, payable):
    ld = _ok("delay_damages_daily", damages_in)
    assert ld["total_amount"] == pytest.approx(damages, rel=_REL)
    pay = _ok("calculate_payment", {"claimed_amount": claimed, "certified_amount": certified,
                                    "retention_rate": ret,
                                    "cumulative_previous_certified": previous,
                                    "contract_value": contract})
    assert pay["retention_held"] == pytest.approx(retention, rel=_REL)
    assert pay["net_payment_due"] == pytest.approx(net, rel=_REL)
    assert pay["cumulative_certified"] == pytest.approx(cumulative, rel=_REL)
    assert pay["percent_complete"] == pytest.approx(pct, rel=_REL)
    assert pay["disputed_amount"] == pytest.approx(disputed, rel=_REL)
    assert pay["net_payment_due"] - ld["total_amount"] == pytest.approx(payable, rel=_REL)


@pytest.mark.parametrize("days, total", [
    (None, None),     # invented: no period given -> a daily figure only, no total invented
    (4, 60_000),      # invented: 15,000 x 4 = 60,000
])
def test_m21_damages_total_only_when_the_days_are_given(days, total):
    params = {"amount_per_day": 15_000}
    if days is not None:
        params["days"] = days
    ld = _ok("delay_damages_daily", params)
    assert ld["daily_amount"] == pytest.approx(15_000, rel=_REL)
    if total is None:
        assert "total_amount" not in ld, ld
    else:
        assert ld["total_amount"] == pytest.approx(total, rel=_REL)
