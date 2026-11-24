"""S-K2 item 8: Monte Carlo P50 / P80 / P90 with a seed and user-supplied ranges.

Every expected figure is worked here from the triangular distribution's own
inverse CDF, independently of the sampler: for low a, most likely c, high b
and a probability p above F(c) = (c - a) / (b - a), the quantile is
x = b - sqrt((1 - p) (b - a) (b - c)); at or below F(c) it is
x = a + sqrt(p (b - a) (c - a)). Each decision has two inputs.
"""
from __future__ import annotations

import math

import pytest

from app.lib.construction_formulas import run_calculation
from app.agents.hats.planning.formulas.construction_formulas_planning import (
    _nearest_rank,
    monte_carlo_percentiles,
)


def _run(**params):
    return run_calculation("monte_carlo_percentiles", params)


def _ok(**params):
    env = _run(**params)
    assert env["status"] == "success", env
    return env["result"]


def _triangular_quantile(a: float, c: float, b: float, p: float) -> float:
    if p <= (c - a) / (b - a):
        return a + math.sqrt(p * (b - a) * (c - a))
    return b - math.sqrt((1 - p) * (b - a) * (b - c))


# ── the percentiles match the distribution the user's range defines ────────

@pytest.mark.parametrize("a,c,b,seed", [(0, 5, 10, 3), (2, 4, 12, 7)])
def test_one_item_percentiles_match_the_triangular_quantiles(a, c, b, seed):
    # (0, 5, 10): 5, 6.8377, 7.7639. (2, 4, 12): F(4) = 0.2, so all three
    # quantiles are above the mode: 5.6754, 8.0, 9.1716.
    r = _ok(low=[a], most_likely=[c], high=[b], iterations=200000, seed=seed)
    for p in (50, 80, 90):
        assert r[f"p{p}"] == pytest.approx(_triangular_quantile(a, c, b, p / 100),
                                           abs=0.03), p


@pytest.mark.parametrize("a,c,b", [(0, 5, 10), (2, 4, 12)])
def test_chance_within_the_most_likely_is_the_cdf_at_the_mode(a, c, b):
    # F(c) = (c - a) / (b - a): 50% for (0, 5, 10), 20% for (2, 4, 12).
    r = _ok(low=[a], most_likely=[c], high=[b], iterations=200000, seed=11)
    assert r["probability_within_most_likely_percent"] == pytest.approx(
        100 * (c - a) / (b - a), abs=0.5)
    assert r["mean"] == pytest.approx((a + b + c) / 3, abs=0.03)


@pytest.mark.parametrize("lows,modes,highs", [
    ([3, 4.5], [3, 4.5], [3, 4.5]),
    ([10, 0, 2], [10, 0, 2], [10, 0, 2]),
])
def test_items_with_no_spread_give_their_sum_on_every_trial(lows, modes, highs):
    # Nothing to sample: every trial total is the sum, and every trial is
    # at (so within) the most likely total.
    total = sum(modes)
    r = _ok(low=lows, most_likely=modes, high=highs, iterations=300, seed=4)
    assert (r["p50"], r["p80"], r["p90"]) == (total, total, total)
    assert r["min_trial"] == r["max_trial"] == r["mean"] == total
    assert r["probability_within_most_likely_percent"] == 100.0


@pytest.mark.parametrize("lows,modes,highs", [
    ([10, 20, 5], [12, 25, 6], [20, 40, 9]),
    ([1000, 2000], [1500, 2000], [3000, 2500]),
])
def test_a_total_of_items_stays_in_its_range_and_its_percentiles_ascend(
        lows, modes, highs):
    r = _ok(low=lows, most_likely=modes, high=highs, iterations=5000, seed=2)
    assert sum(lows) <= r["min_trial"] <= r["p50"] <= r["p80"] <= r["p90"] \
        <= r["max_trial"] <= sum(highs)
    assert r["most_likely_total"] == sum(modes)


# ── the seed makes it reproducible, and the answer says which seed ──────────

@pytest.mark.parametrize("seed", [1, 424242])
def test_the_same_seed_gives_the_same_answer(seed):
    params = dict(low=[10, 20], most_likely=[12, 25], high=[20, 40],
                  iterations=2000, seed=seed)
    first, again = _ok(**params), _ok(**params)
    assert first == again
    assert first["seed"] == seed
    assert f"seed {seed}" in first["working"][0]


@pytest.mark.parametrize("seed_a,seed_b", [(1, 2), (7, 70000)])
def test_a_different_seed_draws_different_trials(seed_a, seed_b):
    base = dict(low=[10, 20], most_likely=[12, 25], high=[20, 40], iterations=500)
    a, b = _ok(seed=seed_a, **base), _ok(seed=seed_b, **base)
    assert (a["p50"], a["mean"]) != (b["p50"], b["mean"])


@pytest.mark.parametrize("iterations", [100, 3000])
def test_the_trial_count_is_the_one_run_and_stated(iterations):
    r = _ok(low=[1], most_likely=[2], high=[4], iterations=iterations, seed=3)
    assert r["iterations"] == iterations
    assert f"{iterations:,} trials" in r["working"][0]


# ── nearest rank ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("ordered,percent,expected", [
    # 10 values: rank ceil(0.8 x 10) = 8 -> the 8th; ceil(0.5 x 10) = 5.
    (list(range(1, 11)), 80, 8),
    (list(range(1, 11)), 50, 5),
    # 7 values: ceil(0.9 x 7) = ceil(6.3) = 7 -> the last; ceil(3.5) = 4.
    ([2, 4, 6, 8, 10, 12, 14], 90, 14),
    ([2, 4, 6, 8, 10, 12, 14], 50, 8),
])
def test_nearest_rank_is_the_smallest_value_with_that_share_at_or_below(
        ordered, percent, expected):
    assert _nearest_rank(ordered, percent) == expected


# ── the working prints figures in full, never as exponents ──────────────────

@pytest.mark.parametrize("scale", [1e6, 2.5e7])
def test_working_writes_large_totals_in_full(scale):
    r = _ok(low=[1 * scale, 2 * scale], most_likely=[1.5 * scale, 2 * scale],
            high=[3 * scale, 2.5 * scale], iterations=500, seed=1)
    text = " ".join(r["working"])
    assert "e+" not in text
    assert f"{3 * scale:,.4f}" in text and f"{5.5 * scale:,.4f}" in text
    assert f"{r['p90']:,.4f}" in text


# ── the user's ranges are required and must be in order ─────────────────────

@pytest.mark.parametrize("lows,modes,highs", [
    ([1, 2], [2], [3]),
    ([1], [2, 3], [4, 5]),
])
def test_ranges_that_do_not_list_the_same_items_are_refused(lows, modes, highs):
    env = _run(low=lows, most_likely=modes, high=highs)
    assert env["status"] == "error"
    assert "same items" in env["error"]


@pytest.mark.parametrize("lows,modes,highs,bad", [
    ([6], [5], [9], "1"),                    # low above most likely
    ([1, 2, 3], [2, 9, 4], [3, 8, 5], "2"),  # most likely above high
])
def test_an_item_out_of_order_is_named_and_refused(lows, modes, highs, bad):
    env = _run(low=lows, most_likely=modes, high=highs)
    assert env["status"] == "error"
    assert env["error"].endswith(f"item(s) {bad}.")


@pytest.mark.parametrize("missing", ["low", "high"])
def test_a_missing_range_is_asked_for_not_assumed(missing):
    params = dict(low=[1], most_likely=[2], high=[3])
    params.pop(missing)
    env = _run(**params)
    assert env["status"] != "success"


@pytest.mark.parametrize("single", [(4, 4, 4), (2, 5, 9)])
def test_a_single_figure_per_range_is_one_item(single):
    a, c, b = single
    direct = dict(iterations=400, seed=6)
    assert monte_carlo_percentiles(a, c, b, **direct) == \
        monte_carlo_percentiles([a], [c], [b], **direct)


def test_no_items_is_refused():
    assert "error" in monte_carlo_percentiles([], [], [])
    assert "error" in monte_carlo_percentiles((), (), ())
