"""S-K2 item 3: interim payment worked from the totals to date.

Every expected value is worked out here from the method, on inputs this
file builds itself -- no import of the engine's arithmetic:

    gross     = work done to date + materials on site
    retention = min(rate x work done, limit)      (no limit given: rate x work)
    net       = gross - retention
    due       = net - total of earlier certificates

The rate and the earlier certificates are the contract's figures and have no
default; a negative due is stated, never clipped to nil.
"""
from __future__ import annotations

import random

import pytest

from app.agents.hats.commercial.formulas.construction_formulas_additions import (
    interim_payment_to_date,
)
from app.lib import formula_registry as fr

_SEEDS = (5, 29)


def _cases(seed: int):
    rng = random.Random(seed)
    out = []
    for _ in range(20):
        work = float(rng.randint(50_000, 9_000_000))
        mos = float(rng.choice([0, rng.randint(1_000, 400_000)]))
        rate = rng.choice([2.5, 3, 5, 7.5, 10])
        prev = round(work * rng.uniform(0.0, 0.8), 2)
        limit = rng.choice([None, round(work * rng.uniform(0.01, 0.12), 2)])
        out.append((work, mos, rate, prev, limit))
    return out


@pytest.mark.parametrize("seed", _SEEDS)
def test_amount_due_follows_the_method(seed):
    for work, mos, rate, prev, limit in _cases(seed):
        r = interim_payment_to_date(work, rate, prev, materials_on_site=mos,
                                    retention_limit=limit)
        at_rate = work * rate / 100
        ret = at_rate if limit is None else min(at_rate, limit)
        assert r["gross_to_date"] == pytest.approx(work + mos, abs=0.01)
        assert r["retention_to_date"] == pytest.approx(ret, abs=0.01)
        assert r["retention_limit_reached"] is (limit is not None and at_rate > limit)
        assert r["net_to_date"] == pytest.approx(work + mos - ret, abs=0.01)
        assert r["amount_due"] == pytest.approx(work + mos - ret - prev, abs=0.01)


@pytest.mark.parametrize("seed", _SEEDS)
def test_working_prints_every_figure_it_reports(seed):
    for work, mos, rate, prev, limit in _cases(seed):
        r = interim_payment_to_date(work, rate, prev, materials_on_site=mos,
                                    retention_limit=limit)
        text = "\n".join(r["working"])
        for key in ("gross_to_date", "retention_to_date", "net_to_date", "amount_due"):
            assert f"{r[key]:,.2f}" in text, (key, text)
        assert f"{rate:g}%" in text
        assert f"{prev:,.2f}" in text


@pytest.mark.parametrize("work,rate,limit", [(1_000_000, 10, 50_000), (240_000, 5, 6_000)])
def test_retention_is_held_at_its_limit(work, rate, limit):
    r = interim_payment_to_date(work, rate, 0, retention_limit=limit)
    assert r["retention_to_date"] == limit
    assert r["retention_limit_reached"] is True
    assert "held at the retention limit" in r["working"][1]


@pytest.mark.parametrize("work,rate,limit", [(1_000_000, 10, 500_000), (240_000, 5, 12_000)])
def test_a_limit_not_reached_leaves_retention_at_the_rate(work, rate, limit):
    r = interim_payment_to_date(work, rate, 0, retention_limit=limit)
    assert r["retention_to_date"] == pytest.approx(work * rate / 100)
    assert r["retention_limit_reached"] is False
    assert "not reached" in r["working"][1]


@pytest.mark.parametrize("work,rate", [(80_000, 10), (3_300_000, 2.5)])
def test_no_limit_given_is_said_not_assumed(work, rate):
    r = interim_payment_to_date(work, rate, 0)
    assert r["retention_to_date"] == pytest.approx(work * rate / 100)
    assert "no retention limit given" in r["working"][1]


@pytest.mark.parametrize("work,mos,rate", [(500_000, 40_000, 10), (90_000, 12_500, 5)])
def test_materials_on_site_carry_no_retention(work, mos, rate):
    with_mos = interim_payment_to_date(work, rate, 0, materials_on_site=mos)
    without = interim_payment_to_date(work, rate, 0)
    assert with_mos["retention_to_date"] == without["retention_to_date"]
    assert with_mos["amount_due"] == pytest.approx(without["amount_due"] + mos)


@pytest.mark.parametrize("work,prev", [(100_000, 150_000), (2_000_000, 1_950_000)])
def test_earlier_certificates_above_net_give_a_stated_deduction(work, prev):
    r = interim_payment_to_date(work, 10, prev)
    assert r["amount_due"] == pytest.approx(work * 0.9 - prev)
    assert r["amount_due"] < 0
    assert "deduction" in r["note"]


@pytest.mark.parametrize("work,prev", [(100_000, 10_000), (2_000_000, 0)])
def test_a_positive_due_carries_no_deduction_note(work, prev):
    r = interim_payment_to_date(work, 10, prev)
    assert r["amount_due"] > 0
    assert "note" not in r


@pytest.mark.parametrize("work,rate", [(200_000, 10), (3_000_000, 5)])
def test_earlier_certificates_equal_to_net_give_nil_due_and_no_note(work, rate):
    # boundary: earlier certificates exactly equal to the net to date
    prev = work - work * rate / 100.0
    r = interim_payment_to_date(work, rate, prev)
    assert r["amount_due"] == 0
    assert "note" not in r


@pytest.mark.parametrize("prev", [0, 25_000])
def test_the_retention_rate_has_no_default(prev):
    r = interim_payment_to_date(100_000, None, prev)
    assert r.get("missing") == ["retention_percent"]
    assert "amount_due" not in r


@pytest.mark.parametrize("rate", [5, 10])
def test_earlier_certificates_have_no_default(rate):
    r = interim_payment_to_date(100_000, rate, None)
    assert r.get("missing") == ["previously_certified"]
    assert "amount_due" not in r


@pytest.mark.parametrize("args,kw", [
    ((-1, 5, 0), {}), ((100, 5, -3), {}), ((100, 5, 0), {"materials_on_site": -2}),
])
def test_a_negative_amount_is_refused(args, kw):
    r = interim_payment_to_date(*args, **kw)
    assert ">= 0" in r["error"]
    assert "amount_due" not in r


@pytest.mark.parametrize("rate", [-0.5, 100.5])
def test_a_rate_outside_0_to_100_is_refused(rate):
    r = interim_payment_to_date(100_000, rate, 0)
    assert "between 0 and 100" in r["error"]


@pytest.mark.parametrize("limit", [-1, -2500])
def test_a_negative_limit_is_refused(limit):
    r = interim_payment_to_date(100_000, 5, 0, retention_limit=limit)
    assert "retention_limit" in r["error"]


@pytest.mark.parametrize(
    "name", ["work_value_to_date", "previously_certified", "materials_on_site"]
)
def test_amounts_declare_they_are_totals_to_date(name):
    params = fr.parameters(fr.get("interim_payment_to_date"))
    assert params[name].basis == "cumulative", name


@pytest.mark.parametrize("name", ["retention_percent", "previously_certified"])
def test_rate_and_earlier_certificates_have_no_default(name):
    params = fr.parameters(fr.get("interim_payment_to_date"))
    assert params[name].required, name


def _nil_cases(seed: int):
    # earlier certificates equal to the net to date, on 2-dp figures with
    # cents, so a float difference that is not exactly zero would show
    rng = random.Random(seed)
    out = []
    for _ in range(400):
        work = rng.randint(5_000_000, 900_000_000) / 100.0
        mos = rng.choice([0.0, rng.randint(100_000, 40_000_000) / 100.0])
        rate = rng.choice([2.5, 5, 7.5, 10])
        ret = round(work * rate / 100.0, 2)
        prev = round(work + mos - ret, 2)
        out.append((work, mos, rate, prev))
    return out


@pytest.mark.parametrize("seed", _SEEDS)
def test_a_nil_due_on_figures_with_cents_is_nil_not_a_deduction(seed):
    for work, mos, rate, prev in _nil_cases(seed):
        r = interim_payment_to_date(work, rate, prev, materials_on_site=mos)
        assert r["amount_due"] == 0, (work, mos, rate, prev)
        assert str(r["amount_due"]) == "0.0"
        assert "note" not in r
        assert "-0.00" not in r["working"][3]


@pytest.mark.parametrize("work,mos,rate", [
    (590_475.60, 152_192.49, 7.5), (3_687_518.80, 0.0, 2.5),
])
def test_a_nil_due_through_the_public_path(work, mos, rate):
    from app.lib import construction_formulas as cf
    prev = round(work + mos - round(work * rate / 100.0, 2), 2)
    env = cf.run_calculation("interim_payment_to_date", {
        "work_value_to_date": work, "materials_on_site": mos,
        "retention_percent": rate, "previously_certified": prev})
    res = env["result"]
    assert res["amount_due"] == 0
    assert "note" not in res
    assert res["working"][3].endswith("= 0.00")


@pytest.mark.parametrize("work,rate,limit,at_rate", [
    (1_000_000, 5, 40_000, "50,000.00"), (240_000, 5, 6_000, "12,000.00"),
])
def test_a_capped_retention_shows_the_at_rate_figure_before_the_limit(
        work, rate, limit, at_rate):
    line = interim_payment_to_date(work, rate, 0, retention_limit=limit)["working"][1]
    assert f"= {at_rate}, held at the retention limit {limit:,.2f}" in line


@pytest.mark.parametrize("prev", [0, 25_000])
def test_the_earlier_certificates_are_reported_as_given(prev):
    r = interim_payment_to_date(400_000, 5, prev)
    assert r["previously_certified"] == prev
    assert r["amount_due"] == pytest.approx(380_000 - prev)


@pytest.mark.parametrize("work", [100_000, 7_250_000])
def test_a_rate_of_exactly_100_is_accepted_and_nets_nil(work):
    r = interim_payment_to_date(work, 100, 0)
    assert "error" not in r
    assert r["retention_to_date"] == work
    assert r["net_to_date"] == 0


@pytest.mark.parametrize("work", [100_000, 7_250_000])
def test_a_rate_of_exactly_0_is_accepted_and_holds_nothing(work):
    r = interim_payment_to_date(work, 0, 0)
    assert "error" not in r
    assert r["retention_to_date"] == 0
    assert r["amount_due"] == work


@pytest.mark.parametrize("work,rate", [(100_000, 5), (2_400_000, 10)])
def test_a_retention_limit_of_exactly_0_holds_nothing(work, rate):
    r = interim_payment_to_date(work, rate, 0, retention_limit=0)
    assert "error" not in r
    assert r["retention_to_date"] == 0
    assert r["retention_limit_reached"] is True
    assert r["amount_due"] == work


@pytest.mark.parametrize("work,prev,deduction", [
    (100_000, 150_000, "60,000.00"), (2_000_000, 1_950_000, "150,000.00"),
])
def test_the_deduction_note_states_the_positive_amount(work, prev, deduction):
    note = interim_payment_to_date(work, 10, prev)["note"]
    assert f"deduction of {deduction}." in note
    assert f"-{deduction}" not in note


@pytest.mark.parametrize("work,mos,gross", [
    (100_000.37, 0.26, 100_000.63), (52_410.18, 1_999.91, 54_410.09),
])
def test_gross_keeps_the_cents(work, mos, gross):
    r = interim_payment_to_date(work, 0, 0, materials_on_site=mos)
    assert r["gross_to_date"] == gross
    assert f"= {gross:,.2f}" in r["working"][0]


@pytest.mark.parametrize("work,rate", [(100_000, 5), (900_000, 10)])
def test_the_result_names_its_standard(work, rate):
    assert interim_payment_to_date(work, rate, 0)["standard"] == "FIDIC Clause 14.3"


def _summed_nil_cases(seed):
    """Earlier certificates whose total a caller adds up in floating point:
    equal to the net to the cent, but the sum carries a residue."""
    rng = random.Random(seed)
    out = []
    while len(out) < 200:
        work = rng.randint(5_000_000, 900_000_000) / 100.0
        rate = rng.choice([0, 2.5, 5, 10])
        net = round(work - round(work * rate / 100.0, 2), 2)
        cents = round(net * 100)
        a = rng.randint(1, cents - 2)
        b = rng.randint(1, cents - a - 1)
        prev = a / 100.0 + b / 100.0 + (cents - a - b) / 100.0
        if prev != net:
            out.append((work, rate, prev))
    return out


@pytest.mark.parametrize("seed", _SEEDS)
def test_a_nil_due_against_a_summed_total_is_nil(seed):
    for work, rate, prev in _summed_nil_cases(seed):
        r = interim_payment_to_date(work, rate, prev)
        assert str(r["amount_due"]) == "0.0", (work, rate, prev)
        assert "note" not in r
        assert r["working"][3].endswith("= 0.00")


@pytest.mark.parametrize("work,rate,parts", [
    (885_330.57, 0, (75_838.48, 425_242.53, 384_249.56)),
    (454_937.80, 5, (155_919.18, 74_828.58, 201_443.15)),
])
def test_a_summed_total_through_the_public_path(work, rate, parts):
    from app.lib import construction_formulas as cf
    prev = sum(parts)
    env = cf.run_calculation("interim_payment_to_date", {
        "work_value_to_date": work, "retention_percent": rate,
        "previously_certified": prev})
    res = env["result"]
    assert res["amount_due"] == 0 and str(res["amount_due"]) == "0.0"
    assert "note" not in res
    assert "-0.00" not in "\n".join(res["working"])


@pytest.mark.parametrize("prev,due,deduction", [
    (100.004, 0.0, False),
    (99.996, 0.0, False),
    (100.006, -0.01, True),
    (100.014, -0.01, True),
])
def test_earlier_certificates_are_taken_to_the_cent(prev, due, deduction):
    r = interim_payment_to_date(100.0, 0, prev)
    assert r["amount_due"] == due
    assert ("note" in r) is deduction
    assert "-0.00" not in r["working"][3]


@pytest.mark.parametrize("limit,reached", [
    (49_999.999, False),
    (50_000.004, False),
    (49_999.99, True),
    (49_999.994, True),
])
def test_the_limit_is_compared_to_the_cent(limit, reached):
    r = interim_payment_to_date(1_000_000.0, 5, 0, retention_limit=limit)
    assert r["retention_limit_reached"] is reached
    assert r["retention_to_date"] == (round(limit, 2) if reached else 50_000.0)
