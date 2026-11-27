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
