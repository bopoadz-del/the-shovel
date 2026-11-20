"""S-K2 slice 3: IPC retention is a required input in both calculators.

Neither ``calculate_interim_payment`` (percent) nor ``calculate_payment``
(fraction) keeps a retention default. A missing rate is an error that
names the input (through the registry, a ``needs_input`` envelope listing
it; called directly, refused by name), and a given rate is the rate used.
Each rule is pinned with two inputs, built here.
"""
from __future__ import annotations

import pytest

from app.agents.hats.commercial.formulas.construction_formulas_additions import (
    calculate_interim_payment,
)
from app.agents.hats.commercial.formulas.construction_knowledge import (
    calculate_payment,
)
from tests.test_calc_param_binding import _err, _ok


# -- missing rate -> error naming the input -----------------------------------

@pytest.mark.parametrize("gross", [900_000, 37_500.5])
def test_interim_payment_without_retention_asks_for_it_by_name(gross):
    env = _err("calculate_interim_payment", {"gross_valuation": gross})
    assert env["missing"] == ["retention_percent"]
    assert "retention_percent" in env["error"]
    assert env.get("needs_input") is True


@pytest.mark.parametrize("claimed, certified", [(100_000, 90_000), (4_200, 4_200)])
def test_payment_without_retention_asks_for_it_by_name(claimed, certified):
    env = _err("calculate_payment",
               {"claimed_amount": claimed, "certified_amount": certified})
    assert env["missing"] == ["retention_rate"]
    assert "retention_rate" in env["error"]
    assert env.get("needs_input") is True


@pytest.mark.parametrize("gross", [900_000, 12_000])
def test_interim_payment_none_retention_is_an_error_naming_it(gross):
    out = calculate_interim_payment(gross, None)
    assert "retention_percent" in out["error"]
    assert out["missing"] == ["retention_percent"]
    assert "net_payment" not in out


@pytest.mark.parametrize("certified", [90_000, 1_000])
def test_payment_none_retention_is_an_error_naming_it(certified):
    out = calculate_payment(certified, certified, None)
    assert "retention_rate" in out["error"]
    assert out["missing"] == ["retention_rate"]
    assert "net_payment_due" not in out


def test_direct_calls_without_retention_are_refused_by_name():
    with pytest.raises(TypeError, match="retention_percent"):
        calculate_interim_payment(900_000)
    with pytest.raises(TypeError, match="retention_rate"):
        calculate_payment(100_000, 90_000)


# -- given rate -> used --------------------------------------------------------

@pytest.mark.parametrize("gross, pct, held, net", [
    (900_000, 10, 90_000.0, 810_000.0),
    (400_000, 7.5, 30_000.0, 370_000.0),
])
def test_interim_payment_uses_the_given_rate(gross, pct, held, net):
    r = _ok("calculate_interim_payment",
            {"gross_valuation": gross, "retention_percent": pct})
    assert r["retention_percent"] == pytest.approx(pct)
    assert r["retention_amount"] == pytest.approx(held)
    assert r["net_payment"] == pytest.approx(net)
    assert f"{pct}%" in r["calculation"]


@pytest.mark.parametrize("certified, rate, held, net", [
    (90_000, 0.05, 4_500.0, 85_500.0),
    (200_000, 0.10, 20_000.0, 180_000.0),
])
def test_payment_uses_the_given_rate(certified, rate, held, net):
    r = _ok("calculate_payment", {"claimed_amount": certified,
                                  "certified_amount": certified,
                                  "retention_rate": rate})
    assert r["retention_held"] == pytest.approx(held)
    assert r["net_payment_due"] == pytest.approx(net)
    assert r["retention_rate_pct"] == pytest.approx(rate * 100)


@pytest.mark.parametrize("gross, pct", [(-1, 10), (-250_000, 5)])
def test_interim_payment_negative_gross_is_refused_whatever_the_rate(gross, pct):
    out = calculate_interim_payment(gross, pct)
    assert "gross_valuation" in out["error"]
    assert "net_payment" not in out


@pytest.mark.parametrize("claimed, certified", [(-1, 900), (1_000, -5)])
def test_payment_negative_amount_is_refused_whatever_the_rate(claimed, certified):
    out = calculate_payment(claimed, certified, 0.05)
    assert "must be >= 0" in out["error"]
    assert "net_payment_due" not in out


@pytest.mark.parametrize("fn, args, key", [
    (calculate_interim_payment, (100_000, 150), "retention_percent"),
    (calculate_payment, (1_000, 900, 1.5), "retention_rate"),
])
def test_out_of_range_rate_is_still_refused(fn, args, key):
    out = fn(*args)
    assert key in out["error"]
    assert "missing" not in out
