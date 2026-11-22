"""A rate the payment certificate assumed is named, never silent.

``ConstructionContainer.payment_certificate`` falls back to its own default
when a rate is neither in the ask nor in the args (retention, and the advance
recovery rate when an advance is being recovered). The figure it then prints
rests on a number nobody stated. The certificate names every such rate in
``defaults_used`` (the shape the formula registry uses) and in its summary
line; a stated rate is never listed. Every ask here is invented, and each
rule is pinned with two inputs.
"""
from __future__ import annotations

import pytest

from app.containers.construction import ConstructionContainer
from app.containers.construction.boq import _payment_figures_from_message


async def _cert(ask: str, params: dict | None = None) -> dict:
    out = await ConstructionContainer().payment_certificate({"message": ask}, params or {})
    assert out["status"] == "success", out
    return out


def _named(out: dict) -> dict:
    return {row["parameter"]: row["value"] for row in out["defaults_used"]}


# -- retention ----------------------------------------------------------------

@pytest.mark.parametrize("gross, held", [(640_000, 64_000.0), (37_500, 3_750.0)])
async def test_retention_not_stated_is_named_as_assumed(gross, held):
    out = await _cert(f"Certify gross valuation {gross:,} for this month.")
    assert out["deductions"]["retention_held"] == pytest.approx(held)
    assert _named(out) == {"retention_percent": 10}
    assert "Assumed, not given: retention 10%." in out["certificate_summary"]


@pytest.mark.parametrize("ask", [
    "Certify gross valuation 640,000 with retention 7.5%.",
    "Certify gross valuation 37,500 at 3% retention.",
])
async def test_retention_stated_in_the_ask_is_not_named(ask):
    out = await _cert(ask)
    assert "retention_percent" not in _named(out)
    assert "Assumed" not in out["certificate_summary"]


@pytest.mark.parametrize("rate", [0, 6])
async def test_retention_given_as_an_arg_is_not_named(rate):
    out = await _cert("Certify gross valuation 210,000.", {"retention_percent": rate})
    assert out["deductions"]["retention_percent"] == pytest.approx(rate)
    assert out["defaults_used"] == []


# -- advance recovery ---------------------------------------------------------

@pytest.mark.parametrize("gross, advance, recovered", [
    (500_000, 300_000, 100_000.0),
    (80_000, 9_000, 9_000.0),
])
async def test_recovery_rate_not_stated_is_named_when_an_advance_is_recovered(
        gross, advance, recovered):
    out = await _cert(
        f"Certify gross valuation {gross:,}, retention 5%, advance payment {advance:,}."
    )
    assert out["deductions"]["advance_recovery"] == pytest.approx(recovered)
    assert _named(out) == {"advance_recovery_percent": 20}
    assert "Assumed, not given: advance recovery 20%." in out["certificate_summary"]


@pytest.mark.parametrize("gross", [500_000, 80_000])
async def test_recovery_rate_is_not_named_when_there_is_no_advance(gross):
    out = await _cert(f"Certify gross valuation {gross:,}, retention 5%.")
    assert out["deductions"]["advance_recovery"] == 0
    assert out["defaults_used"] == []


@pytest.mark.parametrize("ask, recovered", [
    ("Certify gross valuation 500,000, retention 5%, advance payment 300,000, "
     "advance recovery 15%.", 75_000.0),
    ("Certify gross valuation 80,000, retention 5%, advance payment 9,000, "
     "advance recovery 5%.", 4_000.0),
])
async def test_recovery_rate_stated_in_the_ask_is_not_named(ask, recovered):
    out = await _cert(ask)
    assert out["deductions"]["advance_recovery"] == pytest.approx(recovered)
    assert out["defaults_used"] == []


@pytest.mark.parametrize("rate, recovered", [(12, 60_000.0), (0, 0.0)])
async def test_recovery_rate_given_as_an_arg_is_not_named(rate, recovered):
    out = await _cert(
        "Certify gross valuation 500,000, retention 5%, advance payment 300,000.",
        {"advance_recovery_percent": rate},
    )
    assert out["deductions"]["advance_recovery"] == pytest.approx(recovered)
    assert out["defaults_used"] == []


# -- an ask that states the advance AND its recovery rate keeps both ------------

@pytest.mark.parametrize("ask, advance, rate", [
    ("advance payment 300,000, advance recovery 15%", 300_000.0, 15.0),
    ("advance recovery 5% of an advance payment of 9,000", 9_000.0, 5.0),
])
def test_reader_keeps_the_advance_amount_beside_a_recovery_rate(ask, advance, rate):
    fig = _payment_figures_from_message(ask)
    assert fig["advance_payment"] == pytest.approx(advance)
    assert fig["advance_recovery_percent"] == pytest.approx(rate)


@pytest.mark.parametrize("ask", ["advance recovery 15%", "advance payment recovery 7.5 percent"])
def test_reader_never_reads_a_recovery_rate_as_the_advance_amount(ask):
    fig = _payment_figures_from_message(ask)
    assert "advance_payment" not in fig
    assert "advance_recovery_percent" in fig


@pytest.mark.parametrize("ask, gross", [
    ("gross 4% above last month; gross 760,000", 760_000.0),
    ("gross 12 percent up, then gross 51,250", 51_250.0),
])
def test_reader_skips_a_rate_to_the_labels_amount(ask, gross):
    assert _payment_figures_from_message(ask)["gross_valuation"] == pytest.approx(gross)


# -- both, and the figures are unchanged ---------------------------------------

@pytest.mark.parametrize("gross, advance, net", [
    (500_000, 300_000, 350_000.0),
    (80_000, 9_000, 63_000.0),
])
async def test_both_rates_assumed_are_both_named_and_the_net_is_unchanged(gross, advance, net):
    out = await _cert(f"Certify gross valuation {gross:,}, advance payment {advance:,}.")
    assert out["payment"]["net_due_this_period"] == pytest.approx(net)
    assert _named(out) == {"retention_percent": 10, "advance_recovery_percent": 20}
    assert out["certificate_summary"].endswith(
        "Assumed, not given: retention 10%; advance recovery 20%."
    )
