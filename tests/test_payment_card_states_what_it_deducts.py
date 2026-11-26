"""The payment calculator's card says what the function deducts, and no more.

The model fills a calculator call from the card it is offered. A card that
promised "certified amount less retention and previous certificates" while the
function takes only retention off the certified amount (earlier certificates
add to the cumulative, never to the deduction) led the model to pass a to-date
valuation as the certified amount and expect earlier certificates to come off
it. The tool's net was then the to-date figure less retention, and the answer
headlined that.

Every figure here is invented.
"""
from __future__ import annotations

import dataclasses
import re

import pytest

from app.agents.driver.tools import calculator_catalogue
from app.lib import formula_registry
from app.lib.construction_formulas import run_calculation


def _ok(params):
    env = run_calculation("calculate_payment", params)
    assert env["status"] == "success", env
    return env["result"]


def _card_line() -> str:
    _, lines = calculator_catalogue("commercial")
    line = next((ln for ln in lines if ln.startswith("- calculate_payment:")), "")
    assert line, "calculate_payment is not on the card the model is offered"
    return line


@pytest.mark.parametrize("certified, previous", [
    (430_000.0, 0.0),
    (430_000.0, 1_275_000.0),
    (57_250.0, 912_400.0),
])
def test_earlier_certificates_never_move_the_net(certified, previous):
    alone = _ok({"claimed_amount": certified, "certified_amount": certified,
                 "retention_rate": 0.07})
    after = _ok({"claimed_amount": certified, "certified_amount": certified,
                 "retention_rate": 0.07, "cumulative_previous_certified": previous})
    assert after["net_payment_due"] == pytest.approx(alone["net_payment_due"])
    assert after["net_payment_due"] == pytest.approx(certified * 0.93)
    assert after["cumulative_certified"] == pytest.approx(previous + certified)


@pytest.mark.parametrize("certified, rate, previous", [
    (318_000.0, 0.1, 641_000.0),
    (96_500.0, 0.05, 2_140_000.0),
])
def test_card_does_not_promise_a_deduction_the_function_does_not_make(certified, rate, previous):
    # Read the behaviour first: do earlier certificates change the net?
    base = {"claimed_amount": certified, "certified_amount": certified, "retention_rate": rate}
    without = _ok(base)["net_payment_due"]
    with_prev = _ok({**base, "cumulative_previous_certified": previous})["net_payment_due"]
    deducts_previous = abs(with_prev - without) > 1e-6
    line = _card_line().lower()
    assert _promises_earlier_deduction(line) == deducts_previous, line


def test_card_names_the_certified_amount_as_this_periods():
    line = _card_line().lower()
    description = line.split(" inputs:")[0]
    assert "this period" in description
    # A to-date figure is named as the wrong thing to give, not the input.
    assert re.search(r"not a to-date", description)


def test_other_cards_are_unchanged_by_the_payment_card():
    # The interim-payment calculator keeps its own card; only one line moved.
    _, lines = calculator_catalogue("commercial")
    others = [ln for ln in lines if not ln.startswith("- calculate_payment:")]
    assert others and all("not a to-date" not in ln.lower() for ln in others)


# A subtraction word and an earlier-certificate word in one clause, with no
# negation before the subtraction, is a promise that earlier certificates
# come off. Any wording of it, not one phrase.
_SUBTRACT = r"\b(?:less|minus|net\s+of|deduct\w*|subtract\w*|after\s+deducting|takes?\s+off)\b"
_EARLIER = r"\b(?:previous(?:ly)?|prior|earlier|past|before|already)\b"
_NEGATION = r"\b(?:not|never|no)\b"


def _promises_earlier_deduction(text: str) -> bool:
    for clause in re.split(r"[.;:]\s*", text.lower()):
        sub = re.search(_SUBTRACT, clause)
        if sub is None or re.search(_EARLIER, clause) is None:
            continue
        if re.search(_NEGATION, clause[: sub.start()]):
            continue
        return True
    return False


@pytest.mark.parametrize("card, promises", [
    ("Payment due: certified amount less retention and previous certificates.", True),
    ("Payment due: certified amount minus retention and prior certificates.", True),
    ("Net of retention and earlier payments certified.", True),
    ("Deducts retention and certificates already paid.", True),
    ("Amount certified this period less retention; earlier certificates are not deducted here.", False),
    ("Amount certified less retention. Prior certificates only add to the cumulative.", False),
    ("Certificates issued before are never deducted.", False),
    ("", False),
])
def test_a_promise_of_an_earlier_deduction_is_read_in_any_wording(card, promises):
    assert _promises_earlier_deduction(card) is promises


@pytest.mark.parametrize("wording", [
    "Payment due on a claim: certified amount minus retention and prior certificates.",
    "Payment due on a claim: certified amount net of retention and earlier certificates.",
])
def test_a_card_reworded_to_promise_the_deduction_fails_the_card_check(monkeypatch, wording):
    spec = formula_registry.get("calculate_payment")
    monkeypatch.setitem(formula_registry._REGISTRY, "calculate_payment",
                        dataclasses.replace(spec, description=wording))
    line = _card_line().lower()
    assert wording.lower() in line
    # The function still does not deduct earlier certificates, so a card that
    # promises it disagrees with the behaviour, and the card check must see it.
    base = {"claimed_amount": 210_000.0, "certified_amount": 210_000.0, "retention_rate": 0.05}
    without = _ok(base)["net_payment_due"]
    with_prev = _ok({**base, "cumulative_previous_certified": 780_000.0})["net_payment_due"]
    assert with_prev == pytest.approx(without)
    assert _promises_earlier_deduction(line) is True
