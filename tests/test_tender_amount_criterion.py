"""evaluate_tender: a weighted tender sum is scored, not added as a score.

A criterion whose own name says it is a tender sum (price, cost, sum,
amount) and whose tenderers carry the amount itself is scored
lowest / amount x 100. Before, the raw amount was added as if it were a
0-100 score, so the dearest tender won. Weights that sum to 100 are read
as percentages. Every input below is invented.
"""
from __future__ import annotations

import pytest

from app.agents.hats.procurement.formulas.construction_knowledge import evaluate_tender
from app.lib.construction_formulas import run_calculation


def _names(result):
    return [row["name"] for row in result["ranked_tenderers"]]


def _totals(result):
    return {row["name"]: row["weighted_total"] for row in result["ranked_tenderers"]}


@pytest.mark.parametrize(
    "criterion, tenderers, weights, order, totals, scores", [
        # K 3.0/3.0 = 100, L 3.0/6.0 = 50
        # K 0.8*100 + 0.2*50 = 90.0; L 0.8*50 + 0.2*90 = 58.0 -> K, L
        ("price",
         [{"name": "K", "price": 3_000_000, "quality": 50},
          {"name": "L", "price": 6_000_000, "quality": 90}],
         {"price": 0.8, "quality": 0.2},
         ["K", "L"], {"K": 90.0, "L": 58.0}, {"K": 100.0, "L": 50.0}),
        # R 1.2/1.6 = 75, S 100, T 1.2/2.4 = 50
        # R 0.5*75 + 0.5*88 = 81.5; S 50 + 30 = 80.0; T 25 + 49 = 74.0 -> R, S, T
        ("tender_cost",
         [{"name": "R", "tender_cost": 1.6, "technical": 88},
          {"name": "S", "tender_cost": 1.2, "technical": 60},
          {"name": "T", "tender_cost": 2.4, "technical": 98}],
         {"tender_cost": 0.5, "technical": 0.5},
         ["R", "S", "T"], {"R": 81.5, "S": 80.0, "T": 74.0},
         {"R": 75.0, "S": 100.0, "T": 50.0}),
    ])
def test_amount_criterion_scored_lowest_over_amount(criterion, tenderers, weights,
                                                    order, totals, scores):
    result = evaluate_tender(tenderers, weights)
    assert _names(result) == order
    assert _totals(result) == pytest.approx(totals, abs=0.01)
    derived = result["derived_scores"][criterion]
    assert "lowest" in derived["rule"]
    assert derived["scores"] == pytest.approx(scores, abs=0.01)


@pytest.mark.parametrize(
    "tenderers, weights, amount_key", [
        ([{"name": "K", "price": 3_000_000, "quality": 50},
          {"name": "L", "price": 6_000_000, "quality": 90}],
         {"price": 0.8, "quality": 0.2}, "price"),
        ([{"name": "U", "bid_amount": 900, "hse": 40},
          {"name": "V", "bid_amount": 450, "hse": 70}],
         {"bid_amount": 0.6, "hse": 0.4}, "bid_amount"),
    ])
def test_dearer_tender_does_not_gain_from_its_amount(tenderers, weights, amount_key):
    """Raising the dearest tender's amount, all else equal, lowers its total."""
    before = _totals(evaluate_tender(tenderers, weights))
    dearest = max(tenderers, key=lambda t: t[amount_key])["name"]
    raised = [dict(t) for t in tenderers]
    for row in raised:
        if row["name"] == dearest:
            row[amount_key] = row[amount_key] * 1.5
    after = _totals(evaluate_tender(raised, weights))
    assert after[dearest] < before[dearest]


@pytest.mark.parametrize(
    "percent, fraction", [
        ({"price": 70, "quality": 30}, {"price": 0.7, "quality": 0.3}),
        ({"technical": 40, "commercial": 45, "hse": 15},
         {"technical": 0.40, "commercial": 0.45, "hse": 0.15}),
    ])
def test_weights_summing_to_100_read_as_percent(percent, fraction):
    tenderers = [
        {"name": "A1", "price": 5.0, "quality": 70, "technical": 60,
         "commercial": 80, "hse": 90},
        {"name": "B1", "price": 4.0, "quality": 55, "technical": 85,
         "commercial": 65, "hse": 75},
    ]
    as_percent = evaluate_tender(tenderers, percent)
    as_fraction = evaluate_tender(tenderers, fraction)
    assert _totals(as_percent) == pytest.approx(_totals(as_fraction), abs=0.01)
    assert as_percent["weights_read_as_percent"] is True
    assert as_percent["weights_applied"] == pytest.approx(fraction)
    assert "weights_read_as_percent" not in as_fraction


@pytest.mark.parametrize(
    "weights", [
        {"technical": 30, "commercial": 30},
        {"technical": 0.6, "commercial": 0.3},
    ])
def test_weights_not_summing_to_100_are_left_as_given(weights):
    tenderers = [{"name": "W", "technical": 50, "commercial": 70}]
    result = evaluate_tender(tenderers, weights)
    assert result["weights_applied"] == weights
    assert "weights_read_as_percent" not in result
    expected = 50 * weights["technical"] + 70 * weights["commercial"]
    assert result["ranked_tenderers"][0]["weighted_total"] == pytest.approx(expected, abs=0.01)


@pytest.mark.parametrize(
    "tenderers, weights, totals", [
        # given price scores win over the bare amounts: 0.6*100 + 0.4*40 = 76; 0.6*70 + 0.4*90 = 78
        ([{"name": "M", "price": 2.0, "price_score": 100, "quality": 40},
          {"name": "N", "price": 2.9, "price_score": 70, "quality": 90}],
         {"price": 0.6, "quality": 0.4}, {"M": 76.0, "N": 78.0}),
        ([{"name": "P", "cost": 10, "cost_score": 55, "technical": 80},
          {"name": "Q", "cost": 12, "cost_score": 95, "technical": 20}],
         {"cost": 0.5, "technical": 0.5}, {"P": 67.5, "Q": 57.5}),
    ])
def test_stated_scores_are_not_replaced(tenderers, weights, totals):
    result = evaluate_tender(tenderers, weights)
    assert _totals(result) == pytest.approx(totals, abs=0.01)
    assert "derived_scores" not in result


@pytest.mark.parametrize(
    "tenderers, weights, totals", [
        ([{"name": "G", "commercial": 80, "technical": 60},
          {"name": "H", "commercial": 60, "technical": 90}],
         {"commercial": 0.5, "technical": 0.5}, {"G": 70.0, "H": 75.0}),
        ([{"name": "E", "delivery": 30, "quality": 90},
          {"name": "F", "delivery": 95, "quality": 35}],
         {"delivery": 0.25, "quality": 0.75}, {"E": 75.0, "F": 50.0}),
    ])
def test_criteria_not_named_as_amounts_stay_scores(tenderers, weights, totals):
    result = evaluate_tender(tenderers, weights)
    assert _totals(result) == pytest.approx(totals, abs=0.01)
    assert "derived_scores" not in result


@pytest.mark.parametrize(
    "amounts", [
        (0, 3.0),
        (-1.0, 2.0),
    ])
def test_non_positive_amount_is_not_scored_as_an_amount(amounts):
    tenderers = [
        {"name": "Z1", "price": amounts[0], "quality": 50},
        {"name": "Z2", "price": amounts[1], "quality": 50},
    ]
    result = evaluate_tender(tenderers, {"price": 0.5, "quality": 0.5})
    assert "derived_scores" not in result


@pytest.mark.parametrize(
    "params, winner", [
        ({"tenderers": [{"name": "K", "price": 3_000_000, "quality": 50},
                        {"name": "L", "price": 6_000_000, "quality": 90}],
          "weights": {"price": 80, "quality": 20}}, "K"),
        ({"tenderers": [{"name": "R", "tender_cost": 1.6, "technical": 88},
                        {"name": "S", "tender_cost": 1.2, "technical": 60}],
          "weights": {"tender_cost": 50, "technical": 50}}, "R"),
    ])
def test_calculator_entry_ranks_on_derived_scores(params, winner):
    envelope = run_calculation("evaluate_tender", params)
    result = envelope.get("result", envelope)
    assert result["recommended"]["name"] == winner
    assert result["weights_read_as_percent"] is True


@pytest.mark.parametrize(
    "tenderers, weights", [
        ([], {"price": 70, "quality": 30}),
        (None, None),
    ])
def test_no_tenderers_is_named_as_such(tenderers, weights):
    """No tenderers is its own error, not a missing-score question."""
    result = evaluate_tender(tenderers, weights)
    assert result == {"error": "evaluate_tender requires at least one tenderer."}
