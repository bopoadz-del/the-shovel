"""Formulas from app.core.construction_knowledge owned by the procurement hat.

Moved unchanged by scripts/move_formulas.py (F-DRIVER Phase A).
"""
from __future__ import annotations
import re
from app.lib.formula_registry import Param, formula
from typing import Dict, List, Optional


_NO_CRITERION_SCORES = (
    "Which scores does each tenderer have (technical, commercial, HSE out of 100)?"
)

# A plain number, after thousands separators are removed. Rejects anything
# float() would also reject, without catching the exception.
_PLAIN_NUMBER = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$"
)


def _score_number(value: object) -> Optional[float]:
    """A criterion cell as a float, or None when it is not a score."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if not _PLAIN_NUMBER.match(text):
            return None
        return float(text)
    return None


# Words that make a criterion name a tender sum (an amount, lower is better)
# rather than a 0-100 score. "commercial" is not one: in this house it names
# a score (``commercial_score``).
_AMOUNT_WORDS = frozenset({"price", "cost", "sum", "amount"})

_PRICE_SCORE_RULE = "lowest amount / tenderer's amount x 100"


def _names_an_amount(criterion: str) -> bool:
    """True when the criterion's own name says its value is a tender sum."""
    words = re.split(r"[^a-z]+", criterion.lower())
    return any(word in _AMOUNT_WORDS for word in words)


def _amount_scores(tenderers: List[Dict], criterion: str) -> Optional[Dict[int, float]]:
    """Score an amount criterion as lowest / amount x 100, per tenderer index.

    Only when no tenderer already carries ``<criterion>_score`` and the bare
    ``criterion`` value of every tenderer that has one is a positive amount.
    Otherwise None: the bare values are left to be read as scores, as before.
    """
    if any(
        isinstance(t, dict) and _score_number(t.get(f"{criterion}_score")) is not None
        for t in tenderers
    ):
        return None
    amounts: Dict[int, float] = {}
    for index, tenderer in enumerate(tenderers):
        if not isinstance(tenderer, dict) or criterion not in tenderer:
            continue
        number = _score_number(tenderer[criterion])
        if number is None or number <= 0:
            return None
        amounts[index] = number
    if not amounts:
        return None
    lowest = min(amounts.values())
    return {index: lowest / amount * 100.0 for index, amount in amounts.items()}


def _fraction_weights(weights: Dict) -> tuple:
    """Weights written as percentages (summing to 100) as fractions.

    Returns (weights, read_as_percent). Any other set is returned unchanged.
    """
    numbers = []
    for name, value in weights.items():
        if not isinstance(name, str):
            continue
        number = _score_number(value)
        if number is None:
            return weights, False
        numbers.append(number)
    if numbers and abs(sum(numbers) - 100.0) <= 0.5:
        return {
            name: float(_score_number(value)) / 100.0
            for name, value in weights.items() if isinstance(name, str)
        }, True
    return weights, False


def _criterion_score(tenderer: Dict, criterion: str) -> Optional[float]:
    """Read weighted criterion ``c`` from ``<c>_score`` or bare ``<c>``.

    The suffixed key wins when both are present. The same pair is used for
    every name in ``weights``, including names the caller supplies.
    """
    if not isinstance(tenderer, dict):
        return None
    for key in (f"{criterion}_score", criterion):
        if key not in tenderer:
            continue
        number = _score_number(tenderer[key])
        if number is not None:
            return number
    return None


@formula(
    owner='procurement',
    display_name='Tender evaluation',
    description='Scores and ranks tender submissions on weighted technical and commercial criteria.',
    inputs={
        'tenderers': Param(label='tender submissions'),
        'weights': Param(label='criterion weights'),
    },
    outputs={'ranked_tenderers': '-', 'recommended': '-', 'weights_applied': '-', 'procedure': '-'},
)
def evaluate_tender(
    tenderers: List[Dict],
    weights: Optional[Dict] = None,
) -> Dict:
    """
    Score and rank tender submissions per the tender analysis procedure.

    tenderers: list of dicts. Each weighted criterion ``c`` is read from
    ``<c>_score`` or, when that key is absent, from ``c``. Default criteria
    are technical, commercial, hse and local_content, each 0-100.

    weights: optional dict overriding defaults:
        {"technical": 0.45, "commercial": 0.45, "hse": 0.07, "local_content": 0.03}
    Weights that sum to 100 are percentages and are read as fractions.

    A weighted criterion whose name says it is a tender sum (price, cost,
    sum, amount) and whose tenderers carry the amount itself, not a
    ``<c>_score``, is scored lowest / amount x 100. The rule and the
    derived scores are returned in ``derived_scores``.
    """
    if not tenderers:
        return {"error": "evaluate_tender requires at least one tenderer."}
    if weights is None:
        weights = {"technical": 0.45, "commercial": 0.45, "hse": 0.07, "local_content": 0.03}
    weights, read_as_percent = _fraction_weights(weights)

    criteria = [name for name in weights if isinstance(name, str)]
    derived: Dict[str, Dict] = {}
    amount_scores: Dict[str, Dict[int, float]] = {}
    rows = [dict(t) if isinstance(t, dict) else t for t in tenderers]
    for criterion in criteria:
        if not _names_an_amount(criterion):
            continue
        scores = _amount_scores(rows, criterion)
        if scores is None:
            continue
        amount_scores[criterion] = scores
        for index, score in scores.items():
            rows[index][f"{criterion}_score"] = round(score, 2)
        derived[criterion] = {
            "rule": _PRICE_SCORE_RULE,
            "scores": {
                str(rows[i].get("name", i + 1)): round(s, 2) for i, s in scores.items()
            },
        }
    tenderers = rows
    if not any(
        _criterion_score(tenderer, criterion) is not None
        for tenderer in tenderers
        for criterion in criteria
    ):
        return {"error": _NO_CRITERION_SCORES}

    scored = []
    for index, t in enumerate(tenderers):
        total = 0.0
        for criterion in criteria:
            score = amount_scores.get(criterion, {}).get(index)
            if score is None:
                score = _criterion_score(t, criterion)
            total += (0.0 if score is None else score) * float(weights[criterion])
        scored.append({**t, "weighted_total": round(total, 2)})

    ranked = sorted(scored, key=lambda x: x["weighted_total"], reverse=True)
    for i, r in enumerate(ranked):
        r["rank"] = i + 1

    result = {
        "ranked_tenderers": ranked,
        "recommended": ranked[0] if ranked else None,
        "weights_applied": weights,
        "procedure": "tender_analysis",
    }
    if read_as_percent:
        result["weights_read_as_percent"] = True
    if derived:
        result["derived_scores"] = derived
    return result
