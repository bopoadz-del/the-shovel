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
    """
    if not tenderers:
        return {"error": "evaluate_tender requires at least one tenderer."}
    if weights is None:
        weights = {"technical": 0.45, "commercial": 0.45, "hse": 0.07, "local_content": 0.03}

    criteria = [name for name in weights if isinstance(name, str)]
    if not any(
        _criterion_score(tenderer, criterion) is not None
        for tenderer in tenderers
        for criterion in criteria
    ):
        return {"error": _NO_CRITERION_SCORES}

    scored = []
    for t in tenderers:
        total = 0.0
        for criterion in criteria:
            score = _criterion_score(t, criterion)
            total += (0.0 if score is None else score) * float(weights[criterion])
        scored.append({**t, "weighted_total": round(total, 2)})

    ranked = sorted(scored, key=lambda x: x["weighted_total"], reverse=True)
    for i, r in enumerate(ranked):
        r["rank"] = i + 1

    return {
        "ranked_tenderers": ranked,
        "recommended": ranked[0] if ranked else None,
        "weights_applied": weights,
        "procedure": "tender_analysis",
    }
