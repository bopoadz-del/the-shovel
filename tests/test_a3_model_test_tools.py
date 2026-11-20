"""The A3 verdict is computed from scored passes + criteria, never hardcoded.

Two inputs per decision: the same scorer on two different answer sets must
produce different verdicts, and a single flipped cell must flip the verdict.
"""

from __future__ import annotations

import pytest

from tools.a3_model_test.a3_model_test import (
    REPS,
    ROWS,
    _FAIL,
    _PASS,
    pass_rule,
    run_table,
    stub_answer,
)


def _answers(b_passes_overrides: dict[str, int] | None = None) -> dict:
    answers = {
        arm: {row: [stub_answer(arm, row, rep) for rep in range(REPS)] for row in ROWS}
        for arm in ("A", "B")
    }
    for row, n in (b_passes_overrides or {}).items():
        answers["B"][row] = [_PASS[row] if i < n else _FAIL for i in range(REPS)]
    return answers


@pytest.mark.parametrize(
    ("overrides", "verdict"),
    [
        (None, True),
        ({"A3": 3, "A5": 3, "A7": 3}, False),
        ({"B1": 4}, False),
    ],
)
def test_verdict_follows_the_data(overrides, verdict):
    passes, got, _why = run_table(_answers(overrides))
    assert got is verdict, passes


@pytest.mark.parametrize(
    ("row", "good", "bad"),
    [
        ("A3", "The total is 88 No, SAR 3,013,000.", "The total is 72 No."),
        ("B1", "Type 2 high yield deformed bars, grade 420.", "Grade 500 bars."),
    ],
)
def test_pass_rule_distinguishes_good_and_bad(row, good, bad):
    assert pass_rule(row, good) is True
    assert pass_rule(row, bad) is False
