"""The dimension-table scorer decides what pollutes cost answers.

`_score_dimension_table` is the audit trail behind
`exclude_from_cost_grounding`. Drawing dimension tables are full of numbers
with units, and when they leak into cost retrieval the model quotes a
schedule-of-dimensions figure as a rate. The scorer is what keeps them out --
and it was running at 6% of its body, i.e. essentially unexercised.

Each scoring component is tested in isolation so a regression names itself,
rather than only asserting the final boolean, which would leave three of the
four contributions untested.
"""
from __future__ import annotations

import pytest

import app.core.drawing_chunk_classifier as dcc
from app.core.drawing_chunk_classifier import _score_dimension_table


@pytest.fixture
def classify(monkeypatch):
    """classify_chunk is behind the DRAWING_CHUNK_TAGGING flag, which is OFF by
    default -- it returns a zeroed envelope and never calls the scorer. These
    tests are about what it does when ENABLED, so the flag is turned on
    explicitly rather than the tests quietly asserting the disabled shape."""
    monkeypatch.setattr(dcc, "TAGGING_ENABLED", True)
    return dcc.classify_chunk


def test_disabled_flag_returns_a_zeroed_envelope_not_a_wrong_answer(monkeypatch):
    """With tagging off, nothing may be excluded from cost grounding -- and the
    envelope must say so rather than omitting the fields."""
    monkeypatch.setattr(dcc, "TAGGING_ENABLED", False)
    tags = dcc.classify_chunk("Item | Description | Qty\n1 | Column | 12 nos")
    assert tags["tagging_enabled"] is False
    assert tags["exclude_from_cost_grounding"] is False
    assert tags["classifier_score"] == 0


def test_no_signal_scores_zero():
    assert _score_dimension_table("The contractor shall coordinate access.") == 0


def test_a_dimension_table_header_is_worth_three():
    assert _score_dimension_table("Item  Description  Qty") == 3


def test_the_header_bonus_is_counted_once_not_per_pattern():
    """Several header patterns can match the same line. The loop breaks on the
    first, so a header-rich line must not stack to 6 or 9."""
    text = "Item Description Qty | Material Size Quantity | No. Description Qty"
    # header(3) + at most the density/pipe bonuses (2 + 1)
    assert _score_dimension_table(text) <= 6


def test_dense_dimension_numbers_add_two():
    """>15% of words being dimensioned numbers is the strong signal."""
    text = "1200 mm 800 mm 450 mm 300 mm"
    assert _score_dimension_table(text) == 2


def test_a_few_dimension_numbers_add_only_one():
    """Three or more matches, but sparse: the weaker signal."""
    text = (
        "The wall is 1200 mm high and the opening is 800 mm wide with a "
        "clear soffit of 450 mm above finished floor level throughout this "
        "particular area of the works as shown on the relevant drawings."
    )
    assert _score_dimension_table(text) == 1


def test_pipe_delimited_rows_add_one():
    text = "a | b\nc | d"
    assert _score_dimension_table(text) == 1


def test_a_single_pipe_row_is_not_a_table():
    assert _score_dimension_table("a | b") == 0


def test_the_components_accumulate(classify):
    """A real drawing schedule hits header + density + pipe rows."""
    text = (
        "Item | Description | Qty\n"
        "1 | Column C1 | 12 nos\n"
        "2 | Beam B2 | 8 nos\n"
    )
    score = _score_dimension_table(text)
    assert score >= 4, score
    # and it must be classified as a dimension table, not merely score well
    tags = classify(text)
    assert tags["is_drawing_dimension_table"] is True
    assert tags["exclude_from_cost_grounding"] is True
    assert tags["classifier_score"] == score


def test_prose_is_not_excluded_from_cost_grounding(classify):
    """The other direction. Over-excluding is the worse failure: it silently
    removes real content from cost retrieval."""
    tags = classify(
        "Payment shall be made monthly against certified progress."
    )
    assert tags["is_drawing_dimension_table"] is False
    assert tags["exclude_from_cost_grounding"] is False


def test_classify_reports_the_same_score_the_scorer_returns(classify):
    """The audit field must not drift from the decision it explains."""
    for text in ("Item Description Qty", "1200 mm 800 mm", "nothing here"):
        assert classify(text)["classifier_score"] == _score_dimension_table(text)
