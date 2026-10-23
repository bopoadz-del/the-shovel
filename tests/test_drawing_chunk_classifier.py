"""
Test the drawing number-table chunk classifier (W3 class fix).

Regression test for the cost-fabrication incident:
  query: "concrete (250 kg/cm2)" -> drawing dimension tables pulled above
  real rate chunks -> LLM fabricated "450 SAR/m3"
"""

from __future__ import annotations

import pytest

from app.core.drawing_chunk_classifier import (
    TAGGING_ENABLED,
    _looks_like_dimension_table,
    classify_chunk,
    filter_cost_grounding_chunks,
    tag_chunks_for_ingestion,
)


# -- Dimension table fixtures (should be tagged) --------------------------

DIM_TABLE_1 = """
| Item | Description | Size | Qty | Unit |
|------|-------------|------|-----|------|
| 1 | Concrete slab | 250x300 mm | 450 | m3 |
| 2 | Reinforcement | 12mm dia | 1200 | kg |
| 3 | Formwork | plywood | 800 | m2 |
"""

DIM_TABLE_2 = """
No. Description Quantity Unit
1   Concrete C40  450      m3
2   Rebar 16mm    1200     kg
3   Formwork      800      m2
"""

DRAWING_SCHEDULE = """
Drawing No: AB-2023-101
Item | Mark | Description | Size | Qty
A    | SF-1 | Slab formwork | 1200x2400 mm | 45 nos
B    | RB-1 | Rebar mat     | 12mm @ 150   | 1200 kg
"""

# -- Cost table fixtures (should NOT be tagged) ---------------------------

COST_TABLE_1 = """
| Item | Description | Unit | Rate SAR | Amount SAR |
|------|-------------|------|----------|------------|
| 1    | Concrete    | m3   | 450.00   | 202,500    |
| 2    | Rebar       | kg   | 2.60     | 3,120      |
"""

BOQ_TABLE = """
Bill of Quantities -- Package A
Item | Description | Unit | Qty | Rate | Total
1    | Excavation  | m3   | 500 | 25   | 12,500
2    | Concrete    | m3   | 450 | 380  | 171,000
"""

PAYMENT_TABLE = """
Interim Payment Certificate No. 4
Description | Amount SAR
Period work | 1,250,000
Retention   | -62,500
Net due     | 1,187,500
"""

# -- Tests ----------------------------------------------------------------


class TestDimensionTableDetection:
    """Dimension tables should be detected and tagged."""

    def test_pipe_table_with_qty_unit(self):
        assert _looks_like_dimension_table(DIM_TABLE_1) is True

    def test_simple_item_list_with_qty(self):
        assert _looks_like_dimension_table(DIM_TABLE_2) is True

    def test_drawing_schedule(self):
        assert _looks_like_dimension_table(DRAWING_SCHEDULE) is True


class TestCostTableNotTagged:
    """Cost/price/BOQ tables must NOT be tagged as dimension tables."""

    def test_cost_table_with_sar(self):
        assert _looks_like_dimension_table(COST_TABLE_1) is False

    def test_boq_table_with_rate_total(self):
        assert _looks_like_dimension_table(BOQ_TABLE) is False

    def test_payment_certificate(self):
        assert _looks_like_dimension_table(PAYMENT_TABLE) is False


class TestClassifierOutput:
    """classify_chunk returns correct structure."""

    def test_dimension_table_classified_when_enabled(self):
        if not TAGGING_ENABLED:
            pytest.skip("TAGGING_ENABLED is False -- detection logic tested via _looks_like_dimension_table")
        result = classify_chunk(DIM_TABLE_1)
        assert result["is_drawing_dimension_table"] is True
        assert result["exclude_from_cost_grounding"] is True
        assert result["classifier_score"] >= 3

    def test_cost_table_not_classified(self):
        result = classify_chunk(COST_TABLE_1)
        assert result["is_drawing_dimension_table"] is False
        assert result["exclude_from_cost_grounding"] is False

    def test_when_tagging_disabled(self):
        """When TAGGING_ENABLED is False, all chunks pass through untagged."""
        result = classify_chunk(DIM_TABLE_1)
        assert result["tagging_enabled"] is TAGGING_ENABLED
        if not TAGGING_ENABLED:
            assert result["is_drawing_dimension_table"] is False
            assert result["exclude_from_cost_grounding"] is False

    def test_classifier_version_present(self):
        result = classify_chunk(DIM_TABLE_1)
        assert result["classifier_version"] == "1.0"


class TestChunkTagging:
    """tag_chunks_for_ingestion adds _classifier field."""

    def test_tags_dimension_table_chunks(self):
        chunks = [
            {"text": DIM_TABLE_1, "chunk_id": "c1"},
            {"text": COST_TABLE_1, "chunk_id": "c2"},
        ]
        tagged = tag_chunks_for_ingestion(chunks)
        if TAGGING_ENABLED:
            assert "_classifier" in tagged[0]
            assert "_classifier" in tagged[1]
            assert tagged[0]["_classifier"]["exclude_from_cost_grounding"] is True
            assert tagged[1]["_classifier"]["exclude_from_cost_grounding"] is False
        else:
            # When disabled, _classifier field is NOT added
            assert "_classifier" not in tagged[0]
            assert "_classifier" not in tagged[1]


class TestCostGroundingFilter:
    """filter_cost_grounding_chunks removes dimension-table chunks."""

    def test_filters_dimension_chunks(self):
        if not TAGGING_ENABLED:
            pytest.skip("TAGGING_ENABLED is False")

        chunks = [
            {"text": DIM_TABLE_1, "chunk_id": "c1",
             "_classifier": {"exclude_from_cost_grounding": True}},
            {"text": COST_TABLE_1, "chunk_id": "c2",
             "_classifier": {"exclude_from_cost_grounding": False}},
            {"text": "Some regular text about concrete pouring", "chunk_id": "c3",
             "_classifier": {"exclude_from_cost_grounding": False}},
        ]
        filtered = filter_cost_grounding_chunks(chunks)
        ids = {c["chunk_id"] for c in filtered}
        assert "c1" not in ids, "Dimension table should be filtered out"
        assert "c2" in ids, "Cost table should remain"
        assert "c3" in ids, "Regular text should remain"

    def test_noop_when_disabled(self):
        """When TAGGING_ENABLED is False, all chunks pass through."""
        chunks = [
            {"text": DIM_TABLE_1, "chunk_id": "c1"},
            {"text": COST_TABLE_1, "chunk_id": "c2"},
        ]
        filtered = filter_cost_grounding_chunks(chunks)
        assert len(filtered) == 2


class TestRegressionCostFabrication:
    """
    Regression test: the exact incident that caused cost fabrication.
    A dimension table with "450" and "m3" must be tagged so it cannot
    ground a cost claim of "450 SAR/m3".
    """

    def test_incident_chunk_detected(self):
        """The chunk that caused the 450 SAR/m3 fabrication must be detected."""
        incident_chunk = """
        | Item | Description | Size | Qty | Unit |
        | 1 | Concrete | 250 kg/cm2 | 450 | m3 |
        | 2 | Rebar | 16mm | 1200 | kg |
        """
        # Test the core detection logic (works regardless of env flag)
        assert _looks_like_dimension_table(incident_chunk) is True
        # When tagging is enabled, classify_chunk should also tag it
        result = classify_chunk(incident_chunk)
        if TAGGING_ENABLED:
            assert result["is_drawing_dimension_table"] is True
            assert result["exclude_from_cost_grounding"] is True
