"""Tests for the roadmap-named Fast-Track Analyzer shim.

The real compression logic is tested in ``tests/test_pm_computations.py``; this
file only proves that ``app.blocks.fasttrack_analyzer.FastTrackAnalyzerBlock``
delegates to that working implementation and returns the expected shape.
"""

import pytest

from app.blocks.fasttrack_analyzer import FastTrackAnalyzerBlock


@pytest.fixture
def block():
    return FastTrackAnalyzerBlock()


@pytest.mark.asyncio
async def test_fasttrack_analyzer_delegates_to_compress_schedule(block):
    activities = [
        {"id": "A", "duration": 10, "predecessors": []},
        {"id": "B", "duration": 8, "predecessors": [{"predecessor_id": "A"}]},
    ]
    result = await block.process(
        {"activities": activities, "reductions": {"A": 3}, "total_delay_days": 10}
    )

    assert result["status"] == "success"
    assert result["baseline_duration"] == 18
    assert result["revised_duration"] == 15
    assert result["days_saved"] == 3
    assert result["compressed_activity_ids"] == ["A"]
    assert len(result["scenarios"]) == 3


@pytest.mark.asyncio
async def test_fasttrack_analyzer_returns_error_without_activities(block):
    result = await block.process({"reductions": {"A": 3}})
    assert result["status"] == "error"
    assert "No activities" in result["error"]


@pytest.mark.asyncio
async def test_fasttrack_analyzer_returns_error_without_reductions(block):
    result = await block.process(
        {"activities": [{"id": "A", "duration": 5, "predecessors": []}]}
    )
    assert result["status"] == "error"
    assert "No reductions" in result["error"]


def test_fasttrack_analyzer_block_metadata():
    assert FastTrackAnalyzerBlock.name == "fasttrack_analyzer"
    assert "fast-track" in FastTrackAnalyzerBlock.tags
