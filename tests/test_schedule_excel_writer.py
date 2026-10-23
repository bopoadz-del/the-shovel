"""Tests for the roadmap-named Schedule Excel Writer shim.

The real Excel writing is tested in ``tests/test_pm_excel.py``; this file only
proves that ``app.blocks.schedule_excel_writer.ScheduleExcelWriterBlock``
delegates to that working implementation and returns a file path.
"""

import os

import openpyxl
import pytest

from app.blocks.schedule_excel_writer import ScheduleExcelWriterBlock


@pytest.fixture
def block():
    return ScheduleExcelWriterBlock()


@pytest.mark.asyncio
async def test_excel_writer_delegates_to_generate_cost_loaded_schedule(block, tmp_path):
    activities = [
        {"id": "1", "wbs": "1.1", "name": "A", "duration": 5, "predecessors": [], "manpower": 10},
        {"id": "2", "wbs": "1.2", "name": "B", "duration": 5, "predecessors": ["1"], "manpower": 10},
    ]
    result = await block.process(
        {"activities": activities, "meta": {"project": "Test", "currency": "SAR"}}
    )

    assert result["status"] == "success"
    assert os.path.exists(result["file_path"])
    wb = openpyxl.load_workbook(result["file_path"])
    assert {"L2 Schedule", "Cost Loading", "Manpower Histogram", "Milestones", "Summary"} <= set(wb.sheetnames)


@pytest.mark.asyncio
async def test_excel_writer_returns_error_without_input(block):
    result = await block.process({})
    assert result["status"] == "error"
    assert "Provide cpm_output or activities" in result["error"]


def test_excel_writer_block_metadata():
    assert ScheduleExcelWriterBlock.name == "schedule_excel_writer"
    assert "excel" in ScheduleExcelWriterBlock.tags
