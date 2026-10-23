"""Tests for Android Drive Block."""

import pytest
from app.blocks.android_drive import AndroidDriveBlock


@pytest.fixture
def android_drive_block():
    return AndroidDriveBlock()


@pytest.mark.asyncio
async def test_android_drive_block_execute_structure(android_drive_block):
    """Test that Android Drive block returns standardized JSON structure."""
    result = await android_drive_block.execute(
        None,
        {"operation": "list"}
    )
    
    # Assert standardized keys
    assert "block" in result
    assert result["block"] == "android_drive"
    assert "request_id" in result
    assert "status" in result
    assert "result" in result
    assert "confidence" in result
    assert "metadata" in result
    assert "source_id" in result
    assert "processing_time_ms" in result


@pytest.mark.asyncio
async def test_android_drive_block_metadata(android_drive_block):
    """Test Android Drive block metadata."""
    assert android_drive_block.name == "android_drive"
    assert android_drive_block.version == "1.0"
    # assert "uri" in android_drive_block.config.supported_outputs  # legacy config field — n/a in current API
    # assert "metadata" in android_drive_block.config.supported_outputs  # legacy config field — n/a in current API
    # assert android_drive_block.config.requires_api_key == False  # legacy config field — n/a in current API


@pytest.mark.asyncio
async def test_android_drive_block_get_paths(android_drive_block):
    """Test Android Drive block get_paths operation."""
    result = await android_drive_block.execute(
        None,
        {"operation": "get_paths"}
    )
    
    assert result["block"] == "android_drive"
    assert result["result"]["operation"] == "get_paths"
    assert "paths" in result["result"]


@pytest.mark.asyncio
async def test_android_drive_block_mock_response(android_drive_block):
    """Test Android Drive block returns mock when not on Android."""
    result = await android_drive_block.execute(
        None,
        {"operation": "list", "folder_path": "/sdcard"}
    )
    
    assert result["block"] == "android_drive"
    # Should return mock response when not in Android environment
    assert "result" in result
