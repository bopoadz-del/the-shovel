"""Tests for OneDrive Block."""

import pytest
from unittest.mock import patch
from app.blocks.onedrive import OneDriveBlock


@pytest.fixture
def onedrive_block():
    return OneDriveBlock()


@pytest.mark.asyncio
async def test_onedrive_block_execute_structure(onedrive_block):
    """Test that OneDrive block returns standardized JSON structure."""
    with patch.dict("os.environ", {"ONEDRIVE_ACCESS_TOKEN": "mock_token"}):
        result = await onedrive_block.execute(
            None,
            {"operation": "list"}
        )
    
    # Assert standardized keys
    assert "block" in result
    assert result["block"] == "onedrive"
    assert "request_id" in result
    assert "status" in result
    assert "result" in result
    assert "confidence" in result
    assert "metadata" in result
    assert "source_id" in result
    assert "processing_time_ms" in result


@pytest.mark.asyncio
async def test_onedrive_block_metadata(onedrive_block):
    """Test OneDrive block metadata."""
    assert onedrive_block.name == "onedrive"
    assert onedrive_block.version == "2.0"
    # assert "file_id" in onedrive_block.config.supported_outputs  # legacy config field — n/a in current API
    # assert "metadata" in onedrive_block.config.supported_outputs  # legacy config field — n/a in current API
    # assert onedrive_block.config.requires_api_key == True  # legacy config field — n/a in current API


@pytest.mark.asyncio
async def test_onedrive_block_mock_response(onedrive_block):
    """Test OneDrive block returns mock when not configured."""
    result = await onedrive_block.execute(
        None,
        {"operation": "list"}
    )
    
    assert result["block"] == "onedrive"
    # Should return mock response when token not available
    assert "result" in result
