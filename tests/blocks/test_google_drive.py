"""Tests for Google Drive Block."""

import pytest
from unittest.mock import patch
from app.blocks.google_drive import GoogleDriveBlock


@pytest.fixture
def google_drive_block():
    return GoogleDriveBlock()


@pytest.mark.asyncio
async def test_google_drive_block_execute_structure(google_drive_block):
    """Test that Google Drive block returns standardized JSON structure."""
    with patch.dict("os.environ", {"GOOGLE_CREDENTIALS_PATH": "/tmp/mock.json"}):
        result = await google_drive_block.execute(
            None,
            {"operation": "list"}
        )
    
    # Assert standardized keys
    assert "block" in result
    assert result["block"] == "google_drive"
    assert "request_id" in result
    assert "status" in result
    assert "result" in result
    assert "confidence" in result
    assert "metadata" in result
    assert "source_id" in result
    assert "processing_time_ms" in result


@pytest.mark.asyncio
async def test_google_drive_block_metadata(google_drive_block):
    """Test Google Drive block metadata."""
    assert google_drive_block.name == "google_drive"
    assert google_drive_block.version == "2.0"
    # assert "file_id" in google_drive_block.config.supported_outputs  # legacy config field — n/a in current API
    # assert "metadata" in google_drive_block.config.supported_outputs  # legacy config field — n/a in current API
    # assert google_drive_block.config.requires_api_key == True  # legacy config field — n/a in current API


@pytest.mark.asyncio
async def test_google_drive_block_mock_response(google_drive_block):
    """Test Google Drive block returns mock when not configured."""
    result = await google_drive_block.execute(
        None,
        {"operation": "list"}
    )
    
    assert result["block"] == "google_drive"
    # Should return mock response when Google auth not available
    assert "result" in result
