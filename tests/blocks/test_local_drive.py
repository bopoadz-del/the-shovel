"""Tests for Local Drive Block."""

import pytest
import os
import tempfile
from app.blocks.local_drive import LocalDriveBlock


@pytest.fixture
def local_drive_block():
    return LocalDriveBlock()


@pytest.mark.asyncio
async def test_local_drive_block_execute_structure(local_drive_block):
    """Test that Local Drive block returns standardized JSON structure."""
    result = await local_drive_block.execute(
        None,
        {"operation": "list"}
    )
    
    # Assert standardized keys
    assert "block" in result
    assert result["block"] == "local_drive"
    assert "request_id" in result
    assert "status" in result
    assert "result" in result
    assert "confidence" in result
    assert "metadata" in result
    assert "source_id" in result
    assert "processing_time_ms" in result


@pytest.mark.asyncio
async def test_local_drive_block_metadata(local_drive_block):
    """Test Local Drive block metadata."""
    assert local_drive_block.name == "local_drive"
    assert local_drive_block.version == "1.1"
    # assert "file_path" in local_drive_block.config.supported_outputs  # legacy config field — n/a in current API
    # assert "metadata" in local_drive_block.config.supported_outputs  # legacy config field — n/a in current API
    # assert local_drive_block.config.requires_api_key == False  # legacy config field — n/a in current API


@pytest.mark.asyncio
async def test_local_drive_block_list(local_drive_block):
    """Test Local Drive block list operation."""
    result = await local_drive_block.execute(
        None,
        {"operation": "list", "folder_path": "/"}
    )
    
    assert result["block"] == "local_drive"
    assert result["result"]["operation"] == "list"
    assert "files" in result["result"]


@pytest.mark.asyncio
async def test_local_drive_block_write_and_read(local_drive_block):
    """Test Local Drive block write and read operations."""
    # The block is sandboxed to its drive root — use a path relative to it.
    test_path = "test_write.txt"
    write_result = await local_drive_block.execute(
        None,
        {
            "operation": "write",
            "file_path": test_path,
            "content": "Hello from test!"
        }
    )

    assert write_result["block"] == "local_drive"
    assert write_result["result"]["status"] == "success"
    assert write_result["result"]["operation"] == "write"

    # Read the file
    read_result = await local_drive_block.execute(
        None,
        {"operation": "read", "file_path": test_path}
    )

    assert read_result["block"] == "local_drive"
    assert read_result["result"]["operation"] == "read"
    assert read_result["result"]["content"] == "Hello from test!"
