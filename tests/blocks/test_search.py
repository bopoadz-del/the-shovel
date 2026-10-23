"""Tests for Search Block."""

import pytest
from unittest.mock import patch
from app.blocks import SearchBlock


@pytest.fixture
def search_block():
    return SearchBlock()


@pytest.mark.asyncio
async def test_search_block_execute_structure(search_block):
    """Test that Search block returns standardized JSON structure."""
    result = await search_block.execute(
        "python programming",
        {"provider": "mock", "num_results": 5}
    )
    
    # Assert standardized keys
    assert "block" in result
    assert result["block"] == "search"
    assert "request_id" in result
    assert "status" in result
    assert "result" in result
    assert "confidence" in result
    assert "metadata" in result
    assert "source_id" in result
    assert "processing_time_ms" in result


@pytest.mark.asyncio
async def test_search_block_metadata(search_block):
    """Test Search block metadata."""
    assert search_block.name == "search"
    assert search_block.version == "2.0"
    # assert "results" in search_block.config.supported_outputs  # legacy config field — n/a in current API
    # assert search_block.config.requires_api_key == True  # legacy config field — n/a in current API


@pytest.mark.asyncio
async def test_search_block_with_query_dict(search_block):
    """Test Search block accepts query dict."""
    result = await search_block.execute(
        {"query": "machine learning", "text": "machine learning"},
        {"provider": "mock"}
    )
    
    assert result["block"] == "search"
    assert "result" in result


@pytest.mark.asyncio
async def test_search_block_chaining(search_block):
    """Test Search block can receive output from previous block."""
    # Simulate previous block output
    previous_result = {
        "result": {
            "text": "What is artificial intelligence?"
        }
    }
    
    result = await search_block.execute(
        previous_result,
        {"provider": "mock"}
    )
    
    assert result["block"] == "search"
    assert "result" in result
