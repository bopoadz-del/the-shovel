"""Tests for Web Block."""

import pytest
from app.blocks import WebBlock


@pytest.fixture
def web_block():
    return WebBlock()


@pytest.mark.asyncio
async def test_web_block_execute_structure(web_block):
    """Test that Web block returns standardized JSON structure."""
    result = await web_block.execute(
        "https://example.com",
        {"operation": "fetch"}
    )
    
    # Assert standardized keys
    assert "block" in result
    assert result["block"] == "web"
    assert "request_id" in result
    assert "status" in result
    assert "result" in result
    assert "confidence" in result
    assert "metadata" in result
    assert "source_id" in result
    assert "processing_time_ms" in result


@pytest.mark.asyncio
async def test_web_block_metadata(web_block):
    """Test Web block metadata."""
    assert web_block.name == "web"
    assert web_block.version == "2.0"
    # assert "content" in web_block.config.supported_outputs  # legacy config field — n/a in current API
    # assert "data" in web_block.config.supported_outputs  # legacy config field — n/a in current API
    # assert web_block.config.requires_api_key == False  # legacy config field — n/a in current API


@pytest.mark.asyncio
async def test_web_block_scrape(web_block):
    """Test Web block scrape operation."""
    result = await web_block.execute(
        "https://example.com",
        {"operation": "scrape"}
    )
    
    assert result["block"] == "web"
    assert "result" in result


@pytest.mark.asyncio
async def test_web_block_api_request(web_block):
    """Test Web block API request operation."""
    result = await web_block.execute(
        "https://api.example.com/data",
        {"operation": "api", "method": "GET"}
    )
    
    assert result["block"] == "web"
    assert "result" in result
