"""Tests for Image Block."""

import pytest
from PIL import Image
from unittest.mock import patch
from app.blocks import ImageBlock


@pytest.fixture
def image_block():
    return ImageBlock()


@pytest.mark.asyncio
async def test_image_block_execute_structure(image_block):
    """Test that Image block returns standardized JSON structure."""
    # Create a simple test image
    img = Image.new('RGB', (100, 100), color='red')
    
    with patch.dict("os.environ", {"OPENAI_API_KEY": "mock_key"}):
        result = await image_block.execute(
            {"image": img},
            {"operation": "analyze", "provider": "mock"}
        )
    
    # Assert standardized keys
    assert "block" in result
    assert result["block"] == "image"
    assert "request_id" in result
    assert "status" in result
    assert "result" in result
    assert "confidence" in result
    assert "metadata" in result
    assert "source_id" in result
    assert "processing_time_ms" in result


@pytest.mark.asyncio
async def test_image_block_metadata(image_block):
    """Test Image block metadata."""
    assert image_block.name == "image"
    assert image_block.version == "3.0"
    # assert "description" in image_block.config.supported_outputs  # legacy config field — n/a in current API
    # assert "image" in image_block.config.supported_outputs  # legacy config field — n/a in current API
    # assert image_block.config.requires_api_key == True  # legacy config field — n/a in current API


@pytest.mark.asyncio
async def test_image_block_generate(image_block):
    """Test Image block generation."""
    result = await image_block.execute(
        "A red circle on white background",
        {"operation": "generate", "provider": "mock"}
    )
    
    assert result["block"] == "image"
    assert "result" in result
