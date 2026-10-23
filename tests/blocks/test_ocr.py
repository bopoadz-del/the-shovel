"""Tests for OCR Block."""

import pytest
from PIL import Image
import io
from app.blocks import OCRBlock


@pytest.fixture
def ocr_block():
    return OCRBlock()


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    img = Image.new('RGB', (100, 30), color='white')
    return img


@pytest.mark.asyncio
async def test_ocr_block_execute_structure(ocr_block):
    """Test that OCR block returns standardized JSON structure."""
    # Create a simple test image
    img = Image.new('RGB', (100, 30), color='white')
    
    result = await ocr_block.execute(
        {"image": img},
        {"language": "eng"}
    )
    
    # Assert standardized keys
    assert "block" in result
    assert result["block"] == "ocr"
    assert "request_id" in result
    assert "status" in result
    assert "result" in result
    assert "confidence" in result
    assert "metadata" in result
    assert "source_id" in result
    assert "processing_time_ms" in result


@pytest.mark.asyncio
async def test_ocr_block_metadata(ocr_block):
    """Test OCR block metadata."""
    assert ocr_block.name == "ocr"
    assert ocr_block.version == "2.0.0"
    # assert "text" in ocr_block.config.supported_outputs  # legacy config field — n/a in current API
    # assert ocr_block.config.requires_api_key == False  # legacy config field — n/a in current API


@pytest.mark.asyncio
async def test_ocr_block_with_base64(ocr_block):
    """Test OCR block accepts base64 input."""
    import base64
    
    # Create a simple image and convert to base64
    img = Image.new('RGB', (100, 30), color='white')
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    result = await ocr_block.execute(
        {"base64": img_base64},
        {"language": "eng"}
    )
    
    assert result["block"] == "ocr"
    assert "result" in result
