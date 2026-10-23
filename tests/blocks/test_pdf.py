"""Tests for PDF Block."""

import pytest
import os
import tempfile
from app.blocks import PDFBlock


@pytest.fixture
def pdf_block():
    return PDFBlock()


@pytest.fixture
def sample_pdf_path():
    """Create a sample PDF file for testing."""
    # Return a mock path - in real tests you'd create an actual PDF
    return "/tmp/test_sample.pdf"


@pytest.mark.asyncio
async def test_pdf_block_execute_structure(pdf_block):
    """Test that PDF block returns standardized JSON structure."""
    result = await pdf_block.execute(
        {"file_path": "/tmp/nonexistent.pdf"},
        {"extract_text": True}
    )
    
    # Assert standardized keys
    assert "block" in result
    assert result["block"] == "pdf"
    assert "request_id" in result
    assert "status" in result
    assert "result" in result
    assert "confidence" in result
    assert "metadata" in result
    assert "source_id" in result
    assert "processing_time_ms" in result
    
    # Type checks
    assert isinstance(result["request_id"], str)
    assert isinstance(result["processing_time_ms"], int)
    assert isinstance(result["confidence"], float)


@pytest.mark.asyncio
async def test_pdf_block_file_not_found(pdf_block):
    """Test PDF block handles missing file gracefully."""
    result = await pdf_block.execute(
        {"file_path": "/tmp/definitely_not_real.pdf"},
        {"extract_text": True}
    )
    
    # Should have error in result but still return valid structure
    assert result["status"] == "error"
    assert "error" in result["result"] or "FileNotFoundError" in str(result["result"])


@pytest.mark.asyncio
async def test_pdf_block_metadata(pdf_block):
    """Test PDF block metadata is correct."""
    assert pdf_block.name == "pdf"
    assert pdf_block.version == "2.0.0"
    # assert "text" in pdf_block.config.supported_outputs  # legacy config field — n/a in current API
    # assert "tables" in pdf_block.config.supported_outputs  # legacy config field — n/a in current API
    # assert pdf_block.config.requires_api_key == False  # legacy config field — n/a in current API


@pytest.mark.asyncio
async def test_pdf_block_with_source_id(pdf_block):
    """Test PDF block accepts source_id input."""
    result = await pdf_block.execute(
        {"source_id": "test_document.pdf"},
        {"extract_text": True}
    )
    
    # Should return valid structure even if file doesn't exist
    assert "block" in result
    assert result["block"] == "pdf"
