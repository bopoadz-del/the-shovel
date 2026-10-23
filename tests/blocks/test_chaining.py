"""Tests for block chaining (PDF → OCR → Chat)."""

import pytest
from unittest.mock import patch
from app.blocks import PDFBlock, OCRBlock, ChatBlock
from tests.conftest import requires_extended_boot


@pytest.mark.asyncio
async def test_pdf_to_ocr_to_chat_chain():
    """Test chaining PDF → OCR → Chat blocks."""
    
    # Step 1: Simulate PDF block output
    pdf_block = PDFBlock()
    pdf_result = await pdf_block.execute(
        {"file_path": "/tmp/test.pdf"},
        {"extract_text": True}
    )
    
    # Verify PDF result structure
    assert pdf_result["block"] == "pdf"
    assert "result" in pdf_result
    assert "request_id" in pdf_result
    assert "confidence" in pdf_result
    
    # Step 2: OCR block receives PDF output
    ocr_block = OCRBlock()
    
    # Create a mock previous result that looks like PDF output
    ocr_input = {
        "result": {
            "text": "This is extracted text from the PDF document."
        }
    }
    
    ocr_result = await ocr_block.execute(ocr_input, {})
    
    # Verify OCR result structure
    assert ocr_result["block"] == "ocr"
    assert "result" in ocr_result
    assert "request_id" in ocr_result
    assert "confidence" in ocr_result
    
    # Step 3: Chat block receives OCR output
    chat_block = ChatBlock()
    
    with patch.dict("os.environ", {"GROQ_API_KEY": "mock_key"}):
        chat_result = await chat_block.execute(
            ocr_result["result"],  # Pass OCR result as input
            {
                "provider": "mock",
                "prompt": "Summarize this text:",
                "system": "You are a helpful assistant."
            }
        )
    
    # Verify Chat result structure
    assert chat_result["block"] == "chat"
    assert "result" in chat_result
    assert "request_id" in chat_result
    assert "confidence" in chat_result
    
    # Verify standardized keys exist in all results
    for result in [pdf_result, ocr_result, chat_result]:
        assert "block" in result
        assert "request_id" in result
        assert "status" in result
        assert "result" in result
        assert "confidence" in result
        assert "metadata" in result
        assert "source_id" in result
        assert "processing_time_ms" in result
    
    print("✅ PDF → OCR → Chat chain successful!")
    print(f"   PDF processing time: {pdf_result['processing_time_ms']}ms")
    print(f"   OCR processing time: {ocr_result['processing_time_ms']}ms")
    print(f"   Chat processing time: {chat_result['processing_time_ms']}ms")


@requires_extended_boot
@pytest.mark.asyncio
async def test_standardized_response_format():
    """Test that all blocks return identical standardized response format."""
    from app.blocks import (
        PDFBlock, OCRBlock, ChatBlock, VoiceBlock, ImageBlock,
        SearchBlock, TranslateBlock, CodeBlock, WebBlock,
        GoogleDriveBlock, OneDriveBlock, LocalDriveBlock, AndroidDriveBlock,
    )
    
    required_keys = [
        "block", "request_id", "status", "result", 
        "confidence", "metadata", "source_id", "processing_time_ms"
    ]
    
    blocks = [
        (PDFBlock(), {"file_path": "/tmp/test.pdf"}, {"extract_text": False}),
        (OCRBlock(), {"base64": "iVBORw0KGgo="}, {}),
        (ChatBlock(), "hello", {"provider": "mock"}),
        (VoiceBlock(), "hello", {"operation": "tts", "provider": "mock"}),
        (ImageBlock(), "test", {"operation": "generate", "provider": "mock"}),
        (SearchBlock(), "test", {"provider": "mock"}),
        (TranslateBlock(), "hello", {"target": "es", "provider": "mock"}),
        (CodeBlock(), "print(1)", {"operation": "analyze"}),
        (WebBlock(), "https://example.com", {"operation": "fetch"}),
        (GoogleDriveBlock(), None, {"operation": "list"}),
        (OneDriveBlock(), None, {"operation": "list"}),
        (LocalDriveBlock(), None, {"operation": "list"}),
        (AndroidDriveBlock(), None, {"operation": "get_paths"}),
    ]
    
    with patch.dict("os.environ", {
        "GROQ_API_KEY": "mock_key",
        "OPENAI_API_KEY": "mock_key",
        "ONEDRIVE_ACCESS_TOKEN": "mock_token"
    }):
        for block, input_data, params in blocks:
            result = await block.execute(input_data, params)
            
            missing_keys = [key for key in required_keys if key not in result]
            assert not missing_keys, f"{block.name} missing keys: {missing_keys}"
            
            # Type checks
            assert isinstance(result["block"], str)
            assert isinstance(result["request_id"], str)
            assert isinstance(result["status"], str)
            assert isinstance(result["result"], dict)
            # None means "block reported no measured confidence — see Stream E".
            # A numeric value must be in the valid 0..1 range.
            assert result["confidence"] is None or isinstance(result["confidence"], (int, float)), \
                f"{block.name} confidence must be None or numeric, got {result['confidence']!r}"
            if isinstance(result["confidence"], (int, float)):
                assert 0.0 <= result["confidence"] <= 1.0, \
                    f"{block.name} confidence {result['confidence']} out of 0..1 range"
            assert isinstance(result["metadata"], dict)
            assert isinstance(result["processing_time_ms"], int)
            
            print(f"✅ {block.name} v{block.version} - standardized format OK")
