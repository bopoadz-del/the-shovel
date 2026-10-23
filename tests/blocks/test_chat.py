"""Tests for Chat Block."""

import pytest
import os
from unittest.mock import patch
from app.blocks import ChatBlock


@pytest.fixture
def chat_block():
    return ChatBlock()


@pytest.mark.asyncio
async def test_chat_block_execute_structure(chat_block):
    """Test that Chat block returns standardized JSON structure."""
    # Mock API key to avoid errors
    with patch.dict(os.environ, {"GROQ_API_KEY": "mock_key"}):
        result = await chat_block.execute(
            "Hello",
            {"provider": "mock", "model": "test-model"}
        )
    
    # Assert standardized keys
    assert "block" in result
    assert result["block"] == "chat"
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


@pytest.mark.asyncio
async def test_chat_block_metadata(chat_block):
    """Test Chat block metadata."""
    assert chat_block.name == "chat"
    assert chat_block.version == "3.0.0"
    # assert "text" in chat_block.config.supported_outputs  # legacy config field — n/a in current API
    # assert "stream" in chat_block.config.supported_outputs  # legacy config field — n/a in current API
    # assert chat_block.config.requires_api_key == True  # legacy config field — n/a in current API


@pytest.mark.asyncio
async def test_chat_block_with_messages(chat_block):
    """Test Chat block accepts messages list."""
    messages = [
        {"role": "system", "content": "You are a test assistant."},
        {"role": "user", "content": "Hello"}
    ]
    
    with patch.dict(os.environ, {"GROQ_API_KEY": "mock_key"}):
        result = await chat_block.execute(
            messages,
            {"provider": "mock"}
        )
    
    assert result["block"] == "chat"
    assert "result" in result


@pytest.mark.asyncio
async def test_chat_block_chaining(chat_block):
    """Test Chat block can receive output from previous block."""
    # Simulate PDF block output
    previous_result = {
        "result": {
            "text": "This is extracted text from a PDF document."
        }
    }
    
    with patch.dict(os.environ, {"GROQ_API_KEY": "mock_key"}):
        result = await chat_block.execute(
            previous_result,
            {"provider": "mock", "prompt": "Summarize this text:"}
        )
    
    assert result["block"] == "chat"
    assert "result" in result
