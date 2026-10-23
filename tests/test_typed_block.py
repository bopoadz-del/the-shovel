"""
Test for TypedBlock foundation - demonstrates PDF → transform → Chat flow.

This test shows how the new schema foundation enables blocks to connect properly.
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core import (
    TypedBlock,
    TextContent, ChatMessage,
    registry, get_registry,
    transformer, get_transformer, transform,
    validate_text_content,
)
from app.core.universal_base import UniversalBlock
from app.blocks import PDFBlock, OCRBlock, ChatBlock


class TestTypedBlockBasics:
    """Test the new TypedBlock base class."""
    
    def test_typedblock_extends_universalblock(self):
        """TypedBlock should extend UniversalBlock."""
        # Create a simple typed block
        class TestBlock(TypedBlock):
            name = "test"
            version = "1.0"
            input_schema = TextContent
            output_schema = TextContent
            
            async def process(self, input_data, params=None):
                return {"text": input_data.get("text", "")}
        
        block = TestBlock()
        
        # Should be instance of both
        assert isinstance(block, TypedBlock)
        assert isinstance(block, UniversalBlock)
        
        # Should have schema attributes
        assert block.input_schema == TextContent
        assert block.output_schema == TextContent
    
    @pytest.mark.asyncio
    async def test_input_validation(self):
        """Test input schema validation."""
        class TestBlock(TypedBlock):
            name = "test"
            version = "1.0"
            input_schema = TextContent
            output_schema = TextContent
            
            async def process(self, input_data, params=None):
                return {"text": input_data.get("text", "")}
        
        block = TestBlock()
        
        # Valid input
        result = block.validate_input({"text": "Hello world"})
        assert result["valid"] is True
        assert len(result["errors"]) == 0
        
        # Invalid input - missing required field
        result = block.validate_input({"source": "test"})
        assert result["valid"] is False
        assert any("text" in e for e in result["errors"])
    
    @pytest.mark.asyncio
    async def test_output_validation(self):
        """Test output schema validation."""
        class TestBlock(TypedBlock):
            name = "test"
            version = "1.0"
            input_schema = TextContent
            output_schema = TextContent
            
            async def process(self, input_data, params=None):
                return {"text": input_data.get("text", "")}
        
        block = TestBlock()
        
        # Execute and validate output
        result = await block.execute({"text": "Hello"}, {})
        
        # Should have standard UniversalBlock structure
        assert "block" in result
        assert "request_id" in result
        assert "status" in result
        assert "result" in result
        assert "confidence" in result
        
        # Validate output structure
        validation = block.validate_output(result)
        assert "valid" in validation
        assert "errors" in validation


class TestSchemaRegistry:
    """Test the schema registry and type system."""
    
    def test_registry_singleton(self):
        """Registry should be a singleton."""
        r1 = get_registry()
        r2 = get_registry()
        assert r1 is r2
    
    def test_standard_types_registered(self):
        """Standard types should be pre-registered."""
        r = get_registry()
        
        expected_types = [
            "TextContent", "ImageContent", "PDFContent",
            "ConstructionAnalysis", "ChatMessage", "ChatConversation",
            "SearchResult", "VectorEmbedding", "FileContent",
            "AudioContent", "VideoContent", "CodeResult", "TranslationResult"
        ]
        
        for type_name in expected_types:
            assert r.get_schema(type_name) is not None, f"{type_name} should be registered"
    
    def test_type_validation(self):
        """Test type validation."""
        r = get_registry()
        
        # Valid TextContent
        result = r.validate({"text": "Hello"}, "TextContent")
        assert result["valid"] is True
        
        # Invalid TextContent - missing text
        result = r.validate({"source": "test"}, "TextContent")
        assert result["valid"] is False
        assert any("text" in e.lower() for e in result["errors"])
        
        # Valid ChatMessage
        result = r.validate({"role": "user", "content": "Hello"}, "ChatMessage")
        assert result["valid"] is True
    
    def test_type_compatibility(self):
        """Test type compatibility checking."""
        r = get_registry()
        
        # Same type is compatible
        assert r.are_compatible("TextContent", "TextContent")
        
        # Declared compatible types
        assert r.are_compatible("TextContent", "ChatMessage")
        assert r.are_compatible("ChatMessage", "TextContent")
        
        # Incompatible types
        assert not r.are_compatible("TextContent", "ImageContent")


class TestDataTransformer:
    """Test data transformation between block formats."""
    
    def test_transformer_singleton(self):
        """Transformer should be a singleton."""
        t1 = get_transformer()
        t2 = get_transformer()
        assert t1 is t2
    
    def test_pdf_to_text_content(self):
        """Test transforming PDF block output to TextContent."""
        # Simulate PDF block output
        pdf_output = {
            "result": {
                "text": "Extracted PDF text content",
                "pages": 5,
                "filename": "document.pdf",
                "status": "success"
            }
        }
        
        # Transform to TextContent
        text_content, _ = transform(pdf_output, "TextContent", "pdf")
        
        assert text_content["text"] == "Extracted PDF text content"
        assert text_content["source"] == "pdf"
        assert text_content["metadata"]["pages"] == 5
        assert text_content["metadata"]["filename"] == "document.pdf"
        
        # Validate as TextContent
        validation = validate_text_content(text_content)
        assert validation["valid"]
    
    def test_ocr_to_text_content(self):
        """Test transforming OCR block output to TextContent."""
        ocr_output = {
            "result": {
                "text": "Scanned text from image",
                "confidence": 0.95,
                "word_count": 42,
                "engine": "easyocr",
                "preprocessed": True
            }
        }
        
        text_content, _ = transform(ocr_output, "TextContent", "ocr")
        
        assert text_content["text"] == "Scanned text from image"
        assert text_content["source"] == "ocr"
        assert text_content["metadata"]["confidence"] == 0.95
    
    def test_chat_to_text_content(self):
        """Test transforming Chat block output to TextContent."""
        chat_output = {
            "result": {
                "text": "AI response message",
                "provider": "deepseek",
                "model": "deepseek-chat",
                "tokens": {"prompt": 10, "completion": 20}
            }
        }
        
        text_content, _ = transform(chat_output, "TextContent", "chat")
        
        assert text_content["text"] == "AI response message"
        assert text_content["source"] == "chat"
        assert text_content["metadata"]["provider"] == "deepseek"
    
    def test_chat_to_chat_message(self):
        """Test transforming Chat block output to ChatMessage."""
        chat_output = {
            "result": {
                "text": "AI response message",
                "provider": "deepseek",
                "model": "deepseek-chat"
            }
        }
        
        message, _ = transform(chat_output, "ChatMessage", "chat")
        
        assert message["role"] == "assistant"
        assert message["content"] == "AI response message"
        assert "timestamp" in message
        assert message["metadata"]["provider"] == "deepseek"
    
    def test_generic_string_transform(self):
        """Test generic transformation of string to TextContent."""
        text_content, _ = transform("Plain string input", "TextContent")
        
        assert text_content["text"] == "Plain string input"
        assert text_content["source"] == "string"
    
    def test_list_transformers(self):
        """Test listing available transformers."""
        t = get_transformer()
        transformers = t.list_transformers()
        
        # Should have transformers for major blocks
        assert "pdf" in transformers
        assert "ocr" in transformers
        assert "chat" in transformers


class TestPDFToChatFlow:
    """Test the full PDF → Transform → Chat flow."""
    
    @pytest.mark.asyncio
    async def test_pdf_output_to_chat_input(self):
        """Demonstrate PDF block output transformed for Chat block input."""
        # Create blocks
        pdf_block = PDFBlock()
        
        # Mock PDF execution result
        pdf_result = {
            "block": "pdf",
            "request_id": "abc123",
            "status": "success",
            "result": {
                "text": "This is construction specifications. Concrete: 100m³, Steel: 50 tons.",
                "pages": 10,
                "filename": "specs.pdf",
                "status": "success"
            },
            "confidence": 0.95,
            "source_id": "pdf-abc123",
            "metadata": {"version": "1.2"},
            "processing_time_ms": 150
        }
        
        # Step 1: Transform PDF output to TextContent
        text_content, _ = transform(pdf_result, "TextContent", "pdf")
        
        # Verify transformation
        assert text_content["text"] == "This is construction specifications. Concrete: 100m³, Steel: 50 tons."
        assert text_content["source"] == "pdf"
        assert text_content["metadata"]["pages"] == 10
        
        # Validate as TextContent
        validation = validate_text_content(text_content)
        assert validation["valid"], f"Validation errors: {validation['errors']}"
        
        # Step 2: Extract text for Chat block input
        chat_input = text_content["text"]
        assert isinstance(chat_input, str)
        
        # Step 3: Create Chat block with typed schema
        class TypedChatBlock(TypedBlock):
            name = "typed_chat"
            version = "1.0"
            input_schema = TextContent
            output_schema = ChatMessage
            accepted_input_types = ["TextContent", "string"]
            produced_output_types = ["ChatMessage"]
            
            async def process(self, input_data, params=None):
                # Handle both TextContent dict and raw string
                if isinstance(input_data, dict):
                    text = input_data.get("text", "")
                else:
                    text = str(input_data)
                
                return {
                    "role": "assistant",
                    "content": f"Received: {text[:50]}...",
                    "timestamp": "1234567890",
                    "metadata": {"processed": True}
                }
        
        chat_block = TypedChatBlock()
        
        # Step 4: Pass TextContent directly (block accepts it)
        chat_result = await chat_block.execute(text_content, {})
        
        assert chat_result["status"] == "success"
        assert "result" in chat_result
        assert "role" in chat_result["result"]
        assert "content" in chat_result["result"]
        
        print("✅ PDF → TextContent → Chat flow successful!")
        print(f"   PDF extracted {text_content['metadata']['pages']} pages")
        print(f"   Chat received: {chat_result['result']['content'][:40]}...")
    
    @pytest.mark.asyncio
    async def test_ocr_pdf_chat_chain(self):
        """Test OCR → PDF → Chat chain with transformations."""
        # Simulated OCR output
        ocr_result = {
            "result": {
                "text": "Steel beams: 200 units, Concrete blocks: 500 units",
                "confidence": 0.92,
                "word_count": 10,
                "engine": "easyocr"
            }
        }
        
        # Transform OCR output to TextContent
        ocr_text, _ = transform(ocr_result, "TextContent", "ocr")
        assert ocr_text["text"] == "Steel beams: 200 units, Concrete blocks: 500 units"
        assert ocr_text["source"] == "ocr"
        
        # Wrap as PDF content (simulating construction analysis workflow)
        pdf_content = {
            "file_path": "/tmp/construction_plan.pdf",
            "filename": "construction_plan.pdf",
            "text": ocr_text["text"],
            "pages": 1,
            "metadata": {"extracted_from": "image"}
        }
        
        # Transform to TextContent for Chat
        final_text, _ = transform({"result": pdf_content}, "TextContent")
        
        # Chat block input
        chat_prompt = f"Analyze this construction data: {final_text['text']}"
        assert "Steel beams" in chat_prompt
        assert "Concrete blocks" in chat_prompt
        
        print("✅ OCR → PDF → TextContent → Chat chain successful!")


class TestBackwardCompatibility:
    """Ensure existing blocks work with new schema system."""
    
    @pytest.mark.asyncio
    async def test_legacy_universal_block_still_works(self):
        """PDF block should be TypedBlock now."""
        from app.blocks.pdf import PDFBlock
        
        pdf_block = PDFBlock()
        
        # Should be TypedBlock (migrated)
        from app.core.typed_block import TypedBlock
        assert isinstance(pdf_block, TypedBlock)
        
        # Should have TypedBlock attributes
        assert hasattr(pdf_block, 'validate_input')
        assert hasattr(pdf_block, 'input_schema')
        assert pdf_block.input_schema is not None
    
    @pytest.mark.asyncio
    async def test_transformer_works_with_legacy_blocks(self):
        """Transformer should work with legacy block outputs."""
        # Legacy-style block result (as returned by UniversalBlock.execute)
        legacy_result = {
            "block": "pdf",
            "request_id": "xyz789",
            "status": "success",
            "result": {
                "text": "Legacy block output text",
                "pages": 3
            },
            "confidence": 0.95,
            "source_id": "pdf-xyz789",
            "metadata": {"version": "1.0"},
            "processing_time_ms": 100
        }
        
        # Should transform successfully
        text_content, _ = transform(legacy_result, "TextContent", "pdf")
        
        assert text_content["text"] == "Legacy block output text"
        assert text_content["source"] == "pdf"


if __name__ == "__main__":
    # Run tests with pytest if available
    try:
        pytest.main([__file__, "-v"])
    except ImportError:
        # Manual test run
        print("Running manual tests...")
        
        # Test transformer
        pdf_output = {"result": {"text": "Hello from PDF", "pages": 5}}
        text, _ = transform(pdf_output, "TextContent", "pdf")
        print(f"✅ Transform test: {text['text']}")
        
        # Test registry
        r = get_registry()
        print(f"✅ Registry has {len(r.list_types())} types")
        
        print("\n✅ All manual tests passed!")
