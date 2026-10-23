"""Tests for block chaining."""

import pytest
from app.core import CerebrumClient, chain

class TestBlockChaining:
    """Test suite for chaining blocks together."""
    
    @pytest.fixture
    def client(self):
        return CerebrumClient(base_url="http://localhost:8000")
    
    @pytest.mark.asyncio
    async def test_build_chain(self, client):
        """Test building a chain."""
        chain_builder = chain(client)
        
        chain_builder.then("chat", {"provider": "mock"})
        chain_builder.then("translate", {"target": "es"})
        
        assert len(chain_builder.steps) == 2
        assert chain_builder.steps[0]["block"] == "chat"
        assert chain_builder.steps[1]["block"] == "translate"
    
    @pytest.mark.asyncio
    async def test_chain_fluent_interface(self, client):
        """Test chain fluent interface."""
        chain_builder = chain(client)
        
        result = chain_builder.then("pdf", {}).then("ocr", {})
        
        assert result is chain_builder  # Should return self for chaining
        assert len(chain_builder.steps) == 2


class TestChainResult:
    """Tests for ChainResult."""
    
    def test_chain_result_success(self):
        """Test ChainResult success property."""
        from app.core.chain import ChainResult
        
        steps = [
            {"status": "success"},
            {"status": "success"}
        ]
        
        result = ChainResult(steps, {"output": "test"})
        assert result.success == True
    
    def test_chain_result_failure(self):
        """Test ChainResult failure detection."""
        from app.core.chain import ChainResult
        
        steps = [
            {"status": "success"},
            {"status": "error"}
        ]
        
        result = ChainResult(steps, {"output": "test"})
        assert result.success == False
    
    def test_chain_result_total_time(self):
        """Test ChainResult total time calculation."""
        from app.core.chain import ChainResult
        
        steps = [
            {"status": "success", "processing_time_ms": 100},
            {"status": "success", "processing_time_ms": 200}
        ]
        
        result = ChainResult(steps, {})
        assert result.total_time_ms == 300
    
    def test_chain_result_get_step(self):
        """Test ChainResult get_step method."""
        from app.core.chain import ChainResult
        
        steps = [
            {"block": "pdf", "status": "success"},
            {"block": "ocr", "status": "success"}
        ]
        
        result = ChainResult(steps, {})
        assert result.get_step(0)["block"] == "pdf"
        assert result.get_step(1)["block"] == "ocr"
        assert result.get_step(999) == {}  # Out of bounds


class TestCommonPipelines:
    """Tests for common AI pipelines."""
    
    @pytest.mark.asyncio
    async def test_document_processing_pipeline(self):
        """Test PDF -> OCR -> Chat pipeline structure."""
        from app.core import chain, CerebrumClient
        
        client = CerebrumClient()
        chain_builder = chain(client)
        
        # Build document processing pipeline
        chain_builder.then("pdf", {"extract_text": True})
        chain_builder.then("chat", {"prompt": "Summarize this document:"})
        
        assert len(chain_builder.steps) == 2
        assert chain_builder.steps[0]["block"] == "pdf"
        assert chain_builder.steps[1]["block"] == "chat"
    
    @pytest.mark.asyncio
    async def test_multilingual_pipeline(self):
        """Test OCR -> Translate -> Chat pipeline."""
        from app.core import chain, CerebrumClient
        
        client = CerebrumClient()
        chain_builder = chain(client)
        
        chain_builder.then("ocr", {"language": "deu"})
        chain_builder.then("translate", {"target": "en"})
        chain_builder.then("chat", {"prompt": "Extract key information:"})
        
        assert len(chain_builder.steps) == 3
    
    @pytest.mark.asyncio
    async def test_search_pipeline(self):
        """Test Document -> Search -> Chat pipeline.

        Exercises the chain BUILDER (step recording), not the blocks: nothing
        is executed here, so any registered block name serves.
        """
        from app.core import chain, CerebrumClient

        client = CerebrumClient()
        chain_builder = chain(client)

        chain_builder.then("search", {"operation": "add"})
        chain_builder.then("search", {"operation": "search", "top_k": 5})
        chain_builder.then("chat", {"prompt": "Answer based on these documents:"})

        assert len(chain_builder.steps) == 3
        assert chain_builder.steps[0]["block"] == "search"
    
    @pytest.mark.asyncio
    async def test_rag_pipeline(self):
        """Test RAG (Retrieval Augmented Generation) pipeline."""
        from app.core import chain, CerebrumClient
        
        client = CerebrumClient()
        chain_builder = chain(client)
        
        # Retrieve relevant documents then generate answer
        chain_builder.then("search", {"operation": "search", "top_k": 3})
        chain_builder.then("chat", {"prompt": "Based on the retrieved context:"})
        
        assert len(chain_builder.steps) == 2
