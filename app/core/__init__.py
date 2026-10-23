"""Core framework for Cerebrum Blocks."""

from .block import BaseBlock, BlockConfig
from .chain import Chain, chain
from .client import CerebrumClient
from .universal_base import UniversalBlock, UniversalContainer
from .typed_block import TypedBlock
from .schema_registry import (
    SchemaRegistry, registry, get_registry,
    TextContent, ImageContent, PDFContent,
    ConstructionAnalysis, ChatMessage, ChatConversation,
    SearchResult, VectorEmbedding, FileContent,
    AudioContent, VideoContent, CodeResult, TranslationResult,
    validate_text_content, validate_image_content,
    validate_pdf_content, validate_chat_message,
)
from .data_transformer import DataTransformer, transformer, get_transformer, transform

__all__ = [
    # Legacy
    "BaseBlock",
    "BlockConfig", 
    "Chain",
    "chain",
    "CerebrumClient",
    # Universal
    "UniversalBlock",
    "UniversalContainer",
    # Typed
    "TypedBlock",
    # Registry
    "SchemaRegistry",
    "registry",
    "get_registry",
    # Standard Types
    "TextContent",
    "ImageContent",
    "PDFContent",
    "ConstructionAnalysis",
    "ChatMessage",
    "ChatConversation",
    "SearchResult",
    "VectorEmbedding",
    "FileContent",
    "AudioContent",
    "VideoContent",
    "CodeResult",
    "TranslationResult",
    # Validation
    "validate_text_content",
    "validate_image_content",
    "validate_pdf_content",
    "validate_chat_message",
    # Transformer
    "DataTransformer",
    "transformer",
    "get_transformer",
    "transform",
]
