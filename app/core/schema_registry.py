"""
Schema Registry - Global registry for data types in Cerebrum Blocks.

This module provides:
- Standard type definitions (TextContent, ImageContent, etc.)
- Type validation logic
- DataTransformer for converting between compatible types
"""

from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field


# ============================================================================
# STANDARD TYPE DEFINITIONS (JSON Schema format)
# ============================================================================

TextContent = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "source": {"type": "string"},  # "pdf", "ocr", "upload", "chat", etc.
        "metadata": {"type": "object"}
    },
    "required": ["text"]
}
"""Standard text content type - universal format for text data."""

ImageContent = {
    "type": "object",
    "properties": {
        "image_data": {"type": "string"},  # base64 or URL
        "format": {"type": "string"},  # "jpg", "png", "webp", etc.
        "width": {"type": "number"},
        "height": {"type": "number"},
        "source": {"type": "string"},
        "metadata": {"type": "object"}
    },
    "required": ["image_data"]
}
"""Standard image content type - universal format for image data."""

PDFContent = {
    "type": "object",
    "properties": {
        "file_path": {"type": "string"},
        "filename": {"type": "string"},
        "pages": {"type": "number"},
        "text": {"type": "string"},  # extracted text
        "metadata": {"type": "object"}
    },
    "required": ["file_path"]
}
"""Standard PDF content type - contains PDF file and extracted data."""

ConstructionAnalysis = {
    "type": "object",
    "properties": {
        "measurements": {"type": "array"},
        "quantities": {"type": "object"},
        "materials": {"type": "array"},
        "confidence": {"type": "number"},
        "raw_text": {"type": "string"},
        "metadata": {"type": "object"}
    },
    "required": []
}
"""Construction analysis result - measurements, quantities, materials."""

ChatMessage = {
    "type": "object",
    "properties": {
        "role": {"type": "string"},  # "user", "assistant", "system"
        "content": {"type": "string"},
        "timestamp": {"type": "string"},
        "metadata": {"type": "object"}
    },
    "required": ["role", "content"]
}
"""Standard chat message format."""

ChatConversation = {
    "type": "object",
    "properties": {
        "messages": {
            "type": "array",
            "items": ChatMessage
        },
        "metadata": {"type": "object"}
    },
    "required": ["messages"]
}
"""Collection of chat messages (a conversation)."""

SearchResult = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "results": {"type": "array"},
        "total_found": {"type": "number"},
        "source": {"type": "string"},
        "metadata": {"type": "object"}
    },
    "required": ["query", "results"]
}
"""Standard search result format."""

VectorEmbedding = {
    "type": "object",
    "properties": {
        "vector": {"type": "array"},  # list of floats
        "dimension": {"type": "number"},
        "text": {"type": "string"},  # original text
        "id": {"type": "string"},
        "metadata": {"type": "object"}
    },
    "required": ["vector"]
}
"""Vector embedding result."""

FileContent = {
    "type": "object",
    "properties": {
        "file_path": {"type": "string"},
        "filename": {"type": "string"},
        "content": {"type": "string"},  # base64 for binary
        "mime_type": {"type": "string"},
        "size": {"type": "number"},
        "metadata": {"type": "object"}
    },
    "required": ["file_path", "filename"]
}
"""Standard file content type."""

AudioContent = {
    "type": "object",
    "properties": {
        "audio_data": {"type": "string"},  # base64 or URL
        "format": {"type": "string"},  # "mp3", "wav", "ogg", etc.
        "duration": {"type": "number"},
        "text": {"type": "string"},  # transcript if available
        "metadata": {"type": "object"}
    },
    "required": ["audio_data"]
}
"""Standard audio content type."""

VideoContent = {
    "type": "object",
    "properties": {
        "video_data": {"type": "string"},  # base64 or URL
        "format": {"type": "string"},
        "duration": {"type": "number"},
        "width": {"type": "number"},
        "height": {"type": "number"},
        "metadata": {"type": "object"}
    },
    "required": ["video_data"]
}
"""Standard video content type."""

CodeResult = {
    "type": "object",
    "properties": {
        "code": {"type": "string"},
        "language": {"type": "string"},
        "output": {"type": "string"},
        "error": {"type": "string"},
        "analysis": {"type": "string"},
        "metadata": {"type": "object"}
    },
    "required": ["code"]
}
"""Code execution/analysis result."""

TranslationResult = {
    "type": "object",
    "properties": {
        "original_text": {"type": "string"},
        "translated_text": {"type": "string"},
        "source_language": {"type": "string"},
        "target_language": {"type": "string"},
        "metadata": {"type": "object"}
    },
    "required": ["original_text", "translated_text", "target_language"]
}
"""Translation result format."""


# ============================================================================
# TYPE REGISTRY
# ============================================================================

@dataclass
class TypeInfo:
    """Information about a registered type."""
    name: str
    schema: Dict[str, Any]
    description: str
    compatible_types: List[str] = field(default_factory=list)


class SchemaRegistry:
    """
    Global registry for data types in Cerebrum Blocks.
    
    Provides:
    - Type registration and lookup
    - Schema validation
    - Type compatibility checking
    """
    
    _instance = None
    _types: Dict[str, TypeInfo] = {}
    
    def __new__(cls):
        """Singleton pattern - only one registry instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._register_standard_types()
        return cls._instance
    
    def _register_standard_types(self):
        """Register all standard types."""
        self.register("TextContent", TextContent, "Standard text content")
        self.register("ImageContent", ImageContent, "Standard image content")
        self.register("PDFContent", PDFContent, "Standard PDF content")
        self.register("ConstructionAnalysis", ConstructionAnalysis, "Construction analysis result")
        self.register("ChatMessage", ChatMessage, "Standard chat message")
        self.register("ChatConversation", ChatConversation, "Chat conversation")
        self.register("SearchResult", SearchResult, "Search result")
        self.register("VectorEmbedding", VectorEmbedding, "Vector embedding")
        self.register("FileContent", FileContent, "File content")
        self.register("AudioContent", AudioContent, "Audio content")
        self.register("VideoContent", VideoContent, "Video content")
        self.register("CodeResult", CodeResult, "Code execution result")
        self.register("TranslationResult", TranslationResult, "Translation result")
        
        # Mark compatible types
        self._types["TextContent"].compatible_types = ["ChatMessage"]
        self._types["ChatMessage"].compatible_types = ["TextContent"]
        self._types["PDFContent"].compatible_types = ["TextContent", "FileContent"]
        self._types["FileContent"].compatible_types = ["PDFContent"]
    
    def register(self, name: str, schema: Dict, description: str = "") -> None:
        """Register a new type."""
        self._types[name] = TypeInfo(
            name=name,
            schema=schema,
            description=description
        )
    
    def get_schema(self, name: str) -> Optional[Dict]:
        """Get the schema for a type."""
        type_info = self._types.get(name)
        return type_info.schema if type_info else None
    
    def get_type_info(self, name: str) -> Optional[TypeInfo]:
        """Get full type information."""
        return self._types.get(name)
    
    def list_types(self) -> List[str]:
        """List all registered types."""
        return list(self._types.keys())
    
    def validate(self, data: Any, type_name: str) -> Dict[str, Any]:
        """
        Validate data against a type schema.
        
        Returns:
            Dict with validation results:
            {
                "valid": bool,
                "errors": List[str],
                "warnings": List[str]
            }
        """
        schema = self.get_schema(type_name)
        if not schema:
            return {
                "valid": False,
                "errors": [f"Unknown type: {type_name}"],
                "warnings": []
            }
        
        return self._validate_against_schema(data, schema)
    
    def _validate_against_schema(self, data: Any, schema: Dict) -> Dict[str, Any]:
        """Validate data against a JSON schema."""
        errors = []
        warnings = []
        
        schema_type = schema.get("type", "any")
        
        if schema_type == "object":
            if not isinstance(data, dict):
                errors.append(f"Expected object, got {type(data).__name__}")
            else:
                # Check properties
                properties = schema.get("properties", {})
                for prop_name, prop_schema in properties.items():
                    if prop_name in data:
                        prop_value = data[prop_name]
                        prop_type = prop_schema.get("type")
                        
                        if prop_type == "string" and not isinstance(prop_value, str):
                            errors.append(f"Field '{prop_name}' should be string")
                        elif prop_type == "number" and not isinstance(prop_value, (int, float)):
                            errors.append(f"Field '{prop_name}' should be number")
                        elif prop_type == "array" and not isinstance(prop_value, list):
                            errors.append(f"Field '{prop_name}' should be array")
                        elif prop_type == "object" and not isinstance(prop_value, dict):
                            errors.append(f"Field '{prop_name}' should be object")
                
                # Check required fields
                required = schema.get("required", [])
                for field in required:
                    if field not in data:
                        errors.append(f"Missing required field: '{field}'")
        
        elif schema_type == "array":
            if not isinstance(data, list):
                errors.append(f"Expected array, got {type(data).__name__}")
        
        elif schema_type == "string":
            if not isinstance(data, str):
                errors.append(f"Expected string, got {type(data).__name__}")
        
        elif schema_type == "number":
            if not isinstance(data, (int, float)):
                errors.append(f"Expected number, got {type(data).__name__}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    def are_compatible(self, source_type: str, target_type: str) -> bool:
        """Check if two types are compatible (can be converted)."""
        if source_type == target_type:
            return True
        
        source_info = self._types.get(source_type)
        if source_info and target_type in source_info.compatible_types:
            return True
        
        return False
    
    def get_compatibility_chain(self, source_type: str, target_type: str) -> List[str]:
        """
        Get the chain of conversions needed.
        
        Returns empty list if no conversion needed.
        Returns ["transformer_name"] if direct conversion exists.
        Returns multiple items for multi-step conversion.
        """
        if self.are_compatible(source_type, target_type):
            return []
        
        # For now, simple check - more complex pathfinding can be added
        return ["auto_transform"]


# Global registry instance
registry = SchemaRegistry()


def get_registry() -> SchemaRegistry:
    """Get the global schema registry."""
    return registry


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def validate_text_content(data: Any) -> Dict[str, Any]:
    """Quick validation for TextContent."""
    return registry.validate(data, "TextContent")


def validate_image_content(data: Any) -> Dict[str, Any]:
    """Quick validation for ImageContent."""
    return registry.validate(data, "ImageContent")


def validate_pdf_content(data: Any) -> Dict[str, Any]:
    """Quick validation for PDFContent."""
    return registry.validate(data, "PDFContent")


def validate_chat_message(data: Any) -> Dict[str, Any]:
    """Quick validation for ChatMessage."""
    return registry.validate(data, "ChatMessage")


# Export all types
__all__ = [
    # Registry
    "SchemaRegistry",
    "registry",
    "get_registry",
    "TypeInfo",
    
    # Standard types
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
    
    # Validation helpers
    "validate_text_content",
    "validate_image_content",
    "validate_pdf_content",
    "validate_chat_message",
]
