"""
DataTransformer - Convert between block output formats.

This module transforms block-specific output formats to standard types:
- PDF returns {"text": "...", "pages": 5} → transform to standard TextContent
- OCR returns {"result": {"text": "..."}} → extract and normalize
- Construction returns complex dict → wrap properly
"""

import json
from typing import Any, Dict, List, Optional, Callable, Union
from .schema_registry import (
    TextContent,
    ImageContent,
    PDFContent,
    ChatMessage,
    ChatConversation,
    SearchResult,
    VectorEmbedding,
    FileContent,
    registry,
    get_registry
)


class DataTransformer:
    """
    DataTransformer - Converts block output formats to standard types.
    
    Each block may output data in its own format. DataTransformer
    normalizes these to standard types for interoperability.
    
    Example:
        pdf_output = {"text": "Hello", "pages": 5}
        text_content = transformer.transform(pdf_output, "TextContent")
        # Result: {"text": "Hello", "source": "pdf", "metadata": {...}}
    """
    
    _instance = None
    _transformers: Dict[str, Dict[str, Callable]] = {}
    
    # Type constants for Orchestrator compatibility
    UNKNOWN = "Unknown"
    TEXT = "Text"
    JSON = "JSON"
    FILE = "File"
    PDF = "PDF"
    IMAGE = "Image"
    TABLE = "Table"
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._register_default_transformers()
        return cls._instance
    
    def _register_default_transformers(self):
        """Register all default transformers."""
        # PDFBlock output → various types
        self.register("pdf", "TextContent", self._pdf_to_text)
        self.register("pdf_v2", "TextContent", self._pdf_to_text)
        self.register("pdf", "PDFContent", self._pdf_to_pdf_content)
        
        # OCRBlock output → various types
        self.register("ocr", "TextContent", self._ocr_to_text)
        self.register("ocr_v2", "TextContent", self._ocr_to_text)
        
        # ChatBlock output → various types
        self.register("chat", "TextContent", self._chat_to_text)
        self.register("chat", "ChatMessage", self._chat_to_message)
        
        # SearchBlock output → various types
        self.register("search", "SearchResult", self._search_to_result)
        
        # ImageBlock output → various types
        self.register("image", "ImageContent", self._image_to_content)
        
        # File blocks → FileContent
        self.register("google_drive", "FileContent", self._drive_to_file)
        self.register("onedrive", "FileContent", self._drive_to_file)
        self.register("local_drive", "FileContent", self._drive_to_file)
        self.register("android_drive", "FileContent", self._drive_to_file)
        
        # Voice blocks
        self.register("voice", "AudioContent", self._voice_to_audio)
        
        # Translate blocks
        self.register("translate", "TranslationResult", self._translate_to_result)
        
        # Code blocks
        self.register("code", "CodeResult", self._code_to_result)
        
        # Construction blocks
        self.register("construction", "TextContent", self._construction_to_text)
    
    # ========================================================================
    # CONSTRUCTION TRANSFORMERS
    # ========================================================================
    
    def _construction_to_text(self, data: Dict) -> Dict:
        """Transform ConstructionAnalysis output to TextContent for chat block."""
        # Handle UniversalBlock wrapped result
        if "result" in data:
            data = data["result"]
        
        # Extract the most relevant text from construction analysis
        text_parts = []
        
        # Add summary if available
        if "summary" in data:
            text_parts.append(f"Summary: {data['summary']}")
        
        # Add measurements
        if "measurements" in data and data["measurements"]:
            text_parts.append(f"Measurements found: {len(data['measurements'])}")
            for m in data["measurements"][:10]:
                if isinstance(m, dict):
                    text_parts.append(f"  - {m.get('type', 'unknown')}: {m.get('value', 0)} {m.get('unit', '')}")
        
        # Add quantities
        if "quantities" in data and data["quantities"]:
            text_parts.append(f"Quantities: {json.dumps(data['quantities'], indent=2)[:500]}")
        
        # Add specifications
        if "specifications" in data and data["specifications"]:
            text_parts.append(f"Specifications found: {len(data['specifications'])}")
        
        # Add detected disciplines
        if "detected_disciplines" in data and data["detected_disciplines"]:
            text_parts.append(f"Disciplines: {', '.join(data['detected_disciplines'])}")
        
        # Add cost estimate
        if "cost_estimate" in data and data["cost_estimate"]:
            ce = data["cost_estimate"]
            text_parts.append(f"Estimated cost: {ce.get('total_with_overhead', 'N/A')}")
        
        # Add carbon estimate
        if "carbon_estimate" in data and data["carbon_estimate"]:
            ce = data["carbon_estimate"]
            text_parts.append(f"Embodied carbon: {ce.get('total_embodied_carbon_kg', 'N/A')} kg CO2")
        
        # Add raw text if nothing else
        if not text_parts and "raw_text" in data:
            text_parts.append(data["raw_text"][:2000])
        
        # Add file name info
        if "file_name" in data:
            text_parts.insert(0, f"File: {data['file_name']}")
        
        combined_text = "\n\n".join(text_parts) if text_parts else "Construction analysis completed."
        
        return {
            "text": combined_text,
            "source": "construction",
            "metadata": {
                "file_name": data.get("file_name", ""),
                "doc_type": data.get("doc_type", ""),
                "confidence": data.get("confidence", {}),
                "total_pages": data.get("total_pages", 0),
                "original_data": {k: v for k, v in data.items() if k != "original_data"}
            }
        }
    
    # ========================================================================
    # TRANSFORMER METHODS
    # ========================================================================
    
    def _pdf_to_text(self, data: Dict) -> Dict:
        """Transform PDF block output to TextContent."""
        # Handle UniversalBlock wrapped result
        if "result" in data:
            data = data["result"]
        
        return {
            "text": data.get("text", ""),
            "source": "pdf",
            "metadata": {
                "pages": data.get("pages", 0),
                "filename": data.get("filename", ""),
                "status": data.get("status", "unknown")
            }
        }
    
    def _pdf_to_pdf_content(self, data: Dict) -> Dict:
        """Transform PDF block output to PDFContent."""
        if "result" in data:
            data = data["result"]
        
        return {
            "file_path": data.get("file_path", ""),
            "filename": data.get("filename", ""),
            "pages": data.get("pages", 0),
            "text": data.get("text", ""),
            "metadata": {
                "status": data.get("status", "unknown")
            }
        }
    
    def _ocr_to_text(self, data: Dict) -> Dict:
        """Transform OCR block output to TextContent."""
        if "result" in data:
            data = data["result"]
        
        return {
            "text": data.get("text", ""),
            "source": "ocr",
            "metadata": {
                "confidence": data.get("confidence", 0),
                "word_count": data.get("word_count", 0),
                "engine": data.get("engine", ""),
                "preprocessed": data.get("preprocessed", False)
            }
        }
    
    def _chat_to_text(self, data: Dict) -> Dict:
        """Transform Chat block output to TextContent."""
        if "result" in data:
            data = data["result"]
        
        return {
            "text": data.get("text", ""),
            "source": "chat",
            "metadata": {
                "provider": data.get("provider", ""),
                "model": data.get("model", ""),
                "tokens": data.get("tokens", {})
            }
        }
    
    def _chat_to_message(self, data: Dict) -> Dict:
        """Transform Chat block output to ChatMessage."""
        if "result" in data:
            data = data["result"]
        
        import time
        return {
            "role": "assistant",
            "content": data.get("text", ""),
            "timestamp": str(int(time.time())),
            "metadata": {
                "provider": data.get("provider", ""),
                "model": data.get("model", ""),
                "tokens": data.get("tokens", {})
            }
        }
    
    def _search_to_result(self, data: Dict) -> Dict:
        """Transform Search block output to SearchResult."""
        if "result" in data:
            data = data["result"]
        
        return {
            "query": data.get("query", ""),
            "results": data.get("results", []),
            "total_found": data.get("total_found", 0),
            "source": data.get("provider", "search"),
            "metadata": data.get("metadata", {})
        }
    
    def _image_to_content(self, data: Dict) -> Dict:
        """Transform Image block output to ImageContent."""
        if "result" in data:
            data = data["result"]
        
        return {
            "image_data": data.get("image_data", data.get("image_url", "")),
            "format": data.get("format", "png"),
            "width": data.get("width", 0),
            "height": data.get("height", 0),
            "source": data.get("provider", "image"),
            "metadata": {
                "description": data.get("description", ""),
                "status": data.get("status", "unknown")
            }
        }
    
    def _drive_to_file(self, data: Dict) -> Dict:
        """Transform Drive block output to FileContent."""
        if "result" in data:
            data = data["result"]
        
        return {
            "file_path": data.get("file_path", data.get("path", "")),
            "filename": data.get("filename", data.get("name", "")),
            "content": data.get("content", ""),
            "mime_type": data.get("mime_type", data.get("content_type", "")),
            "size": data.get("size", 0),
            "metadata": data.get("metadata", {})
        }
    
    def _voice_to_audio(self, data: Dict) -> Dict:
        """Transform Voice block output to AudioContent."""
        if "result" in data:
            data = data["result"]
        
        return {
            "audio_data": data.get("audio_data", data.get("audio_url", "")),
            "format": data.get("format", "mp3"),
            "duration": data.get("duration", 0),
            "text": data.get("text", data.get("transcript", "")),
            "metadata": data.get("metadata", {})
        }
    
    def _translate_to_result(self, data: Dict) -> Dict:
        """Transform Translate block output to TranslationResult."""
        if "result" in data:
            data = data["result"]
        
        return {
            "original_text": data.get("original_text", data.get("source_text", "")),
            "translated_text": data.get("translated_text", "") or data.get("translation", ""),
            "source_language": data.get("source_language", data.get("source_lang", "auto")),
            "target_language": data.get("target_language", data.get("target_lang", "")),
            "metadata": data.get("metadata", {})
        }
    
    def _code_to_result(self, data: Dict) -> Dict:
        """Transform Code block output to CodeResult."""
        if "result" in data:
            data = data["result"]
        
        return {
            "code": data.get("code", ""),
            "language": data.get("language", ""),
            "output": data.get("output", ""),
            "error": data.get("error", ""),
            "analysis": data.get("analysis", data.get("review", "")),
            "metadata": data.get("metadata", {})
        }
    
    # ========================================================================
    # PUBLIC API
    # ========================================================================
    
    def register(self, source_block: str, target_type: str, transformer: Callable) -> None:
        """
        Register a transformer function.
        
        Args:
            source_block: Block name (e.g., "pdf", "ocr", "chat")
            target_type: Target type name (e.g., "TextContent", "ImageContent")
            transformer: Function that takes data dict and returns transformed dict
        """
        if source_block not in self._transformers:
            self._transformers[source_block] = {}
        self._transformers[source_block][target_type] = transformer
    
    def can_transform(self, source_block: str, target_type: str) -> bool:
        """Check if transformation is available."""
        return (
            source_block in self._transformers and 
            target_type in self._transformers[source_block]
        )
    
    def transform(self, data: Any, target_type: str, source_block: str = None, field_mapping: dict = None) -> tuple:
        """
        Transform data to target type.
        
        Args:
            data: Input data (can be raw block output or wrapped result)
            target_type: Target type name (e.g., "TextContent")
            source_block: Optional source block hint for selecting transformer
            field_mapping: Optional field mapping dict (orchestrator compat)
            
        Returns:
            Tuple of (transformed_data, operation_name)
        """
        # If data is None or empty, return empty target type
        if data is None:
            return self._empty_of_type(target_type), "empty"
        
        # Extract data from UniversalBlock result wrapper if present
        if isinstance(data, dict) and "result" in data:
            inner_data = data["result"]
        else:
            inner_data = data
        
        # Apply field mapping if provided (before type transform)
        if field_mapping and isinstance(inner_data, dict):
            mapped = {}
            for target_field, source_field in field_mapping.items():
                if source_field in inner_data:
                    mapped[target_field] = inner_data[source_field]
            if mapped:
                inner_data = {**inner_data, **mapped}
        
        # Try to infer source block from data if not provided
        if source_block is None and isinstance(inner_data, dict):
            source_block = self._infer_source_block(inner_data)
        
        # Try specific transformer
        if source_block and self.can_transform(source_block, target_type):
            transformer = self._transformers[source_block][target_type]
            return transformer(data), f"{source_block}_to_{target_type}"
        
        # Try generic transformation based on target type
        generic_result = self._generic_transform(inner_data, target_type)
        if generic_result:
            return generic_result, f"generic_{target_type}"
        
        # Return best-effort result
        return self._best_effort_transform(inner_data, target_type), "best_effort"
    
    def _infer_source_block(self, data: Dict) -> Optional[str]:
        """Infer source block from data content."""
        # Check for block-specific fields
        if "pages" in data and "filename" in data:
            return "pdf"
        if "confidence" in data and "word_count" in data:
            return "ocr"
        if "model" in data and "provider" in data:
            return "chat"
        if "results" in data and "query" in data:
            return "search"
        if "translated_text" in data:
            return "translate"
        if "file_id" in data or "file_path" in data:
            return "google_drive"
        return None
    
    def _generic_transform(self, data: Any, target_type: str) -> Optional[Dict]:
        """Try generic transformation based on target type."""
        if target_type == "TextContent":
            return self._to_text_content(data)
        elif target_type == "ChatMessage":
            return self._to_chat_message(data)
        elif target_type == "SearchResult":
            return self._to_search_result(data)
        return None
    
    def _best_effort_transform(self, data: Any, target_type: str) -> Dict:
        """Best-effort transformation when no specific transformer exists."""
        if target_type == "TextContent":
            return {
                "text": str(data) if not isinstance(data, dict) else str(data.get("text", data)),
                "source": "unknown",
                "metadata": {"original": data}
            }
        elif target_type == "ChatMessage":
            import time
            return {
                "role": "assistant",
                "content": str(data) if not isinstance(data, dict) else str(data.get("text", data)),
                "timestamp": str(int(time.time())),
                "metadata": {}
            }
        else:
            # Return data as-is for unknown types
            return data if isinstance(data, dict) else {"data": data}
    
    def _empty_of_type(self, target_type: str) -> Dict:
        """Return empty structure for a type."""
        if target_type == "TextContent":
            return {"text": "", "source": "", "metadata": {}}
        elif target_type == "ChatMessage":
            return {"role": "", "content": "", "timestamp": "", "metadata": {}}
        elif target_type == "SearchResult":
            return {"query": "", "results": [], "total_found": 0, "source": "", "metadata": {}}
        return {}
    
    def _to_text_content(self, data: Any) -> Optional[Dict]:
        """Generic transform to TextContent."""
        if isinstance(data, str):
            return {"text": data, "source": "string", "metadata": {}}
        elif isinstance(data, dict):
            # Try common text field names
            text = (
                data.get("text") or 
                data.get("content") or 
                data.get("message") or 
                data.get("body", "")
            )
            if text:
                return {
                    "text": text,
                    "source": "unknown",
                    "metadata": {k: v for k, v in data.items() if k != "text"}
                }
        return None
    
    def _to_chat_message(self, data: Any) -> Optional[Dict]:
        """Generic transform to ChatMessage."""
        import time
        if isinstance(data, str):
            return {
                "role": "assistant",
                "content": data,
                "timestamp": str(int(time.time())),
                "metadata": {}
            }
        elif isinstance(data, dict):
            content = data.get("text") or data.get("content") or str(data)
            return {
                "role": data.get("role", "assistant"),
                "content": content,
                "timestamp": data.get("timestamp", str(int(time.time()))),
                "metadata": {k: v for k, v in data.items() if k not in ["text", "content", "role", "timestamp"]}
            }
        return None
    
    def _to_search_result(self, data: Any) -> Optional[Dict]:
        """Generic transform to SearchResult."""
        if isinstance(data, dict):
            if "results" in data or "items" in data:
                return {
                    "query": data.get("query", ""),
                    "results": data.get("results") or data.get("items", []),
                    "total_found": data.get("total_found", data.get("total", 0)),
                    "source": data.get("source", "unknown"),
                    "metadata": {}
                }
        return None
    
    def list_transformers(self) -> Dict[str, List[str]]:
        """List all available transformers."""
        return {
            block: list(types.keys()) 
            for block, types in self._transformers.items()
        }
    
    @staticmethod
    def _apply_field_mapping(data: Dict, mapping: Dict[str, str]) -> Dict:
        """Apply field mapping to data (used by orchestrator)."""
        if not mapping or not isinstance(data, dict):
            return data
        result = dict(data)
        for target_field, source_field in mapping.items():
            if source_field in data:
                result[target_field] = data[source_field]
        return result


# Global transformer instance
transformer = DataTransformer()


def get_transformer() -> DataTransformer:
    """Get the global DataTransformer instance."""
    return transformer


def transform(data: Any, target_type: str, source_block: str = None, field_mapping: dict = None) -> tuple:
    """
    Convenience function to transform data to target type.
    
    Returns tuple of (transformed_data, operation_name).
    """
    return transformer.transform(data, target_type, source_block, field_mapping)


# ========================================================================
# STATIC METHODS FOR ORCHESTRATOR COMPATIBILITY
# ========================================================================

def detect_type(data: Any) -> str:
    """Detect the type of data."""
    if data is None:
        return DataTransformer.UNKNOWN
    if isinstance(data, str):
        return DataTransformer.TEXT
    if isinstance(data, dict):
        # Check for block-specific fields
        if "pages" in data and "filename" in data:
            return DataTransformer.PDF
        if "confidence" in data and "word_count" in data:
            return DataTransformer.TEXT  # OCR output
        if "model" in data and "provider" in data:
            return DataTransformer.TEXT  # Chat output
        if "file_path" in data or "path" in data:
            return DataTransformer.FILE
        if "image_data" in data or "image_url" in data:
            return DataTransformer.IMAGE
        if "measurements" in data or "quantities" in data:
            return "ConstructionAnalysis"
        if "text" in data:
            return DataTransformer.TEXT
    return DataTransformer.JSON


def are_compatible(source_type: str, target_type: str) -> bool:
    """Check if two types are compatible for transformation."""
    if source_type == target_type:
        return True
    # Case-insensitive comparison for basic types (text/Text, pdf/PDF, etc.)
    if isinstance(source_type, str) and isinstance(target_type, str):
        if source_type.lower() == target_type.lower():
            return True
    if source_type == DataTransformer.UNKNOWN or target_type == DataTransformer.UNKNOWN:
        return True
    # TextContent can transform to/from various types
    if target_type == "TextContent" and source_type in [DataTransformer.TEXT, DataTransformer.PDF, "ConstructionAnalysis"]:
        return True
    if source_type == "TextContent" and target_type in [DataTransformer.TEXT, DataTransformer.JSON]:
        return True
    # File path dicts can transform to PDFContent or ImageContent
    if target_type == "PDFContent" and source_type == DataTransformer.FILE:
        return True
    if target_type == "ImageContent" and source_type == DataTransformer.FILE:
        return True
    # PDF can be processed by OCR (many OCR engines handle PDFs)
    if target_type.lower() == "image" and source_type.lower() == "pdf":
        return True
    # ConstructionAnalysis can transform to TextContent or Text
    if target_type in ["TextContent", DataTransformer.TEXT] and source_type == "ConstructionAnalysis":
        return True
    return False


# Attach static methods to class for Orchestrator compatibility
DataTransformer.detect_type = staticmethod(detect_type)
DataTransformer.are_compatible = staticmethod(are_compatible)


# Export
__all__ = [
    "DataTransformer",
    "transformer",
    "get_transformer",
    "transform",
]
