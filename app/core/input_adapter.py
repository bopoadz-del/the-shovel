"""
InputAdapter - Translate caller inputs to block-expected formats.

Problem: Callers send strings, None, or simple dicts.
Blocks expect typed dicts with specific fields.
Solution: Auto-wrap inputs based on block input_schema.
"""

from typing import Any, Dict, Optional, Union


class InputAdapter:
    """
    Adapts raw caller inputs to what a block's input_schema expects.
    
    Examples:
        - String "hello" → chat block → {"text": "hello"}
        - String "/path/to.pdf" → pdf_v2 → {"file_path": "/path/to.pdf"}
        - None → chat block → {"text": ""}
        - Dict {"message": "hi"} → chat block → {"text": "hi"}
    """
    
    @classmethod
    def adapt(cls, input_data: Any, block: Any) -> Any:
        """
        Adapt input data to what a block expects.
        
        Args:
            input_data: Raw input from caller (str, dict, None, etc.)
            block: Block instance with input_schema
            
        Returns:
            Adapted input that passes schema validation
        """
        # If input is None, return minimal dict based on schema
        if input_data is None:
            return cls._empty_for_schema(block)
        
        # If input is already a dict, try to normalize field names
        if isinstance(input_data, dict):
            return cls._normalize_dict(input_data, block)
        
        # If input is a string, wrap it appropriately
        if isinstance(input_data, str):
            return cls._wrap_string(input_data, block)
        
        # If input is a list, wrap it
        if isinstance(input_data, list):
            return {"items": input_data}
        
        # Numbers, booleans, etc. — wrap as generic input
        return {"data": input_data}
    
    @classmethod
    def _empty_for_schema(cls, block: Any) -> Dict:
        """Create minimal valid input for a block when caller sends None."""
        schema = cls._get_schema(block)
        
        if not schema:
            return {}
        
        # Check for schema type hints
        if hasattr(block, 'accepted_input_types') and block.accepted_input_types:
            # If block accepts text, return empty text structure
            if 'TextContent' in block.accepted_input_types or 'text' in block.accepted_input_types:
                return {"text": "", "source": "", "metadata": {}}
        
        # Check schema for required fields
        if isinstance(schema, dict):
            # JSON Schema format
            if schema.get("type") == "object":
                props = schema.get("properties", {})
                empty = {}
                for field_name, field_schema in props.items():
                    field_type = field_schema.get("type", "string")
                    if field_type == "string":
                        empty[field_name] = ""
                    elif field_type == "number":
                        empty[field_name] = 0
                    elif field_type == "boolean":
                        empty[field_name] = False
                    elif field_type == "array":
                        empty[field_name] = []
                    elif field_type == "object":
                        empty[field_name] = {}
                return empty
        
        # Schema object format (app.core.schema_registry.Schema)
        if hasattr(schema, 'content_type'):
            ct = schema.content_type.value if hasattr(schema.content_type, 'value') else str(schema.content_type)
            if ct == 'text':
                return {"text": "", "source": "", "metadata": {}}
            if ct == 'pdf':
                return {"file_path": "", "filename": "", "pages": 0, "text": "", "metadata": {}}
            if ct == 'image':
                return {"image_data": "", "format": "", "width": 0, "height": 0, "source": "", "metadata": {}}
        
        return {}
    
    @classmethod
    def _wrap_string(cls, input_str: str, block: Any) -> Dict:
        """Wrap a string into the dict a block expects."""
        schema = cls._get_schema(block)
        block_name = getattr(block, 'name', 'unknown')
        
        # URL — wrap with both url and text so any block can use whichever key it prefers
        if input_str.startswith("http://") or input_str.startswith("https://"):
            return {"url": input_str, "text": input_str, "input": input_str}

        # If it looks like a file path
        if input_str.startswith('/') or input_str.startswith('./') or '.' in input_str.split('/')[-1]:
            # PDF or file-related block
            if 'pdf' in block_name or 'file' in block_name or 'ocr' in block_name:
                return {"file_path": input_str}

        # Text/chat block
        if schema:
            if isinstance(schema, dict):
                props = schema.get("properties", {})
                # If schema has "text" field, use it
                if "text" in props:
                    return {"text": input_str}
                # If schema has "file_path" field, maybe it's a path
                if "file_path" in props:
                    return {"file_path": input_str}
                # If schema has "message" field
                if "message" in props:
                    return {"message": input_str}
                # First string field
                for key, val in props.items():
                    if val.get("type") == "string":
                        return {key: input_str}
            
            # Schema object
            if hasattr(schema, 'content_type'):
                ct = schema.content_type.value if hasattr(schema.content_type, 'value') else str(schema.content_type)
                if ct == 'text':
                    return {"text": input_str}
        
        # Default: wrap as text
        return {"text": input_str}
    
    @classmethod
    def _normalize_dict(cls, input_dict: Dict, block: Any) -> Dict:
        """Normalize dict field names to match block schema."""
        schema = cls._get_schema(block)
        
        if not schema:
            return input_dict
        
        # Get expected field names from schema
        expected_fields = set()
        if isinstance(schema, dict):
            expected_fields = set(schema.get("properties", {}).keys())
        elif hasattr(schema, 'required_fields'):
            expected_fields = set(schema.required_fields)
            if hasattr(schema, 'optional_fields'):
                expected_fields |= set(schema.optional_fields)
        
        # Common aliases
        aliases = {
            "message": ["text", "content", "input"],
            "text": ["message", "content", "input", "body"],
            "file_path": ["path", "url", "filename", "file"],
            "path": ["file_path", "url", "filename"],
            "content": ["text", "message", "body"],
        }
        
        normalized = dict(input_dict)
        
        # If input has field that matches expected, keep it
        # If input has alias but expected has different name, copy it
        for expected_field in expected_fields:
            if expected_field not in normalized:
                # Check aliases
                for alias in aliases.get(expected_field, []):
                    if alias in normalized:
                        normalized[expected_field] = normalized[alias]
                        break
        
        return normalized
    
    @classmethod
    def _get_schema(cls, block: Any) -> Optional[Any]:
        """Get input schema from block."""
        if hasattr(block, 'input_schema'):
            return block.input_schema
        if hasattr(block, 'ui_schema'):
            ui = block.ui_schema
            if isinstance(ui, dict):
                return ui.get("input", {})
        return None


# Convenience function
def adapt_input(input_data: Any, block: Any) -> Any:
    """Adapt input for a block."""
    return InputAdapter.adapt(input_data, block)


__all__ = ["InputAdapter", "adapt_input"]
