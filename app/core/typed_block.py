"""
TypedBlock - Base class with schema validation for block data types.

This module provides TypedBlock which extends UniversalBlock with:
- Input/output schema declarations
- Runtime type validation
- Backward compatibility with existing blocks
- Schema and ContentType classes for declarative type definitions
"""

from abc import ABC
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import uuid

from .universal_base import UniversalBlock


class ContentType(Enum):
    """Standard content types for blocks."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    PDF = "pdf"
    JSON = "json"
    FILE = "file"
    STREAM = "stream"
    CHAT = "chat"
    EMBEDDING = "embedding"
    UNKNOWN = "unknown"


@dataclass
class Schema:
    """
    Declarative schema definition for block I/O.
    
    Example:
        input_schema = Schema(
            content_type=ContentType.TEXT,
            required_fields=["text"],
            optional_fields=["metadata"]
        )
    """
    content_type: ContentType = ContentType.UNKNOWN
    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)
    format_hints: Dict[str, Any] = field(default_factory=dict)
    json_schema: Optional[Dict] = None  # JSON Schema format alternative
    
    def to_json_schema(self) -> Dict:
        """Convert to JSON Schema format."""
        if self.json_schema:
            return self.json_schema
        
        properties = {}
        for field in self.required_fields + self.optional_fields:
            properties[field] = {"type": "string"}  # Default to string
        
        return {
            "type": "object",
            "properties": properties,
            "required": self.required_fields
        }
    
    def validate_data(self, data: Any) -> Dict[str, Any]:
        """Validate data against this schema."""
        errors = []
        
        # Handle string input for TEXT type
        if self.content_type == ContentType.TEXT and isinstance(data, str):
            return {"valid": True, "errors": [], "data": data}
        
        if not isinstance(data, dict):
            return {
                "valid": False,
                "errors": [f"Expected dict, got {type(data).__name__}"],
                "data": data
            }
        
        # Check required fields
        for req_field in self.required_fields:
            if req_field not in data:
                errors.append(f"Missing required field: '{req_field}'")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "data": data
        }



class TypedBlock(UniversalBlock):
    """
    TypedBlock - Base class with schema validation for data types.
    
    Extends UniversalBlock to add:
    - input_schema: Dict or Schema declaring what data types this block accepts
    - output_schema: Dict or Schema declaring what data types this block produces
    - validate_input(data): Runtime type checking
    - validate_output(data): Ensure block returns what it promised
    
    All existing UniversalBlock functionality is preserved for backward compatibility.
    """
    
    # Schema declarations - subclasses define their data contracts
    # Can be either Dict (JSON Schema) or Schema object
    input_schema: Union[Dict, Schema, None] = None
    """Schema declaring what data types this block accepts.
    
    Examples:
        # Dict format (JSON Schema):
        input_schema = {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"]
        }
        
        # Schema object format:
        input_schema = Schema(
            content_type=ContentType.TEXT,
            required_fields=["text"]
        )
    """
    
    output_schema: Union[Dict, Schema, None] = None
    """Schema declaring what data types this block produces."""
    
    # Optional: declare which schema types this block can transform from/to
    accepted_input_types: List[str] = []
    """List of type names this block can accept (e.g., ["TextContent", "PDFContent"])."""
    
    produced_output_types: List[str] = []
    """List of type names this block produces (e.g., ["TextContent"])."""
    
    def __init__(self, hal_block=None, config: Dict = None):
        """Initialize TypedBlock with HAL and config."""
        super().__init__(hal_block=hal_block, config=config)
        
        # Validation stats
        self.validation_errors = []
        self.validation_warnings = []
    
    def validate_input(self, data: Any) -> Dict[str, Any]:
        """
        Validate input data against input_schema.
        
        Args:
            data: The input data to validate
            
        Returns:
            Dict with validation results:
            {
                "valid": bool,
                "errors": List[str],
                "warnings": List[str],
                "data": Any  # Potentially transformed/normalized data
            }
        """
        # If no input_schema defined, accept anything (backward compatibility)
        if not self.input_schema:
            return {
                "valid": True,
                "errors": [],
                "warnings": [],
                "data": data
            }
        
        # Handle Schema object format
        if isinstance(self.input_schema, Schema):
            return self.input_schema.validate_data(data)
        
        # Handle Dict format (JSON Schema)
        errors = []
        warnings = []
        
        schema_type = self.input_schema.get("type", "any")
        
        if schema_type == "object":
            result = self._validate_object(data, self.input_schema)
            errors.extend(result.get("errors", []))
            warnings.extend(result.get("warnings", []))
            data = result.get("data", data)
        elif schema_type == "array":
            result = self._validate_array(data, self.input_schema)
            errors.extend(result.get("errors", []))
            warnings.extend(result.get("warnings", []))
        elif schema_type == "string":
            if not isinstance(data, str):
                errors.append(f"Expected string, got {type(data).__name__}")
        elif schema_type == "number":
            if not isinstance(data, (int, float)):
                errors.append(f"Expected number, got {type(data).__name__}")
        
        self.validation_errors = errors
        self.validation_warnings = warnings
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "data": data
        }
    
    def validate_output(self, data: Dict) -> Dict[str, Any]:
        """
        Validate output data against output_schema.
        
        Args:
            data: The output data to validate (typically the result dict)
            
        Returns:
            Dict with validation results:
            {
                "valid": bool,
                "errors": List[str],
                "warnings": List[str],
                "data": Dict  # The validated data
            }
        """
        # If no output_schema defined, accept anything (backward compatibility)
        if not self.output_schema:
            return {
                "valid": True,
                "errors": [],
                "warnings": [],
                "data": data
            }
        
        # Handle Schema object format
        if isinstance(self.output_schema, Schema):
            # For Schema format, extract result data
            result_data = data.get("result", data)
            return self.output_schema.validate_data(result_data)
        
        # Handle Dict format (JSON Schema)
        errors = []
        warnings = []
        
        # Handle nested result structure (UniversalBlock wraps in result)
        result_data = data.get("result", data)
        
        schema_type = self.output_schema.get("type", "any")
        
        if schema_type == "object":
            result = self._validate_object(result_data, self.output_schema)
            errors.extend(result.get("errors", []))
            warnings.extend(result.get("warnings", []))
        elif schema_type == "array":
            result = self._validate_array(result_data, self.output_schema)
            errors.extend(result.get("errors", []))
            warnings.extend(result.get("warnings", []))
        
        # Check required fields exist
        required = self.output_schema.get("required", [])
        if isinstance(result_data, dict):
            for field in required:
                if field not in result_data:
                    errors.append(f"Missing required output field: '{field}'")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "data": data
        }
    
    def _validate_object(self, data: Any, schema: Dict) -> Dict[str, Any]:
        """Validate an object schema."""
        errors = []
        warnings = []
        
        if not isinstance(data, dict):
            return {
                "valid": False,
                "errors": [f"Expected object, got {type(data).__name__}"],
                "warnings": [],
                "data": data
            }
        
        properties = schema.get("properties", {})
        
        # Validate each property
        for prop_name, prop_schema in properties.items():
            if prop_name in data:
                prop_value = data[prop_name]
                prop_type = prop_schema.get("type")
                
                if prop_type == "string" and not isinstance(prop_value, str):
                    errors.append(f"Field '{prop_name}' should be string, got {type(prop_value).__name__}")
                elif prop_type == "number" and not isinstance(prop_value, (int, float)):
                    errors.append(f"Field '{prop_name}' should be number, got {type(prop_value).__name__}")
                elif prop_type == "boolean" and not isinstance(prop_value, bool):
                    errors.append(f"Field '{prop_name}' should be boolean, got {type(prop_value).__name__}")
                elif prop_type == "array" and not isinstance(prop_value, list):
                    errors.append(f"Field '{prop_name}' should be array, got {type(prop_value).__name__}")
                elif prop_type == "object" and not isinstance(prop_value, dict):
                    errors.append(f"Field '{prop_name}' should be object, got {type(prop_value).__name__}")
        
        # Check required fields
        required = schema.get("required", [])
        for field in required:
            if field not in data:
                errors.append(f"Missing required field: '{field}'")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "data": data
        }
    
    def _validate_array(self, data: Any, schema: Dict) -> Dict[str, Any]:
        """Validate an array schema."""
        errors = []
        warnings = []
        
        if not isinstance(data, list):
            return {
                "valid": False,
                "errors": [f"Expected array, got {type(data).__name__}"],
                "warnings": [],
                "data": data
            }
        
        # Validate items if item schema provided
        items_schema = schema.get("items")
        if items_schema:
            for i, item in enumerate(data):
                item_type = items_schema.get("type")
                if item_type == "string" and not isinstance(item, str):
                    errors.append(f"Item [{i}] should be string, got {type(item).__name__}")
                elif item_type == "number" and not isinstance(item, (int, float)):
                    errors.append(f"Item [{i}] should be number, got {type(item).__name__}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "data": data
        }
    
    async def execute(self, input_data: Any, params: Dict = None) -> Dict:
        """
        Execute with input validation (via UniversalBlock.execute) and
        optional output schema warnings.
        """
        result = await super().execute(input_data, params)

        if self.output_schema:
            output_validation = self.validate_output(result)
            if not output_validation["valid"]:
                if "metadata" not in result:
                    result["metadata"] = {}
                result["metadata"]["output_validation_warnings"] = output_validation["errors"]

        return result
    
    def get_schema_info(self) -> Dict[str, Any]:
        """Get schema information for this block."""
        return {
            "name": self.name,
            "version": self.version,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "accepted_input_types": self.accepted_input_types,
            "produced_output_types": self.produced_output_types
        }
    
    def can_accept(self, data_type: str) -> bool:
        """Check if this block can accept a specific data type."""
        return data_type in self.accepted_input_types or not self.accepted_input_types
    
    def can_produce(self, data_type: str) -> bool:
        """Check if this block can produce a specific data type."""
        return data_type in self.produced_output_types or not self.produced_output_types
