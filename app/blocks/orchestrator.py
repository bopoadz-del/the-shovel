"""Orchestrator Block - The Chain Master with Type Validation"""

from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
from app.core.universal_base import UniversalBlock
from app.core.data_transformer import DataTransformer, transform as _transform_data
from app.core.input_adapter import adapt_input
import logging

logger = logging.getLogger(__name__)


# Input types (case-insensitive) that are satisfied by a plain text string.
_TEXT_LIKE_INPUT_TYPES = {"text", "textcontent", "chatmessage"}

# Field names — priority order — used to pull the human-readable text out of a
# block's JSON output when the next block in a chain expects plain text.
# Ordered most-specific first: a translate block's text is under "translated",
# an LLM block's under "answer"/"text", etc. Extend this list as blocks with
# new output shapes are added to the platform.
_TEXT_OUTPUT_FIELDS = (
    "text", "translated", "translated_text", "translation",
    "answer", "response", "content", "message",
    "result", "output", "summary", "body",
)


def _step_error_text(result: Any) -> str:
    """Best-effort error string from a step result (may be nested)."""
    if isinstance(result, dict):
        for key in ("error", "message"):
            if result.get(key):
                return str(result[key])
        inner = result.get("result")
        if isinstance(inner, dict):
            return _step_error_text(inner)
    return "unknown error"


def _coerce_dict_to_text(data: Dict, source_block: Optional[Any] = None) -> Optional[str]:
    """Extract the primary human-readable string from a block's JSON output.

    Resolution order:

    1. If ``source_block`` declares ``text_output_field`` (a class attribute
       set by the producing block — see ``UniversalBlock``), use that key
       first. This is the per-block contract: a block whose canonical text
       lives under a non-standard key opts in by declaring the field name.

    2. Otherwise, scan the global priority-ordered ``_TEXT_OUTPUT_FIELDS``.
       Works for most existing blocks because their canonical key is
       already in the list (``text``, ``response``, ``answer``, ...).

    3. As a last resort, if exactly one string value exists in the dict
       it's unambiguous; return it.

    Returns ``None`` when no obvious text field is present, so the caller
    can fall back to its normal type handling rather than guess wrongly.
    """
    # 1. Per-block override — the producing block declared its canonical key
    if source_block is not None:
        override_key = getattr(source_block, "text_output_field", None)
        if override_key:
            value = data.get(override_key)
            if isinstance(value, str) and value.strip():
                return value

    # 2. Global fallback list
    for key in _TEXT_OUTPUT_FIELDS:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value

    # 3. Single-string-value heuristic
    string_values = [v for v in data.values() if isinstance(v, str) and v.strip()]
    if len(string_values) == 1:
        return string_values[0]
    return None


@dataclass
class ChainStep:
    """Validated chain step with type information"""
    index: int
    block_name: str
    block: Any
    input_type: str
    output_type: str
    params: Dict
    input_mapping: Optional[Dict[str, str]] = None
    
    def to_dict(self) -> Dict:
        return {
            "index": self.index,
            "block": self.block_name,
            "input_type": self.input_type,
            "output_type": self.output_type,
            "params": self.params,
            "input_mapping": self.input_mapping
        }


@dataclass
class TypeValidationError:
    """Type mismatch between steps"""
    step_index: int
    from_block: str
    to_block: str
    expected_type: str
    actual_type: str
    message: str
    
    def to_dict(self) -> Dict:
        return {
            "step": self.step_index,
            "from": self.from_block,
            "to": self.to_block,
            "expected": self.expected_type,
            "actual": self.actual_type,
            "message": self.message
        }


class OrchestratorBlock(UniversalBlock):
    """Execute chains of blocks with automatic type validation and conversion.
    
    Key features:
    1. Type validation between steps - checks if step N-1 output matches step N input
    2. DataTransformer converts compatible but different formats automatically
    3. Fail-fast with clear error messages showing exactly what's wrong
    4. Execution graph built upfront - validates entire chain before any execution
    5. Standardized context passing with input_mapping support
    """

    name = "orchestrator"
    version = "2.0.0"
    description = "Chain execution engine with type validation"
    layer = 2
    tags = ["ai", "core", "orchestrator", "chain", "typed"]
    requires = ["memory"]

    default_config = {
        "max_steps": 50,
        "persist_steps": True,
        "fail_fast": True,
        "auto_convert": True
    }

    ui_schema = {
        "input": {
            "type": "json",
            "accept": None,
            "placeholder": '{"steps": [{"block": "chat", "params": {}}]}',
            "multiline": True
        },
        "output": {
            "type": "json",
            "fields": [
                {"name": "steps_executed", "type": "number", "label": "Steps"},
                {"name": "final_output", "type": "json", "label": "Output"},
                {"name": "type_conversions", "type": "json", "label": "Conversions"}
            ]
        },
        "quick_actions": [
            {"icon": "⛓️", "label": "Run Chain", "prompt": '{"steps":[{"block":"chat","params":{}}],"initial_input":"Hello"}'},
            {"icon": "📊", "label": "Chain Status", "prompt": "Get status of current chain execution"}
        ]
    }

    def __init__(self, hal_block=None, config=None):
        super().__init__(hal_block, config)
        self._registry = {}
        self._instance_cache = {}
        self._create_block_fn = None
        self._memory_fn = None
        self._progress_callback: Optional[Callable] = None

    def set_platform(self, registry, instance_cache, create_block_fn, memory_fn=None):
        """Wire platform services from main.py"""
        self._registry = registry
        self._instance_cache = instance_cache
        self._create_block_fn = create_block_fn
        self._memory_fn = memory_fn

    def set_progress_callback(self, callback: Callable):
        """Set callback for progress updates"""
        self._progress_callback = callback

    def _report_progress(self, step_index: int, step_label: str, status: str, 
                         details: Dict = None):
        """Report step progress"""
        if self._progress_callback:
            self._progress_callback({
                "step": step_index,
                "label": step_label,
                "status": status,
                "details": details or {}
            })

    def _get_block_type_info(self, block: Any) -> tuple[str, str]:
        """Get input and output types from a block"""
        # Prefer explicit type declarations from TypedBlock
        if hasattr(block, 'accepted_input_types') and block.accepted_input_types:
            input_type = block.accepted_input_types[0]
        elif hasattr(block, 'input_schema') and block.input_schema and hasattr(block.input_schema, 'content_type'):
            input_type = block.input_schema.content_type.value
        elif hasattr(block, 'get_input_type'):
            input_type = block.get_input_type()
        else:
            # Infer from ui_schema or default
            input_type = self._infer_input_type(block)
        
        if hasattr(block, 'produced_output_types') and block.produced_output_types:
            output_type = block.produced_output_types[0]
        elif hasattr(block, 'output_schema') and block.output_schema and hasattr(block.output_schema, 'content_type'):
            output_type = block.output_schema.content_type.value
        elif hasattr(block, 'get_output_type'):
            output_type = block.get_output_type()
        else:
            output_type = self._infer_output_type(block)
        
        return input_type, output_type
    
    def _infer_input_type(self, block: Any) -> str:
        """Infer input type from block ui_schema"""
        ui = getattr(block, 'ui_schema', {})
        input_spec = ui.get('input', {})
        input_type = input_spec.get('type', 'text')
        
        type_mapping = {
            'file': DataTransformer.FILE,
            'text': DataTransformer.TEXT,
            'json': DataTransformer.JSON,
            'image': DataTransformer.IMAGE,
            'pdf': DataTransformer.PDF,
        }
        return type_mapping.get(input_type, DataTransformer.UNKNOWN)
    
    def _infer_output_type(self, block: Any) -> str:
        """Infer output type from block ui_schema"""
        ui = getattr(block, 'ui_schema', {})
        output_spec = ui.get('output', {})
        output_type = output_spec.get('type', 'json')
        
        type_mapping = {
            'text': DataTransformer.TEXT,
            'json': DataTransformer.JSON,
            'table': DataTransformer.TABLE,
            'image': DataTransformer.IMAGE,
        }
        return type_mapping.get(output_type, DataTransformer.UNKNOWN)

    async def _resolve_block(self, block_name: str):
        """Get or create block instance."""
        if block_name in self._instance_cache:
            return self._instance_cache[block_name]
        if block_name in self._registry and self._create_block_fn:
            instance = self._create_block_fn(self._registry[block_name])
            self._instance_cache[block_name] = instance
            return instance
        return None

    async def build_execution_graph(self, steps: List[Dict], initial_input: Any = None) -> tuple[List[ChainStep], List[TypeValidationError]]:
        """Build execution plan and validate entire chain upfront.
        
        Returns:
            (validated_steps, validation_errors)
            If errors exist, chain should not be executed.
        """
        validated_steps = []
        validation_errors = []
        prev_output_type = DataTransformer.detect_type(initial_input) if initial_input else DataTransformer.UNKNOWN
        
        for i, step_config in enumerate(steps):
            block_name = step_config.get("block")
            step_params = dict(step_config.get("params", {}) or {})
            # F38: a per-step ``input`` dict (the natural /v1/execute shape)
            # was silently DROPPED -- boq_processor ran with no file_path
            # while the caller had supplied one. Fold it into the step's
            # params, where merged_params delivers it to the block.
            step_input = step_config.get("input")
            if isinstance(step_input, dict):
                step_params = {**step_input, **step_params}
            input_mapping = step_config.get("input_mapping")
            
            if not block_name:
                validation_errors.append(TypeValidationError(
                    step_index=i,
                    from_block=steps[i-1].get("block", "initial") if i > 0 else "initial",
                    to_block="unknown",
                    expected_type="any",
                    actual_type="none",
                    message=f"Step {i}: Missing block name"
                ))
                continue
            
            # Resolve block
            block = await self._resolve_block(block_name)
            if not block:
                validation_errors.append(TypeValidationError(
                    step_index=i,
                    from_block=steps[i-1].get("block", "initial") if i > 0 else "initial",
                    to_block=block_name,
                    expected_type="any",
                    actual_type="none",
                    message=f"Step {i}: Block '{block_name}' not found"
                ))
                continue
            
            # Skip containers
            if block_name.startswith("container_"):
                validation_errors.append(TypeValidationError(
                    step_index=i,
                    from_block=steps[i-1].get("block", "initial") if i > 0 else "initial",
                    to_block=block_name,
                    expected_type="any",
                    actual_type="container",
                    message=f"Step {i}: Container '{block_name}' cannot be executed in a chain"
                ))
                continue
            
            # Get type info
            input_type, output_type = self._get_block_type_info(block)
            
            # Validate type compatibility with previous step
            if i > 0 and prev_output_type != DataTransformer.UNKNOWN:
                compatible = DataTransformer.are_compatible(prev_output_type, input_type)
                # Fallback: case-insensitive + known cross-type compatibilities
                if not compatible:
                    s = (prev_output_type or "").lower()
                    t = (input_type or "").lower()
                    if s == t:
                        compatible = True
                    elif s == "pdf" and t == "image":
                        compatible = True
                    elif s == "file" and t in ["pdf", "image", "pdfcontent", "imagecontent", "filecontent"]:
                        compatible = True
                if not compatible:
                    prev_block = steps[i-1].get("block", "initial")
                    validation_errors.append(TypeValidationError(
                        step_index=i,
                        from_block=prev_block,
                        to_block=block_name,
                        expected_type=input_type,
                        actual_type=prev_output_type,
                        message=f"Step {i} ({block_name}) expects {input_type} but step {i-1} ({prev_block}) produces {prev_output_type}"
                    ))
            
            # Create validated step
            chain_step = ChainStep(
                index=i,
                block_name=block_name,
                block=block,
                input_type=input_type,
                output_type=output_type,
                params=step_params,
                input_mapping=input_mapping
            )
            validated_steps.append(chain_step)
            
            # Update for next iteration
            prev_output_type = output_type
        
        return validated_steps, validation_errors

    async def process(self, input_data: Any, params: Dict = None) -> Dict:
        """Execute chain with type validation and conversion."""
        params = params or {}
        steps = params.get("steps", [])
        
        if not steps and isinstance(input_data, dict):
            steps = input_data.get("steps", [])
            # F38: without an explicit initial_input the whole
            # {"steps": [...]} dict used to fall through as the chain's
            # input and leak into every step's merged params.
            input_data = input_data.get("initial_input")

        if not steps:
            return {"status": "error", "error": "No steps provided for chain execution"}

        from app.core.privileges import raise_if_privileged_steps

        raise_if_privileged_steps(steps)

        max_steps = params.get("max_steps", self.config.get("max_steps", 50))
        if len(steps) > max_steps:
            return {"status": "error", "error": f"Chain exceeds max_steps ({max_steps})"}

        # Build execution graph and validate upfront
        self._report_progress(-1, "validation", "running", {"message": "Validating chain types..."})
        validated_steps, validation_errors = await self.build_execution_graph(steps, input_data)
        
        if validation_errors and self.config.get("fail_fast", True):
            error_messages = [e.message for e in validation_errors]
            return {
                "status": "error",
                "error": "Chain validation failed",
                "validation_errors": [e.to_dict() for e in validation_errors],
                "details": "Fix the type mismatches above before executing the chain"
            }

        self._report_progress(-1, "validation", "done", {"steps_validated": len(validated_steps)})

        # Execute chain
        context = input_data
        results = []
        type_conversions = []
        # Track the previously-executed step's block for chain output
        # unwrapping (see the _coerce_dict_to_text call below). Indexing
        # `validated_steps[step.index - 1]` would be wrong when
        # validation drops a step (fail_fast=False) and `step.index`
        # diverges from the position in `validated_steps`. Codex P2 fix
        # on PR #32. A local variable is also clearer at the call site.
        prev_block: Optional[Any] = None

        for step in validated_steps:
            self._report_progress(step.index, step.block_name, "running")

            # Transform input if needed (type conversion + field mapping)
            current_type = DataTransformer.detect_type(context)

            # Output-unwrapping: a JSON dict produced by a previous step is
            # coerced to its primary text value when the next block expects
            # text (e.g. translate -> chat). Without this the chain is
            # rejected as a JSON/Text mismatch even though the dict carries
            # exactly the text the next block needs.
            if (
                step.index > 0
                and current_type == DataTransformer.JSON
                and isinstance(context, dict)
                and str(step.input_type).lower() in _TEXT_LIKE_INPUT_TYPES
            ):
                # `prev_block` (the block whose output produced `context`)
                # is tracked across loop iterations rather than looked up
                # via `validated_steps[step.index - 1]` — `step.index`
                # is the ORIGINAL input step number, which can diverge
                # from `validated_steps` position when a step was dropped
                # during validation (fail_fast=False). Codex P2 fix on PR #32.
                unwrapped = _coerce_dict_to_text(context, source_block=prev_block)
                if unwrapped is not None:
                    type_conversions.append({
                        "step": step.index,
                        "block": step.block_name,
                        "from": current_type,
                        "to": step.input_type,
                        "operation": "json_to_text_unwrap",
                    })
                    context = unwrapped
                    current_type = DataTransformer.TEXT

            # Auto-adapt input to block's expected format
            context = adapt_input(context, step.block)
            
            # Skip runtime type check for step 0 (initial input is caller-provided)
            if step.index > 0 and current_type != step.input_type and current_type != DataTransformer.UNKNOWN:
                compatible = DataTransformer.are_compatible(current_type, step.input_type)
                # Fallback: case-insensitive + known cross-type compatibilities
                if not compatible:
                    s = (current_type or "").lower()
                    t = (step.input_type or "").lower()
                    if s == t:
                        compatible = True
                    elif s == "pdf" and t == "image":
                        compatible = True
                    elif s == "file" and t in ["pdf", "image", "pdfcontent", "imagecontent", "filecontent"]:
                        compatible = True
                    elif s == "json" and t in ["file", "pdf", "image"]:
                        # URL/file_path/path dicts are file inputs, not generic JSON
                        if isinstance(context, dict) and any(k in context for k in ("url", "file_path", "path")):
                            compatible = True
                # Special case: file path dicts are valid input for PDF/image/file blocks
                if not compatible and current_type == DataTransformer.FILE:
                    if step.input_type in ["PDFContent", "ImageContent", "FileContent"]:
                        compatible = True
                if compatible:
                    context, conversion_op = _transform_data(
                        context, 
                        step.input_type,
                        field_mapping=step.input_mapping
                    )
                    if conversion_op != "no_change":
                        type_conversions.append({
                            "step": step.index,
                            "block": step.block_name,
                            "from": current_type,
                            "to": step.input_type,
                            "operation": conversion_op
                        })
                else:
                    # Type mismatch - fail or warn
                    if self.config.get("fail_fast", True):
                        return {
                            "status": "error",
                            "step": step.index,
                            "block": step.block_name,
                            "error": f"Type mismatch: {step.block_name} expects {step.input_type} but received {current_type}",
                            "partial_results": results
                        }
            
            # Apply field mapping if no type conversion happened
            if step.input_mapping and isinstance(context, dict):
                context = DataTransformer._apply_field_mapping(context, step.input_mapping)

            # Normalize parameter routing: merge step input into params for blocks that expect params
            merged_params = {**(step.params or {}), **(context if isinstance(context, dict) else {})}
            result = await step.block.execute(context, merged_params)

            results.append({
                "step": step.index,
                "block": step.block_name,
                "success": result.get("status") != "error",
                "result": result
            })

            # Extract typed output for next step
            context = self._extract_output(step.block, result)
            
            self._report_progress(step.index, step.block_name, "done", {
                "success": result.get("status") != "error"
            })

            # Persist to memory
            if self.config.get("persist_steps", True):
                await self._persist_step(step.index, step.block_name, context)

            # Stop on error unless continue_on_error is set
            if result.get("status") == "error" and not params.get("continue_on_error"):
                break

            # The next iteration's chain-output unwrap consults this
            # block's `text_output_field`; remember it for them.
            prev_block = step.block

        # F38: the chain used to return status success UNCONDITIONALLY --
        # a failed step (success: false, loop broken) came back wrapped in
        # a green envelope with the error dict as final_output. The chain's
        # status must reflect its steps.
        failed = [r for r in results if not r["success"]]
        out = {
            "status": "success",
            "steps_executed": len(results),
            "final_output": context,
            "results": results,
            "type_conversions": type_conversions,
            "validation_passed": len(validation_errors) == 0
        }
        if failed:
            first = failed[0]
            out["status"] = (
                "completed_with_errors" if params.get("continue_on_error") else "error"
            )
            out["error"] = (
                f"step {first['step']} ({first['block']}) failed: "
                f"{_step_error_text(first['result'])}"
            )
            out["failed_steps"] = [f["step"] for f in failed]
        return out

    def _extract_output(self, block: Any, result: Dict) -> Any:
        """Extract typed output from execution result."""
        # If block has extract_output method (TypedBlock)
        if hasattr(block, 'extract_output'):
            return block.extract_output(result)
        
        # Otherwise use smart extraction
        if not isinstance(result, dict):
            return result
        
        # Get inner result
        inner = result.get("result", result)
        
        # Remove metadata keys
        if isinstance(inner, dict):
            metadata_keys = {'block', 'request_id', 'status', 'confidence', 
                           'source_id', 'metadata', 'processing_time_ms'}
            cleaned = {k: v for k, v in inner.items() if k not in metadata_keys}
            return cleaned if cleaned else inner
        
        return inner

    async def _persist_step(self, step_index: int, block_name: str, context: Any):
        """Persist chain step to memory."""
        memory = self.get_dep("memory")
        if memory:
            try:
                await memory.execute({
                    "action": "set",
                    "key": f"chain:{step_index}:{block_name}",
                    "value": context,
                    "ttl": 3600
                })
            except Exception:
                logger.debug(
                    "swallowed %s in _persist_step() — continuing",
                    "Exception", exc_info=True,
                )
        elif self._memory_fn:
            try:
                mem = self._memory_fn()
                await mem.execute({
                    "action": "set",
                    "key": f"chain:{step_index}:{block_name}",
                    "value": context,
                    "ttl": 3600
                })
            except Exception:
                logger.debug(
                    "swallowed %s in _persist_step() — continuing",
                    "Exception", exc_info=True,
                )
