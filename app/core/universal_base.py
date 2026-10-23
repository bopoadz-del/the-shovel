"""
Universal Block Base Class - The ONE True Block Pattern

All blocks inherit from this. No more dual systems.
"""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List, Optional
import time
import uuid
import logging

logger = logging.getLogger(__name__)


class UniversalBlock(ABC):
    """
    Universal Block Base Class - Domain Adapter Protocol
    
    All blocks MUST define:
    - name: str - Block identifier
    - version: str - Semver
    - layer: int - Init order (0=infrastructure → 5=interface)
    - tags: List[str] - Categorization
    - requires: List[str] - Dependencies (other block names)
    """
    
    # REQUIRED - define in subclass
    name: str = ""
    version: str = "1.0.0"
    description: str = ""
    layer: int = 3  # Default: domain layer
    tags: List[str] = []
    requires: List[str] = []
    
    # Optional
    default_config: Dict = {}
    author: str = ""

    # Optional declarative input contract (JSON Schema dict or Schema object).
    # When set, execute() validates before process(). TypedBlock subclasses
    # use the same hook via validate_input().
    input_schema: Any = None

    # Lightweight opt-in for UniversalBlock subclasses: require these keys
    # on the adapted input dict (params are also checked).
    required_input_fields: ClassVar[List[str]] = []

    # Require at least one of these keys (after adapt_input merge).
    required_input_one_of: ClassVar[List[str]] = []

    # Skip required-field checks when action/operation matches.
    skip_input_validation_actions: ClassVar[List[str]] = []

    # Blocks that accept empty input by design (orchestrators, stubs).
    allow_empty_input: ClassVar[bool] = False

    # Whether the agent runtime should auto-validate this block's result
    # by running each numeric in the response through validation_pipeline.
    # Defaults True; blocks that produce text-only or vector output should
    # override to False to avoid noisy "skipped — no numeric value found"
    # validation entries.
    auto_validate: bool = True

    # Canonical text field for chain unwrapping.
    #
    # When this block produces a dict (e.g. ``{"status": "success",
    # "translated": "..."}``) and the next block in a chain expects plain
    # text, ``OrchestratorBlock._coerce_dict_to_text`` looks here FIRST
    # before falling back to its priority-ordered global list. Set this
    # in subclasses whose canonical text lives under a non-standard key
    # — e.g. ``text_output_field = "translated"`` on TranslateBlock.
    #
    # ``None`` (default) keeps the legacy behaviour: the global field
    # list is the only signal, which works for most blocks because they
    # already use ``text`` / ``response`` / ``answer`` / etc.
    #
    # See CONTRIBUTING.md ("Block output contracts") for the rule.
    text_output_field: Optional[str] = None
    
    # UI Schema - Auto-configures Universal UI Shell (frontend)
    # Blocks self-describe: what inputs they need, what outputs they produce
    ui_schema: Dict = {
        "input": {
            "type": "text",  # text, file, audio, image, pdf, json
            "accept": None,  # for files: [".pdf", ".jpg"]
            "placeholder": "Enter your request...",
            "multiline": False,
        },
        "output": {
            "type": "text",  # text, table, chart, json, pdf_viewer, image
            "fields": [],    # for table: [{"name": "concrete_m3", "type": "number", "unit": "m³"}]
        },
        "quick_actions": []  # [{"icon": "📄", "label": "Analyze PDF", "prompt": "..."}]
    }
    
    def __init__(self, hal_block=None, config: Dict = None):
        """Initialize with HAL and config"""
        self.hal = hal_block
        self.config = {**self.default_config, **(config or {})}
        
        # Execution stats
        self.execution_count = 0
        self.total_execution_time = 0
        
        # Wired dependencies (filled by assembler)
        self._dependencies: Dict[str, Any] = {}
    
    def wire(self, dep_name: str, dep_instance):
        """Wire a dependency (called by assembler)"""
        self._dependencies[dep_name] = dep_instance
    
    def get_dep(self, name: str) -> Optional[Any]:
        """Get a wired dependency"""
        return self._dependencies.get(name)

    def _prepare_block_input(self, input_data: Any, params: Dict) -> Any:
        """Merge params, then adapt to the block's expected input shape."""
        if input_data is None and params:
            input_data = params
        elif isinstance(input_data, dict) and isinstance(params, dict):
            input_data = {**params, **input_data}
        from app.core.input_adapter import adapt_input
        return adapt_input(input_data, self)

    def _input_validation_failure(
        self,
        errors: List[str],
        request_id: str,
        params: Dict,
        start: float,
    ) -> Dict:
        return {
            "block": self.name,
            "request_id": request_id,
            "status": "error",
            "result": {"error": "Input validation failed", "details": errors},
            "confidence": 0.0,
            "source_id": f"{self.name}-{request_id}",
            "metadata": {
                "version": self.version,
                "validation_errors": errors,
                **params,
            },
            "processing_time_ms": int((time.time() - start) * 1000),
        }

    def _validate_block_input(self, input_data: Any, params: Dict) -> Optional[Dict[str, Any]]:
        """Return a validation result dict when input_schema is set on TypedBlock."""
        schema = getattr(self, "input_schema", None)
        if not schema:
            return None
        if hasattr(self, "validate_input"):
            return self.validate_input(input_data)
        return None

    def _should_skip_input_validation(self, params: Dict, input_data: Any = None) -> bool:
        if getattr(self, "allow_empty_input", False):
            return True
        skipped = getattr(self, "skip_input_validation_actions", None) or []
        data = input_data if isinstance(input_data, dict) else {}
        action = (
            (params or {}).get("action")
            or (params or {}).get("operation")
            or data.get("action")
            or data.get("operation")
        )
        return action in skipped

    def _validate_required_fields(self, input_data: Any, params: Dict) -> List[str]:
        """Check required_input_fields and required_input_one_of."""
        if self._should_skip_input_validation(params, input_data):
            return []
        data = input_data if isinstance(input_data, dict) else {}
        params = params or {}
        errors: List[str] = []

        one_of = getattr(self, "required_input_one_of", None) or []
        if one_of:
            found = False
            for field in one_of:
                val = data.get(field)
                if val is None or (isinstance(val, str) and not val.strip()):
                    val = params.get(field)
                if val is not None and not (isinstance(val, str) and not val.strip()):
                    found = True
                    break
            if not found:
                errors.append(
                    f"At least one of {one_of} is required"
                )

        fields = getattr(self, "required_input_fields", None) or []
        for field in fields:
            val = data.get(field)
            if val is None or (isinstance(val, str) and not val.strip()):
                val = params.get(field)
            if val is None or (isinstance(val, str) and not val.strip()):
                errors.append(f"Missing required field: '{field}'")
        return errors
    
    @abstractmethod
    async def process(self, input_data: Any, params: Dict = None) -> Dict:
        """Main processing - implement in subclass"""
        pass
    
    async def execute(self, input_data: Any, params: Dict = None) -> Dict:
        """Execute with timing and error handling"""
        start = time.time()
        request_id = str(uuid.uuid4())[:12]
        params = params or {}

        input_data = self._prepare_block_input(input_data, params)

        schema_validation = self._validate_block_input(input_data, params)
        if schema_validation is not None and not schema_validation.get("valid", True):
            return self._input_validation_failure(
                schema_validation.get("errors") or ["Invalid input"],
                request_id,
                params,
                start,
            )

        field_errors = self._validate_required_fields(input_data, params)
        if field_errors:
            return self._input_validation_failure(field_errors, request_id, params, start)
        
        try:
            result = await self.process(input_data, params)
            # Respect error status returned by process()
            if isinstance(result, dict) and result.get("status") == "error":
                status = "error"
                confidence = 0.0
            else:
                status = "success"
                confidence = result.get("confidence", None) if isinstance(result, dict) else None
        except Exception as e:
            result = {"error": str(e)}
            status = "error"
            confidence = 0.0
        
        execution_time = int((time.time() - start) * 1000)
        self.execution_count += 1
        self.total_execution_time += execution_time

        try:
            from app.infra.monitoring import record_block_execution
            record_block_execution(self.name, execution_time, status)
        except Exception:
            logger.warning(
                "swallowed %s in execute() — continuing",
                "Exception", exc_info=True,
            )

        if status == "error":
            # So a copy of this error text in the answer is recognised at the
            # exit and replaced by one plain sentence. Never raises.
            from app.agents import answer_exit

            answer_exit.note_tool_result(self.name, {**params, "status": "error", "result": result})
        
        return {
            "block": self.name,
            "request_id": request_id,
            "status": status,
            "result": result,
            "confidence": confidence,
            "source_id": f"{self.name}-{request_id}",
            "metadata": {
                "version": self.version,
                "execution_count": self.execution_count,
                **params
            },
            "processing_time_ms": execution_time
        }
    
    def get_stats(self) -> Dict:
        """Get execution statistics"""
        avg_time = self.total_execution_time / max(self.execution_count, 1)
        return {
            "name": self.name,
            "version": self.version,
            "layer": self.layer,
            "tags": self.tags,
            "execution_count": self.execution_count,
            "avg_execution_time_ms": round(avg_time, 2)
        }


class UniversalContainer(UniversalBlock):
    """
    Universal Container - Multi-block domain system
    
    Containers group related blocks (e.g., all Construction blocks)
    """
    
    # Containers are always layer 3 (domain)
    layer: int = 3
    tags: List[str] = ["container"]
    
    # Sub-blocks this container provides
    sub_blocks: List[str] = []
    
    @abstractmethod
    async def route(self, action: str, input_data: Any, params: Dict) -> Dict:
        """Route to internal action — every container must implement this.

        Declared abstract rather than left as a `raise NotImplementedError`
        body so a container that forgets to route fails at construction time
        instead of on a user's request. `UniversalBlock` is already an ABC, so
        the enforcement is real; both existing subclasses (`DomainContainer`
        and `ConstructionContainer`) define it.
        """
        raise NotImplementedError(f"Action '{action}' not implemented")
    
    async def process(self, input_data: Any, params: Dict = None) -> Dict:
        """Default: route by action param"""
        params = params or {}
        action = params.get("action", "status")
        return await self.route(action, input_data, params)
