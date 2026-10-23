from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.blocks import BLOCK_REGISTRY
from app.dependencies import require_user
from app.dependencies import block_instances, _create_block_instance
from app.core.input_adapter import adapt_input
from app.core.privileges import (
    raise_if_inaccessible_document_ids,
    raise_if_inaccessible_project_ids,
    raise_if_privileged_block,
    raise_if_privileged_steps,
)
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


class ExecuteRequest(BaseModel):
    block: str = Field(..., description="Block name (chat, pdf, ocr, voice, etc.)")
    input: Optional[Any] = Field(default=None, description="Input data for the block")
    params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Block parameters")


@router.post("/execute")
async def execute(request: ExecuteRequest, auth: dict = Depends(require_user)):
    """Execute a single block."""
    block_name = request.block

    if block_name not in BLOCK_REGISTRY:
        raise HTTPException(404, f"Block '{block_name}' not found. Available: {list(BLOCK_REGISTRY.keys())}")

    # Skip containers - they belong to Block Store
    if block_name.startswith("container_"):
        raise HTTPException(400, f"Container '{block_name}' cannot be executed directly. Use Block Store.")

    # Code-execution blocks are admin-only (including via orchestrator steps).
    raise_if_privileged_block(block_name, auth.get("role"))
    if block_name == "orchestrator":
        raise_if_privileged_steps((request.params or {}).get("steps"), auth.get("role"))
    raise_if_inaccessible_project_ids(auth, request.params, request.input)
    raise_if_inaccessible_document_ids(auth, request.params, request.input)

    try:
        if block_name not in block_instances:
            block_instances[block_name] = _create_block_instance(BLOCK_REGISTRY[block_name])

        block = block_instances[block_name]
        
        # Adapt input to what block expects
        adapted_input = adapt_input(request.input, block)
        
        result = await block.execute(adapted_input, request.params or {})

        # A block that refuses its input as a client error (a format the
        # platform does not read) says so with ``client_error_status``; the
        # caller gets that 4xx and the block's message, not a 200.
        inner = result.get("result") if isinstance(result, dict) else None
        if isinstance(inner, dict):
            code = inner.get("client_error_status")
            if isinstance(code, int) and 400 <= code < 500:
                raise HTTPException(code, inner.get("error") or "Unsupported input")

        # Attach artifacts for the side panel (Roadmap V2 · Epic 4).
        try:
            from app.core.artifacts import result_to_artifacts
            if isinstance(result, dict) and "artifacts" not in result:
                inner = result.get("result", result)
                result["artifacts"] = result_to_artifacts(
                    inner if isinstance(inner, dict) else {}
                )
        except Exception:
            logger.warning(
                "swallowed %s in execute() — continuing",
                "Exception", exc_info=True,
            )

        return result

    except HTTPException:
        raise
    except Exception:
        # Do not leak internal exception detail to the client.
        raise HTTPException(500, "Execution failed")


@router.post("/v1/execute")
async def execute_v1(request: ExecuteRequest, auth: dict = Depends(require_user)):
    """Execute a single block (v1 API)."""
    return await execute(request, auth)
