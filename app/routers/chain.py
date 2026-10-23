from typing import Any, Dict, List, Optional
import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.blocks import BLOCK_REGISTRY
from app.core.privileges import (
    raise_if_inaccessible_document_ids,
    raise_if_inaccessible_project_ids,
    raise_if_privileged_block,
)
from app.dependencies import require_user
from app.dependencies import block_instances, _create_block_instance

logger = logging.getLogger(__name__)

router = APIRouter()


class ChainStep(BaseModel):
    block: str = Field(..., description="Block name to execute")
    params: Dict[str, Any] = Field(default={}, description="Block parameters")
    # F38b: stored workflows carry a per-step ``input`` dict; without this
    # field pydantic silently DROPPED it at the model boundary, so a saved
    # workflow ran its blocks with no inputs ("No file path provided")
    # while the direct /v1/execute orchestrator path threaded it fine.
    input: Optional[Dict[str, Any]] = Field(
        default=None, description="Explicit input for this step (merged into params)")
    input_mapping: Optional[Dict[str, str]] = Field(default=None, description="Map previous output fields to expected input fields")
    label: Optional[str] = Field(default=None, description="Human-readable label for this step")


class ChainRequest(BaseModel):
    steps: List[ChainStep] = Field(..., description="Chain of blocks to execute")
    initial_input: Optional[Any] = Field(default=None, description="Starting input")
    fail_fast: bool = Field(default=True, description="Stop on first validation error")
    continue_on_error: bool = Field(default=False, description="Continue chain even if a step fails")


class ChainResponse(BaseModel):
    success: bool
    status: str
    steps_executed: int
    final_output: Optional[Any]
    results: List[Dict]
    type_conversions: List[Dict] = Field(default=[])
    validation_passed: bool = Field(default=True)
    validation_errors: Optional[List[Dict]] = Field(default=None)
    error: Optional[str] = Field(default=None)
    step: Optional[int] = Field(default=None)
    partial_results: Optional[List] = Field(default=None)


@router.post("/chain")
async def chain_execute(request: ChainRequest, auth: dict = Depends(require_user)):
    """Execute a chain of blocks via OrchestratorBlock."""
    if "orchestrator" not in BLOCK_REGISTRY:
        raise HTTPException(500, "Orchestrator block not available")

    for step in request.steps:
        raise_if_privileged_block(step.block, auth.get("role"))
    raise_if_inaccessible_project_ids(
        auth,
        request.initial_input,
        *[step.params for step in request.steps],
        *[step.input for step in request.steps],
    )
    raise_if_inaccessible_document_ids(
        auth,
        request.initial_input,
        *[step.params for step in request.steps],
        *[step.input for step in request.steps],
    )

    try:
        if "orchestrator" not in block_instances:
            block_instances["orchestrator"] = _create_block_instance(BLOCK_REGISTRY["orchestrator"])

        orchestrator = block_instances["orchestrator"]
        
        # Convert steps to dict format expected by orchestrator
        steps = [step.model_dump(exclude_unset=True) for step in request.steps]
        
        orch_result = await orchestrator.execute(
            request.initial_input,
            {
                "steps": steps,
                "fail_fast": request.fail_fast,
                "continue_on_error": request.continue_on_error
            }
        )

        inner = orch_result.get("result", {})
        
        # Handle validation failure (pre-execution)
        if inner.get("status") == "error" and inner.get("validation_errors"):
            return {
                "success": False,
                "status": "validation_failed",
                "error": inner.get("error"),
                "steps_executed": 0,
                "final_output": None,
                "results": [],
                "type_conversions": [],
                "validation_passed": False,
                "validation_errors": inner.get("validation_errors"),
                "details": inner.get("details")
            }
        
        # Handle execution error
        if inner.get("status") == "error":
            return {
                "success": False,
                "status": "error",
                "error": inner.get("error"),
                "step": inner.get("step"),
                "steps_executed": inner.get("steps_executed", 0),
                "final_output": None,
                "results": inner.get("results", []),
                "type_conversions": inner.get("type_conversions", []),
                "validation_passed": inner.get("validation_passed", True),
                "partial_results": inner.get("partial_results")
            }
        
        # Handle queued status
        if inner.get("status") == "queued":
            return {
                "success": True,
                "status": "queued",
                "step": inner.get("step"),
                "block": inner.get("block"),
                "job_id": inner.get("job_id"),
                "steps_executed": inner.get("steps_executed", 0),
                "partial_results": inner.get("partial_results", [])
            }

        # Success
        return {
            "success": True,
            "status": "success",
            "steps_executed": inner.get("steps_executed", 0),
            "final_output": inner.get("final_output"),
            "results": inner.get("results", []),
            "type_conversions": inner.get("type_conversions", []),
            "validation_passed": inner.get("validation_passed", True)
        }

    except HTTPException:
        raise
    except Exception:
        logger.exception("chain execution failed")
        raise HTTPException(500, "Chain execution failed")


@router.post("/v1/chain")
async def chain_execute_v1(request: ChainRequest, auth: dict = Depends(require_user)):
    """Execute a chain of blocks (v1 API)."""
    return await chain_execute(request, auth)
