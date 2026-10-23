"""Block argument adapters owned by the qaqc hat.

Moved from the generic block dispatch in Agent._run_tool_call (F-DRIVER
Phase A); registered with app.agents.core.tool_registry.block_args.
"""
from __future__ import annotations

from typing import Any

from app.agents.core.tool_registry import ToolCall, block_args


@block_args("validation_pipeline", owner="qaqc")
async def adapt_validation_pipeline(call: ToolCall, block_input: Any, block_params: Any):
    """Fold the model's top-level keys into the pipeline input, and the
    user's message in as the ``claim`` when no value was given."""
    args = call.args
    user_message = call.user_message
    # Leftover-hat L4: the model called validation_pipeline with
    # value=null on a prose claim (40 m span / 50 mm beam). The
    # pipeline short-circuited at syntactic and skipped Physical.
    # Fold top-level LLM keys and the user message in as `claim`.
    merged: dict[str, Any] = {}
    if isinstance(block_input, dict):
        merged.update(block_input)
    if isinstance(block_params, dict):
        for k, v in block_params.items():
            merged.setdefault(k, v)
    for k, v in args.items():
        if k not in ("input", "params"):
            merged.setdefault(k, v)
    if merged.get("value") is None and not merged.get("claim") and user_message:
        merged["claim"] = user_message
    block_input = merged
    return block_input, block_params
