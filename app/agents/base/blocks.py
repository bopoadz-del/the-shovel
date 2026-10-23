"""Block argument adapters owned by the base package (every hat).

Moved from the generic block dispatch in Agent._run_tool_call (F-DRIVER
Phase A); registered with app.agents.core.tool_registry.block_args.
"""
from __future__ import annotations

from typing import Any

from app.agents.core.tool_registry import ToolCall, block_args


@block_args("construction", owner="base")
async def adapt_construction(call: ToolCall, block_input: Any, block_params: Any):
    """The generic ``construction`` block: fold top-level action and figures
    into its params, resolve stored file paths in dict payloads, and pass the
    user's words to the drafting actions."""
    from app.agents.runtime import (  # noqa: F401 -- runtime helpers, imported at call time
        _ask_is_ipc_certificate,
        _dispatch_payment_certificate,
        _off_loop,
        _resolve_block_file_input,
    )
    args = call.args
    project_id = call.project_id
    user_message = call.user_message
    if isinstance(args, dict):
        # Forced / NL calls put action + figures at the TOP level
        # (``{action: payment_certificate}`` or ``{}``), not under
        # input/params. Fold them so route() does not fall through
        # to ``status``.
        if not isinstance(block_params, dict):
            block_params = {}
        else:
            block_params = dict(block_params)
        for key, val in args.items():
            if key in ("input", "params"):
                continue
            if block_params.get(key) in (None, ""):
                block_params[key] = val
        action = str(block_params.get("action") or "")
        if action == "payment_certificate" or (
            not action and _ask_is_ipc_certificate(user_message or "")
        ):
            return await _dispatch_payment_certificate(
                block_params, user_message=user_message,
            )
        if not block_input:
            block_input = dict(block_params)
    if project_id:
        # F43: the construction container's file actions (bim_extract,
        # boq_process, drawing takeoffs) received the BARE filename and
        # died on 'File not found: qa_building.ifc' while file-schema
        # blocks got the stored path. Dict payloads only -- a bare-string
        # input here is a natural-language request, and the resolver's
        # substring matching must never rewrite prose into a file path.
        if isinstance(block_input, dict):
            block_input = (await _off_loop(_resolve_block_file_input, project_id, block_input))
        if isinstance(block_params, dict):
            block_params = (await _off_loop(_resolve_block_file_input, project_id, block_params))
    if user_message:
        # Live M14: the model rewrote wir_form scope to "blinding pour"
        # so refuse missed the operator RFP / job-req / claim.
        action = ""
        if isinstance(block_params, dict):
            action = str(block_params.get("action") or "")
        if action in {
            "wir_form", "inspection_request", "job_requisition",
            "rfp_draft", "rfp_management", "claims_builder",
            "payment_certificate",
        }:
            if isinstance(block_params, dict):
                block_params = dict(block_params)
                block_params.setdefault("user_message", user_message)
            else:
                block_params = {"user_message": user_message, "action": action}
            if isinstance(block_input, dict):
                block_input = dict(block_input)
                block_input.setdefault("user_message", user_message)
                block_input.setdefault("text", user_message)
            elif not block_input:
                block_input = {"user_message": user_message, "text": user_message}
    return block_input, block_params
