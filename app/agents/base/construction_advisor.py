"""The construction knowledge base as a chat tool.

The curated rules, thresholds and formulas in app/knowledge/construction_kb.json
are answered by the ``construction_advisor`` block; this module registers that
block as a tool with a declared schema, so a chat turn (driver or not) can
search the rules and evaluate a formula with the values the user stated. Every
result carries the entry's provenance (its source) and credibility tier; the
block, not this module, owns retrieval and evaluation.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from app.agents.core.tool_registry import ToolCall, tool
from app.lib.formula_registry import Param

NAME = "construction_advisor"

SCHEMA = {
    "type": "function",
    "function": {
        "name": NAME,
        "description": (
            "Search the curated construction knowledge base (rules, thresholds, "
            "reference designs and formulas for buildings, concrete, roads / "
            "earthworks / geotech and procurement). Each match comes with its "
            "source and credibility tier. To compute a formula, pass the values "
            "the user stated in `values` (variable -> number) and, when a "
            "previous search returned it, the formula's `rule_id`."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for."},
                "rule_id": {"type": "string",
                            "description": "Id of the formula to evaluate (from a previous result)."},
                "values": {"type": "object",
                           "description": "Formula variable values the user stated, e.g. {\"X\": 1.2}."},
                "domain": {"type": "string",
                           "description": "Optional domain to restrict to, as returned in results."},
                "top_k": {"type": ["integer", "string"], "description": "Max matches (default 5)."},
            },
            "required": ["query"],
        },
    },
}


def _values(raw: Any) -> Optional[Dict[str, Any]]:
    """The stated values as an object; a JSON string is decoded. Anything that
    is not an object is passed through so the block can refuse it."""
    if isinstance(raw, str) and raw.strip():
        import json

        try:
            return json.loads(raw)
        except ValueError:
            return raw  # type: ignore[return-value]
    return raw or None


@tool(NAME, owner="base", display_name="Construction knowledge base", schema=SCHEMA, params={
    "query": Param(label="what to look for", kind="text", required=True),
    "rule_id": Param(label="formula id to evaluate", kind="text", required=False),
    "values": Param(label="stated formula values", kind="object", required=False),
    "domain": Param(label="knowledge domain", kind="text", required=False),
    "top_k": Param("nr", 1, 20, label="number of matches", kind="integer", required=False),
})
async def handle_construction_advisor(call: ToolCall) -> dict:
    """Run the construction_advisor block on the call's arguments."""
    from app.blocks import BLOCK_REGISTRY
    from app.core.privileges import caller_may_use_block

    block_class = BLOCK_REGISTRY.get(NAME)
    if block_class is None or not caller_may_use_block(NAME):
        return {"name": call.name, "ok": False, "result": {
            "status": "error",
            "error": "the construction knowledge base is not installed here",
            "hint": "Answer without it."}}
    args = call.args or {}
    rule_id = str(args.get("rule_id") or "").strip() or None
    query = str(args.get("query") or "").strip() or (rule_id or "").replace(".", " ")
    try:
        top_k = max(1, min(20, int(args.get("top_k") or 5)))
    except (TypeError, ValueError):
        top_k = 5
    params: Dict[str, Any] = {"top_k": top_k}
    for key, val in (("domain", args.get("domain")), ("rule_id", rule_id),
                     ("values", _values(args.get("values")))):
        if val:
            params[key] = val
    result = await block_class().process({"query": query}, params)
    ok = isinstance(result, dict) and result.get("status") == "success"
    return {"name": call.name, "ok": ok, "result": result}
