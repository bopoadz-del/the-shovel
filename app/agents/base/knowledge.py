"""General knowledge as a tool (F-DRIVER Phase B): the model searches the
platform's general-knowledge library when it needs to, instead of the
library being injected into every turn. Base package: every hat has it."""
from __future__ import annotations

from app.agents.core.tool_registry import ToolCall, tool
from app.lib.formula_registry import Param

SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_general_knowledge",
        "description": "Search the platform's general-knowledge library (standards, codes, "
                       "guidance and training material) -- not the project's own documents. "
                       "Results are cited as general knowledge.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for."},
                "top_k": {"type": "integer", "description": "How many passages (default 5)."},
            },
            "required": ["query"],
        },
    },
}


@tool("search_general_knowledge", owner="base", display_name='Knowledge base search', schema=SCHEMA, params={
    "query": Param(label="what to look for", kind="text", required=True),
    "top_k": Param("nr", 1, 20, label="number of passages", kind="integer", required=False),
})
async def handle_search_general_knowledge(call: ToolCall) -> dict:
    """Search every configured general-knowledge project; best passages first."""
    from app.core.doc_index import search_project_documents
    from app.core.rag.retriever import _general_knowledge_project_ids

    query = str(call.args.get("query") or "").strip()
    try:
        top_k = max(1, min(20, int(call.args.get("top_k") or 5)))
    except (TypeError, ValueError):
        top_k = 5
    projects = _general_knowledge_project_ids()
    if not query or not projects:
        return {"name": call.name, "ok": False, "result": {
            "status": "error",
            "error": "no query" if not query else "no general-knowledge library is configured"}}
    found = []
    for project_id in projects:
        for hit in await search_project_documents(project_id, query, top_k) or []:
            if isinstance(hit, dict):
                found.append({**hit, "layer": "general_knowledge", "project_id": project_id})
    found.sort(key=lambda h: float(h.get("score") or 0.0), reverse=True)
    return {"name": call.name, "ok": True, "result": {"results": found[:top_k]}}
