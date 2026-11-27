"""A priced bill's section map as a tool: the model asks for the sections a
document prints (number, title and the chunk span each covers) and decides
itself what to read next. Until now only code called the section reader
(``app.core.rag.bill_sections.section_map`` through
``app.core.rag.evidence_actions.bill_section_maps``); this tool calls the
same two functions, unchanged, over every indexed chunk of the one document
named, read only inside the caller's project. Base package: every hat has it.
"""
from __future__ import annotations

from app.agents.core.tool_registry import ToolCall, tool
from app.lib.formula_registry import Param

NAME = "bill_section_map"

SCHEMA = {
    "type": "function",
    "function": {
        "name": NAME,
        "description": "The sections a bill of quantities prints, read from the document's own "
                       "headings: each section's number, title and the span of the document's "
                       "chunks it covers. An empty list means the document prints no section "
                       "heading.",
        "parameters": {
            "type": "object",
            "properties": {
                "document_id": {"type": "string",
                                "description": "The document, by the id a search or the "
                                               "document list returned."},
            },
            "required": ["document_id"],
        },
    },
}


def _error(message: str) -> dict:
    return {"name": NAME, "ok": False, "result": {"status": "error", "error": message}}


@tool(NAME, owner="base", display_name="Bill section map", schema=SCHEMA, kind="search", params={
    "document_id": Param(label="document id", kind="text", required=True),
}, when_to_use="Before reading a multi-section bill, to see which sections it prints and "
               "where each one lies.",
   when_not="To count or total rows: that is the bill row count and total.")
async def handle_bill_section_map(call: ToolCall) -> dict:
    """``{document_id, sections: [{number, title, label, first_chunk,
    last_chunk}]}`` for one document of the caller's project."""
    from functools import partial

    from app.core.offload import off_loop
    from app.core.rag.evidence_actions import bill_section_maps
    from app.core.rag.vector_store import get_store

    if not call.project_id:
        return _error("no project in scope")
    doc_id = str(call.args.get("document_id") or "").strip()
    if not doc_id:
        return _error("provide document_id")
    store = get_store()
    if not hasattr(store, "chunks_for_docs"):
        return _error("this store cannot read a whole document")
    # The store reads only the caller's project: an id from another project
    # returns no chunk, so it maps to nothing.
    maps = await off_loop(bill_section_maps, partial(store.chunks_for_docs, all_rows=True),
                          call.project_id, [doc_id])
    if doc_id not in maps:
        return _error("no indexed text for that document in this project")
    sections, _chunks = maps[doc_id]
    return {"name": NAME, "ok": True, "result": {
        "status": "success",
        "document_id": doc_id,
        "sections": [{"number": s.number, "title": s.title, "label": s.label,
                      "first_chunk": s.start, "last_chunk": s.end} for s in sections],
    }}
