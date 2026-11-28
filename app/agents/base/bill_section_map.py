"""A priced bill's section map as a tool: the model asks for the sections a
document prints and decides itself what to read next. Until now only code
called the section reader (``app.core.rag.bill_sections.section_map``
through ``app.core.rag.evidence_actions.bill_section_maps``); this tool
calls the same two functions, unchanged, over every indexed chunk of the one
document named, read only inside the caller's project. Base package: every
hat has it.

Each section comes back as a passage in the shape every search tool uses
(``results: [{document_id, filename, section, text}]``): the heading is text
from the document, so a figure the answer quotes from it is backed the same
way as a figure from a searched chunk. Store positions never leave the tool.
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
                       "headings, in the order the document prints them: each section's number "
                       "and its heading. Only headings printed as 'Bill No. <n>' (or 'Bill "
                       "Number <n>', 'Bill # <n>') are read: an empty list means no heading of "
                       "that form, not that the bill has no sections.",
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
    """``{document_id, results: [{document_id, filename, section, text}]}``
    for one document of the caller's project, one result per section."""
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
    # returns no chunk, so it maps to nothing. A store that fails says so --
    # the platform's failure is never told as a fact about the document.
    try:
        chunks = await off_loop(store.chunks_for_docs, call.project_id, [doc_id],
                                all_rows=True, raise_errors=True)
    except Exception as exc:  # noqa: BLE001 -- reported to the model, never raised into the turn
        return _error(f"the document store failed while reading this document ({type(exc).__name__}); "
                      "nothing is known about its text")
    maps = bill_section_maps(lambda _pid, _ids: chunks, call.project_id, [doc_id])
    if doc_id not in maps:
        return _error("no indexed text for that document in this project")
    sections, _chunks = maps[doc_id]
    from app.core import projects

    try:
        filename = (await off_loop(projects.document_names, [doc_id])).get(doc_id) or ""
    except Exception:  # noqa: BLE001 -- a name is optional; the sections stand without it
        filename = ""
    found = []
    for s in sections:
        hit = {"document_id": doc_id, "section": s.number, "text": s.label}
        if filename:
            hit["filename"] = filename
        found.append(hit)
    return {"name": NAME, "ok": True, "result": {"document_id": doc_id, "results": found}}
