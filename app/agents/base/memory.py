"""The model's own JSON memory as three tools: ``write_memory``,
``read_memory``, ``list_memory`` (store: ``app.core.llm_memory``).

A note is how to work -- paths that worked, which tool answers which kind of
question, what to ask first -- never a figure: figures come from the tools at
answer time. The project is the turn's, never an argument. Nothing here puts
a note into a prompt; the model gets one only as the result of its own call,
labelled as its note with the date it was written. Every call returns a
``trace`` entry (operation, name, size, turn) the turn's trace record keeps.
"""
from __future__ import annotations

from typing import Any, Dict

from app.agents.core.tool_registry import ToolCall, tool
from app.core import llm_memory
from app.core.offload import off_loop
from app.lib.formula_registry import Param

_NAME = {
    "type": "string",
    "description": "The note's name: letters, digits, '-' or '_', at most 64 characters; "
                   "'.json' optional.",
}

_NOTES_NOT_FACTS = (
    "Your memory holds YOUR NOTES for this project: how to work in it (paths that "
    "worked, which tool answers which kind of question, what to ask the user first) "
    "and what the user tells you to remember about the project, written as told by "
    "the user. A note is not verified until the owner keeps it; an answer that uses a "
    "note says it comes from the notes and whether it is verified. Calculated figures "
    "always come from the tools at answer time, never from a note. Notes belong to this "
    "project only.")


def _fn(name: str, description: str, properties: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "strict": True,
            "parameters": {"type": "object", "properties": properties,
                           "required": sorted(properties), "additionalProperties": False},
        },
    }


WRITE_SCHEMA = _fn(
    "write_memory",
    _NOTES_NOT_FACTS + " Writes the whole note named `name`, replacing any note of that "
                       "name. Read it first if you mean to add to it.",
    {"name": _NAME,
     "data": {"type": "string",
              "description": "The whole note as JSON text: an object or an array."}})

READ_SCHEMA = _fn(
    "read_memory",
    _NOTES_NOT_FACTS + " Returns the note named `name` with the date it was written.",
    {"name": _NAME})

LIST_SCHEMA = _fn(
    "list_memory",
    _NOTES_NOT_FACTS + " Lists the names of your notes in this project, with their size "
                       "and the date each was written.",
    {})


def turn_of(call: ToolCall) -> str:
    """The turn a call belongs to: its conversation and the number of the
    user's message in it."""
    asked = sum(1 for m in (call.history or []) if isinstance(m, dict) and m.get("role") == "user")
    return f"{call.conversation_id or ''}#{asked + 1}"


def _traced_name(name: Any) -> Any:
    """The note name as the trace may keep it: the stored name when the name
    is a valid one, else None. A refused name is free text from the model
    (it can hold the user's words), and the trace goes to the turn's log
    line, which keeps ids and never text."""
    return llm_memory.valid_name(name)


def _refused(call: ToolCall, op: str, error: str, name: Any = None) -> dict:
    return {"name": call.name, "ok": False,
            "result": {"status": "error", "error": error},
            "trace": {"memory": op, "name": _traced_name(name),
                      "bytes": 0, "turn": turn_of(call), "ok": False}}


def _label(written_at: str, review: str) -> str:
    return (f"Your own note, written {written_at[:10]}, {llm_memory.verified_words(review)}. "
            "From the project notes, not from a document or a calculation.")


@tool("write_memory", owner="base", display_name="Working notes", kind="notes", schema=WRITE_SCHEMA, params={
    "name": Param(label="note name", kind="text", required=True),
    "data": Param(label="note as JSON", kind="text", required=True),
})
async def handle_write_memory(call: ToolCall) -> dict:
    name, turn = call.args.get("name"), turn_of(call)
    try:
        done = await off_loop(llm_memory.write_retrying, call.project_id or "", name,
                              call.args.get("data"), turn)
    except llm_memory.MemoryRefused as exc:
        return _refused(call, "write", str(exc), name)
    return {"name": call.name, "ok": True,
            "result": {"status": "success", "written": done["name"], "bytes": done["bytes"],
                       "replaced": not done["created"]},
            "trace": {"memory": "write", "name": done["name"], "bytes": done["bytes"],
                      "turn": turn, "ok": True}}


@tool("read_memory", owner="base", display_name="Working notes", kind="notes", schema=READ_SCHEMA, params={
    "name": Param(label="note name", kind="text", required=True),
})
async def handle_read_memory(call: ToolCall) -> dict:
    name = call.args.get("name")
    try:
        note = await off_loop(llm_memory.read, call.project_id or "", name)
    except llm_memory.MemoryRefused as exc:
        return _refused(call, "read", str(exc), name)
    return {"name": call.name, "ok": True,
            "result": {"status": "success", "note": _label(note["written_at"], note.get("review", "")),
                       "name": note["name"], "written_at": note["written_at"], "data": note["data"]},
            "trace": {"memory": "read", "name": note["name"], "bytes": note["bytes"],
                      "turn": turn_of(call), "ok": True}}


@tool("list_memory", owner="base", display_name="Working notes", kind="notes", schema=LIST_SCHEMA, params={})
async def handle_list_memory(call: ToolCall) -> dict:
    try:
        notes = await off_loop(llm_memory.listing, call.project_id or "")
    except llm_memory.MemoryRefused as exc:
        return _refused(call, "list", str(exc))
    return {"name": call.name, "ok": True,
            "result": {"status": "success",
                       "note": "Your own notes in this project: how to work, not facts.",
                       "notes": [{"name": n["name"], "bytes": n["bytes"], "written_at": n["written_at"]}
                                 for n in notes]},
            "trace": {"memory": "list", "name": None, "bytes": sum(n["bytes"] for n in notes),
                      "turn": turn_of(call), "ok": True}}
