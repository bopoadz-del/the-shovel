"""The bill section map is a tool the model can call: registered in the base
package, offered on the driver path, listed in the catalogue, and answering
from the document's own indexed chunks inside the caller's project only.
Every test writes its own chunks into the store."""
from __future__ import annotations

import uuid

import numpy as np
import pytest

from app.agents.base.bill_section_map import NAME, _error, handle_bill_section_map
from app.agents.core import tool_registry
from app.core.rag.vector_store import get_store


def _index(project_id: str, doc_id: str, chunks: list[str]) -> None:
    store = get_store()
    store.upsert_chunks(project_id, doc_id, chunks, np.zeros((len(chunks), store.dim), dtype="float32"))


async def _call(project_id, args):
    tool_registry.load()
    # The registry answers the tool's name with this module's handler.
    assert tool_registry.get(NAME).handler is handle_bill_section_map
    return await handle_bill_section_map(tool_registry.ToolCall(agent=None, name=NAME, args=args,
                                                                project_id=project_id))


def _ids():
    tag = uuid.uuid4().hex[:8]
    return f"p-{tag}", f"d-{tag}"


TWO_SECTIONS = [
    "GENERAL SUMMARY\nBill No. 1 Earthworks 100\nBill No. 2 Concrete 200",
    "BILL NO. 1 - EARTHWORKS\nA1 Excavate topsoil m3 40",
    "A2 Fill m3 12\nCarried to General Summary",
    "BILL NO. 2 - CONCRETE\nB1 Blinding m3 9",
    "B2 Slab m3 30",
]
THREE_SECTIONS = [
    "Bill Number 4A\nDrainage\nC1 Pipe m 80",
    "C2 Manhole nr 4\nBill No. 4A carried to collection",
    "Bill No. 5 Roads\nD1 Asphalt m2 500",
    "Bill No. 6 Fencing\nE1 Fence m 70",
]


@pytest.mark.asyncio
@pytest.mark.parametrize("chunks, expected", [
    (TWO_SECTIONS, [("1", "EARTHWORKS", 1, 2), ("2", "CONCRETE", 3, 4)]),
    (THREE_SECTIONS, [("4A", "Drainage", 0, 1), ("5", "Roads", 2, 2), ("6", "Fencing", 3, 3)]),
])
async def test_sections_come_from_the_documents_own_headings(chunks, expected):
    project_id, doc_id = _ids()
    _index(project_id, doc_id, chunks)
    out = await _call(project_id, {"document_id": doc_id})
    assert out["ok"] is True
    got = [(s["number"], s["title"], s["first_chunk"], s["last_chunk"]) for s in out["result"]["sections"]]
    assert got == expected
    assert out["result"]["document_id"] == doc_id
    assert all(s["label"].startswith(f"Bill No. {s['number']}") for s in out["result"]["sections"])


@pytest.mark.asyncio
@pytest.mark.parametrize("chunks", [
    ["Specification clause 3.1 Concrete grade C40", "Clause 3.2 Curing seven days"],
    ["A single page that mentions a bill of quantities in a sentence"],
])
async def test_a_document_with_no_heading_maps_to_an_empty_list(chunks):
    project_id, doc_id = _ids()
    _index(project_id, doc_id, chunks)
    out = await _call(project_id, {"document_id": doc_id})
    assert out["ok"] is True
    assert out["result"]["sections"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("chunks", [TWO_SECTIONS, THREE_SECTIONS])
async def test_another_projects_document_is_never_read(chunks):
    owner, doc_id = _ids()
    other, _ = _ids()
    _index(owner, doc_id, chunks)
    out = await _call(other, {"document_id": doc_id})
    assert out["ok"] is False
    assert "no indexed text" in out["result"]["error"]
    # ... and the owner still reads it.
    assert (await _call(owner, {"document_id": doc_id}))["ok"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("project_id, args, error", [
    (None, {"document_id": "d-x"}, "no project in scope"),
    ("", {"document_id": "d-x"}, "no project in scope"),
    ("p-x", {}, "provide document_id"),
    ("p-x", {"document_id": "   "}, "provide document_id"),
    ("p-x", {"document_id": None}, "provide document_id"),
])
async def test_missing_project_or_document_is_refused(project_id, args, error):
    out = await _call(project_id, args)
    assert out == {"name": NAME, "ok": False, "result": {"status": "error", "error": error}}
    assert out == _error(error)


@pytest.mark.parametrize("message", ["no project in scope", "provide document_id"])
def test_error_is_a_failed_result_carrying_its_message(message):
    assert _error(message) == {"name": NAME, "ok": False,
                               "result": {"status": "error", "error": message}}


@pytest.mark.asyncio
async def test_an_unknown_document_is_refused():
    project_id, doc_id = _ids()
    _index(project_id, doc_id, TWO_SECTIONS)
    out = await _call(project_id, {"document_id": doc_id + "-missing"})
    assert out["ok"] is False


@pytest.mark.parametrize("hat", [None, "qaqc"])
def test_the_tool_is_offered_on_the_driver_path_and_listed_in_the_catalogue(hat):
    from app.agents.driver import catalogue, tools
    from tests.test_driver_mode import _agent

    tool_registry.load()
    assert NAME in tool_registry.tools_of("base")
    offered = tools.offered(_agent(), "synthetic-proj", hat, ["design", "qaqc"])
    names = [(t.get("function") or {}).get("name") for t in offered]
    assert NAME in names
    entry = {e["id"]: e for e in catalogue.entries()}[f"tool:{NAME}"]
    assert entry["code"] == {"tool": NAME}
    assert entry["for"].startswith("The sections a bill of quantities prints")
    assert entry["when_to_use"] and entry["when_not"]
    assert [i["name"] for i in entry["inputs"]] == ["document_id"]
    assert "params" not in entry["gaps"]
    assert entry["source"]["file"] == "app/agents/base/bill_section_map.py"


@pytest.mark.asyncio
@pytest.mark.parametrize("filler, number", [(14, "7"), (25, "9B")])
async def test_a_section_far_into_a_long_document_is_read(filler, number):
    # Every chunk of the document is read, not a first page or so of it.
    project_id, doc_id = _ids()
    chunks = [f"Preamble page {i} general notes" for i in range(filler)]
    chunks += [f"Bill No. {number} Pavements\nF1 Kerb m 120", "F2 Paving m2 300"]
    _index(project_id, doc_id, chunks)
    out = await _call(project_id, {"document_id": doc_id})
    got = [(s["number"], s["first_chunk"], s["last_chunk"]) for s in out["result"]["sections"]]
    assert got == [(number, filler, filler + 1)]


@pytest.mark.parametrize("chunks, numbers", [(TWO_SECTIONS, ["1", "2"]), (THREE_SECTIONS, ["4A", "5", "6"])])
def test_a_driver_turn_calls_the_tool_and_the_model_reads_the_sections(monkeypatch, chunks, numbers):
    # Through the driver loop: the model names the tool, the loop runs it,
    # and the model's next round reads the sections as the tool result.
    import asyncio
    import json

    from app.agents.driver import llm, loop
    from tests.test_driver_mode import _agent

    project_id, doc_id = _ids()
    _index(project_id, doc_id, chunks)
    replies = [{"content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {
                   "name": NAME, "arguments": json.dumps({"document_id": doc_id})}}]},
               {"content": "The bill has its sections."}]
    seen = []

    async def fake_call(agent, messages, offered, model="", timeout=120.0):
        seen.append([dict(m) for m in messages])
        return {"status": "success", "message": replies.pop(0)}

    async def fake_stream(agent, messages, offered, model="", timeout=120.0):
        reply = await fake_call(agent, messages, offered, model=model)
        if reply["message"].get("content"):
            yield "content", reply["message"]["content"]
        yield "done", reply

    monkeypatch.setattr(llm, "call", fake_call)
    monkeypatch.setattr(llm, "stream", fake_stream)

    async def go():
        return [e async for e in loop.stream(_agent(), user_message="Which sections?", project_id=project_id)]
    asyncio.run(go())
    assert len(seen) == 2
    tool_msgs = [m for m in seen[1] if m.get("role") == "tool" and m.get("name") == NAME]
    assert tool_msgs, "the tool result is handed back to the model"
    result = json.loads(tool_msgs[-1]["content"])
    body = result.get("result", result)
    assert [s["number"] for s in body["sections"]] == numbers
