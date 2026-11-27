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
    (TWO_SECTIONS, [("1", "Bill No. 1 - EARTHWORKS"), ("2", "Bill No. 2 - CONCRETE")]),
    (THREE_SECTIONS, [("4A", "Bill No. 4A - Drainage"), ("5", "Bill No. 5 - Roads"),
                      ("6", "Bill No. 6 - Fencing")]),
])
async def test_sections_come_from_the_documents_own_headings(chunks, expected):
    project_id, doc_id = _ids()
    _index(project_id, doc_id, chunks)
    out = await _call(project_id, {"document_id": doc_id})
    assert out["ok"] is True
    got = [(s["section"], s["text"]) for s in out["result"]["results"]]
    assert got == expected
    assert out["result"]["document_id"] == doc_id
    assert all(s["document_id"] == doc_id for s in out["result"]["results"])


@pytest.mark.asyncio
@pytest.mark.parametrize("chunks", [TWO_SECTIONS, THREE_SECTIONS])
async def test_no_store_position_and_no_status_leaves_the_tool(chunks):
    # Store chunk positions are internal: they are never returned, so they
    # can never be shown. With no "status" the loop renders no record table.
    project_id, doc_id = _ids()
    _index(project_id, doc_id, chunks)
    out = await _call(project_id, {"document_id": doc_id})
    assert set(out["result"]) == {"document_id", "results"}
    for hit in out["result"]["results"]:
        assert set(hit) == {"document_id", "section", "text"}  # no documents row: no filename


@pytest.mark.asyncio
@pytest.mark.parametrize("chunks, name", [(TWO_SECTIONS, "Priced bill A.pdf"), (THREE_SECTIONS, "Roads bill.xlsx")])
async def test_each_section_carries_the_documents_own_name(monkeypatch, chunks, name):
    from app.core import projects

    project_id, doc_id = _ids()
    _index(project_id, doc_id, chunks)
    monkeypatch.setattr(projects, "document_names", lambda ids: {i: name for i in ids})
    out = await _call(project_id, {"document_id": doc_id})
    assert {hit["filename"] for hit in out["result"]["results"]} == {name}


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
    assert out["result"]["results"] == []


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
@pytest.mark.parametrize("chunks, suffix", [(TWO_SECTIONS, "-missing"), (THREE_SECTIONS, "x")])
async def test_an_unknown_document_is_refused(chunks, suffix):
    project_id, doc_id = _ids()
    _index(project_id, doc_id, chunks)
    out = await _call(project_id, {"document_id": doc_id + suffix})
    assert out == _error("no indexed text for that document in this project")


@pytest.mark.asyncio
@pytest.mark.parametrize("exc", [RuntimeError("db down"), OSError("connection reset")])
async def test_a_store_failure_is_told_as_a_failure_not_as_an_empty_document(monkeypatch, exc):
    project_id, doc_id = _ids()
    _index(project_id, doc_id, TWO_SECTIONS)
    store = get_store()

    def broken():
        raise exc
    monkeypatch.setattr(store, "_session_factory", broken)
    out = await _call(project_id, {"document_id": doc_id})
    assert out["ok"] is False
    error = out["result"]["error"]
    assert error.startswith("the document store failed") and type(exc).__name__ in error
    assert "no indexed text" not in error
    monkeypatch.undo()
    assert (await _call(project_id, {"document_id": doc_id}))["ok"] is True  # the document is there


@pytest.mark.parametrize("raise_errors", [False, True])
def test_the_store_raises_on_failure_only_when_asked(monkeypatch, raise_errors):
    store = get_store()
    project_id, doc_id = _ids()
    _index(project_id, doc_id, TWO_SECTIONS)
    assert len(store.chunks_for_docs(project_id, [doc_id], all_rows=True, raise_errors=raise_errors)) == 5

    def broken():
        raise RuntimeError("db down")
    monkeypatch.setattr(store, "_session_factory", broken)
    if raise_errors:
        with pytest.raises(RuntimeError):
            store.chunks_for_docs(project_id, [doc_id], all_rows=True, raise_errors=True)
    else:
        assert store.chunks_for_docs(project_id, [doc_id], all_rows=True) == []


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
    assert [s["section"] for s in out["result"]["results"]] == [number]


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
    assert [s["section"] for s in body["results"]] == numbers


def _turn(monkeypatch, project_id, first_call, answer):
    """One driver turn through ``loop.stream``: the scripted model makes one
    tool call, then writes ``answer``. Returns the text the user is shown."""
    import asyncio
    import json

    from app.agents.driver import llm, loop
    from tests.test_driver_mode import _agent

    name, args = first_call
    replies = [{"content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {
                   "name": name, "arguments": json.dumps(args)}}]},
               {"content": answer}]

    async def fake_call(agent, messages, offered, model="", timeout=120.0):
        return {"status": "success", "message": replies.pop(0)}

    async def fake_stream(agent, messages, offered, model="", timeout=120.0):
        reply = await fake_call(agent, messages, offered, model=model)
        if reply["message"].get("content"):
            yield "content", reply["message"]["content"]
        yield "done", reply

    monkeypatch.setattr(llm, "call", fake_call)
    monkeypatch.setattr(llm, "stream", fake_stream)

    async def go():
        return [e async for e in loop.stream(_agent(), user_message="Which sections does the bill print?",
                                             project_id=project_id)]
    events = asyncio.run(go())
    return "".join(e.get("content", "") for e in events if e.get("type") == "token")


PIPES = ["BILL NO. 1 - EARTHWORKS\nA1 Excavate m3 40", "BILL NO. 2 - CONCRETE\nB1 Slab m3 9",
         "BILL NO. 3 - 300MM PIPES 2.5M DEEP\nC1 Pipe m 9"]
DUCTS = ["BILL NO. 6 - ROADS\nR1 Asphalt m2 500", "BILL NO. 7 - 450MM DUCTS 1.2M COVER\nD1 Duct m 70"]


@pytest.mark.parametrize("chunks, answer", [
    (PIPES, "Bill No. 3 covers 300MM pipes 2.5M deep."),
    (PIPES, "The third section is pipes, 300MM, laid 2.5M deep."),
    (DUCTS, "Bill No. 7 covers 450MM ducts with 1.2M cover."),
    (DUCTS, "The last section is ducts, 450MM, under 1.2M of cover."),
])
def test_a_figure_quoted_from_a_heading_the_tool_returned_is_shown_whole(monkeypatch, chunks, answer):
    project_id, doc_id = _ids()
    _index(project_id, doc_id, chunks)
    shown = _turn(monkeypatch, project_id, (NAME, {"document_id": doc_id}), answer)
    assert answer in shown


@pytest.mark.parametrize("chunks, answer, unbacked", [
    (PIPES, "Bill No. 3 covers 900MM pipes 2.5M deep.", "900"),
    (DUCTS, "Bill No. 7 covers 450MM ducts with 3.8M cover.", "3.8"),
])
def test_a_figure_in_no_heading_is_still_removed(monkeypatch, chunks, answer, unbacked):
    # The credit is for what the tool returned, not for anything written
    # after it ran.
    project_id, doc_id = _ids()
    _index(project_id, doc_id, chunks)
    shown = _turn(monkeypatch, project_id, (NAME, {"document_id": doc_id}), answer)
    assert unbacked not in shown


@pytest.mark.parametrize("chunks, answer", [
    (PIPES, "The bill prints 3 sections: Bill No. 1 Earthworks, Bill No. 2 Concrete and Bill No. 3."),
    (DUCTS, "The bill prints 2 sections: Bill No. 6 Roads and Bill No. 7."),
])
def test_an_answer_naming_section_numbers_gets_no_table_added(monkeypatch, chunks, answer):
    project_id, doc_id = _ids()
    _index(project_id, doc_id, chunks)
    shown = _turn(monkeypatch, project_id, (NAME, {"document_id": doc_id}), answer)
    assert answer in shown
    assert "|" not in shown and "chunk" not in shown.lower()
