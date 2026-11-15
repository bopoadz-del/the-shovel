"""F2: a tool message is read in the shape the agent loop writes it.

The streaming and non-streaming loops put ``_tool_result_content(<the tool's
result object>)`` into the tool message: the inner result, never the
``{"name", "ok", "result"}`` wrapper, and past the cap a truncation envelope
whose ``preview`` is the head of that JSON. The F tests built the wrapper, so
a real successful document read was never counted. Every input here is
invented in the test and serialised the way the loop serialises it.
"""

from __future__ import annotations

import json

import pytest

from app.agents.runtime import (
    _documents_read_this_turn,
    _read_tool_result,
    _tool_result_content,
)


def _loop_tool_message(call_id: str, tool: str, tool_result: dict) -> str:
    """The tool message content exactly as app/agents/core/streaming.py and
    loop.py build it from a handler's return value."""
    inner = tool_result.get("result")
    return _tool_result_content(
        inner if isinstance(inner, dict) else {"result": inner}, max_chars=8000)


def _turn(call_id: str, tool: str, args: dict, tool_result: dict) -> list[dict]:
    return [
        {"role": "user", "content": "what does it say about the curing period"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": call_id, "type": "function",
            "function": {"name": tool, "arguments": json.dumps(args)}}]},
        {"role": "tool", "name": tool, "tool_call_id": call_id,
         "content": _loop_tool_message(call_id, tool, tool_result)},
    ]


def _fetch_ok(doc_id: str, name: str, body: str) -> dict:
    return {"name": "fetch_document", "ok": True, "result": {
        "document_id": doc_id, "filename": name, "doc_type": "spec",
        "content": body, "truncated": False, "source": "extracted",
        "partial_read": None}}


@pytest.mark.parametrize("doc_id,name,body", [
    ("d-31", "curing_method_statement.docx", "Cure for 7 days under wet hessian."),
    ("d-77", "precast_yard_plan_rev4.pdf", "Lifting points " * 3000),
], ids=["short", "over_the_cap"])
@pytest.mark.parametrize("attached", [True, False])
def test_a_successful_document_read_counts_in_the_loop_shape(
        doc_id, name, body, attached):
    messages = _turn("c1", "fetch_document", {"document_id": doc_id},
                     _fetch_ok(doc_id, name, body))
    names = _documents_read_this_turn(
        "The curing period is stated in the document.", messages,
        [{"id": doc_id, "original_name": name}] if attached else None)
    assert names == [name]


@pytest.mark.parametrize("error", [
    "document not found",
    "document has no extractable text",
])
def test_a_failed_document_read_does_not_count(error):
    name = "site_diary_week_12.pdf"
    failed = {"name": "fetch_document", "ok": False,
              "result": {"status": "error", "error": error}}
    messages = _turn("c2", "fetch_document", {"filename": name}, failed)
    assert _documents_read_this_turn(
        "I could not read it.", messages,
        [{"id": "d-12", "original_name": name}]) == []


@pytest.mark.parametrize("status,expected", [
    ("success", True),
    ("error", False),
])
def test_a_cut_result_is_judged_by_its_own_status(status, expected):
    """Past the cap the loop writes an envelope; the status in the head of
    its preview decides, and a key inside the document text never does."""
    inner = {"status": status, "filename": "bill_section_4.pdf",
             "note": 'the text says "status": "error" in a quote',
             "rows": ["Item %d Formwork 10 m2" % i for i in range(900)]}
    content = _tool_result_content(inner, max_chars=8000)
    envelope = json.loads(content)
    assert envelope["truncated"] is True and "preview" in envelope
    ok, reported = _read_tool_result(content)
    assert (ok, reported) == (expected, "bill_section_4.pdf")


@pytest.mark.parametrize("content", ["not json at all", "[1, 2, 3]"])
def test_a_non_object_tool_message_is_not_a_success(content):
    assert _read_tool_result(content) == (False, "")
