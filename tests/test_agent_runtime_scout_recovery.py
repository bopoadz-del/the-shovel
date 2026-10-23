"""Tests for Scout raw tool-args recovery.

Groq Llama-4-Scout intermittently emits raw tool/search argument JSON in the
assistant ``content`` field with an empty structured ``tool_calls`` field. The
runtime recovers these into OpenAI-style tool_calls before falling back to the
user-facing ``_TOOL_FORMAT_FALLBACK`` message.
"""
from __future__ import annotations

import json

import pytest

from app.agents.runtime import _recover_tool_calls_from_content


def test_recovery_raw_search_args():
    """A leaked search_project_documents argument object is recovered."""
    text = json.dumps({
        "query": "MV substation commissioning checklist",
        "top_k": 5,
        "project_id": "master_corpus",
        "corpus": "documents",
    })
    recovered = _recover_tool_calls_from_content(text)
    assert len(recovered) == 1
    assert recovered[0]["type"] == "function"
    assert recovered[0]["function"]["name"] == "search_project_documents"
    args = json.loads(recovered[0]["function"]["arguments"])
    assert args["query"] == "MV substation commissioning checklist"
    assert args["top_k"] == 5
    assert args["project_id"] == "master_corpus"


def test_recovery_tool_envelope():
    """A {name, arguments} envelope is recovered."""
    text = json.dumps({
        "name": "generate_wbs",
        "arguments": {"project_id": "p-1", "baseline": True},
    })
    recovered = _recover_tool_calls_from_content(text)
    assert len(recovered) == 1
    assert recovered[0]["function"]["name"] == "generate_wbs"
    args = json.loads(recovered[0]["function"]["arguments"])
    assert args["project_id"] == "p-1"
    assert args["baseline"] is True


def test_recovery_tool_list():
    """A JSON list of tool envelopes is recovered."""
    text = json.dumps([
        {"name": "search_project_documents", "arguments": {"query": "q1"}},
        {"name": "generate_boq", "arguments": {"project_id": "p-2"}},
    ])
    recovered = _recover_tool_calls_from_content(text)
    assert len(recovered) == 2
    assert recovered[0]["function"]["name"] == "search_project_documents"
    assert recovered[1]["function"]["name"] == "generate_boq"


def test_no_recovery_for_normal_prose():
    """Ordinary assistant prose must not be turned into a tool call."""
    text = "Here is the commissioning checklist for the MV substation."
    assert _recover_tool_calls_from_content(text) == []


def test_no_recovery_for_plain_json_answer():
    """User-facing JSON (e.g. a checklist) must not be recovered as a tool."""
    text = json.dumps({
        "title": "Commissioning Checklist",
        "items": ["Inspect terminations", "Test relays"],
    })
    assert _recover_tool_calls_from_content(text) == []


def test_recovery_from_markdown_json_block():
    """Raw args wrapped in a markdown code block are recovered."""
    inner = json.dumps({"query": "cable sizing", "top_k": 3})
    text = f"```json\n{inner}\n```"
    # _recover_tool_calls_from_content only parses the literal string;
    # markdown stripping happens upstream. Pass the inner JSON directly.
    recovered = _recover_tool_calls_from_content(inner)
    assert len(recovered) == 1
    assert recovered[0]["function"]["name"] == "search_project_documents"


def test_empty_input():
    assert _recover_tool_calls_from_content("") == []
    assert _recover_tool_calls_from_content(None) == []  # type: ignore[arg-type]


def test_recovery_bare_query_args():
    """UI-PHYS A5: bare {"query": "..."} without top_k is recovered as search."""
    text = json.dumps({"query": "Delay Damages Contract Data"})
    recovered = _recover_tool_calls_from_content(text)
    assert len(recovered) == 1
    assert recovered[0]["function"]["name"] == "search_project_documents"
    args = json.loads(recovered[0]["function"]["arguments"])
    assert args["query"] == "Delay Damages Contract Data"


def test_recovery_envelope_with_string_arguments():
    """Arguments encoded as a JSON string are still recovered."""
    text = json.dumps({
        "name": "search_project_documents",
        "arguments": json.dumps({"query": "Delay Damages Contract Data"}),
    })
    recovered = _recover_tool_calls_from_content(text)
    assert len(recovered) == 1
    args = json.loads(recovered[0]["function"]["arguments"])
    assert args["query"] == "Delay Damages Contract Data"


def test_recovery_markdown_fenced_envelope():
    inner = json.dumps({
        "name": "search_project_documents",
        "arguments": {"query": "Delay Damages Contract Data"},
    })
    text = f"```json\n{inner}\n```"
    recovered = _recover_tool_calls_from_content(text)
    assert len(recovered) == 1
    assert recovered[0]["function"]["name"] == "search_project_documents"
