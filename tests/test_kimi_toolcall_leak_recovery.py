"""Kimi K2 leaked-tool-call recovery — live 2026-07-26 pilot finding.

Three consecutive lookup questions on the pilot corpus ended as narration
("Let me search the project documents...") with the third leaking Kimi's raw
tool-call markup into the token stream:

    ...specification.<garbage>functions.search_project_documents:0<garbage>
    {"query":"WWPS-01 wastewater pump station total flow rate"}

The turn ended without the search running. These tests pin the recovery: the
leak becomes a real tool call, prose stays prose, truncated JSON is skipped.
"""
import json

from app.agents.runtime import (
    _recover_kimi_leaked_tool_calls,
    _recover_tool_calls_from_content,
)


LEAK = (
    "Let me search the project documents directly for that pump station "
    "specification.��� �functions.search_project_documents:0�"
    '{"query": "WWPS-01 wastewater pump station total flow rate l/s", "top_k": 8}'
)


def test_leak_recovers_search_call():
    calls = _recover_tool_calls_from_content(LEAK)
    assert len(calls) == 1
    fn = calls[0]["function"]
    assert fn["name"] == "search_project_documents"
    args = json.loads(fn["arguments"])
    assert args["query"].startswith("WWPS-01")
    assert args["top_k"] == 8


def test_multiple_leaked_calls_all_recovered():
    text = (
        'x functions.search_project_documents:0 {"query": "a"} '
        'y functions.fetch_document:1 {"document_id": "d1"}'
    )
    calls = _recover_kimi_leaked_tool_calls(text)
    assert [c["function"]["name"] for c in calls] == [
        "search_project_documents", "fetch_document",
    ]


def test_plain_prose_mentioning_functions_is_not_recovered():
    assert _recover_kimi_leaked_tool_calls(
        "The functions.md file documents helper functions in the repo."
    ) == []
    assert _recover_tool_calls_from_content(
        "I will now search the project documents for the flow rate."
    ) == []


def test_truncated_json_is_skipped_not_crashed():
    truncated = 'functions.search_project_documents:0 {"query": "WWPS-01 tot'
    assert _recover_kimi_leaked_tool_calls(truncated) == []


def test_pure_json_recovery_still_works():
    calls = _recover_tool_calls_from_content(
        '{"name": "search_project_documents", "arguments": {"query": "boq total"}}'
    )
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "search_project_documents"
