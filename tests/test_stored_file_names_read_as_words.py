"""A knowledge-library file reaches a reader by its title words, in the
answer and in Sources; a user's own document keeps the name its owner gave
it. Synthetic names, scripted model."""
from __future__ import annotations

import json

from tests.lane5_core_support import answer_text, call, no_retrieval, run_driver, scripted, agent


def _hit(layer):
    return {"snippet": "Cover is 40 mm.", "filename": "qx_alpha_design_values.md",
            "document_id": "d1", "origin": layer, "score": 0.9}


def _drive(monkeypatch, layer):
    from app.agents.driver import llm

    no_retrieval(monkeypatch)
    monkeypatch.setattr(llm, "call", scripted([
        {"content": "", "tool_calls": [call("search_general_knowledge", {"query": "cover"}, "c1")]},
        {"content": "The cover is 40 mm (Source: qx_alpha_design_values.md)."},
    ], []))
    probe = agent()

    async def tool(tool_call, *a, **k):
        return {"ok": True, "name": "search_general_knowledge", "result": {"results": [_hit(layer)]}}

    probe._run_tool_call = tool
    return run_driver(probe, user_message="What cover applies?")


def test_a_library_file_is_named_by_its_words(monkeypatch):
    from tests.lane5_core_support import install_driver_stream

    install_driver_stream(monkeypatch)
    events = _drive(monkeypatch, "general_knowledge")
    text = answer_text(events)
    names = [row["doc_name"] for row in events[-1]["sources"]]
    assert "qx_alpha_design_values" not in text and "qx alpha design values" in text
    assert names == ["qx alpha design values"]


def test_a_users_document_keeps_its_name():
    from app.agents.driver import sources

    trail = [{"role": "tool", "name": "search_project_documents", "content": json.dumps(
        {"ok": True, "result": {"results": [_hit("own")]}})}]
    rows = sources.panel(trail, [], "")
    assert rows[0]["doc_name"] == "qx_alpha_design_values.md"


def test_citing_a_library_file_by_its_stored_name_is_not_an_unbacked_attribution(monkeypatch):
    """The gate matches the stored name the evidence carries; the rename to
    words comes after, so a true citation never draws the removal note."""
    from tests.lane5_core_support import install_driver_stream

    install_driver_stream(monkeypatch)
    events = _drive(monkeypatch, "general_knowledge")
    assert "Unverified attribution removed" not in answer_text(events)
