"""CORE provenance: a figure the user typed is theirs, and a calculator
answer does not cite an unrelated passage."""
from __future__ import annotations

import pytest

from tests.lane5_core_support import (
    Param, answer_text, call, drive, no_retrieval, registered,
)


@pytest.fixture(autouse=True)
def _driver_stream(monkeypatch):
    from tests.lane5_core_support import install_driver_stream

    install_driver_stream(monkeypatch)


def _with_search(monkeypatch, user, replies, hit):
    from app.agents.driver import llm
    from tests.lane5_core_support import agent, run_driver, scripted

    no_retrieval(monkeypatch)
    seen: list = []
    monkeypatch.setattr(llm, "call", scripted(list(replies), seen))
    probe = agent()
    real = probe._run_tool_call

    async def wrapped(tool_call, *args, **kwargs):
        name = (tool_call.get("function") or {}).get("name") or ""
        if name in ("search_project_documents", "search_general_knowledge"):
            return {"ok": True, "name": name, "result": {"results": [hit]}}
        return await real(tool_call, *args, **kwargs)

    probe._run_tool_call = wrapped
    events = run_driver(probe, user_message=user)
    return events, seen


@pytest.mark.xfail(strict=True, reason="CORE: 3 provenance tags user-typed figures as knowledge-base documents")
def test_a_user_typed_magnitude_is_not_tagged_as_a_knowledge_base_document(monkeypatch):
    user = "The reach is 10 million m. State that stored length."
    hit = {
        "text": "The stored length is 10000000 m.",
        "filename": "alpha_store_9f3a",
        "title": "Alpha memo",
        "document_id": "alpha_store_9f3a",
        "origin": "general_knowledge",
        "score": 0.91,
    }
    events, _seen = _with_search(monkeypatch, user, [
        {"content": "", "tool_calls": [call("search_general_knowledge", {"query": "stored length"}, "c1")]},
        {"content": "The stored length is 10000000 m."},
    ], hit)
    end = events[-1]
    sources = end.get("sources") or []
    trail = end.get("provenance") or []
    million = [row for row in trail if "10000000" in str(row.get("figure") or "").replace(",", "")]
    assert million, trail
    assert all(row.get("source") == "user_input" for row in million), million
    assert all(row.get("layer_label") != "Knowledge base" for row in sources)


def test_a_calculator_answer_does_not_cite_an_unrelated_passage(monkeypatch):
    name = "lane5_prov_area"

    def fn(span_m: float, breadth_m: float) -> dict:
        return {"value": span_m * breadth_m}

    user = "calculate the qxnet prov area span 4 m breadth 3 m"
    hit = {
        "text": "The weather note mentions no stored total.",
        "filename": "beta_store_1c2d",
        "title": "Beta memo",
        "document_id": "beta_store_1c2d",
        "origin": "general_knowledge",
        "score": 0.4,
    }
    with registered(name, fn, "Qxnet prov area", {
        "span_m": Param("m", lo=0, hi=1e6, label="span"),
        "breadth_m": Param("m", lo=0, hi=1e6, label="breadth"),
    }, {"value": "m2"}):
        events, _seen = _with_search(monkeypatch, user, [
            {"content": "", "tool_calls": [call("search_general_knowledge", {"query": "stored total"}, "c1")]},
            {"content": "", "tool_calls": [call("construction_calc", {
                "calculation": name, "span_m": 4, "breadth_m": 3,
            }, "c2")]},
            {"content": "The stored total is 12 m2."},
        ], hit)
    text = answer_text(events)
    names = " ".join(str(row.get("doc_name") or "") for row in (events[-1].get("sources") or []))
    assert "12" in text
    assert "beta_store_1c2d" not in names
    assert "Beta memo" not in names
