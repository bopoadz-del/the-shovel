"""CORE rendering hygiene: footers, markdown, markers, titles, and the
words a user reads instead of a tool dump."""
from __future__ import annotations

import re

import pytest
from tests._fixture_values import STUB_KEY_04

from tests.lane5_core_support import (
    Param, answer_text, call, construction_agent, drive, no_retrieval,
    registered,
)


@pytest.fixture(autouse=True)
def _driver_stream(monkeypatch):
    from tests.lane5_core_support import install_driver_stream

    install_driver_stream(monkeypatch)


def _empty_model(messages, api_key=None, project_id=None, with_tools=True,
                 user_id=None, exclude_tools=None, **_kwargs):
    return {"status": "success", "choice": {"message": {"role": "assistant", "content": ""}}}


@pytest.mark.xfail(strict=True, reason="CORE: 27 footer N of M project documents indexed glued onto the last line")
def test_the_coverage_footer_is_on_its_own_line():
    from app.agents.runtime import _postprocess_answer

    body = "The stored total is 42.3 of 12 project documents indexed"
    out = _postprocess_answer(
        body, None, [{"role": "user", "content": "hello"}], coverage=(3, 12),
    )
    assert re.search(r"(?m)^3 of 12 project documents indexed\s*$", out)


@pytest.mark.xfail(strict=True, reason="CORE: 29 unclosed parenthesis and unclosed markdown reach the answer")
def test_unclosed_markdown_does_not_reach_the_answer():
    from app.agents.runtime import _postprocess_answer

    messages = [{"role": "user", "content": "hello"}]
    bold = _postprocess_answer("The **stored total is 42.", None, messages)
    paren = _postprocess_answer("The stored total is 42 (see the note.", None, messages)
    assert bold.count("**") % 2 == 0
    assert paren.count("(") == paren.count(")")


@pytest.mark.xfail(strict=True, reason="CORE: 30 raw (source: -1) marker and a false Unverified attribution removed note")
def test_a_raw_source_marker_and_a_false_note_do_not_reach_the_answer(monkeypatch):
    events, _seen = drive(
        monkeypatch,
        "State the stored total.",
        [{"content": "The stored total is 42 per ZZ-9090-7 (source: -1)."}],
    )
    text = answer_text(events)
    assert "(source: -1)" not in text
    assert "Unverified attribution removed" not in text


def test_a_chunk_marker_is_not_visible_in_the_stream(monkeypatch):
    events, _seen = drive(
        monkeypatch,
        "State the stored total.",
        [{"content": "See chunk 4 for the stored total of 42 units."}],
    )
    text = answer_text(events)
    assert not re.search(r"chunk\s+\d+", text, re.IGNORECASE)


@pytest.mark.xfail(strict=True, reason="CORE: 32 tool signature dump in the answer; the answer names the tool")
def test_empty_model_recovery_does_not_name_the_tool_or_dump_a_signature(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_KEY_04)
    monkeypatch.setenv("DRIVER_MODE", "")
    name = "lane5_render_area"

    def fn(span_m: float, breadth_m: float) -> dict:
        return {"value": span_m * breadth_m}

    user = "calculate the qxnet kestrel total span 4 m breadth 3 m"
    probe = construction_agent()

    async def fake(self, messages, api_key=None, **kwargs):
        return _empty_model(messages, api_key, **kwargs)

    probe._call_llm = fake.__get__(probe, type(probe))
    with registered(name, fn, "Qxnet kestrel total", {
        "span_m": Param("m", lo=0, hi=1e6, label="span"),
        "breadth_m": Param("m", lo=0, hi=1e6, label="breadth"),
    }, {"value": "m2"}):
        events = _stream(probe, user)
    text = answer_text(events)
    assert "Platform calculator" not in text
    assert name not in text
    assert "span_m" not in text
    assert "Qxnet kestrel total" in text
    assert "12" in text


def test_an_echoed_signature_is_rewritten_to_the_display_name(monkeypatch):
    name = "lane5_render_sig"

    def fn(span_m: float) -> dict:
        return {"value": span_m}

    user = "calculate the qxnet render sig span 4 m"
    with registered(name, fn, "Qxnet render sig", {
        "span_m": Param("m", lo=0, hi=1e6, label="clear span"),
    }, {"value": "m"}):
        events, _seen = drive(monkeypatch, user, [
            {"content": "", "tool_calls": [call("construction_calc", {
                "calculation": name, "span_m": 4,
            }, "c1")]},
            {"content": f"{name}(span_m: float) = 4"},
        ])
    text = answer_text(events)
    assert name not in text
    assert "span_m" not in text
    assert "Qxnet render sig" in text


def _search(monkeypatch, hit, answer):
    from app.agents.driver import llm
    from tests.lane5_core_support import agent, run_driver, scripted

    no_retrieval(monkeypatch)
    seen: list = []
    monkeypatch.setattr(llm, "call", scripted([
        {"content": "", "tool_calls": [call("search_project_documents", {"query": "stored total"}, "c1")]},
        {"content": answer},
    ], seen))
    probe = agent()

    async def wrapped(tool_call, *args, **kwargs):
        return {"ok": True, "name": "search_project_documents", "result": {"results": [hit]}}

    probe._run_tool_call = wrapped
    return run_driver(probe, user_message="State the stored total.")


def test_a_source_is_shown_by_its_title(monkeypatch):
    events = _search(monkeypatch, {
        "text": "The stored total is 42 m.",
        "filename": "alpha_store_9f3a",
        "title": "Alpha memo",
        "document_id": "alpha_store_9f3a",
        "origin": "own",
        "score": 0.8,
    }, "The stored total is 42 m.")
    names = [str(row.get("doc_name") or "") for row in (events[-1].get("sources") or [])]
    assert names
    assert all(name == "Alpha memo" for name in names)
    assert all("alpha_store_9f3a" not in name for name in names)


def test_a_credit_line_uses_the_plain_name(monkeypatch):
    events = _search(monkeypatch, {
        "text": "The stored total is 42 m.",
        "filename": "alpha_store_9f3a",
        "title": "Alpha memo",
        "document_id": "alpha_store_9f3a",
        "origin": "general_knowledge",
        "score": 0.8,
    }, "The stored total is 42 m.")
    text = answer_text(events)
    blob = text + " " + " ".join(
        str(row.get("doc_name") or "") for row in (events[-1].get("sources") or [])
    )
    assert "alpha_store_9f3a" not in blob
    assert "Alpha memo" in blob


def _stream(probe, user):
    async def go():
        return [event async for event in probe.chat_stream(
            user_message=user, api_key=STUB_KEY_04,
        )]

    import asyncio
    return asyncio.run(go())
