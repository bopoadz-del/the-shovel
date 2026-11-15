"""A turn that ends with no final model text must never show a tool note.

The answer the user sees is final model text or one of the platform's own
user-facing notices. When the model returns nothing after a search, the
empty-turn recovery used to read any tool payload's ``note`` field as a
deliverable, so the search tool's rows-withheld count and the truncation
envelope's "LAST window" paging note were shown as the answer.

Each case runs on both loops (``Agent.chat`` and ``Agent.chat_stream``) with
two inputs: the model ends on a tool result with empty final text, and the
model ends with final text. Every tool result is built here, from the real
payload builders.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import pytest

from app.agents.base.tools import _search_payload
from app.agents.runtime import (
    _EMPTY_RESPONSE_FALLBACK,
    _TOOL_RESULT_MAX_CHARS,
    Agent,
    _format_tool_error_plain,
    _recover_answer_from_tool_messages,
    _tool_result_content,
)
from tests._fixture_values import STUB_TOKEN_02


@pytest.fixture(autouse=True)
def _stub_llm_key(monkeypatch):
    # Pin the provider so the key guard checks a placeholder; _call_llm is
    # patched, so no request leaves the process.
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_02)


def _row(i: int, snippet_len: int) -> dict:
    return {
        "chunk_id": f"c{i}",
        "file_name": f"spec_section_{i}.pdf",
        "page": i,
        "score": round(1.0 - i * 0.01, 3),
        "snippet": ("word " * (snippet_len // 5)).strip(),
    }


def _withheld_search_result() -> dict:
    """A search result whose lower rows did not fit (it carries the note)."""
    payload = _search_payload([_row(1, 200), _row(2, 200)], withheld=3)
    assert payload.get("note")  # the builder still writes its model note
    return {"name": "search_project_documents", "ok": True, "result": payload}


def _oversized_search_result() -> dict:
    """A search result larger than the tool cap, so the loop windows it."""
    rows = [_row(i, 1200) for i in range(1, 12)]
    out = {"name": "search_project_documents", "ok": True, "result": {"results": rows}}
    assert len(json.dumps(out)) > _TOOL_RESULT_MAX_CHARS
    return out


def _model_notes(tool_result: dict) -> list[str]:
    """Every model-addressed note the tool message can carry."""
    notes = []
    note = (tool_result.get("result") or {}).get("note")
    if note:
        notes.append(note)
    envelope = json.loads(_tool_result_content(tool_result))
    if envelope.get("truncated") and envelope.get("note"):
        notes.append(envelope["note"])
    notes.append("did not fit in one tool message")
    notes.append("This is the LAST window")
    return notes


def _script(final_text: str):
    """1st model call asks for a search; every later call returns ``final_text``."""
    state = {"n": 0}

    async def fake(_self, messages, api_key, **kwargs):
        state["n"] += 1
        if state["n"] == 1:
            return {"status": "success", "choice": {"message": {
                "role": "assistant", "content": "",
                "tool_calls": [{"id": "s1", "type": "function", "function": {
                    "name": "search_project_documents",
                    "arguments": json.dumps({"query": "slab thickness"})}}]}}}
        return {"status": "success", "choice": {"message": {
            "role": "assistant", "content": final_text}}}

    return fake


def _tool(result: dict):
    async def fake(self, tool_call, **kwargs):
        return result
    return fake


def _agent() -> Agent:
    return Agent(
        name="project-assistant", description="pa",
        system_prompt="x", allowed_blocks=["construction"],
    )


def _answer_chat(final_text: str, tool_result: dict) -> str:
    agent = _agent()
    with patch.object(Agent, "_call_llm", _script(final_text)), \
         patch.object(Agent, "_run_tool_call", _tool(tool_result)):
        out = asyncio.run(agent.chat(
            user_message="What slab thickness does the specification give?",
            history=[], project_id=None, conversation_id=None, user_id=None,
        ))
    return str(out.get("response") or out.get("answer") or out.get("content") or "")


def _answer_stream(final_text: str, tool_result: dict) -> str:
    agent = _agent()

    async def go():
        return [ev async for ev in agent.chat_stream(
            user_message="What slab thickness does the specification give?",
            history=[], project_id=None, conversation_id=None, user_id=None,
        )]

    with patch.object(Agent, "_call_llm", _script(final_text)), \
         patch.object(Agent, "_run_tool_call", _tool(tool_result)):
        events = asyncio.run(go())
    tokens = "".join(str(e.get("content") or "") for e in events if e.get("type") == "token")
    finals = " ".join(
        str(e.get(k) or "")
        for e in events if e.get("type") in ("done", "final", "end")
        for k in ("content", "response", "answer")
    )
    return tokens + "\n" + finals


_LOOPS = [pytest.param(_answer_chat, id="chat"), pytest.param(_answer_stream, id="stream")]
_RESULTS = [
    pytest.param(_withheld_search_result, id="rows-withheld-note"),
    pytest.param(_oversized_search_result, id="window-envelope"),
]


@pytest.mark.parametrize("answer", _LOOPS)
@pytest.mark.parametrize("make_result", _RESULTS)
def test_empty_final_text_after_a_tool_result_shows_no_tool_note(answer, make_result):
    tool_result = make_result()
    shown = answer("", tool_result)
    for note in _model_notes(tool_result):
        assert note not in shown, (note, shown)
    assert "char_offset" not in shown
    assert "unable to generate" in shown.lower(), shown


@pytest.mark.parametrize("answer", _LOOPS)
@pytest.mark.parametrize("make_result", _RESULTS)
def test_final_text_after_a_tool_result_is_the_answer(answer, make_result):
    tool_result = make_result()
    final = "The specification gives a slab thickness of 250 mm."
    shown = answer(final, tool_result)
    assert final in shown, shown
    for note in _model_notes(tool_result):
        assert note not in shown, (note, shown)


@pytest.mark.parametrize(
    "name,calc,expected",
    [
        pytest.param(
            "construction_calc",
            {"calculation": "duration", "result": {"note": "Duration is 12 days."}},
            "Duration is 12 days.",
            id="calculator-by-tool-name",
        ),
        pytest.param(
            "construction",
            {"result": {"calculation": "excavation_volume",
                        "summary": "Excavation volume is 84 m3."}},
            "Excavation volume is 84 m3.",
            id="calculator-by-named-calculation",
        ),
        pytest.param(
            "construction",
            {"calculation": "concrete_volume",
             "result": {"message": "Concrete volume is 945 m3."}},
            "Concrete volume is 945 m3.",
            id="calculator-by-envelope-calculation",
        ),
    ],
)
def test_recovery_still_reads_a_calculator_note(name, calc, expected):
    """The guard is on the tool's kind, not on the note field: a calculator's
    own note is still recovered when the model returns nothing."""
    msgs = [{"role": "tool", "name": name, "content": json.dumps(calc)}]
    assert _recover_answer_from_tool_messages("", msgs) == expected


def _withheld_message() -> str:
    return json.dumps(_withheld_search_result())


def _window_message() -> str:
    window = _tool_result_content(_oversized_search_result())
    assert json.loads(window).get("truncated")
    return window


@pytest.mark.parametrize("make_content", [
    pytest.param(_withheld_message, id="rows-withheld-note"),
    pytest.param(_window_message, id="window-envelope"),
])
@pytest.mark.parametrize("text", ["", _EMPTY_RESPONSE_FALLBACK])
def test_recovery_ignores_a_search_note_and_a_window_envelope(make_content, text):
    msgs = [{"role": "tool", "name": "search_project_documents", "content": make_content()}]
    assert _recover_answer_from_tool_messages(text, msgs) == text


def _calc_note_message(name: str = "construction_calc") -> dict:
    return {"role": "tool", "name": name,
            "content": json.dumps({"result": {"note": "Duration is 12 days."}})}


@pytest.mark.parametrize("final", [
    "The specification gives a slab thickness of 250 mm.",
    "Two submittals are still open.",
])
def test_final_model_text_is_kept_over_a_recoverable_draft(final):
    """Recovery only fills an empty turn: real final text is never replaced,
    even when a calculator draft sits in the tool messages."""
    assert _recover_answer_from_tool_messages(final, [_calc_note_message()]) == final


@pytest.mark.parametrize("name,expected", [
    pytest.param("construction_calc", "Duration is 12 days.", id="calculator-name"),
    pytest.param("formula_executor_v2", "Duration is 12 days.", id="formula-name"),
    pytest.param("search_project_documents", "", id="search-name"),
])
def test_the_tool_name_alone_decides_a_note_without_a_calculation_key(name, expected):
    """A payload with no ``calculation`` key is a calculator's only by its
    tool name; the same note from a search stays with the model."""
    assert _recover_answer_from_tool_messages("", [_calc_note_message(name)]) == expected


_PLAIN_ERROR = {"status": "error", "error": "Missing input: trench length"}


@pytest.mark.parametrize("name,recovered", [
    pytest.param("construction_calc", True, id="calculator-name"),
    pytest.param("formula_executor", True, id="formula-name"),
    pytest.param("search_project_documents", False, id="search-name"),
])
def test_the_tool_name_alone_decides_an_error_without_a_calculation_key(name, recovered):
    """A failed calculator still yields its one honest sentence; a failed
    search's error text is never shown as the answer."""
    msgs = [{"role": "tool", "name": name, "content": json.dumps(_PLAIN_ERROR)}]
    expected = _format_tool_error_plain(_PLAIN_ERROR) if recovered else ""
    assert expected != json.dumps(_PLAIN_ERROR)
    assert _recover_answer_from_tool_messages("", msgs) == expected
