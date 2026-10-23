"""Recovery of Llama-native tool-call markup from Groq `tool_use_failed` 400s.

2026-07-24 live probe: Groq rejected a legitimate drawing_qto call because the
model emitted `<function=name{json}</function>` (no `>` terminator after the
JSON). `_parse_llama_native_tool_calls` only accepted `<function=name{json}>`,
so recovery returned [] and — with no fallback provider configured — the raw
HTTP 400 ended the turn. These tests pin every observed markup variant.
"""

from __future__ import annotations

import json

from app.agents.runtime import _parse_llama_native_tool_calls


def _single_call(text: str) -> dict:
    calls = _parse_llama_native_tool_calls(text)
    assert len(calls) == 1, f"expected exactly one recovered call, got {calls!r}"
    return calls[0]


class TestParseLlamaNativeToolCalls:
    def test_canonical_inline_close(self):
        call = _single_call('<function=get_weather{"city": "Riyadh"}>')
        assert call["function"]["name"] == "get_weather"
        assert json.loads(call["function"]["arguments"]) == {"city": "Riyadh"}

    def test_closing_tag_variant_from_live_groq_400(self):
        # Exact failed_generation shape from the 2026-07-24 production probe.
        call = _single_call(
            '<function=drawing_qto{"file_path": "DWG-ZZ-9999 rev C"}</function>'
        )
        assert call["function"]["name"] == "drawing_qto"
        assert json.loads(call["function"]["arguments"]) == {
            "file_path": "DWG-ZZ-9999 rev C"
        }

    def test_canonical_llama3_name_then_json_then_closing_tag(self):
        call = _single_call(
            '<function=lookup>{"query": "manhole spacing"}</function>'
        )
        assert call["function"]["name"] == "lookup"
        assert json.loads(call["function"]["arguments"]) == {
            "query": "manhole spacing"
        }

    def test_prose_returns_empty(self):
        assert _parse_llama_native_tool_calls("The answer is 42.") == []

    def test_invalid_json_returns_empty(self):
        assert _parse_llama_native_tool_calls("<function=f{not json}>") == []

    def test_multiple_calls_recovered_in_order(self):
        calls = _parse_llama_native_tool_calls(
            '<function=a{"x": 1}></function><function=b{"y": 2}</function>'
        )
        assert [c["function"]["name"] for c in calls] == ["a", "b"]
