"""Deterministic LLM stand-in for the UI-PHYS Playwright nightly.

Activated only when ``CEREBRUM_UI_PHYS_STUB=1``. Production chat is
untouched. The stub returns the fixture's expected facts so the UI can
be asserted without a funded provider key; RAG / predispatch / the
identifier-miss short-circuit still run in front of it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from tests._fixture_values import STUB_KEY_02

_CATALOG = Path(__file__).resolve().parents[1] / "fixtures" / "ui_phys" / "questions.json"


def _cases() -> dict[str, dict[str, Any]]:
    data = json.loads(_CATALOG.read_text(encoding="utf-8"))
    return data["cases"]


def _message_text(msg: dict[str, Any]) -> str:
    content = msg.get("content") or ""
    if isinstance(content, list):
        content = " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages or []):
        if msg.get("role") == "user":
            text = _message_text(msg).strip()
            if text:
                return text
    return ""


def _answer_for(user_text: str) -> str | None:
    for cid, case in _cases().items():
        ask = case.get("ask")
        if not ask or ask not in user_text:
            continue
        must = list(case.get("must") or [])
        must_any = list(case.get("must_any") or [])
        cite = (case.get("cite_any") or [None])[0]
        if cid == "G5":
            return (
                "I could not confirm this reference in the indexed project sources. "
                "Drawing FX-2044-0000-AAA-DWG-CA-999999 is not in the corpus."
            )
        if cid == "G1":
            return "Schedule 10: Not Used. (source: S1_contract_data.md)"
        if cid == "E4":
            return (
                "Net raft volume = 30 × 20 × 1.5 = 900 m³. "
                "Documented waste factor is 5% (× 1.05), so 900 × 1.05 = 945 m³ "
                "(about 119 trucks). construction_calc."
            )
        bits = " ".join(must + must_any)
        cite_bit = f" (source: {cite})" if cite else ""
        return f"{bits}{cite_bit}".strip()
    return None


def _answer_from_messages(messages: list[dict[str, Any]]) -> str | None:
    """Match the most recent user turn that contains a catalog ask.

    chat_stream may append a nudge as the last user message (tool-format
    retry, empty-final retry). Walk backwards so those don't miss.
    """
    hit = _answer_for(_last_user_text(messages))
    if hit:
        return hit
    for msg in reversed(messages or []):
        if msg.get("role") != "user":
            continue
        hit = _answer_for(_message_text(msg))
        if hit:
            return hit
    return None


def stub_provider_env() -> dict[str, str]:
    """Provider env that gets a turn past chat_stream's API-key gate.

    Lives here, not in conftest.py, so regular CI can import it without
    Playwright and check it against the runtime. It used to name Groq; when
    Groq was removed the gate started answering "No DEEPSEEK_API_KEY
    configured." before the stub ran, and the nightly was red for days with
    nothing in the PR suite to say why.
    """
    return {
        "LLM_PROVIDER": "deepseek",
        "DEEPSEEK_API_KEY": STUB_KEY_02,
    }


def install_stub() -> None:
    if os.getenv("CEREBRUM_UI_PHYS_STUB", "").strip() not in {"1", "true", "yes"}:
        return

    from app.agents import runtime

    async def _stub(self, messages, api_key, project_id=None, with_tools=True,
                    user_id=None, exclude_tools=None, deadline=None):
        canned = _answer_from_messages(messages)
        if canned:
            # chat_stream reads resp["choice"]["message"] — a bare
            # {"content": ...} raises KeyError and the UI shows a 500 bubble.
            return {
                "status": "success",
                "choice": {
                    "message": {
                        "role": "assistant",
                        "content": canned,
                        "tool_calls": [],
                    },
                    "finish_reason": "stop",
                },
                "raw": {"model": "ui-phys-stub"},
            }
        # Unknown prompt: do not invent. A missing-key error is honest.
        # The nightly sets a dummy provider key (see ``stub_provider_env``)
        # to pass the env-key gate; still refuse unknown asks so we never
        # dial a real provider.
        users = [
            _message_text(m)[:240]
            for m in (messages or [])
            if isinstance(m, dict) and m.get("role") == "user"
        ]
        Path("/tmp/ui_phys_stub_miss.log").write_text(
            "\n---\n".join(users) or "<no user messages>",
            encoding="utf-8",
        )
        return {
            "status": "error",
            "error": "UI-PHYS stub has no canned answer for this prompt.",
        }

    runtime.Agent._call_llm = _stub
