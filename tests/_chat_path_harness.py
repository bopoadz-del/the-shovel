"""Drive a chat turn from the user message to the calculator call.

The entry point is the real ``Agent.chat``. Only the model is stubbed: it
returns a fixed text with no tool call. Everything that reaches a
calculator here therefore got there through deterministic code: the
predispatch probes, the required-input gate and the input read.

Stage boundary (stated, not hidden):
  tested here   - routing (which probe runs), the required-input gate,
                  the input read (named inputs bound from the message),
                  the calculator result.
  NOT tested    - any tool the model itself chooses, the model's wording
                  of the answer, and multi-step chains the model drives.
"""
from __future__ import annotations

import json
from typing import Any

from app.agents.runtime import Agent
from tests._dev_key import DEV_KEY

MODEL_STUB_TEXT = "MODEL STUB: no tool call."


def install_model_stub(monkeypatch) -> None:
    """Replace the provider call; no network, no tool chosen by a model."""

    async def _llm_no_tool(self, messages, api_key, project_id=None, **kwargs):
        return {
            "status": "success",
            "choice": {"message": {"content": MODEL_STUB_TEXT}},
            "raw": {},
        }

    monkeypatch.setattr(Agent, "_call_llm", _llm_no_tool)
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setattr(
        "app.agents.runtime.project_is_rag_ready", lambda _pid: False,
    )


def new_agent() -> Agent:
    return Agent(
        name="project-assistant",
        description="t",
        system_prompt="t",
        allowed_blocks=["construction"],
    )


async def run_turns(turns: list[str]) -> list[dict[str, Any]]:
    """One chat, one output per user turn, history carried between turns."""
    agent = new_agent()
    history: list[dict[str, str]] = []
    outs: list[dict[str, Any]] = []
    for text in turns:
        out = await agent.chat(
            text, api_key=DEV_KEY, project_id=None, history=list(history) or None,
        )
        outs.append(out)
        history += [
            {"role": "user", "content": text},
            {"role": "assistant", "content": out.get("answer") or ""},
        ]
    return outs


def calculator_calls(out: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        t for t in (out.get("tool_calls") or [])
        if t.get("name") == "construction_calc"
    ]


def reached(out: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    """(calculator name, its result dict) of the first calculator call."""
    calls = calculator_calls(out)
    if not calls:
        return None, {}
    result = calls[0].get("result") or {}
    if not isinstance(result, dict):
        return None, {}
    inner = result.get("result") if isinstance(result.get("result"), dict) else {}
    return result.get("calculation"), inner


def call_arguments(out: dict[str, Any]) -> dict[str, Any]:
    """The named inputs the calculator was called with.

    Read from the tool-call envelope the turn wrote into its own thread,
    the same record the answer and the Sources panel read.
    """
    for msg in out.get("messages") or []:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            if fn.get("name") != "construction_calc":
                continue
            raw = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else dict(raw)
            except (TypeError, ValueError):
                return {}
            params = args.get("params")
            return params if isinstance(params, dict) else {}
    return {}
