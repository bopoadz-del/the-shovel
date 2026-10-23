"""Shared helpers for lane-5 CORE fault tests.

Synthetic formulas and tools only. Register into the live registries the
driver and the calculator actually read, and remove them before the test
returns so later tests in the same process are not affected.
"""
from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional

from app.lib.formula_registry import Param, formula


def install_driver_stream(monkeypatch) -> None:
    """Script the driver through ``llm.call``. Stream that reply in small
    pieces, the same way the driver-mode tests do."""
    from app.agents.driver import llm

    async def fake_stream(agent, messages, offered, model="", timeout=120.0):
        reply = await llm.call(agent, messages, offered, model=model)
        if reply.get("status") == "success":
            text = reply["message"].get("content") or ""
            for i in range(0, len(text), 7):
                yield "content", text[i:i + 7]
        yield "done", reply

    monkeypatch.setattr(llm, "stream", fake_stream)


def agent():
    from app.agents.runtime import Agent

    return Agent(name="lane5-probe", description="probe", system_prompt="", allowed_blocks=[])


def construction_agent():
    from app.agents.runtime import Agent

    return Agent(
        name="lane5-probe", description="probe", system_prompt="probe",
        allowed_blocks=["construction"],
    )


def call(name: str, args: Dict[str, Any], cid: str) -> Dict[str, Any]:
    return {
        "id": cid, "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def scripted(replies, seen):
    from app.agents.driver import llm

    async def fake_call(agent, messages, offered, model="", timeout=120.0):
        seen.append({
            "messages": [dict(m) for m in messages],
            "tools": [t["function"]["name"] for t in offered],
        })
        reply = replies.pop(0)
        if callable(reply):
            reply = reply(messages)
        return {"status": "success", "message": reply}

    return fake_call


def no_retrieval(monkeypatch) -> None:
    def boom(*_a, **_k):
        raise AssertionError("this test must not retrieve on its own")

    import app.agents.runtime as rt
    import app.core.rag.inject as inject

    monkeypatch.setattr(inject, "rag_inject", boom)
    monkeypatch.setattr(rt, "rag_inject", boom)


def run_driver(agent, **kwargs) -> List[Dict[str, Any]]:
    from app.agents.driver import loop

    async def go():
        return [event async for event in loop.stream(agent, **kwargs)]

    return asyncio.run(go())


def answer_text(events: List[Dict[str, Any]]) -> str:
    return "".join(event.get("content", "") for event in events if event.get("type") == "token")


_TOOL_RESULTS: List[Dict[str, Any]] = []  # what each tool returned in the last drive()


def tool_envelope(seen) -> Dict[str, Any]:
    """The calculator envelope the last tool run returned. The model reads a
    calculation as prose, so the envelope is taken from the run itself."""
    if _TOOL_RESULTS:
        payload = _TOOL_RESULTS[-1]
        result = payload.get("result") if isinstance(payload, dict) else None
        if isinstance(result, dict) and ("status" in result or "calculation" in result):
            return result
        return payload if isinstance(payload, dict) else {}
    for message in reversed(seen[-1]["messages"]):
        if message.get("role") != "tool":
            continue
        payload = json.loads(message["content"])
        result = payload.get("result") if isinstance(payload, dict) else None
        if isinstance(result, dict) and ("status" in result or "calculation" in result):
            return result
        return payload if isinstance(payload, dict) else {}
    return {}


def _clear_label_cache() -> None:
    from app.lib import source_labels

    source_labels._display_by_id = None
    source_labels._param_units = None
    source_labels._known_units = None
    source_labels._id_pattern = None


def install_formula(name: str, fn: Callable[..., Any], display: str,
                    inputs: Dict[str, Any], outputs: Dict[str, str]) -> Callable[..., Any]:
    if not name.startswith("lane5_"):
        raise AssertionError("synthetic formula names stay on the lane5_ prefix")
    remove_formula(name)
    wrapped = formula(
        owner="base",
        description="Synthetic calculator used to state a general rule.",
        display_name=display,
        inputs=inputs,
        outputs=outputs,
        name=name,
    )(fn)
    from app.lib import construction_formulas as cf

    cf.CALCULATORS[name] = wrapped
    cf._DISPLAY_PHRASES.clear()
    _clear_label_cache()
    return wrapped


def remove_formula(name: str) -> None:
    from app.lib import construction_formulas as cf
    from app.lib import formula_registry as fr

    fr._REGISTRY.pop(name, None)
    cf.CALCULATORS.pop(name, None)
    cf._DISPLAY_PHRASES.clear()
    _clear_label_cache()


@contextmanager
def registered(name: str, fn: Callable[..., Any], display: str,
               inputs: Dict[str, Any], outputs: Dict[str, str]) -> Iterator[None]:
    install_formula(name, fn, display, inputs, outputs)
    try:
        yield
    finally:
        remove_formula(name)


def drive(monkeypatch, user: str, replies, *, project_id: Optional[str] = None,
          history: Optional[list] = None):
    from app.agents.driver import llm

    no_retrieval(monkeypatch)
    seen: list = []
    monkeypatch.setattr(llm, "call", scripted(list(replies), seen))
    probe = agent()
    run_tool = probe._run_tool_call
    _TOOL_RESULTS.clear()

    async def recorded(*args, **kwargs):
        out = await run_tool(*args, **kwargs)
        _TOOL_RESULTS.append(out)
        return out

    monkeypatch.setattr(probe, "_run_tool_call", recorded)
    events = run_driver(probe, user_message=user, project_id=project_id, history=history)
    return events, seen


__all__ = [
    "Param", "agent", "answer_text", "call", "construction_agent", "drive",
    "install_driver_stream", "install_formula", "no_retrieval", "registered",
    "remove_formula", "run_driver", "scripted", "tool_envelope",
]
