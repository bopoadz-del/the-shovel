"""One model call for a driver turn: the active provider, the turn's tools,
``tool_choice`` auto. No tool is forced and no message is rewritten."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

# A call that offers tools is a decision: which hat, tool or workflow comes
# next. The same question must choose the same way every time, so decision
# calls are not sampled. A call with no tools only writes prose and keeps the
# agent's own temperature.
DECISION_TEMPERATURE = 0.0
DECISION_SEED = 7


def _payload(cfg: Dict[str, Any], agent: Any, messages: List[Dict[str, Any]],
             tools: Optional[List[Dict[str, Any]]], model: str, stream: bool) -> Dict[str, Any]:
    from app.agents.runtime import (
        _provider_max_tokens, _provider_temperature, _resolve_attempt_model,
        _sanitize_messages_for_provider,
    )

    payload: Dict[str, Any] = {
        "model": model or _resolve_attempt_model(cfg, agent.model),
        "messages": _sanitize_messages_for_provider(messages),
        "temperature": _provider_temperature(cfg, DECISION_TEMPERATURE if tools else agent.temperature),
        "max_tokens": _provider_max_tokens(cfg, agent.max_tokens),
        "stream": stream,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
        if cfg.get("supports_seed"):
            payload["seed"] = DECISION_SEED
    return payload


async def call(agent: Any, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]],
               model: str = "", timeout: float = 120.0) -> Dict[str, Any]:
    """``{"status": "success", "message": {...}}`` or ``{"status": "error", "error": ...}``."""
    import httpx

    from app.agents.runtime import _llm_config

    cfg = _llm_config()
    key = os.getenv(cfg["env_key"]) or ""
    if not key:
        return {"status": "error", "error": "The language model is not configured."}
    payload = _payload(cfg, agent, messages, tools, model, stream=False)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(cfg["url"], json=payload,
                                     headers={"Authorization": f"Bearer {key}"})
        if resp.status_code != 200:
            return {"status": "error", "error": f"The language model answered HTTP {resp.status_code}."}
        choice = (resp.json().get("choices") or [{}])[0]
        return {"status": "success", "message": choice.get("message") or {}}
    except httpx.HTTPError as exc:
        return {"status": "error", "error": f"The language model could not be reached ({type(exc).__name__})."}


async def stream(agent: Any, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]],
                 model: str = "", timeout: float = 120.0):
    """The same call, streamed. Yields ``("content", text)`` as the model
    writes, then exactly one ``("done", result)`` with the call's whole
    message (content and any tool calls) -- or ``("done", {"status":
    "error", ...})``."""
    import json

    import httpx

    from app.agents.runtime import _llm_config

    cfg = _llm_config()
    key = os.getenv(cfg["env_key"]) or ""
    if not key:
        yield "done", {"status": "error", "error": "The language model is not configured."}
        return
    payload = _payload(cfg, agent, messages, tools, model, stream=True)
    content: List[str] = []
    calls: Dict[int, Dict[str, Any]] = {}
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", cfg["url"], json=payload,
                                     headers={"Authorization": f"Bearer {key}"}) as resp:
                if resp.status_code != 200:
                    yield "done", {"status": "error",
                                   "error": f"The language model answered HTTP {resp.status_code}."}
                    return
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(data)
                    except ValueError:
                        continue
                    delta = ((chunk.get("choices") or [{}])[0].get("delta") or {})
                    text = delta.get("content")
                    if text:
                        content.append(text)
                        yield "content", text
                    for tc in delta.get("tool_calls") or []:
                        slot = calls.setdefault(int(tc.get("index") or 0),
                                                {"id": None, "type": "function",
                                                 "function": {"name": "", "arguments": ""}})
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        slot["function"]["name"] += fn.get("name") or ""
                        slot["function"]["arguments"] += fn.get("arguments") or ""
    except httpx.HTTPError as exc:
        yield "done", {"status": "error",
                       "error": f"The language model could not be reached ({type(exc).__name__})."}
        return
    message: Dict[str, Any] = {"content": "".join(content)}
    if calls:
        message["tool_calls"] = [calls[i] for i in sorted(calls)]
    yield "done", {"status": "success", "message": message}
