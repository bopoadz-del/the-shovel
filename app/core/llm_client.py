"""Provider-agnostic LLM completion — a thin, tool-less call the orchestrator's
dynamic reasoning uses for UNDERSTAND / PLAN.

Reuses `runtime._llm_config()` so it follows whatever provider is configured:
DeepSeek (primary) or OpenRouter free (fallback). The reasoner never hardcodes
a provider — which one is active is a config choice, not a code change.
"""
from __future__ import annotations

import asyncio
import json as _json
import os
from typing import Any, Dict, List, Optional

import httpx
import logging

logger = logging.getLogger(__name__)


async def record_provider_usage(
    provider: str,
    model: str,
    usage: Optional[Dict[str, Any]],
    *,
    user_id: Optional[str] = None,
    agent_name: str = "",
) -> None:
    """The ONE place a non-agent provider round-trip reports to the model-call
    meter. Every provider HTTP egress outside ``app/agents/core/llm.py`` (which
    records at its own two sites) funnels its response usage through here so the
    meter counts calls by construction — a new caller that uses the shared
    egress below is metered without touching this file. Best-effort: a meter or
    DB error never sinks the call. A round-trip with no usage block is still
    recorded as a call with tokens unknown (``usage_tracker.record`` stores the
    token columns as NULL, so unknown is distinguishable from a real zero)."""
    try:
        from app.core import usage_tracker
        from app.core.offload import off_loop
        # A DB write — off the event loop (the agent path measured ~1 s stalls).
        await off_loop(
            usage_tracker.record,
            user_id=user_id,
            agent_name=agent_name or "",
            provider=provider or "",
            model=model or "",
            usage=usage,
        )
    except Exception:  # noqa: BLE001 -- the meter never breaks a provider call
        logger.debug("record_provider_usage swallowed an error", exc_info=True)


class ProviderStreamError(Exception):
    """A streaming provider call failed before/while streaming. Carries the
    HTTP status (0 for a transport error) and a short body so the caller can
    map it to its own stream-error type."""

    def __init__(self, status: int, text: str = ""):
        self.status = status
        self.text = text or ""
        super().__init__(f"provider stream HTTP {status}: {self.text[:200]}")


async def provider_post(
    url: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    *,
    timeout: Optional[float] = None,
    provider: str = "",
    model: str = "",
    user_id: Optional[str] = None,
    agent_name: str = "",
):
    """THE single non-streaming provider HTTP egress. POSTs, and on a 200
    parses the body and records usage via :func:`record_provider_usage`.
    Returns ``(response, parsed_json_or_None)``; the caller keeps its own
    status/retry handling. Every non-streaming ``client.post`` to a provider in
    the app lives here (enforced by tests/test_provider_egress_is_single.py)."""
    client_obj = await asyncio.to_thread(httpx.AsyncClient, timeout=timeout)
    async with client_obj as client:
        r = await client.post(url, json=payload, headers=headers)
    data: Optional[Dict[str, Any]] = None
    if r.status_code < 400:
        try:
            data = r.json()
        except Exception:  # noqa: BLE001
            logger.debug("provider_post: success with a non-JSON body", exc_info=True)
            data = None
        await record_provider_usage(
            provider, model, (data or {}).get("usage"),
            user_id=user_id, agent_name=agent_name,
        )
    return r, data


async def provider_stream(
    url: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    *,
    timeout: Optional[float] = None,
    provider: str = "",
    model: str = "",
    user_id: Optional[str] = None,
    agent_name: str = "",
):
    """THE single streaming provider HTTP egress. Opens the SSE stream; raises
    ``ProviderStreamError`` BEFORE the first yield on a non-200 or a transport
    error; yields each raw ``data:`` payload string (``[DONE]`` skipped); and
    after the stream records usage from the final ``usage`` chunk. Each caller
    keeps its own delta/finish-reason parsing. Every ``client.stream`` to a
    provider in the app lives here (enforced by the egress test)."""
    # Ask for a final usage chunk here so every streaming caller is metered the
    # same way without having to remember to set it (the SSE usage chunk).
    if isinstance(payload, dict):
        payload.setdefault("stream_options", {"include_usage": True})
    usage: Optional[Dict[str, Any]] = None
    started = False
    recorded = False

    async def _record_once(u):
        nonlocal recorded
        if started and not recorded:
            recorded = True
            await record_provider_usage(
                provider, model, u, user_id=user_id, agent_name=agent_name,
            )

    client_obj = await asyncio.to_thread(httpx.AsyncClient, timeout=timeout)
    try:
        async with client_obj as client:
            async with client.stream("POST", url, json=payload, headers=headers) as r:
                if r.status_code != 200:
                    # Read the body for the message — a streaming response's
                    # .text is not populated until read, and some clients don't
                    # expose .text at all.
                    body = await r.aread()
                    text = body.decode("utf-8", "ignore") if isinstance(body, (bytes, bytearray)) else str(body)
                    raise ProviderStreamError(r.status_code, text)
                started = True
                async for line in r.aiter_lines():
                    # Capture usage for the meter, but yield the RAW line so each
                    # caller keeps its own "data:" prefix and [DONE] handling.
                    # Record the moment we have usage OR reach [DONE], BEFORE
                    # yielding it — the caller breaks on [DONE] and abandons us,
                    # so a deferred finally would run too late.
                    if line and line.startswith("data:"):
                        _d = line[5:].strip()
                        if _d == "[DONE]":
                            await _record_once(usage)
                        elif _d:
                            try:
                                evt = _json.loads(_d)
                                if isinstance(evt, dict) and evt.get("usage"):
                                    usage = evt["usage"]
                                    await _record_once(usage)
                            except Exception:  # noqa: BLE001 — best-effort usage
                                logger.debug("provider_stream: usage parse skipped", exc_info=True)
                    yield line
    except ProviderStreamError:
        raise
    except Exception as e:  # noqa: BLE001 — transport/parse before/within the stream
        raise ProviderStreamError(0, str(e))
    finally:
        # Backstop: a stream that started but sent no usage chunk still counts
        # the call (tokens unknown). The guard makes this a no-op when the
        # inline record above already fired (ORDERS_1920Z #2b). Runs on normal
        # end AND on the caller's break (GeneratorExit/aclose).
        await _record_once(usage)


async def simple_complete(
    prompt: str,
    *,
    temperature: float = 0.0,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
    user_id: Optional[str] = None,
    agent_name: str = "",
) -> str:
    """Single-attempt, tool-less completion through the active provider — the
    shared HTTP egress for block reasoners (project_reasoner, formula_executor).

    POSTs one user message, RECORDS usage via :func:`record_provider_usage`, and
    returns the assistant text. Honours the provider's fixed temperature. Raises
    ``RuntimeError`` on an unset key or a non-2xx so callers surface a
    non-secret error (unchanged from the inline egresses this replaces). No
    fallback ladder — callers that want one use :func:`complete`."""
    from app.agents.runtime import _llm_config, _llm_http_timeout
    cfg = _llm_config()
    key = os.getenv(cfg["env_key"]) if cfg.get("env_key") else ""
    if cfg.get("env_key") and not key:
        raise RuntimeError(f"{cfg['env_key']} not configured")
    mdl = model or cfg["default_model"]
    temp = cfg.get("fixed_temperature", temperature)
    payload: Dict[str, Any] = {
        "model": mdl,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temp,
    }
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    r, data = await provider_post(
        cfg["url"], payload, headers,
        timeout=timeout or _llm_http_timeout(),
        provider=cfg.get("provider", ""), model=mdl,
        user_id=user_id, agent_name=agent_name,
    )
    if r.status_code != 200 or not data:
        raise RuntimeError(f"{cfg['provider']} API error (HTTP {r.status_code})")
    return (data["choices"][0]["message"].get("content") or "")


async def complete(
    messages: List[Dict[str, str]],
    *,
    temperature: float = 0.0,
    max_tokens: int = 800,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
    user_id: Optional[str] = None,
    agent_name: str = "",
) -> str:
    """Return the assistant message text for a tool-less completion.

    Mirrors ``Agent._call_llm``'s provider fallback: on a RETRYABLE failure
    from the primary (402/408/413/429/5xx or a network/timeout error) it retries
    once against ``LLM_FALLBACK_PROVIDER`` so the orchestrator's intent routing
    survives a Groq rate-limit or an OpenRouter 402 instead of silently losing
    smart routing for the turn. Auth/validation errors (other 4xx) are not
    retried. The retryable-status set is ``runtime._http_status_is_retryable``.
    """
    from app.agents.runtime import (
        _http_status_is_retryable,
        _llm_config,
        _llm_fallback_config,
        _llm_http_timeout,
        _provider_max_tokens,
    )
    cfg = _llm_config()
    fallback_cfg = _llm_fallback_config(cfg)
    if timeout is None:
        timeout = _llm_http_timeout()

    attempts = [(cfg, os.getenv(cfg["env_key"]) if cfg.get("env_key") else "", model or cfg["default_model"])]
    if fallback_cfg:
        fb_key = os.getenv(fallback_cfg["env_key"]) if fallback_cfg["env_key"] else ""
        attempts.append((fallback_cfg, fb_key, fallback_cfg["default_model"]))

    last_exc: Optional[Exception] = None
    for idx, (a_cfg, a_key, a_model) in enumerate(attempts):
        is_last = idx == len(attempts) - 1
        # Honor a provider's fixed temperature (Kimi k2.6 and other reasoning
        # models REJECT any temperature but 1 with HTTP 400 — the same rule
        # Agent._call_llm applies via fixed_temperature). Without this, the
        # predefined-synthesis path 400'd on Kimi and silently fell back to
        # the deterministic render. max_tokens goes through the same helper
        # so OpenRouter's credit-reservation ceiling applies here too.
        eff_temperature = a_cfg.get("fixed_temperature", temperature)
        payload: Dict[str, Any] = {
            "model": a_model,
            "messages": messages,
            "temperature": eff_temperature,
            "max_tokens": _provider_max_tokens(a_cfg, int(max_tokens)),
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        if a_key:
            headers["Authorization"] = f"Bearer {a_key}"
        try:
            # The single non-streaming provider egress (records usage on 200).
            r, data = await provider_post(
                a_cfg["url"], payload, headers, timeout=timeout,
                provider=a_cfg.get("provider", ""), model=a_model,
                user_id=user_id, agent_name=agent_name,
            )
            if r.status_code >= 400:
                exc = httpx.HTTPStatusError(
                    f"{a_cfg['provider']} HTTP {r.status_code}", request=r.request, response=r
                )
                if _http_status_is_retryable(r.status_code) and not is_last:
                    last_exc = exc
                    continue
                raise exc
            return ((data or {})["choices"][0]["message"].get("content") or "").strip()
        except httpx.HTTPStatusError:
            raise
        except Exception as e:  # network / timeout / parse
            last_exc = e
            if not is_last:
                continue
            raise
    # Unreachable in practice (loop always returns or raises), but keeps types honest.
    raise last_exc or RuntimeError("LLM completion failed")


async def complete_json(
    system: str,
    user: str,
    *,
    max_tokens: int = 500,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """Completion that must return a JSON object. Tolerant parse: strips code
    fences and pulls the first {...} block. Returns {} on unparseable output."""
    text = await complete(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.0, max_tokens=max_tokens, model=model, timeout=timeout,
    )
    return _extract_json_object(text)


def _extract_json_object(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t[:4].lower() == "json":
            t = t[4:]
    try:
        obj = _json.loads(t)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        logger.warning(
            "swallowed %s in _extract_json_object() — continuing",
            "Exception", exc_info=True,
        )
    start, depth = t.find("{"), 0
    if start == -1:
        return {}
    for i in range(start, len(t)):
        if t[i] == "{":
            depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = _json.loads(t[start:i + 1])
                    return obj if isinstance(obj, dict) else {}
                except Exception:
                    logger.warning(
                        "swallowed %s in _extract_json_object() brace match — continuing",
                        "Exception", exc_info=True,
                    )
                    return {}
    return {}
