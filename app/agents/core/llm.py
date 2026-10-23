"""Model calls: one call with tools, and the streamed synthesis.

Moved from the Agent class in app/agents/runtime.py by
scripts/move_agent_methods.py (F-DRIVER Phase A). Each function is still
an Agent method (the class binds it); runtime names are read at call
time, so this module does not import runtime when it is imported.
"""
from __future__ import annotations

from typing import Any, AsyncIterator


async def _call_llm(
    self,
    messages: list[dict[str, Any]],
    api_key: str,
    project_id: str | None = None,
    with_tools: bool = True,
    user_id: str | None = None,
    exclude_tools: set | None = None,
    deadline: float | None = None,
) -> dict[str, Any]:
    from app.agents.runtime import (  # noqa: F401 -- read at call time
        _LOG, _MIN_LLM_ATTEMPT_SECONDS, _OPENROUTER_402_MAX_RETRIES,
        _OPENROUTER_429_MAX_RETRIES, _compact_messages_for_openrouter, _end_cut_answer,
        _finished_on_length, _forced_specific_tool,
        _http_400_is_retryable, _llm_choice_is_empty_or_filtered, _llm_config,
        _llm_fallback_ladder, _llm_http_timeout, _openrouter_402_afford_max_tokens,
        _openrouter_402_is_in_flight, _openrouter_402_should_retry,
        _parse_llama_native_tool_calls, _parse_prompt_token_limit, _provider_max_tokens,
        _provider_retry_delay_seconds, _provider_temperature, _resolve_attempt_model,
        _sanitize_messages_for_provider, _user_intent_requires_tool, asyncio, httpx, json, os,
        time,
    )
    cfg = _llm_config()
    # Soft daily cap: refuse the call when today's spend already meets
    # USAGE_DAILY_CAP_USD for this user. Only enforced for authenticated
    # callers — internal calls without a user_id are not capped (they
    # shouldn't be billable in the first place). A missing / unparseable
    # / non-positive cap disables the check entirely.
    if user_id:
        try:
            cap = float(os.getenv("USAGE_DAILY_CAP_USD") or "0")
        except ValueError:
            cap = 0.0
        if cap > 0:
            try:
                from app.core import usage_tracker
                if usage_tracker.is_over_cap(user_id, cap):
                    today = usage_tracker.daily_total(user_id)
                    return {
                        "status": "error",
                        "error": (
                            f"Daily LLM cost cap reached: "
                            f"${today['cost_usd']:.4f} >= ${cap:.4f} "
                            f"(USAGE_DAILY_CAP_USD). Retry after 00:00 UTC."
                        ),
                    }
            except Exception:  # noqa: BLE001
                # A broken usage tracker must never block a real call.
                _LOG.warning(
                    "swallowed %s in _call_llm() — continuing",
                    "Exception", exc_info=True,
                )
    # An agent that pinned a provider-specific model is left alone; an
    # unpinned/legacy-placeholder (or a foreign pin such as kimi-k2.6
    # on DeepSeek) uses the active provider's default from _llm_config.
    model = _resolve_attempt_model(cfg, self.model)
    # Whitelist-sanitise every outbound message (drops `reasoning` and any
    # other non-standard field that would make a strict provider reject
    # the request). Single chokepoint, covers all callers.
    messages = _sanitize_messages_for_provider(messages)
    payload = {
        "model": model,
        "messages": messages,
        "temperature": self.temperature,
        "max_tokens": self.max_tokens,
        "stream": False,
    }
    tools = self.tool_definitions(project_id=project_id)
    if exclude_tools:
        tools = [
            t for t in tools
            if t.get("function", {}).get("name") not in exclude_tools
        ]
    if tools and with_tools:
        payload["tools"] = tools
        # When the latest user message names a deliverable (schedule,
        # WBS, BOQ, cost estimate, etc.), force the model to emit a
        # tool call instead of drifting into prose. Gated to the
        # project-assistant agent because other agents (e.g.
        # heavy-reasoning) have their own discipline + may legitimately
        # answer in prose on the same keywords. Q&A queries that don't
        # name a deliverable keep tool_choice="auto" as before.
        tool_names = {t.get("function", {}).get("name") for t in tools}
        forced_tool = (
            _forced_specific_tool(messages, tool_names)
            if self.name in (
                "project-assistant", "heavy-reasoning", "construction-pm",
            ) else None
        )
        # Pinned construction-pm keeps its own discipline for the rest
        # of the intent map (histogram / look-ahead stay on predispatch).
        # Duration-override F2 is the one steal that left the hat
        # answering from RAG excerpts instead of generate_wbs.
        if self.name == "construction-pm" and forced_tool != "generate_wbs":
            forced_tool = None
        requires_tool = (
            self.name == "project-assistant"
            and _user_intent_requires_tool(messages)
        )
    else:
        forced_tool = None
        requires_tool = False

    # Provider attempts: primary first, then the full fallback ladder
    # (same-provider model, then cross-provider). Retryable failures
    # (413/429/5xx/timeout) walk the ladder. Auth/validation 401/403/404
    # and generic 400s are not retried. Kimi conversation-shape 400s
    # (orphaned tool_call_id / tokenization) ARE retried, skipping any
    # remaining same-provider hop — another Moonshot model 400s the
    # same way; Groq is the recovery.
    attempts = [(cfg, api_key, model)]
    for fb in _llm_fallback_ladder(cfg):
        fb_key = os.getenv(fb["env_key"]) if fb.get("env_key") else ""
        attempts.append((fb, fb_key, fb["default_model"]))

    def _is_retryable(status: int, provider: str = "") -> bool:
        # DeepSeek HTTP 402 (insufficient credit) is "primary
        # unreachable" — hop to LLM_FALLBACK_PROVIDER. OpenRouter
        # generic 402 stays same-hop only so we do not burn a paid
        # DeepSeek fallback (test_openrouter_402_does_not_fall_back_to_deepseek).
        if status == 402 and provider != "openrouter":
            return True
        return status in (408, 413, 429) or status >= 500

    def _tool_choice_for(provider: str):
        # Neither supported provider is forced. OpenRouter's routed free
        # models must not be forced (OpenAI-compatible, but routing can
        # 400/loop). DeepSeek reasoner / thinking mode 400s when a specific
        # tool is forced. Both call tools cleanly on "auto". Decided
        # PER-ATTEMPT so a fallback to a different provider gets the right
        # value.
        if provider in ("openrouter", "deepseek"):
            return "auto"
        if forced_tool:
            # Force THIS tool by name — "required" alone let the model pick
            # the wrong tool (search) and bypass the authoritative data.
            return {"type": "function", "function": {"name": forced_tool}}
        if requires_tool:
            return "required"
        return "auto"

    # Wall-clock budget. chat_stream caps the whole TURN; _call_llm walks
    # a provider fallback ladder INSIDE that cap, so without a shared
    # deadline the ladder can spend _llm_http_timeout() per hop (150s by
    # default) against a 240s turn cap and return nothing at all.
    #
    # Live request 43e40b3a-e8f: iter=0 took 126.8s and produced a dangling
    # search preamble, the forced no-tools retry started at cum=126.8s, and
    # the turn deadline cancelled the producer 113s later while that retry
    # was still awaiting -- there is no "forced-retry call=" timing line for
    # that request because the await never returned. The user waited four
    # minutes and got a timeout banner.
    #
    # `deadline` is a time.monotonic() instant, not a duration.
    def _attempt_timeout() -> float:
        base = _llm_http_timeout()
        if deadline is None:
            return base
        return max(
            _MIN_LLM_ATTEMPT_SECONDS,
            min(base, deadline - time.monotonic()),
        )

    def _budget_exhausted() -> bool:
        return (
            deadline is not None
            and (deadline - time.monotonic()) <= _MIN_LLM_ATTEMPT_SECONDS
        )

    last_error: dict[str, Any] = {"status": "error", "error": "LLM call failed"}
    skip_providers: set[str] = set()
    for attempt_idx, (a_cfg, a_key, a_model) in enumerate(attempts):
        is_last = attempt_idx == len(attempts) - 1
        if _budget_exhausted():
            _LOG.warning(
                "llm: skipping %s attempt -- wall-clock budget exhausted",
                a_cfg.get("provider"),
            )
            return {
                "status": "error",
                "error": (
                    "LLM call skipped -- no wall-clock budget left in "
                    "this turn."
                ),
            }
        attempt_timeout = _attempt_timeout()
        if a_cfg.get("provider") in skip_providers:
            if is_last:
                return last_error
            continue
        payload["model"] = a_model
        # Per-attempt like tool_choice: a fallback to a different provider
        # must get that provider's accepted temperature (both DeepSeek and
        # OpenRouter accept the agent's own), not the primary's.
        payload["temperature"] = _provider_temperature(a_cfg, self.temperature)
        payload["max_tokens"] = _provider_max_tokens(a_cfg, self.max_tokens)
        if a_cfg.get("provider") == "openrouter":
            payload["messages"] = _compact_messages_for_openrouter(messages)
        else:
            payload["messages"] = messages
        if tools and with_tools:
            payload["tool_choice"] = _tool_choice_for(a_cfg["provider"])
        try:
            headers = (
                {"Authorization": f"Bearer {a_key}", "Content-Type": "application/json"}
                if a_key
                else {"Content-Type": "application/json"}
            )
            openrouter_402_retries = 0
            openrouter_429_retries = 0
            while True:
                async with httpx.AsyncClient(timeout=attempt_timeout) as client:
                    r = await client.post(
                        a_cfg["url"],
                        json=payload,
                        headers=headers,
                    )
                if (
                    r.status_code == 402
                    and a_cfg.get("provider") == "openrouter"
                    and openrouter_402_retries < _OPENROUTER_402_MAX_RETRIES
                    and _openrouter_402_should_retry(r.text)
                ):
                    openrouter_402_retries += 1
                    afford = _openrouter_402_afford_max_tokens(r.text)
                    if afford is not None:
                        current = int(payload.get("max_tokens") or afford)
                        payload["max_tokens"] = max(1, min(current, afford))
                    limit = _parse_prompt_token_limit(r.text)
                    if limit:
                        _have, cap = limit
                        payload["messages"] = _compact_messages_for_openrouter(
                            payload.get("messages") or messages,
                            token_ceiling=cap,
                        )
                        _LOG.warning(
                            "llm: openrouter HTTP 402 prompt-limit "
                            "%s>%s — compacting to cap (retry %s/%s)",
                            _have, cap,
                            openrouter_402_retries,
                            _OPENROUTER_402_MAX_RETRIES,
                        )
                    _LOG.warning(
                        "llm: openrouter HTTP 402 — retry %s/%s "
                        "(afford=%s in_flight=%s max_tokens=%s)",
                        openrouter_402_retries,
                        _OPENROUTER_402_MAX_RETRIES,
                        afford,
                        _openrouter_402_is_in_flight(r.text),
                        payload.get("max_tokens"),
                    )
                    await asyncio.sleep(0.4 * openrouter_402_retries)
                    continue
                if (
                    r.status_code == 429
                    and a_cfg.get("provider") == "openrouter"
                    and openrouter_429_retries < _OPENROUTER_429_MAX_RETRIES
                ):
                    openrouter_429_retries += 1
                    delay = _provider_retry_delay_seconds(
                        r, openrouter_429_retries,
                    )
                    _LOG.warning(
                        "llm: openrouter HTTP 429 — retrying in %.1fs "
                        "(attempt %s)",
                        delay, openrouter_429_retries,
                    )
                    await asyncio.sleep(delay)
                    continue
                break
        except httpx.TimeoutException:
            last_error = {"status": "error", "error": f"{a_cfg['provider']} LLM call timed out ({int(attempt_timeout)}s)."}
            if not is_last:
                _LOG.warning("llm: %s timed out — falling back to %s", a_cfg["provider"], attempts[attempt_idx + 1][0]["provider"])
                continue
            return last_error
        except Exception as e:  # noqa: BLE001 — network/transport
            last_error = {"status": "error", "error": f"{a_cfg['provider']} LLM call failed: {e}"}
            if not is_last:
                _LOG.warning("llm: %s network error (%s) — falling back to %s", a_cfg["provider"], e, attempts[attempt_idx + 1][0]["provider"])
                continue
            return last_error

        if r.status_code >= 400:
            body = r.text
            # Groq's tool-use validator rejects Llama-native function
            # markup (`<function=name{json}>`) with HTTP 400 and
            # `tool_use_failed`. The raw markup lives in
            # `error.failed_generation`. Recover it into OpenAI-style
            # tool_calls so the agent loop can dispatch and continue
            # rather than bubbling a 400 to the user.
            tool_use_failed_unrecovered = False
            try:
                err = json.loads(body)
                err_obj = err.get("error", {}) if isinstance(err, dict) else {}
                if err_obj.get("code") == "tool_use_failed":
                    failed_gen = err_obj.get("failed_generation", "") or ""
                    recovered = _parse_llama_native_tool_calls(failed_gen)
                    if recovered:
                        return {
                            "status": "success",
                            "choice": {
                                "message": {
                                    "role": "assistant",
                                    "content": "",
                                    "tool_calls": recovered,
                                },
                            },
                            "raw": err,
                        }
                    # Llama emitted the answer as PROSE instead of a tool
                    # call (common on Groq when tool_choice is forced with a
                    # large production context). We can't recover a tool
                    # call from prose — but this is a model-side failure a
                    # different provider CAN handle, so treat it as
                    # retryable and fall back rather than erroring the turn.
                    tool_use_failed_unrecovered = True
            except (json.JSONDecodeError, KeyError, TypeError):
                _LOG.warning(
                    "swallowed %s in _call_llm() — continuing",
                    "(json.JSONDecodeError, KeyError, TypeError)", exc_info=True,
                )
            if r.status_code == 402:
                last_error = {
                    "status": "error",
                    "error": (
                        f"{a_cfg['provider']} HTTP 402: insufficient "
                        f"credit — {body[:300]}"
                    ),
                }
            else:
                last_error = {
                    "status": "error",
                    "error": (
                        f"{a_cfg['provider']} HTTP {r.status_code}: "
                        f"{body[:300]}"
                    ),
                }
            shape_400 = r.status_code == 400 and _http_400_is_retryable(body)
            if shape_400:
                skip_providers.add(str(a_cfg.get("provider") or ""))
            if (
                _is_retryable(
                    r.status_code, str(a_cfg.get("provider") or ""),
                )
                or tool_use_failed_unrecovered
                or shape_400
            ) and not is_last:
                reason = (
                    "tool_use_failed (prose)" if tool_use_failed_unrecovered
                    else ("conversation-shape 400" if shape_400 else f"HTTP {r.status_code}")
                )
                nxt_provider = next(
                    (c[0].get("provider") for c in attempts[attempt_idx + 1:]
                     if c[0].get("provider") not in skip_providers),
                    attempts[attempt_idx + 1][0]["provider"],
                )
                _LOG.warning("llm: %s %s — falling back to %s", a_cfg["provider"], reason, nxt_provider)
                continue
            # No fallback taken (non-retryable status, or this was the last
            # provider): this error ENDS the turn. A silently-returned 4xx
            # (e.g. Moonshot's temperature 400) was previously invisible in
            # prod logs, so the acceptance criterion "no HTTP 400 events"
            # could not be verified from logs that never recorded it. Observe
            # only — no retry logic added here.
            _LOG.warning("llm: %s HTTP %s — no fallback taken, turn errors: %s",
                         a_cfg["provider"], r.status_code, body[:200])
            return last_error

        # ── success on this provider ──────────────────────────────────
        active_cfg = a_cfg
        try:
            data = r.json()
            choice = (data.get("choices") or [{}])[0]
            # Best-effort cost tracking — never let it sink an LLM call.
            try:
                from app.core import usage_tracker
                from app.core.offload import off_loop
                # A database write: off the event loop (live: 1.0 s stall at 15 users).
                await off_loop(
                    usage_tracker.record,
                    user_id=user_id,
                    agent_name=self.name,
                    provider=active_cfg.get("provider", ""),
                    model=data.get("model") or active_cfg.get("default_model") or "",
                    usage=data.get("usage"),
                )
            except Exception:  # noqa: BLE001
                _LOG.warning(
                    "swallowed %s in _call_llm() — continuing",
                    "Exception", exc_info=True,
                )
            msg = choice.get("message") or {}
            if msg and not (msg.get("tool_calls") or []):
                # Observe-only tripwire: the model answered a turn whose
                # intent REQUIRED a tool (requires_tool) without calling one.
                # On groq/kimi we can't force tool_choice (it 400s), so the
                # model may skip on "auto" and emit a degenerate prose answer
                # — the exact failure the hardened smoke catches externally.
                # Emit one grep-able marker so a resurgence is queryable from
                # prod logs (list_logs text="TOOL_SKIP"). This does NOT act:
                # no forced retry is triggered here. (DSML-in-prose
                # tool call recovered downstream would false-positive here;
                # this only logs.)
                if requires_tool:
                    _LOG.warning(
                        "TOOL_SKIP agent=%s provider=%s model=%s answer_chars=%d "
                        "— deliverable intent, no tool call",
                        self.name, a_cfg.get("provider"),
                        data.get("model") or a_model, len(msg.get("content") or ""),
                    )
            if _llm_choice_is_empty_or_filtered(choice) and not is_last:
                skip_providers.add(str(a_cfg.get("provider") or ""))
                nxt_provider = next(
                    (c[0].get("provider") for c in attempts[attempt_idx + 1:]
                     if c[0].get("provider") not in skip_providers),
                    attempts[attempt_idx + 1][0]["provider"],
                )
                last_error = {
                    "status": "error",
                    "error": (
                        f"{a_cfg['provider']} empty or content-filtered "
                        "completion"
                    ),
                }
                _LOG.warning(
                    "llm: %s empty/content-filter — falling back to %s",
                    a_cfg["provider"], nxt_provider,
                )
                continue
            if (_finished_on_length(choice) and (msg.get("content") or "").strip()
                    and not msg.get("tool_calls")):
                # The answer stopped where the output token limit fell,
                # usually mid-sentence. It ends on its last whole sentence
                # and says why it stops.
                _LOG.warning(
                    "llm: %s finish_reason=length agent=%s max_tokens=%s chars=%d — "
                    "answer ends on its last whole sentence",
                    a_cfg.get("provider"), self.name, payload.get("max_tokens"),
                    len(msg.get("content") or ""),
                )
                msg["content"] = _end_cut_answer(msg["content"])
            return {"status": "success", "choice": choice, "raw": data}
        except Exception as e:  # noqa: BLE001 — response parse / rewrite
            last_error = {"status": "error", "error": f"{a_cfg['provider']} response error: {e}"}
            if not is_last:
                _LOG.warning("llm: %s response error (%s) — falling back to %s", a_cfg["provider"], e, attempts[attempt_idx + 1][0]["provider"])
                continue
            return last_error

    return last_error


async def _stream_synthesis(
    self,
    messages: list[dict[str, Any]],
    api_key: str,
    project_id: str | None = None,
    user_id: str | None = None,
) -> AsyncIterator[str]:
    """Stream a tool-free synthesis completion, yielding content deltas.

    Used ONLY for the forced (``with_tools=False``) synthesis call, on the
    DeepSeek / OpenRouter OpenAI SSE shape. Mirrors ``_call_llm``'s setup — same
    ``_llm_config``, same leftover-model-name remap, same soft daily-cap
    semantics, and critically the SAME ``_sanitize_messages_for_provider``
    chokepoint — but sets ``stream=True`` and offers NO tools, so no
    tool_call deltas can appear (honours "tool iterations stay
    non-streaming").

    Raises ``_SynthStreamError`` on any PRE-first-token failure (wrong
    provider, over cap, non-200, connect/transport error) so the caller can
    fall back to the non-streaming ``_call_llm`` path. A mid-stream drop is
    also surfaced as ``_SynthStreamError`` — the caller distinguishes the two
    via whether it has already emitted tokens, and never restarts a stream
    it has begun.
    """
    from app.agents.runtime import (  # noqa: F401 -- read at call time
        _LOG, _SynthLengthCut, _SynthStreamError, _compact_messages_for_openrouter, _llm_config,
        _llm_http_timeout, _provider_max_tokens, _provider_temperature, _resolve_attempt_model,
        _sanitize_messages_for_provider, httpx, json, os,
    )
    cfg = _llm_config()
    if cfg["provider"] not in ("openrouter", "deepseek"):
        # DeepSeek and OpenRouter share the OpenAI SSE shape; nothing else
        # is a selectable provider.
        raise _SynthStreamError(
            "streaming synthesis only verified for openrouter/deepseek"
        )
    # Soft daily cap: mirror _call_llm. Over cap -> fall back so the
    # non-streaming path emits the structured cap error the UI expects.
    if user_id:
        try:
            cap = float(os.getenv("USAGE_DAILY_CAP_USD") or "0")
        except ValueError:
            cap = 0.0
        if cap > 0:
            try:
                from app.core import usage_tracker
                over = usage_tracker.is_over_cap(user_id, cap)
            except Exception:  # noqa: BLE001 — broken tracker must not block
                over = False
            if over:
                raise _SynthStreamError("daily cap reached")
    model = _resolve_attempt_model(cfg, self.model)
    messages = _sanitize_messages_for_provider(messages)
    if cfg.get("provider") == "openrouter":
        messages = _compact_messages_for_openrouter(messages)
    temperature = _provider_temperature(cfg, self.temperature)
    stream_max_tokens = _provider_max_tokens(cfg, self.max_tokens)
    headers = (
        {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        if api_key
        else {"Content-Type": "application/json"}
    )
    usage: dict[str, Any] | None = None
    # Reasoning-phase deltas seen but deliberately not yielded (see below).
    reasoning_deltas = 0
    content_deltas = 0
    finish_reason = ""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(_llm_http_timeout(), read=_llm_http_timeout())) as client:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": stream_max_tokens,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            async with client.stream(
                "POST", cfg["url"], json=payload, headers=headers,
            ) as r:
                if r.status_code != 200:
                    await r.aread()
                    raise _SynthStreamError(
                        f"{cfg['provider']} HTTP {r.status_code}: {r.text[:200]}"
                    )
                async for line in r.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        evt = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = evt.get("choices") or []
                    if choices:
                        finish_reason = str(choices[0].get("finish_reason") or finish_reason)
                        _delta_obj = choices[0].get("delta") or {}
                        # Reasoning models (e.g. deepseek-reasoner) stream a
                        # THINKING phase as ``reasoning_content`` before
                        # any ``content`` arrives. It is not the answer:
                        # it contains discarded intermediate claims that
                        # would read as authoritative in the UI, and it
                        # is not what gets persisted. Count it so the
                        # caller can tell "model is working" apart from
                        # "provider returned nothing", but never yield
                        # it as answer text.
                        if _delta_obj.get("reasoning_content"):
                            reasoning_deltas += 1
                        delta = _delta_obj.get("content")
                        if delta:
                            content_deltas += 1
                            yield delta
                    if evt.get("usage"):
                        usage = evt["usage"]
    except _SynthStreamError:
        raise
    except Exception as e:  # noqa: BLE001 — network/transport/parse
        raise _SynthStreamError(f"{cfg['provider']} stream error: {e}")
    if reasoning_deltas and not content_deltas:
        # HTTP 200 with a full thinking phase and no answer. The caller sees
        # zero tokens and falls back to non-streaming, which is correct but
        # looks identical to a dead provider in the logs. Say which it was.
        _LOG.warning(
            "%s streamed %d reasoning deltas and no content; "
            "falling back to non-streaming synthesis",
            cfg["provider"], reasoning_deltas,
        )
    # Best-effort cost tracking — never let it sink the turn.
    if usage is not None:
        try:
            from app.core import usage_tracker
            from app.core.offload import off_loop
            # A database write: off the event loop (live: 628 ms stall at 10 users).
            await off_loop(
                usage_tracker.record,
                user_id=user_id,
                agent_name=self.name,
                provider=cfg.get("provider", ""),
                model=model,
                usage=usage,
            )
        except Exception:  # noqa: BLE001
            _LOG.warning(
                "swallowed %s in _stream_synthesis() — continuing",
                "Exception", exc_info=True,
            )
    if content_deltas and finish_reason.lower() in ("length", "max_tokens"):
        # Every token is already out; the caller ends the answer on its last
        # whole sentence instead of where the limit fell.
        _LOG.warning(
            "%s streamed synthesis stopped at finish_reason=length agent=%s max_tokens=%s",
            cfg["provider"], self.name, stream_max_tokens,
        )
        raise _SynthLengthCut(f"{cfg['provider']} output token limit reached")
