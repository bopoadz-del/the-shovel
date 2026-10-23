"""Chat Block — active cloud provider (DeepSeek/OpenRouter via _llm_config).

The chat must never go completely dark on the user. Order of attempts:

1. **Local fine-tuned model** (opt-in ``use_local_model``) — the LoRA-tuned
   model from the learning subsystem, when installed. Off by default.
2. **Active cloud provider** (DeepSeek primary / OpenRouter fallback, via
   ``_llm_config`` and ``_llm_fallback_ladder``). A primary 402 with
   fallback_ready hops; it must not become the offline template.
3. **Graceful template responder** — a deterministic, non-AI fallback that
   acknowledges the question, surfaces the reason the model layer is down,
   and points the operator at the env vars that would restore it. This
   path always succeeds, so the chat never returns an unhandled error.

The block exposes a single ``provider`` field on the response so callers can
see which path served the answer (``deepseek`` / ``openrouter`` /
``local_lora`` / ``offline_template``).
"""

import json
import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
from app.core.typed_block import TypedBlock, Schema, ContentType


def _shaped_cloud_payload(
    cfg: Dict[str, Any],
    *,
    model: str,
    messages: list,
    max_tokens: int,
    temperature: float,
    stream: bool = False,
) -> Dict[str, Any]:
    """OpenAI-shape body with per-provider temperature / max_tokens pins.

    Moonshot's K2 reasoning models 400 on any temperature but 1. Agent runtime
    and ``llm_client.complete`` already honor ``cfg['fixed_temperature']``;
    ChatBlock used to send its default 0.7 and fall through to the offline
    template. Shape here so both the stream and non-stream branches agree.
    """
    from app.agents.runtime import _provider_max_tokens, _provider_temperature

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": _provider_max_tokens(cfg, int(max_tokens)),
        "temperature": _provider_temperature(cfg, float(temperature)),
    }
    if stream:
        payload["stream"] = True
    return payload


class ChatBlock(TypedBlock):
    """AI chat completions — active cloud provider (DeepSeek/OpenRouter)."""

    auto_validate = False
    name = "chat"
    version = "3.0.0"
    description = "AI chat completions — active cloud provider, local LLM fallback"
    layer = 2
    tags = ["ai", "core", "llm", "chat", "typed"]
    requires = []

    default_config = {
        "default_provider": "deepseek",
        "max_tokens": 2048,
        "temperature": 0.7,
    }

    accepted_input_types = ["Text", "TextContent", "ChatMessage"]
    produced_output_types = ["Text", "TextContent", "ChatMessage"]
    required_input_one_of = ["text", "message"]

    # Canonical text key for chain unwrapping. ChatBlock returns either
    # ``{"text": "..."}`` (cloud path) or ``{"response": "..."}`` (local
    # LoRA path); ``"text"`` is the primary shape, so declaring it locks
    # the contract for the common case. The orchestrator's global fallback
    # still picks up ``"response"`` on the local path because it's in
    # _TEXT_OUTPUT_FIELDS, so no behaviour change either way.
    text_output_field = "text"

    input_schema = Schema(
        content_type=ContentType.TEXT,
        required_fields=[],
        optional_fields=["text", "message", "context"],
        format_hints={"max_length": 100000},
    )

    output_schema = Schema(
        content_type=ContentType.TEXT,
        required_fields=["text"],
        optional_fields=["provider", "model", "tokens", "status"],
        format_hints={},
    )

    ui_schema = {
        "input": {
            "type": "text",
            "accept": None,
            "placeholder": "Ask anything...",
            "multiline": True,
        },
        "output": {
            "type": "text",
            "fields": [
                {"name": "text", "type": "markdown", "label": "Response"},
            ],
        },
        "quick_actions": [
            {"icon": "💡", "label": "Explain", "prompt": "Explain this in simple terms"},
            {"icon": "📝", "label": "Summarize", "prompt": "Summarize the key points"},
        ],
    }

    async def process(self, input_data: Any, params: Dict = None) -> Dict:
        params = params or {}
        if isinstance(input_data, dict):
            message = (
                input_data.get("text")
                or input_data.get("content")
                or input_data.get("extracted_text")
                or str(input_data)
            )
        else:
            message = str(input_data)

        max_tokens = params.get("max_tokens", self.config.get("max_tokens", 2048))
        temperature = params.get("temperature", self.config.get("temperature", 0.7))
        stream = params.get("stream", False)
        model = params.get("model")  # None -> cfg["default_model"] (active provider)

        # System prompt is container/caller-owned — ChatBlock does not inject
        # domain defaults. ConstructionContainer.chat() sets construction_evm.md.

        # ── System prompt resolution — optional, never aborts the chat ─────
        # Accepts either a literal `system_prompt` string or a
        # `system_prompt_file` that names a file inside app/prompts/.
        # Literal wins if both are supplied. Missing file or path-traversal
        # attempt: log a warning and continue with NO system prompt.
        system_prompt_text = self._resolve_system_prompt(input_data, params)

        # ── RAG (PR 2) — strictly opt-in. Defaults preserve existing behavior.
        # When use_rag=True AND project_id is provided AND the embedding
        # stack is installed, retrieve top-k chunks and prepend them as
        # context. Failures here never abort the chat call — retrieval is
        # best-effort enrichment, not a hard dependency.
        use_rag = bool(
            params.get("use_rag")
            or (isinstance(input_data, dict) and input_data.get("use_rag"))
        )
        rag_project_id = (
            params.get("project_id")
            or (isinstance(input_data, dict) and input_data.get("project_id"))
        )
        if use_rag and rag_project_id:
            try:
                from app.core.rag.retriever import retrieve as _retrieve
                rag_k = int(params.get("rag_k", 5))
                chunks = _retrieve(message, str(rag_project_id), k=rag_k)
                if chunks:
                    context = "\n\n".join(
                        f"[{c.doc_id}#{c.chunk_index}] {c.text}" for c in chunks
                    )
                    message = (
                        f"Relevant project context:\n{context}\n\n"
                        f"---\n\nUser question: {message}"
                    )
            except Exception as exc:  # noqa: BLE001
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "RAG retrieval failed for project %s: %s", rag_project_id, exc
                )

        # ── Local fine-tuned model (PR 3a) — opt-in. Defaults preserve
        # existing cloud-first behavior. When use_local_model=true AND the
        # local stack is available (deps installed + adapter present), the
        # generation runs through the LoRA-tuned model and bypasses the
        # cloud provider entirely. Failures fall through to the existing
        # cloud chain — chat never goes dark because of a local-model bug.
        use_local_model = bool(
            params.get("use_local_model")
            or (isinstance(input_data, dict) and input_data.get("use_local_model"))
        )
        if use_local_model:
            from app.core.learning import local_model as _local

            if _local.available():
                local_text = _local.generate(
                    message,
                    max_new_tokens=int(max_tokens),
                    temperature=float(temperature),
                )
                if local_text:
                    return {
                        "status": "success",
                        "response": local_text,
                        "provider": "local_lora",
                        "model": os.getenv("LOCAL_BASE_MODEL") or "Qwen/Qwen2.5-3B-Instruct",
                        "adapter": os.getenv("LOCAL_ADAPTER_DIR") or "data/learning/adapters/construction_v1",
                    }
                # generate() returned None — fall through to cloud chain
                import logging as _logging
                _logging.getLogger(__name__).info(
                    "use_local_model requested but generate() returned None; falling back to cloud"
                )
            else:
                import logging as _logging
                _logging.getLogger(__name__).info(
                    "use_local_model requested but local stack unavailable; falling back to cloud"
                )

        # ── Cloud provider selection via _llm_config() (same as the agent
        # runtime) so LLM_PROVIDER=deepseek|openrouter applies uniformly across
        # the chat block route and the agent path. Walk the same fallback
        # ladder ``_call_llm`` / ``llm_client.complete`` use: a DeepSeek 402
        # with fallback_ready must not become the offline template.
        from app.agents.runtime import (  # local import: avoid cycle at module load
            _llm_config,
            _llm_fallback_ladder,
            _resolve_attempt_model,
        )
        cfg = _llm_config()
        attempts: list[tuple[dict[str, Any], str]] = [
            (cfg, os.getenv(cfg["env_key"]) or "" if cfg.get("env_key") else ""),
        ]
        for fb in _llm_fallback_ladder(cfg):
            attempts.append(
                (fb, os.getenv(fb["env_key"]) or "" if fb.get("env_key") else ""),
            )
        extra_kwargs = {"system_prompt": system_prompt_text} if system_prompt_text else {}
        primary_error = None
        for a_cfg, a_key in attempts:
            if a_cfg.get("env_key") and not a_key:
                primary_error = f"{a_cfg['env_key']} not configured"
                continue
            # Remap leftover ChatRequest defaults (deepseek-chat / kimi-k2.6)
            # the same way the agent path does, or the OpenRouter hop 404s.
            effective_model = _resolve_attempt_model(a_cfg, model)
            result = await self._call_cloud(
                message, effective_model, max_tokens, temperature, stream,
                a_key, a_cfg,
                **extra_kwargs,
            )
            if result.get("status") == "success":
                return result
            primary_error = result.get("error", f"{a_cfg['provider']} call failed")

        # ── Graceful template — chat must not go dark ──────────────────────
        return self._offline_template(
            message, primary_error or "no language model is reachable",
        )

    # ────────────────────────────────────────────────────────────────────────
    # System prompt resolution + message-list construction
    # ────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_messages(message: str, system_prompt: Optional[str]) -> list:
        """Build the OAI-shape messages array, optionally prepending a
        system message. A literal-empty / whitespace-only system_prompt is
        treated as absent so callers can pass any truthy string without
        worrying about producing a stray empty system role."""
        msgs = []
        if system_prompt and system_prompt.strip():
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": message})
        return msgs

    def _resolve_system_prompt(
        self, input_data: Any, params: Dict
    ) -> Optional[str]:
        """Resolve an optional system prompt from input_data or params.

        Precedence:
          1. Literal ``system_prompt`` string (input_data first, then params).
          2. ``system_prompt_file`` naming a file inside ``app/prompts/``.

        Failure modes (missing file, path-traversal, read error) log a
        warning and return None — chat must not go dark over a bad prompt
        reference.
        """
        # 1. Literal — wins outright if non-empty.
        literal = None
        if isinstance(input_data, dict):
            literal = input_data.get("system_prompt")
        if not literal:
            literal = params.get("system_prompt")
        if isinstance(literal, str) and literal.strip():
            return literal

        # 2. Filename — must resolve inside app/prompts/.
        fname = None
        if isinstance(input_data, dict):
            fname = input_data.get("system_prompt_file")
        if not fname:
            fname = params.get("system_prompt_file")
        if not fname:
            return None
        if not isinstance(fname, str):
            return None

        log = logging.getLogger(__name__)
        prompts_dir = (Path(__file__).parent.parent / "prompts").resolve()
        try:
            candidate = (prompts_dir / fname).resolve()
        except (OSError, ValueError) as exc:
            log.warning("system_prompt_file %r could not be resolved: %s", fname, exc)
            return None

        # Reject anything that escapes prompts_dir (path traversal, absolute
        # path that resolves elsewhere, etc.).
        try:
            inside = candidate.is_relative_to(prompts_dir)
        except AttributeError:
            # Python < 3.9 fallback — current target is 3.11+, kept defensively.
            inside = str(candidate).startswith(str(prompts_dir) + os.sep)
        if not inside or candidate == prompts_dir:
            log.warning(
                "system_prompt_file %r rejected — resolves outside app/prompts/",
                fname,
            )
            return None
        if not candidate.is_file():
            log.warning(
                "system_prompt_file %r not found at %s", fname, candidate,
            )
            return None
        try:
            return candidate.read_text(encoding="utf-8")
        except OSError as exc:
            log.warning(
                "system_prompt_file %r failed to read: %s", fname, exc,
            )
            return None

    # ────────────────────────────────────────────────────────────────────────
    # Cloud provider — chat completions (DeepSeek / OpenRouter, OAI-shape)
    # ────────────────────────────────────────────────────────────────────────

    async def _call_cloud(
        self,
        message: str,
        model: str,
        max_tokens: int,
        temperature: float,
        stream: bool,
        api_key: str,
        cfg: Dict[str, Any],
        system_prompt: Optional[str] = None,
    ) -> Dict:
        url = cfg["url"]
        provider_name = cfg["provider"]
        messages = self._build_messages(message, system_prompt)
        payload = _shaped_cloud_payload(
            cfg,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
        )
        if stream:
            async def _stream_generator():
                async with httpx.AsyncClient(timeout=60.0) as client:
                    # Empty-key guard: passing an empty Bearer header makes
                    # httpx raise "Illegal header value b'Bearer '" before the
                    # request even leaves the client. Omit the header entirely
                    # when api_key is empty (mirrors the agent runtime).
                    cloud_headers = {"Content-Type": "application/json"}
                    if api_key:
                        cloud_headers["Authorization"] = f"Bearer {api_key}"
                    async with client.stream(
                        "POST",
                        url,
                        headers=cloud_headers,
                        json=payload,
                    ) as response:
                        if response.status_code != 200:
                            err = await response.aread()
                            yield json.dumps({
                                "type": "error",
                                "message": f"{provider_name} error {response.status_code}: {err[:200]}",
                            })
                            return
                        async for line in response.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            data = line[6:]
                            if data == "[DONE]":
                                continue
                            try:
                                chunk = json.loads(data)
                                content = chunk["choices"][0].get("delta", {}).get("content", "")
                                if content:
                                    yield content
                            except Exception:
                                continue

            return {
                "status": "success",
                "text": "",
                "provider": provider_name,
                "model": model,
                "stream": _stream_generator(),
            }

        try:
            # Same empty-key guard as the stream branch above.
            cloud_headers = {"Content-Type": "application/json"}
            if api_key:
                cloud_headers["Authorization"] = f"Bearer {api_key}"
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    url,
                    headers=cloud_headers,
                    json=payload,
                )
                if response.status_code != 200:
                    return {
                        "status": "error",
                        "error": f"{provider_name} API error (HTTP {response.status_code}): {response.text[:200]}",
                    }
                data = response.json()
                return {
                    "status": "success",
                    "text": data["choices"][0]["message"]["content"],
                    "provider": provider_name,
                    "model": model,
                    "tokens": data.get("usage", {}),
                }
        except httpx.TimeoutException:
            return {"status": "error", "error": f"{provider_name} request timed out"}
        except Exception as e:
            return {"status": "error", "error": f"{provider_name} failed: {e}"}

    # ────────────────────────────────────────────────────────────────────────
    # Graceful offline template — last-resort: chat never goes dark
    # ────────────────────────────────────────────────────────────────────────

    def _offline_template(self, message: str, primary_error: str) -> Dict:
        snippet = (message or "").strip()
        if len(snippet) > 240:
            snippet = snippet[:237] + "..."
        body = (
            "**Chat is running in offline mode.**\n\n"
            "No language model is currently reachable, so I can't "
            "generate an AI response right now. Your message was received intact:\n\n"
            f"> {snippet or '(empty)'}\n\n"
            "**How to restore full chat:**\n"
            "- Set `DEEPSEEK_API_KEY` (or `OPENROUTER_API_KEY`) in `.env` to use a cloud provider.\n\n"
            f"_Primary provider: {primary_error}_"
        )
        return {
            "status": "success",
            "text": body,
            "provider": "offline_template",
            "model": "template:v1",
            "primary_error": primary_error,
        }
