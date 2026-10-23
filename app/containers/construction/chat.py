"""Construction container — chat submodule."""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class ConstructionChatMixin:
    async def chat(self, input_data: Any, params: Dict = None) -> Dict:
        """Delegate a chat turn to ChatBlock with the Construction EVM
        system prompt pre-injected.

        The container owns the policy (which prompt file to use); ChatBlock
        owns the mechanics (loading the file, building the messages list,
        calling the provider). When the caller already supplies either a
        literal ``system_prompt`` or a ``system_prompt_file`` — via params
        OR input_data — we do NOT override it; the caller wins.

        All other params (``stream``, ``model``, ``max_tokens``,
        ``temperature``, ``project_id``, ``use_rag``, ``rag_k``,
        ``use_local_model``, ...) are forwarded to ChatBlock unchanged.

        Returns ChatBlock.process()'s result dict as-is (status / text /
        provider / model / tokens / ...).
        """
        chat_block = self._resolve_block("chat")
        if chat_block is None:
            return {"status": "error", "error": "chat block unavailable"}

        merged = dict(params or {})
        data = input_data if isinstance(input_data, dict) else {}
        # RAG default: ON. The chat block will retrieve from this
        # project's index unless the caller has explicitly set use_rag.
        if "use_rag" not in merged and not (isinstance(input_data, dict)
                                            and "use_rag" in input_data):
            merged["use_rag"] = True
        caller_supplied_prompt = (
            merged.get("system_prompt")
            or merged.get("system_prompt_file")
            or data.get("system_prompt")
            or data.get("system_prompt_file")
        )
        if not caller_supplied_prompt:
            merged["system_prompt_file"] = "construction_evm.md"

        return await chat_block.process(input_data, merged)
