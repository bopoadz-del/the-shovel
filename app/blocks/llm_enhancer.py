"""LLM Enhancer Block - AI text extraction and structuring using chat block."""

import json
import os
from typing import Any, Dict, List, Optional
from app.core.universal_base import UniversalBlock


class LLMEnhancerBlock(UniversalBlock):
    """Enhances raw text/data using LLM-powered extraction and structuring."""

    name = "llm_enhancer"
    version = "1.0.0"
    description = "AI text extraction and structuring using underlying chat block"
    layer = 2
    tags = ["ai", "core", "llm", "enhancer"]
    requires = ["chat"]
    required_input_one_of = ["text", "input", "message"]
    skip_input_validation_actions = ["health_check"]

    default_config = {
        "default_provider": "kimi",
        "max_tokens": 2048,
        "temperature": 0.3
    }

    ui_schema = {
        "input": {
            "type": "text",
            "accept": None,
            "placeholder": "Paste raw text to extract structure from...",
            "multiline": True
        },
        "output": {
            "type": "json",
            "fields": [
                {"name": "structured_data", "type": "json", "label": "Structured Output"}
            ]
        },
        "quick_actions": [
            {"icon": "🔎", "label": "Extract Data", "prompt": "Extract all structured data from this text"},
            {"icon": "📝", "label": "Summarize", "prompt": "Summarize the key information"},
            {"icon": "🏷️", "label": "Classify", "prompt": "Classify and label the type of this content"}
        ]
    }

    async def process(self, input_data: Any, params: Dict = None) -> Dict:
        """Route to appropriate enhancer action."""
        params = params or {}
        action = params.get("action", "extract")
        handlers = {
            "extract": self.extract,
            "summarize": self.summarize,
            "classify": self.classify,
            "translate": self.translate,
            "health_check": self.health_check,
        }
        handler = handlers.get(action)
        if not handler:
            return {"status": "error", "error": f"Unknown action: {action}"}
        return await handler(input_data, params)

    async def extract(self, input_data: Any, params: Dict) -> Dict:
        """Extract structured entities from raw text."""
        text = input_data if isinstance(input_data, str) else str(input_data)
        schema = params.get("schema", "key-value pairs")
        prompt = (
            f"Extract structured information from the following text "
            f"as {schema}. Return ONLY valid JSON.\n\n{text}"
        )
        return await self._call_llm(prompt, params)

    async def summarize(self, input_data: Any, params: Dict) -> Dict:
        """Summarize text to bullet points or a paragraph."""
        text = input_data if isinstance(input_data, str) else str(input_data)
        style = params.get("style", "concise paragraph")
        prompt = f"Summarize the following text as a {style}:\n\n{text}"
        return await self._call_llm(prompt, params)

    async def classify(self, input_data: Any, params: Dict) -> Dict:
        """Classify text into provided categories."""
        text = input_data if isinstance(input_data, str) else str(input_data)
        categories = params.get("categories", ["general"])
        prompt = (
            f"Classify the following text into one of these categories: {categories}. "
            f"Return ONLY the category name.\n\n{text}"
        )
        return await self._call_llm(prompt, params)

    async def translate(self, input_data: Any, params: Dict) -> Dict:
        """Translate text to target language."""
        text = input_data if isinstance(input_data, str) else str(input_data)
        target_lang = params.get("target_language", "English")
        prompt = f"Translate the following text to {target_lang}:\n\n{text}"
        return await self._call_llm(prompt, params)

    async def health_check(self, input_data: Any = None, params: Dict = None) -> Dict:
        """Health check for enhancer block."""
        chat = self.get_dep("chat")
        return {
            "status": "success",
            "block": self.name,
            "version": self.version,
            "chat_block_available": chat is not None
        }

    async def _call_llm(self, prompt: str, params: Dict) -> Dict:
        """Call underlying chat block with prompt."""
        chat = self.get_dep("chat")
        if not chat:
            # Fallback: try BLOCK_REGISTRY
            try:
                from app.blocks import BLOCK_REGISTRY
                chat = BLOCK_REGISTRY.get("chat")()
            except Exception:
                return {"status": "error", "error": "Chat block not available for LLM enhancement"}

        try:
            result = await chat.execute(prompt, {
                "model": params.get("model", self.config.get("default_provider", "kimi-k2.6")),
                "max_tokens": params.get("max_tokens", self.config.get("max_tokens", 2048)),
                "temperature": params.get("temperature", self.config.get("temperature", 0.3)),
                "stream": False
            })
            raw_text = result.get("result", {}).get("text", "")
            model_used = params.get("model", self.config.get("default_provider", "kimi-k2.6"))
            # The extract prompt says "Return ONLY valid JSON". Try to parse it
            # so callers get a real dict/list, not a JSON-shaped string. Fall
            # back to raw text with a warning on parse failure.
            try:
                structured = json.loads(raw_text)
                return {
                    "status": "success",
                    "structured_data": structured,
                    "raw_text": raw_text,
                    "model_used": model_used,
                }
            except (json.JSONDecodeError, TypeError):
                return {
                    "status": "success",
                    "structured_data": None,
                    "raw_text": raw_text,
                    "warning": "LLM returned non-JSON; structured_data is None",
                    "model_used": model_used,
                }
        except Exception as e:
            return {"status": "error", "error": f"LLM enhancement failed: {str(e)}"}

    def get_actions(self) -> Dict[str, Any]:
        """Return all public methods for block registry."""
        return {
            "extract": self.extract,
            "summarize": self.summarize,
            "classify": self.classify,
            "translate": self.translate,
            "health_check": self.health_check,
        }
