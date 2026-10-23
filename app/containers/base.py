"""Domain container host — kit entry point for Cerebrum Block Store installs.

Virgin Fork ships this module plus ~17 generic blocks. Domain kits (e.g.
construction) install container classes that inherit ``DomainContainer`` and
register at boot via ``app.core.domain_kit_loader``.
"""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from app.core.universal_base import UniversalContainer


class DomainContainer(UniversalContainer):
    """Base class for store-published domain kits.

    Subclasses declare prompt paths, optional knowledge facades, and action
    maps. ``chat()`` injects ``system_prompt_file`` unless the caller overrides.
    """

    name: str = ""
    description: str = ""
    version: str = "1.0.0"
    system_prompt_file: str = ""
    knowledge_class: type | None = None
    kit_root: Path | None = None

    def resolve_prompt(self, filename: str | None = None) -> str:
        """Load prompt text from kit prompts directory or app/prompts/."""
        fname = filename or self.system_prompt_file
        if not fname:
            return ""
        candidates: list[Path] = []
        if self.kit_root:
            candidates.append(self.kit_root / "prompts" / fname)
            candidates.append(self.kit_root / "app" / "prompts" / fname)
        app_root = Path(__file__).resolve().parents[1]
        candidates.append(app_root / "prompts" / fname)
        for path in candidates:
            if path.is_file():
                return path.read_text(encoding="utf-8")
        return ""

    def _resolve_block(self, name: str):
        from app.blocks import BLOCK_REGISTRY

        block_cls = BLOCK_REGISTRY.get(name)
        if block_cls is None:
            return None
        from app.dependencies import get_block_instance

        return get_block_instance(name)

    async def chat(self, input_data: Any, params: Dict | None = None) -> Dict:
        """Delegate to ChatBlock with kit default prompt unless caller overrides."""
        chat_block = self._resolve_block("chat")
        if chat_block is None:
            return {"status": "error", "error": "chat block unavailable"}

        merged = dict(params or {})
        data = input_data if isinstance(input_data, dict) else {}
        if "use_rag" not in merged and not (
            isinstance(input_data, dict) and "use_rag" in input_data
        ):
            merged["use_rag"] = True
        caller_supplied_prompt = (
            merged.get("system_prompt")
            or merged.get("system_prompt_file")
            or data.get("system_prompt")
            or data.get("system_prompt_file")
        )
        if not caller_supplied_prompt and self.system_prompt_file:
            merged["system_prompt_file"] = self.system_prompt_file

        return await chat_block.process(input_data, merged)

    @abstractmethod
    def get_actions(self) -> Dict[str, Callable]:
        """Return action name → handler map for ``route()``."""

    async def route(self, action: str, input_data: Any, params: Dict) -> Dict:
        handler = self.get_actions().get(action)
        if handler is None:
            return {"status": "error", "error": f"Action '{action}' not implemented"}
        return await handler(input_data, params)
