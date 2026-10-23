"""Dynamic block/tool catalogue — Reasoning Engine dynamic catalogue awareness.

Thin shim over the existing block registry and the MCP adapter's tool-catalog
builder.  The reasoner (and any other planner) can call this to discover the
blocks available at runtime instead of hard-coding a tool list.
"""

from typing import Any, Dict, List, Optional


def list_blocks() -> List[str]:
    """Return the names of all blocks currently registered."""
    from app.blocks import BLOCK_REGISTRY
    return list(BLOCK_REGISTRY.keys())


def get_block(name: str):
    """Return the block class registered under ``name``, or ``None``."""
    from app.blocks import get_block as _get_block
    return _get_block(name)


def build_tool_catalogue() -> List[Dict[str, Any]]:
    """Return an MCP-style catalog of every registered block.

    Delegates to the existing ``MCPAdapterBlock._build_tool_catalog`` so the
    shape stays identical to what the ``/mcp/sse`` endpoint already exposes.
    """
    from app.blocks.mcp_adapter import MCPAdapterBlock
    return MCPAdapterBlock._build_tool_catalog()


def format_catalogue_for_prompt(
    catalogue: Optional[List[Dict[str, Any]]] = None,
    max_description_len: int = 120,
) -> str:
    """Render a compact, LLM-friendly block catalogue for a system prompt.

    If ``catalogue`` is omitted, the current runtime catalogue is fetched
    dynamically via :func:`build_tool_catalogue`.
    """
    if catalogue is None:
        catalogue = build_tool_catalogue()
    if not catalogue:
        return "AVAILABLE BLOCKS: (none registered)"

    lines = ["AVAILABLE BLOCKS (auto-generated from the block catalogue):"]
    for tool in catalogue:
        name = tool.get("name", "unknown")
        desc = (tool.get("description") or "").strip()
        if len(desc) > max_description_len:
            desc = desc[:max_description_len].rstrip() + "..."
        lines.append(f"  - {name}: {desc}")
    return "\n".join(lines)
