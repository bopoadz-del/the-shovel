"""Smoke tests for the MCP adapter + consumer blocks."""

import pytest

from app.blocks import BLOCK_REGISTRY
from tests.conftest import is_construction_kit_enabled, is_extended_boot

pytestmark = [
    pytest.mark.extended_boot,
    pytest.mark.skipif(not is_extended_boot(), reason="requires CEREBRUM_VIRGIN=false"),
]


@pytest.mark.construction_kit
@pytest.mark.skipif(not is_construction_kit_enabled(), reason="requires CEREBRUM_DOMAIN_KITS=construction")
@pytest.mark.asyncio
async def test_mcp_adapter_lists_every_block_except_itself():
    block = BLOCK_REGISTRY["mcp_adapter"]()
    result = await block.execute({"action": "list_tools"}, {})
    assert result["status"] == "success"
    inner = result.get("result", result)
    tools = inner.get("tools") if isinstance(inner, dict) and "tools" in inner else result.get("tools", [])
    names = {t["name"] for t in tools}
    # Must surface a representative block from each category and skip itself
    assert "chat" in names
    assert "construction" in names
    assert "mcp_adapter" not in names


@pytest.mark.asyncio
async def test_mcp_adapter_describe_unknown_tool():
    block = BLOCK_REGISTRY["mcp_adapter"]()
    result = await block.execute({}, {"action": "describe", "tool": "definitely_not_a_block"})
    inner = result.get("result", result)
    assert inner.get("status") == "error"


@pytest.mark.asyncio
async def test_mcp_consumer_requires_server_and_tool():
    block = BLOCK_REGISTRY["mcp_consumer"]()
    result = await block.execute({}, {})
    inner = result.get("result", result)
    assert inner.get("status") == "error"
    assert "server" in inner.get("error", "").lower()
