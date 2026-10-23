"""MCP consumer — call any external MCP server from inside a Cerebrum chain.

Examples:
    {"server": "github", "tool": "create_issue", "params": {"repo": "...", "title": "..."}}
    {"server": "filesystem", "tool": "read_file", "params": {"path": "..."}}

The block spawns the configured MCP server via stdio (`npx -y @modelcontextprotocol/server-<name>`
by default), opens a ClientSession, calls the requested tool, and returns the result.
"""

import asyncio
import os
from typing import Any, Dict

from app.core.universal_base import UniversalBlock
import logging

logger = logging.getLogger(__name__)

# Servers the platform does not ship. Calling them used to `npx -y` a
# community package with no timeout and hang the chat turn (weather).
_UNCONFIGURED_MCP_SERVERS = frozenset({"weather", "geocode", "currency"})
_DEFAULT_MCP_TIMEOUT_SECONDS = 8.0


def _mcp_timeout_seconds() -> float:
    raw = (os.getenv("MCP_CONSUMER_TIMEOUT_SECONDS") or "").strip()
    if not raw:
        return _DEFAULT_MCP_TIMEOUT_SECONDS
    try:
        return max(0.5, float(raw))
    except ValueError:
        return _DEFAULT_MCP_TIMEOUT_SECONDS


class MCPConsumerBlock(UniversalBlock):
    name = "mcp_consumer"
    version = "1.0.0"
    description = "Call any external MCP server (github, slack, stripe, sentry, ...) from a chain"
    layer = 0
    tags = ["mcp", "agents", "interop", "external"]

    ui_schema = {
        "input": {"type": "json", "placeholder": '{"server": "github", "tool": "create_issue", "params": {...}}'},
        "output": {"type": "json", "fields": [{"name": "result", "type": "json", "label": "Tool result"}]},
        "quick_actions": [
            {"icon": "🐙", "label": "List GitHub repos", "prompt": '{"server":"github","tool":"list_repos","params":{}}'},
        ],
    }

    async def process(self, input_data: Any, params: Dict = None) -> Dict:
        params = params or {}
        data = input_data if isinstance(input_data, dict) else {}

        server_name = data.get("server") or params.get("server")
        tool_name = data.get("tool") or params.get("tool")
        tool_params = data.get("params") or params.get("params") or {}

        if not server_name or not tool_name:
            return {"status": "error", "error": "Provide 'server' and 'tool'."}

        custom_command = data.get("command") or params.get("command")
        if (
            str(server_name).strip().lower() in _UNCONFIGURED_MCP_SERVERS
            and not custom_command
        ):
            return {
                "status": "error",
                "server": server_name,
                "tool": tool_name,
                "error": (
                    f"MCP server '{server_name}' is not configured on this "
                    "platform. Do not spawn npx. Tell the user it is unavailable."
                ),
            }

        # Allow custom command/args for self-hosted MCP servers
        command = custom_command or "npx"
        args = data.get("args") or params.get("args")
        if args is None:
            args = ["-y", f"@modelcontextprotocol/server-{server_name}"]

        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as e:
            return {"status": "error", "error": f"MCP package not installed: {e}"}

        timeout = _mcp_timeout_seconds()

        async def _invoke():
            server_params = StdioServerParameters(command=command, args=args)
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    return await session.call_tool(tool_name, tool_params)

        try:
            result = await asyncio.wait_for(_invoke(), timeout=timeout)
            return {
                "status": "success",
                "server": server_name,
                "tool": tool_name,
                "result": _serialize(result),
            }
        except asyncio.TimeoutError:
            return {
                "status": "error",
                "server": server_name,
                "tool": tool_name,
                "error": (
                    f"MCP server '{server_name}' timed out after {timeout:.0f}s "
                    "and was not spawned further."
                ),
            }
        except Exception as e:
            return {"status": "error", "server": server_name, "tool": tool_name, "error": str(e)}


def _serialize(obj):
    """Best-effort JSON-friendly serialization of an MCP tool result."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_serialize(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            logger.debug(
                "swallowed %s in _serialize() — continuing",
                "Exception", exc_info=True,
            )
    if hasattr(obj, "__dict__"):
        return {k: _serialize(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
    return str(obj)
