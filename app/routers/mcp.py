"""MCP server-sent-events endpoint at /mcp/sse.

Lets MCP-aware clients (Claude Desktop, MCP-compatible IDEs) connect to this
FastAPI app and discover every block in BLOCK_REGISTRY as a callable tool.

This module is import-safe even if the optional `mcp` package is not installed
or doesn't ship the SSE transport — see `mcp_router_available()`.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from app.dependencies import require_api_key, get_block_instance
from app.core.privileges import (
    raise_if_inaccessible_document_ids,
    raise_if_inaccessible_project_ids,
    raise_if_privileged_block,
)

router = APIRouter()


def _require_key_asgi(asgi_app):
    """Wrap an ASGI app so every HTTP request must carry a valid API key.

    Used to gate the /mcp/messages Mount: a Starlette Mount can't take a
    FastAPI `Depends`, so auth is enforced here, reusing the same
    `auth_manager.validate_key` that backs `require_api_key`.
    """
    from fastapi.security import HTTPAuthorizationCredentials
    from app.core.auth import auth as auth_manager

    async def _gated(scope, receive, send):
        if scope.get("type") != "http":
            await asgi_app(scope, receive, send)
            return
        header = ""
        for key, value in scope.get("headers") or []:
            if key == b"authorization":
                header = value.decode("latin-1")
                break
        creds = None
        if header.lower().startswith("bearer "):
            creds = HTTPAuthorizationCredentials(
                scheme="Bearer", credentials=header[7:])
        try:
            auth_manager.validate_key(creds)
        except HTTPException as exc:
            await JSONResponse(
                {"detail": exc.detail}, status_code=exc.status_code,
            )(scope, receive, send)
            return
        await asgi_app(scope, receive, send)

    return _gated


def mcp_router_available() -> bool:
    try:
        from mcp.server import Server  # noqa: F401
        from mcp.server.sse import SseServerTransport  # noqa: F401
        return True
    except Exception:
        return False


@router.get("/mcp/info")
async def mcp_info(auth: dict = Depends(require_api_key)):
    """Lightweight JSON describing the MCP surface — works even without SSE deps."""
    from app.blocks import BLOCK_REGISTRY

    return {
        "available": mcp_router_available(),
        "tool_count": len(BLOCK_REGISTRY),
        "tools": sorted(BLOCK_REGISTRY.keys()),
        "endpoints": {
            "sse": "/mcp/sse" if mcp_router_available() else None,
            "info": "/mcp/info",
        },
    }


if mcp_router_available():
    from mcp.server import Server
    from mcp.server.sse import SseServerTransport
    from mcp.types import TextContent, Tool

    _sse = SseServerTransport("/mcp/messages")

    def mount_message_endpoint(app) -> bool:
        """Mount the POST side of the MCP SSE transport on the FastAPI app.

        Call this on the app itself (from main.py), NOT the APIRouter:
        FastAPI's include_router does not propagate Starlette Mount routes, so
        a Mount added to this router would be silently dropped. /mcp/sse sends
        the client an `endpoint` event pointing at
        /mcp/messages?session_id=...; without this mount every JSON-RPC POST
        there 404s and the `initialize` handshake never completes.
        """
        from starlette.routing import Mount
        app.router.routes.append(
            Mount("/mcp/messages",
                  app=_require_key_asgi(_sse.handle_post_message))
        )
        return True

    def _build_server(auth: Optional[dict] = None) -> "Server":
        from app.blocks import BLOCK_REGISTRY

        role = (auth or {}).get("role")
        server = Server("cerebrum-blocks")

        @server.list_tools()
        async def _list_tools():
            # Single source of truth: delegate to the mcp_adapter catalog so the
            # SSE surface and the mcp_adapter block never drift (and inherit its
            # per-action input schemas). _build_tool_catalog already skips
            # mcp_adapter itself.
            from app.blocks.mcp_adapter import MCPAdapterBlock

            return [
                Tool(
                    name=t["name"],
                    description=t["description"],
                    inputSchema=t["inputSchema"],
                )
                for t in MCPAdapterBlock._build_tool_catalog()
            ]

        @server.call_tool()
        async def _call_tool(name: str, arguments: dict):
            raise_if_privileged_block(name, role)
            raise_if_inaccessible_project_ids(auth, arguments)
            raise_if_inaccessible_document_ids(auth, arguments)
            if name not in BLOCK_REGISTRY:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]
            instance = get_block_instance(name)
            block_input = (arguments or {}).get("input")
            block_params = (arguments or {}).get("params") or {}
            result = await instance.execute(block_input, block_params)
            import json
            return [TextContent(type="text", text=json.dumps(result, default=str))]

        return server

    @router.get("/mcp/sse")
    async def mcp_sse(request: Request,
                      auth: dict = Depends(require_api_key)):
        async with _sse.connect_sse(request.scope, request.receive, request._send) as streams:
            server = _build_server(auth)
            await server.run(streams[0], streams[1], server.create_initialization_options())

else:

    def mount_message_endpoint(app) -> bool:
        """No-op — the MCP SSE transport is unavailable, nothing to mount."""
        return False

    @router.get("/mcp/sse")
    async def mcp_sse_unavailable(auth: dict = Depends(require_api_key)):
        return JSONResponse(
            status_code=503,
            content={"detail": "MCP SSE transport not available — install 'mcp[server]'."},
        )
