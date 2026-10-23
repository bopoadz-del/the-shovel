import os

from fastapi import APIRouter, Response
from fastapi.responses import FileResponse, HTMLResponse

from app.blocks import BLOCK_REGISTRY

router = APIRouter()

_REACT_INDEX = "frontend/dist/index.html"

# When the React bundle is absent (a fresh checkout or CI job that skipped
# `npm run build`), we serve this honest placeholder instead of silently
# falling back to a stale UI. The legacy vanilla-JS dashboard that used to
# live at app/static/index.html was retired — a missing build must fail
# VISIBLY, not quietly ship a different, ancient interface to users.
_BUILD_MISSING_HTML = (
    "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\">"
    "<title>The Fork</title></head><body>"
    "<h1>Frontend build not present</h1>"
    "<p>The React bundle (<code>frontend/dist</code>) has not been built. "
    "Run <code>npm --prefix frontend ci &amp;&amp; npm --prefix frontend run build</code>, "
    "or use the production Docker image which builds it automatically.</p>"
    "</body></html>"
)

# Path prefixes the SPA fallback must NOT shadow. Anything under these is
# either an API endpoint, a mounted StaticFiles directory, or a framework
# route (docs, openapi). Listed without leading slash because we match
# against full_path which FastAPI strips the leading slash from.
_RESERVED_PREFIXES: tuple[str, ...] = (
    "v1/", "api", "dashboard/", "assets/",
    "health", "docs", "redoc", "openapi.json", "mcp",
    # /debug/env must not fall through to the SPA shell. Production does
    # not mount the debug router; a 200 HTML 200 looked like a hole on
    # Windows and on live even though the body had no env payload.
    "debug",
)


def _serve_spa() -> Response:
    if os.path.isfile(_REACT_INDEX):
        return FileResponse(
            _REACT_INDEX,
            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
        )
    # Honest 200 placeholder (valid HTML) — never the old dashboard.
    return HTMLResponse(
        _BUILD_MISSING_HTML,
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


@router.get("/")
async def root():
    return _serve_spa()


@router.get("/api")
def api_info():
    return {
        "name": "Cerebrum Blocks",
        "version": "2.0.0",
        "tagline": "Build AI Like Lego",
        "blocks": len(BLOCK_REGISTRY),
        "endpoints": {
            "blocks": "/v1/blocks",
            "execute": "/v1/execute",
            "chain": "/v1/chain",
            "chat": "/v1/chat",
            "health": "/v1/health",
        },
    }


# SPA fallback: any GET that isn't an API/static path returns the React
# app's index.html so client-side routes like /login or /projects/:id
# resolve to the bundle and React Router takes over.
@router.get("/{full_path:path}")
async def spa_fallback(full_path: str):
    if any(full_path.startswith(p) for p in _RESERVED_PREFIXES):
        return Response(status_code=404)
    return _serve_spa()
