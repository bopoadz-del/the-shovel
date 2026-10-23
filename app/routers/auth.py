from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel

from app.dependencies import AUTH_AVAILABLE, get_auth_block, require_api_key, security

router = APIRouter()


# ── typed request bodies ───────────────────────────────────────────────────────

class KeyRequest(BaseModel):
    """Identifies an API key by either field name (callers use both)."""
    api_key: Optional[str] = None
    key: Optional[str] = None

    def resolved(self) -> Optional[str]:
        return self.api_key or self.key


class CheckPermissionRequest(KeyRequest):
    block: Optional[str] = None


class CreateKeyRequest(BaseModel):
    name: Optional[str] = None
    role: str = "basic"
    owner: Optional[str] = None


def _require_admin(auth_result: dict):
    """Ensure the authenticated key has admin role."""
    if auth_result.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")


def _require_own_key_or_admin(
    auth_result: dict,
    target_key: str,
    credentials: Optional[HTTPAuthorizationCredentials],
) -> None:
    """Inspect another credential only when admin, or when it is this caller.

    JWT users have no API-key identity — they cannot probe keys.
    A presented API key may inspect only itself.
    """
    if auth_result.get("role") == "admin":
        return
    presented = credentials.credentials if credentials is not None else None
    if auth_result.get("auth_method") == "jwt" or not presented or presented != target_key:
        raise HTTPException(status_code=403, detail="Admin role required")


@router.post("/v1/auth/validate")
async def validate_key(
    request: KeyRequest,
    auth: dict = Depends(require_api_key),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    """Validate an API key"""
    if not AUTH_AVAILABLE:
        raise HTTPException(status_code=503, detail="Auth not available")
    api_key = request.resolved()
    if not api_key:
        raise HTTPException(status_code=422, detail="api_key or key required")
    _require_own_key_or_admin(auth, api_key, credentials)
    block = get_auth_block()
    return await block.execute({"action": "validate", "api_key": api_key})


@router.post("/v1/auth/keys")
async def create_key(request: CreateKeyRequest, auth: dict = Depends(require_api_key)):
    """Create a new API key (admin only)"""
    _require_admin(auth)
    if not AUTH_AVAILABLE:
        raise HTTPException(status_code=503, detail="Auth not available")
    block = get_auth_block()
    return await block.execute(
        {"action": "create_key", **request.model_dump(exclude_none=True)}
    )


@router.delete("/v1/auth/keys/{api_key}")
async def delete_key(api_key: str, auth: dict = Depends(require_api_key)):
    """Delete (revoke) an API key by URL (admin only)"""
    _require_admin(auth)
    if not AUTH_AVAILABLE:
        raise HTTPException(status_code=503, detail="Auth not available")
    block = get_auth_block()
    return await block.execute({"action": "revoke_key", "api_key": api_key})


@router.get("/v1/auth/keys")
async def list_keys(admin_key: Optional[str] = None, auth: dict = Depends(require_api_key)):
    """List all API keys (admin only)"""
    _require_admin(auth)
    if not AUTH_AVAILABLE:
        raise HTTPException(status_code=503, detail="Auth not available")
    block = get_auth_block()
    return await block.execute({"action": "list_keys", "admin_key": admin_key})


@router.post("/v1/auth/keys/revoke")
async def revoke_key(request: KeyRequest, auth: dict = Depends(require_api_key)):
    """Revoke an API key (admin only)"""
    _require_admin(auth)
    if not AUTH_AVAILABLE:
        raise HTTPException(status_code=503, detail="Auth not available")
    api_key = request.resolved()
    if not api_key:
        raise HTTPException(status_code=422, detail="api_key or key required")
    block = get_auth_block()
    return await block.execute({"action": "revoke_key", "api_key": api_key})


@router.post("/v1/auth/keys/rotate")
async def rotate_key(request: KeyRequest, auth: dict = Depends(require_api_key)):
    """Rotate an API key (admin only)"""
    _require_admin(auth)
    if not AUTH_AVAILABLE:
        raise HTTPException(status_code=503, detail="Auth not available")
    api_key = request.resolved()
    if not api_key:
        raise HTTPException(status_code=422, detail="api_key or key required")
    block = get_auth_block()
    return await block.execute({"action": "rotate_key", "api_key": api_key})


@router.post("/v1/auth/check")
async def check_permission(
    request: CheckPermissionRequest,
    auth: dict = Depends(require_api_key),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    """Check if key has a permission"""
    if not AUTH_AVAILABLE:
        raise HTTPException(status_code=503, detail="Auth not available")
    api_key = request.resolved()
    if not api_key:
        raise HTTPException(status_code=422, detail="api_key or key required")
    _require_own_key_or_admin(auth, api_key, credentials)
    block = get_auth_block()
    return await block.execute({
        "action": "check_permission",
        "api_key": api_key,
        "block": request.block,
    })


@router.get("/v1/auth/usage")
async def get_usage(
    key: Optional[str] = None,
    api_key: Optional[str] = None,
    auth: dict = Depends(require_api_key),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    """Get usage stats for a key"""
    if not AUTH_AVAILABLE:
        raise HTTPException(status_code=503, detail="Auth not available")
    target = api_key or key
    if not target:
        raise HTTPException(status_code=422, detail="key or api_key query param required")
    _require_own_key_or_admin(auth, target, credentials)
    block = get_auth_block()
    return await block.execute({"action": "get_usage", "api_key": target})
