from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.dependencies import MEMORY_AVAILABLE, get_memory_block, require_api_key

router = APIRouter()

_MEMORY_ACTIONS = {"get", "set", "delete", "flush", "keys", "exists"}


class MemoryOpRequest(BaseModel):
    """Typed body for a memory cache operation. Only these fields reach the
    block — arbitrary client keys are no longer splatted into execute()."""
    key: Optional[str] = None
    value: Optional[Any] = None
    ttl: Optional[int] = None


@router.get("/v1/memory/stats")
async def memory_stats(auth: dict = Depends(require_api_key)):
    """Get memory cache statistics"""
    if not MEMORY_AVAILABLE:
        raise HTTPException(status_code=503, detail="Memory block not available")
    block = get_memory_block()
    return await block.execute({"action": "stats"})


@router.post("/v1/memory/{action}")
async def memory_operation(
    action: str,
    request: MemoryOpRequest,
    auth: dict = Depends(require_api_key),
):
    """Memory cache operations: get, set, delete, flush, keys, exists"""
    if not MEMORY_AVAILABLE:
        raise HTTPException(status_code=503, detail="Memory block not available")

    if action not in _MEMORY_ACTIONS:
        raise HTTPException(status_code=400, detail=f"Unknown action: {action}")

    # Destructive / enumerating actions on the shared in-process cache are
    # admin-only — `flush` wipes it for everyone, `keys` enumerates it.
    if action in {"flush", "keys"} and auth.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin required for this action")

    block = get_memory_block()
    payload = {"action": action, **request.model_dump(exclude_none=True)}
    # Namespace get/set/delete/exists so two JWT users cannot share a key.
    # flush/keys stay unprefixed (admin-only, whole cache).
    if action in {"get", "set", "delete", "exists"} and payload.get("key"):
        uid = auth.get("user_id") or "anon"
        payload["key"] = f"{uid}:{payload['key']}"
    return await block.execute(payload)
