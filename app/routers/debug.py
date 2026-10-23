import os

from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import require_api_key

router = APIRouter()


#: The ONLY environments in which debug routes exist. Fail closed: anything
#: not on this list is production, including an unset variable.
#:
#: This list is shared with ``app.main``, which decides whether to mount the
#: router at all. It used to be duplicated there, and the two copies had
#: already drifted: main.py used this allow-list while ``_is_production``
#: below tested ``env == "production"`` exactly, so ENV="prod-eu" (or "prod",
#: or "production-eu") was NOT production to the endpoint. Nothing was
#: exposed, because such a value also fails the mount test -- but the two
#: answers to "is this production" disagreed, and only one of them was
#: guarding the payload. One list, one answer.
DEV_ENVIRONMENTS = frozenset({"dev", "development", "local", "test", "testing"})


def current_environment() -> str:
    return os.getenv("ENV", os.getenv("ENVIRONMENT", "production")).strip().lower()


def is_dev_environment() -> bool:
    """True only for an explicitly named development environment."""
    return current_environment() in DEV_ENVIRONMENTS


def _is_production() -> bool:
    return not is_dev_environment()


def _require_non_production():
    if _is_production():
        raise HTTPException(status_code=404, detail="Not found")


@router.get("/debug/env")
def debug_env(auth: dict = Depends(require_api_key)):
    """Debug endpoint — gated to non-production + admin only."""
    _require_non_production()
    if auth.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return {
        "environment": os.getenv("ENV", "unknown"),
        "data_dir": os.getenv("DATA_DIR", "not_set"),
    }


@router.get("/v1/debug/env")
def debug_env_v1(auth: dict = Depends(require_api_key)):
    """Debug endpoint (v1 alias)."""
    return debug_env(auth)
