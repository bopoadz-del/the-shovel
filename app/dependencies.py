"""Shared dependencies and block instance management for FastAPI app."""

from app.core.offload import off_loop as _off_loop
import asyncio
import inspect
import logging
import os
from typing import Any, Dict, Optional

from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.blocks import BLOCK_REGISTRY
from app.core.auth import auth as auth_manager
from app.core import jwt_auth
from app.core import users as users_store
from app.core.privileges import set_caller_role

logger = logging.getLogger(__name__)

# HAL initialization
try:
    from app.infra.hal import HALBlock
    _hal = HALBlock()
except Exception as e:
    logger.warning("HALBlock not available during startup: %s", e)
    _hal = None

# Shared block instances
block_instances: Dict[str, Any] = {}


def _create_block_instance(block_class):
    """Create block instance with proper arguments."""
    sig = inspect.signature(block_class.__init__)
    params = list(sig.parameters.keys())

    if "hal_block" in params and "config" in params:
        instance = block_class(hal_block=_hal, config={})
    else:
        instance = block_class()

    if hasattr(instance, "set_platform"):
        try:
            instance.set_platform(BLOCK_REGISTRY, block_instances, _create_block_instance, get_memory_block)
        except Exception:
            logger.warning(
                "swallowed %s in _create_block_instance() — continuing",
                "Exception", exc_info=True,
            )

    return instance


def _wire_block_dependencies(instance, block_class, name: str = None):
    """Wire requires=[] dependencies into a platform block instance.

    Mirrors UniversalAssembler.inject() for the app/blocks/ layer.
    """
    requires = getattr(block_class, "requires", []) or []
    for dep_name in requires:
        if dep_name in block_instances:
            dep_instance = block_instances[dep_name]
            if hasattr(instance, "wire"):
                instance.wire(dep_name, dep_instance)
            elif hasattr(instance, "inject"):
                instance.inject(dep_name, dep_instance)
            else:
                setattr(instance, f"{dep_name}_block", dep_instance)


def get_block_instance(block_name: str) -> Any:
    if block_name not in block_instances:
        block_class = BLOCK_REGISTRY.get(block_name)
        if block_class is None:
            detail = (
                f"{block_name} kit not enabled — set CEREBRUM_DOMAIN_KITS"
                if block_name == "construction"
                else f"Block '{block_name}' is not loaded in the current boot profile"
            )
            raise HTTPException(status_code=503, detail=detail)
        block_instances[block_name] = _create_block_instance(block_class)
        _wire_block_dependencies(block_instances[block_name], block_class, block_name)
    return block_instances[block_name]


# Memory block
_memory_block = None

try:
    from app.infra.memory import MemoryBlock

    def get_memory_block():
        global _memory_block
        if _memory_block is None:
            _memory_block = MemoryBlock(None, {"max_size": 10000, "default_ttl": 3600})
            asyncio.create_task(_memory_block.initialize())
        return _memory_block

    MEMORY_AVAILABLE = True
except Exception as e:
    MEMORY_AVAILABLE = False
    get_memory_block = None  # type: ignore[assignment]
    logger.warning("Memory block not available: %s", e)


# Monitoring block
_monitoring_block = None

try:
    from app.infra.monitoring import MonitoringBlock

    def get_monitoring_block():
        global _monitoring_block
        if _monitoring_block is None:
            _monitoring_block = MonitoringBlock(None, {})
            _monitoring_block.memory_block = get_memory_block()
            asyncio.create_task(_monitoring_block.initialize())
        return _monitoring_block

    MONITORING_AVAILABLE = True
except Exception as e:
    MONITORING_AVAILABLE = False
    get_monitoring_block = None  # type: ignore[assignment]
    logger.warning("Monitoring block not available: %s", e)


# Auth block
_auth_block = None

try:
    from app.infra.auth import AuthBlock

    def get_auth_block():
        global _auth_block
        if _auth_block is None:
            _auth_block = AuthBlock(None, {
                "rate_limit_default": 100,
                "rate_limit_window": 60,
                "master_key": os.getenv("CEREBRUM_MASTER_KEY"),
            })
            _auth_block.memory_block = get_memory_block()
            asyncio.create_task(_auth_block.initialize())
        return _auth_block

    AUTH_AVAILABLE = True
except Exception as e:
    AUTH_AVAILABLE = False
    get_auth_block = None  # type: ignore[assignment]
    logger.warning("Auth block not available: %s", e)


security = HTTPBearer(auto_error=False)


async def require_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Dict[str, Any]:
    """Require valid API key OR valid JWT for protected endpoints.

    Try JWT first: if the bearer token is a valid JWT, resolve the user and
    return a superset dict compatible with both legacy require_api_key callers
    (who read user/tier/valid) and require_user callers (who read user_id/role/
    email/auth_method).

    If JWT decode fails (InvalidTokenError), fall through to the legacy
    auth_manager.validate_key() path unchanged so the development key and real API keys
    keep working exactly as before.
    """
    if credentials is not None:
        try:
            payload = jwt_auth.decode_token(credentials.credentials)
        except jwt_auth.InvalidTokenError:
            payload = None

        if payload is not None:
            user = (await _off_loop(users_store.get_user_by_id, payload.get("user_id")))
            if not user:
                raise HTTPException(status_code=401, detail="Token user no longer exists")
            principal = {
                # Legacy require_api_key keys (callers read these)
                "user": user["email"],
                "tier": user.get("role") or "user",
                "valid": True,
                # require_user / admin-check keys
                "user_id": user["id"],
                "role": user["role"],
                "email": user["email"],
                "auth_method": "jwt",
            }
            set_caller_role(principal.get("role"))
            return principal

    # JWT decode failed or no credentials — fall through to legacy key validation.
    # validate_key(None) raises HTTPException(401) preserving the no-credentials behavior.
    principal = auth_manager.validate_key(credentials)

    # NORMALISE the key principal so both branches guarantee the same keys.
    #
    # Why: validate_key returns the raw key record — user/tier/role/valid, with
    # NO `user_id`. The JWT branch above always has one. That asymmetry caused a
    # live 500 on 2026-08-02: CEREBRUM_MASTER_KEY is minted with role "admin",
    # so a master-key caller passed the admin gate on /v1/admin/drive/scan and
    # then died on `auth["user_id"]` -> KeyError. Every other route that indexes
    # auth["user_id"] behind require_api_key was one edit away from the same bug.
    #
    # `require_user` already resolves an API key to the singleton system user,
    # so an api-key caller ALREADY acts as that user across the whole app. This
    # grants nothing new — it only stops the two dependencies disagreeing about
    # the shape of a principal for the same credential.
    # DELIBERATELY does not touch `role`. `_require_admin` gates on
    # auth.get("role") == "admin", and only CEREBRUM_MASTER_KEY is minted with
    # that role — a plain CEREBRUM_API_KEY_* has none and must keep getting
    # 403. Defaulting role from the system user (which IS admin) would silently
    # promote every standard API key to admin. Identity only, never authority.
    if not principal.get("user_id"):
        principal["user_id"] = users_store.SYSTEM_USER_ID
        principal["auth_method"] = "api_key"
    set_caller_role(principal.get("role"))
    return principal


async def init_blocks():
    """Initialize all block instances at startup."""
    # Pass 1: instantiate
    for name, block_class in BLOCK_REGISTRY.items():
        try:
            if name not in block_instances:
                block_instances[name] = _create_block_instance(block_class)
        except Exception as e:
            logger.warning("Failed to initialize block %s: %s", name, e)

    # Pass 2: wire dependencies (universal connectors)
    for name, instance in block_instances.items():
        block_class = BLOCK_REGISTRY.get(name)
        if block_class:
            _wire_block_dependencies(instance, block_class, name)

    if get_memory_block:
        get_memory_block()
    if get_monitoring_block:
        get_monitoring_block()
    if get_auth_block:
        get_auth_block()


async def require_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Dict[str, Any]:
    """Resolve the caller to a user.

    Accepts a Bearer JWT (decoded -> user_id) OR a legacy API key
    (validated -> the singleton 'system' user). Returns at least
    {user_id, role, email, auth_method}.
    """
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")

    token = credentials.credentials

    # 1) Try JWT first.
    try:
        payload = jwt_auth.decode_token(token)
    except jwt_auth.InvalidTokenError:
        payload = None

    if payload is not None:
        user = (await _off_loop(users_store.get_user_by_id, payload.get("user_id")))
        if not user:
            raise HTTPException(status_code=401, detail="Token user no longer exists")
        principal = {
            "user_id": user["id"],
            "role": user["role"],
            "email": user["email"],
            "auth_method": "jwt",
        }
        set_caller_role(principal.get("role"))
        return principal

    # 2) Fall back to legacy API key -> system user identity.
    # Authority comes from the key record (only CEREBRUM_MASTER_KEY is minted
    # with role=admin). NEVER copy `role` from the system user row — that row
    # IS admin, and copying it would promote every standard API key.
    principal = auth_manager.validate_key(credentials)  # raises 401/429 on bad key
    sys_user = (await _off_loop(users_store.get_user_by_id, users_store.SYSTEM_USER_ID))
    resolved = {
        "user_id": users_store.SYSTEM_USER_ID,
        "role": principal.get("role") or "user",
        "email": sys_user["email"] if sys_user else "system@local",
        "auth_method": "api_key",
    }
    set_caller_role(resolved.get("role"))
    return resolved
