"""Two hardening items from the 2026-08-02 readiness pass.

1. `require_api_key` must return the SAME guaranteed keys as `require_user`.
   The asymmetry (JWT branch has `user_id`, api-key branch does not, while
   CEREBRUM_MASTER_KEY carries role "admin") produced a live 500 on
   /v1/admin/drive/scan. Normalising identity closes the whole latent class.

   CRITICAL: identity only, NEVER authority. The system user IS admin, so
   defaulting `role` from it would silently promote every plain
   CEREBRUM_API_KEY_* to admin. That is the regression these tests pin.

2. Interactive docs + /openapi.json must be CLOSED when ENV=production.
   The route sweep found /openapi.json returning 200 unauthenticated on
   prod, disclosing all 142 operations including every /v1/admin/* route.
"""
from __future__ import annotations

import importlib

import pytest
from fastapi import HTTPException

from app.core import users as users_store


# ── 1. principal shape ───────────────────────────────────────────────────────

class _Creds:
    def __init__(self, token):
        self.credentials = token


def _validate_key_returning(monkeypatch, record):
    """Force the legacy api-key branch to return `record`."""
    from app import dependencies as deps

    monkeypatch.setattr(
        deps.jwt_auth,
        "decode_token",
        lambda t: (_ for _ in ()).throw(deps.jwt_auth.InvalidTokenError("not a jwt")),
    )
    monkeypatch.setattr(deps.auth_manager, "validate_key", lambda c: dict(record))
    return deps


@pytest.mark.asyncio
async def test_master_key_principal_gets_a_user_id(monkeypatch):
    """The exact record auth_manager mints for CEREBRUM_MASTER_KEY."""
    deps = _validate_key_returning(
        monkeypatch,
        {"user": "master", "tier": "unlimited", "role": "admin", "valid": True},
    )

    principal = await deps.require_api_key(_Creds("some-api-key"))

    assert principal["user_id"] == users_store.SYSTEM_USER_ID
    assert principal["auth_method"] == "api_key"
    # admin role preserved — the master key is genuinely admin
    assert principal["role"] == "admin"


@pytest.mark.asyncio
async def test_plain_api_key_is_NOT_promoted_to_admin(monkeypatch):
    """Identity only, never authority.

    A CEREBRUM_API_KEY_* record has no `role`. The singleton system user IS
    admin, so a naive `setdefault("role", sys_user["role"])` would hand every
    standard API key admin rights — `_require_admin` gates on exactly this key.
    """
    deps = _validate_key_returning(
        monkeypatch,
        {"user": "reporting", "tier": "standard", "rate_limit": 1000, "valid": True},
    )

    principal = await deps.require_api_key(_Creds("plain-key"))

    assert principal["user_id"] == users_store.SYSTEM_USER_ID  # identity granted
    assert principal.get("role") != "admin"                    # authority NOT granted

    from app.routers.admin import _require_admin
    with pytest.raises(HTTPException) as exec_exc:
        _require_admin(principal)
    assert exec_exc.value.status_code == 403


@pytest.mark.asyncio
async def test_require_user_plain_api_key_is_NOT_promoted_to_admin(monkeypatch):
    """Same identity-only rule as require_api_key: never copy system-user role."""
    deps = _validate_key_returning(
        monkeypatch,
        {"user": "reporting", "tier": "standard", "rate_limit": 1000, "valid": True},
    )
    monkeypatch.setattr(
        deps.users_store,
        "get_user_by_id",
        lambda uid: {"id": uid, "role": "admin", "email": "system@local"},
    )

    principal = await deps.require_user(_Creds("plain-key"))

    assert principal["user_id"] == users_store.SYSTEM_USER_ID
    assert principal["auth_method"] == "api_key"
    assert principal.get("role") != "admin"


@pytest.mark.asyncio
async def test_jwt_principal_shape_is_unchanged(monkeypatch):
    """Normalisation must not touch the JWT branch."""
    from app import dependencies as deps

    monkeypatch.setattr(deps.jwt_auth, "decode_token", lambda t: {"user_id": "usr_7"})
    monkeypatch.setattr(
        deps.users_store,
        "get_user_by_id",
        lambda uid: {"id": "usr_7", "role": "user", "email": "a@b.c"},
    )

    principal = await deps.require_api_key(_Creds("jwt-token"))

    assert principal["user_id"] == "usr_7"
    assert principal["auth_method"] == "jwt"
    assert principal["role"] == "user"


# ── 2. API surface disclosure ────────────────────────────────────────────────

@pytest.mark.parametrize(
    "env,override,expected",
    [
        ("production", "",       False),  # closed on prod by default
        ("production", "true",   True),   # operator can re-open explicitly
        ("development", "",      True),   # dev keeps the docs a dev expects
        ("development", "false", False),  # and can close them
        ("", "",                 True),   # unset ENV is not production
    ],
)
def test_docs_exposure_policy(monkeypatch, env, override, expected):
    monkeypatch.setenv("ENV", env)
    monkeypatch.setenv("API_DOCS_ENABLED", override)
    import app.main as main_mod
    assert main_mod._api_docs_enabled() is expected


def test_production_app_exposes_no_schema_routes(monkeypatch):
    """End-to-end: with ENV=production the app serves no docs/openapi route."""
    monkeypatch.setenv("ENV", "production")
    monkeypatch.delenv("API_DOCS_ENABLED", raising=False)
    import app.main as main_mod
    reloaded = importlib.reload(main_mod)

    try:
        assert reloaded.app.docs_url is None
        assert reloaded.app.redoc_url is None
        assert reloaded.app.openapi_url is None
        paths = {r.path for r in reloaded.app.routes}
        assert "/openapi.json" not in paths
        assert "/docs" not in paths
    finally:
        # restore the module for the rest of the session
        monkeypatch.delenv("ENV", raising=False)
        importlib.reload(main_mod)
