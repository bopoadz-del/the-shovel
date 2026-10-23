"""The development API key: from the environment only, and fail closed.

Order 0255Z / LANE4 #981: no key value lives in the source. Both auth paths
(app.core.auth and app.infra.auth) accept the development key only in a
development environment and only when CEREBRUM_DEV_KEY is set; unset, no key
is accepted and the refusal names the variable; in production a matching
value is still refused.
"""
import asyncio

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from app.core import auth as core_auth
from app.infra.auth import AuthBlock

KEY = "dev-" + "probe-value-for-this-test"


def _creds(value: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=value)


def _infra(value: str) -> dict:
    return asyncio.run(AuthBlock(None, {})._validate_key(value))


def test_core_auth_accepts_the_key_the_environment_sets(monkeypatch):
    monkeypatch.setenv("CEREBRUM_DEV_KEY", KEY)
    assert core_auth.APIKeyAuth().validate_key(_creds(KEY))["user"] == "dev"


def test_core_auth_with_the_variable_unset_refuses_and_names_it(monkeypatch):
    monkeypatch.delenv("CEREBRUM_DEV_KEY", raising=False)
    with pytest.raises(HTTPException) as refused:
        core_auth.APIKeyAuth().validate_key(_creds(KEY))
    assert refused.value.status_code == 401
    assert "CEREBRUM_DEV_KEY" in refused.value.detail


def test_core_auth_in_production_refuses_even_a_matching_value(monkeypatch):
    monkeypatch.setenv("CEREBRUM_DEV_KEY", KEY)
    monkeypatch.setattr(core_auth, "is_dev_environment", lambda: False)
    with pytest.raises(HTTPException) as refused:
        core_auth.APIKeyAuth().validate_key(_creds(KEY))
    assert refused.value.status_code == 401


def test_infra_auth_follows_the_same_rule(monkeypatch):
    monkeypatch.setenv("CEREBRUM_DEV_KEY", KEY)
    assert _infra(KEY)["valid"] is True
    monkeypatch.delenv("CEREBRUM_DEV_KEY", raising=False)
    refused = _infra(KEY)
    assert refused["valid"] is False and refused["reason"] == "dev_key_not_configured"
    assert "CEREBRUM_DEV_KEY" in refused["message"]
    monkeypatch.setenv("CEREBRUM_DEV_KEY", KEY)
    monkeypatch.setattr(core_auth, "is_dev_environment", lambda: False)
    assert _infra(KEY)["valid"] is False
