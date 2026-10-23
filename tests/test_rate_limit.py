"""Per-caller rate limiting — covers every request, including JWT sessions."""

from unittest.mock import MagicMock

import hashlib

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from app.core import rate_limit
from app.core import redis_client
from tests._fixture_values import BEARER_TOKEN


def test_check_and_record_allows_then_blocks(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "5")
    rate_limit.reset_for_tests()

    ident = "unit-test-id"
    assert all(rate_limit.check_and_record(ident) for _ in range(5))
    # 6th request in the window is over the limit.
    assert rate_limit.check_and_record(ident) is False


def test_disabled_when_limit_is_zero(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "0")
    rate_limit.reset_for_tests()
    assert all(rate_limit.check_and_record("anything") for _ in range(100))


def test_identities_have_independent_budgets(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "2")
    rate_limit.reset_for_tests()

    assert rate_limit.check_and_record("caller-A")
    assert rate_limit.check_and_record("caller-A")
    assert rate_limit.check_and_record("caller-A") is False  # A exhausted
    # A different caller still has its full budget.
    assert rate_limit.check_and_record("caller-B")


def _rate_limit_identity(request: Request) -> str:
    """Identify the caller the same way the production middleware does."""
    authz = request.headers.get("Authorization", "")
    if authz.startswith("Bearer "):
        token = authz[7:].strip()
        return "key:" + hashlib.sha256(token.encode()).hexdigest()[:24]
    host = request.client.host if request.client else "unknown"
    return f"ip:{host}"


async def _rate_limit_middleware(request: Request, call_next):
    """Tiny middleware that exercises app.core.rate_limit.check_and_record."""
    if request.method == "OPTIONS":
        return await call_next(request)
    if not rate_limit.check_and_record(_rate_limit_identity(request)):
        return JSONResponse(
            status_code=429,
            content={"status": "error",
                     "error": "Rate limit exceeded — too many requests."},
        )
    return await call_next(request)


def _rate_limited_app() -> FastAPI:
    """Minimal FastAPI app with only the rate-limit middleware under test."""
    app = FastAPI()
    app.middleware("http")(_rate_limit_middleware)

    @app.get("/v1/projects")
    def _projects():
        return {"ok": True}

    return app


def test_middleware_returns_429_over_the_limit(monkeypatch):
    """The HTTP middleware throttles a caller regardless of auth type."""
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "3")
    rate_limit.reset_for_tests()

    client = TestClient(_rate_limited_app())
    headers = {"Authorization": "Bearer " + BEARER_TOKEN}
    codes = [client.get("/v1/projects", headers=headers).status_code
             for _ in range(5)]

    # The first 3 pass the limiter (whatever the endpoint itself returns);
    # the 4th and 5th are rejected with 429.
    assert codes[0] != 429
    assert codes[3] == 429 and codes[4] == 429, codes


@pytest.fixture(autouse=True)
def _reset_rate_limit_state(monkeypatch):
    """Isolate rate-limiter state and clear REDIS_URL so the suite never
    accidentally talks to a real Redis instance."""
    monkeypatch.delenv("REDIS_URL", raising=False)
    rate_limit.reset_for_tests()
    redis_client.reset_for_tests()
    yield


def test_init_rate_limiter_uses_redis_when_factory_returns_client(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "5")

    mock_client = MagicMock()
    mock_script = mock_client.register_script.return_value
    mock_script.return_value = 1
    monkeypatch.setattr("app.core.rate_limit.get_sync_redis_client", lambda: mock_client)

    assert rate_limit.init_rate_limiter() == "redis"
    assert rate_limit.check_and_record("redis-caller") is True
    mock_script.assert_called_once()
    assert mock_script.call_args.kwargs["keys"] == ["ratelimit:redis-caller"]
    assert mock_script.call_args.kwargs["args"][1] == 60.0
    assert mock_script.call_args.kwargs["args"][2] == 5


def test_redis_unavailable_falls_back_to_in_memory(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "2")
    monkeypatch.setattr("app.core.rate_limit.get_sync_redis_client", lambda: None)

    # init_rate_limiter() must label the backend as in-memory when the
    # factory cannot return a client.
    assert rate_limit.init_rate_limiter() == "in-memory"
    assert not rate_limit._use_redis

    # Redis is unreachable, so the per-request path falls back to in-memory.
    assert rate_limit.check_and_record("fallback-caller")
    assert rate_limit.check_and_record("fallback-caller")
    assert rate_limit.check_and_record("fallback-caller") is False


def test_redis_script_exception_fails_open(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "2")

    mock_client = MagicMock()
    mock_script = mock_client.register_script.return_value
    mock_script.side_effect = Exception("Redis unavailable")
    monkeypatch.setattr("app.core.rate_limit.get_sync_redis_client", lambda: mock_client)

    assert rate_limit.init_rate_limiter() == "redis"
    assert rate_limit._use_redis

    # Redis script failures must fail-open so a cache outage cannot throttle
    # legitimate traffic.
    assert rate_limit.check_and_record("exc-caller") is True
    assert rate_limit.check_and_record("exc-caller") is True
    assert rate_limit.check_and_record("exc-caller") is True
