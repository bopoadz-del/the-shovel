"""Anonymous rate-limit identity trusts a forwarded client only when configured.

The socket peer is the bucket unless both TRUSTED_PROXY_COUNT and
TRUSTED_PROXY_CIDRS name a real proxy tier. Credential routes deny the
request when the shared Redis limiter cannot answer; other routes do not.
"""

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from unittest.mock import MagicMock

import app.core.client_address as client_address
import app.core.rate_limit as rate_limit
from app.core import redis_client
from app.core.offload import off_loop


@pytest.fixture(autouse=True)
def _isolate_limiter(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("TRUSTED_PROXY_COUNT", raising=False)
    monkeypatch.delenv("TRUSTED_PROXY_CIDRS", raising=False)
    rate_limit.reset_for_tests()
    redis_client.reset_for_tests()
    yield
    rate_limit.reset_for_tests()


def _trust(monkeypatch, count="1", cidrs="10.0.0.0/8"):
    monkeypatch.setenv("TRUSTED_PROXY_COUNT", count)
    monkeypatch.setenv("TRUSTED_PROXY_CIDRS", cidrs)


def _limited_app():
    """Same shape as production: bind the peer, then limit off the event loop."""
    app = FastAPI()

    @app.middleware("http")
    async def _limit(request, call_next):
        host = request.client.host if request.client else "unknown"
        allowed = await off_loop(rate_limit.check_and_record, f"ip:{host}")
        if not allowed:
            return JSONResponse(status_code=429, content={"status": "error"})
        return await call_next(request)

    @app.api_route("/{full_path:path}", methods=["GET", "POST"])
    def _any(full_path: str):
        return {"ok": True, "path": full_path}

    app.add_middleware(client_address.ClientAddressMiddleware)
    return app


def _client(app, peer):
    from fastapi.testclient import TestClient

    return TestClient(app, client=(peer, 50000))


def _redis(monkeypatch, script):
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    mock_client = MagicMock()
    mock_client.register_script.return_value = script
    monkeypatch.setattr(rate_limit, "get_sync_redis_client", lambda: mock_client)
    rate_limit.reset_for_tests()
    assert rate_limit.init_rate_limiter() == "redis"
    return mock_client


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, 0),
        ("", 0),
        ("0", 0),
        ("-2", 0),
        ("nope", 0),
        ("1", 1),
        ("2", 2),
    ],
)
def test_trusted_proxy_count_rejects_unset_and_non_positive(raw, expected, monkeypatch):
    if raw is None:
        monkeypatch.delenv("TRUSTED_PROXY_COUNT", raising=False)
    else:
        monkeypatch.setenv("TRUSTED_PROXY_COUNT", raw)
    assert client_address.trusted_proxy_count() == expected


@pytest.mark.parametrize(
    "raw,size",
    [
        (None, 0),
        ("10.0.0.0/8", 1),
        ("10.0.0.0/8, 2001:db8::/32", 2),
        ("10.0.0.0/8, nope", 0),
        ("0.0.0.0/0", 0),
        ("::/0", 0),
    ],
)
def test_trusted_proxy_networks_fail_closed_on_bad_entries(raw, size, monkeypatch):
    if raw is None:
        monkeypatch.delenv("TRUSTED_PROXY_CIDRS", raising=False)
    else:
        monkeypatch.setenv("TRUSTED_PROXY_CIDRS", raw)
    assert len(client_address.trusted_proxy_networks()) == size


@pytest.mark.parametrize(
    "count,cidrs",
    [
        ("", ""),
        ("1", ""),
        ("", "10.0.0.0/8"),
        ("0", "10.0.0.0/8"),
        ("nope", "10.0.0.0/8"),
        ("1", "not-a-cidr"),
        ("1", "0.0.0.0/0"),
        ("1", "::/0"),
        ("-3", "10.0.0.0/8"),
        ("1", "10.0.0.0/8, not-a-cidr"),
    ],
)
def test_incomplete_trust_config_ignores_forwarded_chains(count, cidrs, monkeypatch):
    if count:
        monkeypatch.setenv("TRUSTED_PROXY_COUNT", count)
    if cidrs:
        monkeypatch.setenv("TRUSTED_PROXY_CIDRS", cidrs)
    peer = "10.0.0.8"
    for forwarded in ("203.0.113.9", "198.51.100.4, 203.0.113.9"):
        assert client_address.forwarded_client_host(peer, forwarded) == peer


@pytest.mark.parametrize(
    "peer,cidrs,count,forwarded,expected",
    [
        ("10.0.0.8", "10.0.0.0/8", "1", "198.51.100.9, 203.0.113.50", "203.0.113.50"),
        ("10.0.0.8", "10.0.0.0/8", "1", " 203.0.113.50 ", "203.0.113.50"),
        ("::ffff:10.0.0.8", "10.0.0.0/8", "1", "203.0.113.15", "203.0.113.15"),
        ("2001:db8:1::8", "2001:db8:1::/48", "1", "2001:0db8:0000::9", "2001:db8::9"),
        ("2001:db8:1::8", "2001:db8:1::/48", "1", "[2001:db8::9]", "2001:db8::9"),
        ("10.0.0.8", "10.0.0.0/8", "1", "203.0.113.50:443", "203.0.113.50"),
        ("10.0.0.8", "10.0.0.0/8", "1", "203.0.113.9, 10.0.0.4", "10.0.0.8"),
        (
            "10.0.0.8",
            "10.0.0.0/8",
            "2",
            "198.51.100.1, 203.0.113.9, 10.0.0.4",
            "203.0.113.9",
        ),
        ("2001:db8::1", "10.0.0.0/8", "1", "203.0.113.9", "2001:db8::1"),
        ("10.0.0.8", "2001:db8::/32", "1", "203.0.113.9", "10.0.0.8"),
        ("10.0.0.8", "10.0.0.8/32", "1", "203.0.113.9", "203.0.113.9"),
        ("10.0.0.9", "10.0.0.8/32", "1", "203.0.113.9", "10.0.0.9"),
    ],
)
def test_forwarded_client_respects_count_and_cidr_boundaries(
    peer, cidrs, count, forwarded, expected, monkeypatch
):
    _trust(monkeypatch, count=count, cidrs=cidrs)
    assert client_address.forwarded_client_host(peer, forwarded) == expected


@pytest.mark.parametrize(
    "forwarded",
    ["not-an-ip", "203.0.113.5, :::", " , ", "203.0.113.5, "],
)
def test_malformed_forwarded_window_keeps_the_socket_peer(forwarded, monkeypatch):
    _trust(monkeypatch)
    assert client_address.forwarded_client_host("10.0.0.8", forwarded) == "10.0.0.8"


@pytest.mark.parametrize(
    "forwarded",
    ["203.0.113.9", "198.51.100.4, 203.0.113.9"],
)
def test_untrusted_peer_ignores_a_forged_forwarded_chain(forwarded, monkeypatch):
    _trust(monkeypatch)
    assert client_address.forwarded_client_host("203.0.113.7", forwarded) == "203.0.113.7"


@pytest.mark.parametrize(
    "path,expected",
    [
        ("/v1/users/login", True),
        ("/v1/users/login/", True),
        ("/v1/users/register", True),
        ("/v1/users/forgot-password", True),
        ("/v1/users/reset-password", True),
        ("/v1/users/resend-verification", True),
        ("/v1/users/verify-email", True),
        ("/v1/auth", True),
        ("/v1/auth/keys", True),
        ("/v1/users/me", False),
        ("/v1/projects", False),
        ("/v1/authentication", False),
    ],
)
def test_credential_routes_are_the_login_and_auth_surface(path, expected):
    assert client_address.path_is_credential_route(path) is expected


@pytest.mark.parametrize("scope_type", ["lifespan", "websocket"])
def test_non_http_scopes_do_not_bind_forwarded_headers(scope_type):
    seen = {}

    async def _app(scope, receive, send):
        seen["bound"] = client_address.forwarding_context_bound()
        seen["xff"] = client_address.bound_forwarded_for()

    middleware = client_address.ClientAddressMiddleware(_app)
    scope = {
        "type": scope_type,
        "headers": [(b"x-forwarded-for", b"203.0.113.9")],
        "path": "/v1/users/login",
        "client": ("10.0.0.8", 1),
    }
    asyncio.run(middleware(scope, _noop_receive, _noop_send))
    assert seen["bound"] is False
    assert seen["xff"] is None


async def _noop_receive():
    return {"type": "http.request", "body": b"", "more_body": False}


async def _noop_send(_message):
    return None


@pytest.mark.parametrize(
    "headers,expected",
    [
        (
            [
                (b"x-forwarded-for", b"198.51.100.1"),
                (b"x-forwarded-for", b"203.0.113.60"),
            ],
            "203.0.113.60",
        ),
        (
            [
                (b"x-forwarded-for", b"203.0.113.60, 10.0.0.4"),
                (b"x-forwarded-for", b"203.0.113.61"),
            ],
            "203.0.113.61",
        ),
    ],
)
def test_repeated_forwarded_headers_use_the_rightmost_hop(headers, expected, monkeypatch):
    _trust(monkeypatch)
    seen = {}

    async def _app(scope, receive, send):
        seen["host"] = client_address.forwarded_client_host(
            client_address.bound_peer(),
            client_address.bound_forwarded_for(),
        )

    middleware = client_address.ClientAddressMiddleware(_app)
    scope = {
        "type": "http",
        "headers": headers,
        "path": "/v1/projects",
        "client": ("10.0.0.8", 50000),
        "method": "GET",
    }
    asyncio.run(middleware(scope, _noop_receive, _noop_send))
    assert seen["host"] == expected


@pytest.mark.parametrize(
    "peer,left,right",
    [
        ("10.0.0.8", "203.0.113.10", "203.0.113.11"),
        ("::ffff:10.0.0.8", "2001:db8::10", "2001:db8::11"),
    ],
)
def test_two_forwarded_sources_have_independent_buckets(peer, left, right, monkeypatch):
    _trust(monkeypatch)
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "1")
    app = _limited_app()
    with _client(app, peer) as client:
        first = {
            left: client.get("/v1/projects", headers={"x-forwarded-for": left}).status_code,
            right: client.get("/v1/projects", headers={"x-forwarded-for": right}).status_code,
        }
        again = {
            left: client.get("/v1/projects", headers={"x-forwarded-for": left}).status_code,
            right: client.get("/v1/projects", headers={"x-forwarded-for": right}).status_code,
        }
    assert first[left] == 200 and first[right] == 200
    assert again[left] == 429 and again[right] == 429


@pytest.mark.parametrize(
    "first,second,shared",
    [
        ("2001:0db8::10", "2001:db8::10", True),
        ("203.0.113.10", "203.0.113.11", False),
    ],
)
def test_canonical_address_spelling_shares_one_bucket(first, second, shared, monkeypatch):
    _trust(monkeypatch)
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "1")
    app = _limited_app()
    with _client(app, "10.0.0.8") as client:
        codes = [
            client.get("/v1/projects", headers={"x-forwarded-for": first}).status_code,
            client.get("/v1/projects", headers={"x-forwarded-for": second}).status_code,
        ]
    assert codes[0] == 200
    assert (codes[1] == 429) is shared


@pytest.mark.parametrize("header", ["x-real-ip", "forwarded"])
def test_only_x_forwarded_for_splits_anonymous_buckets(header, monkeypatch):
    _trust(monkeypatch)
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "1")
    app = _limited_app()
    with _client(app, "10.0.0.8") as client:
        codes = [
            client.get("/v1/projects", headers={header: "203.0.113.40"}).status_code,
            client.get("/v1/projects", headers={header: "198.51.100.41"}).status_code,
        ]
    assert codes == [200, 429]


@pytest.mark.parametrize("client_ip", ["203.0.113.10", "2001:db8::20"])
def test_forwarded_header_does_not_stick_to_the_next_request(client_ip, monkeypatch):
    _trust(monkeypatch)
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "1")
    app = _limited_app()
    with _client(app, "10.0.0.8") as client:
        with_header = client.get(
            "/v1/projects", headers={"x-forwarded-for": client_ip}
        ).status_code
        bare = client.get("/v1/projects").status_code
        bare_again = client.get("/v1/projects").status_code
    assert with_header == 200
    assert bare == 200
    assert bare_again == 429


@pytest.mark.parametrize(
    "forwarded,user_id,key_id",
    [
        ("203.0.113.10", "user:one", "key:alpha"),
        ("2001:db8::10", "user:two", "key:beta"),
    ],
)
def test_authenticated_identities_ignore_the_forwarded_client(
    forwarded, user_id, key_id, monkeypatch
):
    _trust(monkeypatch)
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "10")
    seen = {}
    app = FastAPI()

    @app.middleware("http")
    async def _limit(request, call_next):
        host = request.client.host if request.client else "unknown"
        allowed = await off_loop(rate_limit.check_and_record, f"ip:{host}")
        if not allowed:
            return JSONResponse(status_code=429, content={"status": "error"})
        return await call_next(request)

    @app.get("/v1/projects")
    def _projects():
        seen["user"] = rate_limit.resolve_rate_limit_identity(user_id)
        seen["key"] = rate_limit.resolve_rate_limit_identity(key_id)
        seen["forged"] = rate_limit.resolve_rate_limit_identity("ip:198.51.100.9")
        seen["other"] = rate_limit.resolve_rate_limit_identity("ip:203.0.113.8")
        return {"ok": True}

    app.add_middleware(client_address.ClientAddressMiddleware)
    with _client(app, "10.0.0.8") as client:
        status = client.get(
            "/v1/projects", headers={"x-forwarded-for": forwarded}
        ).status_code
    canonical = client_address.forwarded_client_host("10.0.0.8", forwarded)
    assert status == 200
    assert seen["user"] == user_id
    assert seen["key"] == key_id
    assert seen["forged"] == "ip:" + canonical
    assert seen["other"] == "ip:" + canonical


@pytest.mark.parametrize(
    "path,denied",
    [
        ("/v1/users/login", True),
        ("/v1/users/forgot-password", True),
        ("/v1/projects", False),
        ("/v1/execute", False),
    ],
)
def test_redis_script_errors_fail_closed_only_on_credential_routes(
    path, denied, monkeypatch
):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "2")
    script = MagicMock(side_effect=ConnectionError("redis down"))
    _redis(monkeypatch, script)
    app = _limited_app()
    with _client(app, "203.0.113.7") as client:
        codes = [client.post(path).status_code for _ in range(4)]
    assert codes == ([429, 429, 429, 429] if denied else [200, 200, 200, 200])


@pytest.mark.parametrize("path", ["/v1/users/login", "/v1/auth/validate"])
def test_disabled_limiter_allows_credential_routes_when_redis_errors(path, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "0")
    script = MagicMock(side_effect=RuntimeError("redis down"))
    _redis(monkeypatch, script)
    app = _limited_app()
    with _client(app, "203.0.113.7") as client:
        codes = [client.post(path).status_code for _ in range(2)]
    assert codes == [200, 200]


@pytest.mark.parametrize("path", ["/v1/users/login", "/v1/projects"])
def test_healthy_redis_still_enforces_the_limit(path, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "2")
    script = MagicMock(side_effect=[1, 1, 0])
    _redis(monkeypatch, script)
    app = _limited_app()
    with _client(app, "203.0.113.7") as client:
        codes = [client.get(path).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


@pytest.mark.parametrize(
    "path,fail_closed,first",
    [
        ("/v1/users/register", True, 429),
        ("/v1/users/reset-password", True, 429),
        ("/v1/projects", False, 200),
        ("/v1/execute", False, 200),
    ],
)
def test_missing_redis_client_denies_only_credential_routes(
    path, fail_closed, first, monkeypatch
):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "5")
    script = MagicMock(return_value=1)
    _redis(monkeypatch, script)
    limiter = rate_limit._redis_limiter
    limiter._client = None
    limiter._script = None
    monkeypatch.setattr(rate_limit, "get_sync_redis_client", lambda: None)
    app = _limited_app()
    with _client(app, "203.0.113.7") as client:
        status = client.post(path).status_code
    assert status == first
    assert fail_closed is (first == 429)


@pytest.mark.parametrize(
    "outer,inner",
    [
        ("CORSMiddleware", "ClientAddressMiddleware"),
        ("ClientAddressMiddleware", "rate_limit_middleware"),
    ],
)
def test_client_address_middleware_sits_outside_the_limiter(outer, inner):
    from app.main import app

    names = []
    for middleware in app.user_middleware:
        dispatch = middleware.kwargs.get("dispatch")
        if dispatch is not None:
            names.append(dispatch.__name__)
        else:
            names.append(middleware.cls.__name__)
    assert names.index(outer) < names.index(inner)
