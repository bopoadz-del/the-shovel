"""Every Redis constructor in the app is bounded (P-I2).

A Redis that accepts the TCP connection and never answers used to hold the
caller forever: no constructor set ``socket_connect_timeout`` or
``socket_timeout``. The three constructors (async client, sync client, the
session store's boot ping) now take one bound, ``REDIS_SOCKET_TIMEOUT_S``.

Each decision is tested with a hung server and a working one, both real
local sockets built by the test: the hung one returns within the bound and
the caller takes its loss path; the working one is used.
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time

import pytest

from app.core import redis_client as rc
from app.core import session_store as ss

BOUND_S = 0.5
# Generous slack for a slow CI runner; an unbounded call never returns.
CEILING_S = BOUND_S + 2.5


class _LocalServer:
    """A loopback TCP server. ``answer`` False accepts and reads but never
    replies (a hung Redis); True replies ``+PONG`` to every command."""

    def __init__(self, answer: bool):
        self.answer = answer
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(8)
        self._stop = threading.Event()
        self._conns: list[socket.socket] = []
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        return f"redis://127.0.0.1:{self._sock.getsockname()[1]}/0"

    def _serve(self) -> None:
        self._sock.settimeout(0.1)
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except OSError:
                continue
            self._conns.append(conn)
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        conn.settimeout(0.1)
        while not self._stop.is_set():
            try:
                data = conn.recv(4096)
            except socket.timeout:
                continue
            except OSError:
                return
            if not data:
                return
            if self.answer:
                # One reply per command in the buffer (pipelined or not).
                conn.sendall(b"+PONG\r\n" * max(1, data.count(b"*")))

    def close(self) -> None:
        self._stop.set()
        for conn in self._conns:
            try:
                conn.close()
            except OSError:
                pass
        self._sock.close()


def _within_ceiling(fn):
    """Run ``fn`` in a daemon thread; fail (never hang) past CEILING_S."""
    box: dict = {}

    def _run():
        start = time.monotonic()
        try:
            box["value"] = fn()
        except BaseException as exc:  # re-raised in the test thread
            box["error"] = exc
        box["elapsed"] = time.monotonic() - start

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(CEILING_S)
    if worker.is_alive():
        pytest.fail(f"Redis call still blocked after {CEILING_S}s: it is unbounded")
    if "error" in box:
        raise box["error"]
    return box["value"], box["elapsed"]


@pytest.fixture
def server(request):
    srv = _LocalServer(answer=request.param)
    yield srv
    srv.close()


@pytest.fixture(autouse=True)
def _bounded(monkeypatch):
    monkeypatch.setenv("REDIS_SOCKET_TIMEOUT_S", str(BOUND_S))
    rc.reset_for_tests()
    yield
    rc.reset_for_tests()


# ── the setting ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("0.25", 0.25),
        ("7", 7.0),
        ("", rc.DEFAULT_REDIS_SOCKET_TIMEOUT_S),
        ("abc", rc.DEFAULT_REDIS_SOCKET_TIMEOUT_S),
        ("0", rc.DEFAULT_REDIS_SOCKET_TIMEOUT_S),
        ("-1", rc.DEFAULT_REDIS_SOCKET_TIMEOUT_S),
        ("nan", rc.DEFAULT_REDIS_SOCKET_TIMEOUT_S),
        ("inf", rc.DEFAULT_REDIS_SOCKET_TIMEOUT_S),
    ],
)
def test_the_bound_reads_one_setting_and_never_disables_itself(monkeypatch, raw, expected):
    monkeypatch.setenv("REDIS_SOCKET_TIMEOUT_S", raw)
    assert rc.redis_socket_timeout_s() == expected
    assert rc.redis_timeout_kwargs() == {
        "socket_connect_timeout": expected,
        "socket_timeout": expected,
    }


@pytest.mark.parametrize("unset", [True, False], ids=["absent", "blank"])
def test_the_default_is_a_few_seconds(monkeypatch, unset):
    if unset:
        monkeypatch.delenv("REDIS_SOCKET_TIMEOUT_S", raising=False)
    else:
        monkeypatch.setenv("REDIS_SOCKET_TIMEOUT_S", "   ")
    assert 0 < rc.redis_socket_timeout_s() <= 5


# ── the three constructors: hung vs working ─────────────────────────────


@pytest.mark.parametrize("server", [False, True], indirect=True, ids=["hung", "working"])
def test_sync_client_is_bounded(monkeypatch, server):
    monkeypatch.setenv("REDIS_URL", server.url)
    client, elapsed = _within_ceiling(rc.get_sync_redis_client)
    if server.answer:
        assert client is not None
        kwargs = client.connection_pool.connection_kwargs
        assert kwargs["socket_timeout"] == BOUND_S
        assert kwargs["socket_connect_timeout"] == BOUND_S
    else:
        assert client is None  # loss path: callers fall back
        assert elapsed >= BOUND_S * 0.8  # it waited for the bound, not a refusal


@pytest.mark.parametrize("server", [False, True], indirect=True, ids=["hung", "working"])
def test_async_client_is_bounded(monkeypatch, server):
    monkeypatch.setenv("REDIS_URL", server.url)

    async def _go():
        client = await rc.get_redis_client()
        if client is not None:
            kwargs = dict(client.connection_pool.connection_kwargs)
            await client.aclose()
        else:
            kwargs = None
        return client, kwargs

    (client, kwargs), elapsed = _within_ceiling(lambda: asyncio.run(_go()))
    rc.reset_for_tests()
    if server.answer:
        assert client is not None
        assert kwargs["socket_timeout"] == BOUND_S
        assert kwargs["socket_connect_timeout"] == BOUND_S
    else:
        assert client is None
        assert elapsed >= BOUND_S * 0.8


@pytest.mark.parametrize("server", [False, True], indirect=True, ids=["hung", "working"])
def test_session_store_boot_ping_is_bounded(monkeypatch, server):
    monkeypatch.setenv("REDIS_URL", server.url)
    import warnings

    def _boot():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            return ss.get_session_store(), [str(w.message) for w in caught]

    (store, messages), _elapsed = _within_ceiling(_boot)
    fell_back = any("falling back to in-memory" in m for m in messages)
    assert fell_back is (not server.answer)
    expected = ss.RedisSessionStore if server.answer else ss.InMemorySessionStore
    assert type(store) is expected


@pytest.mark.parametrize("server", [False, True], indirect=True, ids=["hung", "working"])
def test_sync_client_caches_only_a_live_client(monkeypatch, server):
    """A live client is built once and reused; a lost one is retried.

    Without the cache every caller would open a new pool and ping again;
    caching the loss would keep the fallback after Redis comes back.
    """
    monkeypatch.setenv("REDIS_URL", server.url)
    built: list = []
    real_from_url = rc.redis.from_url

    def _counting_from_url(*args, **kwargs):
        client = real_from_url(*args, **kwargs)
        built.append(client)
        return client

    monkeypatch.setattr(rc.redis, "from_url", _counting_from_url)
    first, _ = _within_ceiling(rc.get_sync_redis_client)
    second, _ = _within_ceiling(rc.get_sync_redis_client)
    if server.answer:
        assert first is not None and second is first
        assert len(built) == 1
    else:
        assert first is None and second is None
        assert len(built) == 2
