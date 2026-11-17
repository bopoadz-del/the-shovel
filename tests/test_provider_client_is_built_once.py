"""A model call reuses one open provider client; it never builds one per call.

Grader finding (16 Nov): ``provider_post`` and ``provider_stream`` built a new
``httpx.AsyncClient`` on every model call, inside ``asyncio.to_thread`` -- the
loop's DEFAULT executor. Building a client loads a TLS context and the CA
bundle: CPU work that holds the interpreter lock, once per call, for every
user. Now the client is built once per (event loop, timeout) on the turn pool
and reused; the default executor is never asked.

Every test builds its own client factory and its own provider responses.
"""
from __future__ import annotations

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from app.core import llm_client

_REAL_CLIENT = httpx.AsyncClient


class _NoDefaultExecutor(ThreadPoolExecutor):
    """Installed as the loop's default executor: any use fails the call."""

    def __init__(self) -> None:
        super().__init__(max_workers=1)
        self.used: list = []

    def submit(self, fn, *args, **kwargs):
        self.used.append(getattr(fn, "func", fn))
        raise AssertionError("the default executor was used on a model call")


def _handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content or b"{}")
    if body.get("stream"):
        lines = (
            'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
            'data: {"choices":[],"usage":{"total_tokens":3}}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(200, content=lines.encode(),
                              headers={"content-type": "text/event-stream"})
    return httpx.Response(200, json={"choices": [{"message": {"content": body.get("tag")}}],
                                     "usage": {"total_tokens": 2}})


def _factory(builds: list, fail_first: int = 0):
    """A stand-in for ``httpx.AsyncClient`` that counts builds, records the
    building thread, and answers through an in-memory transport."""

    class _Counting(_REAL_CLIENT):
        def __init__(self, *args, **kwargs):
            if len(builds) < fail_first:
                builds.append(("failed", threading.current_thread().name))
                raise OSError("CA bundle unreadable")
            builds.append((kwargs.get("timeout"), threading.current_thread().name))
            super().__init__(*args, transport=httpx.MockTransport(_handler), **kwargs)

    return _Counting


@pytest.fixture
def wired(monkeypatch):
    builds: list = []
    recorded: list = []
    from app.core import usage_tracker
    monkeypatch.setattr(usage_tracker, "record", lambda **kw: recorded.append(kw))
    monkeypatch.setattr(httpx, "AsyncClient", _factory(builds))
    return builds, recorded


async def _guarded(coro_fn):
    guard = _NoDefaultExecutor()
    asyncio.get_running_loop().set_default_executor(guard)
    try:
        return await coro_fn(), guard.used
    finally:
        await llm_client.aclose_shared_clients()


async def _post(tag, timeout=30.0):
    r, data = await llm_client.provider_post(
        "https://provider.invalid/v1/chat/completions", {"tag": tag}, {},
        timeout=timeout, provider="p", model="m")
    return r.status_code, data["choices"][0]["message"]["content"]


async def _stream(timeout=30.0):
    got = []
    async for line in llm_client.provider_stream(
            "https://provider.invalid/v1/chat/completions", {"stream": True}, {},
            timeout=timeout, provider="p", model="m"):
        got.append(line)
    return got


@pytest.mark.parametrize("calls", [2, 5])
def test_sequential_calls_build_one_client_off_the_default_executor(wired, calls):
    builds, recorded = wired

    async def run():
        return [await _post(f"t{i}") for i in range(calls)]

    results, used = asyncio.run(_guarded(run))
    assert results == [(200, f"t{i}") for i in range(calls)]
    assert used == []
    assert len(builds) == 1
    assert builds[0][1].startswith("turn-io")
    assert len(recorded) == calls


@pytest.mark.parametrize("timeouts,expected_builds", [((30.0, 30.0, 30.0), 1), ((30.0, 90.0, 30.0), 2)])
def test_one_client_per_timeout(wired, timeouts, expected_builds):
    builds, _ = wired

    async def run():
        return [await _post("x", t) for t in timeouts]

    results, used = asyncio.run(_guarded(run))
    assert results == [(200, "x")] * len(timeouts)
    assert used == []
    assert sorted({b[0] for b in builds}) == sorted(set(timeouts))
    assert len(builds) == expected_builds


@pytest.mark.parametrize("first", ["post", "stream"])
def test_streaming_and_plain_calls_share_the_client(wired, first):
    builds, recorded = wired

    async def run():
        order = [_stream, _post] if first == "stream" else [_post, _stream]
        out = []
        for fn in order:
            out.append(await (fn("s") if fn is _post else fn()))
        return out

    results, used = asyncio.run(_guarded(run))
    lines = results[0] if first == "stream" else results[1]
    assert [ln for ln in lines if ln][-1] == "data: [DONE]"
    assert used == []
    assert len(builds) == 1
    assert len(recorded) == 2


@pytest.mark.parametrize("users", [2, 15])
def test_concurrent_first_calls_build_once(wired, users):
    builds, _ = wired

    async def run():
        return await asyncio.gather(*(_post(f"u{i}") for i in range(users)))

    results, used = asyncio.run(_guarded(run))
    assert sorted(results) == sorted((200, f"u{i}") for i in range(users))
    assert used == []
    assert len(builds) == 1


@pytest.mark.parametrize("fail_first", [1, 2])
def test_a_failed_build_is_not_kept(monkeypatch, wired, fail_first):
    builds, _ = wired
    monkeypatch.setattr(httpx, "AsyncClient", _factory(builds, fail_first=fail_first))

    async def run():
        errors = 0
        for _ in range(fail_first):
            try:
                await _post("x")
            except OSError:
                errors += 1
        return errors, await _post("ok"), await _post("again")

    (errors, ok, again), used = asyncio.run(_guarded(run))
    assert errors == fail_first
    assert ok == (200, "ok") and again == (200, "again")
    assert [b[0] for b in builds] == ["failed"] * fail_first + [30.0]


@pytest.mark.parametrize("loops", [1, 3])
def test_each_event_loop_gets_its_own_client(wired, loops):
    builds, _ = wired

    async def run():
        return [await _post("a"), await _post("b")]

    for _ in range(loops):
        results, used = asyncio.run(_guarded(run))
        assert results == [(200, "a"), (200, "b")] and used == []
    assert len(builds) == loops


@pytest.mark.parametrize("close_between", [False, True])
def test_shutdown_closes_the_client_and_the_next_call_builds_again(wired, close_between):
    builds, _ = wired
    seen: list = []

    async def run():
        await _post("a")
        seen.append(next(iter(llm_client._CLIENTS[asyncio.get_running_loop()].values())))
        if close_between:
            await llm_client.aclose_shared_clients()
            assert seen[0].is_closed
        await _post("b")
        return None

    _, used = asyncio.run(_guarded(run))
    assert used == []
    assert seen[0].is_closed  # _guarded closes at the end either way
    assert len(builds) == (2 if close_between else 1)


@pytest.mark.parametrize("swap", [False, True])
def test_a_replaced_client_factory_gets_its_own_client(monkeypatch, wired, swap):
    builds, _ = wired
    other: list = []

    async def run():
        await _post("a")
        if swap:
            monkeypatch.setattr(httpx, "AsyncClient", _factory(other))
        await _post("b")

    _, used = asyncio.run(_guarded(run))
    assert used == []
    assert len(builds) == 1
    assert len(other) == (1 if swap else 0)
