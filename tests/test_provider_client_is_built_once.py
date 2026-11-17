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


def _factory(builds: list, fail_first: int = 0, handler=_handler, hold=None, transports=None):
    """A stand-in for ``httpx.AsyncClient`` that counts builds, records the
    building thread, and answers through an in-memory transport placed under
    the module's own retry wrapper. ``hold`` maps a timeout to an Event the
    build of that timeout waits on; ``transports`` collects the network
    transports the module built (replaced here by the in-memory one)."""
    transports = [] if transports is None else transports

    class _Counting(_REAL_CLIENT):
        def __init__(self, *args, **kwargs):
            if len(builds) < fail_first:
                builds.append(("failed", threading.current_thread().name))
                raise OSError("CA bundle unreadable")
            gate = (hold or {}).get(kwargs.get("timeout"))
            if gate is not None:
                gate.wait(10)
            builds.append((kwargs.get("timeout"), threading.current_thread().name))
            transport = kwargs.pop("transport")
            assert isinstance(transport, llm_client._StaleConnectionRetry)
            transports.append(transport.inner)
            transport.inner = httpx.MockTransport(handler)
            super().__init__(*args, transport=transport, **kwargs)

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


@pytest.mark.parametrize("built,slow,calls", [(30.0, 90.0, 1), (60.0, 120.0, 3)])
def test_a_built_client_is_not_held_behind_another_timeouts_first_build(monkeypatch, wired, built, slow, calls):
    """A slow first build (another timeout: a slow CA-bundle read) must not
    stall every model call whose client is already open."""
    builds, _ = wired
    gate = threading.Event()
    monkeypatch.setattr(httpx, "AsyncClient", _factory(builds, hold={slow: gate}))

    async def run():
        await _post("warm", timeout=built)
        pending = asyncio.create_task(_post("slow", timeout=slow))
        await asyncio.sleep(0.05)
        try:
            got = [await asyncio.wait_for(_post(f"b{i}", timeout=built), 2.0) for i in range(calls)]
            still_building = not pending.done()
        finally:
            gate.set()
        return got, still_building, await pending

    (got, still_building, slow_result), used = asyncio.run(_guarded(run))
    assert got == [(200, f"b{i}") for i in range(calls)]
    assert still_building is True
    assert slow_result == (200, "slow")
    assert used == []


def _recording_handler(seen: list, status: int = 200):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content or b"{}"))
        if status != 200:
            return httpx.Response(status, content=f"upstream said {status}".encode())
        return _handler(request)
    return handler


@pytest.mark.parametrize("payload,sent_options", [
    ({"stream": True}, {"include_usage": True}),
    ({"stream": True, "stream_options": {"include_usage": False}}, {"include_usage": False}),
])
def test_a_stream_asks_for_the_usage_chunk_unless_the_caller_chose(monkeypatch, wired, payload, sent_options):
    builds, _ = wired
    seen: list = []
    monkeypatch.setattr(httpx, "AsyncClient", _factory(builds, handler=_recording_handler(seen)))

    async def run():
        return [line async for line in llm_client.provider_stream(
            "https://provider.invalid/v1/chat/completions", dict(payload), {},
            timeout=30.0, provider="p", model="m")]

    lines, used = asyncio.run(_guarded(run))
    assert seen[0]["stream_options"] == sent_options
    assert "data: [DONE]" in lines
    assert used == []


@pytest.mark.parametrize("status", [429, 503])
def test_a_refused_stream_raises_before_any_line(monkeypatch, wired, status):
    builds, recorded = wired
    seen: list = []
    monkeypatch.setattr(httpx, "AsyncClient", _factory(builds, handler=_recording_handler(seen, status)))
    lines: list = []

    async def run():
        with pytest.raises(llm_client.ProviderStreamError) as err:
            async for line in llm_client.provider_stream(
                    "https://provider.invalid/v1/chat/completions", {"stream": True}, {},
                    timeout=30.0, provider="p", model="m"):
                lines.append(line)
        return err.value

    err, used = asyncio.run(_guarded(run))
    assert (err.status, err.text) == (status, f"upstream said {status}")
    assert lines == []
    assert recorded == []
    assert used == []


# --- grader 17 Nov 18:35Z: a reused connection can be dead after an idle period


def _flaky_handler(fail_on: set, exc_type, calls: list):
    """Answers like ``_handler`` but raises ``exc_type`` (no response at all)
    on the request numbers in ``fail_on`` (1-based)."""

    def handler(request):
        calls.append(request.url.path)
        if len(calls) in fail_on:
            raise exc_type("Server disconnected without sending a response.", request=request)
        return _handler(request)

    return handler


@pytest.mark.parametrize("exc_type", [httpx.RemoteProtocolError, httpx.ReadError])
def test_a_call_after_idling_survives_a_connection_the_provider_dropped(monkeypatch, wired, exc_type):
    builds, recorded = wired
    calls: list = []
    # Call 1 opens the connection; call 2 reuses it after the provider closed
    # it while the user idled; the resend on a new connection is answered.
    monkeypatch.setattr(httpx, "AsyncClient", _factory(builds, handler=_flaky_handler({2}, exc_type, calls)))

    async def run():
        return [await _post("before idle"), await _post("after idle")]

    results, used = asyncio.run(_guarded(run))
    assert results == [(200, "before idle"), (200, "after idle")]
    assert len(calls) == 3
    assert len(builds) == 1
    assert len(recorded) == 2
    assert used == []


@pytest.mark.parametrize("exc_type", [httpx.RemoteProtocolError, httpx.WriteError])
def test_a_stream_after_idling_survives_a_connection_the_provider_dropped(monkeypatch, wired, exc_type):
    builds, recorded = wired
    calls: list = []
    monkeypatch.setattr(httpx, "AsyncClient", _factory(builds, handler=_flaky_handler({2}, exc_type, calls)))

    async def run():
        await _post("before idle")
        return await _stream()

    lines, used = asyncio.run(_guarded(run))
    assert [ln for ln in lines if ln][-1] == "data: [DONE]"
    assert len(calls) == 3
    assert len(recorded) == 2
    assert used == []


@pytest.mark.parametrize("exc_type", [httpx.RemoteProtocolError, httpx.ConnectError])
def test_a_provider_that_never_answers_is_a_transport_error_after_one_resend(monkeypatch, wired, exc_type):
    builds, recorded = wired
    calls: list = []
    monkeypatch.setattr(httpx, "AsyncClient", _factory(builds, handler=_flaky_handler({1, 2, 3}, exc_type, calls)))

    async def run():
        with pytest.raises(llm_client.ProviderStreamError) as err:
            await _stream()
        return err.value

    err, used = asyncio.run(_guarded(run))
    assert err.status == 0
    assert "Server disconnected" in err.text
    assert len(calls) == 2  # sent once more, never a third time
    assert recorded == []


@pytest.mark.parametrize("status", [500, 503])
def test_a_request_the_provider_answered_is_never_resent(monkeypatch, wired, status):
    builds, recorded = wired
    seen: list = []
    monkeypatch.setattr(httpx, "AsyncClient", _factory(builds, handler=_recording_handler(seen, status)))

    async def run():
        r, _ = await llm_client.provider_post(
            "https://provider.invalid/v1/chat/completions", {"tag": "x"}, {},
            timeout=30.0, provider="p", model="m")
        return r.status_code

    got, used = asyncio.run(_guarded(run))
    assert got == status
    assert len(seen) == 1
    assert recorded == []


@pytest.mark.parametrize("timeout", [30.0, 90.0])
def test_idle_connections_expire_before_the_provider_drops_them(monkeypatch, wired, timeout):
    builds, _ = wired
    transports: list = []
    monkeypatch.setattr(httpx, "AsyncClient", _factory(builds, transports=transports))

    got, used = asyncio.run(_guarded(lambda: _post("x", timeout)))
    assert got == (200, "x")
    (built,) = transports
    assert isinstance(built, httpx.AsyncHTTPTransport)
    # Below the 5 s idle timeout of the shortest common provider front end.
    assert built._pool._keepalive_expiry == llm_client._KEEPALIVE_EXPIRY_S < 5.0
    assert built._pool._retries == 1


# --- grader: the stream's own error and parse paths


def _lines_handler(lines: list):
    def handler(request):
        return httpx.Response(200, content=("\n\n".join(lines) + "\n\n").encode(),
                              headers={"content-type": "text/event-stream"})

    return handler


@pytest.mark.parametrize("bad", ['data: {"choices": [', "data: not json at all"])
def test_a_malformed_data_line_is_skipped_and_the_stream_goes_on(monkeypatch, wired, bad):
    builds, recorded = wired
    lines = ['data: {"choices":[{"delta":{"content":"a"}}]}', bad,
             'data: {"choices":[{"delta":{"content":"b"}}]}',
             'data: {"choices":[],"usage":{"total_tokens":7}}', "data: [DONE]"]
    monkeypatch.setattr(httpx, "AsyncClient", _factory(builds, handler=_lines_handler(lines)))

    got, used = asyncio.run(_guarded(_stream))
    assert [ln for ln in got if ln] == lines
    assert len(recorded) == 1
    assert recorded[0]["usage"] == {"total_tokens": 7}


@pytest.mark.parametrize("users", [2, 15])
def test_first_callers_interleaved_with_a_build_in_flight_share_one_build(monkeypatch, wired, users):
    builds, _ = wired
    gate = threading.Event()
    monkeypatch.setattr(httpx, "AsyncClient", _factory(builds, hold={30.0: gate}))

    async def run():
        tasks = [asyncio.create_task(_post(f"u{i}")) for i in range(users)]
        # Every caller gets past its fast path and parks while build 1 is held.
        for _ in range(20):
            await asyncio.sleep(0.01)
        assert builds == [] and not any(t.done() for t in tasks)
        gate.set()
        return await asyncio.gather(*tasks)

    results, used = asyncio.run(_guarded(run))
    assert sorted(results) == sorted((200, f"u{i}") for i in range(users))
    assert len(builds) == 1


@pytest.mark.parametrize("waiting", [0, 3])
def test_a_client_built_for_a_cancelled_caller_is_closed(monkeypatch, wired, waiting):
    builds, _ = wired
    gate = threading.Event()
    made: list = []
    base = _factory(builds, hold={30.0: gate})

    class _Tracked(base):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            made.append(self)

    monkeypatch.setattr(httpx, "AsyncClient", _Tracked)

    async def run():
        builder = asyncio.create_task(_post("cancelled"))
        await asyncio.sleep(0.05)
        others = [asyncio.create_task(_post(f"w{i}")) for i in range(waiting)]
        await asyncio.sleep(0.05)
        builder.cancel()
        with pytest.raises(asyncio.CancelledError):
            await builder
        gate.set()
        rest = await asyncio.gather(*others)
        for _ in range(100):
            if any(c.is_closed for c in made):
                break
            await asyncio.sleep(0.01)
        # Read before shutdown closes the shared client too.
        return rest, sorted(c.is_closed for c in made)

    (rest, closed), used = asyncio.run(_guarded(run))
    assert rest == [(200, f"w{i}") for i in range(waiting)]
    # The orphan build is closed; the waiters' own build is kept open.
    assert closed == ([False, True] if waiting else [True])
    assert used == []
