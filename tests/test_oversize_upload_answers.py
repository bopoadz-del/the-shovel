"""An oversize upload must ANSWER, not drop the connection.

Live evidence (2026-08-17, before the fix): POSTing 60 MB to an upload route
died after 34 s with "Server disconnected without sending a response" — the
browser's word for that is "Failed to fetch", which is exactly what the
oversize guard was written to prevent. Cause: the guard answered 413 from the
Content-Length header while the client was still streaming its body, so the
response never reached a peer whose socket was closed mid-send.

The guard still refuses on Content-Length (it must never spool a 345 MB body
to disk), but now drains the body first, under a time budget, so the client
can read the status.
"""
from __future__ import annotations

import pytest


def _oversize_body(app_module) -> int:
    from app.core import upload_limits
    return upload_limits.request_body_limit() + (8 * 1024 * 1024)


@pytest.mark.asyncio
async def test_drain_returns_true_when_body_is_consumed():
    from app import main as app_main

    chunks = [b"x" * 1024] * 5

    class _Req:
        def stream(self):
            async def gen():
                for c in chunks:
                    yield c
            return gen()

    assert await app_main._drain_request_body(_Req(), seconds=30) is True


@pytest.mark.asyncio
async def test_drain_gives_up_on_its_time_budget_instead_of_pinning_a_worker():
    import asyncio

    from app import main as app_main

    class _SlowReq:
        def stream(self):
            async def gen():
                for _ in range(100):
                    await asyncio.sleep(0.01)
                    yield b"x" * 1024
            return gen()

    # Zero budget: the first deadline check must trip and hand back False so
    # the caller answers anyway rather than draining forever.
    assert await app_main._drain_request_body(_SlowReq(), seconds=0.0) is False


@pytest.mark.asyncio
async def test_client_receives_413_json_not_a_dropped_connection():
    """End-to-end through the real middleware stack."""
    import httpx

    from app import main as app_main
    from app.core import upload_limits

    size = upload_limits.request_body_limit() + (8 * 1024 * 1024)
    body = b"0" * size

    transport = httpx.ASGITransport(app=app_main.app)
    async with httpx.AsyncClient(transport=transport,
                                 base_url="http://testserver") as client:
        res = await client.post(
            "/v1/upload", content=body,
            headers={"content-type": "application/octet-stream",
                     "content-length": str(size)},
        )
    assert res.status_code == 413, res.status_code
    # a real, readable payload — not an empty/aborted response
    assert "too large" in res.text.lower(), res.text[:200]
    assert str(size) in res.text


@pytest.mark.asyncio
async def test_the_body_is_actually_consumed_before_the_413_is_sent():
    """The discriminating fence.

    An in-process ASGI test cannot reproduce a socket reset (the body is
    handed over whole), so asserting "413 came back" passes even on the
    broken code. What actually distinguishes the fix is whether the server
    PULLED the client's body before replying — so count the receive() calls.
    Broken: 1 (or 0). Fixed: every chunk drained, then more_body False.
    """
    from app import main as app_main
    from app.core import upload_limits

    size = upload_limits.request_body_limit() + (8 * 1024 * 1024)
    chunk = b"0" * (1024 * 1024)
    n_chunks = size // len(chunk)
    pulled = {"n": 0}

    async def receive():
        pulled["n"] += 1
        if pulled["n"] <= n_chunks:
            return {"type": "http.request", "body": chunk, "more_body": True}
        return {"type": "http.request", "body": b"", "more_body": False}

    sent = []

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": "POST", "scheme": "https", "path": "/v1/upload",
        "raw_path": b"/v1/upload", "query_string": b"", "root_path": "",
        "client": ("test", 1234), "server": ("testserver", 443),
        "headers": [
            (b"host", b"testserver"),
            (b"content-type", b"application/octet-stream"),
            (b"content-length", str(size).encode()),
        ],
    }
    await app_main.app(scope, receive, send)

    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    assert status == 413, status
    assert pulled["n"] >= n_chunks, (
        f"only {pulled['n']} of {n_chunks} body chunks were read before the "
        "413 — the client would be cut off mid-send and see a transport "
        "error instead of this status"
    )


@pytest.mark.asyncio
async def test_within_limit_bodies_are_not_intercepted():
    """The guard must only fire on bodies no route could accept."""
    import httpx

    from app import main as app_main

    transport = httpx.ASGITransport(app=app_main.app)
    async with httpx.AsyncClient(transport=transport,
                                 base_url="http://testserver") as client:
        res = await client.post(
            "/v1/upload", content=b"small",
            headers={"content-type": "application/octet-stream",
                     "content-length": "5"},
        )
    assert res.status_code != 413, res.text[:200]
