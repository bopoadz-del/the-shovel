"""A turn ends with an answer or with a reason — never with silence.

F-SILENT-1, ``FLEET_OPS/artifacts/gate_battery_13b2bf7_2026-08-31.md``,
quoted verbatim:

    a turn that ends with neither an answer nor an error (B6, first pass).
    119 seconds, three events (``start``, ``route``, ``start``), then
    nothing: no tokens, no ``end`` content, no error, HTTP 200. Recorded
    even though the retest passed, because a silent zero-output turn is the
    failure mode #456/#462 were built to end, arriving by another door.

WHY THE EXISTING GUARD DID NOT CATCH IT. ``Agent.chat_stream`` already wraps
``_chat_stream_impl`` with a wall-clock deadline, a heartbeat and a trailing
safety net that synthesises ``end``. That guard begins INSIDE the agent. B6's
last event was the agent's own ``start``, and the turn died with the router's
generator still holding the connection -- past ``route``, outside the agent's
net. Any path the router takes that is not ``Agent.chat_stream`` (the
predefined-workflow path, a routing branch that returns early, an exception
in the router's own frame) has no guard at all.

So the guard is applied where the CONNECTION is, at the SSE boundary, which
is the only place that can see the whole turn. It is deliberately a second
net rather than a replacement: its default timeout sits ABOVE the agent's, so
on the agent path the agent's own structured timeout still wins and this one
never fires. It exists for the paths that have nothing.

WHAT IT WILL NOT DO. It cannot make a stalled turn answer. It converts an
unexplained silence into a named failure, which is the difference between a
bug you can find and a bug you cannot.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from typing import AsyncIterator, Callable, Optional

_LOG = logging.getLogger(__name__)

#: SSE event types that legitimately end a turn.
TERMINAL_TYPES = frozenset({"end", "error"})

#: Shown in the bubble so a failed turn is never an empty one.
SILENT_TURN_FALLBACK = (
    "The assistant did not produce an answer for this turn."
)

DEFAULT_WATCHDOG_SECONDS = 300.0


def watchdog_enabled() -> bool:
    """ON by default; ``CHAT_SSE_WATCHDOG=0`` is the kill switch."""
    return str(os.getenv("CHAT_SSE_WATCHDOG", "1")).strip().lower() not in (
        "0", "false", "no", "off",
    )


def watchdog_seconds() -> float:
    """Wall clock for a whole turn at the SSE boundary.

    Above ``CHAT_STREAM_TIMEOUT_SECONDS`` (the agent's own, default 240) on
    purpose: where the agent has a guard, the agent's structured timeout must
    be what the user sees. A bad value falls back to the default rather than
    breaking the stream -- a watchdog that raises is worse than none.
    """
    try:
        value = float(os.getenv("CHAT_SSE_WATCHDOG_SECONDS") or DEFAULT_WATCHDOG_SECONDS)
    except (TypeError, ValueError):
        return DEFAULT_WATCHDOG_SECONDS
    return value if value > 0 else DEFAULT_WATCHDOG_SECONDS


def _end_carries_answer(raw: str) -> bool:
    """True when an ``end`` frame carries the answer itself: the UI renders a
    non-empty ``content`` on ``end`` in place of the streamed tokens."""
    for line in (raw or "").splitlines():
        line = line.strip()
        if line.startswith("data:"):
            try:
                body = json.loads(line[len("data:"):].strip())
            except (ValueError, TypeError) as exc:
                # Not our JSON: it carries no answer, so the empty-end guard
                # applies. Logged, because a malformed end frame is a fault.
                _LOG.warning("sse watchdog: unreadable end frame (%s)", type(exc).__name__)
                return False
            content = body.get("content") if isinstance(body, dict) else None
            return isinstance(content, str) and bool(content.strip())
    return False


def frame(payload: dict) -> str:
    return f"data: {json.dumps(payload, default=str)}\n\n"


def event_type(raw: str) -> Optional[str]:
    """The ``type`` of one SSE frame, or None when it is not one we wrote.

    Tolerant on purpose: a heartbeat comment, a keep-alive newline or a frame
    from another producer must not make the watchdog think a turn ended.
    """
    if not raw:
        return None
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        body = line[len("data:"):].strip()
        if not body or body == "[DONE]":
            return None
        try:
            data = json.loads(body)
        except ValueError:
            return None
        if isinstance(data, dict):
            got = data.get("type")
            return str(got) if got is not None else None
    return None


#: What a user sees when a turn failed for a reason that is ours to know.
PROVIDER_FAILURE_MESSAGE = (
    "The assistant service is temporarily unavailable. This is a problem on our "
    "side, not with your question. Please try again in a minute."
)

# An error text that names the plumbing: a provider, an HTTP status, an
# upstream JSON body, an exception or an environment variable. Live on the
# fallback path: when DeepSeek AND OpenRouter both failed the user was shown
# `openrouter HTTP 500: {"error":{"message": ...upstream body...}}`.
_PLUMBING_RE = re.compile(
    r"(?i)\b(?:deepseek|openrouter|moonshot|kimi|groq|ollama)\b"
    r"|\bHTTP\s*\d{3}\b|\bLLM call\b|[{}]|Traceback|\bErrno\b"
    r"|\b[A-Z][A-Z0-9]*_(?:API_KEY|KEY|SECRET|TOKEN|URL)\b"
    r"|\b\w*(?:Error|Exception|Timeout)\b\s*[:(]|\bcrashed\b"
)


def user_safe_error(message: object) -> str:
    """The message itself when it is meant for a user; otherwise one plain
    sentence. The request_id travels in its own field, so nothing is lost for
    support -- the original is in the log under that id."""
    text = str(message or "")
    if text and not _PLUMBING_RE.search(text):
        return text
    return PROVIDER_FAILURE_MESSAGE


def sanitize_error_frame(raw: str, request_id: str = "") -> str:
    """Rewrite one SSE ``error`` frame so no plumbing reaches the screen."""
    if not isinstance(raw, str) or '"error"' not in raw:
        return raw
    body = raw[5:].strip() if raw.startswith("data:") else raw.strip()
    try:
        data = json.loads(body)
    except ValueError:
        return raw
    if not isinstance(data, dict) or data.get("type") != "error":
        return raw
    original = data.get("message")
    safe = user_safe_error(original)
    if safe == original:
        return raw
    _LOG.warning("sse: replaced a plumbing error for the user (request_id=%s): %s",
                 data.get("request_id") or request_id, str(original)[:500])
    data["message"] = safe
    return frame(data)


async def guarantee_terminal(
    frames: AsyncIterator[str],
    *,
    request_id: str = "",
    timeout_s: Optional[float] = None,
    clock: Callable[[], float] = time.monotonic,
) -> AsyncIterator[str]:
    """Pass frames through; guarantee the turn ends with ``end`` or ``error``.

    Three exits, all named:

    * the producer finishes having emitted a terminal frame -- nothing added;
    * the producer finishes WITHOUT one (B6) -- a token fallback if the
      bubble is still empty, then an ``error`` naming the last event seen,
      then ``end``;
    * the wall clock runs out -- the same shape, naming the timeout.

    The last-event name is carried because "no answer and no error" is not
    actionable while "ended after ``start`` with no answer" says where to
    look. B6's answer to that question was ``start``.
    """
    if not watchdog_enabled():
        async for raw in frames:
            yield sanitize_error_frame(raw, request_id)
        return

    limit = watchdog_seconds() if timeout_s is None else timeout_s
    deadline = clock() + limit
    saw_terminal = False
    saw_token = False
    last_type = "nothing"
    iterator = frames.__aiter__()

    def _close(reason: str) -> list[str]:
        out: list[str] = []
        if not saw_token:
            out.append(frame({"type": "token", "content": SILENT_TURN_FALLBACK}))
        out.append(frame({
            "type": "error",
            "message": reason,
            "request_id": request_id,
        }))
        out.append(frame({"type": "end", "iterations": 0, "sources": [],
                          "tools": []}))
        return out

    def _timed_out() -> list[str]:
        _LOG.warning(
            "sse watchdog: turn exceeded %.0fs after %r (request_id=%s)",
            limit, last_type, request_id,
        )
        return _close(
            "Response timeout — the turn produced no answer within "
            "%.0fs; the last event was %r." % (limit, last_type)
        )

    try:
        while True:
            remaining = deadline - clock()
            if remaining <= 0:
                for out in _timed_out():
                    yield out
                saw_terminal = True
                return
            try:
                raw = await asyncio.wait_for(iterator.__anext__(), timeout=remaining)
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                # The wait was bounded by what was left of the ABSOLUTE
                # deadline, so its expiry IS the deadline. It is not retried:
                # wait_for cancelled the pending ``__anext__``, which finalises
                # an async-generator producer, so a second read would only see
                # StopAsyncIteration and misreport the turn as "ended with no
                # answer". The event loop's timer may also fire up to one clock
                # tick before ``clock()`` agrees (Windows' tick is ~15 ms), so
                # re-checking ``remaining`` here is what made it misreport.
                for out in _timed_out():
                    yield out
                saw_terminal = True
                return
            kind = event_type(raw)
            if kind == "end" and not saw_token and not saw_terminal and not _end_carries_answer(raw):
                # An `end` with no answer before it and none on it is an
                # empty answer (LANE3: a driver turn ended with no text and no
                # mode; LANE4: no final answer after a tool call). The bubble
                # gets the fallback and an error naming where the turn stopped,
                # then the producer's own `end`, so its sources still arrive.
                _LOG.warning("sse watchdog: turn ended empty after %r (request_id=%s)",
                             last_type, request_id)
                yield frame({"type": "token", "content": SILENT_TURN_FALLBACK})
                yield frame({
                    "type": "error",
                    "message": ("The turn ended after %r with no answer. This is a platform "
                                "fault, not a refusal; please retry." % last_type),
                    "request_id": request_id,
                })
                saw_token = True
            if kind:
                last_type = kind
                if kind == "token":
                    saw_token = True
                if kind in TERMINAL_TYPES:
                    saw_terminal = True
            yield sanitize_error_frame(raw, request_id)
    # No `except asyncio.CancelledError: raise` clause here, deliberately: on
    # Python 3.8+ CancelledError inherits from BaseException, not Exception,
    # so the handler below CANNOT swallow it and a client that went away
    # still cancels cleanly. Verified rather than assumed -- and fenced, so
    # that widening this to `except BaseException` (which WOULD answer an
    # absent client and eat the cancellation) fails a test.
    except Exception as exc:  # noqa: BLE001 - the last net must not have holes
        _LOG.exception("sse watchdog: producer escaped (request_id=%s)", request_id)
        for out in _close(
            "The assistant is temporarily unavailable. Please try again. "
            "(%s after %r)" % (type(exc).__name__, last_type)
        ):
            yield out
        return

    if not saw_terminal:
        _LOG.warning(
            "sse watchdog: turn ended with no terminal event after %r "
            "(request_id=%s)", last_type, request_id,
        )
        for out in _close(
            "The turn ended after %r with no answer and no error. This is a "
            "platform fault, not a refusal — please retry." % last_type
        ):
            yield out


#: The ``error`` text when a layer OUTSIDE the terminal guard fails.
ESCAPE_MESSAGE = "The assistant is temporarily unavailable. Please try again."


async def close_on_escape(frames: AsyncIterator[str], *, request_id: str = "") -> AsyncIterator[str]:
    """The outermost net: a layer wrapped around ``guarantee_terminal``
    (identity redaction, stage timings, the ``accepted`` status) that raises
    mid-stream still ends the turn with ``error`` and ``end``.

    Live f44ddef (warranty_maintenance_schedule long ask): HTTP 200, then the
    connection closed at 9.6 s with no ``end`` and no ``error`` ("Response
    ended prematurely"). ``guarantee_terminal`` cannot see an exception raised
    by a layer that wraps it; such an exception reached the server, which
    drops the connection. The frames added here are fixed text, so nothing
    the redactor would have removed can pass through them. ``CancelledError``
    (a client that went away) is not caught, as in ``guarantee_terminal``.
    """
    saw_terminal = False
    try:
        async for raw in frames:
            if event_type(raw) in TERMINAL_TYPES:
                saw_terminal = True
            yield raw
    except Exception:  # noqa: BLE001 - the last net must not have holes
        _LOG.exception("sse: a layer outside the terminal guard failed (request_id=%s)", request_id)
        if not saw_terminal:
            yield frame({"type": "error", "message": ESCAPE_MESSAGE, "request_id": request_id})
            yield frame({"type": "end", "iterations": 0, "sources": [], "tools": []})
