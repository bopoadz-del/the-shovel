"""A layer that rewrites frames must not be able to end a turn in silence.

Live f44ddef, warranty_maintenance_schedule long ask: HTTP 200, then the
stream closed at 9.6 s after ``start``/``status`` with no ``end`` and no
``error`` (client: "Response ended prematurely"). The identity redactor and
the stage-timings layer wrapped ``guarantee_terminal``, so an exception in
either escaped the guard. These tests drive the real ``_turn_frames`` chain.
"""
import asyncio
import json

from app.routers import chat as chat_mod


def _types(frames):
    out = []
    for raw in frames:
        for line in raw.splitlines():
            if line.startswith("data:"):
                try:
                    out.append(json.loads(line[5:]).get("type"))
                except ValueError:
                    pass
    return out


async def _producer(n_tokens):
    yield 'data: {"type": "start"}\n\n'
    for i in range(n_tokens):
        yield 'data: ' + json.dumps({"type": "token", "content": f"w{i} "}) + '\n\n'
    yield 'data: {"type": "end", "complete": true}\n\n'


class _BrokenRedactor:
    """Raises on the first answer text it is asked to redact."""
    span = 0
    _title_re = None
    _member_re = None

    def __init__(self, fail_after):
        self.calls = 0
        self.fail_after = fail_after

    def text(self, value):
        self.calls += 1
        if self.calls > self.fail_after:
            raise RuntimeError("redactor broke mid-stream")
        return value

    def deep(self, payload):
        return payload


def _chain(stream, identity, timings):
    """The v1 route's chain (timings=True) or the legacy route's (False),
    exactly as ``chat.py`` builds them."""
    from app.core import turn_progress
    from app.routers.chat_watchdog import close_on_escape, guarantee_terminal
    from app.routers.exit_frames import with_answer_exit
    from app.routers.hat_frames import with_hat_signals
    from app.routers.identity_exit import redact_frames

    inner = redact_frames(guarantee_terminal(
        with_hat_signals(with_answer_exit(stream), "warranty schedule"),
        request_id="rid-x"), identity)
    if timings:
        inner = chat_mod._with_stage_timings(inner)
    return close_on_escape(turn_progress.opened(inner), request_id="rid-x")


def _run(stream, identity, timings):
    async def collect():
        got = []
        try:
            async for raw in _chain(stream, identity, timings):
                got.append(raw)
        except Exception as exc:  # noqa: BLE001
            got.append(f"ESCAPED:{type(exc).__name__}")
        return got
    return asyncio.run(collect())


def test_a_redactor_failure_mid_stream_still_ends_with_error_and_end():
    for timings in (True, False):
        for fail_after in (0, 2):
            got = _run(_producer(5), _BrokenRedactor(fail_after), timings)
            assert not any(g.startswith("ESCAPED") for g in got), got
            kinds = _types(got)
            assert kinds[0] == "status", kinds
            assert "error" in kinds and kinds[-1] == "end", kinds


def test_a_stage_timings_failure_still_ends_with_error_and_end(monkeypatch):
    from app.core import turn_timing

    def boom():
        raise ValueError("snapshot broke")
    monkeypatch.setattr(turn_timing, "snapshot", boom)
    got = _run(_producer(1), None, True)
    assert not any(g.startswith("ESCAPED") for g in got), got
    kinds = _types(got)
    assert kinds[-2:] == ["error", "end"], kinds


def test_a_healthy_turn_passes_through_unchanged_with_one_end():
    for n in (1, 7):
        got = _run(_producer(n), None, True)
        kinds = [k for k in _types(got) if k != "hat_signals"]
        assert kinds == ["status", "start"] + ["token"] * n + ["end"], kinds
        end = json.loads(got[-1][6:])
        assert end.get("complete") is True and "timings" in end
    got = _run(_producer(2), None, False)
    assert "timings" not in json.loads(got[-1][6:])


def test_both_streaming_routes_put_the_escape_net_outermost():
    from pathlib import Path
    src = (Path(chat_mod.__file__)).read_text(encoding="utf-8")
    assert src.count("turn_gate.hold_until_done(close_on_escape(turn_progress.opened(") == 2


async def _then_raise(frames, exc):
    for f in frames:
        yield f
    raise exc


def _collect(gen):
    async def go():
        got = []
        try:
            async for raw in gen:
                got.append(raw)
        except BaseException as e:  # noqa: BLE001
            got.append(f"ESCAPED:{type(e).__name__}")
        return got
    return asyncio.run(go())


def test_close_on_escape_adds_nothing_after_a_terminal_was_sent():
    from app.routers.chat_watchdog import close_on_escape
    for last in ('data: {"type": "end"}\n\n', 'data: {"type": "error", "message": "x"}\n\n'):
        frames = ['data: {"type": "start"}\n\n', last]
        got = _collect(close_on_escape(_then_raise(frames, RuntimeError("late")), request_id="r1"))
        assert got == frames


def test_close_on_escape_names_the_request_and_ends_after_a_non_terminal():
    from app.routers.chat_watchdog import ESCAPE_MESSAGE, close_on_escape
    for rid, n in (("r1", 1), ("r2", 3)):
        frames = ['data: {"type": "token", "content": "a"}\n\n'] * n
        got = _collect(close_on_escape(_then_raise(frames, ValueError("v")), request_id=rid))
        assert got[:n] == frames and len(got) == n + 2
        err, end = json.loads(got[n][6:]), json.loads(got[n + 1][6:])
        assert err == {"type": "error", "message": ESCAPE_MESSAGE, "request_id": rid}
        assert end == {"type": "end", "iterations": 0, "sources": [], "tools": []}


def test_close_on_escape_lets_a_cancelled_client_go():
    from app.routers.chat_watchdog import close_on_escape
    got = _collect(close_on_escape(_then_raise(['data: {"type": "start"}\n\n'], asyncio.CancelledError()), request_id="r"))
    assert got[-1] == "ESCAPED:CancelledError" and len(got) == 2


def test_close_on_escape_passes_a_clean_stream_through():
    from app.routers.chat_watchdog import close_on_escape
    for frames in ([], ['data: {"type": "start"}\n\n', ": keepalive\n\n"]):
        async def src(frames=frames):
            for f in frames:
                yield f
        assert _collect(close_on_escape(src(), request_id="r")) == frames
