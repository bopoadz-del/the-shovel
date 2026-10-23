"""TIMING log lines carry the process's resident memory.

The reading comes from /proc on Linux (the deployed container), from psutil
where it is installed, and is -1 when neither is readable. Each path is
exercised with a supplied source so the test means the same on every OS.
"""
import builtins
import io
import sys
import types
from tests._fixture_values import STUB_TOKEN_06


def _proc_status(monkeypatch, text):
    real_open = builtins.open

    def fake_open(path, *a, **k):
        if str(path) == "/proc/self/status":
            if text is None:
                raise OSError("no /proc here")
            return io.StringIO(text)
        return real_open(path, *a, **k)

    monkeypatch.setattr(builtins, "open", fake_open)


def test_rss_is_read_from_proc_status(monkeypatch):
    from app.agents.runtime import _rss_mb

    _proc_status(monkeypatch, "Name:\tpython\nVmRSS:\t  204800 kB\nVmSwap:\t0 kB\n")
    assert _rss_mb() == 200


def test_rss_falls_back_to_psutil_without_proc(monkeypatch):
    from app.agents.runtime import _rss_mb

    _proc_status(monkeypatch, None)
    fake = types.SimpleNamespace(Process=lambda: types.SimpleNamespace(
        memory_info=lambda: types.SimpleNamespace(rss=300 * 1024 * 1024)))
    monkeypatch.setitem(sys.modules, "psutil", fake)
    assert _rss_mb() == 300


def test_rss_is_minus_one_when_nothing_is_readable(monkeypatch):
    from app.agents.runtime import _rss_mb

    _proc_status(monkeypatch, None)
    monkeypatch.setitem(sys.modules, "psutil", None)  # import fails
    assert _rss_mb() == -1


def test_every_timing_line_ends_with_rss(monkeypatch):
    from app.agents import runtime as rt

    calls = []
    monkeypatch.setattr(rt._LOG, "warning", lambda fmt, *args: calls.append(fmt % args))
    monkeypatch.setattr(rt, "_rss_mb", lambda: 123)
    rt._timing_log("TIMING chat_stream iter=%d call=%.1fs", 2, 1.5)
    assert calls == ["TIMING chat_stream iter=2 call=1.5s rss=123MB"]


def test_a_timed_chat_turn_logs_rss_on_every_timing_line(monkeypatch):
    import asyncio

    from app.agents import runtime as rt

    monkeypatch.setenv("AGENT_TIMING_LOG", "1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_06)
    rt.load_agents()
    agent = rt.AGENT_REGISTRY["project-assistant"]

    async def scripted(self, *args, **kw):
        return {"status": "success", "raw": {"model": "scripted"},
                "choice": {"message": {"content": "Done.", "tool_calls": []}}}

    monkeypatch.setattr(rt.Agent, "_call_llm", scripted)
    monkeypatch.setattr(rt, "_rss_mb", lambda: 77)
    lines = []
    monkeypatch.setattr(rt._LOG, "warning", lambda fmt, *args, **kw: lines.append(fmt % args if args else fmt))

    async def turn():
        async for _ in agent.chat_stream("What is the lap length?", user_id="u-timing"):
            pass

    asyncio.run(turn())
    timing = [ln for ln in lines if ln.startswith("TIMING")]
    assert timing and all(ln.endswith(" rss=77MB") for ln in timing)
