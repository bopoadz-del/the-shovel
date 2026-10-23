"""The production LOOP_STALL sources run off the event loop.

Production logged 220 LOOP_STALL events in a day (268 ms to 21.7 s), each
with a [loop] stack naming the blocking call. Each such call is replaced here
by a probe that records the thread it ran on; driving the async entry point
must never run it on the event loop's own thread. On main these fail: the
calls ran inline on the loop.
"""
from __future__ import annotations

import asyncio
import threading

import pytest


class _Probe:
    def __init__(self, result):
        self.result = result
        self.threads: list[int] = []

    def __call__(self, *args, **kwargs):
        self.threads.append(threading.get_ident())
        return self.result


def _drive(coro_factory):
    """Run the coroutine on a fresh loop; return the loop thread's ident."""
    loop_thread: list[int] = []

    async def main():
        loop_thread.append(threading.get_ident())
        return await coro_factory()

    out = asyncio.run(main())
    return loop_thread[0], out


def test_the_driver_resolves_predefined_files_off_the_loop(monkeypatch):
    from app.agents.driver import routes
    from app.routers import chat

    action = next(iter(routes.catalogue()))
    probe = _Probe(({}, {"question": "which file?"}))
    monkeypatch.setattr(chat, "_resolve_predefined_file_params", probe)
    loop_ident, out = _drive(lambda: routes.run({"workflow": action}, "run it", None, None, None))
    assert out.get("needs_input"), out
    assert probe.threads and loop_ident not in probe.threads


def test_the_calculator_check_and_run_happen_off_the_loop(monkeypatch):
    from app.agents import runtime
    from app.agents.base import tools
    from app.agents.core.tool_registry import ToolCall
    from app.lib import construction_formulas

    refusal = _Probe(None)
    run = _Probe({"status": "success", "result": {"value": 1}})
    monkeypatch.setattr(runtime, "_defaulted_run_refusal", refusal)
    monkeypatch.setattr(construction_formulas, "run_calculation", run)
    call = ToolCall(agent=None, name="construction_calc",
                    args={"calculation": "productivity_rate", "params": {"output_quantity": 120, "labor_hours": 40}},
                    user_message="productivity of 120 m2 in 40 man-hours")
    loop_ident, _ = _drive(lambda: tools.handle_construction_calc(call))
    assert refusal.threads and loop_ident not in refusal.threads
    assert run.threads and loop_ident not in run.threads


def test_hydration_reads_and_writes_off_the_loop(monkeypatch):
    from app.core import hydration_store
    from app.core.learning import hydration

    probes = {
        "_projects_active_in_window": _Probe(["p1"]),
        "_filter_existing_projects": _Probe(["p1"]),
        "_collect_project_messages": _Probe([]),
        "_heuristic_project_summary": _Probe("summary"),
        "_writeback_for_next_chat": _Probe({}),
        "_heuristic_global_summary": _Probe("global"),
    }
    for name, probe in probes.items():
        monkeypatch.setattr(hydration, name, probe)
    record = _Probe(None)
    monkeypatch.setattr(hydration_store, "record_run", record)

    async def offline(*_a, **_k):
        return "", "offline_template"

    monkeypatch.setattr(hydration, "_summarize_project", offline)
    monkeypatch.setattr(hydration, "_summarize_global", offline)
    loop_ident, _ = _drive(lambda: hydration.run(target_date="2026-10-15"))
    ran = {name: p.threads for name, p in probes.items() if p.threads}
    assert ran, "hydration called none of its helpers"
    for name, threads in ran.items():
        assert loop_ident not in threads, f"{name} ran on the event loop"
    assert record.threads and loop_ident not in record.threads


def test_the_calculator_schema_is_a_fresh_copy_for_each_caller():
    from app.agents.driver import tools as driver_tools

    first = driver_tools.calculator_schema(None)
    first.clear()
    again = driver_tools.calculator_schema(None)
    assert again, "a caller's change reached the cached schema"


def test_the_calculator_schema_follows_a_newly_registered_formula(monkeypatch):
    from app.agents.driver import tools as driver_tools
    from app.lib import formula_registry

    before = str(driver_tools.calculator_schema(None))
    assert "zz_probe_formula" not in before

    @formula_registry.formula(owner="base", display_name="Probe formula", description="A probe.",
                              inputs={"x": formula_registry.Param("nr", 0, 10, label="probe input")},
                              outputs={"y": "nr"}, name="zz_probe_formula")
    def zz_probe_formula(x: float) -> dict:
        return {"y": x}

    try:
        assert "zz_probe_formula" in str(driver_tools.calculator_schema(None))
    finally:
        formula_registry._REGISTRY.pop("zz_probe_formula", None)
    assert "zz_probe_formula" not in str(driver_tools.calculator_schema(None))


@pytest.fixture(autouse=True)
def _fresh_schema_cache():
    from app.agents.driver import tools as driver_tools

    cache = getattr(driver_tools, "_SCHEMA_CACHE", {})
    cache.clear()
    yield
    cache.clear()


def test_the_driver_exit_check_runs_off_the_loop(monkeypatch):
    """Every streamed piece is exit-checked (citations, figures, identity):
    CPU work that ran inline on the loop for each sentence."""
    from app.agents.driver import llm, loop
    from app.agents.runtime import Agent

    async def fake_stream(agent, messages, offered, model="", timeout=120.0):
        yield "content", "The slab is cast in two pours. "
        yield "done", {"status": "success", "message": {"content": "The slab is cast in two pours. "}}

    monkeypatch.setattr(llm, "stream", fake_stream)
    real = loop.exit_check
    threads: list[int] = []

    def probe(*args, **kwargs):
        threads.append(threading.get_ident())
        return real(*args, **kwargs)

    monkeypatch.setattr(loop, "exit_check", probe)
    agent = Agent(name="loop-probe", description="probe", system_prompt="", allowed_blocks=[])

    async def run():
        return [e async for e in loop.stream(agent, user_message="How is the slab cast?")]

    loop_ident, events = _drive(run)
    assert any(e.get("type") == "end" for e in events)
    assert threads and loop_ident not in threads


def test_the_schedule_cpm_pass_runs_off_the_loop(monkeypatch):
    from app.containers.construction import schedule

    container_cls = next(v for v in vars(schedule).values()
                         if isinstance(v, type) and hasattr(v, "_attach_cpm_to_activities"))
    threads: list[int] = []

    def probe(self, activities, start_date):
        threads.append(threading.get_ident())
        return activities, {}, None

    monkeypatch.setattr(container_cls, "_attach_cpm_to_activities", probe)
    from app.containers.construction import ConstructionContainer

    loop_ident, _ = _drive(lambda: ConstructionContainer().generate_wbs({}, {"brief": "A two-storey office block."}))
    assert threads and loop_ident not in threads
