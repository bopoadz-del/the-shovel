"""The NON-GATING second pass: the fifteen-user load with the live capture ON.

Run as its own process (``python -m tests._loop_capture_pass <data_dir>``) by
``test_loop_stall_live_capture.test_the_capture_runs_over_the_fifteen_user_load``.
The gated measurement (tests/test_the_loop_is_never_blocked_under_load.py)
keeps the capture OFF; this pass runs the same load with it ON and only
prints what it saw. It is a child process because the capture's C timer reads
running threads' frames without the GIL, and once crashed a timing job
(#396 9370c0c, exit 139): a crash here ends this child, never a test process.

Prints one line ``LOOP_CAPTURE_RESULT {json}`` and nothing is asserted here.
The same line carries the cost of the monitor's CPU snapshot with and without
the process stat read (``snapshot_us``): TEN figures each way from one run,
taken in interleaved batches so drift on the runner falls on both sides
alike. The snapshot runs after the heartbeat is stamped, so this extra read
is inside the measured window; ten figures against the 250 ms limit show
what it can move.

Not collected by pytest (no ``test_`` prefix). Synthetic corpus only.
"""
from __future__ import annotations

import asyncio
import json
import os
import statistics
import sys
import time

#: Snapshots timed per batch; each batch's median is one figure.
SNAPSHOT_REPS = 20
#: Figures per variant (batches, interleaved threads-only / threads-and-stat).
SNAPSHOT_FIGURES = 10


def _median_us(fn, reps: int = SNAPSHOT_REPS) -> float:
    costs = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        costs.append((time.perf_counter() - t) * 1e6)
    return statistics.median(costs)


def snapshot_figures(without, with_, figures: int = SNAPSHOT_FIGURES) -> dict:
    """``figures`` batch medians (us) for each variant, batches interleaved,
    and the per-batch extra cost of ``with_`` over ``without``."""
    a, b = [], []
    for _ in range(figures):
        a.append(round(_median_us(without), 1))
        b.append(round(_median_us(with_), 1))
    return {"threads_only": a, "threads_and_stat": b,
            "extra": [round(y - x, 1) for x, y in zip(a, b)]}


def main(data_dir: str) -> dict:
    from tests import test_the_loop_is_never_blocked_under_load as load

    os.environ.update(load.load_env(data_dir))
    load.seed_corpus()

    from app.agents import runtime as rt
    from app.core import loop_watchdog as wd

    rt.load_agents()
    rt.Agent._call_llm = load._scripted_llm()
    rt.project_is_rag_ready = lambda pid: True
    agent = rt.AGENT_REGISTRY["project-assistant"]

    async def user(n: int) -> None:
        history: list = []
        for t in range(load.TURNS):
            q = load._QUESTIONS[(n + t) % len(load._QUESTIONS)]
            async for _ in agent.chat_stream(q, project_id=load.PID, user_id=f"capture-user-{n}",
                                             conversation_id=f"capture-conv-{n}", history=history):
                pass
            history.append({"role": "user", "content": q})

    async def run() -> "wd.LagMonitor":
        async for _ in agent.chat_stream(load._QUESTIONS[0], project_id=load.PID, user_id="capture-warm",
                                         conversation_id="capture-warm", history=[]):
            pass
        mon = wd.LagMonitor(asyncio.get_running_loop(), threshold_ms=load.LIMIT_MS, tick_ms=20,
                            log=False, live_ms=wd.DEFAULT_LIVE_MS).start()
        try:
            await asyncio.gather(*(user(n) for n in range(load.USERS)))
        finally:
            mon.stop()
        return mon

    mon = asyncio.run(run())
    worst = max(mon.live_dumps, default=(0.0, ""))
    return {
        "load_max_ms": round(mon.max_lag_ms),
        "stalls": len(mon.stalls),
        "live_dumps": len(mon.live_dumps),
        "near_misses": mon.live_near_misses,
        "wake_late_max_ms": round(mon.max_wake_late_ms),
        "worst_dump_ms": round(worst[0]),
        "worst_dump": worst[1][:2000],
        "snapshot_us": snapshot_figures(wd.thread_cpu_ms, wd.cpu_snapshot),
    }


if __name__ == "__main__":
    print("LOOP_CAPTURE_RESULT " + json.dumps(main(sys.argv[1])), flush=True)
