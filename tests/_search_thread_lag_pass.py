"""NON-GATING measurement: event-loop lag while 1, 2 and 4 threads run the real search.

Run as its own process (``python -m tests._search_thread_lag_pass <data_dir>``) by
``tests/test_search_thread_loop_lag_is_measured.py`` on the timing job's runner
(Linux, PostgreSQL in CI). It answers one question with numbers: on THAT
runner, does document search running on N threads of the server process delay
the event loop's 20 ms heartbeat, and by how much?

For each mode and each thread count, the loop runs a 20 ms heartbeat (an
``asyncio.sleep`` whose lateness is recorded every tick) while N plain threads
each run the real search over the synthetic corpus the gated load test seeds,
back to back, for ``WINDOW_S`` seconds. Modes:

* ``thread``: the search runs in the calling thread
  (``doc_index._search_project_documents_traced``), as it does in the server
  when the retrieval worker process is off.
* ``process``: the calling thread hands the same search to the retrieval
  worker process pool (``retrieval_worker.call_sync`` +
  ``search_documents_job``), as it does when ``RAG_RETRIEVAL_PROCESS`` is on.
  The waiting thread holds no interpreter lock while the worker computes.

Prints one line ``SEARCH_THREAD_LAG_RESULT {json}``: per mode and thread
count, heartbeat lag p50 / p95 / max (ms), ticks, searches done and CPU per
search in the calling threads (ms); plus the switch interval, CPU count, the
CPUs this process may run on, the Python version, the database driver's
implementation and whether the worker-process path is on by the environment.
Nothing is asserted. Not collected by pytest (no ``test_`` prefix).
Synthetic corpus only.
"""
from __future__ import annotations

import asyncio
import json
import os
import platform
import statistics
import sys
import threading
import time

#: Heartbeat period, as the gated test's monitor uses.
TICK_MS = 20
#: Seconds each (mode, thread count) cell is measured.
WINDOW_S = 4.0
#: Thread counts measured, in order.
THREAD_COUNTS = (1, 2, 4)


def lag_summary(lags_ms: list) -> dict:
    """p50 / p95 / max of the heartbeat lags (ms), rounded; empty -> zeros."""
    if not lags_ms:
        return {"ticks": 0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    ordered = sorted(lags_ms)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
    return {"ticks": len(ordered), "p50_ms": round(statistics.median(ordered), 1),
            "p95_ms": round(p95, 1), "max_ms": round(ordered[-1], 1)}


async def heartbeat(stop: threading.Event, tick_ms: int = TICK_MS) -> list:
    """Sleep ``tick_ms`` until ``stop`` is set; return how late each wake was (ms)."""
    lags = []
    tick = tick_ms / 1000.0
    while not stop.is_set():
        t = time.perf_counter()
        await asyncio.sleep(tick)
        lags.append(max(0.0, (time.perf_counter() - t - tick) * 1000.0))
    return lags


def _searcher(search, queries, stop: threading.Event, out: list) -> None:
    n, cpu0 = 0, time.thread_time()
    while not stop.is_set():
        search(queries[n % len(queries)])
        n += 1
    out.append((n, (time.thread_time() - cpu0) * 1000.0))


def measure(search, queries, threads: int, window_s: float = WINDOW_S) -> dict:
    """Heartbeat lag on a fresh event loop while ``threads`` threads run ``search``."""
    stop = threading.Event()
    done: list = []
    workers = [threading.Thread(target=_searcher, args=(search, queries, stop, done),
                                name=f"lag-search-{i}", daemon=True) for i in range(threads)]

    async def run() -> list:
        for w in workers:
            w.start()
        timer = asyncio.get_running_loop().call_later(window_s, stop.set)
        try:
            return await heartbeat(stop)
        finally:
            timer.cancel()

    lags = asyncio.run(run())
    stop.set()
    for w in workers:
        w.join(timeout=60)
    searches = sum(n for n, _ in done)
    cpu_ms = sum(c for _, c in done)
    cell = lag_summary(lags)
    cell.update({"threads": threads, "searches": searches,
                 "caller_cpu_ms_per_search": round(cpu_ms / searches, 1) if searches else None})
    return cell


def _driver_impl() -> str:
    try:
        import psycopg

        return str(psycopg.pq.__impl__)
    except Exception as exc:  # reported, never raised: this pass only logs
        return f"unavailable ({type(exc).__name__})"


def environment() -> dict:
    from app.core.rag import retrieval_worker

    try:
        affinity = len(os.sched_getaffinity(0))
    except AttributeError:
        affinity = None
    return {"python": platform.python_version(), "platform": sys.platform,
            "switch_interval_ms": round(sys.getswitchinterval() * 1000.0, 3),
            "cpu_count": os.cpu_count(), "cpu_affinity": affinity,
            "vcpus_for_workers": retrieval_worker.available_vcpus(),
            "db": "postgresql" if (os.getenv("DATABASE_URL") or "").startswith("postgres") else "sqlite",
            "psycopg_impl": _driver_impl(),
            "worker_process_on_by_env": retrieval_worker.enabled()}


def main(data_dir: str) -> dict:
    from tests import test_the_loop_is_never_blocked_under_load as load

    os.environ.update(load.load_env(data_dir))
    load.seed_corpus()

    from app.core import doc_index
    from app.core.rag import retrieval_worker

    queries = list(load._QUESTIONS)

    def in_thread(q: str):
        return doc_index._search_project_documents_traced(load.PID, q, 5)

    def in_process(q: str):
        return retrieval_worker.call_sync(retrieval_worker.search_documents_job,
                                          load.PID, q, 5, None, frozenset())

    result = {"env": environment(), "window_s": WINDOW_S, "tick_ms": TICK_MS, "modes": {}}
    in_thread(queries[0])  # warm: imports and caches are not the property measured
    result["idle"] = measure(in_thread, queries, 0)  # control: heartbeat, no search running
    result["modes"]["thread"] = [measure(in_thread, queries, n) for n in THREAD_COUNTS]
    try:
        retrieval_worker.warm_all()
        in_process(queries[0])
        result["modes"]["process"] = [measure(in_process, queries, n) for n in THREAD_COUNTS]
        result["worker_processes"] = retrieval_worker._workers()
    except Exception as exc:  # reported, never raised: this pass only logs
        result["modes"]["process"] = f"not measured: {type(exc).__name__}: {str(exc)[:300]}"
    finally:
        pool = retrieval_worker._POOL
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
    return result


if __name__ == "__main__":
    print("SEARCH_THREAD_LAG_RESULT " + json.dumps(main(sys.argv[1])), flush=True)
