"""Startup must open the port before loading the heavy models.

2026-08-11: the service could not boot on a 512Mi instance. The lifespan
warm-loaded the bge-small embedder inline, BEFORE FastAPI opened the port, and
with no persistent HF cache that warm-up also downloads the model. Startup
outran the platform's port-scan window ("No open ports detected"), the platform
restarted the container while the first boot was still loading, and two
processes each holding torch + a transformer exceeded the instance memory. The
kill looked like a memory leak; the cause was a slow startup.

These tests pin the fix: warm-up is a background task by default, so a slow or
failing model load can never keep the app from becoming reachable.
"""
from __future__ import annotations

import asyncio
import threading

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app


def test_a_hanging_model_load_does_not_prevent_serving(monkeypatch):
    """The defect shape: if warm-up blocks, the app never becomes reachable.

    What discriminates old from new is ORDERING, not wall-clock: startup must
    complete while the fake model load is still stuck. Inline warm-up cannot
    return from startup until the load finishes, so `finished` would already be
    set by the time the client is usable, and the assertion below fails. That
    keeps the test honest without racing the app's real ~45s startup against a
    timer, and keeps it well inside the suite's 120s per-test timeout.
    """
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    # Only needs to outlast the assertions below; the test releases it long
    # before this expires. The cap exists so a bug here can never hang CI.
    HANG_SECONDS = 30

    def _hang():
        # Simulates a slow first-time model download on a host with no cache.
        started.set()
        release.wait(timeout=HANG_SECONDS)
        finished.set()

    monkeypatch.setattr(main_module, "_warm_embedder", _hang)
    monkeypatch.setattr(main_module, "_warm_safety_detector", lambda: None)

    try:
        with TestClient(app) as client:      # runs the full lifespan
            assert started.wait(timeout=30), "warm-up never started"
            assert not finished.is_set(), (
                "startup did not finish until the model load did — the port "
                "was held closed behind the warm-up."
            )
            # Serving WHILE the model load is still stuck.
            assert client.get("/livez").status_code == 200
            # Release INSIDE the context: lifespan shutdown joins the warm-up
            # thread, so leaving it stuck would deadlock teardown, not the code
            # under test.
            release.set()
    finally:
        release.set()


def test_a_failing_model_load_does_not_break_startup(monkeypatch):
    def _boom():
        raise RuntimeError("model repo unreachable")

    monkeypatch.setattr(main_module, "_warm_embedder", _boom)
    monkeypatch.setattr(main_module, "_warm_safety_detector", _boom)

    with TestClient(app) as client:
        assert client.get("/livez").status_code == 200


def test_one_warm_failure_does_not_skip_the_other(monkeypatch):
    """Isolation: the detector failing must not stop the embedder loading."""
    calls = []

    def _boom():
        calls.append("detector")
        raise RuntimeError("no weights")

    monkeypatch.setattr(main_module, "_warm_safety_detector", _boom)
    monkeypatch.setattr(main_module, "_warm_embedder", lambda: calls.append("embedder"))

    asyncio.run(main_module._warm_models())
    assert calls == ["detector", "embedder"], calls


def test_startup_does_not_seed_the_knowledge_base_inline(monkeypatch):
    """The knowledge seed must not run before the port opens.

    This is a DIFFERENT shape from the tests above, and it exists because those
    tests all passed while this bug was live. They only proved `_warm_models`
    was backgrounded — the lifespan was separately calling seed_knowledge()
    inline, which on an unseeded database takes 42s and +677 MB because
    indexing docs/knowledge/*.md constructs the embedder. Measured 2026-08-11:
    that alone kept the port shut past the platform's scan window and exceeded
    a 512Mi instance, and because the restart re-ran the same seed, the deploy
    failed identically every time instead of eventually converging.

    So this asserts on the SEED specifically, not on warm-up in general.
    """
    seeded_during_startup = []

    def _record():
        seeded_during_startup.append("called")

    monkeypatch.setattr(main_module, "_seed_knowledge", _record)
    monkeypatch.setattr(main_module, "_warm_safety_detector", lambda: None)
    monkeypatch.setattr(main_module, "_warm_embedder", lambda: None)
    # Neutralise the background task so this measures the LIFESPAN only: if the
    # seed still runs, it ran inline, not from the warm-up task.
    monkeypatch.setattr(main_module, "_warm_models", _noop_async)

    with TestClient(app) as client:
        assert client.get("/livez").status_code == 200
        assert seeded_during_startup == [], (
            "seed_knowledge ran during startup, before the port opened. On an "
            "unseeded database that is a 42s / +677 MB step, which is why the "
            "service could not boot. It belongs in the background warm-up."
        )


async def _noop_async() -> None:
    """Stand-in for _warm_models when a test needs the lifespan in isolation."""
    return None


def test_the_knowledge_seed_still_runs_in_the_background(monkeypatch):
    """Deferring it must not mean dropping it — the GK corpus still gets seeded."""
    order = []
    monkeypatch.setenv("WARM_MODELS_BLOCKING", "true")
    monkeypatch.setattr(main_module, "_warm_safety_detector", lambda: order.append("detector"))
    monkeypatch.setattr(main_module, "_warm_embedder", lambda: order.append("embedder"))
    monkeypatch.setattr(main_module, "_seed_knowledge", lambda: order.append("seed"))

    with TestClient(app):
        # Embedder before seed: the seed reuses the cached model rather than
        # paying a second load.
        assert order == ["detector", "embedder", "seed"], order


def test_blocking_mode_is_still_available(monkeypatch):
    """WARM_MODELS_BLOCKING=true restores inline warm-up for on-prem, where
    readiness is expected to mean model-loaded."""
    order = []
    monkeypatch.setenv("WARM_MODELS_BLOCKING", "true")
    monkeypatch.setattr(main_module, "_warm_safety_detector", lambda: order.append("detector"))
    monkeypatch.setattr(main_module, "_warm_embedder", lambda: order.append("embedder"))

    with TestClient(app) as client:
        # Both loads completed during startup, before the first request.
        assert order == ["detector", "embedder"], order
        assert client.get("/livez").status_code == 200
