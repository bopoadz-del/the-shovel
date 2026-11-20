"""The app freezes its heap once, when start-up has finished, and releases it
at shutdown; ``GC_FREEZE_AT_STARTUP=0`` turns it off; importing the app
freezes nothing.

Why: a full garbage collection holds the interpreter lock for its whole pass
and re-scans every long-lived object (``GcRecord`` in loop_watchdog measured
the pass on CI). Objects alive at the end of start-up live as long as the
process, so they are taken out of the collector's reach once.
"""
from __future__ import annotations

import gc
import os
import subprocess
import sys

import pytest

import app.main as main_mod
from app.core import loop_watchdog

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def unfrozen():
    gc.unfreeze()
    yield
    gc.unfreeze()


@pytest.mark.parametrize("setting, frozen", [(None, True), ("1", True),
                                             ("0", False), ("false", False),
                                             ("off", False)])
def test_the_off_switch_decides_whether_the_heap_is_frozen(unfrozen, monkeypatch, setting, frozen):
    if setting is None:
        monkeypatch.delenv("GC_FREEZE_AT_STARTUP", raising=False)
    else:
        monkeypatch.setenv("GC_FREEZE_AT_STARTUP", setting)
    got = loop_watchdog.freeze_startup_heap()
    assert (got is not None) is frozen
    assert (gc.get_freeze_count() > 0) is frozen
    if frozen:
        # The count can only fall after gc.freeze(): the temporaries the
        # freeze call itself left alive (the off-switch string) deallocate
        # once the call returns, so any later read is the same count or
        # less. An exact == is therefore a race; >= is the stable invariant
        # and still rejects a freeze that returned a smaller/zero count.
        assert got >= gc.get_freeze_count()
    loop_watchdog.release_startup_heap()
    assert gc.get_freeze_count() == 0


@pytest.mark.parametrize("setting", ["1", "0"])
def test_importing_the_app_freezes_nothing(setting):
    env = dict(os.environ, GC_FREEZE_AT_STARTUP=setting, PYTHONPATH=_REPO)
    out = subprocess.run(
        [sys.executable, "-c",
         "import gc, app.main; print('FREEZE', gc.get_freeze_count())"],
        cwd=_REPO, env=env, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("FREEZE ")][-1]
    assert line == "FREEZE 0"


@pytest.mark.parametrize("setting, frozen_while_serving", [("1", True), ("0", False)])
def test_start_up_freezes_once_after_warm_up_and_shutdown_releases(
        unfrozen, monkeypatch, caplog, setting, frozen_while_serving):
    import logging

    from fastapi.testclient import TestClient

    monkeypatch.setenv("GC_FREEZE_AT_STARTUP", setting)
    # Warm-up inline so start-up has finished when the client is entered;
    # its loads replaced so the test does not load models.
    monkeypatch.setenv("WARM_MODELS_BLOCKING", "true")
    for name in ("_declare_deferred_results", "_warm_safety_detector", "_warm_embedder",
                 "_check_embedder_identity", "_seed_knowledge", "_warm_parse_libraries"):
        monkeypatch.setattr(main_mod, name, lambda: None)
    calls = []
    real = loop_watchdog.freeze_startup_heap

    def counted():
        # Who froze, and how much was frozen before: the end of the warm-up
        # (``_warm_models``), once, with nothing frozen ahead of it.
        calls.append((sys._getframe(1).f_code.co_name, gc.get_freeze_count()))
        return real()

    monkeypatch.setattr(loop_watchdog, "freeze_startup_heap", counted)
    assert gc.get_freeze_count() == 0
    caplog.set_level(logging.INFO, logger=main_mod.logger.name)
    with TestClient(main_mod.app) as client:
        assert client.get("/livez").status_code == 200
        assert calls == [(main_mod._warm_models.__name__, 0)]  # once, at the end of warm-up
        assert (gc.get_freeze_count() > 0) is frozen_while_serving
    assert len(calls) == 1
    assert gc.get_freeze_count() == 0
    # The production figure: the frozen count is logged only when it froze.
    logged = [r.getMessage() for r in caplog.records if "startup heap frozen" in r.getMessage()]
    assert len(logged) == (1 if frozen_while_serving else 0), logged
