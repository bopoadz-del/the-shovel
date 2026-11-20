"""The suite hands a TestClient to a test only after the app's warm-up is done.

The default lifespan serves first and warms up in a background thread. The
knowledge seed in that thread writes the SQLite file on a second connection,
and two red mains came from it interleaving with a test's own upload. These
tests pin the suite-level rule set in tests/conftest.py: no warm-up step is
still running when the test body starts. Each builds its own slow warm-up step.
"""
from __future__ import annotations

import threading
import time

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app


def _slow_step(done: threading.Event, seconds: float):
    def _run():
        time.sleep(seconds)
        done.set()
    return _run


@pytest.mark.parametrize("step", ["_seed_knowledge", "_warm_embedder"])
def test_warm_up_step_has_finished_before_the_test_body(monkeypatch, step):
    done = threading.Event()
    for name in ("_warm_safety_detector", "_warm_embedder", "_check_embedder_identity",
                 "_seed_knowledge", "_warm_parse_libraries"):
        monkeypatch.setattr(main_module, name, lambda: None)
    monkeypatch.setattr(main_module, step, _slow_step(done, 0.5))

    with TestClient(app) as client:
        assert done.is_set(), (
            f"{step} was still running when the test got the client; its "
            "writes would interleave with the test's own requests"
        )
        assert client.get("/livez").status_code == 200
