"""A deferred result declarer is never lost and never half-read.

It leaves the waiting list only after it succeeds (a failure is logged by its
class and retried on the next read), and one lock covers the loop so a second
thread's first read waits for the declarations. Every test builds its own
invented declarer; none relies on what an earlier test already drained.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from app.agents.core import tool_registry


@pytest.fixture
def invented():
    names: list[str] = []
    declarers: list = []

    def make(name, body):
        names.append(name)
        declarers.append(body)
        tool_registry.declare_results_when_read(body)
        return body

    yield make
    for n in names:
        tool_registry._RESULTS.pop(n, None)
    for d in declarers:
        if d in tool_registry._DEFERRED:
            tool_registry._DEFERRED.remove(d)


def _declare(name, kind="search"):
    tool_registry.declare_result(name, owner="base", display_name="Invented", kind=kind)


@pytest.mark.parametrize("name, error", [
    ("zz_flaky_declarer_a", RuntimeError),
    ("zz_flaky_declarer_b", ImportError),
])
def test_a_declarer_that_fails_once_is_retried_and_then_declares(invented, caplog, name, error):
    calls = []

    def declarer():
        calls.append(1)
        if len(calls) == 1:
            raise error("secret detail " + name)
        _declare(name)

    invented(name, declarer)
    with caplog.at_level(logging.WARNING, logger=tool_registry.__name__):
        assert tool_registry.result_spec(name) is None
    assert declarer in tool_registry._DEFERRED
    logged = " ".join(r.getMessage() for r in caplog.records)
    assert error.__name__ in logged and "secret detail" not in logged

    assert tool_registry.result_spec(name).kind == "search"
    assert len(calls) == 2
    assert declarer not in tool_registry._DEFERRED
    tool_registry.result_spec(name)
    assert len(calls) == 2


@pytest.mark.parametrize("failing, healthy", [
    ("zz_fail_first_a", "zz_ok_second_a"),
    ("zz_fail_first_b", "zz_ok_second_b"),
])
def test_one_failing_declarer_does_not_stop_the_others(invented, failing, healthy):
    def bad():
        raise ValueError(failing)

    invented(failing, bad)
    invented(healthy, lambda: _declare(healthy))
    assert healthy in tool_registry.tools_of_kind("search")
    assert bad in tool_registry._DEFERRED


@pytest.mark.parametrize("name", ["zz_slow_declarer_a", "zz_slow_declarer_b"])
def test_a_second_threads_first_read_waits_for_the_declarations(invented, name):
    started, release = threading.Event(), threading.Event()
    runs = []

    def declarer():
        runs.append(1)
        started.set()
        release.wait(10)
        _declare(name)

    invented(name, declarer)
    seen = {}
    first = threading.Thread(target=lambda: seen.setdefault("first", tool_registry.result_spec(name)))
    second = threading.Thread(target=lambda: seen.setdefault("second", tool_registry.result_spec(name)))
    first.start()
    assert started.wait(10)
    second.start()
    time.sleep(0.2)          # the second read is now waiting on the lock
    assert "second" not in seen
    release.set()
    first.join(10)
    second.join(10)
    assert seen["first"].kind == "search" and seen["second"].kind == "search"
    assert runs == [1]


@pytest.mark.parametrize("name", ["zz_reentrant_declarer_a", "zz_reentrant_declarer_b"])
def test_a_read_from_inside_a_declarer_does_not_rerun_it(invented, name):
    runs = []

    def declarer():
        runs.append(1)
        tool_registry.tools_of_kind("search")   # a read made while declaring
        _declare(name)

    invented(name, declarer)
    assert tool_registry.result_spec(name).kind == "search"
    assert runs == [1]


def test_a_calculator_reads_computed_in_a_fresh_process():
    """No earlier test in this process can have drained the deferred list:
    a new interpreter imports the app and reads one calculator's kind."""
    code = (
        "import app.main\n"
        "from app.agents.core import tool_registry as r\n"
        "from app.agents.runtime import _HAT_CALC_ALIASES\n"
        "assert r._DEFERRED, 'the declarations must still be waiting before the first read'\n"
        "for action in (sorted(_HAT_CALC_ALIASES)[0], sorted(_HAT_CALC_ALIASES)[-1]):\n"
        "    assert r.result_spec(action).kind == 'computed', action\n"
        "assert not r._DEFERRED\n"
    )
    env = dict(os.environ)
    env.setdefault("ENV", "development")
    env.setdefault("RAG_EMBEDDING_MODEL", "fake")
    subprocess.run([sys.executable, "-c", code], check=True, timeout=300,
                   cwd=str(Path(__file__).resolve().parents[1]), env=env)
