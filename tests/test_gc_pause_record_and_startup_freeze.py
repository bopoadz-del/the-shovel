"""LOCK HOLDER (advisor 13:32Z 18 Nov): a full garbage collection holds the
interpreter lock for its whole pass, so the loop and the watchdog thread stop
together. ``GcRecord`` names every pass (generation, duration, start);
``freeze_long_lived_heap`` takes the server's startup heap out of every later
full pass, and lifespan does it once, releasing it at shutdown."""
from __future__ import annotations

import gc
import time
import weakref

import pytest
from fastapi.testclient import TestClient

from app.core.loop_watchdog import (GcRecord, freeze_long_lived_heap, gc_summary,
                                    release_long_lived_heap)


class _Node:
    """A cycle only the collector can free (refcounting cannot)."""

    def __init__(self) -> None:
        self.me = self


@pytest.fixture
def gc_quiet():
    """No automatic pass may land inside a measured window."""
    was = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was:
            gc.enable()


@pytest.mark.parametrize("generation", [0, 1, 2])
def test_a_forced_pass_is_recorded_under_its_own_generation(gc_quiet, generation):
    rec = GcRecord().install()
    try:
        t0 = time.monotonic()
        gc.collect(generation)
        t1 = time.monotonic()
    finally:
        rec.remove()
    assert rec.passes == 1
    assert set(rec.worst) == {generation}
    took_ms, began = rec.worst[generation]
    assert 0 <= took_ms <= (t1 - t0) * 1000 + 1
    assert t0 <= began <= t1


@pytest.mark.parametrize("passes", [0, 3])
def test_the_record_counts_exactly_the_passes_run(gc_quiet, passes):
    rec = GcRecord().install()
    try:
        for _ in range(passes):
            gc.collect(2)
    finally:
        rec.remove()
    assert rec.passes == passes
    assert (2 in rec.worst) is (passes > 0)


def test_a_removed_record_sees_no_later_pass(gc_quiet):
    rec = GcRecord().install()
    gc.collect(0)
    rec.remove()
    gc.collect(2)
    assert rec.passes == 1 and set(rec.worst) == {0}
    assert rec._callback not in gc.callbacks


@pytest.mark.parametrize("installs", [1, 2, 3])
def test_installing_again_registers_one_callback(gc_quiet, installs):
    rec = GcRecord()
    for _ in range(installs):
        rec.install()
    try:
        assert gc.callbacks.count(rec._callback) == 1
        gc.collect(1)
    finally:
        rec.remove()
    assert rec.passes == 1


@pytest.mark.parametrize("worst, start, expected", [
    ({}, 10.0, "gc_passes=0 gc2_max=none"),
    ({0: (3.0, 11.0)}, 10.0, "gc_passes=0 gc2_max=none"),
    ({2: (612.4, 13.25), 0: (900.0, 11.0)}, 10.0, "gc_passes=0 gc2_max=612ms@+3.25s"),
    ({2: (45.6, 10.5)}, 10.0, "gc_passes=0 gc2_max=46ms@+0.50s"),
])
def test_the_summary_names_the_worst_full_pass_and_when(worst, start, expected):
    rec = GcRecord()
    rec.worst = dict(worst)
    assert gc_summary(rec, start) == expected


@pytest.mark.parametrize("passes", [0, 7])
def test_the_summary_counts_passes(passes):
    rec = GcRecord()
    rec.passes = passes
    assert gc_summary(rec, 0.0) == f"gc_passes={passes} gc2_max=none"


@pytest.mark.parametrize("made_before_freeze", [True, False])
def test_only_a_cycle_alive_at_the_freeze_survives_full_passes(gc_quiet, made_before_freeze):
    node = _Node() if made_before_freeze else None
    try:
        # The count at the call; another thread may allocate before a re-read.
        assert freeze_long_lived_heap() > 0 and gc.get_freeze_count() > 0
        if node is None:
            node = _Node()
        ref = weakref.ref(node)
        del node
        gc.collect(2)
        # Alive at the freeze: out of the collector's reach. Made after: collected.
        assert (ref() is not None) is made_before_freeze
    finally:
        release_long_lived_heap()
    assert gc.get_freeze_count() == 0
    gc.collect(2)
    assert ref() is None


@pytest.mark.parametrize("lifecycles", [1, 2])
def test_the_server_freezes_its_heap_at_startup_and_releases_it_at_shutdown(lifecycles):
    from app.main import app

    release_long_lived_heap()
    for _ in range(lifecycles):
        assert gc.get_freeze_count() == 0
        with TestClient(app) as client:
            frozen_while_serving = gc.get_freeze_count()
            assert client.get("/livez").status_code == 200
        assert frozen_while_serving > 0
        assert gc.get_freeze_count() == 0


@pytest.mark.parametrize("phases, expected_passes", [
    (["stop"], 0),            # installed mid-pass: a stop with no start
    (["start", "stop"], 1),
])
def test_a_stop_without_its_start_is_not_counted(phases, expected_passes):
    rec = GcRecord()
    for phase in phases:
        rec._callback(phase, {"generation": 2, "collected": 0, "uncollectable": 0})
    assert rec.passes == expected_passes
    assert (2 in rec.worst) is (expected_passes > 0)


@pytest.mark.parametrize("installed", [False, True])
def test_remove_is_safe_whether_or_not_installed(installed):
    rec = GcRecord()
    if installed:
        rec.install()
    rec.remove()
    rec.remove()
    assert rec._callback not in gc.callbacks
