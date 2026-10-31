"""S-9b item 6b: the prompt size that gates first-token latency is recorded.

``turn_timing.mark_value`` stores a non-duration scalar (``prompt_tokens_in``)
for the turn, and ``snapshot`` surfaces it alongside the stage timings, so the
first-token work can be measured before/after a prompt-size change. These pin
that the value round-trips and that a reading taken with no turn started is a
silent no-op (the gate before a dead context var).
"""
from __future__ import annotations

import pytest

import app.core.turn_timing as turn_timing


@pytest.fixture(autouse=True)
def _fresh_turn():
    turn_timing._STATE.set(None)
    yield
    turn_timing._STATE.set(None)


@pytest.mark.parametrize("name,value", [
    ("prompt_tokens_in", 1234),
    ("prompt_tokens_in", 9),
])
def test_mark_value_round_trips_through_snapshot(name, value):
    turn_timing.mark_arrival()
    turn_timing.mark_value(name, value)
    snap = turn_timing.snapshot()
    assert snap[name] == value
    # It is not a duration, so it is not rounded away or zeroed with the stages.
    assert "since_arrival" in snap


@pytest.mark.parametrize("name", ["prompt_tokens_in", "anything"])
def test_mark_value_without_a_turn_is_a_silent_noop(name):
    # No mark_arrival: the context var is None. Recording must not raise, and
    # the value must not appear (dropping the None guard would AttributeError).
    turn_timing.mark_value(name, 5)
    assert name not in turn_timing.snapshot()
