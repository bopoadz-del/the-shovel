"""A streamed turn logs where its time went (order 0520Z item 1).

The driver turn's own timer missed time spent around it; lanes measured
45 s where the turn logged 22-34 s. Each streamed turn now logs one
TURN_TIMING line: milliseconds from receipt to its first event, first
token, end event and close. No content.
"""
import json
import logging

import pytest

from app.agents.core import loop


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.messages: list = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@pytest.fixture
def loop_log():
    """A handler on the loop's own logger, not the root capture (an earlier
    test's logging setup can detach that: CI production-like), proven live
    before use -- a dead capture fails here instead of passing."""
    log = logging.getLogger("app.agents.core.loop")
    handler = _Capture()
    saved = (log.disabled, log.level, logging.root.manager.disable)
    log.disabled = False
    log.setLevel(logging.INFO)
    logging.disable(logging.NOTSET)
    log.addHandler(handler)
    try:
        log.info("TURN_TIMING_CAPTURE_PROBE")
        assert handler.messages == ["TURN_TIMING_CAPTURE_PROBE"], "the loop logger's lines are not captured"
        handler.messages.clear()
        yield handler
    finally:
        log.removeHandler(handler)
        log.disabled = saved[0]
        log.setLevel(saved[1])
        logging.disable(saved[2])


def test_the_timing_line_has_every_mark_in_order(loop_log):
    marks = {"received": 100.0, "first_event": 100.5, "first_token": 101.0, "end_event": 103.0}
    loop._log_turn_timing(marks, "conv-1")
    lines = [m for m in loop_log.messages if m.startswith("TURN_TIMING ")]
    assert len(lines) == 1, loop_log.messages
    data = json.loads(lines[0].split("TURN_TIMING ", 1)[1])
    assert data["conversation"] == "conv-1"
    assert data["first_event"] == 500 and data["first_token"] == 1000 and data["end_event"] == 3000
    assert "closed" in data
    # No steps arg: the step/before-first-token fields are omitted (back-compat).
    assert "tool_calls" not in data and "before_first_token_ms" not in data


def test_the_timing_line_carries_tool_step_counts(loop_log):
    """ORDERS_0410Z long-ask tail: how many tool steps ran, and the wait
    before the first token -- numbers only."""
    marks = {"received": 100.0, "first_event": 100.2, "first_token": 108.0, "end_event": 110.0}
    loop._log_turn_timing(marks, "conv-2", {"tool_calls": 4, "tool_results": 4})
    line = next(m for m in loop_log.messages if m.startswith("TURN_TIMING "))
    data = json.loads(line.split("TURN_TIMING ", 1)[1])
    assert data["tool_calls"] == 4 and data["tool_results"] == 4
    assert data["before_first_token_ms"] == 8000


def test_a_timing_failure_never_breaks_a_turn():
    loop._log_turn_timing({}, None)
    loop._log_turn_timing({"received": "not a number"}, "c")
