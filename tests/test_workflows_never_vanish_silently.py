"""A workflow the platform cannot reach must be visible, never silently
missing (ORDERS_2026-10-13T0733Z step 3). Synthetic failure only."""
from __future__ import annotations

import logging

from app.core import predefined_reasoning as pr


class _Records(logging.Handler):
    def __init__(self):
        super().__init__(logging.ERROR)
        self.records = []

    def emit(self, record):
        self.records.append(record)


def test_a_container_that_does_not_load_is_logged_as_an_error(monkeypatch):
    import app.dependencies as deps

    def broken(name):
        raise RuntimeError("synthetic container failure")
    monkeypatch.setattr(deps, "get_block_instance", broken)
    # Read the module's own logger directly: another test's logging set-up may
    # have stopped it propagating to the root logger caplog listens on.
    log = logging.getLogger(pr.__name__)
    monkeypatch.setattr(log, "disabled", False)
    seen = _Records()
    log.addHandler(seen)
    try:
        assert pr.container_action_names() == set()
    finally:
        log.removeHandler(seen)
    assert any(r.levelno >= logging.ERROR and "workflows are unreachable" in r.getMessage()
               for r in seen.records)


def test_health_counts_the_reachable_workflows(monkeypatch):
    from app.routers import health

    monkeypatch.setattr(pr, "container_action_names", lambda: {"syn_action_a", "syn_action_b"})
    out = health._probe_workflows()
    assert out["reachable"] == len(set(pr.WORKFLOW_REGISTRY) | {"syn_action_a", "syn_action_b"})
