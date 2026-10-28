"""PR#3 RC2 (ORDERS_28OCT_1400Z): the two productivity-kernel duration
estimators must SURFACE a kernel failure (log it) and return UNKNOWN (None),
never swallow it silently. None is the designed "no estimate" outcome (0845Z:
no rate/no computable input -> no estimate); the fix is that a kernel
*exception* is logged, so "nothing computable" and "the kernel blew up" are
not the same invisible answer. scan_exception_pass's RETURN twin stays green
because each handler now logs before returning.
"""
from __future__ import annotations

import logging

import app.agents.hats.planning.formulas.construction_formulas_planning as _kern


def _make_kernel_raise(monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("productivity kernel unavailable")
    monkeypatch.setattr(_kern, "productivity_manpower_duration", boom)


def test_brief_path_estimator_logs_and_returns_unknown_on_kernel_error(monkeypatch, caplog):
    from app.containers.construction import schedule as sched
    _make_kernel_raise(monkeypatch)
    with caplog.at_level(logging.DEBUG, logger="app.containers.construction.schedule"):
        out = sched._estimate_duration_from_rate(100.0, "50 m2 per gang-day")
    assert out is None  # UNKNOWN, not a crash and not a fabricated number
    assert any(
        "could not estimate duration" in r.getMessage() for r in caplog.records
    ), "the kernel failure must be logged, not swallowed silently"


def test_boq_path_estimator_logs_and_returns_unknown_on_kernel_error(monkeypatch, caplog):
    from app.lib import boq_schedule as bs
    _make_kernel_raise(monkeypatch)
    with caplog.at_level(logging.DEBUG, logger="app.lib.boq_schedule"):
        out = bs._kernel_duration_days(200.0, "40 m3 per gang-day")
    assert out is None
    assert any(
        "could not estimate duration" in r.getMessage() for r in caplog.records
    ), "the kernel failure must be logged, not swallowed silently"
