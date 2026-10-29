"""PR#3 RC2 (ORDERS_28OCT_1400Z, re-worked per the 8969ddc grade): a kernel
EXCEPTION in either productivity-duration estimator is SURFACED as a distinct
computed failure — a ``kernel-error`` duration basis that names the failure,
logged at WARNING — never silently collapsed into the plain missing-input
UNKNOWN. A genuinely missing input still yields the ordinary UNKNOWN (0845Z);
only a kernel BREAK is the surfaced, distinguishable outcome.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager

import pytest

import app.agents.hats.planning.formulas.construction_formulas_planning as _kern


def _make_kernel_raise(monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("productivity kernel unavailable")
    monkeypatch.setattr(_kern, "productivity_manpower_duration", boom)


@contextmanager
def _capture(logger_name):
    """Capture records straight off the named logger, independent of handler
    propagation or the ambient log level (pytest ``caplog`` relies on root
    propagation, which app startup can switch off in CI)."""
    lg = logging.getLogger(logger_name)
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.setLevel(logging.DEBUG)
    handler.emit = records.append  # type: ignore[assignment]
    prev_level, prev_disabled = lg.level, lg.disabled
    lg.addHandler(handler)
    lg.setLevel(logging.DEBUG)
    lg.disabled = False
    try:
        yield records
    finally:
        lg.removeHandler(handler)
        lg.setLevel(prev_level)
        lg.disabled = prev_disabled


def test_brief_estimator_surfaces_the_kernel_error(monkeypatch):
    from app.containers.construction import schedule as sched
    _make_kernel_raise(monkeypatch)
    with _capture("app.containers.construction.schedule") as records:
        days, error = sched._estimate_duration_from_rate(100.0, "50 m2 per gang-day")
    assert days is None  # no fabricated number
    assert error and "kernel error" in error.lower() and "RuntimeError" in error
    assert any(r.levelno >= logging.WARNING for r in records), \
        "the kernel break must be logged at a visible level"


def test_boq_estimator_surfaces_the_kernel_error(monkeypatch):
    from app.lib import boq_schedule as bs
    _make_kernel_raise(monkeypatch)
    with _capture("app.lib.boq_schedule") as records:
        days, error = bs._kernel_duration_days(200.0, "40 m3 per gang-day")
    assert days is None
    assert error and "kernel error" in error.lower() and "RuntimeError" in error
    assert any(r.levelno >= logging.WARNING for r in records)


def test_boq_row_duration_basis_names_the_kernel_error_not_unknown(monkeypatch):
    """The row's duration_basis distinguishes a kernel BREAK from a plain
    missing-input UNKNOWN."""
    from app.lib import boq_schedule as bs
    _make_kernel_raise(monkeypatch)
    days, basis = bs.boq_row_duration(
        {"description": "Concrete to slab", "quantity": 50, "unit": "m3",
         "productivity_rate": "20 m3 per gang-day"}
    )
    assert days is None
    assert basis != "unknown"
    assert basis.startswith("kernel-error") and "productivity kernel error" in basis
    # A row with neither a rate nor a norm still yields the ordinary UNKNOWN.
    d2, b2 = bs.boq_row_duration({"description": "Nondescript item", "quantity": 5, "unit": "nr"})
    assert d2 is None and b2 == "unknown"


@pytest.mark.asyncio
async def test_brief_wbs_surfaces_a_kernel_error_on_the_activity(monkeypatch):
    """End-to-end: an activity whose duration needs the kernel carries a
    ``kernel-error`` basis (not ``unknown``) when the kernel raises, so the
    failure is visible in the schedule output."""
    from app.containers.construction import ConstructionContainer
    from app.core import llm_client

    async def fake(system, user, **kwargs):
        return {"activities": [
            {"name": "Plaster", "source_span": "plaster", "duration_days": None,
             "quantity": 100, "unit": "m2", "productivity_rate": "50 m2 per gang-day",
             "predecessors": []},
        ]}
    monkeypatch.setattr(llm_client, "complete_json", fake)
    _make_kernel_raise(monkeypatch)
    r = await ConstructionContainer().generate_wbs(
        {"brief": "Plaster the walls."}, {"target_count": 40},
    )
    assert r["status"] == "success"
    plaster = next(a for a in r["activities"] if a["name"] == "Plaster")
    assert plaster["duration_days"] is None
    assert plaster["duration_basis"].startswith("kernel-error")
    assert "productivity kernel error" in plaster["duration_basis"]
    assert "Plaster" in r["unscheduled_activities"]
