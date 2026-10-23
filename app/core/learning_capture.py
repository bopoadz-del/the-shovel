"""W6 — live activation of the construction learning loop.

The learning engine (``ConstructionLearningEngine``) was fully built and tested
(comprehensive_engine_test 93/93) but NO runtime path ever fed it live data — its
recorders were never called. This module is the wiring: it captures learning
signals from completed container actions and feeds the engine's recorders.

**Default OFF.** Gated by ``FORK_LEARNING_CAPTURE`` (unset/false = no-op), so it
ships inert and the pilot is unchanged until the flag is flipped. Every capture
is best-effort and never raises — a learning-capture failure must never affect
the action's own result.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)


def enabled() -> bool:
    """True when live learning capture is switched on. Default OFF."""
    return (os.getenv("FORK_LEARNING_CAPTURE") or "").strip().lower() in ("1", "true", "yes", "on")


def _engine():
    from app.core.construction_learning import ConstructionLearningEngine
    return ConstructionLearningEngine.shared_instance()


def capture_qa_defects(result: Dict[str, Any], project_id: str = "") -> int:
    """Record each non-conformance from a qa_qc_inspection result into the
    defect learner. Returns the number of defects recorded (0 when off, on
    error, or no defects). Never raises.

    Maps: activity_type <- inspection_type; trade <- defect.trade or
    inspection_type; severity <- defect.severity (falls back to 'medium').
    """
    if not enabled():
        return 0
    try:
        defects = (result or {}).get("defects") or []
        if not defects:
            return 0
        inspection_type = (result or {}).get("inspection_type") or "general"
        eng = _engine()
        n = 0
        for d in defects:
            if not isinstance(d, dict):
                continue
            eng.record_defect(
                activity_type=inspection_type,
                trade=str(d.get("trade") or inspection_type),
                severity=str(d.get("severity") or "medium"),
                activity_name=str(d.get("description") or "")[:120],
                project_id=project_id,
                root_cause=str(d.get("root_cause") or ""),
            )
            n += 1
        logger.info("learning_capture: recorded %d qa/qc defect(s) for %s", n, project_id or "-")
        return n
    except Exception:  # noqa: BLE001 — learning capture must never break the action
        logger.warning("learning_capture: qa defect capture failed", exc_info=True)
        return 0


def capture_schedule_durations(result: Dict[str, Any], project_id: str = "") -> int:
    """Record planned-vs-actual activity durations into the duration learner.
    Records only activities carrying BOTH a planned and an actual duration.
    Returns the count. Never raises.

    NOT wired to parse_primavera_schedule yet (see the note in schedule.py):
    today's schedule parse exposes planned/original durations + start-date delays,
    not a clean per-activity ACTUAL duration. This function is ready for a real
    ``{planned_duration, actual_duration}`` signal (a progress/as-built capture);
    wiring it to the current delay analysis would record nothing. The unit test
    exercises the function against the correct input shape.
    """
    if not enabled():
        return 0
    try:
        # Delay analysis carries the baseline (planned) vs current (actual) view.
        delays = ((result or {}).get("delay_analysis") or {}).get("activities") \
            or (result or {}).get("delayed_activities") or []
        if not delays:
            return 0
        eng = _engine()
        n = 0
        for a in delays:
            if not isinstance(a, dict):
                continue
            planned = a.get("planned_duration") or a.get("baseline_duration")
            actual = a.get("actual_duration") or a.get("current_duration")
            if planned is None or actual is None:
                continue
            eng.record_duration(
                project_id=project_id,
                activity_type=str(a.get("activity_type") or a.get("type") or "general"),
                planned_days=int(planned),
                actual_days=int(actual),
                activity_name=str(a.get("name") or a.get("activity_name") or "")[:120],
            )
            n += 1
        logger.info("learning_capture: recorded %d activity duration(s) for %s", n, project_id or "-")
        return n
    except Exception:  # noqa: BLE001
        logger.warning("learning_capture: schedule duration capture failed", exc_info=True)
        return 0


# Action name -> capture function. Only actions whose result shape a capture
# function actually understands appear here; anything else is a no-op, so
# adding a container action never silently starts writing malformed learning
# data.
_CAPTURE_BY_ACTION = {
    "qa_qc_inspection": capture_qa_defects,
    "wir_form": capture_qa_defects,
    "inspection_request": capture_qa_defects,
    "parse_primavera_schedule": capture_schedule_durations,
    "primavera_parse": capture_schedule_durations,
    "progress_tracker": capture_schedule_durations,
    "track_progress": capture_schedule_durations,
}


def capture_from_action(action: str, result: Any, params: Any = None) -> int:
    """Capture learning signals from one completed container action.

    This is the WRITE side of the learning loop and it is deliberately the only
    thing wired: recording actuals is invisible to users and changes no answer,
    so it can run from day one and accumulate the dataset. The READ side —
    schedule work consulting calibrated durations, procurement consulting
    learned lead times — is NOT connected, and when it is, it must state its
    provenance in the answer ("calibrated from 14 actuals on this project;
    default was 21 days") rather than silently shifting numbers.

    Best-effort by construction: returns 0 and never raises, so a learning
    failure cannot affect the action's own result. Off unless
    ``FORK_LEARNING_CAPTURE`` is set.
    """
    if not enabled():
        return 0
    try:
        fn = _CAPTURE_BY_ACTION.get(action)
        if fn is None or not isinstance(result, dict):
            return 0
        if result.get("status") == "error":
            return 0
        project_id = ""
        if isinstance(params, dict):
            project_id = str(params.get("project_id") or "")
        if not project_id:
            project_id = str(result.get("project_id") or "")
        return fn(result, project_id)
    except Exception:  # noqa: BLE001 — must never break the action
        logger.warning(
            "learning_capture: capture_from_action(%s) failed", action, exc_info=True
        )
        return 0
