"""Construction Learning Engine — Phase 6 of the Construction Management Engine.

Continuous learning and optimization for construction management:
- Duration calibration: actual activity durations refine WBS template defaults
- Procurement lead time learning: actual delivery dates vs planned update the library
- Defect pattern detection: learn which activity types/trades have highest defect rates
- Template effectiveness: learn which workflow templates work best per project type

Extends the existing learning_engine with construction-specific learning.
Uses the same JSON storage pattern — no external dependencies.
"""

from __future__ import annotations

import os
import threading
import time
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# ── Storage ───────────────────────────────────────────────────────────────

def _storage_path() -> str:
    """Resolve learning JSON path at call time (tests can swap via env).

    Prefer CONSTRUCTION_LEARNING_STORAGE; else DATA_DIR/learning/… so
    Render/ephemeral hosts do not default to /tmp (lost on restart).
    """
    explicit = os.environ.get("CONSTRUCTION_LEARNING_STORAGE")
    if explicit:
        return explicit
    data_dir = os.environ.get("DATA_DIR", "./data")
    return os.path.join(data_dir, "learning", "cerebrum_construction_learning.json")


# ── Duration Calibration ──────────────────────────────────────────────────

class DurationCalibration:
    """Learn actual activity durations to refine WBS template defaults.

    Stores: {project_id: {activity_type: [{planned_days, actual_days, date, variance_pct}]}}
    Produces: calibrated duration estimates per activity type with confidence intervals.
    """

    _state_lock = threading.RLock()

    def __init__(self, state: Optional[Dict] = None) -> None:
        self._state = state or {}

    def record_actual_duration(
        self,
        project_id: str,
        activity_type: str,
        planned_days: int,
        actual_days: int,
        activity_name: str = "",
    ) -> Dict[str, Any]:
        """Record an actual duration observation for an activity type."""
        variance_pct = ((actual_days - planned_days) / planned_days * 100) if planned_days > 0 else 0
        observation = {
            "planned_days": planned_days,
            "actual_days": actual_days,
            "variance_pct": round(variance_pct, 2),
            "activity_name": activity_name,
            "recorded_at": datetime.now().isoformat(),
        }
        key = f"{project_id}:{activity_type}"
        with self._state_lock:
            if key not in self._state:
                self._state[key] = []
            self._state[key].append(observation)

        return {
            "status": "recorded",
            "variance_pct": round(variance_pct, 2),
            "samples_for_type": len(self._state.get(key, [])),
        }

    def get_calibrated_duration(
        self,
        project_id: str,
        activity_type: str,
        default_days: int = 5,
        min_samples: int = 3,
    ) -> Dict[str, Any]:
        """Get a calibrated duration estimate based on historical actuals.

        Returns the median actual duration with a confidence score.
        Falls back to default when insufficient samples.
        """
        key = f"{project_id}:{activity_type}"
        samples = self._state.get(key, [])
        if len(samples) < min_samples:
            return {
                "calibrated_days": default_days,
                "confidence": "low",
                "samples": len(samples),
                "source": "default",
                "reason": f"insufficient samples ({len(samples)} < {min_samples})",
            }

        actuals = sorted(s["actual_days"] for s in samples)
        n = len(actuals)
        median = actuals[n // 2] if n % 2 == 1 else (actuals[n // 2 - 1] + actuals[n // 2]) / 2
        mean = sum(actuals) / n
        # Simple std dev
        variance = sum((x - mean) ** 2 for x in actuals) / n
        std_dev = variance ** 0.5

        # Confidence based on sample size and variance
        cv = (std_dev / mean) if mean > 0 else float("inf")
        if cv < 0.15 and n >= 10:
            confidence = "high"
        elif cv < 0.3 and n >= 5:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "calibrated_days": round(median),
            "median": median,
            "mean": round(mean, 1),
            "std_dev": round(std_dev, 1),
            "samples": n,
            "confidence": confidence,
            "source": "calibrated",
            "variance_pct_history": [s["variance_pct"] for s in samples[-10:]],
        }

    def get_type_summary(self, project_id: str) -> Dict[str, Any]:
        """Summary of all calibrated durations for a project."""
        prefix = f"{project_id}:"
        summary = {}
        for key, samples in self._state.items():
            if key.startswith(prefix):
                activity_type = key.split(":", 1)[1]
                actuals = [s["actual_days"] for s in samples]
                variances = [s["variance_pct"] for s in samples]
                summary[activity_type] = {
                    "samples": len(samples),
                    "median_actual": sorted(actuals)[len(actuals) // 2] if actuals else 0,
                    "mean_variance_pct": round(sum(variances) / len(variances), 1) if variances else 0,
                }
        return summary

    def to_dict(self) -> Dict:
        return dict(self._state)

    @classmethod
    def from_dict(cls, data: Dict) -> "DurationCalibration":
        inst = cls()
        inst._state = dict(data)
        return inst


# ── Procurement Lead Time Learning ────────────────────────────────────────

class ProcurementLeadTimeLearner:
    """Learn actual procurement lead times to update the procurement library.

    Stores: {item_category: [{planned_lead_days, actual_lead_days, supplier, date}]}
    Produces: recommended lead times per item category with confidence.
    """

    _state_lock = threading.RLock()

    def __init__(self, state: Optional[Dict] = None) -> None:
        self._state = state or {}

    def record_delivery(
        self,
        item_category: str,
        item_name: str,
        planned_lead_days: int,
        actual_lead_days: int,
        supplier: str = "",
        project_id: str = "",
    ) -> Dict[str, Any]:
        """Record an actual procurement lead time observation."""
        variance_pct = ((actual_lead_days - planned_lead_days) / planned_lead_days * 100) if planned_lead_days > 0 else 0
        observation = {
            "item_name": item_name,
            "planned_lead_days": planned_lead_days,
            "actual_lead_days": actual_lead_days,
            "variance_pct": round(variance_pct, 2),
            "supplier": supplier,
            "project_id": project_id,
            "recorded_at": datetime.now().isoformat(),
        }
        with self._state_lock:
            if item_category not in self._state:
                self._state[item_category] = []
            self._state[item_category].append(observation)

        return {
            "status": "recorded",
            "variance_pct": round(variance_pct, 2),
            "samples_for_category": len(self._state.get(item_category, [])),
        }

    def get_lead_time_estimate(
        self,
        item_category: str,
        default_days: int = 60,
        min_samples: int = 2,
    ) -> Dict[str, Any]:
        """Get a learned lead time estimate for an item category.

        Uses the 80th percentile of actuals to be conservative (procurement
        delays are more common than early deliveries).
        """
        samples = self._state.get(item_category, [])
        if len(samples) < min_samples:
            return {
                "estimated_days": default_days,
                "confidence": "low",
                "samples": len(samples),
                "source": "default",
            }

        actuals = sorted(s["actual_lead_days"] for s in samples)
        n = len(actuals)
        # 80th percentile for conservative estimate
        p80_idx = int(n * 0.8)
        p80 = actuals[min(p80_idx, n - 1)]
        median = actuals[n // 2] if n % 2 == 1 else (actuals[n // 2 - 1] + actuals[n // 2]) / 2
        mean = sum(actuals) / n

        confidence = "high" if n >= 10 else "medium" if n >= 5 else "low"

        return {
            "estimated_days": p80,
            "p80": p80,
            "median": median,
            "mean": round(mean, 1),
            "samples": n,
            "confidence": confidence,
            "source": "learned",
            "variance_pct_history": [s["variance_pct"] for s in samples[-10:]],
        }

    def get_category_summary(self) -> Dict[str, Any]:
        """Summary of all learned lead times."""
        summary = {}
        for category, samples in self._state.items():
            actuals = [s["actual_lead_days"] for s in samples]
            summary[category] = {
                "samples": len(samples),
                "median_lead_days": sorted(actuals)[len(actuals) // 2] if actuals else 0,
                "max_lead_days": max(actuals) if actuals else 0,
            }
        return summary

    def to_dict(self) -> Dict:
        return dict(self._state)

    @classmethod
    def from_dict(cls, data: Dict) -> "ProcurementLeadTimeLearner":
        inst = cls()
        inst._state = dict(data)
        return inst


# ── Defect Pattern Detection ──────────────────────────────────────────────

class DefectPatternDetector:
    """Learn which activity types and trades have the highest defect rates.

    Stores defect observations and produces risk rankings per activity type.
    """

    _state_lock = threading.RLock()

    def __init__(self, state: Optional[Dict] = None) -> None:
        self._state = state or {}

    def record_defect(
        self,
        activity_type: str,
        trade: str,
        severity: str,  # minor | major | critical
        activity_name: str = "",
        project_id: str = "",
        root_cause: str = "",
    ) -> Dict[str, Any]:
        """Record a defect observation."""
        severity_weights = {"minor": 1, "major": 3, "critical": 5}
        weight = severity_weights.get(severity, 1)

        observation = {
            "severity": severity,
            "severity_weight": weight,
            "activity_name": activity_name,
            "project_id": project_id,
            "root_cause": root_cause,
            "recorded_at": datetime.now().isoformat(),
        }

        key = f"{activity_type}:{trade}"
        with self._state_lock:
            if key not in self._state:
                self._state[key] = {
                    "activity_type": activity_type,
                    "trade": trade,
                    "defects": [],
                    "total_weighted": 0,
                    "total_inspections": 0,
                }
            entry = self._state[key]
            entry["defects"].append(observation)
            entry["total_weighted"] += weight

        return {
            "status": "recorded",
            "weighted_score": entry["total_weighted"],
            "total_defects": len(entry["defects"]),
        }

    def record_inspection(
        self,
        activity_type: str,
        trade: str,
        passed: bool,
        project_id: str = "",
    ) -> Dict[str, Any]:
        """Record an inspection outcome (pass or fail) for rate calculation."""
        key = f"{activity_type}:{trade}"
        with self._state_lock:
            if key not in self._state:
                self._state[key] = {
                    "activity_type": activity_type,
                    "trade": trade,
                    "defects": [],
                    "total_weighted": 0,
                    "total_inspections": 0,
                    "failed_inspections": 0,
                }
            entry = self._state[key]
            entry["total_inspections"] += 1
            if not passed:
                entry["failed_inspections"] = entry.get("failed_inspections", 0) + 1

        return {
            "status": "recorded",
            "total_inspections": entry["total_inspections"],
            "failed_inspections": entry.get("failed_inspections", 0),
        }

    def get_defect_risk_ranking(self, min_samples: int = 3) -> List[Dict[str, Any]]:
        """Rank activity_type:trade combinations by defect risk.

        Returns sorted list from highest to lowest risk.
        """
        rankings = []
        for key, entry in self._state.items():
            inspections = entry.get("total_inspections", 0)
            if inspections < min_samples:
                continue
            failed = entry.get("failed_inspections", 0)
            fail_rate = (failed / inspections * 100) if inspections > 0 else 0
            weighted = entry.get("total_weighted", 0)
            # Risk score: weighted defects + fail rate
            risk_score = weighted + fail_rate
            rankings.append({
                "activity_type": entry["activity_type"],
                "trade": entry["trade"],
                "risk_score": round(risk_score, 1),
                "fail_rate_pct": round(fail_rate, 1),
                "weighted_defects": weighted,
                "total_inspections": inspections,
                "failed_inspections": failed,
            })

        rankings.sort(key=lambda x: -x["risk_score"])
        return rankings

    def get_high_risk_types(self, threshold: float = 20.0) -> List[Dict[str, Any]]:
        """Get activity types with fail rate above threshold."""
        all_ranked = self.get_defect_risk_ranking(min_samples=1)
        return [r for r in all_ranked if r["fail_rate_pct"] >= threshold]

    def to_dict(self) -> Dict:
        return dict(self._state)

    @classmethod
    def from_dict(cls, data: Dict) -> "DefectPatternDetector":
        inst = cls()
        inst._state = dict(data)
        return inst


# ── Template Effectiveness Tracker ────────────────────────────────────────

class TemplateEffectivenessTracker:
    """Learn which workflow templates are most effective per project type.

    Tracks template usage outcomes: completion rate, user satisfaction signals,
        and time-to-completion.
    """

    _state_lock = threading.RLock()

    def __init__(self, state: Optional[Dict] = None) -> None:
        self._state = state or {}

    def record_usage(
        self,
        template_id: str,
        project_type: str,
        success: bool,
        steps_completed: int = 0,
        total_steps: int = 0,
    ) -> Dict[str, Any]:
        """Record the outcome of a template execution."""
        key = f"{template_id}:{project_type}"
        with self._state_lock:
            if key not in self._state:
                self._state[key] = {
                    "template_id": template_id,
                    "project_type": project_type,
                    "uses": [],
                    "total_uses": 0,
                    "successful_uses": 0,
                }
            entry = self._state[key]
            entry["total_uses"] += 1
            if success:
                entry["successful_uses"] += 1
            entry["uses"].append({
                "success": success,
                "steps_completed": steps_completed,
                "total_steps": total_steps,
                "recorded_at": datetime.now().isoformat(),
            })
            # Keep only last 50 uses to prevent unbounded growth
            entry["uses"] = entry["uses"][-50:]

        success_rate = (entry["successful_uses"] / entry["total_uses"] * 100) if entry["total_uses"] > 0 else 0
        return {
            "status": "recorded",
            "success_rate_pct": round(success_rate, 1),
            "total_uses": entry["total_uses"],
        }

    def get_best_templates(self, project_type: str, min_uses: int = 2) -> List[Dict[str, Any]]:
        """Get templates ranked by success rate for a project type."""
        results = []
        for key, entry in self._state.items():
            if entry.get("project_type") == project_type:
                if entry["total_uses"] < min_uses:
                    continue
                success_rate = (entry["successful_uses"] / entry["total_uses"] * 100)
                results.append({
                    "template_id": entry["template_id"],
                    "project_type": project_type,
                    "success_rate_pct": round(success_rate, 1),
                    "total_uses": entry["total_uses"],
                    "successful_uses": entry["successful_uses"],
                })
        results.sort(key=lambda x: -x["success_rate_pct"])
        return results

    def get_template_recommendation(self, project_type: str) -> Optional[str]:
        """Recommend the best template for a project type."""
        best = self.get_best_templates(project_type, min_uses=1)
        return best[0]["template_id"] if best else None

    def to_dict(self) -> Dict:
        return dict(self._state)

    @classmethod
    def from_dict(cls, data: Dict) -> "TemplateEffectivenessTracker":
        inst = cls()
        inst._state = dict(data)
        return inst


# ── Unified Construction Learning Block ───────────────────────────────────

class ConstructionLearningEngine:
    """Unified facade for all construction learning operations.

    Manages persistence and provides a single interface for:
    - Duration calibration
    - Procurement lead time learning
    - Defect pattern detection
    - Template effectiveness tracking
    """

    _instance_by_path: Dict[str, "ConstructionLearningEngine"] = {}
    _instance_cache_lock = threading.Lock()

    @classmethod
    def shared_instance(cls) -> "ConstructionLearningEngine":
        path = _storage_path()
        with cls._instance_cache_lock:
            inst = cls._instance_by_path.get(path)
            if inst is None:
                inst = cls()
                cls._instance_by_path[path] = inst
        return inst

    @classmethod
    def reset_shared_instance_cache(cls) -> None:
        with cls._instance_cache_lock:
            cls._instance_by_path.clear()

    def __init__(self) -> None:
        self._state = self._load_state()
        self.duration = DurationCalibration(self._state.get("duration", {}))
        self.procurement = ProcurementLeadTimeLearner(self._state.get("procurement", {}))
        self.defects = DefectPatternDetector(self._state.get("defects", {}))
        self.templates = TemplateEffectivenessTracker(self._state.get("templates", {}))
        self._state_lock = threading.RLock()

    def _load_state(self) -> Dict:
        path = _storage_path()
        try:
            import json
            if os.path.exists(path):
                with open(path, "r") as f:
                    return json.load(f)
        except Exception:
            logger.warning(
                "swallowed %s in _load_state() — continuing",
                "Exception", exc_info=True,
            )
        return {}

    def _save_state(self) -> None:
        path = _storage_path()
        try:
            import json
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with self._state_lock:
                state = {
                    "duration": self.duration.to_dict(),
                    "procurement": self.procurement.to_dict(),
                    "defects": self.defects.to_dict(),
                    "templates": self.templates.to_dict(),
                    "saved_at": datetime.now().isoformat(),
                }
                with open(path, "w") as f:
                    json.dump(state, f, indent=2)
        except Exception:
            logger.warning(
                "swallowed %s in _save_state() — continuing",
                "Exception", exc_info=True,
            )

    # ── Public API ──────────────────────────────────────────────────────

    def record_duration(
        self, project_id: str, activity_type: str,
        planned_days: int, actual_days: int, activity_name: str = "",
    ) -> Dict[str, Any]:
        result = self.duration.record_actual_duration(
            project_id, activity_type, planned_days, actual_days, activity_name
        )
        self._save_state()
        return result

    def get_calibrated_duration(
        self, project_id: str, activity_type: str, default_days: int = 5,
        min_samples: int = 3,
    ) -> Dict[str, Any]:
        return self.duration.get_calibrated_duration(project_id, activity_type, default_days, min_samples)

    def record_procurement_delivery(
        self, item_category: str, item_name: str,
        planned_lead_days: int, actual_lead_days: int,
        supplier: str = "", project_id: str = "",
    ) -> Dict[str, Any]:
        result = self.procurement.record_delivery(
            item_category, item_name, planned_lead_days, actual_lead_days, supplier, project_id
        )
        self._save_state()
        return result

    def get_lead_time_estimate(self, item_category: str, default_days: int = 60) -> Dict[str, Any]:
        return self.procurement.get_lead_time_estimate(item_category, default_days)

    def record_defect(
        self, activity_type: str, trade: str, severity: str,
        activity_name: str = "", project_id: str = "", root_cause: str = "",
    ) -> Dict[str, Any]:
        result = self.defects.record_defect(
            activity_type, trade, severity, activity_name, project_id, root_cause
        )
        self._save_state()
        return result

    def record_inspection(self, activity_type: str, trade: str, passed: bool, project_id: str = "") -> Dict[str, Any]:
        result = self.defects.record_inspection(activity_type, trade, passed, project_id)
        self._save_state()
        return result

    def get_defect_rankings(self) -> List[Dict[str, Any]]:
        return self.defects.get_defect_risk_ranking()

    def record_template_usage(
        self, template_id: str, project_type: str,
        success: bool, steps_completed: int = 0, total_steps: int = 0,
    ) -> Dict[str, Any]:
        result = self.templates.record_usage(
            template_id, project_type, success, steps_completed, total_steps
        )
        self._save_state()
        return result

    def get_best_templates(self, project_type: str) -> List[Dict[str, Any]]:
        return self.templates.get_best_templates(project_type)

    def get_summary(self) -> Dict[str, Any]:
        return {
            "duration_calibrations": len(self.duration.to_dict()),
            "procurement_categories": len(self.procurement.to_dict()),
            "defect_patterns": len(self.defects.to_dict()),
            "template_tracks": len(self.templates.to_dict()),
        }

    def list_surfaces(self) -> Dict[str, Any]:
        """Minimal list surface for CM wiring — no synthetic observations."""
        return {
            "status": "success",
            "execution_mode": "list_only",
            "surfaces": [
                {
                    "id": "duration_calibration",
                    "description": "Record actual vs planned activity durations",
                    "method": "record_duration",
                },
                {
                    "id": "procurement_lead_times",
                    "description": "Learn delivery lead times by category",
                    "method": "record_procurement_delivery",
                },
                {
                    "id": "defect_patterns",
                    "description": "Track defect rates by trade/activity type",
                    "method": "record_defect",
                },
                {
                    "id": "template_effectiveness",
                    "description": "Score workflow template success by project type",
                    "method": "record_template_usage",
                },
            ],
            "summary": self.get_summary(),
            "note": "Recording requires real project observations; no fabricated defaults.",
        }

    def reset(self) -> Dict[str, str]:
        self.duration = DurationCalibration()
        self.procurement = ProcurementLeadTimeLearner()
        self.defects = DefectPatternDetector()
        self.templates = TemplateEffectivenessTracker()
        self._save_state()
        return {"status": "reset"}
