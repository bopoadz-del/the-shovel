"""Tests for app/core/construction_learning.py — Phase 6 Construction Learning."""

from __future__ import annotations

import pytest

from app.core.construction_learning import (
    ConstructionLearningEngine,
    DefectPatternDetector,
    DurationCalibration,
    ProcurementLeadTimeLearner,
    TemplateEffectivenessTracker,
)


class TestDurationCalibration:
    def test_record_and_calibrate(self):
        cal = DurationCalibration()
        cal.record_actual_duration("P-001", "concrete_pour", planned_days=5, actual_days=7)
        cal.record_actual_duration("P-001", "concrete_pour", planned_days=5, actual_days=6)
        cal.record_actual_duration("P-001", "concrete_pour", planned_days=5, actual_days=8)

        result = cal.get_calibrated_duration("P-001", "concrete_pour", default_days=5, min_samples=3)
        assert result["source"] == "calibrated"
        assert result["calibrated_days"] == 7  # median of [6, 7, 8]
        assert result["samples"] == 3
        assert "confidence" in result

    def test_insufficient_samples_fallback(self):
        cal = DurationCalibration()
        cal.record_actual_duration("P-001", "steel", planned_days=10, actual_days=12)
        result = cal.get_calibrated_duration("P-001", "steel", default_days=10, min_samples=5)
        assert result["source"] == "default"
        assert result["calibrated_days"] == 10

    def test_variance_calculation(self):
        cal = DurationCalibration()
        result = cal.record_actual_duration("P-001", "test", planned_days=10, actual_days=12)
        assert result["variance_pct"] == 20.0  # (12-10)/10*100

    def test_high_confidence(self):
        cal = DurationCalibration()
        for i in range(15):
            cal.record_actual_duration("P-002", "painting", planned_days=3, actual_days=3 + (i % 3))
        result = cal.get_calibrated_duration("P-002", "painting", min_samples=10)
        assert result["confidence"] in ("high", "medium", "low")

    def test_type_summary(self):
        cal = DurationCalibration()
        cal.record_actual_duration("P-001", "concrete", 5, 7)
        cal.record_actual_duration("P-001", "steel", 10, 12)
        summary = cal.get_type_summary("P-001")
        assert "concrete" in summary
        assert "steel" in summary

    def test_persistence_roundtrip(self):
        cal = DurationCalibration()
        cal.record_actual_duration("P-001", "test", 5, 7)
        data = cal.to_dict()
        cal2 = DurationCalibration.from_dict(data)
        result = cal2.get_calibrated_duration("P-001", "test", min_samples=1)
        assert result["source"] == "calibrated"


class TestProcurementLeadTimeLearner:
    def test_record_and_estimate(self):
        learner = ProcurementLeadTimeLearner()
        learner.record_delivery("switchgear", "MV Switchgear", 90, 105)
        learner.record_delivery("switchgear", "MV Switchgear", 90, 95)
        learner.record_delivery("switchgear", "LV Switchgear", 60, 75)

        result = learner.get_lead_time_estimate("switchgear", default_days=90, min_samples=2)
        assert result["source"] == "learned"
        assert result["samples"] == 3
        assert result["estimated_days"] >= result["median"]

    def test_p80_conservative(self):
        learner = ProcurementLeadTimeLearner()
        for i in range(10):
            learner.record_delivery("cables", f"Cable-{i}", 30, 30 + i * 2)
        result = learner.get_lead_time_estimate("cables", min_samples=5)
        # P80 should be >= median
        assert result["p80"] >= result["median"]

    def test_insufficient_samples(self):
        learner = ProcurementLeadTimeLearner()
        result = learner.get_lead_time_estimate("pumps", default_days=45, min_samples=5)
        assert result["source"] == "default"
        assert result["estimated_days"] == 45

    def test_category_summary(self):
        learner = ProcurementLeadTimeLearner()
        learner.record_delivery("hvac", "Chiller", 60, 70)
        learner.record_delivery("electrical", "Transformer", 45, 50)
        summary = learner.get_category_summary()
        assert "hvac" in summary
        assert "electrical" in summary

    def test_variance_pct(self):
        learner = ProcurementLeadTimeLearner()
        result = learner.record_delivery("test", "Item", 100, 120)
        assert result["variance_pct"] == 20.0


class TestDefectPatternDetector:
    def test_record_and_rank(self):
        det = DefectPatternDetector()
        det.record_defect("concrete", "carpenter", "major", "Wall L3")
        det.record_defect("concrete", "carpenter", "minor", "Wall L4")
        det.record_defect("concrete", "carpenter", "critical", "Slab L2")
        det.record_inspection("concrete", "carpenter", passed=False)
        det.record_inspection("concrete", "carpenter", passed=True)
        det.record_inspection("concrete", "carpenter", passed=True)

        rankings = det.get_defect_risk_ranking(min_samples=1)
        assert len(rankings) == 1
        assert rankings[0]["activity_type"] == "concrete"
        assert rankings[0]["trade"] == "carpenter"
        assert rankings[0]["weighted_defects"] == 9  # 3 + 1 + 5

    def test_weighted_scoring(self):
        det = DefectPatternDetector()
        det.record_defect("mep", "plumber", "critical")
        det.record_defect("mep", "plumber", "minor")
        rankings = det.get_defect_risk_ranking(min_samples=0)
        assert rankings[0]["weighted_defects"] == 6  # 5 + 1

    def test_fail_rate(self):
        det = DefectPatternDetector()
        det.record_inspection("finishes", "painter", passed=True)
        det.record_inspection("finishes", "painter", passed=True)
        det.record_inspection("finishes", "painter", passed=False)
        rankings = det.get_defect_risk_ranking(min_samples=1)
        assert rankings[0]["fail_rate_pct"] == pytest.approx(33.3, abs=0.5)

    def test_high_risk_types(self):
        det = DefectPatternDetector()
        for _ in range(5):
            det.record_inspection("concrete", "carpenter", passed=False)
        for _ in range(5):
            det.record_inspection("steel", "erector", passed=True)
        high_risk = det.get_high_risk_types(threshold=40.0)
        assert len(high_risk) >= 1
        assert high_risk[0]["activity_type"] == "concrete"

    def test_no_rankings_below_min_samples(self):
        det = DefectPatternDetector()
        det.record_defect("test", "trade", "minor")
        rankings = det.get_defect_risk_ranking(min_samples=5)
        assert len(rankings) == 0

    def test_persistence_roundtrip(self):
        det = DefectPatternDetector()
        det.record_defect("test", "trade", "major")
        data = det.to_dict()
        det2 = DefectPatternDetector.from_dict(data)
        rankings = det2.get_defect_risk_ranking(min_samples=0)
        assert len(rankings) == 1


class TestTemplateEffectivenessTracker:
    def test_record_and_rank(self):
        tracker = TemplateEffectivenessTracker()
        tracker.record_usage("new_project_setup", "data_center", success=True, steps_completed=8, total_steps=8)
        tracker.record_usage("new_project_setup", "data_center", success=True, steps_completed=8, total_steps=8)
        tracker.record_usage("monthly_progress", "data_center", success=True, steps_completed=10, total_steps=10)
        tracker.record_usage("monthly_progress", "data_center", success=False, steps_completed=3, total_steps=10)

        best = tracker.get_best_templates("data_center")
        assert len(best) == 2
        assert best[0]["template_id"] == "new_project_setup"
        assert best[0]["success_rate_pct"] == 100.0

    def test_recommendation(self):
        tracker = TemplateEffectivenessTracker()
        tracker.record_usage("bim_coordination", "data_center", success=True)
        rec = tracker.get_template_recommendation("data_center")
        assert rec == "bim_coordination"

    def test_no_recommendation(self):
        tracker = TemplateEffectivenessTracker()
        rec = tracker.get_template_recommendation("solar")
        assert rec is None

    def test_below_min_uses(self):
        tracker = TemplateEffectivenessTracker()
        tracker.record_usage("test", "type", success=True)
        best = tracker.get_best_templates("type", min_uses=5)
        assert len(best) == 0

    def test_persistence_roundtrip(self):
        tracker = TemplateEffectivenessTracker()
        tracker.record_usage("test", "type", success=True)
        data = tracker.to_dict()
        tracker2 = TemplateEffectivenessTracker.from_dict(data)
        best = tracker2.get_best_templates("type", min_uses=1)
        assert len(best) == 1


class TestConstructionLearningEngine:
    def test_record_duration(self):
        engine = ConstructionLearningEngine()
        result = engine.record_duration("P-001", "concrete", 5, 7)
        assert result["status"] == "recorded"

    def test_get_calibrated_duration(self):
        engine = ConstructionLearningEngine()
        for _ in range(5):
            engine.record_duration("P-001", "masonry", 10, 12)
        result = engine.get_calibrated_duration("P-001", "masonry")
        assert result["source"] == "calibrated"

    def test_record_procurement(self):
        engine = ConstructionLearningEngine()
        result = engine.record_procurement_delivery("switchgear", "MV", 90, 100)
        assert result["status"] == "recorded"

    def test_get_lead_time(self):
        engine = ConstructionLearningEngine()
        for i in range(5):
            engine.record_procurement_delivery("cables", f"Cable-{i}", 30, 30 + i)
        result = engine.get_lead_time_estimate("cables")
        assert result["source"] == "learned"

    def test_record_defect(self):
        engine = ConstructionLearningEngine()
        result = engine.record_defect("concrete", "carpenter", "major")
        assert result["status"] == "recorded"

    def test_defect_rankings(self):
        engine = ConstructionLearningEngine()
        engine.record_defect("concrete", "carpenter", "major")
        engine.record_inspection("concrete", "carpenter", passed=False)
        rankings = engine.get_defect_rankings()
        assert isinstance(rankings, list)

    def test_record_template_usage(self):
        engine = ConstructionLearningEngine()
        result = engine.record_template_usage("new_project_setup", "data_center", success=True)
        assert result["status"] == "recorded"

    def test_get_best_templates(self):
        engine = ConstructionLearningEngine()
        engine.record_template_usage("test_tmpl", "test_type", success=True)
        best = engine.get_best_templates("test_type")
        assert isinstance(best, list)

    def test_summary(self):
        engine = ConstructionLearningEngine()
        summary = engine.get_summary()
        assert "duration_calibrations" in summary
        assert "procurement_categories" in summary
        assert "defect_patterns" in summary
        assert "template_tracks" in summary

    def test_storage_defaults_under_data_dir(self, monkeypatch, tmp_path):
        from app.core import construction_learning as cl

        monkeypatch.delenv("CONSTRUCTION_LEARNING_STORAGE", raising=False)
        monkeypatch.setenv("DATA_DIR", str(tmp_path))
        cl.ConstructionLearningEngine.reset_shared_instance_cache()
        path = cl._storage_path()
        assert str(tmp_path) in path
        assert path.endswith("cerebrum_construction_learning.json")
        # DATA_DIR may itself live under /tmp (pytest tmp_path on Linux); assert
        # the learning file is under DATA_DIR/learning/, not a bare /tmp default.
        norm = path.replace("\\", "/")
        assert "/learning/cerebrum_construction_learning.json" in norm
        assert norm.startswith(str(tmp_path).replace("\\", "/"))

    def test_storage_explicit_env(self, monkeypatch, tmp_path):
        from app.core import construction_learning as cl

        custom = tmp_path / "custom_learning.json"
        monkeypatch.setenv("CONSTRUCTION_LEARNING_STORAGE", str(custom))
        cl.ConstructionLearningEngine.reset_shared_instance_cache()
        assert cl._storage_path() == str(custom)

    def test_reset(self):
        engine = ConstructionLearningEngine()
        engine.record_duration("P-001", "test", 5, 7)
        result = engine.reset()
        assert result["status"] == "reset"
        summary = engine.get_summary()
        assert summary["duration_calibrations"] == 0

    def test_shared_instance(self):
        engine = ConstructionLearningEngine.shared_instance()
        assert isinstance(engine, ConstructionLearningEngine)
        ConstructionLearningEngine.reset_shared_instance_cache()

    def test_persistence(self):
        engine = ConstructionLearningEngine()
        engine.reset()
        engine.record_duration("PERSIST", "test", 5, 10)

        engine2 = ConstructionLearningEngine()
        result = engine2.get_calibrated_duration("PERSIST", "test", min_samples=1)
        assert result["source"] == "calibrated"
        assert result["calibrated_days"] == 10
