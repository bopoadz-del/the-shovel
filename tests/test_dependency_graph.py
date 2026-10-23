"""Tests for app/core/dependency_graph.py — Cross-Domain Dependency Graph."""

from __future__ import annotations

from typing import Any, Dict

import pytest

from app.core.dependency_graph import (
    DEFAULT_RULES,
    DependencyGraph,
    DependencyRule,
    Domain,
    ProjectHealthAggregator,
    SuggestedAction,
    TriggerCondition,
    _domain_to_trigger,
    _status_to_trigger,
)


class TestDefaultRules:
    def test_25_rules_present(self):
        assert len(DEFAULT_RULES) == 25

    def test_all_rules_have_unique_ids(self):
        ids = [r.rule_id for r in DEFAULT_RULES]
        assert len(ids) == len(set(ids))

    def test_all_domains_represented(self):
        source_domains = {r.source_domain for r in DEFAULT_RULES}
        all_domains = set(Domain)
        assert all_domains == source_domains, f"Missing: {all_domains - source_domains}"

    def test_procurement_delay_rule_exists(self):
        rules = [r for r in DEFAULT_RULES if r.source_domain == Domain.PROCUREMENT and r.trigger == TriggerCondition.DELAYED]
        assert len(rules) >= 1
        assert any("schedule" in [d.value for d in r.target_domains] for r in rules)

    def test_safety_incident_rule_exists(self):
        rules = [r for r in DEFAULT_RULES if r.source_domain == Domain.SAFETY and r.trigger == TriggerCondition.CRITICAL_INCIDENT]
        assert len(rules) >= 1

    def test_schedule_delay_claim_rule_exists(self):
        rules = [r for r in DEFAULT_RULES if r.source_domain == Domain.SCHEDULE and r.trigger == TriggerCondition.DELAYED]
        assert len(rules) >= 1
        assert any(Domain.CONTRACT in r.target_domains for r in rules)
        r004 = next(r for r in rules if r.rule_id == "R004")
        assert SuggestedAction.CREATE_CLAIM not in r004.suggested_actions
        assert SuggestedAction.CALCULATE_PROLONGATION_COST not in r004.suggested_actions
        assert "do not recommend a claim" in r004.description.lower()

    def test_bim_clash_rfi_rule_exists(self):
        rules = [r for r in DEFAULT_RULES if r.source_domain == Domain.BIM and r.trigger == TriggerCondition.DETECTED]
        assert len(rules) >= 1
        assert any(SuggestedAction.GENERATE_RFI in r.suggested_actions for r in rules)

    def test_commissioning_to_handover_rule_exists(self):
        rules = [r for r in DEFAULT_RULES if r.source_domain == Domain.COMMISSIONING and r.trigger == TriggerCondition.COMPLETED]
        assert len(rules) >= 1
        assert any(Domain.HANDOVER in r.target_domains for r in rules)


class TestDependencyGraph:
    def test_empty_graph(self):
        g = DependencyGraph(rules=[])
        affected, actions = g.trigger_analysis(Domain.SCHEDULE, TriggerCondition.DELAYED)
        assert len(affected) == 0
        assert len(actions) == 0

    def test_trigger_analysis_procurement_delay(self):
        g = DependencyGraph()
        affected, actions = g.trigger_analysis(Domain.PROCUREMENT, TriggerCondition.DELAYED)
        assert Domain.SCHEDULE in affected
        assert Domain.COST in affected
        assert SuggestedAction.UPDATE_SCHEDULE in actions
        assert SuggestedAction.GENERATE_RECOVERY_OPTIONS in actions

    def test_trigger_analysis_quality_fail(self):
        g = DependencyGraph()
        affected, actions = g.trigger_analysis(Domain.QUALITY, TriggerCondition.FAILED)
        assert Domain.SCHEDULE in affected
        assert SuggestedAction.HOLD_WORK in actions
        assert SuggestedAction.ISSUE_NCR in actions

    def test_trigger_analysis_safety_critical(self):
        g = DependencyGraph()
        affected, actions = g.trigger_analysis(Domain.SAFETY, TriggerCondition.CRITICAL_INCIDENT)
        assert Domain.SCHEDULE in affected
        assert Domain.RISK in affected
        assert SuggestedAction.STOP_WORK in actions

    def test_trigger_analysis_no_match(self):
        g = DependencyGraph()
        affected, actions = g.trigger_analysis(Domain.HANDOVER, TriggerCondition.DELAYED)
        assert len(affected) == 0

    def test_cascade_analysis_procurement(self):
        g = DependencyGraph()
        results = g.cascade_analysis(Domain.PROCUREMENT, TriggerCondition.DELAYED, max_depth=3)
        # Should find: PROCUREMENT(delay) -> SCHEDULE(delay) -> CONTRACT(claim check)
        domains_hit = {r[1] for r in results}
        assert Domain.SCHEDULE in domains_hit
        assert len(results) > 0

    def test_cascade_respects_max_depth(self):
        g = DependencyGraph()
        shallow = g.cascade_analysis(Domain.PROCUREMENT, TriggerCondition.DELAYED, max_depth=1)
        deep = g.cascade_analysis(Domain.PROCUREMENT, TriggerCondition.DELAYED, max_depth=3)
        assert len(shallow) <= len(deep)

    def test_get_rules_for_source_domain(self):
        g = DependencyGraph()
        schedule_rules = g.get_rules_for_source(Domain.SCHEDULE)
        assert len(schedule_rules) > 0
        for r in schedule_rules:
            assert r.source_domain == Domain.SCHEDULE

    def test_get_rules_for_target_domain(self):
        g = DependencyGraph()
        schedule_affected = g.get_rules_for_target(Domain.SCHEDULE)
        assert len(schedule_affected) > 0
        # At least procurement delay and quality fail affect schedule
        sources = {r.source_domain for r in schedule_affected}
        assert Domain.PROCUREMENT in sources or Domain.QUALITY in sources

    def test_full_project_health_rules(self):
        g = DependencyGraph()
        health = g.full_project_health_rules()
        assert len(health) > 0
        assert any("schedule" in key for key in health.keys())


class TestProjectHealthAggregator:
    def test_healthy_project(self):
        g = DependencyGraph()
        agg = ProjectHealthAggregator(g)
        domain_status: Dict[Domain, Dict[str, Any]] = {
            Domain.SCHEDULE: {"status": "healthy", "progress": 85},
            Domain.COST: {"status": "healthy", "variance": 2},
            Domain.QUALITY: {"status": "healthy", "pass_rate": 98},
        }
        health = agg.aggregate(domain_status)
        assert health["overall_status"] == "healthy"
        assert health["overall_score"] >= 80
        assert len(health["cross_domain_flags"]) == 0

    def test_critical_schedule(self):
        g = DependencyGraph()
        agg = ProjectHealthAggregator(g)
        domain_status = {
            Domain.SCHEDULE: {"status": "delayed"},
        }
        health = agg.aggregate(domain_status)
        # Should flag contract and cost as affected
        assert len(health["cross_domain_flags"]) >= 1
        flag = health["cross_domain_flags"][0]
        assert flag["source"] == "schedule"
        assert "contract" in flag["affected_domains"] or "cost" in flag["affected_domains"]

    def test_mixed_project(self):
        g = DependencyGraph()
        agg = ProjectHealthAggregator(g)
        domain_status = {
            Domain.SCHEDULE: {"status": "healthy"},
            Domain.COST: {"status": "variance"},
            Domain.QUALITY: {"status": "failed"},
            Domain.SAFETY: {"status": "healthy"},
        }
        health = agg.aggregate(domain_status)
        assert health["overall_status"] in ("warning", "critical")
        assert len(health["cross_domain_flags"]) >= 2

    def test_empty_domain_status(self):
        g = DependencyGraph()
        agg = ProjectHealthAggregator(g)
        health = agg.aggregate({})
        assert health["overall_status"] == "unknown"
        assert health["overall_score"] == 0


class TestUtilityFunctions:
    def test_domain_to_trigger(self):
        assert _domain_to_trigger(Domain.SCHEDULE) == TriggerCondition.DELAYED
        assert _domain_to_trigger(Domain.COST) == TriggerCondition.VARIANCE_THRESHOLD
        assert _domain_to_trigger(Domain.QUALITY) == TriggerCondition.FAILED
        assert _domain_to_trigger(Domain.SAFETY) == TriggerCondition.CRITICAL_INCIDENT
        assert _domain_to_trigger(Domain.PROCUREMENT) == TriggerCondition.DELAYED
        assert _domain_to_trigger(Domain.CONTRACT) == TriggerCondition.ISSUED
        assert _domain_to_trigger(Domain.RISK) == TriggerCondition.DETECTED
        assert _domain_to_trigger(Domain.BIM) == TriggerCondition.DETECTED

    def test_status_to_trigger(self):
        assert _status_to_trigger("delayed") == TriggerCondition.DELAYED
        assert _status_to_trigger("failed") == TriggerCondition.FAILED
        assert _status_to_trigger("critical") == TriggerCondition.CRITICAL_INCIDENT
        assert _status_to_trigger("variance") == TriggerCondition.VARIANCE_THRESHOLD
        assert _status_to_trigger("overrun") == TriggerCondition.VARIANCE_THRESHOLD
        assert _status_to_trigger("detected") == TriggerCondition.DETECTED
        assert _status_to_trigger("unknown") is None

    def test_quality_critical_maps_to_failed(self):
        assert _status_to_trigger("critical", domain=Domain.QUALITY) == TriggerCondition.FAILED
        assert _status_to_trigger("failed", domain=Domain.QUALITY) == TriggerCondition.FAILED
        assert (
            _status_to_trigger("critical", domain=Domain.SAFETY)
            == TriggerCondition.CRITICAL_INCIDENT
        )
