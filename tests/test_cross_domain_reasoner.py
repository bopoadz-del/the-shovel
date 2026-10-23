"""Tests for app/core/cross_domain_reasoner.py — Phase 4 Cross-Domain Reasoning."""

from __future__ import annotations

import pytest

from app.core.cross_domain_reasoner import (
    CrossDomainIntentDetector,
    CrossDomainReasoner,
    MultiDomainPlanBuilder,
    SystemPromptInjector,
    TemplateMatcher,
    _CROSS_DOMAIN_KEYWORDS,
    _KEYWORD_ROUTES,
    _POST_TOOL_DOMAIN_TRIGGERS,
    _TEMPLATE_TRIGGERS,
)
from app.core.dependency_graph import Domain


class TestTemplateTriggers:
    def test_all_templates_have_triggers(self):
        assert len(_TEMPLATE_TRIGGERS) == 10
        expected = {
            "new_project_setup", "monthly_progress", "change_order_impact",
            "delay_to_claim", "qa_defect_closeout", "submittal_to_install",
            "safety_incident_response", "commissioning_to_handover",
            "bim_coordination", "sustainability_assessment",
        }
        assert set(_TEMPLATE_TRIGGERS.keys()) == expected

    def test_each_template_has_multiple_triggers(self):
        for template_id, triggers in _TEMPLATE_TRIGGERS.items():
            assert len(triggers) >= 3, f"{template_id} has too few triggers"


class TestCrossDomainKeywords:
    def test_keywords_cover_all_relevant_domains(self):
        """Coverage is over the FULL route — the domain a keyword is about plus
        the domains it implicates.

        Asserting over implicated domains alone used to hide a modelling error:
        PROCUREMENT was listed as its own target on "material shortage" and
        "lead time", which is how a table with no source field expressed "this
        keyword is about procurement". Now the source is explicit, so coverage
        has to be measured over both halves or a whole domain looks missing.
        """
        all_domains = set()
        for source, targets in _KEYWORD_ROUTES.values():
            all_domains.add(source)
            all_domains.update(targets)
        assert Domain.SCHEDULE in all_domains
        assert Domain.COST in all_domains
        assert Domain.QUALITY in all_domains
        assert Domain.SAFETY in all_domains
        assert Domain.PROCUREMENT in all_domains
        assert Domain.CONTRACT in all_domains
        assert Domain.RISK in all_domains

    def test_delay_triggers_cost_not_contract(self):
        """Generic delay is schedule + cost. CONTRACT needs explicit claim/EOT."""
        assert Domain.COST in _CROSS_DOMAIN_KEYWORDS["delay"]
        assert Domain.CONTRACT not in _CROSS_DOMAIN_KEYWORDS["delay"]
        assert _KEYWORD_ROUTES["claim"][0] == Domain.CONTRACT
        assert _KEYWORD_ROUTES["extension of time"][0] == Domain.CONTRACT

    def test_claim_triggers_schedule_cost_and_risk(self):
        domains = _CROSS_DOMAIN_KEYWORDS["claim"]
        assert Domain.SCHEDULE in domains
        assert Domain.COST in domains
        assert Domain.RISK in domains


class TestPostToolDomainTriggers:
    def test_generate_wbs_triggers_cost_and_procurement(self):
        domains = _POST_TOOL_DOMAIN_TRIGGERS["generate_wbs"]
        assert Domain.COST in domains
        assert Domain.PROCUREMENT in domains

    def test_parse_primavera_triggers_cost_contract_and_risk(self):
        domains = _POST_TOOL_DOMAIN_TRIGGERS["parse_primavera_schedule"]
        assert Domain.COST in domains
        assert Domain.CONTRACT in domains
        assert Domain.RISK in domains


class TestTemplateMatcher:
    def test_match_new_project_setup(self):
        matcher = TemplateMatcher()
        results = matcher.match("set up a new data center project")
        assert len(results) > 0
        assert results[0][0] == "new_project_setup"
        assert results[0][1] > 0

    def test_match_monthly_progress(self):
        matcher = TemplateMatcher()
        results = matcher.match("give me the monthly progress report")
        assert len(results) > 0
        assert results[0][0] == "monthly_progress"

    def test_match_delay_to_claim(self):
        matcher = TemplateMatcher()
        results = matcher.match("help me build a delay claim for the electrical package")
        assert len(results) > 0
        assert results[0][0] == "delay_to_claim"

    def test_match_commissioning(self):
        matcher = TemplateMatcher()
        results = matcher.match("generate a commissioning checklist for hvac")
        assert len(results) > 0
        assert results[0][0] == "commissioning_to_handover"

    def test_no_match(self):
        matcher = TemplateMatcher()
        results = matcher.match("hello how are you today")
        assert len(results) == 0

    def test_best_match_above_threshold(self):
        matcher = TemplateMatcher()
        template_id = matcher.best_match("new project setup for solar plant", threshold=0.1)
        assert template_id == "new_project_setup"

    def test_best_match_below_threshold(self):
        matcher = TemplateMatcher()
        template_id = matcher.best_match("hello", threshold=0.5)
        assert template_id is None

    def test_match_ordering(self):
        matcher = TemplateMatcher()
        results = matcher.match("monthly progress report for my project")
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)


class TestCrossDomainIntentDetector:
    def test_detect_delay_domains(self):
        detector = CrossDomainIntentDetector()
        domains = detector.detect_additional_domains("we are delayed by 2 weeks on the foundation")
        assert Domain.COST in domains
        assert Domain.CONTRACT not in domains

    def test_detect_claim_domains(self):
        detector = CrossDomainIntentDetector()
        domains = detector.detect_additional_domains("file a claim for the delay")
        assert Domain.SCHEDULE in domains
        assert Domain.COST in domains
        assert Domain.RISK in domains

    def test_detect_quality_domains(self):
        detector = CrossDomainIntentDetector()
        domains = detector.detect_additional_domains("defect found in concrete pour")
        assert Domain.SCHEDULE in domains
        assert Domain.SAFETY in domains

    def test_no_additional_domains(self):
        detector = CrossDomainIntentDetector()
        domains = detector.detect_additional_domains("hello")
        assert len(domains) == 0

    def test_cross_domain_context(self):
        detector = CrossDomainIntentDetector()
        context = detector.get_cross_domain_context("we are delayed")
        assert "schedule" in context.lower() or "cost" in context.lower()

    def test_cross_domain_context_empty(self):
        detector = CrossDomainIntentDetector()
        context = detector.get_cross_domain_context("hello")
        assert context == ""

    def test_suggest_follow_up_tools(self):
        detector = CrossDomainIntentDetector()
        tools = detector.suggest_follow_up_tools("generate_wbs", "what about the cost?")
        assert len(tools) > 0

    def test_suggest_follow_up_tools_unknown(self):
        detector = CrossDomainIntentDetector()
        tools = detector.suggest_follow_up_tools("unknown_tool", "test")
        assert len(tools) == 0


class TestMultiDomainPlanBuilder:
    def test_build_new_project_setup(self):
        builder = MultiDomainPlanBuilder()
        plan = builder.build_from_template("new_project_setup", {"project_type": "data_center"})
        assert plan is not None
        assert plan["template_id"] == "new_project_setup"
        assert plan["understanding"]
        assert len(plan["steps"]) > 0
        assert "schedule" in plan["domains"]

    def test_build_delay_to_claim(self):
        builder = MultiDomainPlanBuilder()
        plan = builder.build_from_template("delay_to_claim")
        assert plan is not None
        assert plan["template_id"] == "delay_to_claim"
        assert len(plan["steps"]) >= 5

    def test_build_unknown_template(self):
        builder = MultiDomainPlanBuilder()
        plan = builder.build_from_template("nonexistent")
        assert plan is None

    def test_build_commissioning(self):
        builder = MultiDomainPlanBuilder()
        plan = builder.build_from_template("commissioning_to_handover")
        assert plan is not None
        step_types = [s.get("step_type", "") for s in plan["steps"]]
        actions = [s["params"].get("action", "") for s in plan["steps"]]
        assert "test_results" in step_types
        assert "commissioning_checklist" in actions

    def test_step_type_block_map_coverage(self):
        builder = MultiDomainPlanBuilder()
        mapping = builder._step_type_block_map()
        # Key step types must be mapped
        assert mapping["build_wbs"] == "construction"
        assert mapping["extract_boq"] == "boq_processor"
        assert mapping["cost_load"] == "construction"
        assert mapping["defect_register"] == "construction"
        assert mapping["incident_report"] == "construction"
        assert mapping["esg_report"] == "construction"


class TestSystemPromptInjector:
    def test_inject_with_template_match(self):
        injector = SystemPromptInjector()
        base = "You are a construction assistant."
        enhanced = injector.inject_for_turn(base, "set up a new solar project")
        assert "new_project_setup" in enhanced.lower() or "workflow" in enhanced.lower()
        assert len(enhanced) > len(base)

    def test_inject_with_cross_domain(self):
        injector = SystemPromptInjector()
        base = "You are a construction assistant."
        enhanced = injector.inject_for_turn(base, "we are delayed on the foundation")
        assert "cross-domain" in enhanced.lower() or "domain" in enhanced.lower()

    def test_inject_no_change(self):
        injector = SystemPromptInjector()
        base = "You are a construction assistant."
        enhanced = injector.inject_for_turn(base, "hello")
        assert enhanced == base

    def test_history_suggestions(self):
        injector = SystemPromptInjector()
        history = [
            {"role": "assistant", "content": "I ran generate_wbs for your project."},
        ]
        result = injector._history_based_suggestions(history)
        assert "cost" in result.lower() or "procurement" in result.lower()


class TestCrossDomainReasoner:
    def test_analyze_turn_new_project(self):
        reasoner = CrossDomainReasoner()
        result = reasoner.analyze_turn("set up a new data center project")
        assert result["matched_template"] == "new_project_setup"
        assert result["matched_template_score"] > 0
        assert isinstance(result["additional_domains"], list)
        assert isinstance(result["suggested_tools"], list)

    def test_analyze_turn_with_cross_domain(self):
        reasoner = CrossDomainReasoner()
        result = reasoner.analyze_turn("we have a delay on the critical path")
        assert len(result["additional_domains"]) > 0
        assert "cost" in result["additional_domains"] or "contract" in result["additional_domains"]

    def test_build_plan(self):
        reasoner = CrossDomainReasoner()
        plan = reasoner.build_plan("monthly_progress")
        assert plan is not None
        assert len(plan["steps"]) > 0

    def test_inject_prompt(self):
        reasoner = CrossDomainReasoner()
        base = "You are a construction assistant."
        enhanced = reasoner.inject_prompt(base, "new project setup")
        assert len(enhanced) > len(base)

    def test_inject_prompt_no_change(self):
        reasoner = CrossDomainReasoner()
        base = "You are a construction assistant."
        enhanced = reasoner.inject_prompt(base, "hello")
        assert enhanced == base

    def test_get_post_tool_suggestions(self):
        reasoner = CrossDomainReasoner()
        tools = reasoner.get_post_tool_suggestions("generate_wbs", "what about cost?")
        assert len(tools) > 0

    def test_get_post_tool_suggestions_unknown(self):
        reasoner = CrossDomainReasoner()
        tools = reasoner.get_post_tool_suggestions("unknown", "test")
        assert len(tools) == 0

    def test_all_templates_matchable(self):
        reasoner = CrossDomainReasoner()
        for template_id, triggers in _TEMPLATE_TRIGGERS.items():
            # Use first trigger phrase for each template
            test_message = triggers[0] + " for my project"
            result = reasoner.analyze_turn(test_message)
            assert result["matched_template"] == template_id, f"Failed to match {template_id}"
