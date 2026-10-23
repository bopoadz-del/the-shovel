"""Tests for app/core/workflow_templates.py — Multi-Domain Workflow Templates."""

from __future__ import annotations

import pytest

from app.core.workflow_templates import WorkflowTemplateLibrary
from app.schemas.execution_plan import ExecutionPlan


class TestWorkflowTemplateLibrary:
    def test_10_templates_registered(self):
        lib = WorkflowTemplateLibrary()
        ids = lib.all_template_ids()
        assert len(ids) == 10

    def test_all_template_ids(self):
        lib = WorkflowTemplateLibrary()
        ids = set(lib.all_template_ids())
        expected = {
            "new_project_setup",
            "monthly_progress",
            "change_order_impact",
            "delay_to_claim",
            "qa_defect_closeout",
            "submittal_to_install",
            "safety_incident_response",
            "commissioning_to_handover",
            "bim_coordination",
            "sustainability_assessment",
        }
        assert ids == expected

    def test_get_existing_template(self):
        lib = WorkflowTemplateLibrary()
        t = lib.get("new_project_setup")
        assert t is not None
        assert t.template_id == "new_project_setup"
        assert "schedule" in t.domains
        assert "cost" in t.domains

    def test_get_missing_template(self):
        lib = WorkflowTemplateLibrary()
        assert lib.get("nonexistent") is None

    def test_contains(self):
        lib = WorkflowTemplateLibrary()
        assert "new_project_setup" in lib
        assert "nonexistent" not in lib


class TestTemplatePlans:
    def _validate_plan(self, plan: ExecutionPlan) -> None:
        assert plan.understanding
        assert len(plan.steps) > 0
        for step in plan.steps:
            assert step.type
            assert step.description

    def test_new_project_setup_plan(self):
        lib = WorkflowTemplateLibrary()
        plan = lib.get("new_project_setup").build_plan({"project_type": "solar"})
        self._validate_plan(plan)
        assert "solar" in plan.understanding.lower() or "Set up" in plan.understanding

    def test_monthly_progress_plan(self):
        lib = WorkflowTemplateLibrary()
        plan = lib.get("monthly_progress").build_plan({})
        self._validate_plan(plan)
        step_types = [s.type for s in plan.steps]
        assert "progress_tracker" in step_types

    def test_change_order_impact_plan(self):
        lib = WorkflowTemplateLibrary()
        plan = lib.get("change_order_impact").build_plan({})
        self._validate_plan(plan)
        step_types = [s.type for s in plan.steps]
        assert "time_impact_analysis" in step_types
        assert "cost_impact" in step_types
        assert "entitlement_check" in step_types

    def test_delay_to_claim_plan(self):
        lib = WorkflowTemplateLibrary()
        plan = lib.get("delay_to_claim").build_plan({})
        self._validate_plan(plan)
        step_types = [s.type for s in plan.steps]
        assert "delay_analysis" in step_types
        assert "prolongation_cost" in step_types
        assert "claim_strength" in step_types

    def test_qa_defect_closeout_plan(self):
        lib = WorkflowTemplateLibrary()
        plan = lib.get("qa_defect_closeout").build_plan({})
        self._validate_plan(plan)
        step_types = [s.type for s in plan.steps]
        assert "defect_register" in step_types
        assert "hold_point" in step_types
        assert "rework_cost" in step_types
        assert "ncr_issue" in step_types

    def test_submittal_to_install_plan(self):
        lib = WorkflowTemplateLibrary()
        plan = lib.get("submittal_to_install").build_plan({})
        self._validate_plan(plan)
        step_types = [s.type for s in plan.steps]
        assert "spec_analyze" in step_types
        assert "submittal_log" in step_types
        assert "po_trigger" in step_types

    def test_safety_incident_response_plan(self):
        lib = WorkflowTemplateLibrary()
        plan = lib.get("safety_incident_response").build_plan({})
        self._validate_plan(plan)
        step_types = [s.type for s in plan.steps]
        assert "incident_report" in step_types
        assert "stop_work_order" in step_types
        assert "root_cause" in step_types

    def test_commissioning_to_handover_plan(self):
        lib = WorkflowTemplateLibrary()
        plan = lib.get("commissioning_to_handover").build_plan({})
        self._validate_plan(plan)
        step_types = [s.type for s in plan.steps]
        assert "test_results" in step_types
        assert "pc_cert" in step_types
        assert "om_manual" in step_types

    def test_bim_coordination_plan(self):
        lib = WorkflowTemplateLibrary()
        plan = lib.get("bim_coordination").build_plan({})
        self._validate_plan(plan)
        step_types = [s.type for s in plan.steps]
        assert "bim_analysis" in step_types
        assert "clash_detection" in step_types
        assert "qto_extraction" in step_types
        assert "wbs_from_bim" in step_types

    def test_sustainability_assessment_plan(self):
        lib = WorkflowTemplateLibrary()
        plan = lib.get("sustainability_assessment").build_plan({})
        self._validate_plan(plan)
        step_types = [s.type for s in plan.steps]
        assert "carbon_calc" in step_types
        assert "carbon_cost_premium" in step_types
        assert "esg_report" in step_types


class TestTemplateDomainCoverage:
    def test_schedule_domain_covered(self):
        lib = WorkflowTemplateLibrary()
        templates = lib.find_by_domain("schedule")
        assert len(templates) >= 3  # new_project, monthly, change_order, delay_to_claim, bim

    def test_cost_domain_covered(self):
        lib = WorkflowTemplateLibrary()
        templates = lib.find_by_domain("cost")
        assert len(templates) >= 3

    def test_quality_domain_covered(self):
        lib = WorkflowTemplateLibrary()
        templates = lib.find_by_domain("quality")
        assert len(templates) >= 2

    def test_safety_domain_covered(self):
        lib = WorkflowTemplateLibrary()
        templates = lib.find_by_domain("safety")
        assert len(templates) >= 1

    def test_procurement_domain_covered(self):
        lib = WorkflowTemplateLibrary()
        templates = lib.find_by_domain("procurement")
        assert len(templates) >= 2

    def test_contract_domain_covered(self):
        lib = WorkflowTemplateLibrary()
        templates = lib.find_by_domain("contract")
        assert len(templates) >= 2

    def test_risk_domain_covered(self):
        lib = WorkflowTemplateLibrary()
        templates = lib.find_by_domain("risk")
        assert len(templates) >= 3

    def test_bim_domain_covered(self):
        lib = WorkflowTemplateLibrary()
        templates = lib.find_by_domain("bim")
        assert len(templates) >= 1

    def test_commissioning_domain_covered(self):
        lib = WorkflowTemplateLibrary()
        templates = lib.find_by_domain("commissioning")
        assert len(templates) >= 1

    def test_handover_domain_covered(self):
        lib = WorkflowTemplateLibrary()
        templates = lib.find_by_domain("handover")
        assert len(templates) >= 1


class TestTemplateLibraryQueries:
    def test_list_templates(self):
        lib = WorkflowTemplateLibrary()
        items = lib.list_templates()
        assert len(items) == 10
        for item in items:
            assert "id" in item
            assert "name" in item
            assert "description" in item
            assert "domains" in item

    def test_list_templates_unique_ids(self):
        lib = WorkflowTemplateLibrary()
        items = lib.list_templates()
        ids = [i["id"] for i in items]
        assert len(ids) == len(set(ids))

    def test_find_by_domain_returns_typed(self):
        lib = WorkflowTemplateLibrary()
        templates = lib.find_by_domain("schedule")
        for t in templates:
            assert "schedule" in t.domains
