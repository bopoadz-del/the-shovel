"""Tests for CM step → ConstructionContainer / block action aliases."""

from __future__ import annotations

import pytest

from app.core.cm_step_aliases import (
    ACTION_ALIASES,
    NO_AUTO_DISPATCH,
    PLAN_ONLY_STEPS,
    STEP_TO_TARGET,
    all_template_step_types,
    is_auto_dispatch,
    is_plan_only,
    resolve_action,
    resolve_step,
    unmapped_step_types,
)
from app.core.cross_domain_reasoner import MultiDomainPlanBuilder
from app.core.workflow_templates import WorkflowTemplateLibrary


def _known_construction_actions() -> set[str]:
    """The route keys ConstructionContainer.route() actually accepts.

    Derived from the handlers table, NOT hand-copied. Until 2026-08-12 this was
    a literal set of thirty names maintained by hand, which made the test below
    a false-green machine: deleting `"schedule_risk"` from the real handlers
    table left `STEP_TO_TARGET["recovery_options"] -> ("construction",
    "schedule_risk")` dangling, and this file stayed green because it was
    checking the alias against its own copy rather than against the router.
    Measured — that exact mutation was run, and only
    tests/test_construction_actions_reachable.py went red.

    Parsed rather than imported because the table is a local inside route().
    Keys, not `self.<method>` names: the table aliases on purpose.
    """
    import re
    from pathlib import Path

    src = Path("app/containers/construction/__init__.py").read_text(encoding="utf-8")
    start = src.index("handlers = {")
    block = src[start:src.index("\n        }", start)]
    keys = set(re.findall(r'"([a-z0-9_]+)":\s*self\.', block))
    assert len(keys) > 40, f"parsed only {len(keys)} route keys — has route() changed shape?"
    return keys


_KNOWN_CONSTRUCTION_ACTIONS = _known_construction_actions()

_KNOWN_BLOCKS = {
    "construction",
    "boq_processor",
    "document_engine",
    "spec_analyzer",
}

_CALLER_HANDLED_STEPS = {"render_artifact", "deliver"} | set(PLAN_ONLY_STEPS)


class TestStepAliasCoverage:
    def test_every_template_step_type_is_mapped(self):
        missing = unmapped_step_types()
        assert missing == [], f"Unmapped STEP_* constants: {missing}"

    def test_all_targets_use_known_blocks(self):
        for step, (block, action) in STEP_TO_TARGET.items():
            if step in _CALLER_HANDLED_STEPS or step in PLAN_ONLY_STEPS:
                assert (block, action) == NO_AUTO_DISPATCH
                continue
            assert block in _KNOWN_BLOCKS, f"{step} → unknown block {block}"
            if block == "construction":
                assert action in _KNOWN_CONSTRUCTION_ACTIONS, (
                    f"{step} → unknown construction action {action}"
                )

    def test_render_and_deliver_are_not_health_check(self):
        assert resolve_step("render_artifact") == NO_AUTO_DISPATCH
        assert resolve_step("deliver") == NO_AUTO_DISPATCH
        assert not is_auto_dispatch(resolve_step("render_artifact"))
        assert not is_auto_dispatch(resolve_step("deliver"))
        assert resolve_action("render_artifact") == "render_artifact"
        assert resolve_action("deliver") == "deliver"
        assert "render_artifact" not in ACTION_ALIASES
        assert "deliver" not in ACTION_ALIASES
        assert ACTION_ALIASES.get("render_artifact") != "health_check"
        assert STEP_TO_TARGET["render_artifact"][1] != "health_check"
        assert STEP_TO_TARGET["deliver"][1] != "health_check"

    def test_resolve_action_aliases(self):
        assert resolve_action("payment_cert") == "payment_certificate"
        assert resolve_action("procurement_plan") == "procurement_list_generator"
        assert resolve_action("float_analysis") == "parse_primavera_schedule"
        assert resolve_action("extract_boq") == "extract_boq"  # specialist block
        assert resolve_action("payment_certificate") == "payment_certificate"

    def test_resolve_step(self):
        assert resolve_step("build_wbs") == ("construction", "generate_wbs")
        assert resolve_step("extract_boq") == ("boq_processor", None)
        assert resolve_step("nope") is None

    def test_formal_instruments_do_not_auto_dispatch(self):
        for step in ("ncr_issue", "stop_work_order", "pc_cert"):
            assert resolve_step(step) == NO_AUTO_DISPATCH
            assert not is_auto_dispatch(resolve_step(step))
            assert is_plan_only(step)
            assert resolve_action(step) == step
            assert step not in ACTION_ALIASES


class TestTemplatePlansResolve:
    def test_every_template_step_resolves(self):
        lib = WorkflowTemplateLibrary()
        builder = MultiDomainPlanBuilder(lib)
        for tid in lib.all_template_ids():
            plan = builder.build_from_template(tid)
            assert plan is not None
            assert len(plan["steps"]) > 0
            for step in plan["steps"]:
                if step.get("dispatch") is False:
                    assert step["block"] is None
                    assert step["params"].get("action") is None
                    assert step["step_type"] in _CALLER_HANDLED_STEPS
                    if is_plan_only(step["step_type"]):
                        assert step.get("needs_operator") is True
                        assert step.get("needs_caller_render") is False
                    else:
                        assert step.get("needs_caller_render") is True
                    continue
                assert step["block"] in _KNOWN_BLOCKS
                action = step["params"].get("action")
                if step["block"] == "construction":
                    assert action in _KNOWN_CONSTRUCTION_ACTIONS, (
                        f"{tid}: {step.get('step_type')} → {action}"
                    )

    def test_new_project_setup_uses_canonical_actions(self):
        plan = MultiDomainPlanBuilder().build_from_template("new_project_setup")
        actions = [s["params"].get("action") for s in plan["steps"] if s["block"] == "construction"]
        assert "generate_wbs" in actions
        assert "procurement_list_generator" in actions
        assert "risk_register_auto_populate" in actions
        # Must NOT leave template vocabulary as the action
        assert "procurement_plan" not in actions
        assert "build_wbs" not in actions

    def test_step_type_preserved(self):
        plan = MultiDomainPlanBuilder().build_from_template("monthly_progress")
        assert any(s.get("step_type") == "payment_cert" for s in plan["steps"])
        payment = next(s for s in plan["steps"] if s.get("step_type") == "payment_cert")
        assert payment["params"]["action"] == "payment_certificate"

    def test_render_artifact_is_caller_handled_not_health_check(self):
        plan = MultiDomainPlanBuilder().build_from_template("new_project_setup")
        render_steps = [s for s in plan["steps"] if s.get("step_type") == "render_artifact"]
        assert render_steps, "new_project_setup should include render_artifact"
        for step in render_steps:
            assert step["dispatch"] is False
            assert step["needs_caller_render"] is True
            assert step["block"] is None
            assert step["params"].get("action") is None
            assert step["params"].get("action") != "health_check"
