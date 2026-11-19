"""Every calculator name a hat offers is a DECLARED calculator, and the five
procedure-rule actions (document number, critical-rules gate, NCR next
status, design status, review timeline) answer through the real tool
dispatch instead of 'unknown calculation'.

The old guard checked ``resolve_binding``, which also resolves plain
``construction_knowledge`` functions by bare name, so a name could pass it
and still be unknown to ``construction_calc``. The guard here reads what
``construction_calc`` actually runs: ``formula_registry.calculators()``.

Every input is built by the test; each decision is checked with two inputs.
"""
from __future__ import annotations

import asyncio
import glob
import json

import pytest

import app.agents.runtime as rt
from app.lib import formula_registry


def _agent():
    return rt.Agent(name="u1", description="u1", system_prompt="u1",
                    allowed_blocks=["construction"])


def _dispatch(name: str, args: dict) -> dict:
    tc = {"id": "c1", "function": {"name": name, "arguments": json.dumps(args)}}
    return asyncio.run(_agent()._run_tool_call(tc))


def _calc(calculation: str, params: dict) -> dict:
    out = _dispatch("construction_calc", {"calculation": calculation, "params": params})
    return out["result"] if isinstance(out.get("result"), dict) else out


def _manifest_calc_names() -> set:
    names = set()
    for path in glob.glob("app/agents/manifests/*.json"):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        for action in data.get("allowed_actions") or []:
            if str(action).startswith("construction_calc:"):
                names.add(str(action).split(":", 1)[1])
    return names


# -- the general guard ---------------------------------------------------

def test_every_manifest_calculator_action_is_a_declared_calculator():
    names = _manifest_calc_names()
    assert names, "no construction_calc:<name> action found in any manifest"
    missing = sorted(names - set(formula_registry.calculators()))
    assert not missing, f"offered as construction_calc but not declared: {missing}"


def test_every_hat_calc_alias_target_is_a_declared_calculator():
    missing = sorted({calc for calc in rt._HAT_CALC_ALIASES.values()}
                     - set(formula_registry.calculators()))
    assert not missing, f"alias targets not declared as calculators: {missing}"


@pytest.mark.parametrize("name, owner", [
    ("generate_doc_number", "contracts"),
    ("enforce_critical_rules", "contracts"),
    ("next_ncr_status", "contracts"),
    ("validate_design_status", "qaqc"),
    ("check_review_timeline", "qaqc"),
])
def test_each_rule_action_is_declared_with_its_owner(name, owner):
    spec = formula_registry.get(name)
    assert spec is not None and spec.owner == owner
    assert spec.display_name and spec.description


def test_the_manifest_binding_names_the_declared_function():
    from app.agents.formulas import resolve_binding
    for path in ("app/agents/manifests/fork.hat.contracts.json",
                 "app/agents/manifests/fork.hat.qaqc.json"):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        for f in data.get("formulas") or []:
            spec = formula_registry.get(f["id"])
            if spec is None or not f.get("implemented_by"):
                continue
            assert resolve_binding(f["implemented_by"]) is spec.fn, f["id"]


# -- each action through the real construction_calc dispatch -------------

@pytest.mark.parametrize("params, number", [
    ({"doc_type": "NCR", "sequence": 7, "year": 2025}, "NCR-2025-007"),
    ({"doc_type": "rfi", "sequence": 42}, "RFI-0042"),
])
def test_document_number_through_dispatch(params, number):
    env = _calc("generate_doc_number", params)
    assert env["status"] == "success", env
    assert env["result"]["document_number"] == number


@pytest.mark.parametrize("body, compliant", [
    ("The design drawing package is approved by the consultant.", False),
    ("Concrete pour to the level 3 slab is complete.", True),
])
def test_critical_rules_gate_through_dispatch(body, compliant):
    env = _calc("enforce_critical_rules", {"document_text": body})
    assert env["status"] == "success", env
    assert env["result"]["compliant"] is compliant
    assert env["result"]["violation_count"] == (0 if compliant else 1)


@pytest.mark.parametrize("given, following, done", [
    ("raised", "ACKNOWLEDGED", False),
    ("Verifying", "CLOSED", False),
    ("CLOSED", None, True),
])
def test_ncr_next_status_through_dispatch(given, following, done):
    env = _calc("next_ncr_status", {"ncr_status": given})
    assert env["status"] == "success", env
    assert env["result"]["next_status"] == following
    assert env["result"]["workflow_complete"] is done


@pytest.mark.parametrize("given", ["banana", "OPENED"])
def test_an_unknown_ncr_status_is_an_error_not_a_blank_answer(given):
    env = _calc("next_ncr_status", {"ncr_status": given})
    assert env["status"] == "error"
    assert given in env["error"] and "RAISED" in env["error"]


@pytest.mark.parametrize("given, valid", [
    ("approved", False),
    ("for comment", True),
])
def test_design_status_through_dispatch(given, valid):
    env = _calc("validate_design_status", {"design_status": given})
    assert env["status"] == "success", env
    assert env["result"]["valid"] is valid


@pytest.mark.parametrize("dist, workshop, days, compliant", [
    ("2026-03-01", "2026-03-05", 4, False),
    ("2026-03-01", "2026-03-10", 9, True),
])
def test_review_timeline_through_dispatch(dist, workshop, days, compliant):
    env = _calc("check_review_timeline",
                {"distribution_date": dist, "workshop_date": workshop})
    assert env["status"] == "success", env
    assert env["result"]["distribution_period_days"] == days
    assert env["result"]["compliant"] is compliant


@pytest.mark.parametrize("dist, workshop", [
    ("March 1", "2026-03-10"),
    ("2026-03-01", "next week"),
])
def test_an_unreadable_date_is_an_error(dist, workshop):
    env = _calc("check_review_timeline",
                {"distribution_date": dist, "workshop_date": workshop})
    assert env["status"] == "error"
    assert "YYYY-MM-DD" in env["error"]


# -- the two hat aliases through the real dispatch -----------------------

@pytest.mark.parametrize("body, compliant", [
    ("Please approve the design package.", False),
    ("Site cleared and levels taken.", True),
])
def test_compliance_gate_checker_alias(body, compliant):
    out = _dispatch("compliance_gate_checker", {"params": {"document_text": body}})
    assert out["ok"], out
    assert out["result"]["aliased_to"] == "construction_calc:enforce_critical_rules"
    assert out["result"]["result"]["compliant"] is compliant


@pytest.mark.parametrize("given, following", [
    ("APPROVED", "IMPLEMENTING"),
    ("acknowledged", "DISPOSITION_PROPOSED"),
])
def test_ncr_tracker_alias(given, following):
    out = _dispatch("ncr_tracker", {"params": {"ncr_status": given}})
    assert out["ok"], out
    assert out["result"]["aliased_to"] == "construction_calc:next_ncr_status"
    assert out["result"]["result"]["next_status"] == following


# -- the rule functions themselves are unchanged -------------------------

def test_the_knowledge_functions_keep_their_own_contract():
    from app.core import construction_knowledge as ck
    assert ck.generate_doc_number("VO", 15) == "VO-015"
    assert ck.validate_design_status("APPROVED")[0] is False
    assert ck.next_ncr_status("CLOSED") is None
    assert isinstance(ck.enforce_critical_rules("approved design drawing"), list)


# -- the declarations' own input handling ---------------------------------

@pytest.mark.parametrize("given, following", [
    ("disposition proposed", "DISPOSITION_REVIEWED"),
    ("Disposition-Reviewed", "APPROVED"),
])
def test_an_ncr_status_written_with_spaces_or_hyphens_is_read(given, following):
    env = _calc("next_ncr_status", {"ncr_status": given})
    assert env["status"] == "success", env
    assert env["result"]["next_status"] == following


@pytest.mark.parametrize("dist, workshop, days", [
    (" 2026-05-01 ", "2026-05-09", 8),
    ("2026-05-01", " 2026-05-16 ", 15),
])
def test_review_dates_with_surrounding_spaces_are_read(dist, workshop, days):
    from app.agents.hats.qaqc.formulas.design_review_rules import check_review_timeline
    out = check_review_timeline(dist, workshop)
    assert out["distribution_period_days"] == days
    assert "error" not in out


@pytest.mark.parametrize("blank", ["   ", "\t"])
def test_a_blank_rule_input_is_an_error_never_a_verdict(blank):
    from app.agents.hats.contracts.formulas import document_rules as dr
    from app.agents.hats.qaqc.formulas import design_review_rules as qr
    for out in (dr.generate_doc_number(blank, 3), dr.enforce_critical_rules(blank),
                dr.next_ncr_status(blank), qr.validate_design_status(blank)):
        assert isinstance(out.get("error"), str) and out["error"], out
