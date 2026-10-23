"""5-stage gates per KB entry for the Construction Knowledge Base MVP.

The five gates per entry:

1. Syntactic   — sympy.sympify parses formula expressions; workflow guards
                 parse via the safe guard parser without raising.
2. Dimensional — pint balances the units (or, for the thermal entry, we
                 explicitly check structure since 168 is a dimensioned
                 constant that pint cannot derive).
3. Physical    — worked examples reproduce hand-computed values.
4. Empirical   — out-of-band sanity (formula in a physically plausible
                 range; workflow rejects when guard preconditions absent).
5. Operational — every result carries provenance + credibility + warnings;
                 region/project/tier triggers append a "verify against
                 your project spec or applicable standards" warning.

A dedicated security test exercises the safe guard parser against
``ast.Call`` and ``ast.Lambda`` payloads.
"""
from __future__ import annotations

import math

import pint
import pytest
import sympy

from app.blocks import _knowledge as kb
from app.blocks._knowledge import GuardEvalError, _safe_guard_eval


# ---------------------------------------------------------------------------
# Entry A — thermal.equilibrium_time
# ---------------------------------------------------------------------------


def test_thermal_syntactic_sympify():
    entry = kb.get_rule("thermal.equilibrium_time")
    assert entry is not None
    # Must parse cleanly.
    sympy.sympify(entry["expression"])


def test_thermal_dimensional_structure():
    """The constant 168 carries hours; (X/1.5) is dimensionless because
    both numerator and denominator are meters. pint cannot derive the
    "hours" tag from the expression alone, so we assert the structural
    invariant: with X in meters and the denominator in meters, the
    ratio is dimensionless and the entry's declared output unit is hour.
    """
    ureg = pint.UnitRegistry()
    x = 1.2 * ureg.meter
    half_thickness_ref = 1.5 * ureg.meter
    ratio = (x / half_thickness_ref) ** 2
    assert ratio.dimensionless
    entry = kb.get_rule("thermal.equilibrium_time")
    assert entry["thresholds"]["unit"] == "hour"
    # Sanity: pint accepts "hour" as a real unit (catches typos).
    assert (1 * ureg(entry["thresholds"]["unit"])).to("second").magnitude == 3600


def test_thermal_physical_worked_example():
    out = kb.evaluate("thermal.equilibrium_time", X=1.2)
    assert out["result"] == pytest.approx(107.52, abs=1e-6)
    assert out["unit"] == "hour"


def test_thermal_empirical_range():
    out = kb.evaluate("thermal.equilibrium_time", X=5)
    # 168 * (5/1.5)**2 = 1866.666...
    assert out["result"] == pytest.approx(168 * (5 / 1.5) ** 2, abs=1e-6)
    assert 1.0 < out["result"] < 100_000.0


def test_thermal_operational_envelope():
    out = kb.evaluate("thermal.equilibrium_time", X=1.2)
    assert "provenance" in out
    # Provenance preserves the source citation. Entries are kept GENERAL (no
    # location/project tie — region_specific/project_specific false, project
    # null), so we assert the source citation exists rather than a project.
    assert out["provenance"].get("source")
    assert isinstance(out["credibility_tier"], int)
    assert isinstance(out["warnings"], list)
    # Tier 3 (site-experience) entries surface the verify-against-spec warning.
    assert any("verify against your project spec" in w for w in out["warnings"])


# ---------------------------------------------------------------------------
# Entry B — earthworks.swelling_factor
# ---------------------------------------------------------------------------


def test_earthworks_syntactic_sympify():
    entry = kb.get_rule("earthworks.swelling_factor")
    assert entry is not None
    sympy.sympify(entry["expression"])


def test_earthworks_dimensional_pint():
    """A and B both density (ton/meter**3); C dimensionless => result
    dimensionless. pint balances this fully."""
    ureg = pint.UnitRegistry()
    A = 2.18 * ureg("ton/meter**3")
    B = 1.6 * ureg("ton/meter**3")
    C = 96 * ureg.dimensionless
    result = (A * C) / B
    assert result.dimensionless


def test_earthworks_physical_worked_example():
    out = kb.evaluate("earthworks.swelling_factor", A=2.18, B=1.6, C=96)
    # Raw C=96 reproduces 130.8 (the unit-convention note in the entry's
    # statement explains why this does NOT equal the published 1.31).
    assert out["result"] == pytest.approx(130.8, abs=0.01)
    assert out["unit"] == "dimensionless"


def test_earthworks_empirical_decimal_convention():
    """With C=0.96 the formula reproduces the published 1.308 ~ 1.31."""
    out = kb.evaluate("earthworks.swelling_factor", A=2.18, B=1.6, C=0.96)
    assert out["result"] == pytest.approx(1.308, abs=0.01)


def test_earthworks_operational_envelope():
    out = kb.evaluate("earthworks.swelling_factor", A=2.18, B=1.6, C=96)
    assert "provenance" in out
    assert out["provenance"]["verified_against_standard"] == (
        "AASHTO T 180-93 (Modified Proctor)"
    )
    assert isinstance(out["credibility_tier"], int)
    assert isinstance(out["warnings"], list)
    # Tier 3 (site-experience) entries surface the verify-against-spec warning.
    assert any("verify against your project spec" in w for w in out["warnings"])


# ---------------------------------------------------------------------------
# Entry C — procurement.tender_lifecycle
# ---------------------------------------------------------------------------


def test_procurement_syntactic_guards_parse():
    entry = kb.get_rule("procurement.tender_lifecycle")
    assert entry is not None
    # Every guard must parse via the safe walker. We feed an empty
    # context — the walker still has to traverse the AST and resolve
    # the missing keys as None without raising.
    for tr in entry["transitions"]:
        _safe_guard_eval(tr["guard"], context={})


def test_procurement_dimensional_states_and_transitions():
    """Workflows have no units; the 'dimensional' analogue is structural
    integrity — every transition's from/to references a declared state."""
    entry = kb.get_rule("procurement.tender_lifecycle")
    states = set(entry["states"])
    for tr in entry["transitions"]:
        assert tr["from"] in states
        assert tr["to"] in states


def test_procurement_physical_allowed_transition():
    out = kb.validate_transition(
        "procurement.tender_lifecycle",
        "RAT_ISSUED",
        {"to": "RFP_ISSUED"},
        {"rat_number": "RAT-12345"},
    )
    assert out["allowed"] is True


def test_procurement_empirical_rejected_without_rat_number():
    out = kb.validate_transition(
        "procurement.tender_lifecycle",
        "RAT_ISSUED",
        {"to": "RFP_ISSUED"},
        {},  # no rat_number
    )
    assert out["allowed"] is False
    # The guard expression must be surfaced for diagnosis.
    assert out["guard"] == "context.rat_number is not None"


def test_procurement_operational_envelope():
    out = kb.validate_transition(
        "procurement.tender_lifecycle",
        "RAT_ISSUED",
        {"to": "RFP_ISSUED"},
        {"rat_number": "RAT-12345"},
    )
    assert "provenance" in out
    assert isinstance(out["credibility_tier"], int)
    # Tier 4 (controlled documents) entries do not auto-warn — the
    # workflow is generic enough to adopt as-is. The envelope still
    # carries the warnings list shape so callers don't have to branch.
    assert isinstance(out["warnings"], list)
    assert "missing_documents" in out
    assert "missing_approvals" in out


# ---------------------------------------------------------------------------
# Cross-cutting loader behaviour
# ---------------------------------------------------------------------------


def test_load_knowledge_filters_by_domain():
    buildings = kb.load_knowledge("construction.buildings")
    roads = kb.load_knowledge("construction.roads")
    procurement = kb.load_knowledge("construction.procurement")
    assert any(e["id"] == "thermal.equilibrium_time" for e in buildings)
    assert any(e["id"] == "earthworks.swelling_factor" for e in roads)
    assert any(e["id"] == "procurement.tender_lifecycle" for e in procurement)
    # No cross-contamination.
    assert not any(e["id"] == "thermal.equilibrium_time" for e in roads)


def test_load_knowledge_no_filter_returns_all():
    all_entries = kb.load_knowledge()
    ids = {e["id"] for e in all_entries}
    # The full general corpus (buildings + roads/earthworks + concrete +
    # procurement). The three pilot entries must remain; the corpus is large.
    assert {
        "thermal.equilibrium_time",
        "earthworks.swelling_factor",
        "procurement.tender_lifecycle",
    } <= ids
    assert len(ids) >= 40, f"expected the full general corpus, got {len(ids)}"


def test_every_entry_is_general_and_valid():
    """No entry may be tied to a location/project, and every formula must
    sympy-parse, every workflow guard must be safe-evaluable."""
    import sympy
    for e in kb.load_knowledge():
        ap = e.get("applicability", {})
        assert ap.get("region_specific") in (False, None), e["id"]
        assert ap.get("project_specific") in (False, None), e["id"]
        assert not (e.get("provenance") or {}).get("project"), f"{e['id']} ties to a project"
        if e.get("type") == "formula" and e.get("expression"):
            sympy.sympify(e["expression"])  # raises on bad expression


def test_new_corpus_worked_examples_reproduce():
    """Spot-check the load-bearing formulas across domains evaluate correctly."""
    assert kb.evaluate("foundation.uplift_fos", counterweight=34, uplift=100)["result"] == pytest.approx(0.34, abs=1e-6)
    assert kb.evaluate("ground_improvement.dynamic_compaction", M=20, H=20)["result"] == pytest.approx(10.0, abs=1e-6)
    assert kb.evaluate("earthworks.compacted_material", E_loose=18, D=1.35)["result"] == pytest.approx(13.333, abs=1e-2)
    assert kb.evaluate("compaction.field_density", field_dry_density=2.104, lab_mdd=2.131)["result"] == pytest.approx(98.73, abs=1e-1)
    assert kb.evaluate("roads.heavy_lift_feasibility", beam_wt=66.6, rigging_factor=1.1, n_cranes=2)["result"] == pytest.approx(36.63, abs=1e-2)


def test_new_procurement_workflows_enforce_key_rules():
    """Payment Advice blocked without a passed checklist; VSR blocked without
    an attached back-up sheet."""
    pay_block = kb.validate_transition("procurement.interim_payment_flow", "CHECKLIST_REVIEW",
                                       {"to": "PAYMENT_ADVICE_ISSUED"}, {"payment_certificate_checklist_passed": False})
    assert pay_block["allowed"] is False
    pay_ok = kb.validate_transition("procurement.interim_payment_flow", "CHECKLIST_REVIEW",
                                    {"to": "PAYMENT_ADVICE_ISSUED"}, {"payment_certificate_checklist_passed": True})
    assert pay_ok["allowed"] is True
    vsr_block = kb.validate_transition("procurement.change_management", "BACKUP_SHEET",
                                       {"to": "VSR"}, {"back_up_sheet_attached": False, "cost_quality_schedule_assessed": False})
    assert vsr_block["allowed"] is False


def test_search_knowledge_retrieves_rules_by_nl_query():
    """Free-text query → ranked KB rule ids by token overlap on id+title+statement."""
    def ids(q):
        return [e["id"] for e in kb.search_knowledge(q, top_k=4)]
    assert "thermal.equilibrium_time" in ids("mass concrete equilibrium time for a thick pour")
    assert "compaction.acceptance" in ids("subgrade compaction acceptance criteria")
    assert "asphalt.bitumen_content" in ids("bitumen percentage for the wearing course")
    assert "dewatering.method_selection" in ids("dewatering well point selection")
    assert "roads.heavy_lift_feasibility" in ids("crane heavy lift feasibility for a precast beam")
    assert kb.search_knowledge("") == []
    # domain filter restricts to entries that apply to that namespace
    res = kb.search_knowledge("concrete", top_k=8, domain="construction.concrete")
    assert res and all(
        "construction.concrete" in (e.get("applicability", {}).get("applies_to") or [])
        for e in res
    )


def test_construction_advisor_block_answers_with_cited_rules():
    import asyncio
    from app.blocks.construction_advisor import ConstructionAdvisorBlock
    b = ConstructionAdvisorBlock()
    out = asyncio.run(b.process("what is the mass concrete equilibrium time for a thick pour"))
    assert out["status"] == "success" and out["count"] >= 1
    assert "thermal.equilibrium_time" in [m["id"] for m in out["matches"]]
    # every answer is cited (provenance + tier present)
    assert all("provenance" in m and "credibility_tier" in m for m in out["matches"])
    # evaluates the top formula when values are supplied
    out2 = asyncio.run(b.process({"query": "mass concrete equilibrium time", "values": {"X": 1.2}}))
    assert out2["evaluation"]["result"] == pytest.approx(107.52, abs=1e-6)
    # empty query handled
    assert asyncio.run(b.process(""))["status"] == "error"


def test_orchestrator_routes_kb_queries_to_advisor():
    import asyncio
    from app.blocks.smart_orchestrator import SmartOrchestratorBlock
    from app.core.action_router import best_action
    orch = SmartOrchestratorBlock()
    for q in ["what is the mass concrete equilibrium time",
              "subgrade compaction acceptance and field density",
              "dewatering well point selection"]:
        res = asyncio.run(orch.process({"user_message": q}))
        action, conf = best_action(res)
        assert action == "construction_advisor", f"{q!r} routed to {action}"


def test_evaluate_rejects_workflow():
    with pytest.raises(ValueError):
        kb.evaluate("procurement.tender_lifecycle")


def test_validate_transition_rejects_formula():
    with pytest.raises(ValueError):
        kb.validate_transition(
            "thermal.equilibrium_time",
            "ANY",
            {"to": "OTHER"},
            {},
        )


# ---------------------------------------------------------------------------
# Security tests for the safe guard parser
# ---------------------------------------------------------------------------


def test_guard_rejects_import_payload():
    """An attempted module import must NOT execute and must raise."""
    with pytest.raises(GuardEvalError):
        _safe_guard_eval("__import__('os').system('rm -rf /')", context={})


def test_guard_rejects_ast_call():
    with pytest.raises(GuardEvalError):
        _safe_guard_eval("len(context)", context={"a": 1})


def test_guard_rejects_ast_lambda():
    with pytest.raises(GuardEvalError):
        _safe_guard_eval("(lambda: 1)()", context={})


def test_guard_rejects_arbitrary_name():
    with pytest.raises(GuardEvalError):
        _safe_guard_eval("os.system('x')", context={})


def test_guard_rejects_arithmetic_binop():
    """BinOp is not in the allowlist — even harmless arithmetic is out
    of scope for transition guards."""
    with pytest.raises(GuardEvalError):
        _safe_guard_eval("context.x + 1 == 2", context={"x": 1})


def test_guard_accepts_simple_literal_true():
    assert _safe_guard_eval("True", context={}) is True


def test_guard_accepts_context_attribute_missing_key_is_none():
    """context.foo on missing key returns None (falsy) rather than raising."""
    assert _safe_guard_eval("context.missing is None", context={}) is True


def test_guard_accepts_boolean_combination():
    ctx = {"a": True, "b": False}
    assert _safe_guard_eval("context.a and not context.b", ctx) is True
    assert _safe_guard_eval("context.a or context.b", ctx) is True


def test_guard_accepts_subscript_on_context():
    assert _safe_guard_eval("context['x'] == 1", context={"x": 1}) is True
