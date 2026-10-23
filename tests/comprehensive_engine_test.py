"""
Comprehensive Engine Test Suite
Covers: All 12 activity types, 6 statuses, 4 risk levels, 25 dependency rules,
10 workflow templates, cross-domain reasoning, and learning engine.
Scoring: Each test group contributes to an overall coverage score.
"""
from __future__ import annotations
import sys, os
from datetime import date, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.schemas.construction_activity import (
    ActivityType, Status, RiskLevel, ConstructionActivity, ActivityGraph,
    ScheduleMeta, CostMeta, QualityGate, SafetyGate, ProcurementLink
)
from app.core.dependency_graph import (
    DependencyGraph, Domain, TriggerCondition, ProjectHealthAggregator, DEFAULT_RULES
)
from app.core.workflow_templates import WorkflowTemplateLibrary
from app.core.cross_domain_reasoner import CrossDomainReasoner
from app.core.construction_learning import (
    ConstructionLearningEngine, DurationCalibration,
    ProcurementLeadTimeLearner, DefectPatternDetector, TemplateEffectivenessTracker
)

RESULTS = {"passed": 0, "failed": 0, "groups": {}}

def test_group(name):
    def decorator(fn):
        def wrapper():
            gp, gf, errs = 0, 0, []
            try:
                fn(gp, gf, errs)
            except Exception as e:
                errs.append(f"GROUP EXCEPTION: {e}")
                gf += 1
            RESULTS["groups"][name] = {"passed": RESULTS["passed"] - sum(g["passed"] for g in RESULTS["groups"].values()), "failed": RESULTS["failed"] - sum(g["failed"] for g in RESULTS["groups"].values()), "errors": errs}
            print(f"  {name}: {RESULTS['groups'][name]['passed']} passed, {RESULTS['groups'][name]['failed']} failed")
            for err in errs:
                print(f"    ERROR: {err}")
        return wrapper
    return decorator

test_group.__test__ = False  # decorator factory, not a pytest test

def check(condition, label, errors_list):
    if condition:
        RESULTS["passed"] += 1
        return True
    else:
        RESULTS["failed"] += 1
        errors_list.append(label)
        return False

@test_group("Activity Types (13/13)")
def test_activity_types(gp, gf, errs):
    expected = {"schedule", "cost", "quality", "safety", "procurement", "contract", "risk",
                "resource", "commissioning", "handover", "document", "reporting", "bim"}
    check({t.value for t in ActivityType} == expected, f"ActivityType mismatch", errs)

@test_group("Status (6/6) + RiskLevel (4/4)")
def test_status_risk(gp, gf, errs):
    check({s.value for s in Status} == {"planned", "in_progress", "complete", "blocked", "overdue", "on_hold"}, "Status enum", errs)
    check({r.value for r in RiskLevel} == {"low", "medium", "high", "critical"}, "RiskLevel enum", errs)

@test_group("ConstructionActivity per Type (13 types)")
def test_all_activity_types(gp, gf, errs):
    for atype, name, extras in [
        (ActivityType.SCHEDULE, "Pour Concrete", {"schedule": {"duration_days": 5, "is_critical": True}}),
        (ActivityType.COST, "Cost Control", {"cost": {"budgeted": 10000, "currency": "USD"}}),
        (ActivityType.QUALITY, "Inspect Rebar", {"quality_gates": [{"gate_type": "inspection", "result": "pass"}]}),
        (ActivityType.SAFETY, "Safety Audit", {"safety_gates": [{"gate_type": "permit", "status": "planned"}]}),
        (ActivityType.PROCUREMENT, "Order Switchgear", {"procurement_links": [{"item_id": "P-001", "is_long_lead": True}]}),
        (ActivityType.CONTRACT, "Review FIDIC", {"contract_clauses": ["Clause 14"]}),
        (ActivityType.RISK, "Weather Risk", {"risk_level": RiskLevel.HIGH}),
        (ActivityType.RESOURCE, "Crew Assignment", {"responsible_party": "MEP"}),
        (ActivityType.COMMISSIONING, "HVAC Test", {"schedule": {"duration_days": 3}}),
        (ActivityType.HANDOVER, "PC Cert", {"documents": ["PC"]}),
        (ActivityType.DOCUMENT, "RFI-042", {"bim_element_guids": ["guid-123"]}),
        (ActivityType.REPORTING, "Monthly Report", {"meta": {"month": "2025-01"}}),
        (ActivityType.BIM, "Clash Detection", {"bim_element_guids": ["guid-456"]}),
    ]:
        a = ConstructionActivity(id=f"ACT-{atype.value.upper()}", name=name, type=atype, **extras)
        check(a.type == atype, f"Type {atype.value}", errs)

@test_group("Blocking + Overdue Logic")
def test_blocking_overdue(gp, gf, errs):
    check(ConstructionActivity(id="B1", quality_gates=[QualityGate(result="fail", status=Status.BLOCKED)]).is_blocked(), "Blocked by QA", errs)
    check(ConstructionActivity(id="B2", safety_gates=[SafetyGate(status=Status.BLOCKED)]).is_blocked(), "Blocked by safety", errs)
    check(not ConstructionActivity(id="B3", quality_gates=[QualityGate(result="pass")]).is_blocked(), "Not blocked", errs)
    past = date.today() - timedelta(days=5)
    check(ConstructionActivity(id="O1", status=Status.OVERDUE, schedule=ScheduleMeta(finish_date=past)).days_overdue() == 5, "Overdue 5d", errs)

@test_group("ActivityGraph Cross-Domain Linking")
def test_activity_graph(gp, gf, errs):
    g = ActivityGraph(project_id="P-001")
    for a in [
        ConstructionActivity(id="PROC-001", type=ActivityType.PROCUREMENT),
        ConstructionActivity(id="SCH-001", type=ActivityType.SCHEDULE, cost=CostMeta(budgeted=50000)),
        ConstructionActivity(id="COST-001", type=ActivityType.COST),
        ConstructionActivity(id="Q-001", type=ActivityType.QUALITY),
        ConstructionActivity(id="R-001", type=ActivityType.RISK),
        ConstructionActivity(id="SAF-001", type=ActivityType.SAFETY),
        ConstructionActivity(id="COM-001", type=ActivityType.COMMISSIONING),
        ConstructionActivity(id="H-001", type=ActivityType.HANDOVER),
        ConstructionActivity(id="B-001", type=ActivityType.BIM),
        ConstructionActivity(id="DOC-001", type=ActivityType.DOCUMENT),
        ConstructionActivity(id="RES-001", type=ActivityType.RESOURCE),
        ConstructionActivity(id="REP-001", type=ActivityType.REPORTING),
        ConstructionActivity(id="CON-001", type=ActivityType.CONTRACT),
    ]:
        g.add_activity(a)
    check(len(g.activities) == 13, "13 activities", errs)
    g.link_activities("PROC-001", "SCH-001")
    check("SCH-001" in g.get_by_id("PROC-001").dependents, "Proc->Schedule link", errs)
    s = g.summary()
    check(s["total_activities"] == 13, "Summary 13", errs)
    check(s["by_type"].get("contract") == 1, "Has contract", errs)

@test_group("25 Dependency Rules")
def test_dependency_rules(gp, gf, errs):
    check(len(DEFAULT_RULES) == 25, "25 rules", errs)
    check(len({r.rule_id for r in DEFAULT_RULES}) == 25, "Unique IDs", errs)
    check(set(Domain) == {r.source_domain for r in DEFAULT_RULES}, "All domains", errs)

@test_group("Trigger Analysis (all 12 domains)")
def test_trigger_analysis(gp, gf, errs):
    g = DependencyGraph()
    for domain in Domain:
        check(len(g.get_rules_for_source(domain)) >= 1, f"Rules for {domain.value}", errs)
    check(Domain.SCHEDULE in g.trigger_analysis(Domain.PROCUREMENT, TriggerCondition.DELAYED)[0], "Proc->Schedule", errs)
    check(Domain.SCHEDULE in g.trigger_analysis(Domain.QUALITY, TriggerCondition.FAILED)[0], "Quality->Schedule", errs)
    check(Domain.CONTRACT in g.trigger_analysis(Domain.SCHEDULE, TriggerCondition.DELAYED)[0], "Schedule->Contract", errs)
    check(Domain.DOCUMENT in g.trigger_analysis(Domain.BIM, TriggerCondition.DETECTED)[0], "BIM->Document", errs)
    check(Domain.HANDOVER in g.trigger_analysis(Domain.COMMISSIONING, TriggerCondition.COMPLETED)[0], "Comm->Handover", errs)

@test_group("Cascade Analysis")
def test_cascade(gp, gf, errs):
    g = DependencyGraph()
    results = g.cascade_analysis(Domain.PROCUREMENT, TriggerCondition.DELAYED, max_depth=3)
    check(Domain.SCHEDULE in {r[1] for r in results}, "Cascade reaches schedule", errs)

@test_group("Health Aggregator")
def test_health_aggregator(gp, gf, errs):
    g = DependencyGraph()
    agg = ProjectHealthAggregator(g)
    h = agg.aggregate({Domain.SCHEDULE: {"status": "healthy"}, Domain.COST: {"status": "healthy"}})
    check(h["overall_status"] == "healthy", "Healthy", errs)
    h = agg.aggregate({Domain.SCHEDULE: {"status": "delayed"}})
    check(len(h["cross_domain_flags"]) >= 1, "Flags for delayed", errs)
    h = agg.aggregate({Domain.SCHEDULE: {"status": "delayed"}, Domain.COST: {"status": "overrun"}, Domain.QUALITY: {"status": "failed"}, Domain.SAFETY: {"status": "critical"}})
    check(h["overall_status"] == "critical", "Critical multi-domain", errs)

@test_group("10 Workflow Templates")
def test_workflow_templates(gp, gf, errs):
    lib = WorkflowTemplateLibrary()
    check(set(lib.all_template_ids()) == {"new_project_setup", "monthly_progress", "change_order_impact", "delay_to_claim",
                "qa_defect_closeout", "submittal_to_install", "safety_incident_response",
                "commissioning_to_handover", "bim_coordination", "sustainability_assessment"}, "10 templates", errs)
    for tid in lib.all_template_ids():
        plan = lib.get(tid).build_plan({})
        check(len(plan.steps) > 0, f"{tid} has steps", errs)

@test_group("Cross-Domain Reasoner")
def test_cross_domain_reasoner(gp, gf, errs):
    r = CrossDomainReasoner()
    check(r.analyze_turn("set up new project for data center")["matched_template"] == "new_project_setup", "Match new project", errs)
    check(r.analyze_turn("monthly progress report")["matched_template"] == "monthly_progress", "Match monthly", errs)
    check(r.analyze_turn("delay claim for electrical")["matched_template"] == "delay_to_claim", "Match delay claim", errs)
    check(r.analyze_turn("safety incident in zone A")["matched_template"] == "safety_incident_response", "Match safety", errs)
    check(r.analyze_turn("commissioning and handover")["matched_template"] == "commissioning_to_handover", "Match comm", errs)
    check("cost" in r.analyze_turn("we are delayed by 2 weeks")["additional_domains"] or "contract" in r.analyze_turn("we are delayed by 2 weeks")["additional_domains"], "Delay->cost/contract", errs)
    check(r.analyze_turn("hello")["matched_template"] is None, "No match for hello", errs)
    plan = r.build_plan("delay_to_claim")
    check(plan is not None and len(plan["steps"]) >= 5, "Plan built", errs)
    check(len(r.get_post_tool_suggestions("generate_wbs", "cost?")) > 0, "Post-tool suggestions", errs)

@test_group("Learning: Duration Calibration")
def test_learning_duration(gp, gf, errs):
    cal = DurationCalibration()
    for i in range(5):
        cal.record_actual_duration("P-001", "concrete_pour", 5, 6 + i)
    r = cal.get_calibrated_duration("P-001", "concrete_pour", min_samples=3)
    check(r["source"] == "calibrated", "Calibrated", errs)
    check(r["samples"] == 5, "5 samples", errs)

@test_group("Learning: Procurement Lead Times")
def test_learning_procurement(gp, gf, errs):
    learner = ProcurementLeadTimeLearner()
    for i in range(8):
        learner.record_delivery("switchgear", f"SG-{i}", 90, 90 + i * 3)
    r = learner.get_lead_time_estimate("switchgear", min_samples=3)
    check(r["source"] == "learned", "Learned", errs)
    check(r["estimated_days"] >= r["median"], "P80 >= median", errs)

@test_group("Learning: Defect Patterns")
def test_learning_defects(gp, gf, errs):
    det = DefectPatternDetector()
    det.record_defect("concrete", "carpenter", "major")
    det.record_defect("concrete", "carpenter", "critical")
    for _ in range(5):
        det.record_inspection("concrete", "carpenter", passed=False)
    for _ in range(10):
        det.record_inspection("mep", "plumber", passed=True)
    rankings = det.get_defect_risk_ranking(min_samples=1)
    check(len(rankings) == 2, "2 rankings", errs)
    check(rankings[0]["activity_type"] == "concrete", "Concrete highest risk", errs)

@test_group("Learning: Template Effectiveness")
def test_learning_templates(gp, gf, errs):
    tracker = TemplateEffectivenessTracker()
    for _ in range(5):
        tracker.record_usage("new_project_setup", "data_center", success=True)
    for _ in range(3):
        tracker.record_usage("bim_coordination", "data_center", success=True)
    best = tracker.get_best_templates("data_center")
    check(len(best) == 2, "2 templates ranked", errs)
    check(best[0]["success_rate_pct"] == 100.0, "100% success", errs)

@test_group("End-to-End Scenarios")
def test_e2e(gp, gf, errs):
    reasoner = CrossDomainReasoner()
    lib = WorkflowTemplateLibrary()
    graph = ActivityGraph(project_id="E2E-001")
    learning = ConstructionLearningEngine()
    learning.reset()

    analysis = reasoner.analyze_turn("we are delayed by 3 weeks on electrical")
    check("cost" in analysis["additional_domains"] or "contract" in analysis["additional_domains"], "E2E: delay->cost/contract", errs)

    plan = reasoner.build_plan("delay_to_claim")
    check(plan["template_id"] == "delay_to_claim", "E2E: plan template", errs)
    check(len([s for s in plan["steps"] if s["step_type"] == "delay_analysis"]) > 0, "E2E: has delay_analysis", errs)

    for a in [
        ConstructionActivity(id="ELEC-001", type=ActivityType.SCHEDULE, schedule=ScheduleMeta(duration_days=14, is_critical=True)),
        ConstructionActivity(id="PROC-ELEC", type=ActivityType.PROCUREMENT, procurement_links=[ProcurementLink(item_id="E-001", is_long_lead=True)]),
        ConstructionActivity(id="QA-ELEC", type=ActivityType.QUALITY),
    ]:
        graph.add_activity(a)
    graph.link_activities("PROC-ELEC", "ELEC-001")
    s = graph.summary()
    check(s["total_activities"] == 3, "E2E: 3 activities", errs)

    for d in [(14, 21), (14, 18), (14, 20)]:
        learning.record_duration("E2E-001", "electrical", d[0], d[1])
    r = learning.get_calibrated_duration("E2E-001", "electrical", default_days=14, min_samples=2)
    check(r["source"] == "calibrated", "E2E: calibrated", errs)

    g_dep = DependencyGraph()
    health = ProjectHealthAggregator(g_dep).aggregate({
        Domain.SCHEDULE: {"status": "delayed"}, Domain.COST: {"status": "overrun"},
        Domain.QUALITY: {"status": "failed"}, Domain.SAFETY: {"status": "critical"},
        Domain.PROCUREMENT: {"status": "delayed"}, Domain.CONTRACT: {"status": "healthy"},
    })
    check(health["overall_status"] in ("warning", "critical"), "E2E: health status", errs)
    check(len(health["cross_domain_flags"]) >= 3, "E2E: flags", errs)

    for tid in lib.all_template_ids():
        plan = reasoner.build_plan(tid)
        check(plan is not None and len(plan["steps"]) > 0, f"E2E: {tid} plan", errs)

if __name__ == "__main__":
    print("=" * 70)
    print("  THE FORK - Construction Management Engine - Comprehensive Test Suite")
    print("=" * 70)
    print()

    test_activity_types()
    test_status_risk()
    test_all_activity_types()
    test_blocking_overdue()
    test_activity_graph()
    test_dependency_rules()
    test_trigger_analysis()
    test_cascade()
    test_health_aggregator()
    test_workflow_templates()
    test_cross_domain_reasoner()
    test_learning_duration()
    test_learning_procurement()
    test_learning_defects()
    test_learning_templates()
    test_e2e()

    total = RESULTS["passed"] + RESULTS["failed"]
    score = (RESULTS["passed"] / total * 100) if total > 0 else 0

    print()
    print(f"  Total Tests: {total} | Passed: {RESULTS['passed']} | Failed: {RESULTS['failed']} | Score: {score:.1f}%")
    print()
    for name, data in RESULTS["groups"].items():
        status = "PASS" if not data.get("errors") else "FAIL"
        print(f"  [{status}] {name}")

    print()
    print("=" * 70)
    if RESULTS["failed"] == 0:
        print(f"  ENGINE STATUS: FULLY OPERATIONAL - {score:.0f}/100")
    else:
        print(f"  ENGINE STATUS: {score:.0f}/100 - {RESULTS['failed']} failures")
    print("=" * 70)
    sys.exit(0 if RESULTS["failed"] == 0 else 1)
