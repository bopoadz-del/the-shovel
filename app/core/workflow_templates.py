"""Multi-Domain Workflow Templates — Phase 3 of the Construction Management Engine.

Predefined cross-domain plans that tie multiple construction management
domains together. Each template is a typed ExecutionPlan the orchestrator
can invoke. Templates are parameterized (project_type, scope, thresholds).

Usage:
    from app.core.workflow_templates import WorkflowTemplateLibrary
    lib = WorkflowTemplateLibrary()
    plan = lib.get("new_project_setup").build_plan(params={"project_type": "data_center"})
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from app.schemas.execution_plan import ExecutionPlan, PlanStep


# ── Typed step vocabulary (shared across all templates) ─────────────────────
# These are the step types PlanExecutor must support for multi-domain workflows.
# Each step type maps to a block or container action.

STEP_EXTRACT_DOCUMENT = "extract_document"          # document_engine
STEP_BUILD_WBS = "build_wbs"                        # ConstructionContainer.generate_wbs
STEP_EXTRACT_BOQ = "extract_boq"                    # boq_processor
STEP_COST_LOAD = "cost_load"                        # schedule_bridge + pm_excel
STEP_EXTRACT_RISKS = "extract_risks"                # document_engine risk extraction
STEP_POPULATE_RISK_REGISTER = "populate_risk_register"  # risk_register_auto_populate
STEP_UPDATE_RISK_REGISTER = "update_risk_register"   # risk register maintenance
STEP_EXTRACT_EQUIPMENT_LIST = "extract_equipment_list"  # document_engine
STEP_PROCUREMENT_PLAN = "procurement_plan"          # procurement_list_generator
STEP_EXTRACT_CLAUSES = "extract_clauses"            # contract analysis
STEP_OBLIGATION_REGISTER = "obligation_register"    # process_contract
STEP_PROGRESS_TRACKER = "progress_tracker"          # progress_tracker
STEP_FLOAT_ANALYSIS = "float_analysis"              # parse_primavera_schedule
STEP_RECOVERY_OPTIONS = "recovery_options"          # schedule risk analysis
STEP_EARNED_VALUE = "earned_value"                  # cost analysis
STEP_COST_VARIANCE = "cost_variance"                # sympy_reason
STEP_INSPECTION_SUMMARY = "inspection_summary"      # qa_qc_inspection
STEP_DEFECT_TRENDS = "defect_trends"                # quality trending
STEP_SAFETY_SUMMARY = "safety_summary"              # safety_compliance_audit
STEP_INCIDENT_SUMMARY = "incident_summary"          # safety incidents
STEP_PAYMENT_CERT = "payment_cert"                  # payment_certificate
STEP_CASH_FLOW_UPDATE = "cash_flow_update"          # cash_flow_forecast
STEP_TIME_IMPACT_ANALYSIS = "time_impact_analysis"  # forensic_delay_analysis
STEP_COST_IMPACT_QUANTIFICATION = "cost_impact"     # cost loading
STEP_ENTITLEMENT_CHECK = "entitlement_check"        # contract clause check
STEP_CLAUSE_REFERENCE = "clause_reference"          # process_contract
STEP_DELAY_ANALYSIS = "delay_analysis"              # forensic_delay_analysis
STEP_CRITICAL_PATH_IMPACT = "critical_path_impact"  # CPM analysis
STEP_PROLONGATION_COST = "prolongation_cost"        # claims_builder
STEP_EOT_ENTITLEMENT = "eot_entitlement"            # contract analysis
STEP_CLAIM_STRENGTH = "claim_strength"              # claims_builder
STEP_COUNTER_ARGUMENTS = "counter_arguments"        # claims_builder
STEP_DEFECT_REGISTER = "defect_register"            # qa_qc_inspection
STEP_SEVERITY_SCORING = "severity_scoring"          # quality scoring
STEP_HOLD_POINT = "hold_point"                      # schedule hold
STEP_REWORK_COST = "rework_cost"                    # cost estimation
STEP_NCR_ISSUE = "ncr_issue"                        # non-conformance
STEP_RECTIFICATION_TRACK = "rectification_track"    # defect tracking
STEP_CLOSE_OUT = "close_out"                        # handover close
STEP_SPEC_ANALYZE = "spec_analyze"                  # spec_analyzer
STEP_SUBMITTAL_LOG = "submittal_log"                # submittal_log_generator
STEP_APPROVAL_TRACKING = "approval_tracking"        # submittal tracking
STEP_PO_TRIGGER = "po_trigger"                      # procurement_optimizer
STEP_LINK_PROCUREMENT = "link_procurement"          # schedule lead_time injection
STEP_INCOMING_INSPECTION_TRIGGER = "incoming_inspection"  # qa_qc_inspection
STEP_INCIDENT_REPORT = "incident_report"            # safety reporting
STEP_ROOT_CAUSE = "root_cause"                      # safety analysis
STEP_STOP_WORK_ORDER = "stop_work_order"            # safety hold
STEP_MITIGATION_PLAN = "mitigation_plan"            # risk planning
STEP_LESSONS_LEARNED = "lessons_learned"            # knowledge capture
STEP_TEST_RESULTS = "test_results"                  # commissioning results
STEP_DEFICIENCY_TRACKING = "deficiency_tracking"    # commissioning issues
STEP_PC_CERT = "pc_cert"                            # practical completion
STEP_OM_MANUAL = "om_manual"                        # om_manual_generator
STEP_TRAINING_PLAN = "training_plan"                # training requirements
STEP_WARRANTY_REGISTER = "warranty_register"        # warranty_maintenance_schedule
STEP_BIM_ANALYSIS = "bim_analysis"                  # bim_analysis
STEP_CLASH_DETECTION = "clash_detection"            # bim_clash_detection
STEP_QTO_EXTRACTION = "qto_extraction"              # bim_extractor
STEP_WBS_FROM_BIM = "wbs_from_bim"                  # schedule from BIM
STEP_BOQ_RECONCILE = "boq_reconcile"                # BOQ vs model
STEP_COORDINATION_RISKS = "coordination_risks"      # BIM risk identification
STEP_RFI_GENERATION = "rfi_generation"              # rfi_generator
STEP_CARBON_CALC = "carbon_calc"                    # carbon_footprint_calculator
STEP_CARBON_COST_PREMIUM = "carbon_cost_premium"    # cost analysis
STEP_SUSTAINABLE_ALTERNATIVES = "sustainable_alternatives"  # procurement
STEP_REGULATORY_COMPLIANCE = "regulatory_compliance"  # risk
STEP_ESG_REPORT = "esg_report"                      # esg_sustainability_report
STEP_RENDER_ARTIFACT = "render_artifact"            # materialize output
STEP_DELIVER = "deliver"                            # final delivery step


class WorkflowTemplate:
    """One named multi-domain workflow template."""

    def __init__(
        self,
        template_id: str,
        name: str,
        description: str,
        domains: List[str],
        builder: Callable[[Dict[str, Any]], ExecutionPlan],
    ) -> None:
        self.template_id = template_id
        self.name = name
        self.description = description
        self.domains = domains
        self._builder = builder

    def build_plan(self, params: Optional[Dict[str, Any]] = None) -> ExecutionPlan:
        return self._builder(params or {})


# ── Template builders ───────────────────────────────────────────────────────

def _build_new_project_setup(params: Dict[str, Any]) -> ExecutionPlan:
    """Template A: New Project Setup."""
    project_type = params.get("project_type", "data_center")
    understanding = (
        f"Set up a new {project_type} project: extract scope from BOD/RFP, "
        "generate WBS, build cost estimate, populate risk register, "
        "and create procurement plan."
    )
    steps = [
        PlanStep(type=STEP_EXTRACT_DOCUMENT, description="Extract scope, milestones, lead times, risks from BOD/RFP"),
        PlanStep(type=STEP_BUILD_WBS, description=f"Generate WBS for {project_type} with extracted milestones and lead times"),
        PlanStep(type=STEP_EXTRACT_BOQ, description="Extract BOQ from documents for cost loading"),
        PlanStep(type=STEP_COST_LOAD, description="Build cost-loaded schedule from WBS + BOQ"),
        PlanStep(type=STEP_EXTRACT_RISKS, description="Extract risks from documents"),
        PlanStep(type=STEP_POPULATE_RISK_REGISTER, description="Populate risk register with extracted risks"),
        PlanStep(type=STEP_EXTRACT_EQUIPMENT_LIST, description="Extract equipment list with lead times"),
        PlanStep(type=STEP_PROCUREMENT_PLAN, description="Create procurement plan from equipment list"),
        PlanStep(type=STEP_EXTRACT_CLAUSES, description="Extract key contract clauses"),
        PlanStep(type=STEP_OBLIGATION_REGISTER, description="Build obligation register from contract clauses"),
        PlanStep(type=STEP_RENDER_ARTIFACT, description="Deliver project setup package (multi-sheet Excel)"),
    ]
    return ExecutionPlan(understanding=understanding, steps=steps)


def _build_monthly_progress(params: Dict[str, Any]) -> ExecutionPlan:
    """Template B: Monthly Progress Cycle."""
    understanding = (
        "Monthly progress review: compare actual vs planned progress, "
        "analyze schedule float, compute earned value, summarize quality and safety, "
        "and update payment/cash flow."
    )
    steps = [
        PlanStep(type=STEP_PROGRESS_TRACKER, description="Track actual vs planned progress"),
        PlanStep(type=STEP_FLOAT_ANALYSIS, description="Analyze schedule float and identify constraints"),
        PlanStep(type=STEP_RECOVERY_OPTIONS, description="Generate recovery options if behind schedule"),
        PlanStep(type=STEP_EARNED_VALUE, description="Compute earned value and SPI/CPI"),
        PlanStep(type=STEP_COST_VARIANCE, description="Calculate cost variance and forecast"),
        PlanStep(type=STEP_INSPECTION_SUMMARY, description="Summarize QA inspections and results"),
        PlanStep(type=STEP_DEFECT_TRENDS, description="Analyze defect trends over time"),
        PlanStep(type=STEP_SAFETY_SUMMARY, description="Summarize safety compliance status"),
        PlanStep(type=STEP_INCIDENT_SUMMARY, description="Summarize incidents and near-misses"),
        PlanStep(type=STEP_PAYMENT_CERT, description="Generate payment certificate for period"),
        PlanStep(type=STEP_CASH_FLOW_UPDATE, description="Update cash flow forecast"),
        PlanStep(type=STEP_RENDER_ARTIFACT, description="Deliver monthly progress report (dashboard + Excel)"),
    ]
    return ExecutionPlan(understanding=understanding, steps=steps)


def _build_change_order_impact(params: Dict[str, Any]) -> ExecutionPlan:
    """Template C: Change Order Impact."""
    understanding = (
        "Assess full impact of a variation order: time impact on schedule, "
        "cost quantification, contract entitlement check, and risk register update."
    )
    steps = [
        PlanStep(type=STEP_EXTRACT_DOCUMENT, description="Extract VO details and scope change"),
        PlanStep(type=STEP_TIME_IMPACT_ANALYSIS, description="Analyze schedule impact of scope change"),
        PlanStep(type=STEP_COST_IMPACT_QUANTIFICATION, description="Quantify cost impact of variation"),
        PlanStep(type=STEP_ENTITLEMENT_CHECK, description="Check contractual entitlement for VO"),
        PlanStep(type=STEP_CLAUSE_REFERENCE, description="Reference relevant contract clauses"),
        PlanStep(type=STEP_UPDATE_RISK_REGISTER, description="Update risk register with VO-related risks"),
        PlanStep(type=STEP_RENDER_ARTIFACT, description="Deliver change order impact report"),
    ]
    return ExecutionPlan(understanding=understanding, steps=steps)


def _build_delay_to_claim(params: Dict[str, Any]) -> ExecutionPlan:
    """Template D: Delay to Claim Pipeline."""
    understanding = (
        "Convert schedule delay into a formal claim: delay analysis, "
        "prolongation costs, EOT entitlement, and claim strength assessment."
    )
    steps = [
        PlanStep(type=STEP_DELAY_ANALYSIS, description="Analyze delays between current and baseline schedule"),
        PlanStep(type=STEP_CRITICAL_PATH_IMPACT, description="Assess critical path impact of delays"),
        PlanStep(type=STEP_PROLONGATION_COST, description="Calculate prolongation/preliminaries costs"),
        PlanStep(type=STEP_EOT_ENTITLEMENT, description="Check extension of time entitlement"),
        PlanStep(type=STEP_CLAIM_STRENGTH, description="Assess overall claim strength"),
        PlanStep(type=STEP_COUNTER_ARGUMENTS, description="Anticipate counter-arguments from employer"),
        PlanStep(type=STEP_RENDER_ARTIFACT, description="Deliver claim package (narrative + quantum + entitlement)"),
    ]
    return ExecutionPlan(understanding=understanding, steps=steps)


def _build_qa_defect_closeout(params: Dict[str, Any]) -> ExecutionPlan:
    """Template E: QA Defect to Close-Out."""
    understanding = (
        "Track defects from detection through rectification to close-out: "
        "defect register, schedule hold, rework cost, NCR, and final sign-off."
    )
    steps = [
        PlanStep(type=STEP_DEFECT_REGISTER, description="Register defects with severity classification"),
        PlanStep(type=STEP_SEVERITY_SCORING, description="Score defect severities and prioritize"),
        PlanStep(type=STEP_HOLD_POINT, description="Place schedule hold on affected activities"),
        PlanStep(type=STEP_REWORK_COST, description="Estimate rework cost for defects"),
        PlanStep(type=STEP_NCR_ISSUE, description="Issue Non-Conformance Reports for major/critical defects"),
        PlanStep(type=STEP_RECTIFICATION_TRACK, description="Track rectification progress"),
        PlanStep(type=STEP_CLOSE_OUT, description="Close out defects after verification"),
        PlanStep(type=STEP_RENDER_ARTIFACT, description="Deliver defect tracking report with cost + schedule impact"),
    ]
    return ExecutionPlan(understanding=understanding, steps=steps)


def _build_submittal_to_install(params: Dict[str, Any]) -> ExecutionPlan:
    """Template F: Submittal to Installation Chain."""
    understanding = (
        "Manage material submittals from specification through approval "
        "to procurement, schedule linking, and incoming inspection."
    )
    steps = [
        PlanStep(type=STEP_SPEC_ANALYZE, description="Analyze specifications for required submittals"),
        PlanStep(type=STEP_SUBMITTAL_LOG, description="Generate submittal log from specs"),
        PlanStep(type=STEP_APPROVAL_TRACKING, description="Track submittal approval status"),
        PlanStep(type=STEP_PO_TRIGGER, description="Trigger purchase order on submittal approval"),
        PlanStep(type=STEP_LINK_PROCUREMENT, description="Link procurement lead times to schedule activities"),
        PlanStep(type=STEP_INCOMING_INSPECTION_TRIGGER, description="Trigger incoming inspection on goods receipt"),
        PlanStep(type=STEP_RENDER_ARTIFACT, description="Deliver submittal tracker linked to schedule + procurement"),
    ]
    return ExecutionPlan(understanding=understanding, steps=steps)


def _build_safety_incident_response(params: Dict[str, Any]) -> ExecutionPlan:
    """Template G: Safety Incident Response."""
    understanding = (
        "Respond to safety incident: incident report, root cause analysis, "
        "stop-work order, risk update, and lessons learned."
    )
    steps = [
        PlanStep(type=STEP_INCIDENT_REPORT, description="Document incident details and immediate actions"),
        PlanStep(type=STEP_ROOT_CAUSE, description="Analyze root cause of incident"),
        PlanStep(type=STEP_STOP_WORK_ORDER, description="Issue stop-work order for affected zone"),
        PlanStep(type=STEP_HOLD_POINT, description="Place hold on related schedule activities"),
        PlanStep(type=STEP_MITIGATION_PLAN, description="Develop mitigation plan to prevent recurrence"),
        PlanStep(type=STEP_UPDATE_RISK_REGISTER, description="Update risk register with incident risk"),
        PlanStep(type=STEP_LESSONS_LEARNED, description="Capture lessons learned for procedure update"),
        PlanStep(type=STEP_RENDER_ARTIFACT, description="Deliver incident response package"),
    ]
    return ExecutionPlan(understanding=understanding, steps=steps)


def _build_commissioning_to_handover(params: Dict[str, Any]) -> ExecutionPlan:
    """Template H: Commissioning to Handover."""
    understanding = (
        "Drive commissioning through to practical completion: test results, "
        "deficiency tracking, PC certificate, O&M manuals, training, and warranty."
    )
    steps = [
        PlanStep(type=STEP_TEST_RESULTS, description="Record commissioning test results by system"),
        PlanStep(type=STEP_DEFICIENCY_TRACKING, description="Track deficiencies found during commissioning"),
        PlanStep(type=STEP_PC_CERT, description="Generate Practical Completion certificate"),
        PlanStep(type=STEP_OM_MANUAL, description="Compile O&M manuals for all systems"),
        PlanStep(type=STEP_TRAINING_PLAN, description="Plan operations team training"),
        PlanStep(type=STEP_WARRANTY_REGISTER, description="Generate warranty register with expiry dates"),
        PlanStep(type=STEP_RENDER_ARTIFACT, description="Deliver handover package (PC + O&M + warranty + training)"),
    ]
    return ExecutionPlan(understanding=understanding, steps=steps)


def _build_bim_coordination(params: Dict[str, Any]) -> ExecutionPlan:
    """Template I: BIM-Driven Coordination."""
    understanding = (
        "Use BIM model to drive coordination: clash detection, quantity extraction, "
        "WBS generation, BOQ reconciliation, and RFI creation."
    )
    steps = [
        PlanStep(type=STEP_BIM_ANALYSIS, description="Analyze IFC model for elements and properties"),
        PlanStep(type=STEP_CLASH_DETECTION, description="Run clash detection across disciplines"),
        PlanStep(type=STEP_QTO_EXTRACTION, description="Extract quantities from BIM model"),
        PlanStep(type=STEP_WBS_FROM_BIM, description="Generate WBS activities from BIM elements"),
        PlanStep(type=STEP_BOQ_RECONCILE, description="Reconcile model quantities with BOQ"),
        PlanStep(type=STEP_COORDINATION_RISKS, description="Identify coordination risks from clashes"),
        PlanStep(type=STEP_RFI_GENERATION, description="Generate RFIs for unresolved clashes/design issues"),
        PlanStep(type=STEP_RENDER_ARTIFACT, description="Deliver BIM coordination report"),
    ]
    return ExecutionPlan(understanding=understanding, steps=steps)


def _build_sustainability_assessment(params: Dict[str, Any]) -> ExecutionPlan:
    """Template J: Sustainability Assessment."""
    understanding = (
        "Assess project sustainability: carbon footprint, cost premium analysis, "
        "sustainable procurement alternatives, regulatory compliance, and ESG reporting."
    )
    steps = [
        PlanStep(type=STEP_CARBON_CALC, description="Calculate embodied carbon from materials and energy"),
        PlanStep(type=STEP_CARBON_COST_PREMIUM, description="Analyze carbon cost premium vs baseline"),
        PlanStep(type=STEP_SUSTAINABLE_ALTERNATIVES, description="Identify sustainable material alternatives"),
        PlanStep(type=STEP_REGULATORY_COMPLIANCE, description="Check regulatory compliance for sustainability"),
        PlanStep(type=STEP_ESG_REPORT, description="Generate ESG/sustainability report"),
        PlanStep(type=STEP_RENDER_ARTIFACT, description="Deliver sustainability assessment report"),
    ]
    return ExecutionPlan(understanding=understanding, steps=steps)


# ── Template registry ───────────────────────────────────────────────────────

class WorkflowTemplateLibrary:
    """Registry of all multi-domain workflow templates."""

    def __init__(self) -> None:
        self._templates: Dict[str, WorkflowTemplate] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        templates: List[WorkflowTemplate] = [
            WorkflowTemplate(
                "new_project_setup", "New Project Setup",
                "Extract scope from BOD/RFP and generate schedule + cost + risk + procurement",
                ["schedule", "cost", "risk", "procurement", "contract"],
                _build_new_project_setup,
            ),
            WorkflowTemplate(
                "monthly_progress", "Monthly Progress Cycle",
                "Unified monthly review: schedule + cost + quality + safety + payment",
                ["schedule", "cost", "quality", "safety", "contract"],
                _build_monthly_progress,
            ),
            WorkflowTemplate(
                "change_order_impact", "Change Order Impact",
                "Full VO impact: schedule + cost + contract + risk",
                ["schedule", "cost", "contract", "risk"],
                _build_change_order_impact,
            ),
            WorkflowTemplate(
                "delay_to_claim", "Delay to Claim Pipeline",
                "Convert schedule delay into formal claim package",
                ["schedule", "cost", "contract", "risk"],
                _build_delay_to_claim,
            ),
            WorkflowTemplate(
                "qa_defect_closeout", "QA Defect to Close-Out",
                "Track defects from detection through rectification to sign-off",
                ["quality", "schedule", "cost"],
                _build_qa_defect_closeout,
            ),
            WorkflowTemplate(
                "submittal_to_install", "Submittal to Installation Chain",
                "Manage submittals from spec through approval to procurement and inspection",
                ["document", "procurement", "schedule", "quality"],
                _build_submittal_to_install,
            ),
            WorkflowTemplate(
                "safety_incident_response", "Safety Incident Response",
                "Respond to safety incident: report, stop-work, risk update, lessons learned",
                ["safety", "schedule", "risk"],
                _build_safety_incident_response,
            ),
            WorkflowTemplate(
                "commissioning_to_handover", "Commissioning to Handover",
                "Drive commissioning through to PC, O&M, warranty, and training",
                ["commissioning", "handover", "quality"],
                _build_commissioning_to_handover,
            ),
            WorkflowTemplate(
                "bim_coordination", "BIM-Driven Coordination",
                "Use BIM for clashes, quantities, WBS, BOQ reconciliation, and RFIs",
                ["bim", "schedule", "cost", "document"],
                _build_bim_coordination,
            ),
            WorkflowTemplate(
                "sustainability_assessment", "Sustainability Assessment",
                "Carbon footprint, cost premium, sustainable alternatives, ESG report",
                ["cost", "procurement", "risk", "document"],
                _build_sustainability_assessment,
            ),
        ]
        for t in templates:
            self._templates[t.template_id] = t

    def get(self, template_id: str) -> Optional[WorkflowTemplate]:
        return self._templates.get(template_id)

    def list_templates(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": t.template_id,
                "name": t.name,
                "description": t.description,
                "domains": t.domains,
            }
            for t in self._templates.values()
        ]

    def find_by_domain(self, domain: str) -> List[WorkflowTemplate]:
        return [t for t in self._templates.values() if domain in t.domains]

    def all_template_ids(self) -> List[str]:
        return list(self._templates.keys())

    def __contains__(self, template_id: str) -> bool:
        return template_id in self._templates
