"""Cross-Domain Dependency Graph — Phase 2 of the Construction Management Engine.

Maps real-world relationships between construction management domains.
When an activity changes in one domain, the dependency graph identifies
which other domains are affected and what actions should be considered.

These are NOT hardcoded workflows — they are dependency rules the
orchestrator uses to chain activities dynamically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class Domain(str, Enum):
    """Construction management domains."""
    SCHEDULE = "schedule"
    COST = "cost"
    QUALITY = "quality"
    SAFETY = "safety"
    PROCUREMENT = "procurement"
    CONTRACT = "contract"
    RISK = "risk"
    BIM = "bim"
    COMMISSIONING = "commissioning"
    HANDOVER = "handover"
    RESOURCE = "resource"
    DOCUMENT = "document"


class TriggerCondition(str, Enum):
    """What condition on the source activity triggers the rule."""
    DELAYED = "delayed"  # activity is behind schedule
    FAILED = "failed"  # inspection/test failed
    CRITICAL_INCIDENT = "critical_incident"  # safety incident
    APPROVED = "approved"  # document/VO/payment approved
    ISSUED = "issued"  # RFI, NCR, submittal issued
    RECEIVED = "received"  # goods, responses, payments received
    DETECTED = "detected"  # clash, defect, risk detected
    COMPLETED = "completed"  # activity finished
    VARIANCE_THRESHOLD = "variance_threshold"  # cost/schedule variance exceeded
    MILESTONE_REACHED = "milestone_reached"
    PLANNED = "planned"  # activity planned (e.g., handover planning)


class SuggestedAction(str, Enum):
    """Actions the orchestrator may suggest based on dependency rules."""
    UPDATE_SCHEDULE = "update_schedule"
    CHECK_EOT_ENTITLEMENT = "check_eot_entitlement"
    CALCULATE_PROLONGATION_COST = "calculate_prolongation_cost"
    ISSUE_NCR = "issue_ncr"
    HOLD_WORK = "hold_work"
    STOP_WORK = "stop_work"
    UPDATE_RISK_REGISTER = "update_risk_register"
    GENERATE_RFI = "generate_rfi"
    UPDATE_DRAWINGS = "update_drawings"
    CREATE_SUBMITTAL = "create_submittal"
    ISSUE_PURCHASE_ORDER = "issue_purchase_order"
    INSPECT_GOODS = "inspect_goods"
    RELEASE_HOLD_POINT = "release_hold_point"
    GENERATE_PAYMENT_CERT = "generate_payment_cert"
    UPDATE_CASH_FLOW = "update_cash_flow"
    GENERATE_RECOVERY_OPTIONS = "generate_recovery_options"
    UPDATE_BUDGET = "update_budget"
    ESCALATE_RISK = "escalate_risk"
    CREATE_CLAIM = "create_claim"
    UPDATE_COMMISSIONING_PLAN = "update_commissioning_plan"
    GENERATE_PC_CERT = "generate_pc_cert"
    GENERATE_OM_MANUAL = "generate_om_manual"
    CONDUCT_TRAINING = "conduct_training"
    ACTIVATE_WARRANTY = "activate_warranty"
    NONE = "none"


@dataclass
class DependencyRule:
    """One cross-domain dependency rule.

    When an activity in `source_domain` meets `trigger`, activities in
    `target_domains` may be affected, and `suggested_actions` should be
    considered by the orchestrator.
    """
    rule_id: str
    name: str
    source_domain: Domain
    trigger: TriggerCondition
    target_domains: List[Domain] = field(default_factory=list)
    suggested_actions: List[SuggestedAction] = field(default_factory=list)
    description: str = ""
    auto_propagate: bool = False  # if True, auto-execute; else suggest only


# ── The 25 cross-domain dependency rules ───────────────────────────────────

DEFAULT_RULES: List[DependencyRule] = [
    # Procurement → Schedule
    DependencyRule(
        rule_id="R001", name="Procurement delivery delays schedule",
        source_domain=Domain.PROCUREMENT, trigger=TriggerCondition.DELAYED,
        target_domains=[Domain.SCHEDULE, Domain.COST],
        suggested_actions=[
            SuggestedAction.UPDATE_SCHEDULE,
            SuggestedAction.GENERATE_RECOVERY_OPTIONS,
        ],
        description=(
            "Equipment/material delivery delay pushes dependent schedule "
            "activities. Update the programme and recovery options — do not "
            "assume prolongation or a claim."
        ),
    ),
    # Quality → Schedule
    DependencyRule(
        rule_id="R002", name="Failed inspection blocks schedule",
        source_domain=Domain.QUALITY, trigger=TriggerCondition.FAILED,
        target_domains=[Domain.SCHEDULE, Domain.COST],
        suggested_actions=[
            SuggestedAction.HOLD_WORK,
            SuggestedAction.ISSUE_NCR,
            SuggestedAction.UPDATE_SCHEDULE,
            SuggestedAction.CALCULATE_PROLONGATION_COST,
        ],
        description="Failed QA inspection creates a hold point; schedule activities blocked until rectified.",
    ),
    # Safety → Schedule
    DependencyRule(
        rule_id="R003", name="Critical safety incident stops work",
        source_domain=Domain.SAFETY, trigger=TriggerCondition.CRITICAL_INCIDENT,
        target_domains=[Domain.SCHEDULE, Domain.RISK],
        suggested_actions=[
            SuggestedAction.STOP_WORK,
            SuggestedAction.UPDATE_RISK_REGISTER,
            SuggestedAction.UPDATE_SCHEDULE,
        ],
        description="Critical safety incident triggers stop-work order in affected zone.",
    ),
    # Schedule → Contract
    DependencyRule(
        rule_id="R004", name="Schedule delay needs entitlement split",
        source_domain=Domain.SCHEDULE, trigger=TriggerCondition.DELAYED,
        target_domains=[Domain.CONTRACT, Domain.COST, Domain.RISK],
        suggested_actions=[
            SuggestedAction.UPDATE_SCHEDULE,
            SuggestedAction.UPDATE_RISK_REGISTER,
        ],
        description=(
            "Schedule delay noted — do not assume EOT or cost entitlement. "
            "Split into EOT-only / cost-only / concurrent / culpable, or ask "
            "when ownership of the risk is unclear. Do not recommend a claim "
            "by default."
        ),
    ),
    # Contract → Cost
    DependencyRule(
        rule_id="R005", name="Variation order updates budget",
        source_domain=Domain.CONTRACT, trigger=TriggerCondition.APPROVED,
        target_domains=[Domain.COST, Domain.SCHEDULE],
        suggested_actions=[
            SuggestedAction.UPDATE_BUDGET,
            SuggestedAction.UPDATE_SCHEDULE,
        ],
        description="Approved variation order changes budget and potentially schedule scope.",
    ),
    # Cost → Risk
    DependencyRule(
        rule_id="R006", name="Budget variance escalates risk",
        source_domain=Domain.COST, trigger=TriggerCondition.VARIANCE_THRESHOLD,
        target_domains=[Domain.RISK, Domain.CONTRACT],
        suggested_actions=[
            SuggestedAction.ESCALATE_RISK,
            SuggestedAction.UPDATE_BUDGET,
        ],
        description="Cost overrun exceeds threshold — escalate to risk register.",
    ),
    # BIM → RFI (Document)
    DependencyRule(
        rule_id="R007", name="BIM clash triggers RFI",
        source_domain=Domain.BIM, trigger=TriggerCondition.DETECTED,
        target_domains=[Domain.DOCUMENT, Domain.SCHEDULE],
        suggested_actions=[
            SuggestedAction.GENERATE_RFI,
            SuggestedAction.UPDATE_SCHEDULE,
        ],
        description=(
            "MEP clash detected in BIM model — draft and post an RFI to the "
            "CDE (Aconex) and assess time and cost impact. Do not allocate a "
            "local Fork RFI number; the CDE is the register."
        ),
    ),
    # RFI → Design → Submittal
    DependencyRule(
        rule_id="R008", name="RFI response enables design update and submittal",
        source_domain=Domain.DOCUMENT, trigger=TriggerCondition.RECEIVED,
        target_domains=[Domain.DOCUMENT, Domain.QUALITY],
        suggested_actions=[
            SuggestedAction.UPDATE_DRAWINGS,
            SuggestedAction.CREATE_SUBMITTAL,
        ],
        description="RFI answer received — update drawings and trigger material submittal.",
    ),
    # Submittal → Procurement
    DependencyRule(
        rule_id="R009", name="Submittal approval triggers procurement",
        source_domain=Domain.DOCUMENT, trigger=TriggerCondition.APPROVED,
        target_domains=[Domain.PROCUREMENT],
        suggested_actions=[SuggestedAction.ISSUE_PURCHASE_ORDER],
        description="Material submittal approved — issue purchase order.",
    ),
    # Procurement → Quality
    DependencyRule(
        rule_id="R010", name="Goods received trigger inspection",
        source_domain=Domain.PROCUREMENT, trigger=TriggerCondition.RECEIVED,
        target_domains=[Domain.QUALITY],
        suggested_actions=[SuggestedAction.INSPECT_GOODS],
        description="Materials delivered to site — incoming inspection required before use.",
    ),
    # Quality → Schedule (release)
    DependencyRule(
        rule_id="R011", name="Inspection passed releases hold point",
        source_domain=Domain.QUALITY, trigger=TriggerCondition.APPROVED,
        target_domains=[Domain.SCHEDULE],
        suggested_actions=[SuggestedAction.RELEASE_HOLD_POINT],
        description="QA inspection passed — release schedule hold point, work may resume.",
    ),
    # Schedule → Payment
    DependencyRule(
        rule_id="R012", name="Milestone reached triggers payment",
        source_domain=Domain.SCHEDULE, trigger=TriggerCondition.MILESTONE_REACHED,
        target_domains=[Domain.COST],
        suggested_actions=[
            SuggestedAction.GENERATE_PAYMENT_CERT,
            SuggestedAction.UPDATE_CASH_FLOW,
        ],
        description="Schedule milestone achieved — generate payment certificate and update cash flow.",
    ),
    # Payment → Cash Flow
    DependencyRule(
        rule_id="R013", name="Payment approved records income",
        source_domain=Domain.COST, trigger=TriggerCondition.APPROVED,
        target_domains=[Domain.COST],
        suggested_actions=[SuggestedAction.UPDATE_CASH_FLOW],
        description="Payment certificate approved — record income in cash flow forecast.",
    ),
    # Commissioning → Handover
    DependencyRule(
        rule_id="R014", name="Commissioning tests enable handover",
        source_domain=Domain.COMMISSIONING, trigger=TriggerCondition.COMPLETED,
        target_domains=[Domain.HANDOVER, Domain.QUALITY],
        suggested_actions=[
            SuggestedAction.GENERATE_PC_CERT,
            SuggestedAction.GENERATE_OM_MANUAL,
        ],
        description="Commissioning tests complete — generate Practical Completion certificate and O&M manuals.",
    ),
    # Handover → Warranty
    DependencyRule(
        rule_id="R015", name="DLP end activates warranty",
        source_domain=Domain.HANDOVER, trigger=TriggerCondition.COMPLETED,
        target_domains=[Domain.HANDOVER],
        suggested_actions=[SuggestedAction.ACTIVATE_WARRANTY],
        description="Defects Liability Period ended — activate warranty period.",
    ),
    # Schedule → Resource
    DependencyRule(
        rule_id="R016", name="Schedule peak drives resource demand",
        source_domain=Domain.SCHEDULE, trigger=TriggerCondition.VARIANCE_THRESHOLD,
        target_domains=[Domain.RESOURCE],
        suggested_actions=[SuggestedAction.GENERATE_RECOVERY_OPTIONS],
        description="Schedule shows resource peak — check resource availability and leveling options.",
    ),
    # Risk → Schedule
    DependencyRule(
        rule_id="R017", name="Risk event materializes as schedule delay",
        source_domain=Domain.RISK, trigger=TriggerCondition.DETECTED,
        target_domains=[Domain.SCHEDULE, Domain.COST],
        suggested_actions=[
            SuggestedAction.UPDATE_SCHEDULE,
            SuggestedAction.UPDATE_RISK_REGISTER,
        ],
        description="Risk event triggers — update schedule with contingency and risk register.",
    ),
    # Resource → Cost
    DependencyRule(
        rule_id="R018", name="Resource overrun runs budget",
        source_domain=Domain.RESOURCE, trigger=TriggerCondition.VARIANCE_THRESHOLD,
        target_domains=[Domain.COST],
        suggested_actions=[SuggestedAction.UPDATE_BUDGET],
        description="Resource overruns (overtime, additional crew) — update cost forecast.",
    ),
    # Quality → Risk
    DependencyRule(
        rule_id="R019", name="Repeated defects escalate risk",
        source_domain=Domain.QUALITY, trigger=TriggerCondition.VARIANCE_THRESHOLD,
        target_domains=[Domain.RISK],
        suggested_actions=[SuggestedAction.ESCALATE_RISK],
        description="Defect rate exceeds threshold — escalate as project risk.",
    ),
    # Contract → Risk
    DependencyRule(
        rule_id="R020", name="Contract dispute escalates risk",
        source_domain=Domain.CONTRACT, trigger=TriggerCondition.ISSUED,
        target_domains=[Domain.RISK],
        suggested_actions=[SuggestedAction.UPDATE_RISK_REGISTER],
        description="Contract dispute or claim issued — update risk register.",
    ),
    # Safety → Quality
    DependencyRule(
        rule_id="R021", name="Safety incident triggers quality review",
        source_domain=Domain.SAFETY, trigger=TriggerCondition.CRITICAL_INCIDENT,
        target_domains=[Domain.QUALITY],
        suggested_actions=[SuggestedAction.HOLD_WORK],
        description="Safety incident may indicate quality issue — review related work.",
    ),
    # Procurement → Risk
    DependencyRule(
        rule_id="R022", name="Procurement delay is a risk event",
        source_domain=Domain.PROCUREMENT, trigger=TriggerCondition.DELAYED,
        target_domains=[Domain.RISK],
        suggested_actions=[SuggestedAction.UPDATE_RISK_REGISTER],
        description="Long-lead procurement item delayed — register as active risk.",
    ),
    # Commissioning → Schedule
    DependencyRule(
        rule_id="R023", name="Commissioning failure delays handover",
        source_domain=Domain.COMMISSIONING, trigger=TriggerCondition.FAILED,
        target_domains=[Domain.SCHEDULE, Domain.HANDOVER],
        suggested_actions=[
            SuggestedAction.UPDATE_SCHEDULE,
            SuggestedAction.UPDATE_COMMISSIONING_PLAN,
        ],
        description="Commissioning test failed — delay handover, update commissioning plan.",
    ),
    # Handover → Resource
    DependencyRule(
        rule_id="R024", name="Handover requires training resources",
        source_domain=Domain.HANDOVER, trigger=TriggerCondition.PLANNED,
        target_domains=[Domain.RESOURCE],
        suggested_actions=[SuggestedAction.CONDUCT_TRAINING],
        description="Handover planned — allocate training resources for operations team.",
    ),
    # Document → Schedule (design approval)
    DependencyRule(
        rule_id="R025", name="Design approval enables construction start",
        source_domain=Domain.DOCUMENT, trigger=TriggerCondition.APPROVED,
        target_domains=[Domain.SCHEDULE],
        suggested_actions=[SuggestedAction.RELEASE_HOLD_POINT],
        description="Design documents approved — release construction start hold.",
    ),
]


class DependencyGraph:
    """Traverses dependency rules to find cross-domain impacts."""

    def __init__(self, rules: Optional[List[DependencyRule]] = None) -> None:
        self.rules = list(DEFAULT_RULES) if rules is None else rules
        self._by_source: Dict[Domain, List[DependencyRule]] = {}
        self._by_target: Dict[Domain, List[DependencyRule]] = {}
        self._index()

    def _index(self) -> None:
        for rule in self.rules:
            self._by_source.setdefault(rule.source_domain, []).append(rule)
            for t in rule.target_domains:
                self._by_target.setdefault(t, []).append(rule)

    def get_rules_for_source(
        self, domain: Domain, trigger: Optional[TriggerCondition] = None
    ) -> List[DependencyRule]:
        rules = self._by_source.get(domain, [])
        if trigger:
            rules = [r for r in rules if r.trigger == trigger]
        return rules

    def get_rules_for_target(self, domain: Domain) -> List[DependencyRule]:
        return self._by_target.get(domain, [])

    def trigger_analysis(
        self,
        domain: Domain,
        trigger: TriggerCondition,
    ) -> Tuple[Set[Domain], List[SuggestedAction]]:
        """Given a domain event, find all affected domains and suggested actions.

        Returns:
            affected_domains: set of domains that may be impacted
            suggested_actions: list of actions the orchestrator should consider
        """
        rules = self.get_rules_for_source(domain, trigger)
        affected: Set[Domain] = set()
        actions: List[SuggestedAction] = []
        for rule in rules:
            affected.update(rule.target_domains)
            actions.extend(rule.suggested_actions)
        return affected, actions

    def cascade_analysis(
        self,
        domain: Domain,
        trigger: TriggerCondition,
        max_depth: int = 3,
    ) -> List[Tuple[int, Domain, List[SuggestedAction]]]:
        """Multi-hop cascade: find indirect impacts up to max_depth.

        Example: PROCUREMENT(delayed) -> SCHEDULE(delayed) -> CONTRACT(claim check)
        """
        results: List[Tuple[int, Domain, List[SuggestedAction]]] = []
        seen: Set[Domain] = {domain}
        queue: List[Tuple[int, Domain, TriggerCondition]] = [(0, domain, trigger)]

        while queue:
            depth, current_domain, current_trigger = queue.pop(0)
            if depth >= max_depth:
                continue
            affected, actions = self.trigger_analysis(current_domain, current_trigger)
            for aff_domain in affected:
                if aff_domain not in seen:
                    seen.add(aff_domain)
                    # Map domain to likely follow-on trigger
                    follow_trigger = _domain_to_trigger(aff_domain)
                    results.append((depth + 1, aff_domain, actions))
                    queue.append((depth + 1, aff_domain, follow_trigger))
        return results

    def full_project_health_rules(self) -> Dict[str, List[str]]:
        """Return all rules organized by source domain for dashboard use."""
        health: Dict[str, List[str]] = {}
        for rule in self.rules:
            key = f"{rule.source_domain.value}.{rule.trigger.value}"
            health.setdefault(key, []).append(rule.name)
        return health


def _domain_to_trigger(domain: Domain) -> TriggerCondition:
    """Heuristic: what trigger is most likely to propagate from a domain."""
    mapping = {
        Domain.SCHEDULE: TriggerCondition.DELAYED,
        Domain.COST: TriggerCondition.VARIANCE_THRESHOLD,
        Domain.QUALITY: TriggerCondition.FAILED,
        Domain.SAFETY: TriggerCondition.CRITICAL_INCIDENT,
        Domain.PROCUREMENT: TriggerCondition.DELAYED,
        Domain.CONTRACT: TriggerCondition.ISSUED,
        Domain.RISK: TriggerCondition.DETECTED,
        Domain.BIM: TriggerCondition.DETECTED,
        Domain.COMMISSIONING: TriggerCondition.FAILED,
        Domain.HANDOVER: TriggerCondition.PLANNED,
        Domain.RESOURCE: TriggerCondition.VARIANCE_THRESHOLD,
        Domain.DOCUMENT: TriggerCondition.APPROVED,
    }
    return mapping.get(domain, TriggerCondition.DETECTED)


class ProjectHealthAggregator:
    """Aggregates cross-domain health signals into a unified project view.

    Consumes domain-specific status summaries and produces a unified
    health score with cross-domain flags.
    """

    def __init__(self, dependency_graph: Optional[DependencyGraph] = None) -> None:
        self.graph = dependency_graph or DependencyGraph()

    def aggregate(
        self,
        domain_status: Dict[Domain, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Produce unified project health from per-domain status.

        domain_status format:
            {Domain.SCHEDULE: {"status": "delayed", "variance_days": 14, ...}, ...}
        """
        health: Dict[str, Any] = {
            "overall_status": "healthy",
            "domain_scores": {},
            "cross_domain_flags": [],
            "suggested_actions": [],
        }

        # Expanded score map handles all common domain status strings
        score_map = {
            "healthy": 100,
            "warning": 70,
            "critical": 40,
            "unknown": 75,
            "delayed": 60,
            "overrun": 55,
            "failed": 40,
            "on_hold": 60,
            "variance": 65,
            "incident": 35,
            "detected": 70,
            "planned": 85,
            "in_progress": 80,
            "complete": 100,
        }
        scores: List[int] = []

        for domain, status in domain_status.items():
            s = status.get("status", "unknown")
            scores.append(score_map.get(s, 75))
            health["domain_scores"][domain.value] = status

            # Check for cross-domain triggers (domain-aware: quality "critical"
            # must map to FAILED → R002 NCR/hold, not CRITICAL_INCIDENT/safety)
            trigger = _status_to_trigger(s, domain=domain)
            if trigger:
                affected, actions = self.graph.trigger_analysis(domain, trigger)
                if affected:
                    health["cross_domain_flags"].append({
                        "source": domain.value,
                        "trigger": trigger.value,
                        "affected_domains": [d.value for d in affected],
                    })
                    health["suggested_actions"].extend([a.value for a in actions])

        # Overall score is weighted average; any critical domain pulls it down
        if scores:
            avg = sum(scores) / len(scores)
            health["overall_score"] = round(avg, 1)
            health["overall_status"] = (
                "healthy" if avg >= 80 else "warning" if avg >= 60 else "critical"
            )
        else:
            health["overall_score"] = 0
            health["overall_status"] = "unknown"

        # Deduplicate suggested actions
        health["suggested_actions"] = list(dict.fromkeys(health["suggested_actions"]))
        return health


def _status_to_trigger(
    status: str,
    domain: Optional[Domain] = None,
) -> Optional[TriggerCondition]:
    """Map a domain status string to the most likely trigger condition.

    Domain-aware: quality failures must fire FAILED (R002 NCR/hold), not
    CRITICAL_INCIDENT (R003 safety stop-work). Safety keeps critical→incident.
    """
    s = (status or "").lower().strip()
    if domain == Domain.QUALITY and s in ("failed", "critical", "blocked", "ncr", "fail"):
        return TriggerCondition.FAILED
    if domain == Domain.SAFETY and s in ("critical", "incident", "failed"):
        return TriggerCondition.CRITICAL_INCIDENT
    if domain == Domain.COMMISSIONING and s in ("failed", "critical"):
        return TriggerCondition.FAILED

    mapping = {
        "delayed": TriggerCondition.DELAYED,
        "failed": TriggerCondition.FAILED,
        "critical": TriggerCondition.CRITICAL_INCIDENT,
        "variance": TriggerCondition.VARIANCE_THRESHOLD,
        "overrun": TriggerCondition.VARIANCE_THRESHOLD,
        "detected": TriggerCondition.DETECTED,
        "incident": TriggerCondition.CRITICAL_INCIDENT,
        "blocked": TriggerCondition.FAILED,
        "ncr": TriggerCondition.FAILED,
    }
    return mapping.get(s)
