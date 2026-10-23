"""Cross-Domain Reasoner — Phase 4 of the Construction Management Engine.

Enables the orchestrator to reason dynamically across construction management
domains. When a user asks about one domain, the reasoner detects if other
domains are implicitly relevant, matches the query to multi-domain workflow
templates, and builds cross-domain execution plans.

Purely deterministic — no LLM calls. Runs cheaply on every turn.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from app.core.dependency_graph import (
    DependencyGraph,
    Domain,
    TriggerCondition,
    _status_to_trigger,
)
from app.core.workflow_templates import WorkflowTemplateLibrary
from app.schemas.construction_activity import ActivityType

# The confidence a template match must reach before anything acts on it. Lives
# here, applied inside `analyze_turn`, so every consumer inherits it — a gate
# that each caller has to remember is a gate that some caller will forget.
TEMPLATE_MATCH_FLOOR = 0.15

logger = logging.getLogger(__name__)

# ── Miss telemetry (issue #419) ───────────────────────────────────────────
# Keyword matching is literal. It will keep missing phrasings nobody thought
# to add, and every miss is SILENT: the turn simply gets no cross-domain help,
# nothing fails and nothing logs. That makes it impossible to say whether this
# layer helps on 5% of construction turns or 60%, which is exactly the number
# needed to justify (or drop) the move to embedding similarity.
#
# Counted only when cross-domain intent WAS detected. A miss on "hello" is not
# a miss — it is the layer correctly declining, and counting it would bury the
# signal under conversational turns.
_MATCH_OUTCOMES = ("matched", "below_floor", "no_candidate")
_MATCH_STATS: Dict[str, int] = {k: 0 for k in _MATCH_OUTCOMES}
_PROM_TEMPLATE_MATCH = None


def _bump_template_match(outcome: str, score: float, message: str) -> None:
    """Record one construction-shaped turn's template-match outcome.

    Never raises: telemetry must not be able to break a turn.
    """
    try:
        _MATCH_STATS[outcome] = _MATCH_STATS.get(outcome, 0) + 1

        global _PROM_TEMPLATE_MATCH
        if _PROM_TEMPLATE_MATCH is None:
            try:
                from prometheus_client import Counter
                _PROM_TEMPLATE_MATCH = Counter(
                    "the_fork_cm_template_match_total",
                    "Cross-domain template match outcomes on turns with detected intent",
                    ["outcome"],
                )
            # Absent library must not break a turn.
            except Exception:  # noqa: BLE001
                _PROM_TEMPLATE_MATCH = False  # library absent; in-process stats only
        if _PROM_TEMPLATE_MATCH:
            _PROM_TEMPLATE_MATCH.labels(outcome=outcome).inc()

        if outcome == "matched":
            return

        # The miss itself is the useful record; the phrasing that caused it is
        # the training signal for the embedding work. Raw user text is only
        # emitted when explicitly switched on, since these are customer
        # messages and the default log sink is not the place for them.
        if os.getenv("CM_LOG_TEMPLATE_MISSES") == "1":
            logger.info(
                "cm_template_miss outcome=%s score=%.3f message=%r",
                outcome, score, message[:200],
            )
        else:
            logger.debug("cm_template_miss outcome=%s score=%.3f", outcome, score)
    # Telemetry must never break the turn it measures.
    except Exception:
        logger.debug("cm_template_miss telemetry failed", exc_info=True)


def get_template_match_stats() -> Dict[str, int]:
    """Snapshot of match outcomes since process start (construction turns only).

    `below_floor` + `no_candidate` over the total is the miss rate that issue
    #419 exists to measure.

    Read it as "of the turns this layer recognised as construction, how many
    did it fail to help" — NOT "how many construction turns went unhelped".
    A turn that fails keyword intent detection as well never reaches the
    counter at all: "the client is unhappy about the concrete finish on the
    podium" is plainly construction, matches no domain keyword, and is
    invisible here. So this is a LOWER BOUND on the true miss rate, and the
    gap between the two cannot be closed by the same keyword tables that
    create it. Treat a bad number as conclusive and a good one as partial.
    """
    stats = dict(_MATCH_STATS)
    total = sum(stats.values())
    stats["total"] = total
    stats["miss_rate"] = (
        round((stats["below_floor"] + stats["no_candidate"]) / total, 4) if total else 0.0
    )
    return stats


def reset_template_match_stats() -> None:
    """Zero the counters. For tests and for bounding a measurement window."""
    for k in _MATCH_OUTCOMES:
        _MATCH_STATS[k] = 0


# Single-word triggers that are ordinary business English rather than
# construction terms of art. On their own they are not enough to commit to a
# multi-domain workflow — "dashboard" is a request to look at something, not a
# statement that a monthly progress cycle is under way. Terms of art
# ("variation", "snag", "prolongation") are NOT listed here: in this domain
# they are unambiguous, and demoting them is what made site language invisible.
_GENERIC_TRIGGERS = frozenset({"dashboard", "turnover", "omission"})

# ── Template keyword patterns ─────────────────────────────────────────────
# Each template's trigger phrases — matched against the lowercased user message.
#
# VOCABULARY NOTE: these must carry the words site teams actually use, not only
# the canonical textbook phrasing. Measured 2026-08-24 before the site-language
# pass below: of five realistic turns, three matched no template at all
# ("the chiller delivery has slipped four weeks", "lost-time injury on level 3",
# "claiming 45 days for the late foundation release") and one matched the wrong
# template ("snagging list ... 200 open items" -> commissioning_to_handover
# rather than qa_defect_closeout). Misses here are SILENT — the turn simply
# gets no cross-domain help — so they do not show up as errors.
_TEMPLATE_TRIGGERS: Dict[str, List[str]] = {
    "new_project_setup": [
        "new project", "project setup", "start project", "initialize project",
        "set up project", "project kickoff", "begin project", "create project",
    ],
    "monthly_progress": [
        "monthly report", "progress report", "monthly review", "status report",
        "progress update", "where are we", "how is the project", "project health",
        "project status", "dashboard", "monthly cycle",
    ],
    "change_order_impact": [
        "change order", "variation order", "vo impact", "change impact",
        "variation impact", "scope change", "change request", "extra work",
        # A variation is called a "variation" on site. The bare term was
        # missing, so "client issued a variation for the facade" only matched
        # when the word "impact" also happened to appear somewhere in the
        # sentence — the matcher accepts non-adjacent word sets, which made the
        # gap look like intermittent success rather than a missing trigger.
        "variation", "vo",
        "field instruction", "additional works", "omission",
    ],
    "delay_to_claim": [
        # Explicit claim language only. A reported slip ("delivery has
        # slipped") is not entitlement — classify_delay stays on ASK.
        "delay claim", "eot claim", "prolongation",
        "time impact", "delay analysis", "schedule delay claim", "claim for delay",
        "claiming days", "days delay", "days of delay",
        "disruption claim", "acceleration cost", "late release",
    ],
    "qa_defect_closeout": [
        "defect tracking", "defect closeout", "ncr tracking", "quality defect",
        "defect register", "rectification", "close out defect", "defect report",
        # "Snagging"/"punch list" IS defect close-out. Before this it matched
        # commissioning_to_handover instead, purely because "handover" appeared
        # in the sentence — a vocabulary gap reading as a reasoning error.
        "snag", "snag list", "snagging list",
        "punch list", "punchlist", "punch item", "open items",
        "make good", "remedial works",
    ],
    "submittal_to_install": [
        "submittal log", "material submittal", "approval tracking",
        "submittal tracker", "shop drawing", "submittal schedule",
    ],
    "safety_incident_response": [
        "safety incident", "incident report", "accident report",
        "stop work", "incident response", "safety investigation", "root cause",
        # Site language. "LTI" is the standard term on every project that
        # reports safety statistics and matched nothing before this.
        "lost time injury", "lost-time injury", "lti", "near miss",
        "first aid case", "injury", "fatality", "dangerous occurrence",
        "man down", "hse incident",
    ],
    "commissioning_to_handover": [
        "commissioning", "practical completion", "handover",
        "pc certificate", "o&m manual", "testing and commissioning",
        "system testing", "facility handover", "turnover",
    ],
    "bim_coordination": [
        "bim model", "clash detection", "model coordination",
        "ifc model", "quantity from model", "bim analysis",
        "3d model", "digital model",
    ],
    "sustainability_assessment": [
        "carbon footprint", "sustainability", "esg report",
        "green building", "leed", "breeam", "embodied carbon",
        "carbon calculation",
    ],
}

# ── Cross-domain intent keywords ──────────────────────────────────────────
# When these keywords appear in a user message, additional domains are relevant.
_KEYWORD_ROUTES: Dict[str, Tuple[Domain, Tuple[Domain, ...]]] = {
    # keyword -> (domain the keyword is ABOUT, domains it implicates)
    #
    # The source domain is what makes relevance ranking possible: a dependency
    # rule is relevant when it FIRES FROM what the user is discussing, not
    # merely when it happens to point at an implicated domain. Without it the
    # inject could only rank by target and fell back to declaration order.

    # ── Schedule ──
    # Generic delay/slip is schedule + cost only. CONTRACT (and therefore
    # claims_builder) requires explicit EOT / claim / prolongation language.
    "delay": (Domain.SCHEDULE, (Domain.COST,)),
    "behind schedule": (Domain.SCHEDULE, (Domain.COST,)),
    "behind programme": (Domain.SCHEDULE, (Domain.COST,)),
    "schedule slip": (Domain.SCHEDULE, (Domain.COST,)),
    "slip": (Domain.SCHEDULE, (Domain.COST,)),
    "slippage": (Domain.SCHEDULE, (Domain.COST,)),
    "pushed out": (Domain.SCHEDULE, (Domain.COST,)),
    "late": (Domain.SCHEDULE, (Domain.COST,)),
    "overdue": (Domain.SCHEDULE, (Domain.COST,)),
    "milestone missed": (Domain.SCHEDULE, (Domain.COST,)),
    "critical path": (Domain.SCHEDULE, (Domain.COST, Domain.RESOURCE)),
    "float": (Domain.SCHEDULE, (Domain.COST, Domain.RESOURCE)),
    "compress": (Domain.SCHEDULE, (Domain.COST, Domain.RESOURCE)),
    "crash": (Domain.SCHEDULE, (Domain.COST, Domain.RESOURCE)),
    "fast track": (Domain.SCHEDULE, (Domain.COST, Domain.RESOURCE, Domain.QUALITY)),
    "acceleration": (Domain.SCHEDULE, (Domain.COST, Domain.RESOURCE)),

    # ── Cost ──
    "over budget": (Domain.COST, (Domain.SCHEDULE, Domain.RISK)),
    "cost overrun": (Domain.COST, (Domain.SCHEDULE, Domain.RISK)),
    "budget variance": (Domain.COST, (Domain.SCHEDULE, Domain.RISK)),
    "cost impact": (Domain.COST, (Domain.SCHEDULE, Domain.RISK)),

    # ── Contract ──
    # A variation is a CONTRACT instrument. It sat under the cost section
    # before, which sent the ranking looking for cost-sourced rules.
    "change order": (Domain.CONTRACT, (Domain.SCHEDULE, Domain.COST, Domain.RISK)),
    "variation": (Domain.CONTRACT, (Domain.SCHEDULE, Domain.COST, Domain.RISK)),
    "variation order": (Domain.CONTRACT, (Domain.SCHEDULE, Domain.COST, Domain.RISK)),
    # SI is a design/site instrument, not a VO. Do not implicate COST as a VO.
    "site instruction": (Domain.DOCUMENT, (Domain.SCHEDULE, Domain.QUALITY)),
    "claim": (Domain.CONTRACT, (Domain.SCHEDULE, Domain.COST, Domain.RISK)),
    "dispute": (Domain.CONTRACT, (Domain.SCHEDULE, Domain.COST, Domain.RISK)),
    "eot": (Domain.CONTRACT, (Domain.SCHEDULE, Domain.COST)),
    "extension of time": (Domain.CONTRACT, (Domain.SCHEDULE, Domain.COST)),
    "liquidated damages": (Domain.CONTRACT, (Domain.SCHEDULE, Domain.COST)),
    "prolongation": (Domain.CONTRACT, (Domain.SCHEDULE, Domain.COST)),
    "fidic": (Domain.CONTRACT, (Domain.SCHEDULE, Domain.COST, Domain.RISK)),

    # ── Quality ──
    "defect": (Domain.QUALITY, (Domain.SCHEDULE, Domain.SAFETY)),
    "snag": (Domain.QUALITY, (Domain.SCHEDULE, Domain.HANDOVER)),
    "punch list": (Domain.QUALITY, (Domain.SCHEDULE, Domain.HANDOVER)),
    "punchlist": (Domain.QUALITY, (Domain.SCHEDULE, Domain.HANDOVER)),
    "inspection failed": (Domain.QUALITY, (Domain.SCHEDULE, Domain.SAFETY)),
    "ncr": (Domain.QUALITY, (Domain.SCHEDULE, Domain.SAFETY)),
    "non conformance": (Domain.QUALITY, (Domain.SCHEDULE, Domain.SAFETY)),
    "hold point": (Domain.QUALITY, (Domain.SCHEDULE,)),
    "wir": (Domain.QUALITY, (Domain.SCHEDULE,)),
    "work inspection request": (Domain.QUALITY, (Domain.SCHEDULE,)),
    "rework": (Domain.QUALITY, (Domain.SCHEDULE, Domain.COST)),
    "remedial works": (Domain.QUALITY, (Domain.SCHEDULE, Domain.COST)),

    # ── Safety ──
    "incident": (Domain.SAFETY, (Domain.SCHEDULE, Domain.QUALITY)),
    "accident": (Domain.SAFETY, (Domain.SCHEDULE, Domain.QUALITY)),
    "lost time injury": (Domain.SAFETY, (Domain.SCHEDULE, Domain.QUALITY, Domain.RISK)),
    "lost-time injury": (Domain.SAFETY, (Domain.SCHEDULE, Domain.QUALITY, Domain.RISK)),
    "lti": (Domain.SAFETY, (Domain.SCHEDULE, Domain.QUALITY, Domain.RISK)),
    "near miss": (Domain.SAFETY, (Domain.SCHEDULE, Domain.QUALITY)),
    "injury": (Domain.SAFETY, (Domain.SCHEDULE, Domain.QUALITY, Domain.RISK)),
    "fatality": (Domain.SAFETY, (Domain.SCHEDULE, Domain.QUALITY, Domain.RISK)),
    "stop work": (Domain.SAFETY, (Domain.SCHEDULE, Domain.QUALITY)),
    "safety violation": (Domain.SAFETY, (Domain.SCHEDULE, Domain.QUALITY)),

    # ── Procurement ──
    "delivery delayed": (Domain.PROCUREMENT, (Domain.SCHEDULE, Domain.COST)),
    "delivery slipped": (Domain.PROCUREMENT, (Domain.SCHEDULE, Domain.COST)),
    "nominated sub": (Domain.PROCUREMENT, (Domain.CONTRACT, Domain.SCHEDULE)),
    "nominated subcontractor": (Domain.PROCUREMENT, (Domain.CONTRACT, Domain.SCHEDULE)),
    "long lead": (Domain.PROCUREMENT, (Domain.SCHEDULE, Domain.RISK)),
    "procurement": (Domain.PROCUREMENT, (Domain.SCHEDULE,)),
    "purchase order": (Domain.PROCUREMENT, (Domain.SCHEDULE, Domain.QUALITY)),
    "material shortage": (Domain.PROCUREMENT, (Domain.SCHEDULE, Domain.COST)),
    "lead time": (Domain.PROCUREMENT, (Domain.SCHEDULE, Domain.COST)),
    "expediting": (Domain.PROCUREMENT, (Domain.SCHEDULE,)),
    "shipment": (Domain.PROCUREMENT, (Domain.SCHEDULE,)),

    # ── Commissioning / handover ──
    "testing": (Domain.COMMISSIONING, (Domain.HANDOVER, Domain.QUALITY)),
    "performance test": (Domain.COMMISSIONING, (Domain.HANDOVER, Domain.QUALITY)),
    "handover": (Domain.HANDOVER, (Domain.QUALITY, Domain.COMMISSIONING)),

    # ── BIM ──
    "clash": (Domain.BIM, (Domain.SCHEDULE, Domain.DOCUMENT)),
    "quantity takeoff": (Domain.BIM, (Domain.COST,)),
    "qto": (Domain.BIM, (Domain.COST,)),
    # Issued For Construction drawings — drawings, not a BIM filename.
    "ifc drawings": (Domain.DOCUMENT, (Domain.QUALITY,)),
    "ifc drawing": (Domain.DOCUMENT, (Domain.QUALITY,)),
    "issued for construction": (Domain.DOCUMENT, (Domain.QUALITY,)),

    # ── Risk ──
    "risk register": (Domain.RISK, (Domain.SCHEDULE, Domain.COST)),
    "mitigation": (Domain.RISK, (Domain.SCHEDULE, Domain.COST)),
}

# Back-compat view: keyword -> implicated domains. Derived, never edited by
# hand, so the two can never drift apart.
_CROSS_DOMAIN_KEYWORDS: Dict[str, List[Domain]] = {
    kw: list(targets) for kw, (_src, targets) in _KEYWORD_ROUTES.items()
}

# Single-token keywords and triggers match on WORD BOUNDARIES with common
# inflections (delays, delayed, claiming, snags) — never as raw substrings.
# Substring matching is how "ncr" fired inside "concrete", "eot" inside
# "geotechnical", "late" inside "calculated", and — after the site-language
# pass added them — "vo" inside "invoice" and "lti" inside "multi", which
# published the variation-order and safety-incident templates on turns that
# contain neither. Multi-word phrases keep substring semantics: they are
# specific enough that a containing sentence genuinely says them.
_DOUBLES_FINAL_CONSONANT = re.compile(r"[aeiou][bdglmnprt]$")


def _word_pattern(token: str) -> "re.Pattern[str]":
    """Match ``token`` as a whole word, allowing common inflections.

    Tokens ending consonant-after-vowel also accept the doubled-consonant
    forms, so "snag" reaches "snagging"/"snagged" and "de-snag" reaches
    "de-snagging". Without it those turns matched only the exact stem, which is
    the one spelling site teams do not use.

    The test over-accepts on multi-syllable tokens — it generates alternatives
    like "variationned" for words English does not double. That is deliberate:
    such strings never occur in text, so they cost a little regex and can
    produce no false positive, whereas hand-maintaining a list of which words
    double is a table that goes stale the first time someone adds a keyword.
    """
    stem = re.escape(token)
    suffixes = ["s", "es", "ed", "ing"]
    if _DOUBLES_FINAL_CONSONANT.search(token):
        doubled = re.escape(token[-1])
        suffixes += [doubled + "ed", doubled + "ing"]
    return re.compile(r"\b" + stem + r"(?:" + "|".join(suffixes) + r")?\b")


_ROUTE_PATTERNS: Dict[str, "re.Pattern[str]"] = {
    kw: _word_pattern(kw) for kw in _KEYWORD_ROUTES if " " not in kw
}

_TRIGGER_PATTERNS: Dict[str, "re.Pattern[str]"] = {
    t: _word_pattern(t)
    for triggers in _TEMPLATE_TRIGGERS.values()
    for t in triggers
    if " " not in t
}

# ── Post-tool cross-domain triggers ───────────────────────────────────────
# After a specific tool runs, these domains may be relevant next.
_POST_TOOL_DOMAIN_TRIGGERS: Dict[str, List[Domain]] = {
    "generate_wbs": [Domain.COST, Domain.PROCUREMENT],
    "boq_processor": [Domain.SCHEDULE, Domain.COST],
    "qa_qc_inspection": [Domain.SCHEDULE, Domain.COST],
    "safety_compliance_audit": [Domain.SCHEDULE, Domain.RISK],
    "parse_primavera_schedule": [Domain.COST, Domain.CONTRACT, Domain.RISK],
    "progress_tracker": [Domain.COST, Domain.QUALITY, Domain.SAFETY],
    "procurement_list_generator": [Domain.SCHEDULE, Domain.QUALITY],
    "claims_builder": [Domain.SCHEDULE, Domain.COST],
    "commissioning_checklist": [Domain.HANDOVER],
    "bim_analysis": [Domain.SCHEDULE, Domain.COST],
    "bim_clash_detection": [Domain.DOCUMENT, Domain.SCHEDULE, Domain.COST],
    "cash_flow_forecast": [Domain.CONTRACT],
    "payment_certificate": [Domain.COST],
    "esg_sustainability_report": [Domain.COST, Domain.PROCUREMENT],
}


class TemplateMatcher:
    """Match user messages to multi-domain workflow templates."""

    def __init__(self, template_library: Optional[WorkflowTemplateLibrary] = None) -> None:
        self._library = template_library or WorkflowTemplateLibrary()

    def match(self, user_message: str) -> List[Tuple[str, float]]:
        """Return template IDs sorted by match score (highest first).

        Scoring is deliberately INDEPENDENT of how many trigger phrases a
        template happens to list. The previous ``hits / len(triggers)`` divided
        by the list length, which meant every synonym added to a template
        lowered the score of every phrase already in it — so enriching the
        vocabulary (the whole point of the trigger table) pushed real matches
        below TEMPLATE_MATCH_FLOOR. A template with 7 triggers scored 0.143 on
        a single hit and a template with 20 scored 0.05, for identical evidence.

        Instead each matched trigger contributes weight by SPECIFICITY: a
        multi-word phrase ("lost time injury") is strong evidence, a single
        construction term of art ("snag", "variation") is weaker but still
        clears the floor, and a single word that is ordinary business English
        (_GENERIC_TRIGGERS) does not clear it alone. Two units of evidence
        saturate the score.

        That last tier is what the old length division was doing by accident:
        "dashboard" scored 1/11 because monthly_progress happens to list eleven
        triggers, which kept a bare one-word turn from publishing a five-domain
        workflow plan. That guard is real and worth keeping — it just needed
        saying out loud instead of depending on how long a list is.
        """
        text = user_message.lower()
        words = set(re.findall(r"[a-z0-9']+", text))
        scores: List[Tuple[str, float]] = []
        for template_id, triggers in _TEMPLATE_TRIGGERS.items():
            weight = 0.0
            for t in triggers:
                if " " in t:
                    hit = t in text or all(w in words for w in t.split())
                else:
                    hit = bool(_TRIGGER_PATTERNS[t].search(text))
                if hit:
                    if " " in t:
                        weight += 1.0
                    elif t in _GENERIC_TRIGGERS:
                        weight += 0.25
                    else:
                        weight += 0.5
            if weight > 0:
                scores.append((template_id, min(1.0, weight / 2.0)))
        # Ties are broken by template id so the ranking is stable across runs
        # rather than dependent on dict insertion order.
        scores.sort(key=lambda x: (-x[1], x[0]))
        return scores

    def best_match(
        self, user_message: str, threshold: float = TEMPLATE_MATCH_FLOOR
    ) -> Optional[str]:
        """Return the best matching template ID, or None if below threshold."""
        matches = self.match(user_message)
        if matches and matches[0][1] >= threshold:
            return matches[0][0]
        return None

    def get_matched_template(
        self, user_message: str, threshold: float = TEMPLATE_MATCH_FLOOR
    ):
        """Return the actual WorkflowTemplate object for the best match."""
        template_id = self.best_match(user_message, threshold)
        if template_id:
            return self._library.get(template_id)
        return None


class CrossDomainIntentDetector:
    """Detect when a query about one domain implies other domains."""

    def __init__(self, dependency_graph: Optional[DependencyGraph] = None) -> None:
        self._graph = dependency_graph or DependencyGraph()

    def detect_domains(self, user_message: str) -> Tuple[Set[Domain], Set[Domain]]:
        """Return (source domains, implicated domains) for a message.

        The source set is what the message is ABOUT; the implicated set is what
        may be affected. Only the second is exposed to callers as
        ``detect_additional_domains``, but ranking needs both.
        """
        text = user_message.lower()
        sources: Set[Domain] = set()
        targets: Set[Domain] = set()
        for keyword, (source, implicated) in _KEYWORD_ROUTES.items():
            if " " in keyword:
                hit = keyword in text
            else:
                hit = bool(_ROUTE_PATTERNS[keyword].search(text))
            if hit:
                sources.add(source)
                targets.update(implicated)
        return sources, targets

    def detect_additional_domains(self, user_message: str) -> Set[Domain]:
        """Find domains that are implicitly relevant based on message keywords."""
        return self.detect_domains(user_message)[1]

    def _rank_rules(
        self, sources: Set[Domain], targets: Set[Domain]
    ) -> List[Any]:
        """Dependency rules ordered by relevance to this message.

        This used to be ``get_rules_for_target(domain)[:3]`` — the first three
        rules DECLARED for each implicated domain, which is an artefact of the
        order somebody typed DEFAULT_RULES in, not a statement about the query.
        A variation-order question was told "Critical safety incident triggers
        stop-work order in affected zone" as a relevant linkage, because R003
        happens to be an early rule targeting SCHEDULE. Wrong, and stated with
        the same confidence as a right answer, inside the system prompt.

        A rule earns its place by firing FROM something the message is about
        (weight 3), or from something the message implicates (weight 1, the
        downstream chain), plus one per implicated domain it points at. The
        floor of 3 means a rule with no connection to the source — however many
        targets it happens to share — does not appear at all.
        """
        detected = sources | targets
        scored: List[Tuple[int, str, Any]] = []
        for rule in self._graph.rules:
            if rule.source_domain in sources:
                score = 3
            elif rule.source_domain in targets:
                score = 1
            else:
                continue
            score += sum(1 for d in rule.target_domains if d in detected)
            if score >= 3:
                # rule_id breaks ties deterministically
                scored.append((score, rule.rule_id, rule))
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [r for _score, _rid, r in scored]

    def get_cross_domain_context(self, user_message: str) -> str:
        """Generate a context string for the system prompt about cross-domain relevance.

        Returns empty string when no cross-domain intent is detected, and also
        when intent is detected but no dependency rule is genuinely relevant to
        it. Silence is the correct output there: this text enters the system
        prompt of the construction agents as assertive domain guidance, so a
        plausible-but-wrong linkage is worse than none.
        """
        sources, targets = self.detect_domains(user_message)
        if not targets:
            return ""

        rules = self._rank_rules(sources, targets)
        if not rules:
            return ""

        domain_names = sorted(d.value for d in targets)
        context_lines = [
            f"Cross-domain relevance detected: {', '.join(domain_names)} may be affected.",
            "Relevant construction management linkages:",
        ]
        from app.core.delay_advice import rewrite_r004_description

        listed = 0
        for rule in rules:
            if listed >= 5:
                break
            description = rule.description
            if rule.rule_id == "R004":
                description = rewrite_r004_description(user_message, description)
            if not description:
                continue
            listed += 1
            context_lines.append(f"  {listed}. {description}")
        if listed == 0:
            return ""
        return "\n".join(context_lines)

    def suggest_follow_up_tools(
        self, last_tool_name: str, user_message: str
    ) -> List[str]:
        """Suggest tool names that may be relevant after a tool has run.

        Maps from the post-tool domain triggers to likely block/tool names.
        """
        triggered_domains = _POST_TOOL_DOMAIN_TRIGGERS.get(last_tool_name, [])
        if not triggered_domains:
            return []

        # Map domains to common tool names
        domain_tools: Dict[Domain, List[str]] = {
            Domain.SCHEDULE: ["generate_wbs", "parse_primavera_schedule", "progress_tracker"],
            Domain.COST: ["boq_processor", "cash_flow_forecast", "cost_estimate"],
            Domain.QUALITY: ["qa_qc_inspection"],
            Domain.SAFETY: ["safety_compliance_audit"],
            Domain.PROCUREMENT: ["procurement_list_generator"],
            Domain.CONTRACT: ["process_contract", "claims_builder"],
            Domain.DOCUMENT: ["cde_post_rfi", "search_project_documents"],
            Domain.RISK: ["risk_register"],
            Domain.COMMISSIONING: ["commissioning_checklist"],
            Domain.HANDOVER: ["handover_checklist"],
            Domain.BIM: ["bim_analysis"],
            Domain.RESOURCE: ["resource_histogram"],
        }

        suggestions: List[str] = []
        for domain in triggered_domains:
            suggestions.extend(domain_tools.get(domain, []))
        from app.core.delay_advice import claims_builder_permitted

        if not claims_builder_permitted(user_message):
            suggestions = [t for t in suggestions if t != "claims_builder"]
        return suggestions[:5]  # cap suggestions


class MultiDomainPlanBuilder:
    """Build execution plans that chain multiple domains.

    Takes a workflow template and converts it into a sequence of
    block executions the orchestrator can run.
    """

    def __init__(self, template_library: Optional[WorkflowTemplateLibrary] = None) -> None:
        self._library = template_library or WorkflowTemplateLibrary()

    def build_from_template(
        self, template_id: str, params: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Build an executable plan from a workflow template.

        Returns a dict with:
            - template_id, template_name
            - understanding (the template's plan understanding)
            - steps: list of {block, params, description}
        """
        template = self._library.get(template_id)
        if template is None:
            return None

        plan = template.build_plan(params or {})
        from app.core.cm_step_aliases import is_auto_dispatch, resolve_step

        steps: List[Dict[str, Any]] = []
        for step in plan.steps:
            target = resolve_step(step.type)
            if target is None:
                continue
            # Caller-handled delivery/render: keep step_type visible, do not
            # invent a block/action that would silently succeed (e.g. health_check).
            if not is_auto_dispatch(target):
                from app.core.cm_step_aliases import is_plan_only

                plan_only = is_plan_only(step.type)
                steps.append({
                    "block": None,
                    "description": step.description,
                    "params": {**step.args, "action": None},
                    "step_type": step.type,
                    "dispatch": False,
                    "needs_caller_render": not plan_only,
                    "needs_operator": plan_only,
                    "plan_note": (
                        "Formal instrument — plan or ask. Do not pretend "
                        "an analysis template executed this step. CDE is "
                        "the register."
                        if plan_only
                        else None
                    ),
                })
                continue
            block_name, canonical_action = target
            params_out: Dict[str, Any] = {**step.args}
            # Prefer canonical ConstructionContainer / block action so
            # /v1/execute can dispatch without a second alias hop.
            if canonical_action:
                params_out["action"] = canonical_action
            elif block_name == "construction":
                params_out["action"] = step.type
            steps.append({
                "block": block_name,
                "description": step.description,
                "params": params_out,
                "step_type": step.type,
                "dispatch": True,
            })

        return {
            "template_id": template_id,
            "template_name": template.name,
            "understanding": plan.understanding,
            "steps": steps,
            "domains": template.domains,
        }

    @staticmethod
    def _step_type_block_map() -> Dict[str, str]:
        """Map workflow step types to block names (compat wrapper).

        Omits NO_AUTO_DISPATCH targets (empty block) — those are caller-handled.
        """
        from app.core.cm_step_aliases import STEP_TO_TARGET

        return {
            step: block
            for step, (block, _action) in STEP_TO_TARGET.items()
            if block
        }


class SystemPromptInjector:
    """Inject cross-domain awareness into agent system prompts.

    This is a minimal, deterministic prompt extension — no LLM calls.
    """

    def __init__(
        self,
        template_matcher: Optional[TemplateMatcher] = None,
        intent_detector: Optional[CrossDomainIntentDetector] = None,
    ) -> None:
        self._matcher = template_matcher or TemplateMatcher()
        self._detector = intent_detector or CrossDomainIntentDetector()
        self._plan_builder = MultiDomainPlanBuilder()

    def inject_for_turn(
        self, base_prompt: str, user_message: str, conversation_history: Optional[List[Dict]] = None
    ) -> str:
        """Return an enhanced system prompt with cross-domain context.

        If no cross-domain context is relevant, returns the base prompt unchanged.
        """
        extras: List[str] = []

        # 1. Check for template match
        template = self._matcher.get_matched_template(user_message)
        if template:
            extras.append(
                f"\n[Workflow Template Match: {template.name}]\n"
                f"This request matches the '{template.name}' workflow "
                f"({', '.join(template.domains)}). "
                f"When appropriate, chain across these domains rather than "
                f"answering from a single domain."
            )

        # 2. Check for cross-domain intent
        cross_domain = self._detector.get_cross_domain_context(user_message)
        if cross_domain:
            extras.append(f"\n[Cross-Domain Context]\n{cross_domain}")

        # 3. Check conversation history for prior tool calls that suggest follow-ups
        if conversation_history:
            follow_up = self._history_based_suggestions(conversation_history)
            if follow_up:
                extras.append(f"\n[Follow-up Suggestions]\n{follow_up}")

        if not extras:
            return base_prompt

        injected = base_prompt + "\n" + "\n".join(extras)
        return injected

    def _history_based_suggestions(self, history: List[Dict]) -> str:
        """Analyze conversation history for cross-domain follow-up opportunities.

        Looks for prior assistant tool calls and suggests related domains.
        """
        if not history:
            return ""

        # Find the last assistant message with tool calls
        last_tool_domains: List[str] = []
        for turn in reversed(history):
            if turn.get("role") == "assistant":
                content = turn.get("content", "")
                # Detect tool call mentions in history
                for tool_name, domains in _POST_TOOL_DOMAIN_TRIGGERS.items():
                    if tool_name in content:
                        last_tool_domains.extend(d.value for d in domains)
                break

        if not last_tool_domains:
            return ""

        unique = list(dict.fromkeys(last_tool_domains))  # dedupe preserve order
        return (
            "Based on the previous tool call, these domains may be relevant next: "
            f"{', '.join(unique)}. Consider chaining if the user asks a follow-up."
        )


class CrossDomainReasoner:
    """Unified facade for Phase 4 cross-domain reasoning.

    Combines template matching, intent detection, plan building, and
    prompt injection into one interface.
    """

    def __init__(self) -> None:
        self.template_matcher = TemplateMatcher()
        self.intent_detector = CrossDomainIntentDetector()
        self.plan_builder = MultiDomainPlanBuilder()
        self.prompt_injector = SystemPromptInjector(
            self.template_matcher, self.intent_detector
        )

    def analyze_turn(
        self, user_message: str, conversation_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """Analyze a single turn and return cross-domain reasoning results.

        Returns:
            - matched_template: best matching template above TEMPLATE_MATCH_FLOOR,
              else None
            - matched_template_score: match confidence, reported even when it is
              below the floor so callers can log or tune against it
            - additional_domains: implicitly relevant domains
            - cross_domain_context: prompt-ready context string
            - suggested_tools: tools to consider based on cross-domain links

        The floor is applied HERE rather than left to each caller. It previously
        returned whatever ranked first — a 0.125 match was reported as
        `matched_template` with no indication it was below the documented
        threshold — and the one consumer that existed re-checked the score
        itself. That works exactly until a second consumer trusts the field.
        The gate belongs in the funnel, not in whichever caller remembers it.
        """
        from app.core.delay_advice import DelayKind, classify_delay, claims_builder_permitted

        template_scores = self.template_matcher.match(user_message)
        best_score = template_scores[0][1] if template_scores else 0.0
        best_template = (
            template_scores[0][0]
            if template_scores and best_score >= TEMPLATE_MATCH_FLOOR
            else None
        )
        kind = classify_delay(user_message)
        # A reported slip is not a delay-to-claim workflow.
        if best_template == "delay_to_claim" and kind != DelayKind.CLAIM:
            best_template = None

        source_domains, additional_domains = self.intent_detector.detect_domains(
            user_message
        )
        cross_domain_context = self.intent_detector.get_cross_domain_context(user_message)

        # Record the outcome only for turns this layer considers its own — ones
        # where construction intent was actually detected. Without this the
        # miss rate is invisible: an unmatched turn produces no error, no log
        # and no user-visible difference, so nobody finds out the vocabulary
        # has a hole until someone reads a transcript by hand.
        if source_domains or additional_domains:
            if best_template is not None:
                outcome = "matched"
            elif template_scores:
                outcome = "below_floor"
            else:
                outcome = "no_candidate"
            _bump_template_match(outcome, best_score, user_message)

        # Build suggested tools from additional domains
        domain_tools = {
            Domain.SCHEDULE: ["generate_wbs", "parse_primavera_schedule"],
            Domain.COST: ["boq_processor", "cash_flow_forecast"],
            Domain.QUALITY: ["qa_qc_inspection"],
            Domain.SAFETY: ["safety_compliance_audit"],
            Domain.PROCUREMENT: ["procurement_list_generator"],
            Domain.CONTRACT: ["process_contract", "claims_builder"],
            Domain.RISK: ["risk_register"],
            Domain.BIM: ["bim_analysis"],
            Domain.COMMISSIONING: ["commissioning_checklist"],
            Domain.HANDOVER: ["handover_checklist"],
            Domain.DOCUMENT: ["cde_post_rfi", "search_project_documents"],
            Domain.RESOURCE: ["resource_histogram"],
        }
        # Tools for the domain the message is ABOUT come first, then tools for
        # the domains it implicates. Drawing only from the implicated set meant
        # a procurement question never suggested the procurement tool — the
        # source domain is the one most likely to have the right tool.
        suggested_tools: List[str] = []
        for domain in sorted(source_domains, key=lambda d: d.value):
            suggested_tools.extend(domain_tools.get(domain, []))
        for domain in sorted(additional_domains, key=lambda d: d.value):
            suggested_tools.extend(domain_tools.get(domain, []))
        suggested_tools = list(dict.fromkeys(suggested_tools))[:6]
        if not claims_builder_permitted(user_message):
            suggested_tools = [t for t in suggested_tools if t != "claims_builder"]

        return {
            "matched_template": best_template,
            "matched_template_score": round(best_score, 3),
            "additional_domains": [d.value for d in additional_domains],
            "cross_domain_context": cross_domain_context,
            "suggested_tools": suggested_tools,
        }

    def build_plan(self, template_id: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Build a multi-domain execution plan from a template."""
        return self.plan_builder.build_from_template(template_id, params)

    def inject_prompt(
        self, base_prompt: str, user_message: str, history: Optional[List[Dict]] = None
    ) -> str:
        """Inject cross-domain awareness into a system prompt."""
        return self.prompt_injector.inject_for_turn(base_prompt, user_message, history)

    def get_post_tool_suggestions(self, last_tool_name: str, user_message: str) -> List[str]:
        """Get tool suggestions for follow-up after a tool has run."""
        return self.intent_detector.suggest_follow_up_tools(last_tool_name, user_message)
