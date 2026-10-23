"""Smart Orchestrator Block - 58-action keyword router for construction workflows.

The runtime action list is built by prepending PROCEDURE_ROUTING_ADDITIONS
(17 procedure-kind actions from the catalogue) to the in-file ACTION_PATTERNS
list. Six action names appear in both lists; their keyword
lists are MERGED at scoring time so neither source loses coverage.
Net unique actions: 17 + 47 smart_orch − 6 collisions = 58
(look_ahead + evm_calculate; earlier cde_post_rfi + cde_poll_events).
"""

import re
from typing import Any, Dict, List, Optional, Tuple
from app.core.universal_base import UniversalBlock
from app.blocks._procedure_routing import PROCEDURE_ROUTING_ADDITIONS
from app.core.clash_intent import message_wants_clash
from app.core.answer_report_intent import (
    ANSWER_REPORT_BLOCKED_ACTIONS,
    message_wants_answer_report,
)
from app.core.contract_lookup_intent import (
    CONTRACT_LOOKUP_BLOCKED_ACTIONS,
    message_is_contract_data_lookup,
)
import logging

logger = logging.getLogger(__name__)


# Word-boundary keyword cache. Substring `kw in text` is unsafe — 2-char keys
# like "co" / "vo" match inside "concrete", "construction", "voltage", and so
# every chat message gets routed to change-order / variation-order actions.
# Use a regex with negative-alphanumeric lookarounds so a keyword matches only
# at a word boundary (yet still matches when wrapped in punctuation or spaces).
_KW_REGEX_CACHE: Dict[str, "re.Pattern[str]"] = {}


def _kw_pattern(kw: str) -> "re.Pattern[str]":
    pat = _KW_REGEX_CACHE.get(kw)
    if pat is None:
        # `(?<![a-z0-9])` / `(?![a-z0-9])` reject neighbours that are letters
        # or digits but allow punctuation (so ".xlsx" still matches in ".xlsx file").
        pat = re.compile(
            r"(?<![a-z0-9])" + re.escape(kw) + r"(?![a-z0-9])",
            flags=re.IGNORECASE,
        )
        _KW_REGEX_CACHE[kw] = pat
    return pat


def _matches_keyword(kw: str, text: str) -> bool:
    return bool(_kw_pattern(kw).search(text))


# ── Action → keyword patterns ─────────────────────────────────────────────────
# PROCEDURE_ROUTING_ADDITIONS are prepended so procedure-specific queries (RFI,
# NCR, design review, payment cert, etc.) are matched BEFORE the generic
# construction keywords below — otherwise "raise an NCR" falls through to a
# bare contract-analysis action.
ACTION_PATTERNS: List[Tuple[str, List[str]]] = PROCEDURE_ROUTING_ADDITIONS + [
    # Construction knowledge base (cited engineering rules/formulas). Keywords
    # are specific multi-word engineering phrasings so they don't collide with
    # the schedule / manpower / cost actions below.
    ("construction_advisor", [
        "mass concrete", "equilibrium time", "thermal control", "core temperature",
        "compaction acceptance", "field density", "proctor", "swelling factor",
        "dewatering", "well point", "ground improvement", "dynamic compaction",
        "soil stabilization", "roller compacted", "rcc mix", "bitumen content",
        "wearing course", "heavy lift", "crane lift", "lifting feasibility",
        "uplift", "factor of safety", "diaphragm wall", "post-tension",
        "sealant joint", "rebar coupler", "tunnel cross section",
        "earthworks production", "haul distance", "design formula",
    ]),
    # Leftover L6: "bank volume of a rectangular trench 14.5 × 3.2 × 1.75"
    # used to miss every keyword, fall through to intelligent_workflow, then
    # the hat called construction action=construction_calc (Unknown action).
    ("construction_calc", [
        "bank volume", "excavation volume", "trench volume", "pit volume",
        "rectangular trench", "rectangular pit", "cut volume",
        # Live UI pack E4 — leftover L6 keywords were earthwork-only, so
        # "Concrete volume for a raft … waste factor" never reached the
        # calculator and the model reported net 900.
        "concrete volume", "raft volume", "waste factor",
    ]),
    # BOQ / Cost
    ("boq_process",           ["boq", "bill of quantities", "bill of quantity", "quantities sheet", "cost sheet", "price list", ".xlsx", ".csv"]),
    # WBS / schedule generation — matched ahead of extract_quantities so
    # "200 activities" / "schedule" requests don't fall through to QTO.
    # Keywords cover the real-world phrasings: "L2 schedule", "level 2 schedule",
    # "300 activities" (any 3-digit count), "draft schedule", etc. Previously
    # narrow patterns missed "create L2 schedule with around 300 activities"
    # which scored 0 and went to the single-shot fast path — where the LLM
    # invents manpower histograms because it can't call generate_wbs.
    ("generate_wbs",          [
        "wbs", "work breakdown",
        # Activity-count phrasings — any "N activities" / "N tasks".
        "activities schedule", "activity schedule",
        "200 activities", "300 activities", "400 activities", "500 activities",
        "100 activities", "150 activities", "250 activities", "350 activities",
        "schedule activities",
        # Action verbs + schedule
        "create schedule", "generate schedule", "build schedule", "draft schedule",
        "produce schedule", "make schedule", "develop schedule", "prepare schedule",
        "schedule template", "activity list",
        # Schedule levels (L1/L2/L3/L4)
        "l1 schedule", "l2 schedule", "l3 schedule", "l4 schedule",
        "level 1 schedule", "level 2 schedule", "level 3 schedule", "level 4 schedule",
        # Construction-specific
        "construction schedule", "project schedule", "master schedule",
        "baseline schedule", "epc schedule",
    ]),
    # Combined "extract_quantities" entry — there used to be two; the dedupe
    # guard in _match_actions silently dropped the second list. Merged here.
    ("extract_quantities",    ["extract quantities", "take off", "qto", "quantity take", "measure", "count items", "area calculation", "room area", "floor area", "calculate area", "quantities", "takeoff"]),
    ("estimate_costs",        ["estimate cost", "cost estimate", "budget", "pricing", "price estimate", "how much"]),
    ("tender_bid_analysis",   ["tender", "bid", "proposal", "quote comparison", "contractor bid",
                                "contractor bids", "compare bids", "score bids", "tender scoring", "evaluate bids"]),
    ("procurement_list_generator", ["procurement", "material list", "purchase list", "buy list", "vendor list", "procurement list", "what materials", "need to buy", "materials list"]),
    ("procurement_optimizer", ["optimize procurement", "best supplier", "cheapest", "optimize cost"]),
    ("payment_certificate",   ["payment cert", "valuation", "progress payment", "invoice certificate", "payment invoice", "certificate"]),
    ("cash_flow_forecast",    ["cash_flow_forecast", "cash_flow", "cashflow", "cash-flow", "cash flow", "s-curve", "payment schedule", "fund flow", "s curve", "spend curve", "cumulative spend", "cumulative cost curve", "drawdown"]),
    # Specifications
    ("spec_analyze",          ["spec", "specification", "material spec", "grade requirement", "astm", "aci", "saso", "standard", "compliance check", "specification requirements", "material specs", "material specifications", "concrete specification", "specs"]),
    ("process_specification_full", ["full specification", "spec section", "csi division", "masterformat"]),
    # Drawings
    ("drawing_qto",           ["drawing", "dxf", "dwg", "floor plan", "blueprint", "autocad", "measure drawing", "drawings", "quantity takeoff", "takeoff", "take-off",
                                "ifc drawing", "ifc drawings", "issued for construction"]),
    # Schedule
    ("parse_primavera_schedule", ["primavera", "xer", "p6", "schedule", "gantt", "programme", "baseline", "milestones", "milestone", "key milestones", "milestone report", "completion dates", "major completion dates"]),
    ("progress_tracker",      ["progress", "completion", "percent complete", "actual vs planned", "delay",
                                "progress tracking", "tracking against planned", "slipping", "behind schedule", "schedule slippage",
                                "delivery has slipped", "delivery slipped", "has slipped"]),
    ("evm_calculate",         ["earned value", "evm", "cpi", "spi", "cost performance index",
                                "schedule performance index", "estimate at completion", "eac",
                                "cost variance", "schedule variance", "bcwp", "bcws", "acwp"]),
    ("resource_histogram",    ["resource", "manpower", "histogram", "crew", "labor loading", "workforce"]),
    ("look_ahead",            ["look ahead", "lookahead", "look-ahead", "3 week look", "4 week look",
                                "three week look", "four week look", "rolling look ahead",
                                "short term programme", "short-term programme"]),
    ("forensic_delay_analysis", ["delay analysis", "eot", "extension of time", "delay claim", "forensic"]),
    # BIM / IFC
    ("bim_analysis",          ["bim", "ifc", "revit", "3d model", "building model", "navisworks"]),
    # Bare "conflict" removed — too broad (schedule conflict). Synonyms must
    # also be recognized by message_wants_clash or the post-match filter
    # strips bim_clash_detection whenever the literal word "clash" is absent.
    ("bim_clash_detection",   ["clash", "clash detection", "interference", "mep conflict"]),
    ("bim_extract",           ["extract bim", "ifc quantities", "bim quantities", "model quantities"]),
    ("digital_twin_sync",     ["digital twin", "asset data", "sync model", "as-built bim"]),
    # QA/QC
    ("qa_qc_inspection",      ["qa", "qc", "quality", "inspection", "test report", "punch list"]),
    ("commissioning_checklist", ["commissioning", "handover", "pre-commissioning", "startup checklist", "commissioning checklist", "commissioning steps", "t&c steps", "energising", "energisation", "energizing", "energization"]),
    # Contracts / Claims
    ("process_contract",      ["contract", "subcontract", "agreement", "terms", "clause", "fidic", "nec",
                                "nominated sub", "nominated subcontractor", "nominated subcontract"]),
    # Note: 2-char abbreviations "co" / "vo" removed — they were substrings of
    # "concrete", "construction", "cost", "compliance", "voltage", so every
    # construction message used to route here. The multi-word forms cover
    # the same intent without false positives.
    ("change_order_impact",   ["change order", "variation", "scope change", "amendment"]),
    ("variation_order_manager", ["variation order", "variation management", "change log"]),
    # "extension of time" is EOT-only (forensic_delay_analysis). Do not
    # auto-route a time request onto claims_builder.
    ("claims_builder",        ["claim", "dispute", "loss and expense", "damages", "delay claim", "eot claim"]),
    ("rfi_generator",         ["rfi", "request for information", "query", "clarification", "design query"]),
    ("cde_post_rfi",          [
        "post this rfi to aconex", "post the rfi to aconex", "post rfi to aconex",
        "send this rfi to aconex", "send the rfi to aconex", "send rfi to aconex",
        "post this draft to aconex", "post the draft to aconex",
        "post this rfi to the cde", "send this rfi to the cde",
    ]),
    ("cde_poll_events",       [
        "poll aconex", "poll the cde", "poll cde events", "ingest cde events",
        "pull aconex mail", "subscribe cde mail", "cde event poll",
    ]),
    # Safety
    ("safety_compliance_audit", ["safety", "hse", "osha", "risk assessment", "hazard", "ppe", "toolbox",
                                  "hse compliance audit", "working at height", "work at height", "fall protection",
                                  "site safety audit", "compliance audit checklist"]),
    ("risk_register_auto_populate", ["risk register", "risk log", "risk matrix", "risk assessment"]),
    # Sustainability
    ("carbon_footprint_calculator", ["carbon", "co2", "emissions", "sustainability", "embodied carbon", "lca"]),
    ("esg_sustainability_report", ["esg", "green", "leed", "breeam", "environmental report", "sustainability report"]),
    # Reports
    ("daily_site_report",     ["daily report", "site diary", "dsr", "site report", "daily log"]),
    ("submittal_log_generator", ["submittal", "transmittal", "document log", "material approval", "shop drawing",
                                  "submittal log", "submittal register", "approval status", "material submittal", "shop drawing submittal"]),
    ("as_built_deviation_report", ["as-built", "as built", "deviation", "red-line", "record drawing",
                                    "as-built deviations", "as built deviations", "deviations from design",
                                    "redline drawing", "as-built report"]),
    ("warranty_maintenance_schedule", ["warranty", "maintenance", "service schedule", "pppm", "o&m"]),
    ("om_manual_generator",   ["o&m", "operation manual", "maintenance manual", "handover manual",
                                "o&m manual", "operation and maintenance manual", "o&m outline", "o and m manual"]),
    # Value / Analysis
    ("value_engineering",     ["value engineering", "ve study", "cost reduction", "alternative", "optimization",
                                "value engineer", "value engineering options", "cut cost", "cost reduction options",
                                "optimize cost", "value engineering study"]),
    ("sympy_reason",          ["variance analysis", "reasoning", "compare cost", "benchmark", "symbolic", "formula"]),
    # Documents
    ("process_document",      ["document", "pdf", "report", "upload", "analyse file", "analyze file", "execution plan", "project execution plan", "what documents", "list the documents", "which documents", "which drawings", "project documents", "what files", "documents in this project"]),
    # Reasoning / AI
    ("intelligent_workflow",  ["workflow", "automate", "chain", "pipeline", "multi-step", "full analysis"]),
    ("health_check",          ["health", "status", "ping", "alive", "system check"]),
]

# File extension → action
FILE_TYPE_MAP: Dict[str, str] = {
    ".xlsx": "boq_process",
    ".xls":  "boq_process",
    ".csv":  "boq_process",
    ".xer":  "parse_primavera_schedule",
    ".ifc":  "bim_analysis",
    ".dxf":  "drawing_qto",
    ".dwg":  "drawing_qto",
    ".pdf":  "process_document",
    ".jpg":  "qa_qc_inspection",
    ".jpeg": "qa_qc_inspection",
    ".png":  "qa_qc_inspection",
}

# Actions that can run in parallel safely
PARALLEL_GROUPS: Dict[str, List[str]] = {
    "full_analysis": [
        "boq_process", "spec_analyze", "drawing_qto", "parse_primavera_schedule"
    ],
    "cost_suite": [
        "extract_quantities", "estimate_costs", "procurement_list_generator"
    ],
    "compliance_suite": [
        "spec_analyze", "qa_qc_inspection", "safety_compliance_audit"
    ],
    "reporting_suite": [
        "daily_site_report", "progress_tracker", "cash_flow_forecast"
    ],
}


class SmartOrchestratorBlock(UniversalBlock):
    auto_validate = False
    name = "smart_orchestrator"
    version = "1.1.0"
    description = "58-action construction keyword router: maps user messages to action queues with parallel execution hints"
    layer = 2
    tags = ["infrastructure", "construction", "orchestration", "routing", "nlp"]
    requires = []
    allow_empty_input = True

    default_config = {
        "max_actions": 5,
        "confidence_threshold": 0.3,
        "fallback_agent": "intelligent_workflow",
    }

    ui_schema = {
        "input": {
            "type": "text",
            "placeholder": "Describe what you need (e.g. 'analyze the BOQ and check specs')...",
            "multiline": True,
        },
        "output": {
            "type": "json",
            "fields": [
                {"name": "action_queue", "type": "list", "label": "Action Queue"},
                {"name": "parallel_flag", "type": "boolean", "label": "Run Parallel"},
                {"name": "fallback_agent", "type": "text", "label": "Fallback"},
            ],
        },
        "quick_actions": [
            {"icon": "🔀", "label": "Route Message", "prompt": "What actions should I run for this request?"},
            {"icon": "📋", "label": "List Actions", "prompt": "List all available construction actions"},
        ],
    }

    async def process(self, input_data: Any, params: Dict = None) -> Dict:
        params = params or {}
        data = input_data if isinstance(input_data, dict) else {}

        user_message = (
            data.get("user_message")
            or data.get("message")
            or data.get("text")
            or data.get("input")
            or (input_data if isinstance(input_data, str) else "")
        )
        session_context = data.get("session_context", {})
        # A procedure code the user typed ("XYZ-101") carries no meaning in
        # the product. Resolve it at run time against the project's own
        # document names and let keyword matching see that document's kind.
        match_message = user_message
        _pid = (session_context or {}).get("project_id") if isinstance(session_context, dict) else None
        if _pid and user_message:
            from app.core import procedure_catalogue as _pc

            if _pc.has_code(user_message):
                match_message = _pc.expand_codes(user_message, _pc.live_document_names(_pid))
        file_type = (
            data.get("file_type")
            or params.get("file_type")
            or self._detect_file_type(data, session_context)
        )
        promote_cm_followups = bool(
            data.get("promote_cm_followups")
            or params.get("promote_cm_followups")
            or self.config.get("promote_cm_followups", False)
        )

        if user_message.strip().lower() in ("list actions", "help", "what can you do"):
            return self._list_actions()

        # ── Learned routing (PR 1) — opt-in via routing_mode="learned" ──
        # The learned classifier predicts ONE action with a calibrated
        # confidence. Below the threshold (or when the model isn't loaded),
        # we fall back to the keyword regex so we never regress on what
        # already worked. Every learned dispatch is recorded as a
        # routing_decisions pattern on learning_engine so the next retrain
        # has live data, not just seed keywords.
        routing_mode = data.get("routing_mode") or params.get("routing_mode") or self.config.get("routing_mode", "auto")
        learned_prediction: Optional[Dict[str, Any]] = None
        if routing_mode == "learned":
            learned_prediction = self._predict_learned(user_message)
            if learned_prediction and not learned_prediction.get("fallback_recommended"):
                action = learned_prediction["action"]
                action_queue = [action]
                max_actions = int(params.get("max_actions", self.config.get("max_actions", 5)))
                # Keep keyword secondary matches as supporting context but
                # surface the learned pick as primary.
                matched_secondary = self._match_actions(match_message, file_type)
                secondary = [m for m in matched_secondary if m["action"] != action][:max_actions - 1]
                action_queue.extend(m["action"] for m in secondary)

                parallel_group = self._detect_parallel_group(action_queue)
                parallel_flag = parallel_group is not None or len(action_queue) > 1
                fallback = self.config.get("fallback_agent", "intelligent_workflow")

                self._record_routing_decision(
                    user_message, action, learned_prediction["confidence"],
                    source="learned", session_context=session_context,
                )

                cm_meta = self._cross_domain_enrichment(
                    user_message, action_queue, promote_cm_followups=promote_cm_followups
                )
                # Enrichment may append high-confidence follow-ups; refresh parallel.
                parallel_group = self._detect_parallel_group(action_queue)
                parallel_flag = parallel_group is not None or len(action_queue) > 1

                return {
                    "status": "success",
                    "action_queue": action_queue,
                    "parallel_flag": parallel_flag,
                    "parallel_group": parallel_group,
                    "fallback_agent": fallback,
                    "matched_actions": [{
                        "action": action,
                        "score": learned_prediction["confidence"],
                        "source": "learned",
                    }] + secondary,
                    "file_type_hint": file_type,
                    "session_context": session_context,
                    "routing_mode": "learned",
                    "model_confidence": learned_prediction["confidence"],
                    "fallback_used": False,
                    "top_k": learned_prediction.get("top_k", []),
                    **cm_meta,
                }
            # else: model not loaded OR low confidence → fall through to keyword

        matched = self._match_actions(match_message, file_type)
        max_actions = int(params.get("max_actions", self.config.get("max_actions", 5)))
        action_queue = [m["action"] for m in matched[:max_actions]]

        parallel_group = self._detect_parallel_group(action_queue)
        parallel_flag = parallel_group is not None or len(action_queue) > 1

        fallback = self.config.get("fallback_agent", "intelligent_workflow")
        if not action_queue:
            action_queue = [fallback]
            parallel_flag = False

        # Record keyword-mode dispatches too so the classifier has training
        # data even before anyone opts into learned mode. Skip the empty /
        # fallback case (nothing useful to learn from). Note _match_actions
        # returns dicts with key "confidence", not "score".
        if routing_mode == "learned" and action_queue and action_queue[0] != fallback:
            self._record_routing_decision(
                user_message, action_queue[0],
                (matched[0].get("confidence", 0.0) if matched else 0.0),
                source="keyword_fallback", session_context=session_context,
            )

        # Cross-domain reasoner hook (non-competing): after keyword routing,
        # attach template match + post-tool follow-ups. May append up to two
        # high-confidence follow-ups for WBS/procurement seeds only — still
        # does not replace ACTION_PATTERNS ownership.
        cm_meta = self._cross_domain_enrichment(
            user_message, action_queue, promote_cm_followups=promote_cm_followups
        )
        parallel_group = self._detect_parallel_group(action_queue)
        parallel_flag = parallel_group is not None or len(action_queue) > 1

        return {
            "status": "success",
            "action_queue": action_queue,
            "parallel_flag": parallel_flag,
            "parallel_group": parallel_group,
            "fallback_agent": fallback,
            "matched_actions": matched[:max_actions],
            "file_type_hint": file_type,
            "session_context": session_context,
            "routing_mode": routing_mode,
            **cm_meta,
            **({"fallback_used": True, "model_confidence": (learned_prediction or {}).get("confidence", 0.0),
                "fallback_reason": (learned_prediction or {}).get("reason")}
               if routing_mode == "learned" else {}),
        }

    # Orchestrator keyword → CrossDomainReasoner post-tool trigger keys.
    # ACTION_PATTERNS uses boq_process; reasoner triggers are keyed boq_processor.
    _POST_TOOL_SEED_ALIASES: Dict[str, str] = {
        "boq_process": "boq_processor",
    }
    # Same floor as TemplateMatcher.best_match / get_matched_template.
    _CM_TEMPLATE_SCORE_THRESHOLD: float = 0.15
    # Seeds that may promote post-tool follow-ups onto action_queue.
    # Contract: only these primary actions may grow the queue; other seeds
    # keep follow-ups as cm_follow_up_tools metadata only (no second router).
    _CM_QUEUE_APPEND_SEEDS = frozenset({
        "generate_wbs",
        "procurement_list_generator",
    })
    # Max follow-ups promoted onto action_queue per turn.
    _CM_QUEUE_APPEND_CAP: int = 2

    def _normalize_post_tool_seed(self, seed: Optional[str]) -> Optional[str]:
        if not seed:
            return seed
        return self._POST_TOOL_SEED_ALIASES.get(seed, seed)

    def _known_orchestrator_actions(self) -> set:
        """Action names owned by ACTION_PATTERNS (single router vocabulary)."""
        return {action for action, _ in ACTION_PATTERNS}

    def _safe_queue_follow_ups(
        self,
        action_queue: List[str],
        follow_ups: List[str],
        *,
        seed: Optional[str],
        template_score: float,
        suggested_tools: List[str],
        promote_cm_followups: bool = False,
    ) -> List[str]:
        """Promote high-confidence follow-ups onto action_queue (in place).

        Safety gates (all required):
          1. Seed is generate_wbs, procurement_list_generator, or (when
             promote_cm_followups=true) boq_process.
          2. Template match score >= 0.15 OR the tool also appears in
             analyze_turn suggested_tools (cross-domain intent overlap).
          3. Tool is a known ACTION_PATTERNS action (no alien reasoner names).
          4. Cap at _CM_QUEUE_APPEND_CAP; never duplicate queue entries.

        Returns the list of tools actually appended (for cm_queue_appended).
        """
        allowed_seeds = set(self._CM_QUEUE_APPEND_SEEDS)
        if promote_cm_followups:
            allowed_seeds.add("boq_process")
        if seed not in allowed_seeds or not follow_ups:
            return []
        known = self._known_orchestrator_actions()
        # boq_processor is the reasoner key; orchestrator owns boq_process.
        known_aliases = {"boq_processor": "boq_process", "cost_estimate": "estimate_costs"}
        suggested = set(suggested_tools or [])
        high_conf_template = template_score >= self._CM_TEMPLATE_SCORE_THRESHOLD
        queued = set(action_queue)
        appended: List[str] = []
        for raw in follow_ups:
            if len(appended) >= self._CM_QUEUE_APPEND_CAP:
                break
            tool = known_aliases.get(raw, raw)
            if tool not in known or tool in queued:
                continue
            if not (high_conf_template or raw in suggested or tool in suggested):
                continue
            action_queue.append(tool)
            queued.add(tool)
            appended.append(tool)
        return appended

    def _cross_domain_enrichment(
        self, user_message: str, action_queue: List[str], *, promote_cm_followups: bool = False
    ) -> Dict[str, Any]:
        """Attach CM template / post-tool suggestions; optionally grow the queue.

        Prefer generate_wbs / procurement_list_generator as the post-tool
        seed when present; otherwise use the primary matched action.

        Document contract for consumers (chat hint, agent gate, UI):
          - cm_follow_up_tools: advisory list (always, capped).
          - cm_suggested_tools: cross-domain intent tools from analyze_turn.
          - cm_matched_template / cm_workflow_plan: only when score >= 0.15.
          - cm_queue_appended: tools promoted onto action_queue this turn
            (empty unless seed is WBS/procurement/BOQ-with-flag and safety gates pass).
          - cm_prompt_inject: CrossDomainReasoner.inject_prompt fragment for
            runtime system-prompt enrichment (empty when no cross-domain hit).
        """
        try:
            from app.core.cross_domain_reasoner import CrossDomainReasoner

            reasoner = CrossDomainReasoner()
            analysis = reasoner.analyze_turn(user_message)
            seed = None
            for preferred in ("generate_wbs", "procurement_list_generator"):
                if preferred in action_queue:
                    seed = preferred
                    break
            if seed is None and action_queue:
                seed = action_queue[0]
            seed_for_triggers = self._normalize_post_tool_seed(seed)
            follow_ups: List[str] = []
            if seed_for_triggers:
                follow_ups = reasoner.get_post_tool_suggestions(
                    seed_for_triggers, user_message
                )
            # Drop tools already in the queue
            queued = set(action_queue)
            follow_ups = [t for t in follow_ups if t not in queued]

            score = float(analysis.get("matched_template_score") or 0.0)
            suggested = analysis.get("suggested_tools") or []
            appended = self._safe_queue_follow_ups(
                action_queue,
                follow_ups,
                seed=seed,
                template_score=score,
                suggested_tools=suggested,
                promote_cm_followups=promote_cm_followups,
            )

            # Prompt inject for chat/agent runtime (deterministic, no LLM).
            injected = reasoner.inject_prompt("", user_message)
            prompt_fragment = injected.strip() if injected and injected.strip() else ""

            out: Dict[str, Any] = {
                "cm_follow_up_tools": follow_ups[:5],
                "cm_suggested_tools": suggested,
                "cm_queue_appended": appended,
            }
            if prompt_fragment:
                out["cm_prompt_inject"] = prompt_fragment
            # analyze_turn returns the raw top hit; only publish a workflow plan
            # when score clears TemplateMatcher.best_match's 0.15 threshold.
            template_id = analysis.get("matched_template")
            if template_id and score >= self._CM_TEMPLATE_SCORE_THRESHOLD:
                out["cm_matched_template"] = template_id
                out["cm_matched_template_score"] = analysis["matched_template_score"]
                plan = reasoner.build_plan(template_id)
                if plan:
                    out["cm_workflow_plan"] = {
                        "template_id": plan["template_id"],
                        "template_name": plan["template_name"],
                        "step_count": len(plan["steps"]),
                        "steps": [
                            {
                                "block": s["block"],
                                "action": s["params"].get("action"),
                                "step_type": s.get("step_type"),
                                "description": s.get("description"),
                                "dispatch": s.get("dispatch", True),
                                "needs_caller_render": s.get("needs_caller_render", False),
                            }
                            for s in plan["steps"][:12]
                        ],
                    }
            elif template_id:
                # Expose weak score for debugging without advertising a plan.
                out["cm_matched_template_score"] = analysis["matched_template_score"]
            return out
        except Exception:  # noqa: BLE001 — never break keyword routing
            return {}

    def _predict_learned(self, message: str) -> Optional[Dict[str, Any]]:
        """Consult learning_engine's predict_route op. Returns None on any error
        (caller treats that as "fall back to keyword router").

        Uses shared_instance() to avoid a full JSON load per chat dispatch —
        before the singleton, every call instantiated a fresh LearningEngineBlock
        which read _state from disk in __init__. Reviewer fix from PRs #19-#23 retro.
        """
        from app.blocks import BLOCK_REGISTRY

        cls = BLOCK_REGISTRY.get("learning_engine")
        if cls is None:
            return None
        try:
            le = cls.shared_instance()
            # We bypass execute() and call the op directly to avoid the
            # envelope wrap; this is a hot path, ms matter.
            return le._predict_route({"text": message}, {})
        except Exception:  # noqa: BLE001
            return None

    def _record_routing_decision(
        self, message: str, action: str, score: float,
        source: str, session_context: Dict,
    ) -> None:
        """Log the dispatch as a routing_decisions pattern on learning_engine
        so the next train_router has live data. Best-effort — failures here
        never break the dispatch.

        Uses shared_instance() so concurrent dispatches share one instance
        with a lock around _record_pattern's read-modify-write window;
        before the singleton, concurrent writes raced and silently lost
        observations. Reviewer fix from PRs #19-#23 retro.
        """
        import json
        from app.blocks import BLOCK_REGISTRY

        try:
            cls = BLOCK_REGISTRY.get("learning_engine")
            if cls is None:
                return
            le = cls.shared_instance()
            project_id = (session_context or {}).get("project_id") or "default"
            le._record_pattern({
                "project_id": project_id,
                "category": "routing_decisions",
                "observation": json.dumps({
                    "text": message[:500],
                    "action": action,
                    "score": float(score),
                    "source": source,
                    "corrected": False,
                }, ensure_ascii=False),
                "source": "smart_orchestrator",
            }, {})
        except Exception:  # noqa: BLE001
            logger.debug(
                "swallowed %s in _record_routing_decision() — continuing",
                "Exception", exc_info=True,
            )

    def _match_actions(self, message: str, file_type: Optional[str]) -> List[Dict]:
        scores: Dict[str, float] = {}

        # File type gives strong signal
        if file_type and file_type in FILE_TYPE_MAP:
            action = FILE_TYPE_MAP[file_type]
            scores[action] = scores.get(action, 0.0) + 0.8

        # Build the runtime keyword view by MERGING duplicate action names.
        # Pre-1.1.0 the matcher skipped any ACTION_PATTERNS entry whose
        # action name had already been seen — that silently dropped the
        # second occurrence's keyword list. Real example: PROCEDURE_ROUTING
        # owns `safety_compliance_audit` with HSE-audit / stop-work /
        # near-miss keywords, while smart_orchestrator's own list adds the
        # everyday phrasings `safety`, `hse`, `osha`, `ppe`, `toolbox`,
        # `risk assessment`, `hazard`. The skip dropped all seven, so
        # "any safety concerns?" never routed to safety_compliance_audit.
        # Merging guarantees every keyword from every source is matched.
        merged_patterns: Dict[str, List[str]] = {}
        for action, keywords in ACTION_PATTERNS:
            existing = merged_patterns.setdefault(action, [])
            for kw in keywords:
                if kw not in existing:
                    existing.append(kw)

        # Keyword matching — uses word-boundary regex (see _matches_keyword).
        # Pure substring matching was unsafe: 2-char keys like "co" matched
        # "concrete", "compliance", "cost"; "qa" matched "quantity"; etc.
        for action, keywords in merged_patterns.items():
            for kw in keywords:
                if _matches_keyword(kw, message):
                    weight = len(kw.split()) * 0.2  # multi-word keywords score higher
                    scores[action] = scores.get(action, 0.0) + weight

        # "use 6 days per slab and re-run" must beat payment/calc keywords.
        try:
            from app.lib.wbs_duration_overrides import message_wants_wbs_duration_rerun
            if message_wants_wbs_duration_rerun(message):
                scores["generate_wbs"] = max(scores.get("generate_wbs", 0.0), 0.85)
        except Exception:  # noqa: BLE001
            logger.debug(
                "duration-override generate_wbs boost skipped",
                exc_info=True,
            )

        # Per-action gate-1 threshold (operator brief 2026-06-19, PR #80).
        # GENERATIVE_INTENTS — the actions that legitimately need a heavy-
        # reasoning dispatch (generate_wbs, bim_analysis, drawing_qto, ...)
        # — clear gate-1 at a relaxed 0.2 floor so single-word matches like
        # the bare "wbs" in "generate a WBS for a 10-floor tower" surface
        # in matched_actions for action_router.needs_planning to evaluate.
        # Non-generative actions keep the historical 0.3 to suppress thin
        # matches that would just add noise to the action_queue.
        #
        # Local import to avoid the runtime → action_router cycle: the
        # routing helper in app.agents.runtime imports action_router, which
        # imports nothing from app.blocks; loading GENERATIVE_INTENTS at
        # module top would invert that ordering and break the agent runtime
        # boot path.
        from app.core.action_router import GENERATIVE_INTENTS

        threshold = float(self.config.get("confidence_threshold", 0.3))
        generative_threshold = float(
            self.config.get("generative_confidence_threshold", 0.2)
        )

        def _gate_for(action: str) -> float:
            return generative_threshold if action in GENERATIVE_INTENTS else threshold

        results = [
            {
                "action": action,
                "confidence": round(min(score, 1.0), 3),
                "keywords_matched": [
                    kw for kw in merged_patterns.get(action, [])
                    if _matches_keyword(kw, message)
                ],
            }
            for action, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
            if score >= _gate_for(action)
        ]
        # "do not run clash detection" still contains the word clash.
        # Negated phrasing is not a clash request (leftover project-assistant).
        if not message_wants_clash(message):
            results = [r for r in results if r["action"] != "bim_clash_detection"]
        # Contract Data Q&A (Time for Completion, milestones, delay damages)
        # is RAG on project-assistant — "milestone" / "completion" must not
        # dispatch generate_wbs or parse_primavera_schedule.
        if message_is_contract_data_lookup(message):
            results = [
                r for r in results
                if r["action"] not in CONTRACT_LOOKUP_BLOCKED_ACTIONS
            ]
        if message_wants_answer_report(message):
            results = [
                r for r in results
                if r["action"] not in ANSWER_REPORT_BLOCKED_ACTIONS
            ]
        from app.core.conversation_wbs import message_wants_wbs_export
        if message_wants_wbs_export(message):
            results = [r for r in results if r["action"] != "generate_wbs"]
        from app.core.action_router import message_wants_look_ahead
        if message_wants_look_ahead(message):
            # Schedule nouns ("construction schedule", "programme", "xer")
            # outscore the look-ahead cue and dispatch generate_wbs /
            # parse_primavera. Promote look_ahead; drop the WBS builder.
            results = [r for r in results if r["action"] != "generate_wbs"]
            if not any(r["action"] == "look_ahead" for r in results):
                results.insert(0, {
                    "action": "look_ahead",
                    "confidence": 0.4,
                    "keywords_matched": ["look-ahead"],
                })
            else:
                results.sort(
                    key=lambda r: (
                        0 if r["action"] == "look_ahead" else 1,
                        -float(r.get("confidence") or 0.0),
                    )
                )

        from app.core.action_router import message_wants_cash_flow
        if message_wants_cash_flow(message):
            # "schedule" / "programme" nouns outscore cash_flow_forecast and
            # dispatch generate_wbs (live tip 50c37f ask1). Promote cash flow;
            # drop the WBS builder.
            results = [r for r in results if r["action"] != "generate_wbs"]
            if not any(r["action"] == "cash_flow_forecast" for r in results):
                results.insert(0, {
                    "action": "cash_flow_forecast",
                    "confidence": 0.4,
                    "keywords_matched": ["cash_flow_forecast"],
                })
            else:
                results.sort(
                    key=lambda r: (
                        0 if r["action"] == "cash_flow_forecast" else 1,
                        -float(r.get("confidence") or 0.0),
                    )
                )

        from app.core.action_router import message_wants_rfi_draft
        if message_wants_rfi_draft(message):
            # Drawing S-### / A-### refs outscore rfi_generator onto
            # drawing_qto (live tip 50c37f ask1 → tools=[] RAG-miss).
            # Clash + RFI must not collapse onto cde_post_rfi.
            results = [
                r for r in results
                if r["action"] not in {
                    "drawing_qto",
                    "bim_clash_detection",
                    "extract_quantities",
                    "cde_post_rfi",
                }
            ]
            if not any(r["action"] == "rfi_generator" for r in results):
                results.insert(0, {
                    "action": "rfi_generator",
                    "confidence": 0.4,
                    "keywords_matched": ["rfi_generator"],
                })
            else:
                results.sort(
                    key=lambda r: (
                        0 if r["action"] == "rfi_generator" else 1,
                        -float(r.get("confidence") or 0.0),
                    )
                )
        return self._apply_correctness_filters(message, results)

    def _apply_correctness_filters(
        self, message: str, results: List[Dict]
    ) -> List[Dict]:
        """Chat-path guards: no default claim, no alias collapse, site vocab."""
        from app.core.delay_advice import claims_builder_permitted
        from app.core.site_vocab import (
            message_is_site_instruction_not_vo,
            message_issues_ncr,
            message_issues_pc_cert,
            message_issues_stop_work,
            message_reports_slipped_delivery,
            message_wants_clash_cde_rfi,
            message_wants_ifc_drawings,
            message_wants_vo_draft,
        )

        if not claims_builder_permitted(message) or message_reports_slipped_delivery(message):
            results = [r for r in results if r["action"] != "claims_builder"]
        if message_issues_ncr(message):
            results = [r for r in results if r["action"] != "qa_qc_inspection"]
        if message_issues_stop_work(message):
            results = [r for r in results if r["action"] != "safety_compliance_audit"]
        if message_issues_pc_cert(message):
            results = [
                r for r in results
                if r["action"] not in {"commissioning_checklist", "payment_certificate"}
            ]
        if message_wants_ifc_drawings(message):
            results = [r for r in results if r["action"] != "bim_analysis"]
        if message_is_site_instruction_not_vo(message):
            results = [
                r for r in results
                if r["action"] not in {"change_order_impact", "variation_order_manager"}
            ]
        if message_wants_vo_draft(message):
            # Live Phase 2 ask1: "variation" scored change_order_impact first
            # and the turn never drafted. Prefer variation_order_manager.
            results = [
                r for r in results if r["action"] != "change_order_impact"
            ]
            vo = [
                r for r in results if r["action"] == "variation_order_manager"
            ]
            rest = [
                r for r in results if r["action"] != "variation_order_manager"
            ]
            if not vo:
                vo = [{
                    "action": "variation_order_manager",
                    "confidence": 0.8,
                    "keywords_matched": ["variation order"],
                }]
            results = vo + rest
        # Tool-shaped / draft RFI asks stay on rfi_generator (clash/rebar/lap OK).
        # The CDE collapse is for "post this clash RFI to Aconex/CDE" only.
        from app.core.action_router import message_wants_rfi_draft
        if (
            message_wants_clash(message)
            and message_wants_clash_cde_rfi(message)
            and not message_wants_rfi_draft(message)
        ):
            results = [r for r in results if r["action"] != "rfi_generator"]
            if not any(r["action"] == "cde_post_rfi" for r in results):
                results.append({
                    "action": "cde_post_rfi",
                    "confidence": 0.4,
                    "keywords_matched": ["clash rfi"],
                })
        return results

    def _detect_file_type(self, data: Dict, context: Dict) -> Optional[str]:
        for key in ("file_path", "filename", "file"):
            val = data.get(key) or context.get(key, "")
            if val:
                ext = "." + str(val).rsplit(".", 1)[-1].lower() if "." in val else ""
                if ext in FILE_TYPE_MAP:
                    return ext
        return None

    def _detect_parallel_group(self, actions: List[str]) -> Optional[str]:
        action_set = set(actions)
        for group_name, group_actions in PARALLEL_GROUPS.items():
            if len(action_set & set(group_actions)) >= 2:
                return group_name
        return None

    def _list_actions(self) -> Dict:
        unique = list(dict.fromkeys(a for a, _ in ACTION_PATTERNS))
        return {
            "status": "success",
            "action_queue": [],
            "parallel_flag": False,
            "fallback_agent": "intelligent_workflow",
            "all_actions": unique,
            "total_actions": len(unique),
            "parallel_groups": PARALLEL_GROUPS,
            "file_type_routing": FILE_TYPE_MAP,
        }


# Compile every keyword pattern once, at import: compiled lazily, the first
# requests after a deploy compiled hundreds of patterns on the event loop
# (live: a 627 ms stall at 15 users).
for _action, _keywords in ACTION_PATTERNS:
    for _kw in _keywords:
        _kw_pattern(_kw)
