"""Action → domain hint translation for the chat router.

The `smart_orchestrator` block returns action strings like `boq_process` or
`spec_analyze` based on keyword matches. The chat router uses this module to
turn those action strings into a short domain-aware system prompt that biases
the LLM toward the right kind of answer.

Why a hint, not a block dispatch:
  - Most actions returned by the orchestrator (`tender_bid_analysis`,
    `claims_builder`, `safety_compliance_audit`, ...) don't have a 1:1 block
    in BLOCK_REGISTRY — only ~9 of 39 do.
  - The blocks that exist (`boq_processor`, `spec_analyzer`, `bim`, ...)
    expect structured input (file paths, parameters), not a freeform chat
    message. Dispatching a raw user message to them would fail in ways the
    user can't recover from.
  - The chat block is already the right thing to answer freeform
    questions about uploaded files — it just needs a nudge about *which*
    domain the user is asking inside of.

So this module returns a one-sentence system-prompt fragment per action.
Unknown / weak matches return None, and the chat router falls through to
plain chat unchanged.
"""

import re
from typing import Optional


# Per-action system-prompt hint. Each value tells the LLM what domain frame to
# answer inside. Kept short so it never overwhelms the user's question.
ACTION_HINTS: dict[str, str] = {
    # BOQ / Cost
    "boq_process":
        "The user is asking about a Bill of Quantities. Focus on line items, "
        "quantities, unit rates, totals, and any cost structure visible in the file.",
    "extract_quantities":
        "The user wants quantity takeoff data — areas, counts, volumes. "
        "Extract concrete numbers with units; flag any missing measurements.",
    "estimate_costs":
        "The user wants a cost estimate. Use any uploaded BOQ / pricing data "
        "to give specific figures (with currency) and a clear total.",
    "tender_bid_analysis":
        "The user is analysing tender/bid documents. Focus on price comparisons, "
        "scope coverage, conditions, and discrepancies between bids.",
    "procurement_list_generator":
        "The user wants a procurement / material list. Group by trade or "
        "material category, give quantities + unit, leave price columns blank "
        "if no rate data is present.",
    "procurement_optimizer":
        "The user wants procurement optimisation. Compare suppliers, identify "
        "cheapest viable options, flag any quality/lead-time tradeoffs.",
    "payment_certificate":
        "The user is dealing with a payment certificate / valuation. Focus on "
        "work done to date, retention, materials on site, and net amount due.",
    "cash_flow_forecast":
        "The user wants a cash flow / S-curve projection. Walk through phasing, "
        "monthly drawdowns, and the timing of payments.",
    # Specifications
    "spec_analyze":
        "The user is asking about a technical specification. Identify "
        "applicable standards (ASTM, ACI, SASO, etc.), grade requirements, "
        "and compliance checks.",
    "process_specification_full":
        "The user wants a structured walk-through of a full specification "
        "section. Organise by clause/sub-clause and call out compliance items.",
    # Drawings
    "drawing_qto":
        "The user is taking off quantities from drawings (DXF/DWG/floor plan). "
        "Focus on areas, lengths, counts, and call out measurement assumptions.",
    # Schedule
    "parse_primavera_schedule":
        "The user is reviewing a Primavera P6 / .xer schedule. Focus on "
        "activities, durations, critical path, and milestones.",
    "progress_tracker":
        "The user wants a progress / actual-vs-planned analysis. Identify "
        "completion %, slippages, and recovery options.",
    "evm_calculate":
        "The user wants classic Earned Value Management (PV/EV/AC → SPI, CPI, "
        "SV, CV, and EAC/ETC/VAC when BAC is given). Do not invent actuals.",
    "resource_histogram":
        "The user wants a resource histogram (manpower / crew loading). "
        "Time-phase by week or month and flag peaks/troughs.",
    "look_ahead":
        "The user wants a 3–4 week look-ahead from the uploaded Primavera "
        ".xer. List activities overlapping the window with ES/EF, float, "
        "and critical flag — never fabricate activities.",
    "forensic_delay_analysis":
        "The user is building a forensic delay / EOT analysis. Focus on "
        "as-planned vs as-built, concurrent delays, and entitlement basis.",
    # BIM / IFC
    "bim_analysis":
        "The user is reviewing a BIM / IFC model. Focus on building elements, "
        "quantities by type, and any model-data quality issues.",
    "bim_clash_detection":
        "The user is asking about BIM clashes. Focus on hard/soft clashes by "
        "discipline (MEP/structure/architecture) and priority.",
    "bim_extract":
        "The user wants quantities extracted from a BIM model. Group by "
        "element type with totals; include units.",
    "digital_twin_sync":
        "The user is asking about digital-twin / as-built model sync. Focus on "
        "deltas between design and built state.",
    # QA/QC
    "qa_qc_inspection":
        "The user is dealing with QA/QC, inspections, NCRs, or punch lists. "
        "Focus on defects, root cause if known, and required corrective actions.",
    "commissioning_checklist":
        "The user wants a commissioning / handover checklist. Organise by "
        "system (HVAC, electrical, plumbing, ...) with pre/post tests.",
    # Contracts / Claims
    "process_contract":
        "The user is reviewing a contract. Focus on parties, scope, "
        "key clauses (FIDIC / NEC where applicable), and risk allocation.",
    "change_order_impact":
        "The user wants change-order / variation impact analysis. Walk through "
        "cost impact, time impact, and contract basis.",
    "variation_order_manager":
        "The user is managing variation orders. Track status, value, and "
        "approval state per VO.",
    "claims_builder":
        "The user is building a construction claim. Identify the event, basis "
        "in contract, loss/expense quantum, and time entitlement.",
    "rfi_generator":
        "The user wants help drafting an RFI. State the issue clearly, the "
        "drawings/specs involved, and the question to designer. Do not treat "
        "the chat bubble as the RFI register — Aconex / the CDE is the system "
        "of record.",
    "cde_post_rfi":
        "The user wants the drafted RFI posted to Aconex / the CDE. Post the "
        "draft via the CDE connector. Do not allocate a Fork RFI number and "
        "do not keep a local RFI log.",
    "job_requisition":
        "The user wants a job requisition / prequalification pack. Draft "
        "scope, NOC constraints, and a shortlist rubric from operator facts.",
    "rfp_draft":
        "The user wants an RFP / invitation to tender. Write the invitation, "
        "scope, prequalification, evaluation method, and key dates.",
    "rfp_management":
        "The user wants an RFP / invitation to tender. Write the invitation, "
        "scope, prequalification, evaluation method, and key dates.",
    "as_built_deviation_report":
        "The user wants an as-built deviation note. State planned vs actual "
        "figures and the shortfall; do not invent drawing dimensions.",
    # Safety
    "safety_briefing":
        "The user wants a live-haul-road / public-interface safety briefing. "
        "Cover required signage, speed control, and pedestrian crossings "
        "from the named diversion drawings. Do not invent an approved TMP.",
    "safety_compliance_audit":
        "The user is doing a safety / HSE compliance check. Reference OSHA "
        "or local regs where applicable; focus on hazards and PPE.",
    "risk_register_auto_populate":
        "The user is populating a risk register. Each risk: likelihood, "
        "impact, owner, mitigation, residual rating.",
    # Sustainability
    "carbon_footprint_calculator":
        "The user wants embodied / operational carbon calculations. Use "
        "EPDs where available; output in tCO2e with assumptions stated.",
    "esg_sustainability_report":
        "The user is preparing an ESG / sustainability report. Use LEED / "
        "BREEAM categories if appropriate.",
    # Reports
    "daily_site_report":
        "The user wants a daily site report. Include manpower, weather, "
        "work done, materials received, incidents, photos referenced.",
    "submittal_log_generator":
        "The user is managing a submittal log. Track item, spec section, "
        "submitted-on, approved-on, status.",
    "as_built_deviation_report":
        "The user wants an as-built deviation report. Compare to design, "
        "list deviations with rationale and approval refs.",
    "warranty_maintenance_schedule":
        "The user wants a warranty / planned maintenance schedule. Organise "
        "by asset/system with frequency.",
    "om_manual_generator":
        "The user wants an O&M manual outline. Standard sections: "
        "operation, maintenance, spares, contacts.",
    # Value / Analysis
    "value_engineering":
        "The user wants value-engineering options. List alternatives with "
        "cost/quality tradeoffs and lifecycle implications.",
    "sympy_reason":
        "The user wants symbolic / variance analysis. Show formulas, "
        "substitutions, and the final numeric result.",
    # Documents
    "process_document":
        "The user wants help analysing the uploaded document. Be specific "
        "about what's in it; don't ask for clarification if the content is clear.",
    # Reasoning / AI
    "intelligent_workflow":
        "The user wants a multi-step workflow. Lay out the plan, then "
        "execute step-by-step.",
    "health_check":
        "The user is asking about system status. Be brief and factual.",
    # Procedure routing (catalogue kinds) — metadata-first; delegate when runnable.
    "design_review_workflow":
        "The user is asking about design review / acceptance. "
        "Never use 'APPROVED' on design documents — use accepted/for comment/buy-off.",
    "rfi_management":
        "The user is managing RFIs. Focus on open/overdue counts, "
        "required-by dates, and factual responses — RFIs are not design changes.",
    "ncr_management":
        "The user is raising or tracking NCRs. Cover disposition, "
        "corrective action, and closure — do not invent NCR numbers.",
    "handover_management":
        "The user is asking about practical completion / handover. "
        "Cover prerequisites, DLP, snag lists, and O&M without fabricating certs.",
    "inspection_request":
        "The user wants an inspection request / WIR. Write the full form: "
        "scope, mix/volume, hold and witness points, and signatories. Do not stop "
        "at 'let me search'. Do not claim an IR was issued without data.",
    "wir_form":
        "The user wants a Work Inspection Request. Fill the WIR from operator facts "
        "(location, mix, volume, hold/witness). Do not search-loop the template.",
}


# Minimum orchestrator confidence score before we apply a hint. Below this,
# the match is too speculative — fall through to plain chat.
#
# Sized for the orchestrator's word-boundary matcher: 0.4 means "at least one
# multi-word match (e.g. 'bill of quantities' → 3 * 0.2 = 0.6, well over)" OR
# "two independent single-word matches (e.g. 'aci' + 'specification' → 0.4)".
# The previous 0.6 was tuned against the old substring matcher whose scores
# were inflated by spurious substring hits ('spec' inside 'specification', etc.).
HINT_CONFIDENCE_THRESHOLD = 0.4


# ─────────────────────────────────────────────────────────────────────────────
# Intent-based routing: chat → heavy-reasoning agent
# ─────────────────────────────────────────────────────────────────────────────
#
# Some user intents need *multi-step* generative reasoning, not a single LLM
# turn over the prompt. "Create a 200-activity schedule", "Generate a
# procurement list", "Run the forensic delay analysis" — these are NOT
# answerable by prepending file content to a chat block. They need:
#   - tool calls (search_project_documents, construction.generate_wbs,
#     formula_executor_v2)
#   - iterative refinement
#   - validation
#
# The runtime `heavy-reasoning` agent already supports all of this. When the
# orchestrator classifies a message into one of the GENERATIVE_INTENTS below
# at sufficient confidence, the chat router routes the message there instead
# of the fast single-shot ChatBlock path.
#
# Fast-path intents (everything else) stay on the chat block — single LLM
# call, sub-2s latency, fine for Q&A like "what is the total cost".

# Confidence threshold for routing.
#
# Originally 0.5 (stricter than the 0.4 hint threshold) on the theory that
# routing is more invasive (5-15s vs sub-2s) and should require firmer
# evidence. In practice the orchestrator's keyword matcher gives the real-
# world phrasing "Create a 200 activities schedule" exactly 0.4 — so 0.5
# made routing unreachable for the canonical use case we built it for.
#
# At 0.4 the whitelist (GENERATIVE_INTENTS) does the safety work: a 0.4
# match against a non-generative action (e.g. "process_document") still
# stays on the fast path because the action isn't whitelisted. Only
# genuinely generative intents with at least one multi-word OR two single-
# word keyword matches reach the heavy-reasoning path.
ROUTING_CONFIDENCE_THRESHOLD = 0.4

# Gate-2 for actions in GENERATIVE_INTENTS only. Operator brief PR #80:
# the global 0.4 threshold made the literal phrasing "generate a WBS for
# a 10-floor tower" unreachable — the single matched keyword "wbs" scores
# 0.2 (1 word), gets boosted past gate-1 by the matching relaxed gate in
# smart_orchestrator._match_actions, but used to be blocked here. Lowered
# to 0.2 so generative intents route from sparse phrasings; non-generative
# matches still need 0.4 (preserved by leaving ROUTING_CONFIDENCE_THRESHOLD
# as the global constant for any caller that consults it directly).
GENERATIVE_ROUTING_THRESHOLD = 0.2

# Generative intents — the ones that require multi-step reasoning or
# synthesis. Keep this list short and only include actions whose answer
# legitimately needs tool calls, NOT every action that has a hint.
GENERATIVE_INTENTS = frozenset({
    # Multi-step / programmatic generation
    "generate_wbs",                # create a WBS / schedule activity list
    "intelligent_workflow",        # generic multi-step workflow
    "sympy_reason",                # symbolic / variance analysis pipeline
    # Schedule analysis (real CPM, not just prose answer)
    "parse_primavera_schedule",
    "forensic_delay_analysis",
    "resource_histogram",
    "look_ahead",
    "evm_calculate",
    # Cost/financial workflows
    "cash_flow_forecast",
    "estimate_costs",
    "procurement_list_generator",
    "procurement_optimizer",
    "payment_certificate",
    # Specs / standards full pipeline
    "process_specification_full",
    # BIM full pipeline
    "bim_analysis",
    "bim_clash_detection",
    # The container action is `bim_extract` — the orchestrator briefly emitted
    # the non-existent name `bim_extractor`, which routed but could never
    # deliver (2026-07-24 seam sweep). tests/test_capability_seams.py pins this.
    "bim_extract",
    # Digital-twin payload prep is a real implementation that previously only
    # ever produced a chat hint (in no delivery gate). progress_tracker is
    # deliberately NOT here — tests/test_routing_matrix.py pins it fast-path.
    "digital_twin_sync",
    # Drawing quantity take-off — real tool dispatch, not RAG-only prose.
    # Without this, "extract quantities from the floor plan" stays on the
    # fast chat path and the model has no way to read the DXF/vector PDF,
    # so it refuses to answer instead of calling drawing_qto. Surfaced as
    # the lone FAIL in QA v3 (2026-06-17).
    "drawing_qto",
    # Contracts / Claims (real document synthesis)
    "claims_builder",
    "rfi_generator",
    "cde_post_rfi",
    "inspection_request",
    "wir_form",
    "job_requisition",
    "rfp_draft",
    "rfp_management",
    "as_built_deviation_report",
    "safety_briefing",
    "value_engineering",
    "change_order_impact",
    "variation_order_manager",
})


# Rolling / N-day look-ahead from a programme. These share schedule-like
# nouns ("programme", "construction schedule", "master schedule") with
# generate_wbs; the keyword scorer then picks WBS. Detector is the steal-guard.
_LOOKAHEAD_PHRASES = (
    "look ahead", "look-ahead", "lookahead",
    "3 week look", "4 week look", "three week look", "four week look",
    "2 week look", "2-week look", "two week look",
    "rolling look ahead", "rolling look-ahead",
    "short term programme", "short-term programme",
    "short term program", "short-term program",
)
_LOOKAHEAD_WINDOW_RE = re.compile(
    r"\b(?:\d+|two|three|four)\s*[-]?\s*(?:day|week)s?\s+[-]?\s*look"
    r"|\blook\s*[-]?\s*ahead\b"
    r"|\blookahead\b"
    r"|\brolling\s+look"
    r"|\bshort[-\s]term\s+program(?:me)?\b",
    re.IGNORECASE,
)
_LOOKAHEAD_QA_RE = re.compile(
    r"\b(what is|what's|whats|explain|define)\b",
    re.IGNORECASE,
)
# "what falls in the next 14 days" / "over the next 3 weeks" never says
# look-ahead. Digits and the same week-words the window parser accepts.
_NEXT_WINDOW_NUM = r"(?:\d+|two|three|four)"
_NEXT_WINDOW_UNIT = r"(?:day|week)s?"
_NEXT_WINDOW_RE = re.compile(
    rf"\b(?:in|within|over|during|for)\s+the\s+next\s+{_NEXT_WINDOW_NUM}"
    rf"\s*[-]?\s*{_NEXT_WINDOW_UNIT}\b"
    rf"|\bnext\s+{_NEXT_WINDOW_NUM}\s*[-]?\s*{_NEXT_WINDOW_UNIT}\b"
    rf"|\b(?:coming|upcoming)\s+up\s+over\s+the\s+next\s+{_NEXT_WINDOW_NUM}"
    rf"\s*[-]?\s*{_NEXT_WINDOW_UNIT}\b",
    re.IGNORECASE,
)
# Imperative WBS / schedule produce. "Create a look-ahead programme" does
# not match: the noun is not the token right after the article.
_WBS_PRODUCE_RE = re.compile(
    r"\b(?:wbs|work\s+breakdown(?:\s+structure)?)\b"
    r"|\b(?:l[1-4]|level\s+[1-4])\s+schedule\b"
    r"|\b\d{2,4}\s+activities\b"
    r"|\b(?:generate|create|build|produce|draft|make|develop|prepare)\s+"
    r"(?:a\s+|an\s+|the\s+)?(?:wbs|schedule|programme|program)\b",
    re.IGNORECASE,
)


def _next_window_look_ahead(raw: str) -> bool:
    """True for a next-N-days/weeks question that is not some other deliverable.

    Checked before the definition veto. 'What's coming up…' and 'what is
    due in the next 14 days' contain 'what is' / 'what's' and would
    otherwise be thrown out with 'what is a look-ahead?'.
    """
    if not _NEXT_WINDOW_RE.search(raw):
        return False
    if _WBS_PRODUCE_RE.search(raw):
        return False
    if message_wants_cash_flow(raw):
        return False
    if message_wants_rfi_draft(raw):
        return False
    if "histogram" in raw.lower():
        return False
    from app.core.contract_lookup_intent import message_is_contract_data_lookup
    if message_is_contract_data_lookup(raw):
        return False
    return True


def message_wants_look_ahead(text: str) -> bool:
    """True when the turn asks for a rolling / N-day look-ahead, not a WBS.

    Live Phase 2: "14-day look-ahead from the synthetic programme" and
    "look-ahead from the construction schedule" were classified as
    generate_wbs because those schedule nouns score higher than "look-ahead".

    A live look-ahead ask: "As of today, 21 September 2026, what falls in the next 14
    days?" never says look-ahead, so the phrase guard let UNDERSTAND map
    it to generate_wbs. A next-N-days/weeks question is the same ask.
    """
    raw = text or ""
    if not raw.strip():
        return False
    if _next_window_look_ahead(raw):
        return True
    if _LOOKAHEAD_QA_RE.search(raw):
        return False
    low = raw.lower()
    if any(p in low for p in _LOOKAHEAD_PHRASES):
        return True
    if _LOOKAHEAD_WINDOW_RE.search(raw):
        return True
    return "look" in low and "ahead" in low and any(
        t in low for t in (
            ".xer", "primavera", "p6", "schedule", "programme", "program",
        )
    )



# Cash-flow / S-curve asks share "schedule" / "programme" nouns with
# generate_wbs; the keyword scorer then picks WBS. Detector is the steal-guard.
_CASH_FLOW_PHRASES = (
    "cash_flow_forecast",
    "cash_flow",
    "cash-flow",
    "cash flow",
    "cashflow",
    "s-curve",
    "s curve",
    "spend curve",
    "drawdown",
)
_CASH_FLOW_QA_RE = re.compile(
    r"\b(what is|what's|whats|explain|define)\b",
    re.IGNORECASE,
)
# Dewatering "drawdown" is the water-table drop (required_drawdown_m on
# dewatering_well_point_spacing), not an S-curve. A cash-flow phrase still
# wins when both appear.
_DEWATERING_DRAWDOWN_RE = re.compile(
    r"well[_\s\-]?points?|dewater|permeabilit|water[_\s\-]?table",
    re.IGNORECASE,
)
_CASH_FLOW_EXPLICIT_RE = re.compile(
    r"cash[_\- ]?flow|cashflow|s-curve|s curve|spend curve",
    re.IGNORECASE,
)


def dewatering_drawdown_not_cash_flow(text: str) -> bool:
    """True when ``drawdown`` is a dewatering input, not a cash-flow curve."""
    raw = text or ""
    if not re.search(r"drawdown", raw, re.IGNORECASE):
        return False
    if _CASH_FLOW_EXPLICIT_RE.search(raw):
        return False
    return bool(_DEWATERING_DRAWDOWN_RE.search(raw))


def message_wants_cash_flow(text: str) -> bool:
    """True when the turn asks for a cash-flow / S-curve, not a WBS.

    Live tip 50c37f: "Use cash_flow_forecast. Synthetic schedule/costs…"
    classified as generate_wbs because schedule nouns outscored cash flow
    (orchestrator keywords lacked the underscore tool name).

    A well-point / permeability ask that names drawdown is dewatering, not
    this detector. Matching it locked the turn and skipped construction_calc.
    """
    raw = text or ""
    if not raw.strip() or _CASH_FLOW_QA_RE.search(raw):
        return False
    if dewatering_drawdown_not_cash_flow(raw):
        return False
    low = raw.lower()
    return any(p in low for p in _CASH_FLOW_PHRASES)


# RFI drafts often name Drawing S-### / A-### refs; drawing_qto keywords
# then win. Detector is the steal-guard (live tip 50c37f ask1).
_RFI_DRAFT_PHRASES = (
    "rfi_generator",
    "request for information",
    "follow-on rfi",
    "follow on rfi",
    "draft a rfi",
    "draft an rfi",
    "draft one rfi",
    "generate a rfi",
    "generate an rfi",
    "create a rfi",
    "create an rfi",
    "raise an rfi",
    "raise a rfi",
    "write an rfi",
    "write a rfi",
    "produce an rfi",
    "produce a rfi",
    "issue an rfi",
    "issue a rfi",
)
_RFI_QA_RE = re.compile(
    r"\b(what is|what's|whats|explain|define|how many)\b",
    re.IGNORECASE,
)


def message_wants_rfi_draft(text: str) -> bool:
    """True for an RFI-draft deliverable, not a drawing_qto / definition ask.

    Live tip 50c37f: "Use rfi_generator. Draft ONE RFI … Drawing S-201"
    classified as drawing_qto then RAG-miss refused. Clash + the word RFI
    is still a draft, not a CDE post.
    """
    raw = text or ""
    if not raw.strip() or _RFI_QA_RE.search(raw):
        return False
    low = raw.lower()
    return any(p in low for p in _RFI_DRAFT_PHRASES)


def needs_planning(action: Optional[str], confidence: float) -> bool:
    """True iff this orchestrator classification warrants the heavy-reasoning
    agent path instead of the fast single-shot chat block.

    Uses the relaxed GENERATIVE_ROUTING_THRESHOLD (0.2) rather than the
    global ROUTING_CONFIDENCE_THRESHOLD (0.4) so sparse generative
    phrasings like "generate a WBS for a 10-floor tower" route to
    heavy-reasoning. The whitelist (GENERATIVE_INTENTS) does the safety
    work: non-generative actions still return False even at high
    confidence. Ambiguous classifications fall through to the fast path
    with a domain hint.
    """
    if not action:
        return False
    if action not in GENERATIVE_INTENTS:
        return False
    return confidence >= GENERATIVE_ROUTING_THRESHOLD


def hint_for_action(action: str) -> Optional[str]:
    """Return the LLM system-prompt hint for a routed action, or None."""
    return ACTION_HINTS.get(action)


def best_action(orchestrator_result: dict) -> tuple[Optional[str], float]:
    """Pull the highest-confidence action out of a SmartOrchestratorBlock result.

    Returns (action_name, confidence). When the orchestrator returns nothing
    routable, returns (None, 0.0). SmartOrchestratorBlock emits matches as
    {"action": str, "confidence": float, "keywords_matched": [str]} ordered
    by descending confidence.
    """
    matched = orchestrator_result.get("matched_actions") or []
    if not matched:
        return None, 0.0
    top = matched[0]
    return top.get("action"), float(top.get("confidence") or 0.0)


def hint_for_orchestrator_result(orchestrator_result: dict) -> Optional[str]:
    """End-to-end: orchestrator result → hint sentence (or None).

    Returns None when the top match is below the confidence threshold or when
    the matched action has no registered hint *and* there is no CM prompt
    inject / follow-up context to surface.

    When smart_orchestrator attaches ``cm_prompt_inject`` (from
    CrossDomainReasoner.inject_prompt) or high-confidence follow-ups, those
    are folded into the hint so the chat path actually uses reasoner output
    beyond metadata — without a second router.
    """
    parts: list[str] = []
    action, score = best_action(orchestrator_result)
    if action and score >= HINT_CONFIDENCE_THRESHOLD:
        base = hint_for_action(action)
        if base:
            parts.append(base)

    cm_inject = (orchestrator_result.get("cm_prompt_inject") or "").strip()
    if cm_inject:
        # Keep inject compact — chat piggybacks on the user message.
        parts.append(cm_inject[:800])

    follow_ups = orchestrator_result.get("cm_follow_up_tools") or []
    appended = orchestrator_result.get("cm_queue_appended") or []
    usable = appended or follow_ups[:2]
    if usable:
        parts.append(
            "Consider related construction actions next: "
            + ", ".join(usable)
            + "."
        )

    template = orchestrator_result.get("cm_matched_template")
    if template and orchestrator_result.get("cm_workflow_plan"):
        parts.append(
            f"A multi-domain workflow template matched ({template}); "
            "prefer chaining across its domains when answering."
        )

    if not parts:
        return None
    return " ".join(parts)
