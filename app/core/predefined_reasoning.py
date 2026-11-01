"""Predefined reasoning — the orchestrator block's own PLAN -> EXECUTE ->
DELIVER path for KNOWN workflows (absorbed from project_reasoner).

For a known workflow the step list is a fixed template, so no LLM planning call
is needed — deterministic, auditable, no step hallucination. The orchestrator
block calls `run_workflow`; unknown actions return handled=False so the block
falls through to a dynamic agent.

DELIVER is intent-gated: a request to PRODUCE/EXPORT materializes a document
(render_artifact is in the plan); a question answers inline from the staged
summary (no artifact). This is the fix to "produce a document to answer a
question" — the plan composition, not "a tool ran", decides.
"""
from __future__ import annotations

import re
import logging
from typing import Any, Dict, List, Optional, Tuple

from app.schemas.execution_plan import ExecutionPlan, PlanStep
from app.schemas.project_session import ProjectSession
from app.core.plan_executor import PlanExecutor

logger = logging.getLogger(__name__)


# Scoped Dispatch v2 — the EXPLICIT allowlist of container actions eligible for
# reasoner-driven predefined dispatch. These are deliverable-PRODUCING actions
# (they generate a register / schedule / certificate / report from project data)
# where deterministic container execution + LLM synthesis beats the agent tool
# loop's short answer. Q&A / lookup / document-processing / RAG-grounded actions
# (chat, process_document, process_contract, process_specification_full,
# benchmark_lookup, recommend, learn, analyze_spec, spec_analyze, health_check,
# status) are DELIBERATELY excluded: they must stay on the RAG-grounded agent
# path, never be intercepted. Membership is explicit, not a heuristic — adding an
# action here is a reviewed decision.
SCOPED_DISPATCH_ALLOWLIST = frozenset({
    "cash_flow_forecast",
    "resource_histogram",
    "look_ahead",
    "evm_calculate",
    "risk_register_auto_populate",
    "procurement_list_generator",
    "procurement_optimizer",
    "commissioning_checklist",
    "rfi_generator",
    "cde_post_rfi",
    "submittal_log_generator",
    "payment_certificate",
    "change_order_impact",
    "value_engineering",
    "claims_builder",
    "variation_order_manager",
    "warranty_maintenance_schedule",
    "as_built_deviation_report",
    "om_manual_generator",
    "esg_sustainability_report",
    "carbon_footprint_calculator",
    "safety_compliance_audit",
    "wir_form",
    "inspection_request",
    "job_requisition",
    "rfp_draft",
    "rfp_management",
    "tender_bid_analysis",
    "daily_site_report",
    "forensic_delay_analysis",
})


def container_action_names() -> set:
    """Construction-container actions eligible for reasoner-driven predefined
    dispatch: the explicit SCOPED_DISPATCH_ALLOWLIST intersected with the actions
    the loaded container actually implements. Empty when the kit isn't loaded.
    Never returns Q&A/lookup actions even if they exist on the container."""
    try:
        from app.dependencies import get_block_instance
        con = get_block_instance("construction")
        impl = set(con.get_actions().keys()) if hasattr(con, "get_actions") else set()
        return impl & SCOPED_DISPATCH_ALLOWLIST
    except Exception:  # noqa: BLE001 -- reported loudly below, never silently
        # A container that does not load takes every one of its workflows
        # out of reach. That must be visible: logged as an error here and
        # counted by /health (workflows_reachable), never an empty set quietly.
        logging.getLogger(__name__).exception(
            "construction container did not load: its workflows are unreachable")
        return set()


def workflows_reachable() -> int:
    """How many predefined workflows the model can reach right now."""
    return len(set(WORKFLOW_REGISTRY) | container_action_names())

# Verbs that mean "I want the artifact", not "answer my question".
_DELIVERABLE_RE = re.compile(
    r"\b(produce|export|generate|create|build|make|prepare|develop|draft|"
    r"give me|download|issue|render)\b", re.IGNORECASE)


def is_deliverable_request(message: str) -> bool:
    return bool(_DELIVERABLE_RE.search(message or ""))


_INTERROGATIVE_RE = re.compile(
    r"^\s*(what|what's|when|within|how|who|whose|where|which|why|does|do|did|"
    r"is|are|was|were|can|could|should|would|will|has|have)\b", re.IGNORECASE)


# A DEFINITION question asks what a concept IS. It names the concept, which is
# exactly why the keyword classifier scores it HIGH: "What is a forensic delay
# analysis?" matches both "forensic" and "delay analysis" and lands at 0.6.
# So the confidence that was meant to signal "this user wants the tool run"
# is, for this shape, only signalling "this user said the tool's name".
_DEFINITIONAL_RE = re.compile(
    r"^\s*(?:"
    r"what\s+(?:is|are)\s+(?:a|an)\b"           # What is a/an X
    r"|what(?:'s|\s+is)\s+meant\s+by\b"        # What is meant by X
    r"|what\s+does\s+.+?\s+(?:mean|stand\s+for)\b"
    r"|(?:define|explain|describe)\b"
    r"|how\s+does\s+.+?\s+work\b"
    r"|(?:is|are)\s+.+?\s+the\s+same\s+as\b"
    r")"
    r"|\bdifference\s+between\b",
    re.IGNORECASE,
)

# ...unless it points at the user's own material. "Explain the delay on MY
# project" and "What is the critical path of THIS schedule?" have data behind
# them and a real tool to run; they are analysis, not definition.
_POINTS_AT_PROJECT_DATA_RE = re.compile(
    r"\b(?:this|these|that|those|my|our|uploaded|attached|above|below)\b"
    r"|\bthe\s+(?:project|schedule|programme|program|baseline|contract|boq|"
    r"bill|model|drawing|drawings|file|files|xer|tender|site)\b"
    r"|\.(?:xer|ifc|dxf|dwg|xlsx|pdf)\b",
    re.IGNORECASE,
)


def is_definition_question(message: str) -> bool:
    """True for "what is X" asked about a concept, not about project data.

    Reads the message only. It must never learn which project it was asked
    in: the corpus is one-to-all and a definition is a definition anywhere.
    """
    msg = (message or "").strip()
    if not msg or not _DEFINITIONAL_RE.search(msg):
        return False
    if _POINTS_AT_PROJECT_DATA_RE.search(msg):
        return False
    return not is_deliverable_request(msg)


# "How long will X take?" with no figures in it. A live bare-duration ask: six words
# routed to generate_wbs and came back "Schedule built: 204 activities over
# 688 working days" -- a whole programme invented for a scope nobody named.
# A duration ask that carries its own quantities and rates is a calculation
# (a productivity / rate duration ask) and is deliberately NOT matched here.
_BARE_DURATION_RE = re.compile(
    r"\bhow\s+long\b.{0,60}?\b(?:take|takes|last|complete|finish)\b"
    r"|\bhow\s+(?:many|much)\s+(?:working\s+|calendar\s+)?"
    r"(?:days?|weeks?|months?|time)\b.{0,60}?\b(?:take|takes|need|require)\b"
    r"|\bwhat\s+is\s+the\s+duration\s+of\b",
    re.IGNORECASE | re.DOTALL,
)
_ANY_FIGURE_RE = re.compile(r"\d")


def is_bare_duration_question(message: str) -> bool:
    """True for "how long will the foundations take?" -- a scope question.

    Bare means it states no quantity and no rate: there is nothing to compute
    with, so any duration produced is a duration for a scope the operator
    never specified. A figure anywhere in the message takes it off this path
    and back to the ordinary calculation route.
    """
    msg = (message or "").strip()
    if not msg or not _BARE_DURATION_RE.search(msg):
        return False
    if _ANY_FIGURE_RE.search(msg):
        return False
    return not is_deliverable_request(msg)


def lookup_question_hijack(message: str, confidence: float) -> bool:
    """True when a LOW-confidence route is about to intercept a lookup QUESTION.

    2026-07-24 golden-gate repro: "Within how many days must the Contractor
    give notice of an EOT claim on this project?" — the "eot" keyword routed
    this to forensic_delay_analysis at confidence 0.2, which errored asking
    for XER files; the RAG path (which holds the answer in the project's own
    contract note) never ran. The allowlist's contract is explicit: Q&A stays
    on the RAG-grounded path. High-confidence routes and deliverable-verbed
    messages ("generate...", "calculate...") are untouched.

    Contract Data TfC / milestone lookups are always hijacks, even at high
    keyword confidence — "milestone" matching parse_primavera_schedule at
    0.4 must not run generate_wbs.
    """
    from app.core.contract_lookup_intent import message_is_contract_data_lookup
    from app.core.answer_report_intent import message_wants_answer_report
    from app.core.action_router import message_wants_look_ahead
    if message_is_contract_data_lookup(message):
        return True
    if message_wants_answer_report(message):
        return True
    if message_wants_look_ahead(message):
        return False
    from app.core.conversation_wbs import message_wants_wbs_export
    if message_wants_wbs_export(message):
        return True
    # Checked BEFORE the confidence cut-off, deliberately. For every other
    # shape a confident route is evidence the user wants the tool. For a
    # definition it is only evidence they named it -- see _DEFINITIONAL_RE.
    if is_definition_question(message):
        return True
    # Also before the cut-off, and for the same reason: a confident route is
    # evidence the user wants the tool for every shape EXCEPT one that has
    # nothing for the tool to work from. The live bare-duration ask was routed at high
    # confidence and answered with an invented 204-activity programme.
    if is_bare_duration_question(message):
        return True
    if confidence >= 0.5:
        return False
    msg = (message or "").strip()
    if not msg:
        return False
    question_shaped = bool(_INTERROGATIVE_RE.match(msg)) or msg.endswith("?")
    return question_shaped and not is_deliverable_request(msg)


def build_schedule_plan(context: Dict[str, Any]) -> ExecutionPlan:
    """A work breakdown structure and baseline schedule for a whole project
    brief, as an exportable deliverable. Fixed compute template: extract (if
    docs) -> build_wbs -> cost_load. No render step here — materialization is an
    intent-gated export descriptor built in run_workflow (endpoints are the
    render mechanism / API calls)."""
    p = context.get("params") or {}
    doc_ids = context.get("document_ids") or []
    steps: List[PlanStep] = []
    if doc_ids:
        steps.append(PlanStep(type="extract_document", args={"document_ids": doc_ids},
                              description="extract equipment lead times + milestones"))
    from app.lib.wbs_duration_overrides import collect_overrides
    msg = context.get("message", "") or ""
    duration_overrides = p.get("duration_overrides") or collect_overrides(msg)
    steps.append(PlanStep(type="build_wbs", args={
        "brief": msg,
        "target_count": p.get("target_count", 200),
        "project_type": p.get("project_type"),
        "start_date": p.get("start_date"),
        "user_message": msg,
        "duration_overrides": duration_overrides or None,
        "project_id": context.get("project_id") or p.get("project_id"),
        "boq_items": p.get("boq_items") or context.get("boq_items"),
    }, description="build the WBS (+ long-lead procurement)"))
    steps.append(PlanStep(type="cost_load", args={
        "project_name": context.get("project_name") or "Project",
        "currency": p.get("currency") or "",
        "day_rate": p.get("day_rate"),
        "crew_per_trade": p.get("crew_per_trade", 4),
        "start_date": p.get("start_date"),
    }, description="cost-load the schedule (man-days S-curve, milestones)"))
    return ExecutionPlan(understanding="produce/answer a construction schedule", steps=steps)


# Live leftover F1: a WBS / work-breakdown ask used to return only
# "Schedule built: N activities…" + a workbook offer. The tree was already
# staged on session.data["wbs"]["wbs_tree"]; DELIVER just never printed it.
_WBS_OUTLINE_ASK_RE = re.compile(
    r"\b(?:wbs|work\s+breakdown(?:\s+structure)?|"
    r"high[- ]level\s+(?:breakdown|outline|structure)|"
    r"breakdown\s+of\s+(?:the\s+)?(?:work|scope|works))\b",
    re.IGNORECASE,
)


def message_wants_wbs_outline(message: str) -> bool:
    """True when the turn asked to see the WBS hierarchy, not only a schedule."""
    return bool(_WBS_OUTLINE_ASK_RE.search(message or ""))


# Live BOQ-scope WBS ask after #521: the hierarchy rendered, but generate_wbs still
# A WBS over a project's measured BOQ scope must consume the retrieved rows.
# Contract Data particulars, spec-precedence and delay-damages asks stay off
# this path.
def message_wants_boq_scope_wbs(message: str) -> bool:
    """True for a high-level WBS over a project's measured BOQ scope, ANY trade.

    Gate on STRUCTURE, not keywords (N8, ORDERS_28OCT_0815Z): the ask must be a
    WBS-outline ask AND a bill-of-quantities ask. There is no demolition/
    site-clearance keyword requirement and no ``BOQ_SCOPE_WBS`` kill-switch —
    any measured BOQ scope elects the WBS path. generate_wbs derives the WBS
    from the rows; there is no template scaffold to fall back to.
    """
    msg = message or ""
    if not message_wants_wbs_outline(msg):
        return False
    from app.core.rag.retriever import query_asks_for_boq_scope
    return query_asks_for_boq_scope(msg)


def format_wbs_outline(wbs_tree: Any, *, title: str = "High-level WBS") -> str:
    """Render numbered WBS codes as a hierarchical outline.

    ``wbs_tree`` maps ``"1"`` → phase, ``"1.1"`` → package. Only codes already
    in the tree are printed — this does not invent names or levels.
    """
    if not isinstance(wbs_tree, dict) or not wbs_tree:
        return ""

    def _sort_key(code: str) -> tuple:
        parts: list = []
        for piece in str(code).split("."):
            try:
                parts.append(int(piece))
            except (TypeError, ValueError):
                parts.append(piece)
        return tuple(parts)

    rows: list[tuple[str, str]] = []
    for raw_code, raw_name in wbs_tree.items():
        code = str(raw_code).strip()
        name = str(raw_name).strip() if raw_name is not None else ""
        if not code or not name:
            continue
        rows.append((code, name))
    if not rows:
        return ""
    rows.sort(key=lambda item: _sort_key(item[0]))

    lines = [f"## {title}", ""]
    for code, name in rows:
        # Phase rows (no dot) as headings so the glass shows levels, not a flat list.
        if "." not in code:
            lines.append(f"### {code} {name}")
        else:
            lines.append(f"{code} {name}")
    return "\n".join(lines).strip()


def deliver_schedule(session: ProjectSession, deliverable: bool,
                     message: str = "") -> str:
    """Deterministic DELIVER: a written answer from the staged summary. (LLM
    polish is optional and can wrap this later — the numbers are fixed.)

    A WBS / work-breakdown ask also surfaces the staged ``wbs_tree`` as a
    numbered hierarchy. The tree is what ``generate_wbs`` already returned;
    this does not invent corpus content.
    """
    s = session.data.get("schedule_summary") or {}
    if not s:
        return "I could not build the schedule for this request."
    parts = [
        f"Schedule built: {s.get('activities')} activities over "
        f"{s.get('duration_days')} working days "
        f"({s.get('critical_count')} on the critical path).",
    ]
    overrides = s.get("duration_overrides_applied") or []
    if overrides:
        bits = ", ".join(
            f"{o.get('match')} = {o.get('days')} days "
            f"({o.get('activities_updated')} activities)"
            for o in overrides if isinstance(o, dict)
        )
        if bits:
            parts.append(f"User duration override applied and CPM recomputed: {bits}.")
    if s.get("procurement_injected"):
        parts.append(
            f"{s['procurement_injected']} long-lead procurement item(s) were "
            "injected from the source documents' real lead times, each linked "
            "ahead of its install.")
    if s.get("total_man_days"):
        parts.append(f"Total effort is about {s['total_man_days']:,} man-days.")
    ms = s.get("target_milestones") or []
    if ms:
        named = ", ".join(f"{m.get('name')} ({m.get('target_date')})" for m in ms[:5] if m.get('name'))
        if named:
            parts.append(f"Target milestones from the source: {named}.")
    if deliverable:
        parts.append(
            "The cost-loaded workbook (CPM, cumulative man-days S-curve, "
            "manpower histogram, milestones) is ready to download.")
    summary = " ".join(parts)

    wbs = session.data.get("wbs") if isinstance(session.data.get("wbs"), dict) else {}
    tree = wbs.get("wbs_tree") if isinstance(wbs.get("wbs_tree"), dict) else {}
    scaffold = wbs.get("scaffold") if isinstance(wbs.get("scaffold"), dict) else {}
    if message_wants_wbs_outline(message) and tree:
        blocks: list[str] = []
        declaration = scaffold.get("declaration")
        # Template-scaffold wording is honest only when the tree is a scaffold.
        if (
            declaration
            and scaffold.get("derived_from_boq")
            and "template scaffold" in str(declaration).lower()
        ):
            declaration = ""
        if declaration:
            blocks.append(f"_{declaration}_")
        outline = format_wbs_outline(tree)
        if outline:
            blocks.append(outline)
        blocks.append(summary)
        return "\n\n".join(blocks)
    return summary


def _export_descriptor(context: Dict[str, Any]) -> Dict[str, Any] | None:
    """Intent-gated materialization: the export endpoint (the render API call)
    the deliverable should be produced from, on click. None when no project.

    When a conversation already has a staged WBS, the offer binds to that
    snapshot. Re-running schedule-from-brief from the current
    message used to serve the 204-activity building scaffold instead.
    """
    project_id = context.get("project_id")
    if not project_id:
        return None
    conversation_id = context.get("conversation_id")
    if conversation_id:
        from app.core.conversation_wbs import (
            conversation_schedule_export_descriptor,
            load_conversation_wbs,
        )
        staged = load_conversation_wbs(str(conversation_id))
        if staged:
            n = len(staged.get("activities") or [])
            return conversation_schedule_export_descriptor(
                project_id, str(conversation_id), n,
            )
    p = context.get("params") or {}
    doc_ids = context.get("document_ids") or []
    if doc_ids:
        endpoint = f"/v1/projects/{project_id}/export/schedule-from-document"
        payload: Dict[str, Any] = {"document_ids": doc_ids}
    else:
        endpoint = f"/v1/projects/{project_id}/export/schedule-from-brief"
        payload = {"brief": context.get("message", "")}
    payload.update({"target_count": p.get("target_count", 200)})
    for k in ("project_type", "start_date", "day_rate", "duration_overrides"):
        if p.get(k) is not None:
            payload[k] = p[k]
    if conversation_id:
        payload["conversation_id"] = conversation_id
    return {"label": "Schedule (Excel)", "format": "xlsx", "method": "POST",
            "endpoint": endpoint, "payload": payload}


def build_histogram_plan(context: Dict[str, Any]) -> ExecutionPlan:
    """Manpower histogram from a brief: build a norms-based WBS, then
    time-phase its crews into periods. Used when the request carries no
    resource-loaded .xer — the deliver text labels the result as a planning
    estimate, never as site resource returns (the container action's
    no-fabrication contract stays intact for the schedule-file path)."""
    p = context.get("params") or {}
    msg = context.get("message", "") or ""
    unit = "month" if re.search(r"\bmonth", msg, re.IGNORECASE) else "week"
    steps = [
        PlanStep(type="build_wbs", args={
            "brief": msg,
            "target_count": p.get("target_count", 150),
            "project_type": p.get("project_type"),
            "start_date": p.get("start_date"),
            "user_message": msg,
            "duration_overrides": p.get("duration_overrides") or None,
        }, description="build a norms-based WBS from the brief"),
        PlanStep(type="resource_histogram", args={"period_unit": unit},
                 description="time-phase the WBS crews into periods"),
    ]
    return ExecutionPlan(understanding="produce a manpower histogram", steps=steps)


def deliver_histogram(session: ProjectSession, deliverable: bool,
                      message: str = "") -> str:
    """Deterministic DELIVER for the histogram plan — numbers straight from
    the session, provenance stated up front."""
    h = session.data.get("manpower") or {}
    periods = h.get("periods") or []
    if not periods:
        return "I could not build the manpower histogram for this request."
    parts = [
        "Manpower histogram — norms-derived from a generated WBS for this "
        "brief (no resource-loaded P6 schedule was provided; these are "
        "planning estimates, not site resource returns).",
    ]
    for per in periods[:24]:
        parts.append(f"- {per.get('label')}: {int(per.get('total') or 0)} workers")
    if h.get("peak_total"):
        parts.append(
            f"Peak manpower {int(h['peak_total'])} in {h.get('peak_period')}; "
            f"about {int(h.get('total_manhours') or 0):,} man-hours overall."
        )
    if deliverable:
        parts.append(
            "The cost-loaded workbook (CPM, S-curve, manpower histogram, "
            "milestones) is ready to download.")
    return "\n".join(parts)


# action (from smart_orchestrator) -> plan builder
WORKFLOW_REGISTRY = {
    "generate_wbs": build_schedule_plan,
    "resource_histogram": build_histogram_plan,
}

# action -> DELIVER renderer for registry plans (default: the schedule text)
DELIVER_REGISTRY = {
    "generate_wbs": deliver_schedule,
    "resource_histogram": deliver_histogram,
}


def _render_container_result(action: str, result: Any) -> str:
    """Turn a container action's return value into a human-readable answer.
    Prefers an explicit answer/summary/markdown/report field; otherwise renders
    the salient fields so the deliverable is substantial rather than a stub."""
    import json as _json
    if isinstance(result, str):
        return result
    if not isinstance(result, dict):
        return str(result)
    for k in ("answer", "markdown", "report", "summary", "text", "narrative"):
        v = result.get(k)
        if isinstance(v, str) and v.strip():
            return v
    # Build a readable rendering from the structured payload.
    title = action.replace("_", " ").title()
    lines = [f"# {title}", ""]
    for k, v in result.items():
        if k in ("status", "action", "ok"):
            continue
        if isinstance(v, (str, int, float, bool)):
            lines.append(f"- **{k.replace('_',' ')}:** {v}")
        elif isinstance(v, list) and v:
            lines.append(f"\n## {k.replace('_',' ').title()} ({len(v)})")
            for item in v[:40]:
                if isinstance(item, dict):
                    lines.append("- " + "; ".join(f"{ik}: {iv}" for ik, iv in list(item.items())[:6]))
                else:
                    lines.append(f"- {item}")
        elif isinstance(v, dict) and v:
            lines.append(f"\n## {k.replace('_',' ').title()}")
            for ik, iv in list(v.items())[:30]:
                lines.append(f"- **{ik}:** {iv}")
    rendered = "\n".join(lines).strip()
    return rendered or _json.dumps(result, indent=2, default=str)


# ── Grounding gate (Phase-2 2a) ─────────────────────────────────────────────
# Post-synthesis, FLAG-only checks. The synthesis prompt already TELLS the LLM to
# ground every figure in the tool result, but nothing ENFORCES it. These checks
# add honest enforcement WITHOUT ever blocking or regenerating a deliverable — a
# heuristic must not suppress a real answer. Flag-gated so it can be turned off
# instantly in prod. Soundness: this is correct because predefined-dispatch
# synthesis receives ONLY the container `result`; if RAG chunks are ever added to
# the synthesis input, the backing/source sets below MUST expand to cover them.
import os as _os

def _grounding_gate_enabled() -> bool:
    return _os.getenv("GROUNDING_GATE", "1") not in ("0", "false", "False", "")

# A confidence/validation STAMP asserted in prose ("confidence: high",
# "validation passed", "quality-assured"). Deliberately NARROW — bare
# "verify"/"validated" appears in legitimate checklist content, so only
# stamp-shaped phrases are matched to keep false-positives near zero.
_CONF_RATING = r"(?:very\s+high|high|moderate|medium|low|strong|\d{1,3}\s*%|0?\.\d\d?)"
_CONFIDENCE_STAMP_RE = re.compile(
    # rating on either side of the word: "confidence: high" OR "95% confidence".
    rf"\bconfidence\b[\s:*_()-]{{0,4}}{_CONF_RATING}"
    rf"|{_CONF_RATING}[\s-]{{0,4}}\bconfidence\b",
    re.IGNORECASE,
)
_VALIDATION_STAMP_RE = re.compile(
    r"\b(validation\s+(passed|complete|completed|successful|status)|"
    r"independently\s+validated|cross[- ]validated|quality[- ]assured|"
    r"verified\s+against\s+(the\s+)?(tool|engine|calculation|model))\b",
    re.IGNORECASE,
)
# Result keys that legitimately BACK a confidence/validation stamp. When any of
# these is present in the structured result, a stamp in the prose is honest and
# is left untouched (e.g. drawing_qto / historical_benchmark carry real
# confidence / source_note; validation_pipeline results carry pass/checks).
_STAMP_BACKING_KEYS = frozenset({
    "confidence", "confidence_score", "confidence_level", "confidence_interval",
    "validation", "validated", "verified", "validation_report",
    "validation_status", "source_note", "checks", "pass", "passed",
    "quality_check", "qa", "overall_pass", "stage_results",
})

def _result_has_backing_field(result: Any) -> bool:
    """True iff the structured result carries any confidence/validation-style
    field that would legitimately back a stamp in the prose. Walks the whole
    result (dicts + lists), case-insensitive on keys."""
    stack = [result]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            for k, v in cur.items():
                if str(k).lower() in _STAMP_BACKING_KEYS:
                    return True
                stack.append(v)
        elif isinstance(cur, list):
            stack.extend(cur)
    return False

def _gate_confidence_stamps(text: str, result: Any) -> str:
    """FLAG (append a caveat) when the prose asserts a validation/confidence
    stamp that the computed result does NOT back with a corresponding field.
    Never edits a stamp the result actually backs; never blocks the answer."""
    if not (_CONFIDENCE_STAMP_RE.search(text) or _VALIDATION_STAMP_RE.search(text)):
        return text
    if _result_has_backing_field(result):
        return text
    return text + (
        "\n\n> _Note: this response mentions validation/confidence, but the "
        "computed result carries no validation or confidence field — treat any "
        "such rating as unverified._"
    )

# ── Grounding gate increment 2 — money/rate figure grounding ────────────────
# NARROWLY scoped to money/rate-shaped figures ONLY (currency symbol / SAR|USD…
# / rate unit like "/day", "per m³"). Deliberately does NOT check bare integers
# (counts, weeks, years) — a derived count like "42 trades" is not currency-
# shaped, so it is never checked and never false-flagged. A fabricated
# "SAR 1,200/day" that appears in a financial deliverable but nowhere in the
# computed result is exactly what this catches.
_CURRENCY = r"(?:SAR|USD|AED|EUR|GBP|QAR|KWD|OMR|BHD|\$|€|£)"
_RATE_UNIT = (r"(?:/|per\s+)\s*(?:day|month|week|year|hour|hr|man-?hour|m3|m²|m2|"
              r"m³|sqm|sq\s?m|kg|tonne|ton|t|lm|l\.?m|no\.?|unit|each|ea)")
_NUM = r"\d[\d,]*(?:\.\d+)?"
# money figure with the currency on either side, OR a number followed by a rate unit.
_MONEY_FIG_RE = re.compile(
    rf"{_CURRENCY}\s*({_NUM})"                      # $ 1,200  / SAR 300,000
    rf"|({_NUM})\s*{_CURRENCY}"                     # 300,000 SAR
    rf"|({_NUM})\s*{_RATE_UNIT}",                   # 1,200 /day  / 55 per m³
    re.IGNORECASE,
)
_NUM_IN_TEXT_RE = re.compile(_NUM)

def _to_number(s: str):
    try:
        return float(str(s).replace(",", "").strip())
    except (ValueError, TypeError):
        logger.warning("could not parse grounding number %r", s, exc_info=True)
        return None

def _collect_result_numbers(result: Any) -> set:
    """Every numeric value anywhere in the result — real fields AND numbers
    embedded in result strings (e.g. 'IPC: SAR 120,000' in a summary). The LLM
    only saw the payload truncated to 6000 chars, so matching against the FULL
    result strictly REDUCES false-positives."""
    nums: set = set()
    stack = [result]
    while stack:
        cur = stack.pop()
        if isinstance(cur, bool):
            continue
        if isinstance(cur, (int, float)):
            nums.add(round(float(cur), 2))
        elif isinstance(cur, str):
            for m in _NUM_IN_TEXT_RE.findall(cur):
                v = _to_number(m)
                if v is not None:
                    nums.add(round(v, 2))
        elif isinstance(cur, dict):
            stack.extend(cur.values())
        elif isinstance(cur, (list, tuple)):
            stack.extend(cur)
    return nums

def _is_grounded_number(value: float, result_nums: set) -> bool:
    """value grounds if some result number matches within a rounding tolerance
    (0.5 absolute or 0.5% relative — covers '300,000.0' vs '300000' and the
    LLM rounding 120,000.50 -> 120,001)."""
    tol = max(0.5, abs(value) * 0.005)
    return any(abs(value - r) <= tol for r in result_nums)

def _gate_money_figures(text: str, result: Any) -> str:
    """FLAG (append a caveat) money/rate figures in the prose that do NOT trace
    to any number in the computed result — treat them as unverified estimates.
    FLAG-only; never blocks. Bare non-currency numbers are never checked."""
    result_nums = _collect_result_numbers(result)
    ungrounded: list = []
    for m in _MONEY_FIG_RE.finditer(text):
        raw = next((g for g in m.groups() if g), None)
        v = _to_number(raw)
        if v is None or v == 0:
            continue
        if not _is_grounded_number(v, result_nums):
            frag = m.group(0).strip()
            if frag not in ungrounded:
                ungrounded.append(frag)
    if not ungrounded:
        return text
    listed = "; ".join(ungrounded[:8])
    return text + (
        "\n\n> _Note: the following figure(s) could not be traced to the computed "
        f"result and should be treated as estimates, not calculated values: {listed}._"
    )

def _apply_grounding_gate(text: str, result: Any) -> str:
    """Run the enabled post-synthesis grounding checks (FLAG-only)."""
    if not _grounding_gate_enabled():
        return text
    for check in (_gate_confidence_stamps, _gate_money_figures):
        try:
            text = check(text, result)
        except Exception:  # noqa: BLE001 — a gate must never break a deliverable
            logger.exception("grounding gate %s failed; skipping it", getattr(check, "__name__", "?"))
    return text


async def _synthesize_answer(action: str, result: Any, message: Optional[str]) -> str:
    """LLM synthesis of a container action's structured result into a grounded,
    professional narrative. Scoped Dispatch v2 BANS raw-render as the delivered
    answer — the deterministic render is kept only as an emergency fallback if
    the LLM call fails, so we never emit nothing. The LLM is instructed to ground
    every figure strictly in the tool result (no invention)."""
    raw = _render_container_result(action, result)
    try:
        import json as _json
        from app.core import llm_client
        payload = (_json.dumps(result, default=str)[:6000]
                   if isinstance(result, (dict, list)) else str(result)[:6000])
        system = (
            "You are a construction project assistant. A deterministic tool has "
            "already run and produced the structured result below. Write a clear, "
            "professional answer for a construction PM that presents that result. "
            "Ground EVERY figure, name and date strictly in the tool result - do "
            "not invent, estimate, or add facts not present. Be concise, use "
            "markdown. If the result is empty or an error, say so plainly."
        )
        user = (f"User request: {message or action}\n\nTool: {action}\n"
                f"Structured result (JSON):\n{payload}")
        text = await llm_client.complete(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0, max_tokens=900,
        )
        return _apply_grounding_gate(text.strip() or raw, result)
    except Exception:
        logger.exception("synthesis for %s failed; using deterministic render", action)
        return _apply_grounding_gate(raw, result)


async def _run_container_action(action: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Scoped reasoner-driven dispatch: if `action` is in the explicit
    SCOPED_DISPATCH_ALLOWLIST (deliverable-producing) AND implemented by the
    container, run it via the container's own execute() and LLM-synthesize the
    result. Returns None otherwise (so the caller keeps its handled=False
    fallthrough to the RAG-grounded agent path). Q&A / lookup / document actions
    are never in the allowlist, so they are never intercepted here."""
    if action not in SCOPED_DISPATCH_ALLOWLIST:
        return None
    try:
        from app.dependencies import get_block_instance
        con = get_block_instance("construction")
    except Exception:
        logger.warning("construction block unavailable for action %s", action, exc_info=True)
        return None
    try:
        actions = set(con.get_actions().keys()) if hasattr(con, "get_actions") else set()
    except Exception:
        actions = set()
    if action not in actions:
        return None
    envelope: Dict[str, Any] = {
        "action": action,
        "project_id": context.get("project_id"),
        "message": context.get("message"),
        "document_ids": context.get("document_ids") or [],
    }
    envelope.update({k: v for k, v in (context.get("params") or {}).items() if v is not None})
    if action == "payment_certificate":
        try:
            from app.containers.construction.boq import ipc_args_from_ask
            filled = ipc_args_from_ask(context.get("message") or "", envelope)
            envelope.update({k: v for k, v in filled.items() if v is not None})
        except Exception:  # noqa: BLE001
            logger.debug("IPC ask coercion skipped", exc_info=True)
    try:
        result = await con.execute(envelope)
    except Exception as e:  # noqa: BLE001
        logger.exception("generic container action %s failed", action)
        return {"handled": True, "status": "error",
                "answer": f"The {action.replace('_',' ')} step could not complete: {e}",
                "deliverable": context.get("deliverable"), "exports": [], "plan_steps": [action]}
    answer = await _synthesize_answer(action, result, context.get("message"))
    exports = result.get("exports", []) if isinstance(result, dict) else []
    return {
        "handled": True,
        "status": (result.get("status") if isinstance(result, dict) else None) or "success",
        "answer": answer,
        "deliverable": context.get("deliverable"),
        "export": exports[0] if exports else None,
        "exports": exports,
        "plan_steps": [action],
    }


async def run_workflow(action: str, context: Dict[str, Any],
                       session: ProjectSession) -> Dict[str, Any]:
    """Run the predefined workflow for `action`. Returns handled=False for
    unknown actions so the orchestrator falls through to a dynamic agent."""
    builder = WORKFLOW_REGISTRY.get(action)
    if builder is None:
        # No bespoke plan builder — try the generic container dispatch. This is
        # what makes every container-implemented action a first-class,
        # reasoner-driven deliverable instead of falling to a short chat answer.
        generic = await _run_container_action(action, context)
        if generic is not None:
            return generic
        return {"handled": False}
    # Prefer the dynamic UNDERSTAND verdict (context["deliverable"]) when the
    # orchestrator supplied it; fall back to the keyword heuristic otherwise.
    deliverable = (context["deliverable"] if "deliverable" in context
                   else is_deliverable_request(context.get("message", "")))
    plan = builder(context)
    run = await PlanExecutor().run(plan, session)
    deliver = DELIVER_REGISTRY.get(action, deliver_schedule)
    answer = deliver(session, deliverable, context.get("message") or "")
    wbs = session.data.get("wbs") if isinstance(session.data.get("wbs"), dict) else {}
    cid = context.get("conversation_id")
    if cid and wbs:
        from app.core.conversation_wbs import stage_conversation_wbs
        stage_conversation_wbs(str(cid), wbs)
    export = _export_descriptor(context) if deliverable else None
    return {
        "handled": True,
        "status": run.status,
        "answer": answer,
        "deliverable": deliverable,
        "export": export,
        "exports": [export] if export else [],
        "plan_steps": [st.type for st in plan.steps],
    }
