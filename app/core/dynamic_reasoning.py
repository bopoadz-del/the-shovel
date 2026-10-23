"""Dynamic UNDERSTAND — the orchestrator's LLM-driven intent read.

Replaces the brittle keyword classifier for the workflow decision: the LLM reads
the message and returns which known workflow it is (if any) and whether the user
wants a DELIVERABLE or an ANSWER. This is the "dynamic reasoning over predefined
routes" the operator asked for — the model understands "how long is procurement"
is a QUESTION, which keyword routing (procurement_list_generator) cannot.

Runs on the configured cloud ladder (DeepSeek primary, OpenRouter fallback).
Bounded output: a known workflow name or "none", so a bad/hallucinated read
simply falls through to the dynamic agent — never invents a step.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from app.core.llm_client import complete_json

logger = logging.getLogger(__name__)

# Semantic workflow -> smart_orchestrator action (WORKFLOW_REGISTRY key).
WORKFLOW_TO_ACTION = {
    "schedule": "generate_wbs",
}

_SYSTEM = (
    "You are the intent router for a construction project platform. Read the "
    "user's message and return ONLY a JSON object, no prose:\n"
    '{"workflow": "<schedule|none>", "mode": "<produce|question>", "params": {}}\n'
    "- workflow \"schedule\": the user wants a construction schedule / WBS / "
    "programme BUILT (generate, create, L2 schedule, N activities).\n"
    "- workflow \"none\": anything else, including Contract Data lookups "
    "(Time for Completion, milestones, delay damages).\n"
    "- mode \"produce\": wants a deliverable created/exported (produce, generate, "
    "create, build, export, make, prepare, draft a schedule).\n"
    "- mode \"question\": is ASKING about a schedule (how long, what, when, which, "
    "status, explain) — NOT requesting a file.\n"
    "- params (optional): target_count (int), project_type (string), start_date "
    "(YYYY-MM-DD) only if the user states them.\n"
    "Examples:\n"
    '"produce a 300 activity data center schedule" -> '
    '{"workflow":"schedule","mode":"produce","params":{"target_count":300,"project_type":"data_center"}}\n'
    '"how long is procurement on the schedule?" -> '
    '{"workflow":"schedule","mode":"question","params":{}}\n'
    '"what is the Time for Completion for the whole of the Works?" -> '
    '{"workflow":"none","mode":"question","params":{}}\n'
    '"what is the tallest building" -> {"workflow":"none","mode":"question","params":{}}'

)


def _intent_model() -> Optional[str]:
    """Return ORCHESTRATOR_INTENT_MODEL only when the active provider can serve it.

    Cloud prod is DeepSeek (+ OpenRouter fallback). A colon-tag id
    (``name:tag``, e.g. ``gpt-oss:20b-cloud``) is not a DeepSeek/OpenRouter
    model and 400s every chat turn — the exception is swallowed, so
    predefined schedule routing never fires. Ignore such an override.
    """
    override = (os.getenv("ORCHESTRATOR_INTENT_MODEL") or "").strip()
    if not override:
        return None
    if ":" in override:
        logger.warning(
            "ORCHESTRATOR_INTENT_MODEL=%r ignored: not a DeepSeek/OpenRouter "
            "model id; using the provider default so intent routing does not "
            "400 every chat turn",
            override,
        )
        return None
    return override


async def understand_intent(message: str, has_documents: bool = False) -> Dict[str, Any]:
    """Return {action, mode, params, workflow} for a message, or workflow
    'none' (action None) when it is not a known workflow. Never raises."""
    empty = {"workflow": "none", "action": None, "mode": "question", "params": {}}
    if not message or not message.strip():
        return empty
    from app.core.contract_lookup_intent import message_is_contract_data_lookup
    from app.core.answer_report_intent import message_wants_answer_report
    from app.core.action_router import (
        message_wants_cash_flow,
        message_wants_look_ahead,
    )
    if message_is_contract_data_lookup(message):
        return empty
    if message_wants_answer_report(message):
        return empty
    if message_wants_look_ahead(message):
        from app.containers.construction.schedule import _stated_look_ahead_date
        stated = _stated_look_ahead_date(message)
        return {
            "workflow": "look_ahead",
            "action": "look_ahead",
            "mode": "produce",
            "deliverable": True,
            "params": {"as_of": stated} if stated else {},
        }
    if message_wants_cash_flow(message):
        # Live tip 53b8b29 ask1: "Use cash_flow_forecast. Synthetic
        # schedule/costs…" still went predefined generate_wbs after #669
        # because UNDERSTAND maps "schedule" + "Produce" to workflow=schedule.
        return {
            "workflow": "cash_flow_forecast",
            "action": "cash_flow_forecast",
            "mode": "produce",
            "deliverable": True,
            "params": {},
        }
    from app.core.conversation_wbs import message_wants_wbs_export
    if message_wants_wbs_export(message):
        return empty
    user = message.strip()
    if has_documents:
        user += "\n\n(Note: the user has attached document(s) to this project.)"
    try:
        # Intent routing is cheap work — pin it to a lighter model when
        # ORCHESTRATOR_INTENT_MODEL is set to an id the ACTIVE provider can
        # serve. Unset -> provider default (DeepSeek in cloud prod).
        model = _intent_model()
        # Fail FAST: a per-turn router must never hang a chat for 2 minutes on a
        # slow/broken cloud LLM. Short timeout -> exception -> fall through to
        # the agent path. Configurable via ORCHESTRATOR_INTENT_TIMEOUT.
        try:
            tmo = float(os.getenv("ORCHESTRATOR_INTENT_TIMEOUT") or "20")
        except ValueError:
            tmo = 20.0
        if os.getenv("AGENT_TIMING_LOG") == "1":
            import time as _t, logging as _lg
            _t0 = _t.monotonic()
            out = await complete_json(_SYSTEM, user, max_tokens=300, model=model, timeout=tmo)
            _lg.getLogger("app.core.dynamic_reasoning").warning(
                "TIMING understand_intent call=%.1fs model=%s", _t.monotonic() - _t0, model)
        else:
            out = await complete_json(_SYSTEM, user, max_tokens=300, model=model, timeout=tmo)
    except Exception:
        return empty
    workflow = str(out.get("workflow") or "none").lower()
    action = WORKFLOW_TO_ACTION.get(workflow)
    if not action:
        return empty
    mode = str(out.get("mode") or "question").lower()
    params = out.get("params") if isinstance(out.get("params"), dict) else {}
    return {
        "workflow": workflow,
        "action": action,
        "mode": mode,
        "deliverable": mode == "produce",
        "params": params,
    }
