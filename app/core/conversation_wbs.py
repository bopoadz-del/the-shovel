"""Conversation-scoped WBS / schedule snapshot.

After a chat-built WBS (e.g. a BOQ-scope WBS with 2 BOQ-derived
activities) the export path used to re-run ``generate_wbs`` from the
export ask or from ``schedule-from-brief`` without the conversation's
staged activities. That silently served the 204-activity building
template. The snapshot here is the bind: export that WBS, or fail
honestly. Nothing marks a template scaffold as the conversation WBS.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_EXPORT_VERB_RE = re.compile(
    r"\b(export|download|save|send\s+me|give\s+me)\b",
    re.IGNORECASE,
)
_WBS_SCOPE_RE = re.compile(
    r"\b(wbs|work\s+breakdown(?:\s+structure)?|schedule|workbook)\b",
    re.IGNORECASE,
)
# Produce / generate still means "build a new WBS", not "export the last one".
_BUILD_VERB_RE = re.compile(
    r"\b(generate|produce|create|build|make|prepare|develop|draft)\b",
    re.IGNORECASE,
)
_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9._-]+")

_WBS_ANSWER_MARKERS = (
    "schedule built:",
    "high-level wbs",
    "work breakdown structure",
    "template scaffold",
    "boq-derived",
    "derived from this project's boq",
)


def conversation_wbs_dir() -> Path:
    root = Path(os.getenv("DATA_DIR", "./data")) / "conversation_wbs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _safe_conversation_id(conversation_id: str) -> str:
    raw = (conversation_id or "").strip()
    if not raw:
        raise ValueError("conversation_id is required")
    safe = _SAFE_ID_RE.sub("_", raw)[:200]
    if not safe or safe in {".", ".."}:
        raise ValueError("conversation_id is not a usable file key")
    return safe


def staged_wbs_path(conversation_id: str) -> Path:
    return conversation_wbs_dir() / f"{_safe_conversation_id(conversation_id)}.json"


def message_wants_wbs_export(text: str) -> bool:
    """True when the turn wants the already-built WBS as a file.

    Positive: the live 'Export ... WBS as xlsx' ask; 'download the WBS as excel';
    'export this schedule as xlsx'.

    Negative: 'generate a high-level WBS…' (build, not export);
    'produce the schedule'; answer-range report exports; BOQ-to-Excel.
    """
    raw = text or ""
    if not raw.strip():
        return False
    from app.core.action_router import message_wants_look_ahead
    if message_wants_look_ahead(raw):
        return False
    from app.core.answer_report_intent import message_wants_answer_report
    if message_wants_answer_report(raw):
        return False
    if not _EXPORT_VERB_RE.search(raw):
        return False
    if _BUILD_VERB_RE.search(raw):
        return False
    if not _WBS_SCOPE_RE.search(raw):
        return False
    return True


def message_looks_like_wbs_answer(text: str) -> bool:
    """True when an assistant turn is the glass for a generated WBS."""
    t = (text or "").lower()
    return any(marker in t for marker in _WBS_ANSWER_MARKERS)


def is_outline_wbs_code(value: Any) -> bool:
    """True for chat-style codes (``1``, ``1.1``, ``2.1.3``), not phase slugs."""
    raw = str(value or "").strip()
    if not raw:
        return False
    return all(part.isdigit() for part in raw.split("."))


def hierarchical_wbs_code(
    activity: dict[str, Any] | None,
    wbs_tree: dict[str, Any] | None = None,
) -> str:
    """Package code the chat printed for this activity.

    Prefers an existing outline ``wbs_code`` / ``wbs``, then the longest
    ``wbs_tree`` key that prefixes the activity id (``1.1.3`` → ``1.1``).
    Sequential integers and phase slugs are never invented here.
    """
    if not isinstance(activity, dict):
        return ""
    aid = str(activity.get("id") or activity.get("code") or "").strip()
    existing = str(activity.get("wbs_code") or activity.get("wbs") or "").strip()
    tree_keys = [
        str(k).strip() for k in (wbs_tree or {}) if str(k).strip()
    ]
    tree_keys.sort(key=lambda k: (len(k.split(".")), k), reverse=True)
    if is_outline_wbs_code(existing) and (
        not tree_keys or existing in (wbs_tree or {}) or existing == aid
    ):
        return existing
    for key in tree_keys:
        if aid == key or aid.startswith(f"{key}."):
            return key
    if is_outline_wbs_code(aid):
        parts = aid.split(".")
        return ".".join(parts[:2]) if len(parts) >= 2 else aid
    return existing or aid


def stamp_hierarchical_wbs_codes(
    activities: list[dict[str, Any]] | None,
    wbs_tree: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Write chat ``wbs_tree`` codes onto each activity as ``wbs`` / ``wbs_code``."""
    stamped: list[dict[str, Any]] = []
    for raw in activities or []:
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        code = hierarchical_wbs_code(row, wbs_tree)
        if code:
            row["wbs"] = code
            row["wbs_code"] = code
        stamped.append(row)
    return stamped


def snapshot_from_wbs(wbs: dict[str, Any] | None) -> Optional[dict[str, Any]]:
    """Compact, JSON-safe snapshot. None when there is nothing to export."""
    if not isinstance(wbs, dict):
        return None
    acts = wbs.get("activities")
    if not isinstance(acts, list) or not acts:
        return None
    scaffold = wbs.get("scaffold") if isinstance(wbs.get("scaffold"), dict) else {}
    summary = wbs.get("summary") if isinstance(wbs.get("summary"), dict) else {}
    tree = wbs.get("wbs_tree") if isinstance(wbs.get("wbs_tree"), dict) else {}
    acts = stamp_hierarchical_wbs_codes(acts, tree)
    return {
        "activities": acts,
        "wbs_tree": tree,
        "scaffold": scaffold,
        "summary": summary,
        "brief": wbs.get("brief") or "",
        "project_type": wbs.get("project_type"),
        "start_date": wbs.get("start_date"),
        "actual_count": wbs.get("actual_count") or len(acts),
        "duration_overrides_applied": wbs.get("duration_overrides_applied") or [],
        "target_milestones": wbs.get("target_milestones") or [],
    }


def stage_conversation_wbs(
    conversation_id: str | None,
    wbs: dict[str, Any] | None,
) -> Optional[dict[str, Any]]:
    """Persist the conversation's WBS. No-op when id or activities are missing."""
    if not conversation_id:
        return None
    snap = snapshot_from_wbs(wbs)
    if snap is None:
        return None
    # A BOQ-scope ask that fell through to the building template must not
    # become the conversation's exportable WBS.
    refusal = refuse_scaffold_for_boq_wbs_ask(
        str((wbs or {}).get("brief") or ""),
        str((wbs or {}).get("user_message") or (wbs or {}).get("brief") or ""),
        wbs,
    )
    if refusal:
        return None
    path = staged_wbs_path(conversation_id)
    path.write_text(json.dumps(snap, default=str), encoding="utf-8")
    return snap


def load_conversation_wbs(conversation_id: str | None) -> Optional[dict[str, Any]]:
    """Return the staged snapshot, or None when absent / unreadable / empty."""
    if not conversation_id:
        return None
    try:
        path = staged_wbs_path(conversation_id)
    except ValueError:
        logger.warning("refusing unusable conversation_id for WBS load")
        return None
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("could not read staged WBS for %s", conversation_id, exc_info=True)
        return None
    if not isinstance(raw, dict):
        return None
    acts = raw.get("activities")
    if not isinstance(acts, list) or not acts:
        return None
    return raw


def clear_conversation_wbs(conversation_id: str | None) -> bool:
    """Delete the staged snapshot. True when a file was removed."""
    if not conversation_id:
        return False
    try:
        path = staged_wbs_path(conversation_id)
    except ValueError:
        logger.warning("refusing unusable conversation_id for WBS clear")
        return False
    if not path.is_file():
        return False
    path.unlink()
    return True


def conversation_schedule_export_descriptor(
    project_id: str,
    conversation_id: str,
    activity_count: int = 0,
) -> dict[str, Any]:
    """SSE ``exports`` offer bound to this conversation's staged WBS."""
    from app.core.answer_report_intent import export_workspace_project_id

    scoped = export_workspace_project_id(project_id, conversation_id)
    label = (
        f"Schedule — {activity_count} activities (Excel)"
        if activity_count
        else "Schedule (Excel)"
    )
    return {
        "label": label,
        "format": "xlsx",
        "method": "POST",
        "endpoint": (
            f"/v1/projects/{scoped}/conversations/{conversation_id}"
            f"/export/schedule"
        ),
        "payload": {"conversation_id": conversation_id},
    }


def refuse_scaffold_for_boq_wbs_ask(
    brief: str,
    user_message: str,
    wbs: dict[str, Any] | None,
) -> Optional[str]:
    """Honest refusal when a BOQ-scope WBS was asked and the result is a template.

    Empty string / None means the result may be exported. A string is the
    422 detail — never serve the 204-activity building programme as that
    WBS.
    """
    from app.core.predefined_reasoning import message_wants_boq_scope_wbs

    if not (
        message_wants_boq_scope_wbs(brief or "")
        or message_wants_boq_scope_wbs(user_message or "")
    ):
        return None
    scaffold = (wbs or {}).get("scaffold") if isinstance(wbs, dict) else {}
    if isinstance(scaffold, dict) and scaffold.get("derived_from_boq"):
        return None
    return (
        "BOQ-derived WBS was requested, but no demolition / site-clearance "
        "BOQ rows were retrieved. Refusing the generic template scaffold."
    )


def fulfill_wbs_export(
    user_message: str,
    project_id: str | None,
    conversation_id: str | None,
    agent_name: str,
    owner_id: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Compile the confirmation + export offer from staged conversation WBS.

    No LLM. Persists the turn when ``conversation_id`` is set.
    """
    from app.core import agent_memory

    if conversation_id:
        agent_memory.get_or_create_conversation(
            conversation_id, agent_name, project_id, owner_id=owner_id,
        )
        agent_memory.append_message(conversation_id, "user", user_message)

    staged = load_conversation_wbs(conversation_id)
    if not staged:
        answer = (
            "There is no WBS or schedule in this conversation to export. "
            "Generate a WBS first, then export that workbook. "
            "I will not substitute a generic template scaffold."
        )
        if conversation_id:
            agent_memory.append_message(conversation_id, "assistant", answer)
        return answer, []

    acts = staged.get("activities") or []
    n = len(acts)
    scaffold = staged.get("scaffold") if isinstance(staged.get("scaffold"), dict) else {}
    derived = bool(scaffold.get("derived_from_boq"))
    provenance = "BOQ-derived" if derived else "the WBS built in this conversation"
    overrides = staged.get("duration_overrides_applied") or []
    override_bit = ""
    if overrides:
        bits = ", ".join(
            f"{o.get('match')} = {o.get('days')} days"
            for o in overrides if isinstance(o, dict)
        )
        if bits:
            override_bit = f" Duration overrides present: {bits}."
    answer = (
        f"Exporting {provenance} ({n} activities). "
        "Download the Excel workbook with the button below. "
        "This is that conversation's WBS, not a generic template scaffold."
        + override_bit
    )
    exports: list[dict[str, Any]] = []
    if project_id and conversation_id:
        exports = [conversation_schedule_export_descriptor(
            project_id, conversation_id, n,
        )]
    if conversation_id:
        agent_memory.append_message(conversation_id, "assistant", answer)
    return answer, exports
