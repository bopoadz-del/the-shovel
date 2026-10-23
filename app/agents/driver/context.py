"""What the model is given in driver mode: standing rules, the hats it can
choose, the project's profile and -- once chosen -- the hat's guidance.

Everything here applies to every project and question alike: the rules
are a short file, the hats come from their manifests and the profile from
the project's own record. Nothing is retrieved or injected for the
question; the model searches with tools when it needs to.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

_RULES = Path(__file__).resolve().parents[3] / "config" / "driver_standing_rules.md"

#: Project fields shown to the model, in this order, when the record has them.
_PROFILE_FIELDS = ("name", "client", "location", "status")


def standing_rules() -> str:
    return _RULES.read_text(encoding="utf-8").strip()


def hats() -> List[Dict[str, str]]:
    """Every hat the catalogue loads: its discipline and description."""
    from app.agents import catalog

    return [{"discipline": h.discipline.value, "name": h.name, "description": h.description}
            for h in sorted(catalog.list_hats(), key=lambda h: h.discipline.value)]


def hat_guidance(discipline: str) -> Optional[str]:
    from app.agents import catalog

    for h in catalog.list_hats():
        if h.discipline.value == discipline:
            return h.system_prompt_guidance
    return None


def project_profile(project_id: Optional[str], user_id: Optional[str]) -> str:
    if not project_id:
        return "No project is open; answer from general knowledge."
    from app.core import projects

    try:
        project = projects.get_project_accessible(project_id, user_id)
    except Exception:  # noqa: BLE001 -- a profile lookup never blocks the turn
        project = None
    if not project:
        return f"Project {project_id}: no profile on record."
    lines = [f"{field}: {project[field]}" for field in _PROFILE_FIELDS if project.get(field)]
    profile = project.get("profile")
    if isinstance(profile, dict):
        lines += [f"{k}: {v}" for k, v in profile.items() if v not in (None, "")]
    return "Project profile\n" + "\n".join(lines) if lines else f"Project {project_id}: no profile on record."


def system_message(project_id: Optional[str], user_id: Optional[str], hat: Optional[str]) -> Dict[str, Any]:
    hat_lines = "\n".join(f"- {h['discipline']}: {h['description']}" for h in hats())
    parts = [
        "You are The Fork, a construction project assistant.",
        "Standing rules\n" + standing_rules(),
        "Hats you can wear (choose with select_hat)\n" + hat_lines,
        project_profile(project_id, user_id),
    ]
    guidance = hat_guidance(hat) if hat else None
    if guidance:
        parts.append(f"You are wearing the {hat} hat.\n{guidance}")
    return {"role": "system", "content": "\n\n".join(parts)}


def messages(user_message: str, history: Optional[list], project_id: Optional[str],
             user_id: Optional[str], hat: Optional[str]) -> List[Dict[str, Any]]:
    out = [system_message(project_id, user_id, hat)]
    for turn in history or []:
        if isinstance(turn, dict) and turn.get("role") in ("user", "assistant") and turn.get("content"):
            out.append({"role": turn["role"], "content": str(turn["content"])})
    out.append({"role": "user", "content": user_message})
    return out
