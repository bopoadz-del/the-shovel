"""Dynamic reasoner system prompt — Reasoning Engine Plan 5.

Pure string building. The prompt is rebuilt every turn so it reflects what the
session already knows — a follow-up question then builds on prior state.
"""

from typing import Iterable, Mapping, Optional

from app.core.catalogue import format_catalogue_for_prompt
from app.schemas.project_session import ProjectSession

_PHASES = """\
You are the Project Reasoner for a construction-project intelligence platform.
Work in four phases:
  UNDERSTAND — restate what the user is really asking for.
  PLAN       — produce an ordered list of steps that answers it.
  EXECUTE    — (the platform runs your plan; you do not run it yourself).
  DELIVER    — write a clear answer from the executed results.

When asked to PLAN, reply with ONLY a JSON object:
  {"understanding": "...", "steps": [{"type": "...", "args": {...},
   "output_key": "...", "description": "..."}]}

AVAILABLE STEP TYPES:
  - compute_cpm          run the Critical Path Method over loaded activities.
  - resource_histogram   time-phased manpower; args: {"period_unit": "week"|"month"}.
  - gantt                Gantt bars for the loaded schedule.
  - compress             shorten the schedule; args: {"reductions": {"<id>": <days>}}.
  - generate_code        novel/custom logic; args: {"task": "...",
                         "variables": {...}, "from_session": ["<key>", ...]}.
"""


def _state_summary(session: ProjectSession) -> str:
    d = session.data
    lines = []
    activities = d.get("activities")
    if activities:
        lines.append(f"- activities: {len(activities)} loaded.")
    else:
        lines.append("- activities: NOT loaded (no schedule yet).")
    for key, label in (
        ("cpm_results", "CPM already computed"),
        ("manpower", "resource histogram already computed"),
        ("gantt", "Gantt data already computed"),
        ("compressed", "a compressed schedule already exists"),
    ):
        if key in d:
            lines.append(f"- {label} (session key '{key}') — reuse it, "
                         f"do not recompute unless the user changed inputs.")
    if session.artifacts:
        lines.append(f"- {len(session.artifacts)} artifact(s) already produced.")
    return "\n".join(lines)


# Per-snippet hard cap so a single big document can't blow out the prompt.
_MAX_SNIPPET_CHARS = 800


def _format_document_excerpts(
    excerpts: Optional[Iterable[Mapping]] = None,
) -> str:
    """Render top-k document snippets as a labelled prompt section.

    ``excerpts`` is the shape produced by ``doc_index.search_project_documents``:
    a list of {document_id, filename, snippet, score} dicts. Returns the empty
    string when no excerpts are available, so the reasoner sees the section
    only when there is real content.
    """
    if not excerpts:
        return ""
    rendered = []
    for i, e in enumerate(excerpts, start=1):
        filename = e.get("filename") or e.get("document_id") or f"doc-{i}"
        snippet = str(e.get("snippet") or "").strip()
        if not snippet:
            continue
        if len(snippet) > _MAX_SNIPPET_CHARS:
            snippet = snippet[:_MAX_SNIPPET_CHARS].rstrip() + "..."
        rendered.append(f"[{i}] {filename}\n{snippet}")
    if not rendered:
        return ""
    return (
        "\n\nRELEVANT DOCUMENT EXCERPTS (extracted from project documents — "
        "use as evidence when answering):\n"
        + "\n\n".join(rendered)
    )


def build_reasoner_prompt(
    session: ProjectSession,
    request: str,
    document_excerpts: Optional[Iterable[Mapping]] = None,
) -> str:
    """Assemble the reasoner system prompt for the current turn.

    ``document_excerpts`` (optional) is the result of
    ``doc_index.search_project_documents(project_id, request)``: a list of
    {document_id, filename, snippet, score} dicts. When provided, the
    snippets are folded into the prompt so the reasoner can ground its plan
    in the actual uploaded files instead of only the session's structured
    state (activities, CPM results, etc.).
    """
    excerpts_block = _format_document_excerpts(document_excerpts)
    return (
        f"{_PHASES}\n\n"
        f"{format_catalogue_for_prompt()}\n\n"
        f"CURRENT SESSION STATE:\n{_state_summary(session)}"
        f"{excerpts_block}\n\n"
        f"USER REQUEST:\n{request}"
    )
