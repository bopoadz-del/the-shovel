"""The route catalogue as the model's tool (F-DRIVER Addendum 2: "routes
are the model's tools"). Every predefined workflow -- the plan builders and
the construction container's scoped actions -- is one ``run_workflow`` call
away; the model chooses which, code does not. Workflows run exactly as the
chat router runs them (same tenant gate, same file-parameter resolution)."""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

RUN_WORKFLOW = "run_workflow"


def catalogue() -> List[str]:
    """Every predefined workflow, once. Two action names bound to the same
    method are one workflow; the method's own name stands for it."""
    from app.core.predefined_reasoning import WORKFLOW_REGISTRY, container_action_names

    actions = container_action_names()
    bound: Dict[str, Any] = {}
    if actions:
        try:
            from app.dependencies import get_block_instance
            impl = get_block_instance("construction").get_actions()
            bound = {a: getattr(impl.get(a), "__func__", impl.get(a)) for a in actions}
        except Exception:  # noqa: BLE001 -- container_action_names already logs a failed load
            bound = {}

    def _alias(name: str) -> bool:
        fn = bound.get(name)
        own = getattr(fn, "__name__", name)
        return fn is not None and own != name and own in bound and bound[own] is fn

    return sorted((set(WORKFLOW_REGISTRY) | actions) - {a for a in actions if _alias(a)})


_CARDS_PATH = Path(__file__).resolve().parents[3] / "config" / "workflow_cards.yaml"
CARD_FIELDS = ("produces", "use_when", "needs", "inputs")


@lru_cache(maxsize=1)
def cards() -> Dict[str, Dict[str, Any]]:
    """config/workflow_cards.yaml: what each workflow produces, when to use
    it, what it needs and its inputs with display labels."""
    import yaml

    with open(_CARDS_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_FILLER = frozenset("a an and any asks for from how in is its of on or the to user what when which with".split())
OVERLAP_LIMIT = 0.5


def _content_words(card: Dict[str, Any]) -> set:
    text = f"{card.get('produces') or ''} {card.get('use_when') or ''}".lower()
    return set(re.findall(r"[a-z]+", text)) - _FILLER


def card_gaps(workflows: List[str], declared: Dict[str, Dict[str, Any]]) -> List[str]:
    """Every catalogue entry needs a complete card; every card names a
    workflow that exists."""
    gaps = []
    for w in workflows:
        card = declared.get(w) or {}
        for field in CARD_FIELDS[:-1]:
            if len(str(card.get(field) or "").split()) < 3:
                gaps.append(f"{w}: {field} missing or under three words")
        inputs = card.get("inputs")
        if not isinstance(inputs, dict) or not inputs:
            gaps.append(f"{w}: no inputs declared")
        else:
            gaps += [f"{w}: input {k} has no display label" for k, v in inputs.items()
                     if not str(v or "").strip() or str(v).strip() == k]
    gaps += [f"{name}: card for a workflow that is not in the catalogue"
             for name in sorted(set(declared) - set(workflows))]
    return gaps


def card_overlaps(declared: Dict[str, Dict[str, Any]], limit: float = OVERLAP_LIMIT) -> List[str]:
    """Pairs of cards whose output and use read alike: the model cannot
    choose between them reliably. Overlap is shared content words over all."""
    names, out = sorted(declared), []
    for i, a in enumerate(names):
        wa = _content_words(declared[a])
        for b in names[i + 1:]:
            wb = _content_words(declared[b])
            share = len(wa & wb) / max(1, len(wa | wb))
            if share >= limit:
                out.append(f"{a} / {b}: {share:.2f}")
    return out


def call_again_note(refused: Dict[str, Dict[str, Any]]) -> str:
    """The one reminder a model gets when it answers while a workflow it
    called is still refusing for a missing input."""
    parts = []
    for workflow, refusal in refused.items():
        inputs = ", ".join(f"{key} ({label})" for key, label in (refusal.get("inputs") or {}).items())
        parts.append(f"{workflow}: {str(refusal.get('missing') or '').strip()} Inputs: {inputs or 'see its entry'}.")
    return ("A workflow you called has not run yet, so it has produced nothing. Call run_workflow "
            "for it again now, with params filled from the user's own words and the documents you read, "
            "keyed as listed. Do not write the deliverable yourself. " + " ".join(parts))


def input_request(workflow: str, refusal: Dict[str, Any]) -> str:
    """What the user is asked when a workflow cannot run for a missing input:
    what it produces and what it needs, in its card's words."""
    card = cards().get(workflow) or {}
    produces = card.get("produces")
    needs = card.get("needs")
    if produces and needs:
        return (f"To produce {_lower_first(produces)}, I need {_lower_first(needs)}. "
                "Please provide them, or tell me which project document holds them.")
    return (f"The {workflow.replace('_', ' ')} workflow could not run: "
            f"{str(refusal.get('missing') or 'an input it needs is missing').strip()}")


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:]


def _entry(name: str) -> str:
    """The catalogue line the model reads for one workflow."""
    card = cards().get(name)
    if not card:
        return f"- {name}: {_summary(name)}."
    inputs = ", ".join(f"{label} ({key})" for key, label in (card.get("inputs") or {}).items())
    return (f"- {name}: produces {_lower_first(card['produces'])}. "
            f"Use when {_lower_first(card['use_when'])}. Needs {_lower_first(card['needs'])}. "
            f"Inputs: {inputs}.")


def _summary(name: str) -> str:
    """What a workflow produces, in its own words: the first sentence of its
    docstring (a container action's or a plan builder's)."""
    from app.core.predefined_reasoning import WORKFLOW_REGISTRY

    fn = WORKFLOW_REGISTRY.get(name)
    if fn is None:
        # A construction container action: its method documents it.
        try:
            from app.dependencies import get_block_instance
            fn = getattr(get_block_instance("construction"), name, None)
        except Exception:  # noqa: BLE001 -- a description is best-effort
            fn = None
    doc = " ".join(((getattr(fn, "__doc__", "") or "") if fn else "").split())
    if doc:
        return doc.split(". ")[0].rstrip(".")
    # No docstring: the action's name in words still says what it produces.
    return name.replace("_", " ").capitalize()


def schema(workflows: List[str]) -> Optional[Dict[str, Any]]:
    """The route catalogue as one tool, every workflow described by what it
    produces -- so the model can tell a deliverable from a calculation."""
    if not workflows:
        return None
    lines = [_entry(w) for w in workflows]
    return {
        "type": "function",
        "function": {
            "name": RUN_WORKFLOW,
            "description": "Run one of the platform's predefined workflows. Use one only when the user "
                           "asks for the deliverable it produces. A figure, duration or programme that a "
                           "calculator computes from the user's own inputs goes to that calculator, "
                           "not to a workflow. Pass the inputs it needs in params, under the keys "
                           "listed; look for them in the project's documents first. It returns the "
                           "workflow's answer and any export offers, or what is still missing.\n"
                           + "\n".join(lines),
            "parameters": {
                "type": "object",
                "properties": {
                    "workflow": {"type": "string", "enum": workflows},
                    "deliverable": {"type": "boolean",
                                    "description": "True when the user asked for the artifact itself."},
                    "params": {"type": "object", "description": "Inputs the workflow takes, if known."},
                },
                "required": ["workflow"],
            },
        },
    }


async def run(args: Dict[str, Any], user_message: str, project_id: Optional[str],
              user_id: Optional[str], conversation_id: Optional[str]) -> Dict[str, Any]:
    from app.core import projects
    from app.core.predefined_reasoning import run_workflow
    from app.routers.chat import _resolve_predefined_file_params
    from app.schemas.project_session import ProjectSession

    action = str(args.get("workflow") or "")
    if action not in catalogue():
        return {"ok": False, "error": f"No workflow named {action!r}."}
    safe_project_id, project_name = project_id, None
    if project_id:
        from app.core.offload import off_loop

        proj = await off_loop(projects.get_project_accessible, project_id, user_id)
        safe_project_id = project_id if proj else None
        project_name = (proj or {}).get("name")
    params = {k: v for k, v in (args.get("params") or {}).items() if v is not None} \
        if isinstance(args.get("params"), dict) else {}
    context: Dict[str, Any] = {
        "message": user_message, "project_id": safe_project_id,
        "project_name": project_name or "Project", "document_ids": [],
        "params": params, "conversation_id": conversation_id,
    }
    if isinstance(args.get("deliverable"), bool):
        context["deliverable"] = args["deliverable"]
    from app.core.offload import off_loop as _off

    # A database read: never on the event loop.
    resolved, ask = await _off(_resolve_predefined_file_params, action, safe_project_id, params, user_message)
    if ask:
        return {"ok": False, "needs_input": ask}
    context["params"] = resolved
    session = ProjectSession.new(f"driver-{user_id or 'anon'}-{conversation_id or 'turn'}",
                                 user_id=user_id or "system")
    out = await run_workflow(action, context, session)
    if not out.get("handled"):
        return {"ok": False, "error": f"The workflow {action!r} could not run."}
    if out.get("status") == "error":
        # The workflow ran and refused: an input it needs is missing. Its own
        # words say which; the documents may hold it.
        card = cards().get(action) or {}
        return {"ok": False, "workflow": action, "missing": out.get("answer"),
                "needs": card.get("needs"), "inputs": card.get("inputs") or {},
                "next": "Find these inputs in the project's documents (search them, then read the "
                        "passages), and call run_workflow again with them in params, keyed as "
                        "listed in inputs. Do not write this workflow's deliverable yourself. "
                        "Ask the user only for what the documents do not hold."}
    return {"ok": True, "workflow": action, "answer": out.get("answer"),
            "exports": out.get("exports") or []}
