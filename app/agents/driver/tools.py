"""The tools a driver turn offers: ``select_hat`` and the tools the base
package and the chosen hat's package register. The schemas are the agent's
own; which tools exist is the package registry's answer, never a choice
made from the question's words."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

SELECT_HAT = "select_hat"


def select_hat_schema(disciplines: List[str]) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": SELECT_HAT,
            "description": "Wear the hat whose description fits the question. Its tools and "
                           "guidance are added for the rest of the turn.",
            "parameters": {
                "type": "object",
                "properties": {"hat": {"type": "string", "enum": sorted(disciplines)}},
                "required": ["hat"],
            },
        },
    }


def offered(agent: Any, project_id: Optional[str], hat: Optional[str], disciplines: List[str]) -> List[Dict[str, Any]]:
    import dataclasses

    from app.agents.core import tool_registry

    names = set(tool_registry.tools_of("base"))
    if hat:
        names |= set(tool_registry.tools_of(hat))
    # The package tools' schemas are written next to the construction block's
    # in tool_definitions and emitted only when that block is allowed; in
    # driver mode the registry, not the agent's block list, decides.
    schema_source = dataclasses.replace(
        agent, allowed_blocks=sorted(set(agent.allowed_blocks) | {"construction"}))
    tools = [t for t in schema_source.tool_definitions(project_id=project_id)
             if (t.get("function") or {}).get("name") in names]
    # The calculator's schema comes from the registry, for this hat.
    tools = [calculator_schema(hat) if (t.get("function") or {}).get("name") == "construction_calc" else t
             for t in tools]
    # Tools that declare their schema where they are defined (e.g. general
    # knowledge) are offered from the registry.
    have = {(t.get("function") or {}).get("name") for t in tools}
    for name in sorted(names - have):
        spec = tool_registry.get(name)
        if spec is not None and spec.schema:
            tools.append(spec.schema)
    from app.agents.driver import routes

    workflows = routes.catalogue()
    workflow = routes.schema(workflows)
    if workflow:
        # A package tool named like a catalogue workflow runs that workflow
        # by a second door; with two doors the model's choice between them
        # varies. The workflow is offered once, through run_workflow.
        tools = [t for t in tools if (t.get("function") or {}).get("name") not in set(workflows)]
    return [select_hat_schema(disciplines)] + tools + ([workflow] if workflow else [])


#: Calculations listed in full (inputs with units and ranges) at most: base
#: plus one hat stays well under it; a larger offer is cut by registry order,
#: never silently. Every other registered calculator is still listed, briefly.
_MAX_CALCULATIONS = 80


def _input_line(key: str, p: Any) -> str:
    bits = [p.unit] if p.unit and p.unit != "-" else []
    if p.lo is not None and p.hi is not None:
        bits.append(f"{p.lo:g} to {p.hi:g}")
    if not p.required:
        bits.append("optional")
    label = p.label or key.replace("_", " ")
    return f"{key} = {label}" + (f" ({', '.join(bits)})" if bits else "")


_SCHEMA_CACHE: Dict[Any, Dict[str, Any]] = {}


def calculator_schema(hat: Optional[str]) -> Dict[str, Any]:
    """The calculator schema for ``hat``, built once per registry state: it
    was rebuilt on every turn, on the event loop (a 268 ms LOOP_STALL in
    production). Keyed on the hat and the registry's exact contents, so a
    registered, replaced or removed formula is never stale; each caller gets
    its own copy."""
    import copy

    from app.lib import formula_registry

    key = (hat, tuple((spec.name, id(spec)) for spec in formula_registry.all_specs()))
    if key not in _SCHEMA_CACHE:
        _SCHEMA_CACHE[key] = _build_calculator_schema(hat)
    return copy.deepcopy(_SCHEMA_CACHE[key])


def _build_calculator_schema(hat: Optional[str]) -> Dict[str, Any]:
    """The platform calculator as the model sees it in driver mode, built from
    the formula registry: the formulas offered to this hat (base plus its own),
    each with its display name, what it does, and every input's label, unit,
    range and whether it is required. The model fills ``params`` from the
    user's words by those labels and units -- no per-formula phrase lists."""
    from app.lib import formula_registry

    specs = formula_registry.specs_for(hat)[:_MAX_CALCULATIONS]
    lines = []
    for spec in specs:
        inputs = "; ".join(_input_line(k, p) for k, p in formula_registry.parameters(spec).items())
        lines.append(f"- {spec.name}: {spec.display_name}. {spec.description} Inputs: {inputs or 'none'}.")
    # Every other registered calculator stays callable before its hat is
    # chosen: a calculator the turn needs is never out of reach, so the model
    # never works its figure by hand or reaches for a different tool.
    listed = {s.name for s in specs}
    others = [s for s in formula_registry.all_specs() if s.name not in listed]
    if others:
        lines.append("Other calculators (in another hat; call them the same way):")
        for spec in others:
            # Optional inputs are marked here too: an input the model cannot
            # see is optional reads as one it must find or ask for (LANE3:
            # "using your standard rates" refused, or priced from a document).
            inputs = "; ".join(f"{k} = {p.label or k.replace('_', ' ')}" + ("" if p.required else " (optional)")
                               for k, p in formula_registry.parameters(spec).items())
            lines.append(f"- {spec.name}: {spec.display_name} ({spec.owner}). Inputs: {inputs or 'none'}.")
    specs = specs + others
    return {
        "type": "function",
        "function": {
            "name": "construction_calc",
            "description": (
                "Run one of the platform's deterministic calculators. Choose the calculation "
                "that does what the user asks, then fill params from the user's own words: "
                "match each figure to an input by its label and unit. Give each figure exactly as "
                "the user wrote it, as a string with its sign and unit (\"-120 m2 per day\", "
                "\"21,528 sq ft\"): never change its sign, never convert or normalise its unit. "
                "The calculator reads the sign and unit from that text, converts what it can, and "
                "decides whether a negative or zero figure can be used. Leave out an input "
                "the user did not give: a required one the calculator asks for; an optional one "
                "runs on the calculator's own declared default, and the answer states that "
                "default. A figure read from a document goes in only when the user asks for "
                "that document's or project's figures. Never invent a value. "
                "In the answer, name inputs and the calculator by their labels and display "
                "name, never by the identifiers below.\n" + "\n".join(lines)
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "calculation": {"type": "string", "enum": [s.name for s in specs],
                                    "description": "Which calculator to run."},
                    "params": {"type": "object", "additionalProperties": True,
                               "description": ("The calculator's inputs, keyed as listed for it; each "
                                               "figure as the user wrote it, sign and unit included.")},
                },
                "required": ["calculation", "params"],
            },
        },
    }
