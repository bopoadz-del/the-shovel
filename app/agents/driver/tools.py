"""The tools a driver turn offers: ``select_hat`` and every tool the base
package and the hat packages register. A hat changes the order, never what
is offered: the worn hat's tools come first, then the base package's, then
the other hats'. The schemas are the agent's own; which tools exist is the
package registry's answer, never a choice made from the question's words."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

SELECT_HAT = "select_hat"


def select_hat_schema(disciplines: List[str]) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": SELECT_HAT,
            "description": "Wear the hat whose description fits the question. Its guidance is "
                           "added for the rest of the turn and its tools are listed first; "
                           "every tool stays offered whichever hat is worn.",
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

    # Every package's tools, whichever hat is worn (or none): a hat is focus,
    # never access. A tool the turn needs is never out of reach.
    names = {n for owner in tool_registry.OWNERS for n in tool_registry.tools_of(owner)}
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
    return [select_hat_schema(disciplines)] + lead_first(tools, hat) + ([workflow] if workflow else [])


def lead_rank(name: str, hat: Optional[str]) -> int:
    """0 for a tool the worn hat owns, 1 for a base tool, 2 for another
    hat's tool. A name the registry does not know ranks as base."""
    from app.agents.core import tool_registry

    spec = tool_registry.get(name or "")
    owner = spec.owner if spec is not None else "base"
    if hat and owner == hat:
        return 0
    return 1 if owner == "base" else 2


def lead_first(tools: List[Dict[str, Any]], hat: Optional[str]) -> List[Dict[str, Any]]:
    """The worn hat's tools first, then the base package's, then the other
    hats' -- each group in its existing order. Nothing is added or removed."""
    return sorted(tools, key=lambda t: lead_rank((t.get("function") or {}).get("name") or "", hat))


#: Calculations listed in full (inputs with units and ranges) at most: base
#: plus one hat stays well under it; a larger offer is cut by registry order,
#: never silently. Every other registered calculator is still listed, briefly.
_MAX_CALCULATIONS = 80


#: How an input's declared basis reads in the offer. A to-date figure bound
#: to a period input is counted twice once earlier payments are taken off, so
#: the line says which reading the input takes.
BASIS_WORDS = {
    "period": "this period's amount only, not a total to date",
    "cumulative": "total to date, including earlier periods",
}


def _input_line(key: str, p: Any) -> str:
    bits = [p.unit] if p.unit and p.unit != "-" else []
    if p.lo is not None and p.hi is not None:
        bits.append(f"{p.lo:g} to {p.hi:g}")
    if getattr(p, "basis", ""):
        bits.append(BASIS_WORDS[p.basis])
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


def calculator_catalogue(hat: Optional[str]) -> "tuple[List[Any], List[str]]":
    """Every registered calculator as the model is offered it, one line each:
    name, display name and every input's label, unit, range and whether it is
    required -- built from the formula registry, the one source both chat paths
    (driver and default) read. The hat's own formulas come first with what they
    do; every other calculator follows, still with its labelled inputs, units
    and ranges, so the model fills ``params`` by label and unit whichever path
    or hat the turn is on. Returns ``(specs, lines)`` in offer order."""
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
            # The same labelled line as above (label, unit, range, optional):
            # an input the model cannot see is optional reads as one it must
            # find or ask for, and an input without its unit is filled in the
            # wrong one.
            inputs = "; ".join(_input_line(k, p) for k, p in formula_registry.parameters(spec).items())
            lines.append(f"- {spec.name}: {spec.display_name} ({spec.owner}). Inputs: {inputs or 'none'}.")
    return specs + others, lines


def _build_calculator_schema(hat: Optional[str]) -> Dict[str, Any]:
    """The platform calculator as the model sees it in driver mode, built from
    the formula registry: the formulas offered to this hat (base plus its own),
    each with its display name, what it does, and every input's label, unit,
    range and whether it is required. The model fills ``params`` from the
    user's words by those labels and units -- no per-formula phrase lists."""
    specs, lines = calculator_catalogue(hat)
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


#: The working-sheet tools a catalogue-path turn adds (DESIGN V2).
SHEET_UPDATE = "sheet_update"
RUN_STEP = "run_step"


def sheet_schemas() -> List[Dict[str, Any]]:
    """``sheet_update`` (the model writes givens, plan and missing) and
    ``run_step`` (code runs one planned formula step and writes its result).
    The same two schemas for every turn and project."""
    given = {"type": "object", "properties": {
        "id": {"type": "string", "description": "g1, g2, ..."},
        "what": {"type": "string"}, "value": {"type": "number"},
        "unit": {"type": "string", "description": "the value's unit; '-' for a pure number"},
        "source": {"type": "string", "description": "user, or the document and page it was read from"}},
        "required": ["id", "value", "unit"]}
    step = {"type": "object", "properties": {
        "id": {"type": "string", "description": "s1, s2, ..."},
        "entry": {"type": "string", "description": "formula:<calculation name>, e.g. formula:arithmetic_step"},
        "inputs": {"type": "object", "additionalProperties": True,
                   "description": "each input of the formula: a given id (g1), an earlier step id (s1), "
                                  "or a literal option such as an operator"},
        "output": {"type": "string", "description": "which output of the formula this step keeps "
                                                    "(optional when the formula has one main output)"}},
        "required": ["id", "entry", "inputs"]}
    missing = {"type": "object", "properties": {
        "for": {"type": "string"}, "input": {"type": "string"}, "label": {"type": "string"}},
        "required": ["input"]}
    return [
        {"type": "function", "function": {
            "name": SHEET_UPDATE,
            "description": ("Write this turn's working sheet: the givens (each figure with its unit and "
                            "source), the plan (one step per formula, inputs by reference) and the inputs "
                            "still missing. A given or step with an existing id replaces it. Results are "
                            "never written here: run_step writes them. The sheet is handed back after "
                            "every call."),
            "parameters": {"type": "object", "properties": {
                "givens": {"type": "array", "items": given},
                "plan": {"type": "array", "items": step},
                "missing": {"type": "array", "items": missing}}}}},
        {"type": "function", "function": {
            "name": RUN_STEP,
            "description": ("Run one plan step. Code resolves its inputs from the sheet, converts them to "
                            "the formula's units, runs the formula and writes the result on the sheet. "
                            "Every arithmetic step, even one division, is a step (formula:arithmetic_step)."),
            "parameters": {"type": "object", "properties": {"step": {"type": "string"}},
                           "required": ["step"]}}},
    ]
