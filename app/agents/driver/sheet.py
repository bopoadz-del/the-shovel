"""The per-turn working sheet: the single state of a chained answer.

One JSON object per turn, handed back to the model on every round until the
chain is done::

    {"turn": "...", "question": "...",
     "givens":  [{"id": "g1", "what": "...", "value": 20, "unit": "nr", "source": "user"}],
     "plan":    [{"id": "s1", "entry": "formula:<name>", "inputs": {"<input>": "g1"}}],
     "steps":   [{"id": "s1", "call_id": "...", "status": "done",
                  "result": {"name": "...", "value": 1.5, "unit": "m3"}}],
     "missing": [{"for": "s2", "input": "...", "label": "...", "asked": true}],
     "done": false}

Who writes what -- the point of the sheet:

* the model writes ``givens``, ``plan`` and ``missing`` (``apply_update``);
* code writes every ``steps[].result`` from the tool's own result
  (``record_result``); an update from the model that carries a step or a
  result is refused, so no figure on the sheet was typed by the model;
* every write (``apply_update``, ``run_step``, ``record_result``) closes a
  missing input whose step has its result (``close_answered``), so what is
  left in ``missing`` is always waiting on a step that has not run;
* a plan input names a given (``g1``) or an earlier step (``s1``); code
  resolves the reference (``resolve_inputs``), so a value is never re-typed.

``run_step`` runs a plan step whose entry is a registered formula
(``formula:<name>``): code resolves its inputs from the sheet, holds them to
the formula's declared units and ranges, runs it and writes the result. An
in-between step (a sum, a difference, a product, a quotient) is the
``arithmetic_step`` formula like any other: arithmetic is never done in the
model's text.

Nothing here names a formula, a project or a document: the sheet holds what
the model and the tools put on it.
"""
from __future__ import annotations

import copy
import json
import re
from typing import Any, Dict, List, Optional, Tuple

#: Fields of the sheet the model may write. ``steps`` and ``done`` are code's.
MODEL_FIELDS = ("givens", "plan", "missing")

STEP_STATUSES = ("done", "failed", "waiting_input")

_GIVEN_ID = re.compile(r"^g\d+$")
_STEP_ID = re.compile(r"^s\d+$")


def new_sheet(turn: str, question: str, previous: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """A fresh sheet for ``turn``. The previous turn's givens and missing
    inputs carry over (a later value or a correction replaces a given by its
    id), so an input already asked for is not asked again."""
    sheet: Dict[str, Any] = {"turn": str(turn), "question": str(question or ""),
                             "givens": [], "plan": [], "steps": [], "missing": [],
                             "done": False}
    if isinstance(previous, dict):
        sheet["givens"] = [dict(g, source=g.get("source") or "earlier_turn")
                           for g in previous.get("givens") or [] if isinstance(g, dict)]
        sheet["missing"] = [dict(m) for m in previous.get("missing") or [] if isinstance(m, dict)]
    return sheet


def _number(value: Any) -> Optional[float]:
    from app.agents.base.formulas.arithmetic import finite_number

    return finite_number(value)


def _given_problems(given: Any) -> List[str]:
    if not isinstance(given, dict):
        return ["a given must be an object"]
    gid = str(given.get("id") or "")
    problems = []
    if not _GIVEN_ID.match(gid):
        problems.append(f"given id {gid!r} must look like g1, g2, ...")
    if _number(given.get("value")) is None:
        problems.append(f"given {gid or '?'} has no numeric value")
    if not str(given.get("unit") or "").strip():
        problems.append(f"given {gid or '?'} has no unit (write '-' for a pure number)")
    return problems


def _plan_problems(step: Any, known: set) -> List[str]:
    if not isinstance(step, dict):
        return ["a plan step must be an object"]
    sid = str(step.get("id") or "")
    problems = []
    if not _STEP_ID.match(sid):
        problems.append(f"plan step id {sid!r} must look like s1, s2, ...")
    if not str(step.get("entry") or "").strip():
        problems.append(f"plan step {sid or '?'} names no catalogue entry")
    inputs = step.get("inputs")
    if not isinstance(inputs, dict):
        problems.append(f"plan step {sid or '?'} inputs must be an object")
        return problems
    for name, ref in inputs.items():
        if isinstance(ref, str) and (_GIVEN_ID.match(ref) or _STEP_ID.match(ref)):
            if ref not in known:
                problems.append(f"plan step {sid} input {name} refers to {ref}, which is not on the sheet")
            elif ref == sid:
                problems.append(f"plan step {sid} input {name} refers to itself")
    return problems


def apply_update(sheet: Dict[str, Any], update: Any) -> Tuple[Dict[str, Any], List[str]]:
    """Apply the model's ``sheet_update``. Returns ``(new sheet, problems)``.

    Only ``givens``, ``plan`` and ``missing`` are read. An update that carries
    ``steps``, ``done`` or a result is refused whole -- results are code's.
    A given or plan step with the id of an existing one replaces it (a
    correction); a new id is appended. When a correction changes a given or
    a plan step, that step and every step depending on it (directly or
    through another step, by the plan's inputs) go back to open: their
    results are cleared and the sheet is not done. On any problem the sheet
    is returned unchanged with the problems, so the model can correct its
    update."""
    if not isinstance(update, dict):
        return sheet, ["sheet_update takes an object with givens, plan and missing"]
    foreign = sorted(k for k in update if k not in MODEL_FIELDS)
    if foreign:
        return sheet, [f"only {', '.join(MODEL_FIELDS)} can be written; "
                       f"{', '.join(foreign)} are written by code from the tool results"]
    for key in MODEL_FIELDS:
        if key in update and not isinstance(update[key], list):
            return sheet, [f"{key} must be a list"]
    out = copy.deepcopy(sheet)
    problems: List[str] = []
    replaced: set = set()  # ids whose given or plan step now differs
    for given in update.get("givens") or []:
        found = _given_problems(given)
        if found:
            problems.extend(found)
            continue
        entry = {"id": given["id"], "what": str(given.get("what") or ""),
                 "value": _number(given["value"]), "unit": str(given["unit"]).strip(),
                 "source": str(given.get("source") or "user")}
        if _changed(out["givens"], entry, ("value", "unit")):
            replaced.add(entry["id"])
        _put(out["givens"], entry)
    known = {g["id"] for g in out["givens"]} | {s.get("id") for s in out["plan"]}
    known |= {s.get("id") for s in update.get("plan") or [] if isinstance(s, dict)}
    for step in update.get("plan") or []:
        found = _plan_problems(step, known)
        if found:
            problems.extend(found)
            continue
        planned = {"id": step["id"], "entry": str(step["entry"]).strip(),
                   "inputs": dict(step["inputs"])}
        if str(step.get("output") or "").strip():
            planned["output"] = str(step["output"]).strip()
        if _changed(out["plan"], planned, ("entry", "inputs", "output")):
            replaced.add(planned["id"])
        _put(out["plan"], planned)
    if "missing" in update:
        # The model's list replaces the old one (an input it now has is left
        # off); an item already asked for keeps its mark, so it is not
        # asked a second time.
        asked = {(m.get("for"), m.get("input")) for m in out["missing"] if m.get("asked")}
        missing: List[Dict[str, Any]] = []
        for item in update.get("missing") or []:
            if not isinstance(item, dict) or not str(item.get("input") or "").strip():
                problems.append("a missing item must name the input it needs")
                continue
            key = (str(item.get("for") or ""), str(item["input"]))
            if any((m["for"], m["input"]) == key for m in missing):
                continue
            missing.append({"for": key[0], "input": key[1],
                            "label": str(item.get("label") or item["input"]),
                            "asked": key in asked})
        out["missing"] = missing
    if problems:
        return sheet, problems
    stale = dependents(out, replaced) | {i for i in replaced if _STEP_ID.match(i)}
    out["steps"] = [st for st in out["steps"] if st.get("id") not in stale]
    out = close_answered(out)
    out["done"] = is_done(out)
    return out, []


def _changed(items: List[Dict[str, Any]], entry: Dict[str, Any], fields: Tuple[str, ...]) -> bool:
    """True when ``entry`` replaces an item of the same id whose ``fields``
    differ (a correction); a new id or the same content is no change."""
    old = next((i for i in items if i.get("id") == entry["id"]), None)
    return old is not None and any(old.get(f) != entry.get(f) for f in fields)


def dependents(sheet: Dict[str, Any], ids: Any) -> set:
    """Plan steps that depend on any of ``ids`` (a given or a step), directly
    or through another step. Dependency is read from the plan's own inputs
    and nothing else."""
    found: set = set()
    frontier = set(ids)
    while frontier:
        nxt = {p["id"] for p in sheet.get("plan") or []
               if p["id"] not in found
               and any(isinstance(r, str) and r in frontier for r in (p.get("inputs") or {}).values())}
        found |= nxt
        frontier = nxt
    return found


def _put(items: List[Dict[str, Any]], entry: Dict[str, Any]) -> None:
    for i, existing in enumerate(items):
        if existing.get("id") == entry["id"]:
            items[i] = entry
            return
    items.append(entry)


def _step(sheet: Dict[str, Any], sid: str) -> Optional[Dict[str, Any]]:
    return next((s for s in sheet.get("steps") or [] if s.get("id") == sid), None)


def resolve_inputs(sheet: Dict[str, Any], step_id: str) -> Tuple[Dict[str, Any], List[str]]:
    """The values a plan step runs on: each reference to a given or to an
    earlier step's result replaced by ``{"value", "unit"}``; a literal input
    (a text option, a number the model wrote) is passed as written.
    A reference to a step that is not done is a problem, never a guess."""
    plan = next((p for p in sheet.get("plan") or [] if p.get("id") == step_id), None)
    if plan is None:
        return {}, [f"{step_id} is not a plan step"]
    givens = {g["id"]: g for g in sheet.get("givens") or []}
    values: Dict[str, Any] = {}
    problems: List[str] = []
    for name, ref in (plan.get("inputs") or {}).items():
        if isinstance(ref, str) and _GIVEN_ID.match(ref):
            given = givens.get(ref)
            if given is None:
                problems.append(f"{name}: given {ref} is not on the sheet")
                continue
            values[name] = {"value": given["value"], "unit": given["unit"]}
        elif isinstance(ref, str) and _STEP_ID.match(ref):
            step = _step(sheet, ref)
            if step is None or step.get("status") != "done" or not step.get("result"):
                problems.append(f"{name}: step {ref} has no result yet")
                continue
            result = step["result"]
            values[name] = {"value": result["value"], "unit": result.get("unit") or "-"}
        else:
            values[name] = ref
    return values, problems


def record_result(sheet: Dict[str, Any], step_id: str, call_id: str,
                  tool_result: Any, output: str = "") -> Dict[str, Any]:
    """Write a step's result from the tool's own result (code, never the
    model). ``output`` names the result field to keep; without it the tool's
    main ``result`` is used. A tool error or a missing / non-numeric output
    marks the step failed with the tool's own words."""
    out = copy.deepcopy(sheet)
    entry: Dict[str, Any] = {"id": step_id, "call_id": str(call_id or "")}
    value, unit, name, error = _result_value(tool_result, output)
    if error:
        entry.update(status="failed", error=error)
    else:
        entry.update(status="done", result={"name": name, "value": value, "unit": unit})
    _put(out["steps"], entry)
    out = close_answered(out)
    out["done"] = is_done(out)
    return out


def _result_value(tool_result: Any, output: str) -> Tuple[Optional[float], str, str, str]:
    if not isinstance(tool_result, dict):
        return None, "", "", "the tool returned no result object"
    if tool_result.get("error") or tool_result.get("status") == "error":
        return None, "", "", str(tool_result.get("error") or "the tool reported an error")
    body = tool_result.get("result", tool_result)
    unit = str(tool_result.get("unit") or "")
    units = tool_result.get("units") if isinstance(tool_result.get("units"), dict) else {}
    if output:
        if not isinstance(body, dict) or output not in body:
            return None, "", "", f"the tool result has no output {output!r}"
        number = _number(body[output])
        name = output
        unit = str(units.get(output) or unit or "-")
    else:
        number = _number(body)
        name = str(tool_result.get("name") or "result")
        unit = unit or "-"
    if number is None:
        return None, "", "", f"output {name!r} is not a number"
    return number, unit, name, ""


def is_done(sheet: Dict[str, Any]) -> bool:
    """Done when the plan has at least one step and every step is done."""
    plan = sheet.get("plan") or []
    if not plan:
        return False
    return all((_step(sheet, p["id"]) or {}).get("status") == "done" for p in plan)


def close_answered(sheet: Dict[str, Any]) -> Dict[str, Any]:
    """The sheet with every missing input whose step has its result taken
    off: the step ran, so what it was waiting for is no longer missing (a
    fact code reads from the steps, not a judgment). An item tied to no
    step, or to a step without a result, stays as the model wrote it."""
    out = copy.deepcopy(sheet)
    ran = {st.get("id") for st in out.get("steps") or []
           if st.get("status") == "done" and st.get("result")}
    out["missing"] = [m for m in out.get("missing") or [] if not m.get("for") or m.get("for") not in ran]
    return out


def code_results(sheet: Any) -> List[Dict[str, Any]]:
    """The results code wrote on the sheet: one per done step, with its
    step id, entry, output name, value and unit exactly as written. The
    model cannot write a step (``apply_update`` refuses it), so every row
    here is a figure from code. Anything not shaped like a sheet gives []."""
    if not isinstance(sheet, dict):
        return []
    rows: List[Dict[str, Any]] = []
    for st in sheet.get("steps") or []:
        if not isinstance(st, dict) or st.get("status") != "done" or not isinstance(st.get("result"), dict):
            continue
        result = st["result"]
        value = _number(result.get("value"))
        if value is None:
            continue
        rows.append({"step": str(st.get("id") or ""), "entry": str(st.get("entry") or ""),
                     "name": str(result.get("name") or ""), "value": value,
                     "unit": str(result.get("unit") or "")})
    return rows


def mark_asked(sheet: Dict[str, Any]) -> Dict[str, Any]:
    """Every missing input has now been asked for (once, this turn)."""
    out = copy.deepcopy(sheet)
    for item in out["missing"]:
        item["asked"] = True
    return out


def unasked(sheet: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Missing inputs not yet asked for in any turn."""
    return [m for m in sheet.get("missing") or [] if not m.get("asked")]


def render(sheet: Dict[str, Any]) -> str:
    """The sheet as the model reads it each round: compact JSON, keys in a
    fixed order, so two runs of the same turn see the same bytes."""
    return json.dumps(sheet, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


#: The kind of catalogue entry a plan step runs through ``run_step``.
FORMULA_ENTRY = "formula"


def open_steps(sheet: Dict[str, Any]) -> List[str]:
    """Plan steps that are not done yet (never run, failed, or waiting for an
    input). While any is open the turn keeps its tools."""
    return [p["id"] for p in sheet.get("plan") or []
            if (_step(sheet, p["id"]) or {}).get("status") != "done"]


def _formula_values(spec: Any, values: Dict[str, Any]) -> Dict[str, Any]:
    """Resolved sheet values as a formula takes them. A value with a unit goes
    to an input that declares a companion ``<input>_unit`` as the bare number
    plus that unit; otherwise it is written with its unit, so the registry's
    declared units convert it (or refuse a unit of another kind)."""
    from app.lib import formula_registry

    params = formula_registry.parameters(spec)
    out: Dict[str, Any] = {}
    for name, value in values.items():
        if not (isinstance(value, dict) and "value" in value):
            out[name] = value
            continue
        number, unit = value["value"], str(value.get("unit") or "-").strip() or "-"
        if f"{name}_unit" in params:
            out[name] = number
            out.setdefault(f"{name}_unit", unit)
        elif unit == "-":
            out[name] = number
        else:
            out[name] = f"{number:g} {unit}"
    return out


def _output_of(spec: Any, chosen: str) -> str:
    if chosen:
        return chosen
    if spec.main_output:
        return spec.main_output
    numeric = [k for k in spec.outputs if not k.endswith("_unit")]
    return numeric[0] if len(numeric) == 1 else ""


def run_step(sheet: Dict[str, Any], step_id: str, call_id: str = "") -> Dict[str, Any]:
    """Run plan step ``step_id`` and write its result (code, never the model).

    The step's entry names a registered formula (``formula:<name>``). Its
    inputs are resolved from the sheet, converted to the formula's declared
    units and held to its declared ranges; a required input that is absent
    leaves the step ``waiting_input`` with the inputs' labels; a refused
    value, an unknown entry or the formula's own error leaves it ``failed``
    with the reason. Returns the new sheet."""
    from app.lib import formula_registry

    out = copy.deepcopy(sheet)
    plan = next((p for p in out.get("plan") or [] if p.get("id") == step_id), None)
    if plan is None:
        _put(out["steps"], {"id": step_id, "call_id": str(call_id or ""), "status": "failed",
                            "error": f"{step_id} is not a plan step"})
        out["done"] = is_done(out)
        return out
    kind, _, name = str(plan.get("entry") or "").partition(":")
    spec = formula_registry.get(name) if kind == FORMULA_ENTRY else None

    def _end(**fields: Any) -> Dict[str, Any]:
        _put(out["steps"], {"id": step_id, "call_id": str(call_id or ""), "entry": plan["entry"], **fields})
        closed = close_answered(out)
        closed["done"] = is_done(closed)
        return closed

    if spec is None:
        return _end(status="failed", error=(f"{plan.get('entry')!r} is not a registered formula; "
                                            f"a step runs an entry written as {FORMULA_ENTRY}:<name>"))
    values, problems = resolve_inputs(out, step_id)
    if problems:
        return _end(status="failed", error="; ".join(problems))
    params = formula_registry.parameters(spec)
    unknown = sorted(k for k in values if k not in params)
    if unknown:
        return _end(status="failed", error=f"{', '.join(unknown)} are not inputs of {name}")
    supplied = _formula_values(spec, values)
    supplied, _conversions, rejected = formula_registry.convert_written_units(name, supplied)
    if not rejected:
        supplied, rejected = formula_registry.check_inputs(name, supplied)
    if rejected:
        return _end(status="failed", error="; ".join(
            f"{r.get('label') or r.get('input') or r.get('name')}: {r.get('reason')}" for r in rejected))
    absent = [p for k, p in params.items() if p.required and k not in supplied]
    if absent:
        return _end(status="waiting_input", needs=[p.label or p.name for p in absent])
    try:
        result = spec.fn(**supplied)
    except Exception as exc:  # noqa: BLE001 -- the formula's failure is the step's, in its words
        return _end(status="failed", error=f"{name} could not run: {exc}")
    output = _output_of(spec, str(plan.get("output") or ""))
    if isinstance(result, dict) and result.get("error"):
        return _end(status="failed", error=str(result["error"]))
    if not isinstance(result, dict) or not output or output not in result:
        return _end(status="failed", error=f"{name} returned no output {output or '(none named)'!r}")
    number = _number(result[output])
    if number is None:
        return _end(status="failed", error=f"output {output!r} of {name} is not a number")
    declared = spec.outputs.get(output) or "-"
    unit = str(result.get(f"{output}_unit") or declared).strip() or "-"
    done = {"status": "done", "result": {"name": output, "value": number, "unit": unit},
            "inputs": supplied}
    if result.get("method"):
        done["method"] = str(result["method"])
    return _end(**done)
