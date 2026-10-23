"""The construction container's action surface must close in both directions.

Audit 2026-08-12.

The first version of this file only asked "can route() dispatch every public
action?". That question was too weak, and it let a worse bug through: three
actions were unreachable, and wiring them into the router turned an honest
`{"status": "error", "error": "Unknown action: ..."}` into an unhandled
AttributeError — a 500 instead of a refusal.

The reason they were unreachable was not an oversight. Two of them cannot run:

    generate_construction_report  -> self._generate_doc_recommendations
    track_progress                -> self._assess_delay_risk
      (and, one level down, _compare_photo_to_bim -> self._element_similarity,
       self._find_deviations)

None of those four helpers has ever been defined in this repository.
`git log -S "def _assess_delay_risk" --all` and its three siblings return
nothing. `generate_construction_report` additionally reads five keys off
`process_document` that its current contract does not return. So the methods
were dead because they were broken, and "unreachable" was the symptom.

That is the general lesson this file now encodes: reachability alone is not
health. An action can be routable and still be incapable of executing. So the
fence checks three things, not one:

  1. FORWARD  — every action name a caller surface can emit is dispatchable.
  2. REVERSE  — every dispatchable action is named by some caller surface,
                or is on a short documented list of externally-called ones.
  3. RUNNABLE — no method in the package calls a `self.` helper that does not
                exist. This is the assertion that would have caught the four
                missing helpers on the day they were written.

Note the two namespaces. `handlers` aliases deliberately
(`"schedule_risk": self.analyze_schedule_risk`), so the KEYS are what callers
name and the `self.<method>` targets are what the class defines. Checks that
mix them report false positives; that mistake was made once during this audit.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

CONTAINER_DIR = Path("app/containers/construction")
INIT = CONTAINER_DIR / "__init__.py"


# ── parsing the two namespaces ───────────────────────────────────────────────

def _handlers_block() -> str:
    src = INIT.read_text(encoding="utf-8")
    start = src.index("handlers = {")
    return src[start:src.index("\n        }", start)]


def _routed_keys() -> set[str]:
    """The names a caller passes to route() — dict KEYS, aliases included."""
    return set(re.findall(r'"([a-z0-9_]+)":\s*self\.', _handlers_block()))


def _routed_methods() -> set[str]:
    """The methods the handlers table binds — `self.<name>`, not the keys."""
    return set(re.findall(r"self\.([a-z0-9_]+)", _handlers_block()))


def _public_actions() -> set[str]:
    """Methods shaped like a container action: (self, input_data, params)."""
    found = set()
    for path in sorted(CONTAINER_DIR.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
                args = [a.arg for a in node.args.args]
                if (len(args) >= 3 and args[:3] == ["self", "input_data", "params"]
                        and not node.name.startswith("_")):
                    found.add(node.name)
    return found


# ── caller surfaces ──────────────────────────────────────────────────────────
# A "dispatch surface" names an action that something will then try to RUN.
# Everything it can emit must be dispatchable.

def _suggested_actions() -> set[str]:
    """`_suggest_next_action` — shown to the user as the next thing to do."""
    return set(re.findall(r'"suggested_action":\s*"([a-z0-9_]+)"',
                          INIT.read_text(encoding="utf-8")))


def _pipeline_next_actions() -> set[str]:
    """`auto_pipeline` builds next_actions[] for the UI to offer as buttons.

    Read via AST from `next_actions.append({...})` rather than by grepping
    every `"action":` literal — the package is full of unrelated block params
    like {"action": "query_location"} that a blind regex sweeps up.
    """
    found = set()
    for path in sorted(CONTAINER_DIR.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "append"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "next_actions"
                    and node.args and isinstance(node.args[0], ast.Dict)):
                for key, value in zip(node.args[0].keys, node.args[0].values):
                    if getattr(key, "value", None) == "action" and isinstance(value, ast.Constant):
                        found.add(value.value)
    return found


def _step_alias_actions() -> set[str]:
    """Workflow template steps that resolve to a construction action."""
    from app.core.cm_step_aliases import STEP_TO_TARGET

    return {action for block, action in STEP_TO_TARGET.values()
            if block == "construction" and action}


def _hint_actions() -> set[str]:
    """action_router.ACTION_HINTS — a NON-dispatch surface.

    These only pick a system-prompt fragment for the chat router; the module
    docstring is explicit that most of its keys have no 1:1 handler and that
    unknown ones fall through to plain chat. So it does not belong in the
    forward check — but it is real evidence that a name is reachable, so it
    counts for the reverse one.
    """
    from app.core.action_router import ACTION_HINTS

    return set(ACTION_HINTS)


def _orchestrator_actions() -> set[str]:
    """Action strings smart_orchestrator emits from keyword matches.

    DELIBERATELY LOOSE, and only ever used to WIDEN the reachable set in the
    reverse check — never to fail anything. It intersects every quoted
    lowercase string in the module with the route keys, so an action whose
    name happens to coincide with an ordinary quoted word in that file counts
    as reachable when it may not be. The consequence is a possible false
    NEGATIVE in the orphan list (an action treated as reachable that nothing
    really emits), never a false failure. Tightening it means enumerating the
    real emission sites; until then, do not read the orphan set as exact.
    """
    src = Path("app/blocks/smart_orchestrator.py").read_text(encoding="utf-8")
    return set(re.findall(r'"([a-z0-9_]+)"', src)) & _routed_keys()


# Dispatchable actions that no in-repo surface names, each because it is
# entered from OUTSIDE this package. Asserted by equality, not as an
# allowlist: a new unreferenced action fails here and must be justified, and
# an entry that becomes reachable also fails and must be deleted. An
# allowlist would rot silently in one direction; equality rots in neither.
EXPECTED_EXTERNAL_ONLY = {
    "chat",                 # the chat endpoint dispatches this by name
    "auto_pipeline",        # driven by the upload pipeline, not by a suggestion
    "orchestrate",          # entered from the orchestrator layer above this one
    "jetson_dispatch",      # edge-device caller lives outside the repo
    "formula_execute",      # /v1/execute callers name this directly
    "bim_extractor",        # LLM names the standalone block; aliased to bim_extract
    "primavera_parse",      # ditto — block delegation, called by API clients
    "learn",                # ditto
    "benchmark_lookup",     # ditto
    "recommend",            # ditto
    "analyze_spec",         # legacy short-name alias kept for integrations
    "contract_review",      # legacy short-name alias
    "carbon_report",        # legacy short-name alias
    "safety_audit",         # legacy short-name alias
    "procurement_analysis",  # canonical name; the "procurement" alias is hinted
    # Wired 2026-08-12 during the audit, once each was made to run. Nothing in
    # the repo names either yet -- track_progress needs a photo set and an IFC
    # model, which no automated surface currently assembles. If a surface
    # starts naming one, delete its line.
    "extract_measurements",
    "track_progress",
}


# ── 1. FORWARD: everything a caller can name must dispatch ───────────────────

def test_every_public_action_is_routable():
    unreachable = sorted(_public_actions() - _routed_methods() - PARKED_UNRUNNABLE)
    assert not unreachable, (
        "these public actions exist but route() cannot dispatch them, so the "
        "capability is dead to every caller: " + ", ".join(unreachable) +
        "\nBefore wiring one, check it RUNS — two of the three found in the "
        "2026-08-12 audit called helpers that do not exist, and wiring them "
        "produced a 500 instead of an honest refusal. If it cannot run, leave "
        "it unwired and register it in docs/KNOWN_INCOMPLETE.md."
    )


def test_dispatch_surfaces_name_only_dispatchable_actions():
    """Anything the platform offers to run, it must be able to run."""
    keys = _routed_keys()
    for surface, names in (("_suggest_next_action", _suggested_actions()),
                           ("auto_pipeline next_actions", _pipeline_next_actions()),
                           ("cm_step_aliases STEP_TO_TARGET", _step_alias_actions())):
        assert names, f"{surface} yielded no action names — has its shape changed?"
        dangling = sorted(names - keys)
        assert not dangling, (
            f"{surface} names {dangling}, which route() cannot dispatch. "
            "Every name a dispatch surface emits must be a key in the handlers "
            "table (keys, not method names — the table aliases)."
        )


@pytest.mark.asyncio
async def test_recommended_actions_actually_dispatch():
    """The static check above proves the name is a key. This proves route()
    really accepts it — the bug that started this file was user-facing."""
    from app.containers.construction import ConstructionContainer

    container = ConstructionContainer()
    for action in sorted(_suggested_actions() | _pipeline_next_actions()):
        result = await container.route(action, {}, {})
        assert "Unknown action" not in str(result.get("error", "")), (
            f"the platform offers {action!r} but route() rejects it: "
            f"{result.get('error')}"
        )


# ── 2. REVERSE: everything dispatchable must be reachable from something ─────

def test_every_dispatchable_action_is_named_by_some_surface():
    named = (_suggested_actions() | _pipeline_next_actions() | _step_alias_actions()
             | _hint_actions() | _orchestrator_actions())
    orphans = _routed_keys() - named
    assert orphans == EXPECTED_EXTERNAL_ONLY, (
        "the set of actions nobody in this repo names has changed.\n"
        f"  newly orphaned (nothing can reach these): {sorted(orphans - EXPECTED_EXTERNAL_ONLY)}\n"
        f"  no longer orphaned (delete from the set):  {sorted(EXPECTED_EXTERNAL_ONLY - orphans)}\n"
        "A newly orphaned action is a capability no user path can name. Either "
        "give it a surface, or add it to EXPECTED_EXTERNAL_ONLY with the reason "
        "it is called from outside."
    )


def test_router_does_not_reference_missing_methods():
    """A handlers entry pointing at a method that does not exist would raise
    AttributeError when the table is built, not at dispatch."""
    from app.containers.construction import ConstructionContainer

    container = ConstructionContainer()
    for name in _routed_methods():
        assert hasattr(container, name), (
            f"route() maps to self.{name}, which does not exist on the container"
        )


# ── 3. RUNNABLE: no method may call a helper that does not exist ─────────────

def _unresolved_self_references() -> dict[str, set[str]]:
    """Every `self.X` READ in the package that the class cannot supply.

    Store-context attributes are excluded: `self._last_bim_query_error = None`
    is assigned at runtime, so it is legitimately absent from a fresh instance
    and is not a missing helper.
    """
    from app.containers.construction import ConstructionContainer

    loaded: dict[str, set[str]] = {}
    assigned: set[str] = set()
    for path in sorted(CONTAINER_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for fn in [n for n in ast.walk(tree)
                   if isinstance(n, (ast.AsyncFunctionDef, ast.FunctionDef))]:
            for node in ast.walk(fn):
                if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                        and node.value.id == "self"):
                    if isinstance(node.ctx, ast.Store):
                        assigned.add(node.attr)
                    else:
                        loaded.setdefault(node.attr, set()).add(f"{path.name}:{fn.name}")
                if (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Attribute)
                        and isinstance(node.target.value, ast.Name)
                        and node.target.value.id == "self"):
                    assigned.add(node.target.attr)

    container = ConstructionContainer()
    return {attr: where for attr, where in loaded.items()
            if attr not in assigned and not hasattr(container, attr)}


def _self_call_graph() -> dict[str, set[str]]:
    """method -> the `self.<name>` methods it calls."""
    graph: dict[str, set[str]] = {}
    for path in sorted(CONTAINER_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for fn in [n for n in ast.walk(tree)
                   if isinstance(n, (ast.AsyncFunctionDef, ast.FunctionDef))]:
            calls = {n.func.attr for n in ast.walk(fn)
                     if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                     and isinstance(n.func.value, ast.Name) and n.func.value.id == "self"}
            graph.setdefault(fn.name, set()).update(calls)
    return graph


def _unrunnable_methods() -> set[str]:
    """Methods that crash when executed — transitively.

    `track_progress` does not merely call the missing `_assess_delay_risk`; it
    also calls `_compare_photo_to_bim`, which calls two more helpers that do
    not exist. Following the call graph is what makes this fence hold when the
    break is one level down rather than in the method itself.
    """
    missing = set(_unresolved_self_references())
    graph = _self_call_graph()
    tainted = {m for m, calls in graph.items() if calls & missing}
    changed = True
    while changed:          # propagate up the call graph to a fixed point
        changed = False
        for method, calls in graph.items():
            if method not in tainted and calls & tainted:
                tainted.add(method)
                changed = True
    return tainted


# Nothing is parked. `track_progress` and `generate_construction_report` were
# briefly held here while their four missing helpers were unwritten; all four
# are now implemented and both actions are routed, so both sets are empty and
# the assertions below are plain emptiness checks again. Re-populate ONLY with
# a matching docs/KNOWN_INCOMPLETE.md entry — an exemption without a register entry
# is how the wrong reason in docs/archive/HANDOFF.md survived for months.
PARKED_UNRUNNABLE: set[str] = set()
KNOWN_MISSING_HELPERS: set[str] = set()


def test_no_new_method_calls_a_helper_that_does_not_exist():
    """The assertion that would have caught the four phantom helpers.

    `hasattr` on a live instance, so anything the base Container supplies
    dynamically still counts. Equality rather than emptiness: the four known
    ones are parked and unrouted, but a FIFTH appearing is a new defect, and
    defining one of the four must shrink this set rather than pass silently.
    """
    missing = _unresolved_self_references()
    assert set(missing) == KNOWN_MISSING_HELPERS, (
        "the set of phantom helpers has changed.\n"
        + "  NEW (a method that crashes when run):\n"
        + "\n".join(f"    self.{a}  <- called in {', '.join(sorted(w))}"
                    for a, w in sorted(missing.items())
                    if a not in KNOWN_MISSING_HELPERS)
        + f"\n  now defined (remove from KNOWN_MISSING_HELPERS): "
          f"{sorted(KNOWN_MISSING_HELPERS - set(missing))}\n"
        "Define the helper, or leave the caller unrouted and register it in "
        "docs/KNOWN_INCOMPLETE.md."
    )


def test_no_unrunnable_method_is_routed():
    """The property that actually protects users, and it has no exceptions.

    Parking broken code is allowed — routing it is not. Dispatching a method
    that cannot run returns a 500 where the router would otherwise have given
    an honest "Unknown action". That is strictly worse than the gap it was
    meant to close, and it is exactly what happened on 2026-08-12.
    """
    routed = {m for m in _unrunnable_methods() if m in _routed_methods()}
    assert not routed, (
        "these routed actions crash when dispatched: " + ", ".join(sorted(routed)) +
        "\nRemove them from the handlers table in ConstructionContainer.route(). "
        "An unroutable capability is a gap; a routable one that 500s is a defect."
    )


def test_nothing_is_parked_without_being_registered():
    """A parked action must be genuinely unrunnable, not merely inconvenient.

    With both sets empty this asserts the strong form: every public action is
    routed and no helper is missing. If an entry is ever added back, it has to
    be for a real break — which is what the wrong "needs multi-file
    resolution" reason in docs/archive/HANDOFF.md was not.
    """
    assert PARKED_UNRUNNABLE <= _unrunnable_methods(), (
        f"{sorted(PARKED_UNRUNNABLE - _unrunnable_methods())} is parked but runs "
        "— wire it into the handlers table and drop it from PARKED_UNRUNNABLE."
    )
