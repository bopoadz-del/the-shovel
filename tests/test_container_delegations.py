"""The container's block delegations must reach the block they name.

Audit bars 2 and 3, 2026-08-12, for the formulas / orchestration cluster --
`sympy_reason`, `orchestrate`, `formula_execute`, `bim_extract`, `learn`,
`recommend`, `drawing_qto` and their siblings.

These are five-line pass-throughs: resolve a block by name, return an honest
error if it is absent, otherwise `await block.process(input_data, params)`.
There is no arithmetic to get wrong, so a control-delete of the body says
little. The two things that CAN be wrong are both silent:

  1. The block name is a typo or has drifted. `_resolve_block("formulaexec")`
     returns None and the action answers "block unavailable" -- which reads as
     a deployment/profile problem, not as a bug, and would survive a release.
  2. The delegation stops delegating. Nothing downstream notices if the action
     returns its own shaped dict instead of the block's answer.

Coverage before this file: `formula_execute` and `orchestrate` were referenced
by NO test in the repository except the router fence. `sympy_reason` and
`bim_extract` had one file each.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

CONTAINER_DIR = Path("app/containers/construction")


def _delegated_block_names() -> dict[str, set[str]]:
    """Every block name the container resolves, mapped to its callers.

    Parsed rather than listed, so a new delegation is covered the day it is
    written instead of the day someone remembers to add it here.
    """
    names: dict[str, set[str]] = {}
    for path in sorted(CONTAINER_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for fn in [n for n in ast.walk(tree)
                   if isinstance(n, (ast.AsyncFunctionDef, ast.FunctionDef))]:
            for node in ast.walk(fn):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr in ("_resolve_block", "get_dep")
                        and node.args and isinstance(node.args[0], ast.Constant)
                        and isinstance(node.args[0].value, str)):
                    names.setdefault(node.args[0].value, set()).add(fn.name)
    return names


def _pure_delegations() -> dict[str, str]:
    """Public actions whose entire body is `resolve -> guard -> block.process`.

    Detected structurally: the method resolves exactly one block and its last
    statement returns an awaited `.process(...)` call. Anything with real logic
    is excluded, because those belong in their own payload tests.
    """
    found: dict[str, str] = {}
    for path in sorted(CONTAINER_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for fn in [n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef)]:
            args = [a.arg for a in fn.args.args]
            if args[:3] != ["self", "input_data", "params"] or fn.name.startswith("_"):
                continue
            resolved = [n.args[0].value for n in ast.walk(fn)
                        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                        and n.func.attr == "_resolve_block" and n.args
                        and isinstance(n.args[0], ast.Constant)]
            if len(resolved) != 1 or len(fn.body) > 4:
                continue
            last = fn.body[-1]
            if (isinstance(last, ast.Return) and isinstance(last.value, ast.Await)
                    and isinstance(last.value.value, ast.Call)
                    and isinstance(last.value.value.func, ast.Attribute)
                    and last.value.value.func.attr == "process"):
                found[fn.name] = resolved[0]
    return found


def _every_declared_block_name() -> set[str]:
    """Block names declared anywhere, independent of the boot profile.

    NOT `BLOCK_REGISTRY`, which is profile-dependent: the virgin profile loads
    ~15 blocks and the production-like one ~48, so checking a delegation
    against it fails in virgin for names that are perfectly correct. That is
    exactly what happened -- this test passed locally and in production-like
    and failed the virgin leg, which is the whole reason both profiles are run.

    The invariant worth holding is that the name is a REAL block somewhere in
    the codebase (catching typos and drift), not that this particular boot
    happened to load it.
    """
    from app.blocks import _EXTENDED_PLATFORM_SPECS, _GENERIC_BLOCK_SPECS
    from app.core.domain_kit_loader import _KIT_BLOCK_SPECS

    names = {spec[0] for spec in _GENERIC_BLOCK_SPECS}
    names |= {spec[0] for spec in _EXTENDED_PLATFORM_SPECS}
    for kit_specs in _KIT_BLOCK_SPECS.values():
        names |= {spec[0] for spec in kit_specs}
    return names


def test_every_delegated_block_name_is_a_real_block():
    """A drifted name degrades to 'block unavailable', which reads as config.

    That is the failure this catches: it looks like the block was not loaded
    in this profile, so it survives review and deployment.
    """
    declared = _every_declared_block_name()
    assert len(declared) > 20, (
        f"only {len(declared)} block names declared -- the spec lists have moved"
    )

    names = _delegated_block_names()
    assert names, "no block delegations found -- has _resolve_block been renamed?"
    unknown = {name: sorted(callers) for name, callers in names.items()
               if name not in declared}
    assert not unknown, (
        "these block names are declared nowhere, so the actions calling them "
        "can only ever return 'block unavailable':\n"
        + "\n".join(f"  {n!r} <- {', '.join(c)}" for n, c in sorted(unknown.items()))
    )


# The delegations as they stand, asserted by EQUALITY.
#
# Not decoration, and not a >= length check -- that was tried and it did not
# bite. `_pure_delegations()` recognises a delegation by its shape, so gutting
# one ("return {'status': 'success'}") stops it matching and it silently drops
# out of the set the tests below iterate. The check excluded exactly the thing
# that breaks it. Pinning the set means a delegation that stops delegating
# fails here, and a genuinely new one fails until it is added.
EXPECTED_DELEGATIONS = {
    "bim_extract": "bim_extractor",
    "boq_process": "boq_processor",
    "drawing_qto": "drawing_qto",
    "formula_execute": "formula_executor_v2",
    "learn": "learning_engine",
    "orchestrate": "smart_orchestrator",
    "primavera_parse": "primavera_parser",
    "recommend": "recommendation_template",
    "spec_analyze": "spec_analyzer",
    "sympy_reason": "sympy_reasoning",
}


def test_the_delegation_set_is_exactly_what_is_expected():
    delegations = _pure_delegations()
    assert delegations == EXPECTED_DELEGATIONS, (
        "the set of pure block delegations changed.\n"
        f"  no longer delegating (gutted, or grew logic): "
        f"{sorted(set(EXPECTED_DELEGATIONS) - set(delegations))}\n"
        f"  new (add to EXPECTED_DELEGATIONS): "
        f"{sorted(set(delegations) - set(EXPECTED_DELEGATIONS))}\n"
        f"  re-pointed at a different block: "
        f"{ {k: (EXPECTED_DELEGATIONS[k], delegations[k]) for k in set(delegations) & set(EXPECTED_DELEGATIONS) if delegations[k] != EXPECTED_DELEGATIONS[k]} }"
    )


@pytest.mark.asyncio
async def test_every_pure_delegation_returns_the_blocks_answer():
    """The block's result must come back unaltered.

    A sentinel the container cannot invent: if the action returns anything
    else, it stopped delegating. Each action is exercised through `route()`
    too, so a delegation that works on a direct call but is unrouted still
    fails here.
    """
    from app.containers.construction import ConstructionContainer

    class SentinelBlock:
        def __init__(self, name):
            self.name = name

        async def process(self, input_data, params):
            return {"status": "success", "_sentinel": f"answered-by-{self.name}"}

    delegations = _pure_delegations()
    container = ConstructionContainer()
    for action, block_name in delegations.items():
        container.wire(block_name, SentinelBlock(block_name))

    for action, block_name in sorted(delegations.items()):
        direct = await getattr(container, action)({}, {})
        assert direct.get("_sentinel") == f"answered-by-{block_name}", (
            f"{action}() did not return {block_name}'s answer: {direct}"
        )
        routed = await container.route(action, {}, {"action": action})
        assert routed.get("_sentinel") == f"answered-by-{block_name}", (
            f"route({action!r}) did not reach {block_name}: {routed}"
        )


@pytest.mark.asyncio
async def test_an_unresolvable_block_is_refused_and_named():
    """The guard branch, exercised by forcing resolution to fail.

    Calling with nothing wired does NOT test this: `_resolve_block` falls back
    to the global registry, so a block loaded in the current profile resolves
    and returns its own honest error instead ("No file_path provided"). That
    is correct behaviour and it hides the branch. Forcing None is the only way
    to reach it.
    """
    from app.containers.construction import ConstructionContainer

    container = ConstructionContainer()
    container._resolve_block = lambda name: None

    for action, block_name in sorted(_pure_delegations().items()):
        result = await getattr(container, action)({}, {})
        if result.get("status") == "success":
            pytest.fail(
                f"{action}() reported success with {block_name} unresolvable: {result}"
            )
        assert result.get("status") == "error", f"{action}: {result}"
        assert block_name in result.get("error", ""), (
            f"{action}() refused without naming the missing block: {result}"
        )


# NOT asserted here: "no delegation returns success on empty input".
#
# It was written, it failed, and it was removed rather than given an exception
# list. `learn` legitimately succeeds with no input -- learning_engine is a
# QUERY that reports the current state (0 formulas, empty tier distribution),
# and zeros are the true answer, not a fabricated one. The invariant is real
# per block but not across all of them, so it belongs in each block's own
# tests. An exception list here would have turned a false generalisation into
# a permanent stale twin.
